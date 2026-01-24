# Issue #132: Streaming I/O - Overlap File Loading with GPU Search

## Problem Statement

Current pipeline is **sequential**:
```
Load ALL files (283ms) → THEN search (50ms) = 333ms total
```

We wait for ALL files to load before starting ANY GPU work. This wastes GPU cycles.

**Benchmark Evidence:**
- Small (134 MB, 14K files): GPU 283ms vs ripgrep 180ms
- File loading dominates the time budget
- GPU sits idle during I/O phase

## Solution: Pipelined Streaming

```
Load chunk 1 → Load chunk 2 → Load chunk 3 → Load chunk 4
              Search chunk 1 → Search chunk 2 → Search chunk 3 → Search chunk 4
```

**Expected improvement:**
- I/O and compute overlap = ~40% time reduction
- 283ms → ~170ms (competitive with ripgrep's 180ms)

## Architecture

### Current (Sequential)
```
CPU: [====== Load All Files ======]
GPU:                               [== Search ==]
     0ms                          283ms         333ms
```

### New (Streaming)
```
CPU: [Load 1][Load 2][Load 3][Load 4]
GPU:        [Search1][Search2][Search3][Search4]
     0ms    70ms    140ms    210ms    ~200ms total
```

## Implementation

### Data Structures

```rust
/// Streaming chunk for pipelined I/O
#[repr(C)]
pub struct StreamChunk {
    pub buffer: Buffer,         // GPU buffer for this chunk
    pub file_count: u32,        // Number of files in this chunk
    pub total_bytes: u64,       // Total data in chunk
    pub descriptors: Buffer,    // File descriptors for this chunk
    pub ready: AtomicBool,      // Set when I/O complete
}

/// Double-buffer for streaming
pub struct StreamingPipeline {
    chunks: [StreamChunk; 4],   // Quad-buffering
    load_idx: AtomicU32,        // Current loading chunk
    search_idx: AtomicU32,      // Current searching chunk
}
```

### Metal Kernel (unchanged - operates on chunks)

```metal
// Same search kernel, but called per-chunk instead of once
kernel void search_chunk(
    device const uchar4* data [[buffer(0)]],
    device const FileDescriptor* files [[buffer(1)]],
    device SearchResult* results [[buffer(2)]],
    constant SearchParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Existing O(1) chunked search - no changes needed
    // Just operates on smaller data set per invocation
}
```

### Rust Pipeline

```rust
impl StreamingSearch {
    pub fn search_streaming(&mut self, files: &[PathBuf], pattern: &str) -> Vec<SearchResult> {
        let chunk_size = files.len() / 4;  // 4 chunks
        let mut all_results = Vec::new();

        // Start first chunk loading
        self.start_load_chunk(0, &files[0..chunk_size]);

        for i in 0..4 {
            // Wait for current chunk to finish loading
            self.wait_chunk_ready(i);

            // Start loading NEXT chunk (overlapped)
            if i < 3 {
                let next_start = (i + 1) * chunk_size;
                let next_end = ((i + 2) * chunk_size).min(files.len());
                self.start_load_chunk(i + 1, &files[next_start..next_end]);
            }

            // Search current chunk (while next is loading)
            let results = self.search_chunk(i, pattern);
            all_results.extend(results);
        }

        all_results
    }

    fn start_load_chunk(&self, chunk_idx: usize, files: &[PathBuf]) {
        // Use MTLIOCommandQueue for async loading
        let cmd_buffer = self.io_queue.command_buffer().unwrap();

        for (i, file) in files.iter().enumerate() {
            let handle = GpuIOFileHandle::open(&self.device, file).unwrap();
            let offset = self.chunks[chunk_idx].compute_offset(i);
            cmd_buffer.load_buffer(
                &self.chunks[chunk_idx].buffer,
                offset,
                file.metadata().unwrap().len(),
                &handle,
                0,
            );
        }

        // Commit but DON'T wait - returns immediately
        cmd_buffer.commit();
    }

    fn wait_chunk_ready(&self, chunk_idx: usize) {
        // Spin until I/O completes (or use MTLSharedEvent)
        while !self.chunks[chunk_idx].ready.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    fn search_chunk(&mut self, chunk_idx: usize, pattern: &str) -> Vec<SearchResult> {
        let encoder = self.command_buffer.compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.search_pipeline);
        encoder.set_buffer(0, Some(&self.chunks[chunk_idx].buffer), 0);
        encoder.set_buffer(1, Some(&self.chunks[chunk_idx].descriptors), 0);
        // ... dispatch and collect results
    }
}
```

### Synchronization with MTLSharedEvent

```rust
// Better than spin-waiting: use GPU events
pub struct StreamingPipelineV2 {
    shared_event: SharedEvent,
    load_complete_values: [u64; 4],  // Event values for each chunk
}

impl StreamingPipelineV2 {
    fn start_load_chunk(&self, chunk_idx: usize, files: &[PathBuf]) {
        let cmd_buffer = self.io_queue.command_buffer().unwrap();
        // ... load commands ...

        // Signal event when load completes
        cmd_buffer.encode_signal_event(
            &self.shared_event,
            self.load_complete_values[chunk_idx],
        );
        cmd_buffer.commit();
    }

    fn wait_chunk_ready(&self, chunk_idx: usize) {
        // CPU waits on GPU event (efficient, no spinning)
        self.shared_event.wait_until_signaled(
            self.load_complete_values[chunk_idx],
        );
    }
}
```

## Benchmarks

### Test Cases

1. **Small directory** (14K files, 134 MB) - Target: Match ripgrep
2. **Large directory** (61K files, 657 MB) - Target: Improve 3x → 4x
3. **Cold cache** - Maximum benefit from overlap

### Expected Results

| Scale | Current | Streaming | Speedup | vs ripgrep |
|-------|---------|-----------|---------|------------|
| Small (134 MB) | 283ms | ~170ms | 1.7x | Tied |
| Large (657 MB) | 1,853ms | ~1,200ms | 1.5x | 4.5x faster |

### Benchmark Code

```rust
#[test]
fn benchmark_streaming_vs_batch() {
    let device = Device::system_default().unwrap();
    let files = collect_test_files(".", 10_000);

    // Current: batch load then search
    let batch_start = Instant::now();
    let batch_loader = GpuBatchLoader::new(&device).unwrap();
    let batch_result = batch_loader.load_batch(&files).unwrap();
    let mut searcher = GpuContentSearch::new(&device, batch_result.file_count()).unwrap();
    searcher.load_from_batch(&batch_result).unwrap();
    let _ = searcher.search("TODO", false);
    let batch_time = batch_start.elapsed();

    // New: streaming pipeline
    let stream_start = Instant::now();
    let mut streaming = StreamingSearch::new(&device).unwrap();
    let _ = streaming.search_streaming(&files, "TODO");
    let stream_time = stream_start.elapsed();

    println!("Batch: {:.1}ms, Streaming: {:.1}ms, Speedup: {:.2}x",
        batch_time.as_secs_f64() * 1000.0,
        stream_time.as_secs_f64() * 1000.0,
        batch_time.as_secs_f64() / stream_time.as_secs_f64());

    assert!(stream_time < batch_time, "Streaming should be faster");
}
```

## Success Criteria

1. **Correctness:** Same search results as batch mode
2. **Performance:** ≥30% reduction in total search time for small directories
3. **Overlap efficiency:** GPU utilization >50% during I/O phase
4. **Memory:** No increase in peak memory (reuse chunk buffers)

## Implementation Steps

1. Add `StreamChunk` and `StreamingPipeline` structs
2. Implement `start_load_chunk` with async MTLIOCommandQueue
3. Implement chunk-wise search dispatch
4. Add MTLSharedEvent synchronization
5. Integrate into `gpu_ripgrep` with `--stream` flag
6. Benchmark and tune chunk count/size
7. Make streaming the default if benchmarks confirm improvement
