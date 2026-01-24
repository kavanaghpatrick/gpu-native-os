# PRD: GPU-Initiated Batch I/O with MTLIOCommandQueue (Issue #125)

## THE GPU IS THE COMPUTER

**Current Problem**: Sequential mmap loading = 163ms bottleneck (55% of total time)
**Root Cause**: CPU initiates each file load one at a time
**Solution**: GPU-initiated batch I/O - queue ALL files at once, GPU handles scheduling

## Architecture

```
BEFORE (Sequential CPU):
  CPU: open(file1) → mmap → open(file2) → mmap → ... → 163ms
  GPU: [idle waiting for data]

AFTER (GPU Batch I/O):
  CPU: Queue 10,000 load commands in single batch → commit → done
  GPU: MTLIOCommandQueue schedules all loads in parallel → ~20ms
```

## Algorithm: GPU-Parallel File Loading

### Phase 1: Batch Command Generation
```rust
// Single CPU pass: queue ALL file loads
fn queue_batch_loads(queue: &GpuIOQueue, files: &[PathBuf]) -> Vec<GpuIOPendingLoad> {
    // Pre-calculate total size and offsets using parallel iterator
    let sizes: Vec<u64> = files.par_iter()
        .filter_map(|p| fs::metadata(p).ok().map(|m| m.len()))
        .collect();

    let total_size: u64 = sizes.iter().sum();

    // Allocate ONE large destination buffer
    let mega_buffer = device.new_buffer(
        align_to_page(total_size as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Queue ALL loads into single command buffer
    let cmd_buffer = queue.command_buffer().unwrap();

    let mut offset = 0u64;
    for (file, size) in files.iter().zip(sizes.iter()) {
        if let Some(handle) = GpuIOFileHandle::open(queue.device(), file) {
            cmd_buffer.load_buffer(&mega_buffer, offset, *size, &handle, 0);
            offset += align_to_page(*size as usize) as u64;
        }
    }

    // Single commit - GPU handles ALL scheduling
    cmd_buffer.commit();
    cmd_buffer
}
```

### Phase 2: GPU-Side Status Polling
```metal
// GPU kernel polls completion status - no CPU round-trip
kernel void poll_io_status(
    device atomic_uint* status_flags [[buffer(0)]],  // Per-file completion
    device atomic_uint* ready_count [[buffer(1)]],   // Files ready to search
    uint tid [[thread_position_in_grid]]
) {
    uint status = atomic_load_explicit(&status_flags[tid], memory_order_relaxed);
    if (status == IO_STATUS_COMPLETE) {
        atomic_fetch_add_explicit(ready_count, 1, memory_order_relaxed);
    }
}
```

### Phase 3: Overlapped Search (Search While Loading)
```metal
// Start searching files as they complete - don't wait for all
kernel void search_with_io_overlap(
    device uint8_t* mega_buffer [[buffer(0)]],
    device FileDescriptor* file_descs [[buffer(1)]],
    device atomic_uint* io_status [[buffer(2)]],
    device SearchResult* results [[buffer(3)]],
    constant SearchParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint file_idx = tid / THREADS_PER_FILE;

    // Check if this file's data is ready (GPU-side check, no CPU)
    if (atomic_load_explicit(&io_status[file_idx], memory_order_acquire) != IO_COMPLETE) {
        return; // Data not ready yet, skip
    }

    // Search this file's region
    FileDescriptor desc = file_descs[file_idx];
    search_region(mega_buffer + desc.offset, desc.size, params, results);
}
```

## Performance Model

| Metric | Current | With GPU Batch I/O |
|--------|---------|-------------------|
| File Opens | Sequential 10K calls | Batched (amortized) |
| I/O Scheduling | CPU | GPU MTLIOCommandQueue |
| Search Start | After ALL files load | As files complete |
| Total Phase 3 | ~163ms | Target: <30ms |

## GPU Algorithms Used

1. **Parallel Prefix Sum (Scan)** - Calculate file offsets in O(log n)
2. **Atomic Completion Tracking** - GPU tracks which files are ready
3. **Work Stealing** - Idle threads search completed files while others load
4. **Coalesced Memory Access** - Mega-buffer ensures sequential GPU reads

## Implementation Plan

### Step 1: Batch MTLIOCommandQueue Wrapper
```rust
pub struct GpuBatchLoader {
    queue: GpuIOQueue,
    mega_buffer: Buffer,
    file_descriptors: Buffer,  // GPU-readable file metadata
    io_status: Buffer,         // Per-file completion status
}

impl GpuBatchLoader {
    /// Load up to 50,000 files in single GPU batch
    pub fn load_batch(&mut self, files: &[PathBuf]) -> GpuBatchHandle;

    /// Check how many files are ready (GPU-side, no CPU)
    pub fn ready_count(&self) -> u32;

    /// Get mega-buffer for searching (may be partially loaded)
    pub fn search_buffer(&self) -> &Buffer;
}
```

### Step 2: Overlapped Search Kernel
```metal
// Search files as they become available
[[kernel]] void overlapped_search(
    device uint8_t* data,
    device FileDescriptor* files,
    device atomic_uint* status,
    device Match* matches,
    device atomic_uint* match_count,
    constant Pattern& pattern,
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
);
```

### Step 3: Integration with GpuContentSearch
```rust
impl GpuContentSearch {
    /// Load files using GPU batch I/O (replaces load_files)
    pub fn load_files_gpu_batch(&mut self, files: &[&Path]) -> Result<usize, SearchError> {
        let loader = GpuBatchLoader::new(&self.device)?;
        let handle = loader.load_batch(files);

        // Don't wait - start searching immediately
        self.search_buffer = loader.search_buffer().clone();
        self.file_descriptors = loader.file_descriptors().clone();
        self.batch_handle = Some(handle);

        Ok(files.len())
    }

    /// Search with I/O overlap - searches completed files while others load
    pub fn search_overlapped(&self, pattern: &str) -> Vec<ContentMatch>;
}
```

## Test Plan

### Test 1: Batch Loading Correctness
```rust
#[test]
fn test_batch_load_matches_sequential() {
    let files = collect_test_files("tests/fixtures", 1000);

    // Load with sequential mmap
    let seq_data = load_sequential(&files);

    // Load with GPU batch
    let batch_data = load_gpu_batch(&files);

    // Verify identical content
    assert_eq!(seq_data, batch_data);
}
```

### Test 2: Overlapped Search Correctness
```rust
#[test]
fn test_overlapped_search_finds_all_matches() {
    // Search with wait-for-all
    let matches_wait = search_after_load(&files, "pattern");

    // Search with overlap
    let matches_overlap = search_overlapped(&files, "pattern");

    // Same results (order may differ)
    assert_eq!(matches_wait.len(), matches_overlap.len());
}
```

### Test 3: Performance Benchmark
```rust
#[test]
fn benchmark_batch_vs_sequential() {
    let files = collect_files(".", 10_000);

    // Baseline: sequential mmap
    let seq_time = time(|| load_sequential(&files));

    // GPU batch
    let batch_time = time(|| load_gpu_batch(&files));

    println!("Sequential: {:.1}ms", seq_time);
    println!("GPU Batch:  {:.1}ms", batch_time);
    println!("Speedup:    {:.1}x", seq_time / batch_time);

    assert!(batch_time < seq_time / 3.0, "Expected 3x+ speedup");
}
```

## Success Metrics

- [ ] Phase 3 time reduced from 163ms to <30ms (5x improvement)
- [ ] GPU batch I/O handles 10,000+ files
- [ ] Overlapped search starts within 5ms of first file ready
- [ ] Zero CPU involvement during I/O (only initial queue setup)

## Dependencies

- Metal 3+ (MTLIOCommandQueue)
- Apple Silicon (unified memory)
- macOS 13+ (Ventura)
