# PRD: GPU Parallel Compaction with Prefix Sum (Issue #126)

## THE GPU IS THE COMPUTER

**Problem**: 10,000 small mmap buffers = scattered memory, poor cache locality
**Solution**: GPU parallel compaction into single contiguous mega-buffer using parallel prefix sum

## The Algorithm: Blelloch Scan + Parallel Scatter

This is the **gold standard** GPU algorithm for variable-length data compaction.

### Why Parallel Prefix Sum?

Given N files with sizes [s₀, s₁, s₂, ...], we need offsets [0, s₀, s₀+s₁, ...].

**CPU approach**: O(N) sequential - terrible
**GPU approach**: O(log N) parallel depth - optimal

```
Input sizes:   [100, 200, 150, 300, 50]
Prefix sum:    [0, 100, 300, 450, 750]  ← Each file knows its offset!
All threads write in parallel to their computed offset
```

## Architecture

```
Phase 1: GPU Prefix Sum (compute offsets)
  Input:  [size₀, size₁, size₂, ..., sizeₙ]
  Output: [off₀,  off₁,  off₂,  ..., offₙ]
  Depth:  O(log N) = ~14 steps for 10,000 files

Phase 2: GPU Parallel Scatter (copy data)
  Thread i: memcpy(mega_buffer + off[i], file_data[i], size[i])
  All 10,000 copies happen simultaneously!

Phase 3: GPU Search (on contiguous buffer)
  Single mega-buffer = perfect coalesced access
```

## Metal Shader: Blelloch Parallel Prefix Sum

```metal
// Work-efficient parallel prefix sum (Blelloch scan)
// O(N) work, O(log N) depth - optimal!

kernel void prefix_sum_upsweep(
    device uint* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) {
        data[idx] += data[idx - stride];
    }
}

kernel void prefix_sum_downsweep(
    device uint* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint idx = (tid + 1) * stride * 2 - 1;
    if (idx < n) {
        uint temp = data[idx - stride];
        data[idx - stride] = data[idx];
        data[idx] += temp;
    }
}

// Complete scan in log(N) dispatches
void gpu_exclusive_scan(device uint* data, uint n, command_buffer& cmd) {
    // Up-sweep (reduce)
    for (uint stride = 1; stride < n; stride *= 2) {
        dispatch(prefix_sum_upsweep, n / (stride * 2), stride);
    }

    // Set last element to 0 for exclusive scan
    data[n-1] = 0;

    // Down-sweep
    for (uint stride = n / 2; stride >= 1; stride /= 2) {
        dispatch(prefix_sum_downsweep, n / (stride * 2), stride);
    }
}
```

## Metal Shader: Parallel Scatter

```metal
// Each thread copies one file to its computed offset
kernel void parallel_scatter(
    device uint8_t* mega_buffer [[buffer(0)]],      // Destination
    device uint8_t* source_data [[buffer(1)]],      // All files concatenated
    device uint* source_offsets [[buffer(2)]],      // Where each file starts in source
    device uint* dest_offsets [[buffer(3)]],        // Computed by prefix sum
    device uint* sizes [[buffer(4)]],               // File sizes
    uint tid [[thread_position_in_grid]]            // File index
) {
    uint src_off = source_offsets[tid];
    uint dst_off = dest_offsets[tid];
    uint size = sizes[tid];

    // Vectorized copy (16 bytes at a time)
    device uint4* src = (device uint4*)(source_data + src_off);
    device uint4* dst = (device uint4*)(mega_buffer + dst_off);
    uint vec_count = size / 16;

    for (uint i = 0; i < vec_count; i++) {
        dst[i] = src[i];
    }

    // Handle remainder bytes
    uint remainder_off = vec_count * 16;
    for (uint i = remainder_off; i < size; i++) {
        mega_buffer[dst_off + i] = source_data[src_off + i];
    }
}

// Optimized: Use threadgroup memory for small files
kernel void parallel_scatter_coalesced(
    device uint8_t* mega_buffer [[buffer(0)]],
    device uint8_t* source_data [[buffer(1)]],
    device FileDescriptor* files [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Each SIMD group (32 threads) handles one file
    uint file_idx = simd_id;
    FileDescriptor file = files[file_idx];

    // All 32 threads copy 32 bytes per iteration
    uint bytes_per_iter = 32;
    uint iterations = (file.size + bytes_per_iter - 1) / bytes_per_iter;

    for (uint i = 0; i < iterations; i++) {
        uint byte_idx = i * bytes_per_iter + lane;
        if (byte_idx < file.size) {
            mega_buffer[file.dest_offset + byte_idx] =
                source_data[file.source_offset + byte_idx];
        }
    }
}
```

## Rust Implementation

```rust
pub struct GpuParallelCompactor {
    device: Device,
    upsweep_pipeline: ComputePipelineState,
    downsweep_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
}

impl GpuParallelCompactor {
    /// Compact multiple buffers into one using GPU parallel prefix sum
    pub fn compact(&self, sources: &[MmapBuffer]) -> CompactedBuffer {
        // Step 1: Upload sizes to GPU
        let sizes: Vec<u32> = sources.iter()
            .map(|b| b.file_size() as u32)
            .collect();
        let sizes_buffer = self.device.new_buffer_with_data(&sizes, ...);

        // Step 2: GPU parallel prefix sum to compute offsets
        let offsets_buffer = self.gpu_prefix_sum(&sizes_buffer, sizes.len());

        // Step 3: Read total size (last offset + last size)
        let total_size = self.read_total_size(&offsets_buffer, &sizes_buffer, sizes.len());

        // Step 4: Allocate mega-buffer
        let mega_buffer = self.device.new_buffer(total_size, ...);

        // Step 5: GPU parallel scatter
        self.gpu_scatter(sources, &offsets_buffer, &mega_buffer);

        CompactedBuffer {
            buffer: mega_buffer,
            file_offsets: offsets_buffer,
            file_count: sources.len(),
        }
    }

    fn gpu_prefix_sum(&self, sizes: &Buffer, n: usize) -> Buffer {
        let cmd = self.queue.command_buffer();
        let offsets = self.device.new_buffer(n * 4, ...);

        // Copy sizes to offsets (we'll scan in place)
        // ... blit encoder copy ...

        // Up-sweep phase: log(n) dispatches
        let mut stride = 1;
        while stride < n {
            let encoder = cmd.compute_encoder();
            encoder.set_pipeline(&self.upsweep_pipeline);
            encoder.set_buffer(0, &offsets);
            encoder.set_bytes(1, &(n as u32));
            encoder.set_bytes(2, &(stride as u32));
            encoder.dispatch_threads(n / (stride * 2), 1, 1);
            encoder.end();
            stride *= 2;
        }

        // Clear last element for exclusive scan
        // ... (set to 0) ...

        // Down-sweep phase: log(n) dispatches
        while stride >= 1 {
            let encoder = cmd.compute_encoder();
            encoder.set_pipeline(&self.downsweep_pipeline);
            // ... same pattern ...
            stride /= 2;
        }

        cmd.commit();
        cmd.wait_until_completed();

        offsets
    }
}
```

## Performance Analysis

### Parallel Prefix Sum Complexity
- **Work**: O(N) - same as sequential
- **Depth**: O(log N) - massively parallel
- **For 10,000 files**: 14 dispatches total

### Parallel Scatter Complexity
- **Work**: O(total bytes)
- **Parallelism**: 10,000 threads (one per file)
- **Memory bandwidth**: Full GPU bandwidth (~200 GB/s on M4 Pro)

### Expected Performance

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Compute 10K offsets | 10µs | 50µs (14 dispatches) | 0.2x (GPU overhead) |
| Copy 110MB | 55ms (memcpy) | 0.5ms (200 GB/s) | 110x |
| **Total** | **55ms** | **<1ms** | **>50x** |

Note: The scan itself is slower on GPU due to dispatch overhead, but the scatter
dominates and is massively faster.

## Test Plan

### Test 1: Prefix Sum Correctness
```rust
#[test]
fn test_prefix_sum_correctness() {
    let sizes = vec![100, 200, 150, 300, 50];
    let expected = vec![0, 100, 300, 450, 750];

    let gpu_offsets = gpu_prefix_sum(&sizes);

    assert_eq!(gpu_offsets, expected);
}

#[test]
fn test_prefix_sum_large() {
    let sizes: Vec<u32> = (0..10000).map(|i| (i % 1000) + 100).collect();

    // CPU reference
    let mut cpu_offsets = vec![0u32; sizes.len()];
    for i in 1..sizes.len() {
        cpu_offsets[i] = cpu_offsets[i-1] + sizes[i-1];
    }

    let gpu_offsets = gpu_prefix_sum(&sizes);

    assert_eq!(gpu_offsets, cpu_offsets);
}
```

### Test 2: Scatter Correctness
```rust
#[test]
fn test_scatter_preserves_data() {
    let files: Vec<Vec<u8>> = (0..1000)
        .map(|i| vec![i as u8; 100 + i])
        .collect();

    let compacted = gpu_compact(&files);

    // Verify each file's data at its offset
    for (i, file) in files.iter().enumerate() {
        let offset = compacted.offset(i);
        let slice = &compacted.data()[offset..offset + file.len()];
        assert_eq!(slice, file.as_slice());
    }
}
```

### Test 3: Performance Benchmark
```rust
#[test]
fn benchmark_compaction() {
    let files = load_mmap_buffers(".", 10_000);

    let cpu_time = time(|| cpu_compact(&files));
    let gpu_time = time(|| gpu_compact(&files));

    println!("CPU compact: {:.1}ms", cpu_time);
    println!("GPU compact: {:.1}ms", gpu_time);
    println!("Speedup: {:.1}x", cpu_time / gpu_time);

    assert!(gpu_time < cpu_time / 10.0);
}
```

## Integration with GPU Ripgrep

```rust
impl GpuContentSearch {
    pub fn load_files_compacted(&mut self, paths: &[&Path]) -> Result<usize, Error> {
        // Load files with mmap (still sequential for now)
        let mmap_buffers: Vec<MmapBuffer> = paths.iter()
            .filter_map(|p| MmapBuffer::from_file(&self.device, p).ok())
            .collect();

        // GPU compact into mega-buffer
        let compactor = GpuParallelCompactor::new(&self.device)?;
        let compacted = compactor.compact(&mmap_buffers);

        self.search_buffer = compacted.buffer;
        self.file_offsets = compacted.file_offsets;
        self.file_count = compacted.file_count;

        Ok(self.file_count)
    }
}
```

## Success Metrics

- [ ] Prefix sum computes 10,000 offsets in <1ms
- [ ] Scatter copies 110MB in <1ms (>100 GB/s effective)
- [ ] Total compaction time <2ms (vs 163ms current)
- [ ] Search performance unchanged (same data, better layout)
