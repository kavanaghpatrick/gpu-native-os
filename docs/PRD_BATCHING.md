# PRD: Batch Path Lookup (Priority 1)

**Issue**: #25 - Implement True Batching for Path Lookups
**Priority**: ⭐⭐⭐ Critical (100x speedup potential)
**Status**: Not Started
**Effort**: 2-3 days

---

## Problem Statement

Current implementation dispatches one GPU kernel per path lookup, resulting in:
- **200µs dispatch overhead per lookup** (dominates total latency)
- **5,000 lookups/sec throughput** (far below GPU capability)
- **~5% GPU utilization** (wasting 95% of parallel capacity)
- **50 GB/s memory bandwidth** (using 12% of 400 GB/s available)

**Root Cause**: Metal command buffer creation, encoding, and synchronization overhead is amortized over only 1 lookup.

**Impact**: GPU is slower than CPU for all realistic workloads.

---

## Solution Overview

Implement **batch path lookup** that queues multiple path resolution requests and dispatches them to GPU in a single kernel invocation.

### Key Benefits

- **Amortize dispatch cost**: 200µs ÷ 100 paths = **2µs per path**
- **Increase throughput**: 100 paths in 300µs = **333,000 lookups/sec** (66x improvement)
- **Better GPU utilization**: Process multiple directories in parallel
- **Higher memory bandwidth**: More concurrent memory accesses

---

## Technical Design

### Architecture

```
User API:
  lookup_batch(&[&str]) → Vec<Result<u32>>

Internal Flow:
  1. Parse all paths into PathComponent arrays
  2. Allocate single large GPU buffer for all path data
  3. Dispatch ONE compute kernel that processes ALL paths in parallel
  4. Read all results from single output buffer
  5. Return Vec of results matching input order
```

### Data Structures

#### Rust Side

```rust
/// Batch lookup request
pub struct BatchPathLookup {
    /// Paths to resolve
    paths: Vec<String>,

    /// Maximum batch size before auto-flush
    max_batch_size: usize,

    /// Parsed path components (flattened)
    components: Vec<PathComponent>,

    /// Path metadata: (start_index, component_count)
    path_metadata: Vec<(u32, u32)>,
}

/// Single path's metadata in batch
#[repr(C)]
struct PathMetadata {
    start_idx: u32,       // Index into components buffer
    component_count: u32, // Number of components in this path
    start_inode: u32,     // Starting inode (ROOT or current_dir)
    _padding: u32,
}

/// Batch results (one per path)
#[repr(C)]
struct BatchResult {
    inode_id: u32,   // Result inode (or INVALID_INODE)
    status: u32,     // 0=success, 1=not_found, 2=error
}
```

#### Metal Side

```metal
kernel void batch_path_lookup_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* entries [[buffer(1)]],
    constant uint32_t& total_inodes [[buffer(2)]],
    constant uint32_t& total_entries [[buffer(3)]],

    // Batch-specific buffers
    device PathComponent* all_components [[buffer(4)]],  // Flattened components
    device PathMetadata* path_metadata [[buffer(5)]],    // Per-path metadata
    device BatchResult* results [[buffer(6)]],           // Output results
    constant uint32_t& batch_size [[buffer(7)]],         // Number of paths

    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    // Each threadgroup processes one path
    uint path_idx = gid / 1024;  // Which path this threadgroup handles
    uint local_tid = tid;         // Thread within threadgroup

    if (path_idx >= batch_size) return;

    PathMetadata meta = path_metadata[path_idx];
    uint32_t current_inode = meta.start_inode;

    // Walk through components for this path
    for (uint32_t comp_idx = 0; comp_idx < meta.component_count; comp_idx++) {
        PathComponent component = all_components[meta.start_idx + comp_idx];

        // Parallel search within directory (same as single lookup)
        threadgroup uint32_t found_inode;
        threadgroup atomic_uint found_flag;

        if (local_tid == 0) {
            found_inode = INVALID_INODE;
            atomic_store_explicit(&found_flag, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread searches different entries
        for (uint32_t i = local_tid; i < total_entries; i += 1024) {
            // Same search logic as single lookup...
            // Check parent_id, hash, and name match
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (found_inode == INVALID_INODE) {
            // Not found - write error and exit
            if (local_tid == 0) {
                results[path_idx].inode_id = INVALID_INODE;
                results[path_idx].status = 1; // NOT_FOUND
            }
            return;
        }

        current_inode = found_inode;
    }

    // Success - write result
    if (local_tid == 0) {
        results[path_idx].inode_id = current_inode;
        results[path_idx].status = 0; // SUCCESS
    }
}
```

### GPU Dispatch Configuration

```rust
// Dispatch N threadgroups, one per path
let threadgroups = MTLSize::new(batch_size as u64, 1, 1);
let threads_per_group = MTLSize::new(1024, 1, 1);

encoder.dispatch_thread_groups(threadgroups, threads_per_group);
```

**Key Insight**: Each threadgroup (1024 threads) processes one path. Multiple threadgroups run in parallel on GPU.

---

## API Design

### Public Interface

```rust
impl GpuFilesystem {
    /// Lookup multiple paths in a single GPU dispatch
    ///
    /// # Example
    /// ```
    /// let paths = vec!["/src/main.rs", "/src/lib.rs", "/tests/test.rs"];
    /// let results = fs.lookup_batch(&paths)?;
    ///
    /// for (path, result) in paths.iter().zip(results.iter()) {
    ///     match result {
    ///         Ok(inode_id) => println!("{} → {}", path, inode_id),
    ///         Err(e) => println!("{} → Error: {}", path, e),
    ///     }
    /// }
    /// ```
    pub fn lookup_batch(&self, paths: &[&str]) -> Result<Vec<Result<u32, String>>, String>;

    /// Async batch lookup (returns immediately, poll for results)
    pub fn lookup_batch_async(&self, paths: &[&str])
        -> Result<BatchLookupFuture, String>;
}

/// Builder for configuring batch lookups
pub struct BatchLookupBuilder {
    max_batch_size: usize,
    timeout_ms: Option<u64>,
}

impl BatchLookupBuilder {
    pub fn max_batch_size(mut self, size: usize) -> Self;
    pub fn timeout(mut self, ms: u64) -> Self;
    pub fn build(self) -> BatchPathLookup;
}
```

### Internal Implementation

```rust
pub struct GpuFilesystem {
    // ... existing fields ...

    // Batch lookup pipeline
    batch_lookup_pipeline: ComputePipelineState,

    // Batch buffers (sized for max batch)
    batch_components_buffer: Buffer,      // PathComponent array
    batch_metadata_buffer: Buffer,        // PathMetadata array
    batch_results_buffer: Buffer,         // BatchResult array
    batch_params_buffer: Buffer,          // Batch parameters

    // Configuration
    max_batch_size: usize,
}
```

---

## Algorithm Pseudocode

### Rust Side

```rust
fn lookup_batch(&self, paths: &[&str]) -> Result<Vec<Result<u32, String>>, String> {
    // 1. Validate input
    if paths.is_empty() {
        return Ok(vec![]);
    }

    let batch_size = paths.len().min(self.max_batch_size);
    if batch_size > self.max_batch_size {
        return Err(format!("Batch size {} exceeds max {}",
            paths.len(), self.max_batch_size));
    }

    // 2. Parse all paths into components
    let mut all_components = Vec::new();
    let mut metadata = Vec::new();

    for path in paths.iter().take(batch_size) {
        let (start_inode, components) = self.parse_path(path)?;

        metadata.push(PathMetadata {
            start_idx: all_components.len() as u32,
            component_count: components.len() as u32,
            start_inode,
            _padding: 0,
        });

        all_components.extend(components);
    }

    // 3. Write to GPU buffers
    unsafe {
        // Write components
        let comp_ptr = self.batch_components_buffer.contents() as *mut PathComponent;
        for (i, comp) in all_components.iter().enumerate() {
            *comp_ptr.add(i) = *comp;
        }

        // Write metadata
        let meta_ptr = self.batch_metadata_buffer.contents() as *mut PathMetadata;
        for (i, meta) in metadata.iter().enumerate() {
            *meta_ptr.add(i) = *meta;
        }

        // Write batch parameters
        let params = self.batch_params_buffer.contents() as *mut BatchParams;
        *params = BatchParams {
            total_inodes: self.current_inode_count as u32,
            total_entries: self.current_entry_count as u32,
            batch_size: batch_size as u32,
            _padding: 0,
        };
    }

    // 4. Dispatch GPU kernel
    let queue = self.device.new_command_queue();
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&self.batch_lookup_pipeline);
    encoder.set_buffer(0, Some(&self.inode_buffer), 0);
    encoder.set_buffer(1, Some(&self.dir_entry_buffer), 0);
    encoder.set_buffer(2, Some(&self.batch_params_buffer), 0);
    encoder.set_buffer(3, Some(&self.batch_params_buffer), 4);
    encoder.set_buffer(4, Some(&self.batch_components_buffer), 0);
    encoder.set_buffer(5, Some(&self.batch_metadata_buffer), 0);
    encoder.set_buffer(6, Some(&self.batch_results_buffer), 0);
    encoder.set_buffer(7, Some(&self.batch_params_buffer), 8);

    // One threadgroup per path, 1024 threads per group
    let threadgroups = MTLSize::new(batch_size as u64, 1, 1);
    let threads_per_group = MTLSize::new(1024, 1, 1);
    encoder.dispatch_thread_groups(threadgroups, threads_per_group);

    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // 5. Read results
    let mut results = Vec::with_capacity(batch_size);
    unsafe {
        let result_ptr = self.batch_results_buffer.contents() as *const BatchResult;

        for i in 0..batch_size {
            let result = *result_ptr.add(i);

            if result.status == 0 {
                results.push(Ok(result.inode_id));
            } else {
                results.push(Err(format!("Path not found: {}", paths[i])));
            }
        }
    }

    Ok(results)
}
```

### GPU Kernel (Detailed)

```metal
kernel void batch_path_lookup_kernel(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* entries [[buffer(1)]],
    constant uint32_t& total_inodes [[buffer(2)]],
    constant uint32_t& total_entries [[buffer(3)]],
    device PathComponent* all_components [[buffer(4)]],
    device PathMetadata* path_metadata [[buffer(5)]],
    device BatchResult* results [[buffer(6)]],
    constant uint32_t& batch_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    const uint32_t INVALID_INODE = 0xFFFFFFFF;

    // Calculate which path this threadgroup is processing
    uint path_idx = gid / 1024;

    if (path_idx >= batch_size) return;

    // Load this path's metadata
    PathMetadata meta = path_metadata[path_idx];
    uint32_t current_inode = meta.start_inode;

    // Walk through each component of the path
    for (uint32_t comp_idx = 0; comp_idx < meta.component_count; comp_idx++) {
        PathComponent component = all_components[meta.start_idx + comp_idx];

        // Shared variables for this threadgroup
        threadgroup uint32_t found_inode;
        threadgroup atomic_uint found_flag;

        if (tid == 0) {
            found_inode = INVALID_INODE;
            atomic_store_explicit(&found_flag, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel search: each thread checks different entries
        for (uint32_t i = tid; i < total_entries; i += 1024) {
            uint32_t entry_inode = entries[i].inode_id;

            // Check if this entry is in the current directory
            if (inodes[entry_inode].parent_id == current_inode) {
                // Check hash match
                if (entries[i].name_hash == component.hash) {
                    // Verify name (handle hash collisions)
                    bool name_matches = true;
                    if (entries[i].name_len == component.len) {
                        for (uint16_t j = 0; j < component.len; j++) {
                            if (entries[i].name[j] != component.name[j]) {
                                name_matches = false;
                                break;
                            }
                        }
                    } else {
                        name_matches = false;
                    }

                    if (name_matches) {
                        // Found it! First thread to find wins
                        uint32_t old = atomic_exchange_explicit(&found_flag, 1,
                            memory_order_relaxed);
                        if (old == 0) {
                            found_inode = entry_inode;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if component was found
        if (found_inode == INVALID_INODE) {
            // Path component not found - mark as error and exit
            if (tid == 0) {
                results[path_idx].inode_id = INVALID_INODE;
                results[path_idx].status = 1; // NOT_FOUND
            }
            return;
        }

        // Move to next level
        current_inode = found_inode;
    }

    // All components resolved successfully
    if (tid == 0) {
        results[path_idx].inode_id = current_inode;
        results[path_idx].status = 0; // SUCCESS
    }
}
```

---

## Test Specification

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_batch_lookup_empty() {
        let device = Device::system_default().unwrap();
        let fs = GpuFilesystem::new(&device, 1024).unwrap();

        let results = fs.lookup_batch(&[]).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_single() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        let results = fs.lookup_batch(&["/src"]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unwrap(), src_id);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_multiple() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();
        let docs_id = fs.add_file(0, "docs", FileType::Directory).unwrap();
        let tests_id = fs.add_file(0, "tests", FileType::Directory).unwrap();

        let paths = vec!["/src", "/docs", "/tests"];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].unwrap(), src_id);
        assert_eq!(results[1].unwrap(), docs_id);
        assert_eq!(results[2].unwrap(), tests_id);
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_large() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 2048).unwrap();

        // Create 100 files
        let mut expected_ids = Vec::new();
        let mut paths = Vec::new();

        for i in 0..100 {
            let name = format!("file{:03}", i);
            let id = fs.add_file(0, &name, FileType::Regular).unwrap();
            expected_ids.push(id);
            paths.push(format!("/file{:03}", i));
        }

        // Batch lookup all 100
        let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let results = fs.lookup_batch(&path_refs).unwrap();

        assert_eq!(results.len(), 100);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.as_ref().unwrap(), &expected_ids[i]);
        }
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_mixed_success_fail() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        let paths = vec!["/src", "/missing", "/src", "/also_missing"];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].as_ref().unwrap(), &src_id);
        assert!(results[1].is_err());
        assert_eq!(results[2].as_ref().unwrap(), &src_id);
        assert!(results[3].is_err());
    }

    #[test]
    #[ignore]
    fn test_batch_lookup_nested_paths() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();
        let gpu_os_id = fs.add_file(src_id, "gpu_os", FileType::Directory).unwrap();
        let main_id = fs.add_file(src_id, "main.rs", FileType::Regular).unwrap();
        let fs_id = fs.add_file(gpu_os_id, "filesystem.rs", FileType::Regular).unwrap();

        let paths = vec![
            "/src/main.rs",
            "/src/gpu_os",
            "/src/gpu_os/filesystem.rs",
        ];
        let results = fs.lookup_batch(&paths).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap(), &main_id);
        assert_eq!(results[1].as_ref().unwrap(), &gpu_os_id);
        assert_eq!(results[2].as_ref().unwrap(), &fs_id);
    }
}
```

### Performance Tests

```rust
#[test]
#[ignore]
fn bench_batch_vs_individual() {
    let device = Device::system_default().unwrap();
    let mut fs = GpuFilesystem::new(&device, 2048).unwrap();

    // Create 100 files
    let mut paths = Vec::new();
    for i in 0..100 {
        let name = format!("file{:03}", i);
        fs.add_file(0, &name, FileType::Regular).unwrap();
        paths.push(format!("/file{:03}", i));
    }

    let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();

    // Benchmark individual lookups
    let start = Instant::now();
    for path in &path_refs {
        let _ = fs.lookup_path(path);
    }
    let individual_time = start.elapsed();

    // Benchmark batch lookup
    let start = Instant::now();
    let _ = fs.lookup_batch(&path_refs);
    let batch_time = start.elapsed();

    let speedup = individual_time.as_secs_f64() / batch_time.as_secs_f64();

    println!("Individual: {:.2}ms", individual_time.as_secs_f64() * 1000.0);
    println!("Batch: {:.2}ms", batch_time.as_secs_f64() * 1000.0);
    println!("Speedup: {:.1}x", speedup);

    // Should be at least 10x faster
    assert!(speedup > 10.0, "Batch should be >10x faster, got {:.1}x", speedup);
}
```

---

## Success Criteria

### Functional

- ✅ Correctly resolves all paths in batch
- ✅ Handles mixed success/failure cases
- ✅ Maintains result ordering (result[i] matches paths[i])
- ✅ Supports batch sizes from 1 to max_batch_size
- ✅ Error handling for invalid paths
- ✅ All unit tests passing

### Performance

- ✅ Batch of 100 paths: **<500µs total** (5µs per path)
- ✅ **>10x speedup** vs individual lookups
- ✅ Throughput: **>100,000 lookups/sec** for batch size 100
- ✅ GPU utilization: **>20%** (multiple threadgroups active)
- ✅ Memory bandwidth: **>200 GB/s**

---

## Implementation Plan

1. **Add batch data structures** (2 hours)
   - PathMetadata, BatchResult, BatchParams
   - Update GpuFilesystem struct with batch buffers

2. **Implement Metal kernel** (4 hours)
   - batch_path_lookup_kernel shader
   - Test with simple cases

3. **Implement Rust API** (4 hours)
   - lookup_batch() method
   - Path parsing and buffer management
   - GPU dispatch and result reading

4. **Add tests** (3 hours)
   - Unit tests (empty, single, multiple, nested)
   - Performance benchmarks
   - Edge cases

5. **Optimize and tune** (3 hours)
   - Buffer sizing
   - Threadgroup configuration
   - Error handling

**Total Effort**: 16 hours (~2 days)

---

## Future Enhancements (Out of Scope for Phase 2)

- Auto-batching with timeout (queue paths, auto-flush after 10ms)
- Async batch lookup with futures
- Adaptive batch sizing based on directory sizes
- Multi-level batching (batch directories within batch paths)
- Persistent batch mode for high-throughput scenarios

---

## References

- Benchmark results: `docs/PERFORMANCE_ANALYSIS.md`
- Single lookup implementation: `src/gpu_os/filesystem.rs:lookup_path()`
- Metal compute shaders: Apple Metal Programming Guide
