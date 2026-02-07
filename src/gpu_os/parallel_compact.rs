// Issue #126: GPU Parallel Compaction with Prefix Sum (Blelloch Scan)
//
// THE GPU IS THE COMPUTER. Use optimal GPU algorithms.
//
// Problem: 10,000 small buffers = scattered memory, poor cache locality
// Solution: GPU parallel compaction using Blelloch scan for O(log N) offset computation
//
// Algorithm:
// 1. Parallel prefix sum (scan) to compute offsets: O(log N) depth
// 2. Parallel scatter to copy data to computed offsets: O(1) depth
//
// Total: O(N) work, O(log N) depth - optimal for GPU!

use metal::*;

/// Metal shader source for parallel compaction
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Blelloch Parallel Prefix Sum (Exclusive Scan)
// Work-efficient: O(N) work, O(log N) depth
// ============================================================================

// Up-sweep (reduce) phase
// Builds a tree of partial sums
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

// Down-sweep phase
// Propagates partial sums down the tree
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

// Clear last element for exclusive scan
kernel void clear_last(
    device uint* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0 && n > 0) {
        data[n - 1] = 0;
    }
}

// ============================================================================
// Parallel Scatter - Copy data to computed offsets
// Each thread handles one chunk
// ============================================================================

// File descriptor for scatter operation
struct FileDescriptor {
    uint src_offset;   // Source offset in input buffer
    uint dst_offset;   // Destination offset (from prefix sum)
    uint size;         // Size in bytes
    uint file_index;   // Original file index
};

// Basic scatter: each thread copies one file
kernel void parallel_scatter(
    device uint8_t* dest [[buffer(0)]],
    device uint8_t* source [[buffer(1)]],
    device FileDescriptor* descriptors [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    FileDescriptor desc = descriptors[tid];

    // Copy bytes
    for (uint i = 0; i < desc.size; i++) {
        dest[desc.dst_offset + i] = source[desc.src_offset + i];
    }
}

// Vectorized scatter: copy 16 bytes at a time using uint4
kernel void parallel_scatter_vectorized(
    device uint4* dest [[buffer(0)]],
    device uint4* source [[buffer(1)]],
    device FileDescriptor* descriptors [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    FileDescriptor desc = descriptors[tid];

    // Aligned offsets for uint4 (16 bytes)
    uint src_vec_off = desc.src_offset / 16;
    uint dst_vec_off = desc.dst_offset / 16;
    uint vec_count = desc.size / 16;

    // Copy 16 bytes at a time
    for (uint i = 0; i < vec_count; i++) {
        dest[dst_vec_off + i] = source[src_vec_off + i];
    }

    // Handle remainder (not implemented for simplicity)
    // In production, would handle non-aligned sizes
}

// SIMD-group scatter: 32 threads cooperate to copy one large file
kernel void parallel_scatter_simd(
    device uint8_t* dest [[buffer(0)]],
    device uint8_t* source [[buffer(1)]],
    device FileDescriptor* descriptors [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Each SIMD group handles one file
    FileDescriptor desc = descriptors[simd_id];

    // Each lane copies every 32nd byte
    for (uint i = lane; i < desc.size; i += 32) {
        dest[desc.dst_offset + i] = source[desc.src_offset + i];
    }
}

// ============================================================================
// Utility kernels
// ============================================================================

// Copy sizes to offsets buffer (for in-place scan)
kernel void copy_sizes_to_offsets(
    device uint* sizes [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    offsets[tid] = sizes[tid];
}

// Compute total size (read last offset + last size after scan)
kernel void compute_total_size(
    device uint* offsets [[buffer(0)]],
    device uint* sizes [[buffer(1)]],
    device uint* total [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0 && n > 0) {
        *total = offsets[n - 1] + sizes[n - 1];
    }
}
"#;

/// File descriptor for scatter operation
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ScatterDescriptor {
    pub src_offset: u32,
    pub dst_offset: u32,
    pub size: u32,
    pub file_index: u32,
}

/// Result of parallel compaction
pub struct CompactedBuffer {
    /// Single contiguous buffer with all data
    pub buffer: Buffer,
    /// Per-file offsets in the buffer
    pub offsets: Vec<u32>,
    /// Per-file sizes
    pub sizes: Vec<u32>,
    /// Total bytes
    pub total_bytes: usize,
}

impl CompactedBuffer {
    /// Get slice for a specific file
    pub fn file_slice(&self, index: usize) -> Option<&[u8]> {
        let offset = *self.offsets.get(index)? as usize;
        let size = *self.sizes.get(index)? as usize;

        let ptr = self.buffer.contents() as *const u8;
        Some(unsafe {
            std::slice::from_raw_parts(ptr.add(offset), size)
        })
    }

    /// Get file offset
    pub fn offset(&self, index: usize) -> Option<u32> {
        self.offsets.get(index).copied()
    }

    /// Get file size
    pub fn size(&self, index: usize) -> Option<u32> {
        self.sizes.get(index).copied()
    }

    /// Number of files
    pub fn file_count(&self) -> usize {
        self.offsets.len()
    }
}

/// GPU Parallel Compactor using Blelloch scan
pub struct GpuParallelCompactor {
    device: Device,
    queue: CommandQueue,
    upsweep_pipeline: ComputePipelineState,
    downsweep_pipeline: ComputePipelineState,
    clear_last_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
    #[allow(dead_code)]
    copy_sizes_pipeline: ComputePipelineState,
    #[allow(dead_code)]
    compute_total_pipeline: ComputePipelineState,
}

impl GpuParallelCompactor {
    /// Create a new parallel compactor
    pub fn new(device: &Device) -> Result<Self, String> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile shaders: {}", e))?;

        let upsweep = library.get_function("prefix_sum_upsweep", None)
            .map_err(|e| format!("Failed to get upsweep function: {}", e))?;
        let downsweep = library.get_function("prefix_sum_downsweep", None)
            .map_err(|e| format!("Failed to get downsweep function: {}", e))?;
        let clear_last = library.get_function("clear_last", None)
            .map_err(|e| format!("Failed to get clear_last function: {}", e))?;
        let scatter = library.get_function("parallel_scatter", None)
            .map_err(|e| format!("Failed to get scatter function: {}", e))?;
        let copy_sizes = library.get_function("copy_sizes_to_offsets", None)
            .map_err(|e| format!("Failed to get copy_sizes function: {}", e))?;
        let compute_total = library.get_function("compute_total_size", None)
            .map_err(|e| format!("Failed to get compute_total function: {}", e))?;

        Ok(Self {
            device: device.clone(),
            queue: device.new_command_queue(),
            upsweep_pipeline: device.new_compute_pipeline_state_with_function(&upsweep)
                .map_err(|e| format!("Failed to create upsweep pipeline: {}", e))?,
            downsweep_pipeline: device.new_compute_pipeline_state_with_function(&downsweep)
                .map_err(|e| format!("Failed to create downsweep pipeline: {}", e))?,
            clear_last_pipeline: device.new_compute_pipeline_state_with_function(&clear_last)
                .map_err(|e| format!("Failed to create clear_last pipeline: {}", e))?,
            scatter_pipeline: device.new_compute_pipeline_state_with_function(&scatter)
                .map_err(|e| format!("Failed to create scatter pipeline: {}", e))?,
            copy_sizes_pipeline: device.new_compute_pipeline_state_with_function(&copy_sizes)
                .map_err(|e| format!("Failed to create copy_sizes pipeline: {}", e))?,
            compute_total_pipeline: device.new_compute_pipeline_state_with_function(&compute_total)
                .map_err(|e| format!("Failed to create compute_total pipeline: {}", e))?,
        })
    }

    /// Compute exclusive prefix sum on GPU
    /// Input: [3, 1, 7, 0, 4, 1, 6, 3]
    /// Output: [0, 3, 4, 11, 11, 15, 16, 22]
    pub fn prefix_sum(&self, sizes: &[u32]) -> Vec<u32> {
        let n = sizes.len();
        if n == 0 {
            return vec![];
        }

        // For small arrays, CPU is faster
        if n < 256 {
            return Self::cpu_prefix_sum(sizes);
        }

        // Round up to power of 2 for Blelloch scan
        let padded_n = n.next_power_of_two();

        // Create buffer with padding
        let mut padded_sizes = sizes.to_vec();
        padded_sizes.resize(padded_n, 0);

        let buffer = self.device.new_buffer_with_data(
            padded_sizes.as_ptr() as *const _,
            (padded_n * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create stride buffers (pre-allocate for all phases)
        let log_n = (padded_n as f64).log2().ceil() as usize;
        let mut stride_buffers: Vec<Buffer> = Vec::with_capacity(log_n);
        let mut stride = 1u32;
        while stride < padded_n as u32 {
            stride_buffers.push(self.device.new_buffer_with_data(
                &stride as *const _ as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            ));
            stride *= 2;
        }

        let n_buffer = self.device.new_buffer_with_data(
            &(padded_n as u32) as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        // Single command buffer for entire scan
        let cmd = self.queue.new_command_buffer();

        // Up-sweep phase
        stride = 1;
        let mut stride_idx = 0;
        while stride < padded_n as u32 {
            let threads = padded_n as u32 / (stride * 2);
            if threads == 0 {
                break;
            }

            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.upsweep_pipeline);
            encoder.set_buffer(0, Some(&buffer), 0);
            encoder.set_buffer(1, Some(&n_buffer), 0);
            encoder.set_buffer(2, Some(&stride_buffers[stride_idx]), 0);
            encoder.dispatch_threads(
                MTLSize::new(threads as u64, 1, 1),
                MTLSize::new(threads.min(256) as u64, 1, 1),
            );
            encoder.end_encoding();

            stride *= 2;
            stride_idx += 1;
        }

        // Clear last element
        {
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.clear_last_pipeline);
            encoder.set_buffer(0, Some(&buffer), 0);
            encoder.set_buffer(1, Some(&n_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
        }

        // Down-sweep phase
        stride = padded_n as u32 / 2;
        while stride >= 1 {
            let threads = padded_n as u32 / (stride * 2);
            if threads == 0 {
                break;
            }

            stride_idx = stride_idx.saturating_sub(1);
            let encoder = cmd.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.downsweep_pipeline);
            encoder.set_buffer(0, Some(&buffer), 0);
            encoder.set_buffer(1, Some(&n_buffer), 0);
            encoder.set_buffer(2, Some(&stride_buffers[stride_idx]), 0);
            encoder.dispatch_threads(
                MTLSize::new(threads as u64, 1, 1),
                MTLSize::new(threads.min(256) as u64, 1, 1),
            );
            encoder.end_encoding();

            if stride == 1 {
                break;
            }
            stride /= 2;
        }

        // Single commit and wait
        cmd.commit();
        cmd.wait_until_completed();

        // Read result (only first n elements)
        let result_ptr = buffer.contents() as *const u32;
        let result: Vec<u32> = unsafe {
            std::slice::from_raw_parts(result_ptr, n).to_vec()
        };

        result
    }

    /// CPU prefix sum for small arrays (faster due to no GPU overhead)
    fn cpu_prefix_sum(sizes: &[u32]) -> Vec<u32> {
        let mut offsets = vec![0u32; sizes.len()];
        for i in 1..sizes.len() {
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        }
        offsets
    }

    /// Compact multiple data chunks into a single contiguous buffer
    pub fn compact(&self, chunks: &[&[u8]]) -> Result<CompactedBuffer, String> {
        if chunks.is_empty() {
            return Err("No chunks to compact".to_string());
        }

        // Get sizes
        let sizes: Vec<u32> = chunks.iter().map(|c| c.len() as u32).collect();

        // Compute offsets using GPU prefix sum
        let offsets = self.prefix_sum(&sizes);

        // Calculate total size
        let total_bytes: usize = sizes.iter().map(|&s| s as usize).sum();

        // Concatenate source data
        let mut source_data = Vec::with_capacity(total_bytes);
        for chunk in chunks {
            source_data.extend_from_slice(chunk);
        }

        // Create source buffer
        let source_buffer = self.device.new_buffer_with_data(
            source_data.as_ptr() as *const _,
            total_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create destination buffer
        let dest_buffer = self.device.new_buffer(
            total_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create scatter descriptors
        let mut src_offset = 0u32;
        let descriptors: Vec<ScatterDescriptor> = sizes.iter()
            .zip(offsets.iter())
            .enumerate()
            .map(|(i, (&size, &dst_offset))| {
                let desc = ScatterDescriptor {
                    src_offset,
                    dst_offset,
                    size,
                    file_index: i as u32,
                };
                src_offset += size;
                desc
            })
            .collect();

        let desc_buffer = self.device.new_buffer_with_data(
            descriptors.as_ptr() as *const _,
            (descriptors.len() * std::mem::size_of::<ScatterDescriptor>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch scatter kernel
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.scatter_pipeline);
        encoder.set_buffer(0, Some(&dest_buffer), 0);
        encoder.set_buffer(1, Some(&source_buffer), 0);
        encoder.set_buffer(2, Some(&desc_buffer), 0);
        encoder.dispatch_threads(
            MTLSize::new(chunks.len() as u64, 1, 1),
            MTLSize::new(chunks.len().min(256) as u64, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        Ok(CompactedBuffer {
            buffer: dest_buffer,
            offsets,
            sizes,
            total_bytes,
        })
    }

    /// Compact MmapBuffers into a single buffer
    pub fn compact_buffers(&self, buffers: &[&super::mmap_buffer::MmapBuffer]) -> Result<CompactedBuffer, String> {
        // Extract data slices
        let chunks: Vec<&[u8]> = buffers.iter()
            .map(|b| unsafe {
                std::slice::from_raw_parts(b.as_ptr(), b.file_size())
            })
            .collect();

        self.compact(&chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Reference CPU implementation (wrapper for tests)
    fn cpu_prefix_sum(sizes: &[u32]) -> Vec<u32> {
        GpuParallelCompactor::cpu_prefix_sum(sizes)
    }

    #[test]
    fn test_compactor_creation() {
        let device = Device::system_default().expect("No Metal device");
        let compactor = GpuParallelCompactor::new(&device).expect("Failed to create compactor");
        println!("GpuParallelCompactor created successfully");
    }

    #[test]
    fn test_prefix_sum_small() {
        let device = Device::system_default().expect("No Metal device");
        let compactor = GpuParallelCompactor::new(&device).expect("Failed to create compactor");

        let sizes = vec![3, 1, 7, 0, 4, 1, 6, 3];
        let expected = cpu_prefix_sum(&sizes);
        let gpu_result = compactor.prefix_sum(&sizes);

        println!("Input:    {:?}", sizes);
        println!("Expected: {:?}", expected);
        println!("GPU:      {:?}", gpu_result);

        assert_eq!(gpu_result, expected);
    }

    #[test]
    fn test_prefix_sum_large() {
        let device = Device::system_default().expect("No Metal device");
        let compactor = GpuParallelCompactor::new(&device).expect("Failed to create compactor");

        // Test with 10,000 elements
        let sizes: Vec<u32> = (0..10000).map(|i| (i % 1000 + 100) as u32).collect();
        let expected = cpu_prefix_sum(&sizes);

        let start = Instant::now();
        let gpu_result = compactor.prefix_sum(&sizes);
        let elapsed = start.elapsed();

        println!("Prefix sum of {} elements in {:.2}ms", sizes.len(), elapsed.as_secs_f64() * 1000.0);

        assert_eq!(gpu_result, expected);
    }

    #[test]
    fn test_compact_basic() {
        let device = Device::system_default().expect("No Metal device");
        let compactor = GpuParallelCompactor::new(&device).expect("Failed to create compactor");

        // Create test chunks
        let chunks: Vec<Vec<u8>> = vec![
            vec![1, 2, 3],
            vec![4, 5],
            vec![6, 7, 8, 9],
            vec![10],
        ];

        let chunk_refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let result = compactor.compact(&chunk_refs).expect("Compact failed");

        // Verify offsets
        assert_eq!(result.offsets, vec![0, 3, 5, 9]);

        // Verify data
        for (i, chunk) in chunks.iter().enumerate() {
            let actual = result.file_slice(i).expect("Failed to get slice");
            assert_eq!(actual, chunk.as_slice(), "Chunk {} mismatch", i);
        }

        println!("Compact basic: OK");
    }

    #[test]
    fn benchmark_compact() {
        let device = Device::system_default().expect("No Metal device");
        let compactor = GpuParallelCompactor::new(&device).expect("Failed to create compactor");

        // Create 1000 chunks of varying sizes
        let chunks: Vec<Vec<u8>> = (0..1000)
            .map(|i| vec![(i % 256) as u8; 100 + (i % 500) as usize])
            .collect();

        let total_bytes: usize = chunks.iter().map(|c| c.len()).sum();
        println!("\n=== Compaction Benchmark ===");
        println!("Chunks: {}, Total: {:.1} KB", chunks.len(), total_bytes as f64 / 1024.0);

        // CPU baseline
        let cpu_start = Instant::now();
        let sizes: Vec<u32> = chunks.iter().map(|c| c.len() as u32).collect();
        let _cpu_offsets = cpu_prefix_sum(&sizes);
        let mut _cpu_result = Vec::with_capacity(total_bytes);
        for chunk in &chunks {
            _cpu_result.extend_from_slice(chunk);
        }
        let cpu_time = cpu_start.elapsed();

        // GPU compact
        let chunk_refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let gpu_start = Instant::now();
        let _gpu_result = compactor.compact(&chunk_refs).expect("Compact failed");
        let gpu_time = gpu_start.elapsed();

        println!("CPU: {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("GPU: {:.2}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    }

    #[test]
    fn benchmark_large_scatter() {
        let device = Device::system_default().expect("No Metal device");
        let compactor = GpuParallelCompactor::new(&device).expect("Failed to create compactor");

        // Create 10,000 chunks totaling ~50MB (where GPU scatter should win)
        let chunks: Vec<Vec<u8>> = (0..10000)
            .map(|i| vec![(i % 256) as u8; 5000 + (i % 1000) as usize])
            .collect();

        let total_bytes: usize = chunks.iter().map(|c| c.len()).sum();
        println!("\n=== Large Scatter Benchmark ===");
        println!("Chunks: {}, Total: {:.1} MB", chunks.len(), total_bytes as f64 / 1024.0 / 1024.0);

        // CPU baseline: just memcpy
        let cpu_start = Instant::now();
        let mut cpu_result = Vec::with_capacity(total_bytes);
        for chunk in &chunks {
            cpu_result.extend_from_slice(chunk);
        }
        let cpu_time = cpu_start.elapsed();
        drop(cpu_result);

        // GPU compact (includes prefix sum + scatter)
        let chunk_refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let gpu_start = Instant::now();
        let _gpu_result = compactor.compact(&chunk_refs).expect("Compact failed");
        let gpu_time = gpu_start.elapsed();

        let cpu_throughput = (total_bytes as f64 / (1024.0 * 1024.0)) / cpu_time.as_secs_f64();
        let gpu_throughput = (total_bytes as f64 / (1024.0 * 1024.0)) / gpu_time.as_secs_f64();

        println!("CPU memcpy: {:.1}ms ({:.1} GB/s)", cpu_time.as_secs_f64() * 1000.0, cpu_throughput / 1024.0);
        println!("GPU scatter: {:.1}ms ({:.1} GB/s)", gpu_time.as_secs_f64() * 1000.0, gpu_throughput / 1024.0);
        println!("GPU/CPU: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    }
}
