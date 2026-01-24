// Issue #126: GPU Parallel Compaction with Prefix Sum
//
// THE GPU IS THE COMPUTER - Use Blelloch scan for O(log N) offset computation

use metal::*;
use std::time::Instant;

// These will be implemented in src/gpu_os/parallel_compact.rs
// use rust_experiment::gpu_os::parallel_compact::{GpuParallelCompactor, CompactedBuffer};

/// Reference CPU implementation of exclusive prefix sum
fn cpu_prefix_sum(sizes: &[u32]) -> Vec<u32> {
    let mut offsets = vec![0u32; sizes.len()];
    for i in 1..sizes.len() {
        offsets[i] = offsets[i - 1] + sizes[i - 1];
    }
    offsets
}

/// Reference CPU compaction
fn cpu_compact(data: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>) {
    let sizes: Vec<u32> = data.iter().map(|d| d.len() as u32).collect();
    let offsets = cpu_prefix_sum(&sizes);
    let total_size: usize = sizes.iter().map(|&s| s as usize).sum();

    let mut result = vec![0u8; total_size];
    for (i, chunk) in data.iter().enumerate() {
        let offset = offsets[i] as usize;
        result[offset..offset + chunk.len()].copy_from_slice(chunk);
    }

    (result, offsets)
}

#[test]
fn test_cpu_prefix_sum_reference() {
    // Verify our reference implementation
    let sizes = vec![100, 200, 150, 300, 50];
    let expected = vec![0, 100, 300, 450, 750];

    let offsets = cpu_prefix_sum(&sizes);

    assert_eq!(offsets, expected);
    println!("CPU prefix sum reference: OK");
}

#[test]
fn test_cpu_compact_reference() {
    // Test data: 5 chunks of varying sizes
    let data: Vec<Vec<u8>> = vec![
        vec![1, 2, 3],
        vec![4, 5],
        vec![6, 7, 8, 9],
        vec![10],
        vec![11, 12, 13, 14, 15],
    ];

    let (compacted, offsets) = cpu_compact(&data);

    // Verify offsets
    assert_eq!(offsets, vec![0, 3, 5, 9, 10]);

    // Verify data integrity
    assert_eq!(compacted.len(), 15);
    assert_eq!(&compacted[0..3], &[1, 2, 3]);
    assert_eq!(&compacted[3..5], &[4, 5]);
    assert_eq!(&compacted[5..9], &[6, 7, 8, 9]);
    assert_eq!(&compacted[9..10], &[10]);
    assert_eq!(&compacted[10..15], &[11, 12, 13, 14, 15]);

    println!("CPU compact reference: OK");
}

#[test]
fn test_prefix_sum_metal_shader() {
    let device = Device::system_default().expect("No Metal device");

    // Compile prefix sum shaders
    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        // Up-sweep (reduce) phase of Blelloch scan
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

        // Down-sweep phase of Blelloch scan
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

        // Set last element to zero (for exclusive scan)
        kernel void clear_last(
            device uint* data [[buffer(0)]],
            constant uint& n [[buffer(1)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid == 0) {
                data[n - 1] = 0;
            }
        }
    "#;

    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .expect("Failed to compile shader");

    let upsweep = library.get_function("prefix_sum_upsweep", None).unwrap();
    let downsweep = library.get_function("prefix_sum_downsweep", None).unwrap();
    let clear_last = library.get_function("clear_last", None).unwrap();

    let upsweep_pipeline = device.new_compute_pipeline_state_with_function(&upsweep).unwrap();
    let downsweep_pipeline = device.new_compute_pipeline_state_with_function(&downsweep).unwrap();
    let clear_last_pipeline = device.new_compute_pipeline_state_with_function(&clear_last).unwrap();

    // Test data - power of 2 for simplicity
    let n = 8u32;
    let input: Vec<u32> = vec![3, 1, 7, 0, 4, 1, 6, 3];
    let expected = cpu_prefix_sum(&input);

    // Create GPU buffer
    let buffer = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (n as usize * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();

    // Up-sweep phase
    let mut stride = 1u32;
    while stride < n {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&upsweep_pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.set_bytes(1, 4, &n as *const _ as *const _);
        encoder.set_bytes(2, 4, &stride as *const _ as *const _);

        let threads = n / (stride * 2);
        encoder.dispatch_threads(
            MTLSize::new(threads as u64, 1, 1),
            MTLSize::new(threads.min(256) as u64, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        stride *= 2;
    }

    // Clear last element
    {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&clear_last_pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.set_bytes(1, 4, &n as *const _ as *const _);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Down-sweep phase
    stride = n / 2;
    while stride >= 1 {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&downsweep_pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.set_bytes(1, 4, &n as *const _ as *const _);
        encoder.set_bytes(2, 4, &stride as *const _ as *const _);

        let threads = n / (stride * 2);
        encoder.dispatch_threads(
            MTLSize::new(threads as u64, 1, 1),
            MTLSize::new(threads.min(256) as u64, 1, 1),
        );
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        if stride == 1 {
            break;
        }
        stride /= 2;
    }

    // Read result
    let result_ptr = buffer.contents() as *const u32;
    let result: Vec<u32> = unsafe {
        std::slice::from_raw_parts(result_ptr, n as usize).to_vec()
    };

    println!("Input:    {:?}", input);
    println!("Expected: {:?}", expected);
    println!("GPU:      {:?}", result);

    assert_eq!(result, expected, "GPU prefix sum mismatch!");
    println!("GPU prefix sum: OK");
}

#[test]
fn test_parallel_scatter_shader() {
    let device = Device::system_default().expect("No Metal device");

    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        // Parallel scatter: each thread copies one chunk
        kernel void parallel_scatter(
            device uint8_t* dest [[buffer(0)]],
            device uint8_t* source [[buffer(1)]],
            device uint* src_offsets [[buffer(2)]],
            device uint* dst_offsets [[buffer(3)]],
            device uint* sizes [[buffer(4)]],
            uint tid [[thread_position_in_grid]]
        ) {
            uint src_off = src_offsets[tid];
            uint dst_off = dst_offsets[tid];
            uint size = sizes[tid];

            for (uint i = 0; i < size; i++) {
                dest[dst_off + i] = source[src_off + i];
            }
        }
    "#;

    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .expect("Failed to compile shader");

    let scatter_fn = library.get_function("parallel_scatter", None).unwrap();
    let scatter_pipeline = device.new_compute_pipeline_state_with_function(&scatter_fn).unwrap();

    // Test data: 4 chunks
    let chunks: Vec<Vec<u8>> = vec![
        vec![1, 2, 3],
        vec![4, 5],
        vec![6, 7, 8, 9],
        vec![10],
    ];

    // Flatten source data
    let source_data: Vec<u8> = chunks.iter().flatten().copied().collect();
    let sizes: Vec<u32> = chunks.iter().map(|c| c.len() as u32).collect();

    // Compute offsets
    let src_offsets = cpu_prefix_sum(&sizes);
    let dst_offsets = cpu_prefix_sum(&sizes); // Same for this test

    let total_size: u32 = sizes.iter().sum();

    // Create buffers
    let dest_buffer = device.new_buffer(total_size as u64, MTLResourceOptions::StorageModeShared);
    let source_buffer = device.new_buffer_with_data(
        source_data.as_ptr() as *const _,
        source_data.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let src_offsets_buffer = device.new_buffer_with_data(
        src_offsets.as_ptr() as *const _,
        (src_offsets.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dst_offsets_buffer = device.new_buffer_with_data(
        dst_offsets.as_ptr() as *const _,
        (dst_offsets.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sizes_buffer = device.new_buffer_with_data(
        sizes.as_ptr() as *const _,
        (sizes.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Dispatch scatter
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&scatter_pipeline);
    encoder.set_buffer(0, Some(&dest_buffer), 0);
    encoder.set_buffer(1, Some(&source_buffer), 0);
    encoder.set_buffer(2, Some(&src_offsets_buffer), 0);
    encoder.set_buffer(3, Some(&dst_offsets_buffer), 0);
    encoder.set_buffer(4, Some(&sizes_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new(chunks.len() as u64, 1, 1),
        MTLSize::new(chunks.len() as u64, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify result
    let result_ptr = dest_buffer.contents() as *const u8;
    let result: Vec<u8> = unsafe {
        std::slice::from_raw_parts(result_ptr, total_size as usize).to_vec()
    };

    assert_eq!(result, source_data, "Scatter mismatch!");
    println!("GPU parallel scatter: OK");
}

#[test]
fn benchmark_prefix_sum_scaling() {
    let device = Device::system_default().expect("No Metal device");

    println!("\n=== Prefix Sum Scaling Benchmark ===\n");

    for &n in &[1000, 10000, 100000, 1000000] {
        // Generate random sizes
        let sizes: Vec<u32> = (0..n).map(|i| (i % 1000 + 100) as u32).collect();

        // CPU timing
        let cpu_start = Instant::now();
        let _cpu_result = cpu_prefix_sum(&sizes);
        let cpu_time = cpu_start.elapsed();

        // Note: GPU timing requires full implementation
        // For now, just estimate based on O(log N) depth

        let log_n = (n as f64).log2().ceil() as u32;
        let estimated_gpu_dispatches = log_n * 2; // up-sweep + down-sweep

        println!("N = {:>7}: CPU {:.2}ms, GPU ~{} dispatches (O(log N) = {})",
            n,
            cpu_time.as_secs_f64() * 1000.0,
            estimated_gpu_dispatches,
            log_n);
    }
}

// ============================================================================
// Tests for full compactor (to be implemented)
// ============================================================================

#[test]
#[ignore = "Requires GpuParallelCompactor implementation"]
fn test_compactor_correctness() {
    // let device = Device::system_default().expect("No Metal device");
    // let compactor = GpuParallelCompactor::new(&device).unwrap();

    // // Generate test data
    // let chunks: Vec<Vec<u8>> = (0..1000)
    //     .map(|i| vec![(i % 256) as u8; 100 + (i % 500)])
    //     .collect();

    // // GPU compact
    // let gpu_result = compactor.compact(&chunks);

    // // CPU reference
    // let (cpu_data, cpu_offsets) = cpu_compact(&chunks);

    // // Verify offsets
    // let gpu_offsets = gpu_result.offsets();
    // assert_eq!(gpu_offsets, cpu_offsets, "Offset mismatch");

    // // Verify data
    // let gpu_data = gpu_result.data();
    // assert_eq!(gpu_data, cpu_data.as_slice(), "Data mismatch");
}

#[test]
#[ignore = "Requires GpuParallelCompactor implementation"]
fn benchmark_compactor_vs_cpu() {
    // let device = Device::system_default().expect("No Metal device");
    // let compactor = GpuParallelCompactor::new(&device).unwrap();

    // // Generate 10,000 chunks totaling ~100MB
    // let chunks: Vec<Vec<u8>> = (0..10000)
    //     .map(|i| vec![0u8; 10000 + (i % 5000)])
    //     .collect();

    // let total_bytes: usize = chunks.iter().map(|c| c.len()).sum();
    // println!("Compacting {} chunks ({:.1} MB)", chunks.len(), total_bytes as f64 / 1e6);

    // // CPU timing
    // let cpu_start = Instant::now();
    // let _cpu_result = cpu_compact(&chunks);
    // let cpu_time = cpu_start.elapsed();

    // // GPU timing
    // let gpu_start = Instant::now();
    // let _gpu_result = compactor.compact(&chunks);
    // let gpu_time = gpu_start.elapsed();

    // println!("CPU: {:.1}ms", cpu_time.as_secs_f64() * 1000.0);
    // println!("GPU: {:.1}ms", gpu_time.as_secs_f64() * 1000.0);
    // println!("Speedup: {:.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());

    // // GPU should be at least 10x faster for large data
    // assert!(gpu_time < cpu_time / 10, "Expected 10x+ speedup");
}
