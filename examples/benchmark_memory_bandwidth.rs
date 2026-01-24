// Benchmark: What's our theoretical maximum?
//
// Tests raw memory bandwidth vs search performance to identify bottlenecks.

use metal::*;
use std::time::Instant;

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Test 1: Raw memory read (theoretical max)
// Just sum all bytes - measures pure memory bandwidth
kernel void bandwidth_test(
    device const uchar4* data [[buffer(0)]],
    device atomic_uint& result [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        uchar4 v = data[gid];
        uint sum = v.x + v.y + v.z + v.w;
        atomic_fetch_add_explicit(&result, sum, memory_order_relaxed);
    }
}

// Test 2: Vectorized search (4 bytes at a time)
// Searches for a single byte pattern using SIMD
kernel void vectorized_search(
    device const uchar4* data [[buffer(0)]],
    device atomic_uint& match_count [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uchar& pattern [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        uchar4 v = data[gid];
        uint matches = 0;
        if (v.x == pattern) matches++;
        if (v.y == pattern) matches++;
        if (v.z == pattern) matches++;
        if (v.w == pattern) matches++;
        if (matches > 0) {
            atomic_fetch_add_explicit(&match_count, matches, memory_order_relaxed);
        }
    }
}

// Test 3: Brute force multi-byte search (vectorized load, scalar compare)
kernel void brute_force_search(
    device const uchar* data [[buffer(0)]],
    device atomic_uint& match_count [[buffer(1)]],
    constant uint& byte_count [[buffer(2)]],
    constant uchar* pattern [[buffer(3)]],
    constant uint& pattern_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread checks one position
    if (gid + pattern_len <= byte_count) {
        bool match = true;
        for (uint i = 0; i < pattern_len && match; i++) {
            if (data[gid + i] != pattern[i]) {
                match = false;
            }
        }
        if (match) {
            atomic_fetch_add_explicit(&match_count, 1, memory_order_relaxed);
        }
    }
}

// Test 4: Warp-cooperative search (32 threads search together)
// Each thread loads 4 bytes, warp checks for pattern across all
kernel void warp_search(
    device const uchar4* data [[buffer(0)]],
    device atomic_uint& match_count [[buffer(1)]],
    constant uint& vec4_count [[buffer(2)]],
    constant uchar& target [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each SIMD lane loads 4 bytes
    if (gid < vec4_count) {
        uchar4 v = data[gid];

        // Count matches in this thread's 4 bytes
        uint local_matches = 0;
        if (v.x == target) local_matches++;
        if (v.y == target) local_matches++;
        if (v.z == target) local_matches++;
        if (v.w == target) local_matches++;

        // Reduce across SIMD group (32 threads)
        uint total = simd_sum(local_matches);

        // Only lane 0 writes
        if (simd_lane == 0 && total > 0) {
            atomic_fetch_add_explicit(&match_count, total, memory_order_relaxed);
        }
    }
}
"#;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    println!("=== Memory Bandwidth Benchmark ===\n");
    println!("Device: {}", device.name());

    // Create large test data (100 MB)
    let data_size = 100 * 1024 * 1024;
    let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

    println!("Data size: {} MB\n", data_size / 1024 / 1024);

    // Create buffer
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        data_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let result_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let count_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let pattern_buffer = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
    let pattern_len_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    // Compile shaders
    let library = device.new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
        .expect("Failed to compile shaders");

    let bandwidth_fn = library.get_function("bandwidth_test", None).unwrap();
    let vectorized_fn = library.get_function("vectorized_search", None).unwrap();
    let brute_force_fn = library.get_function("brute_force_search", None).unwrap();
    let warp_fn = library.get_function("warp_search", None).unwrap();

    let bandwidth_pipeline = device.new_compute_pipeline_state_with_function(&bandwidth_fn).unwrap();
    let vectorized_pipeline = device.new_compute_pipeline_state_with_function(&vectorized_fn).unwrap();
    let brute_force_pipeline = device.new_compute_pipeline_state_with_function(&brute_force_fn).unwrap();
    let warp_pipeline = device.new_compute_pipeline_state_with_function(&warp_fn).unwrap();

    let queue = device.new_command_queue();

    // Test 1: Raw bandwidth
    println!("{}", "─".repeat(60));
    println!("Test 1: Raw Memory Bandwidth (uchar4 loads)");
    {
        let vec4_count = data_size / 4;
        unsafe {
            *(result_buffer.contents() as *mut u32) = 0;
            *(count_buffer.contents() as *mut u32) = vec4_count as u32;
        }

        // Warmup
        for _ in 0..3 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&bandwidth_pipeline);
            enc.set_buffer(0, Some(&buffer), 0);
            enc.set_buffer(1, Some(&result_buffer), 0);
            enc.set_buffer(2, Some(&count_buffer), 0);
            enc.dispatch_threads(
                MTLSize::new(vec4_count as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&bandwidth_pipeline);
            enc.set_buffer(0, Some(&buffer), 0);
            enc.set_buffer(1, Some(&result_buffer), 0);
            enc.set_buffer(2, Some(&count_buffer), 0);
            enc.dispatch_threads(
                MTLSize::new(vec4_count as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed();

        let total_bytes = data_size * iterations;
        let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
        println!("  Throughput: {:.1} GB/s", throughput);
    }

    // Test 2: Vectorized single-byte search
    println!("\nTest 2: Vectorized Single-Byte Search (uchar4 + compare)");
    {
        let vec4_count = data_size / 4;
        let pattern: u8 = b'e';
        unsafe {
            *(result_buffer.contents() as *mut u32) = 0;
            *(count_buffer.contents() as *mut u32) = vec4_count as u32;
            *(pattern_buffer.contents() as *mut u8) = pattern;
        }

        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            unsafe { *(result_buffer.contents() as *mut u32) = 0; }

            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&vectorized_pipeline);
            enc.set_buffer(0, Some(&buffer), 0);
            enc.set_buffer(1, Some(&result_buffer), 0);
            enc.set_buffer(2, Some(&count_buffer), 0);
            enc.set_buffer(3, Some(&pattern_buffer), 0);
            enc.dispatch_threads(
                MTLSize::new(vec4_count as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed();

        let matches = unsafe { *(result_buffer.contents() as *const u32) };
        let total_bytes = data_size * iterations;
        let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
        println!("  Throughput: {:.1} GB/s ({} matches per iteration)", throughput, matches);
    }

    // Test 3: Brute force multi-byte search
    println!("\nTest 3: Brute Force Multi-Byte Search (one thread per position)");
    {
        let pattern = b"fn ";
        unsafe {
            *(result_buffer.contents() as *mut u32) = 0;
            *(count_buffer.contents() as *mut u32) = data_size as u32;
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), pattern_buffer.contents() as *mut u8, pattern.len());
            *(pattern_len_buffer.contents() as *mut u32) = pattern.len() as u32;
        }

        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            unsafe { *(result_buffer.contents() as *mut u32) = 0; }

            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&brute_force_pipeline);
            enc.set_buffer(0, Some(&buffer), 0);
            enc.set_buffer(1, Some(&result_buffer), 0);
            enc.set_buffer(2, Some(&count_buffer), 0);
            enc.set_buffer(3, Some(&pattern_buffer), 0);
            enc.set_buffer(4, Some(&pattern_len_buffer), 0);
            enc.dispatch_threads(
                MTLSize::new(data_size as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed();

        let matches = unsafe { *(result_buffer.contents() as *const u32) };
        let total_bytes = data_size * iterations;
        let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
        println!("  Throughput: {:.1} GB/s ({} matches per iteration)", throughput, matches);
    }

    // Test 4: SIMD-cooperative search
    println!("\nTest 4: SIMD-Cooperative Search (simd_sum reduction)");
    {
        let vec4_count = data_size / 4;
        let pattern: u8 = b'e';
        unsafe {
            *(result_buffer.contents() as *mut u32) = 0;
            *(count_buffer.contents() as *mut u32) = vec4_count as u32;
            *(pattern_buffer.contents() as *mut u8) = pattern;
        }

        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            unsafe { *(result_buffer.contents() as *mut u32) = 0; }

            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&warp_pipeline);
            enc.set_buffer(0, Some(&buffer), 0);
            enc.set_buffer(1, Some(&result_buffer), 0);
            enc.set_buffer(2, Some(&count_buffer), 0);
            enc.set_buffer(3, Some(&pattern_buffer), 0);
            enc.dispatch_threads(
                MTLSize::new(vec4_count as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed();

        let matches = unsafe { *(result_buffer.contents() as *const u32) };
        let total_bytes = data_size * iterations;
        let throughput = total_bytes as f64 / elapsed.as_secs_f64() / 1e9;
        println!("  Throughput: {:.1} GB/s ({} matches per iteration)", throughput, matches);
    }

    println!("\n{}", "─".repeat(60));
    println!("\n=== Analysis ===\n");
    println!("To reach 200 GB/s, we need:");
    println!("1. Vectorized loads (uchar4/uint4) - avoid byte-by-byte access");
    println!("2. SIMD reductions (simd_sum) - avoid per-thread atomics");
    println!("3. Large data (100MB+) - amortize dispatch overhead");
    println!("4. Simplified algorithm - brute force can beat Boyer-Moore on GPU");
    println!("5. Skip line counting - defer to CPU for the few matches");
}
