// Issue #104: GPU Branch Optimization - SIMD Warp-Level Primitives for Collective Operations
//
// PRD: Replace loop-based reductions with simd_shuffle, simd_reduce_add, simd_prefix_exclusive_sum.
//
// Target: Close the 1.6x CPU advantage in reduction workloads.
//
// This test compares:
// 1. Loop-based reduction (current baseline)
// 2. SIMD shuffle reduction (register-only, no shared memory for warp)
// 3. SIMD reduce_add (hardware primitive)

use metal::*;
use std::time::Instant;

const DATA_SIZE: u32 = 10_000_000;
const WARM_UP_RUNS: usize = 3;
const TIMED_RUNS: usize = 10;

// ============================================================================
// GPU Shaders
// ============================================================================

const GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// BASELINE: Loop-based reduction with shared memory
kernel void reduce_loop_based(
    device const float* data [[buffer(0)]],
    device atomic_float* result [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Load into shared memory
    shared[local_tid] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup (loop-based)
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared[local_tid] += shared[local_tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 adds to global result
    if (local_tid == 0) {
        atomic_fetch_add_explicit(result, shared[0], memory_order_relaxed);
    }
}

// SIMD SHUFFLE: Use simd_shuffle_xor for warp-level reduction
// No shared memory needed for warp-local reduction (32 threads)
kernel void reduce_simd_shuffle(
    device const float* data [[buffer(0)]],
    device atomic_float* result [[buffer(1)]],
    threadgroup float* warp_sums [[threadgroup(0)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Grid-stride loop for better memory bandwidth utilization
    float local_sum = 0.0;
    uint grid_size = tg_size * ((element_count + tg_size - 1) / tg_size);
    for (uint i = tid; i < element_count; i += grid_size) {
        local_sum += data[i];
    }

    // Warp-level reduction using shuffle (register-only, no shared memory!)
    // This is 32 threads reducing to 1 value using butterfly pattern
    local_sum += simd_shuffle_xor(local_sum, 16);
    local_sum += simd_shuffle_xor(local_sum, 8);
    local_sum += simd_shuffle_xor(local_sum, 4);
    local_sum += simd_shuffle_xor(local_sum, 2);
    local_sum += simd_shuffle_xor(local_sum, 1);

    // Lane 0 of each warp writes to shared memory
    uint num_warps = tg_size / 32;
    if (simd_lane == 0) {
        warp_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp reduces warp sums
    if (simd_group == 0 && simd_lane < num_warps) {
        float val = warp_sums[simd_lane];
        val += simd_shuffle_xor(val, 4);
        val += simd_shuffle_xor(val, 2);
        val += simd_shuffle_xor(val, 1);

        if (simd_lane == 0) {
            atomic_fetch_add_explicit(result, val, memory_order_relaxed);
        }
    }
}

// SIMD REDUCE_ADD: Use hardware primitive for warp reduction
kernel void reduce_simd_reduce(
    device const float* data [[buffer(0)]],
    device atomic_float* result [[buffer(1)]],
    threadgroup float* warp_sums [[threadgroup(0)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Grid-stride loop
    float local_sum = 0.0;
    uint grid_size = tg_size * ((element_count + tg_size - 1) / tg_size);
    for (uint i = tid; i < element_count; i += grid_size) {
        local_sum += data[i];
    }

    // Hardware warp reduction - single instruction!
    float warp_sum = simd_sum(local_sum);

    // Lane 0 of each warp writes to shared memory
    uint num_warps = tg_size / 32;
    if (simd_lane == 0) {
        warp_sums[simd_group] = warp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp reduces warp sums using simd_sum
    if (simd_group == 0 && simd_lane < num_warps) {
        float val = warp_sums[simd_lane];
        // Reduce the 8 warp sums (for 256-thread threadgroup)
        val = simd_sum(val);  // This sums all active lanes

        if (simd_lane == 0) {
            atomic_fetch_add_explicit(result, val, memory_order_relaxed);
        }
    }
}

// DECENTRALIZED: Last threadgroup does final sum (no atomic contention)
kernel void reduce_decentralized(
    device const float* data [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    device atomic_uint* completion_counter [[buffer(2)]],
    device float* final_result [[buffer(3)]],
    constant uint& element_count [[buffer(4)]],
    constant uint& num_threadgroups [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Grid-stride loop
    float local_sum = 0.0;
    uint grid_size = num_threadgroups * tg_size;
    for (uint i = tid; i < element_count; i += grid_size) {
        local_sum += data[i];
    }

    // Warp reduction
    local_sum = simd_sum(local_sum);

    // Threadgroup reduction using shared memory
    threadgroup float warp_sums[8];
    if (simd_lane == 0) {
        warp_sums[simd_group] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < (tg_size / 32)) {
        float val = warp_sums[simd_lane];
        val = simd_sum(val);

        if (simd_lane == 0) {
            // Write threadgroup sum to partial sums array
            partial_sums[tg_id] = val;
        }
    }

    // Ensure partial sum is written before incrementing counter
    threadgroup_barrier(mem_flags::mem_device);

    // Last threadgroup to finish does final sum
    if (local_tid == 0) {
        uint completed = atomic_fetch_add_explicit(completion_counter, 1, memory_order_relaxed) + 1;
        if (completed == num_threadgroups) {
            // We're the last threadgroup - sum all partial results
            float total = 0.0;
            for (uint i = 0; i < num_threadgroups; i++) {
                total += partial_sums[i];
            }
            *final_result = total;
        }
    }
}
"#;

// ============================================================================
// CPU Reference
// ============================================================================

fn cpu_reduce_sum(data: &[f32]) -> f32 {
    let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    let partial_sums: Vec<f32> = std::thread::scope(|s| {
        let chunk_size = (data.len() + num_threads - 1) / num_threads;
        data.chunks(chunk_size)
            .map(|chunk| s.spawn(move || chunk.iter().sum::<f32>()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    partial_sums.iter().sum()
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    median_ms: f64,
    std_dev_ms: f64,
    sum: f32,
    correct: bool,
}

impl BenchResult {
    fn new(name: &str, times: Vec<f64>, sum: f32, expected: f32) -> Self {
        let n = times.len() as f64;
        let mean = times.iter().sum::<f64>() / n;
        let variance = if times.len() > 1 {
            times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Check correctness (allow 0.1% error for f32 accumulation)
        let rel_error = (sum - expected).abs() / expected.abs().max(1.0);
        let correct = rel_error < 0.001;

        BenchResult {
            name: name.to_string(),
            median_ms: median,
            std_dev_ms: std_dev,
            sum,
            correct,
        }
    }
}

fn create_pipeline(device: &Device, function_name: &str) -> ComputePipelineState {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(GPU_SHADERS, &options)
        .expect("Shader compile failed");
    let function = library.get_function(function_name, None).expect("Function not found");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed")
}

// ============================================================================
// Benchmarks
// ============================================================================

fn run_loop_based(device: &Device, queue: &CommandQueue, expected: f32) -> BenchResult {
    let pipeline = create_pipeline(device, "reduce_loop_based");
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    let data_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut final_sum = 0.0f32;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let result_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(result_buf.contents() as *mut f32) = 0.0;
        }

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&result_buf), 0);
        enc.set_threadgroup_memory_length(0, 256 * 4);
        enc.dispatch_threads(
            MTLSize::new(DATA_SIZE as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
            final_sum = unsafe { *(result_buf.contents() as *const f32) };
        }
    }

    BenchResult::new("Loop-based", times, final_sum, expected)
}

fn run_simd_shuffle(device: &Device, queue: &CommandQueue, expected: f32) -> BenchResult {
    let pipeline = create_pipeline(device, "reduce_simd_shuffle");
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    let data_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut final_sum = 0.0f32;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let result_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(result_buf.contents() as *mut f32) = 0.0;
        }

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&result_buf), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        enc.set_threadgroup_memory_length(0, 8 * 4); // Only 8 warp sums
        enc.dispatch_threads(
            MTLSize::new(DATA_SIZE as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
            final_sum = unsafe { *(result_buf.contents() as *const f32) };
        }
    }

    BenchResult::new("SIMD Shuffle", times, final_sum, expected)
}

fn run_simd_reduce(device: &Device, queue: &CommandQueue, expected: f32) -> BenchResult {
    let pipeline = create_pipeline(device, "reduce_simd_reduce");
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    let data_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut final_sum = 0.0f32;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let result_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(result_buf.contents() as *mut f32) = 0.0;
        }

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&result_buf), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        enc.set_threadgroup_memory_length(0, 8 * 4);
        enc.dispatch_threads(
            MTLSize::new(DATA_SIZE as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
            final_sum = unsafe { *(result_buf.contents() as *const f32) };
        }
    }

    BenchResult::new("SIMD Reduce", times, final_sum, expected)
}

fn run_decentralized(device: &Device, queue: &CommandQueue, expected: f32) -> BenchResult {
    let pipeline = create_pipeline(device, "reduce_decentralized");
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    let data_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let threadgroup_size = 256u32;
    let num_threadgroups = (DATA_SIZE + threadgroup_size - 1) / threadgroup_size;

    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let num_tg_buf = device.new_buffer_with_data(
        &num_threadgroups as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut final_sum = 0.0f32;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let partial_buf = device.new_buffer(
            (num_threadgroups * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counter_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let result_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(counter_buf.contents() as *mut u32) = 0;
            *(result_buf.contents() as *mut f32) = 0.0;
        }

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&partial_buf), 0);
        enc.set_buffer(2, Some(&counter_buf), 0);
        enc.set_buffer(3, Some(&result_buf), 0);
        enc.set_buffer(4, Some(&count_buf), 0);
        enc.set_buffer(5, Some(&num_tg_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_threadgroups as u64, 1, 1),
            MTLSize::new(threadgroup_size as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
            final_sum = unsafe { *(result_buf.contents() as *const f32) };
        }
    }

    BenchResult::new("Decentralized", times, final_sum, expected)
}

fn run_cpu_benchmark(expected: f32) -> BenchResult {
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    let mut times = Vec::new();
    let mut final_sum = 0.0f32;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let start = Instant::now();
        let sum = cpu_reduce_sum(&data);
        let elapsed = start.elapsed();
        std::hint::black_box(sum);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
            final_sum = sum;
        }
    }

    BenchResult::new("CPU", times, final_sum, expected)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_simd_primitives_correctness() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n=== Issue #104: SIMD Primitives Correctness Test ===\n");

    let expected = DATA_SIZE as f32;  // Sum of 10M ones

    let results = vec![
        run_loop_based(&device, &queue, expected),
        run_simd_shuffle(&device, &queue, expected),
        run_simd_reduce(&device, &queue, expected),
        run_decentralized(&device, &queue, expected),
    ];

    for result in &results {
        let status = if result.correct { "✓ PASS" } else { "✗ FAIL" };
        println!("  {}: {} (sum={:.0}, expected={:.0})",
            result.name, status, result.sum, expected);
        assert!(result.correct, "{} produced incorrect results", result.name);
    }

    println!("\nAll SIMD reduction kernels produce correct results!");
}

#[test]
fn bench_simd_primitives() {
    assert!(
        !cfg!(debug_assertions),
        "Benchmark must run in release mode! Use: cargo test --release --test test_issue_104_simd_primitives"
    );

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Issue #104: SIMD Warp-Level Primitives Benchmark                      ║");
    println!("║  Data Size: {} elements | CPU Cores: {} | Trials: {}                          ║",
        DATA_SIZE, num_cpus, TIMED_RUNS);
    println!("║  Target: Close the 1.6x CPU advantage in reduction                             ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    let expected = DATA_SIZE as f32;

    println!("\n[1/5] Running CPU benchmark...");
    let cpu_result = run_cpu_benchmark(expected);
    println!("  CPU: {:.2}ms ± {:.2}ms (sum={:.0})",
        cpu_result.median_ms, cpu_result.std_dev_ms, cpu_result.sum);

    println!("\n[2/5] Running GPU Loop-based...");
    let loop_result = run_loop_based(&device, &queue, expected);
    println!("  Loop-based: {:.2}ms ± {:.2}ms (sum={:.0})",
        loop_result.median_ms, loop_result.std_dev_ms, loop_result.sum);

    println!("\n[3/5] Running GPU SIMD Shuffle...");
    let shuffle_result = run_simd_shuffle(&device, &queue, expected);
    println!("  SIMD Shuffle: {:.2}ms ± {:.2}ms (sum={:.0})",
        shuffle_result.median_ms, shuffle_result.std_dev_ms, shuffle_result.sum);

    println!("\n[4/5] Running GPU SIMD Reduce...");
    let reduce_result = run_simd_reduce(&device, &queue, expected);
    println!("  SIMD Reduce: {:.2}ms ± {:.2}ms (sum={:.0})",
        reduce_result.median_ms, reduce_result.std_dev_ms, reduce_result.sum);

    println!("\n[5/5] Running GPU Decentralized...");
    let decentralized_result = run_decentralized(&device, &queue, expected);
    println!("  Decentralized: {:.2}ms ± {:.2}ms (sum={:.0})",
        decentralized_result.median_ms, decentralized_result.std_dev_ms, decentralized_result.sum);

    // Results summary
    println!("\n╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESULTS SUMMARY                                   ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Implementation         Median (ms)   vs CPU        vs Loop      Correct      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");

    let results = vec![
        &cpu_result,
        &loop_result,
        &shuffle_result,
        &reduce_result,
        &decentralized_result,
    ];

    for result in &results {
        let vs_cpu = if result.name == "CPU" {
            "---".to_string()
        } else if cpu_result.median_ms / result.median_ms > 1.0 {
            format!("{:.2}x faster", cpu_result.median_ms / result.median_ms)
        } else {
            format!("{:.2}x slower", result.median_ms / cpu_result.median_ms)
        };

        let vs_loop = if result.name == "CPU" || result.name == "Loop-based" {
            "---".to_string()
        } else if loop_result.median_ms / result.median_ms > 1.0 {
            format!("{:.2}x faster", loop_result.median_ms / result.median_ms)
        } else {
            format!("{:.2}x slower", result.median_ms / loop_result.median_ms)
        };

        let correct_str = if result.correct { "✓" } else { "✗" };

        println!("║  {:20} {:>8.2}      {:>12}  {:>10}  {}            ║",
            result.name, result.median_ms, vs_cpu, vs_loop, correct_str);
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Key findings
    println!("\n📊 Key Findings:");

    let gpu_results = [&loop_result, &shuffle_result, &reduce_result, &decentralized_result];
    let best_gpu = gpu_results
        .iter()
        .min_by(|a, b| a.median_ms.partial_cmp(&b.median_ms).unwrap())
        .unwrap();

    let improvement_vs_loop = loop_result.median_ms / best_gpu.median_ms;
    let vs_cpu = cpu_result.median_ms / best_gpu.median_ms;

    println!("  • Best GPU implementation: {} ({:.2}ms)", best_gpu.name, best_gpu.median_ms);
    if improvement_vs_loop > 1.0 {
        println!("  • {:.1}x improvement over loop-based ✓", improvement_vs_loop);
    }
    if vs_cpu > 1.0 {
        println!("  • GPU is now {:.1}x FASTER than CPU ✓", vs_cpu);
    } else {
        println!("  • CPU is still {:.1}x faster than best GPU", 1.0 / vs_cpu);
    }

    // Assert correctness
    for result in &results {
        assert!(result.correct, "{} produced incorrect results", result.name);
    }
}
