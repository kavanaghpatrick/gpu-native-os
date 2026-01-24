// FAIR GPU vs CPU Benchmark
//
// Addresses methodological issues identified by external review:
// 1. SAME workload on GPU and CPU (no hidden shortcuts)
// 2. WARM-UP runs before timing
// 3. MULTIPLE trials with statistics
// 4. CORRECTNESS verification (GPU results must match CPU)
// 5. CONSISTENT timing (same setup/teardown costs)
// 6. SAME memory access patterns
// 7. SAME math operations (no fast intrinsics vs libm difference)

use metal::*;
use std::time::Instant;
use std::thread;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const DATA_SIZE: u32 = 10_000_000;  // 10M elements
const WARM_UP_RUNS: usize = 3;
const TIMED_RUNS: usize = 10;  // Increased from 5 for better statistics
const TOLERANCE: f32 = 0.001;  // For float comparison

#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    workload: String,
    times_ms: Vec<f64>,
    mean_ms: f64,
    median_ms: f64,
    std_dev_ms: f64,
    min_ms: f64,
    max_ms: f64,
    throughput: f64,
    correct: bool,
}

impl BenchResult {
    fn new(name: &str, workload: &str, times: Vec<f64>, correct: bool) -> Self {
        let n = times.len() as f64;
        let mean = times.iter().sum::<f64>() / n;

        // Use sample variance (n-1 divisor) for unbiased estimate
        let variance = if times.len() > 1 {
            times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Calculate median
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = *sorted.first().unwrap_or(&0.0);
        let max = *sorted.last().unwrap_or(&0.0);
        let throughput = DATA_SIZE as f64 / (mean / 1000.0);  // elements/sec

        BenchResult {
            name: name.to_string(),
            workload: workload.to_string(),
            times_ms: times,
            mean_ms: mean,
            median_ms: median,
            std_dev_ms: std_dev,
            min_ms: min,
            max_ms: max,
            throughput,
            correct,
        }
    }
}

// ============================================================================
// GPU Shaders - Deliberately simple to match CPU exactly
// ============================================================================

const GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 1. Parallel Compute - 100 FMA operations per element
// Uses same math as CPU: multiply and add (no special intrinsics)
kernel void parallel_compute(
    device float* data [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    float val = data[tid];
    for (uint i = 0; i < 100; i++) {
        val = val * 1.001f + 0.001f;  // Simple FMA, no intrinsics
    }
    data[tid] = val;
}

// 2. Reduction - Proper hierarchical reduction
// Two-phase: local reduction in shared memory, then atomic add
kernel void reduce_sum(
    device float* data [[buffer(0)]],
    device atomic_float* result [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint num_threadgroups [[threadgroups_per_grid]]
) {
    // Load into shared memory
    shared[local_tid] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared[local_tid] += shared[local_tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 of each threadgroup adds to global result
    if (local_tid == 0) {
        atomic_fetch_add_explicit(result, shared[0], memory_order_relaxed);
    }
}

// 3. Random Memory Access - FULL dataset, not truncated
kernel void random_access(
    device float* data [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& data_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Chase through random indices - use data_size not hardcoded value
    uint idx = indices[tid];
    float val = data[idx % data_size];

    // 10 chases through the FULL dataset
    for (uint i = 0; i < 10; i++) {
        idx = indices[idx % data_size];
        val += data[idx % data_size];
    }

    output[tid] = val;
}

// 4. Branch Heavy - Simple branches, same operations as CPU
kernel void branch_compute(
    device float* data [[buffer(0)]],
    device uint* categories [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float val = data[tid];
    uint cat = categories[tid] % 4;  // Only 4 branches to reduce divergence impact

    // Same math operations as CPU - no special intrinsics
    if (cat == 0) {
        val = val * 2.0f + 1.0f;
    } else if (cat == 1) {
        val = val * 0.5f - 1.0f;
    } else if (cat == 2) {
        val = val * val;
    } else {
        val = 1.0f / (val + 1.0f);
    }

    data[tid] = val;
}

// 5. Sequential Chain - Each thread processes a chain of dependent ops
kernel void sequential_chain(
    device float* data [[buffer(0)]],
    constant uint& chain_length [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint start = tid * chain_length;
    float val = data[start];

    // Sequential dependency within chain
    for (uint i = 1; i < chain_length; i++) {
        val = val * 1.001f + data[start + i];
        data[start + i] = val;
    }
}
"#;

// ============================================================================
// CPU Reference Implementations - THE GROUND TRUTH
// ============================================================================

fn cpu_parallel_compute(data: &mut [f32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    thread::scope(|s| {
        let chunk_size = (data.len() + num_threads - 1) / num_threads;
        for chunk in data.chunks_mut(chunk_size) {
            s.spawn(move || {
                for val in chunk.iter_mut() {
                    let mut v = *val;
                    for _ in 0..100 {
                        v = v * 1.001 + 0.001;
                    }
                    *val = v;
                }
            });
        }
    });
}

// CPU reduction using f32 to match GPU precision exactly
fn cpu_reduce_sum(data: &[f32]) -> f32 {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    let partial_sums: Vec<f32> = thread::scope(|s| {
        let chunk_size = (data.len() + num_threads - 1) / num_threads;
        data.chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || chunk.iter().sum::<f32>())  // f32 like GPU
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    partial_sums.iter().sum()
}

fn cpu_random_access(data: &[f32], indices: &[u32], output: &mut [f32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let data_size = data.len();

    thread::scope(|s| {
        let chunk_size = (output.len() + num_threads - 1) / num_threads;
        for (chunk_idx, out_chunk) in output.chunks_mut(chunk_size).enumerate() {
            let base = chunk_idx * chunk_size;
            s.spawn(move || {
                for (i, out) in out_chunk.iter_mut().enumerate() {
                    let tid = base + i;
                    let mut idx = indices[tid] as usize;
                    let mut val = data[idx % data_size];

                    for _ in 0..10 {
                        idx = indices[idx % data_size] as usize;
                        val += data[idx % data_size];
                    }

                    *out = val;
                }
            });
        }
    });
}

fn cpu_branch_compute(data: &mut [f32], categories: &[u32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    thread::scope(|s| {
        let chunk_size = (data.len() + num_threads - 1) / num_threads;
        for (chunk, cats) in data.chunks_mut(chunk_size).zip(categories.chunks(chunk_size)) {
            s.spawn(move || {
                for (val, &cat) in chunk.iter_mut().zip(cats.iter()) {
                    let cat = cat % 4;
                    *val = match cat {
                        0 => *val * 2.0 + 1.0,
                        1 => *val * 0.5 - 1.0,
                        2 => *val * *val,
                        _ => 1.0 / (*val + 1.0),
                    };
                }
            });
        }
    });
}

fn cpu_sequential_chain(data: &mut [f32], chain_length: usize) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let num_chains = data.len() / chain_length;
    let chains_per_thread = (num_chains + num_threads - 1) / num_threads;

    thread::scope(|s| {
        // Use chunks_mut to properly split the data
        let elements_per_thread = chains_per_thread * chain_length;
        for chunk in data.chunks_mut(elements_per_thread) {
            let local_chains = chunk.len() / chain_length;
            s.spawn(move || {
                for chain in 0..local_chains {
                    let chain_start = chain * chain_length;
                    let mut val = chunk[chain_start];
                    for i in 1..chain_length {
                        val = val * 1.001 + chunk[chain_start + i];
                        chunk[chain_start + i] = val;
                    }
                }
            });
        }
    });
}

// ============================================================================
// GPU Benchmark Runners
// ============================================================================

fn create_gpu_pipeline(device: &Device, function_name: &str) -> (ComputePipelineState, CommandQueue) {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(GPU_SHADERS, &options)
        .expect("Shader compile failed");
    let function = library.get_function(function_name, None).expect("Function not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed");
    let queue = device.new_command_queue();
    (pipeline, queue)
}

// Check correctness by comparing ENTIRE buffer (not sampling)
// This ensures no GPU bugs slip through undetected
fn verify_correctness_full(gpu_ptr: *const f32, reference: &[f32]) -> bool {
    let data_size = reference.len();
    let mut errors = 0;
    let mut first_error_idx = None;

    // Compare EVERY element for complete correctness verification
    for idx in 0..data_size {
        let gpu_val = unsafe { *gpu_ptr.add(idx) };
        let cpu_val = reference[idx];
        let diff = (gpu_val - cpu_val).abs();
        let tolerance = TOLERANCE * cpu_val.abs().max(1.0);

        if diff > tolerance {
            if errors < 5 {
                println!("  Mismatch at {}: GPU={:.6} CPU={:.6} diff={:.6}",
                    idx, gpu_val, cpu_val, diff);
            }
            if first_error_idx.is_none() {
                first_error_idx = Some(idx);
            }
            errors += 1;
        }
    }

    if errors > 0 {
        println!("  Total errors: {} / {} elements ({:.4}%)",
            errors, data_size, (errors as f64 / data_size as f64) * 100.0);
    }
    errors == 0
}

fn run_gpu_parallel_compute(device: &Device, reference: &[f32]) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "parallel_compute");

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        // Fresh data each run
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = i as f32;
            }
        }

        // Time includes command buffer creation (fair comparison)
        let start = Instant::now();

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
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
        }

        // Verify correctness on EVERY timed run with distributed sampling
        if run >= WARM_UP_RUNS {
            let ptr = data_buf.contents() as *const f32;
            if !verify_correctness_full(ptr, reference) {
                println!("Parallel compute failed on run {}", run);
                all_correct = false;
            }
        }
    }

    BenchResult::new("GPU", "Parallel Compute", times, all_correct)
}

fn run_gpu_reduction(device: &Device, expected_sum: f32) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "reduce_sum");

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let result_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = 1.0;  // Sum should equal DATA_SIZE
            }
            let res_ptr = result_buf.contents() as *mut f32;
            *res_ptr = 0.0;
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
        }

        // Verify correctness on EVERY timed run
        if run >= WARM_UP_RUNS {
            let gpu_sum = unsafe { *(result_buf.contents() as *const f32) };
            let rel_error = (gpu_sum - expected_sum).abs() / expected_sum.abs().max(1.0);
            if rel_error > 0.001 {  // 0.1% tolerance for f32 accumulation
                println!("Reduction mismatch run {}: GPU={} Expected={} Error={}%",
                    run, gpu_sum, expected_sum, rel_error * 100.0);
                all_correct = false;
            }
        }
    }

    BenchResult::new("GPU", "Reduction", times, all_correct)
}

fn run_gpu_random_access(device: &Device, reference: &[f32]) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "random_access");

    // Create consistent random indices
    let indices: Vec<u32> = (0..DATA_SIZE)
        .map(|i| ((i as u64 * 1103515245 + 12345) % DATA_SIZE as u64) as u32)
        .collect();

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let idx_buf = device.new_buffer_with_data(
            indices.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let size_buf = device.new_buffer_with_data(
            &DATA_SIZE as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = i as f32;
            }
        }

        let start = Instant::now();

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&idx_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_buffer(3, Some(&size_buf), 0);
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
        }

        // Verify on every timed run
        if run >= WARM_UP_RUNS {
            let ptr = out_buf.contents() as *const f32;
            if !verify_correctness_full(ptr, reference) {
                println!("Random access failed on run {}", run);
                all_correct = false;
            }
        }
    }

    BenchResult::new("GPU", "Random Access", times, all_correct)
}

fn run_gpu_branch(device: &Device, reference: &[f32], categories: &[u32]) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "branch_compute");

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cat_buf = device.new_buffer_with_data(
            categories.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = (i as f32) * 0.001;  // Small values to avoid overflow
            }
        }

        let start = Instant::now();

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&cat_buf), 0);
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
        }

        // Verify on every timed run with distributed sampling
        if run >= WARM_UP_RUNS {
            let ptr = data_buf.contents() as *const f32;
            if !verify_correctness_full(ptr, reference) {
                println!("Branch compute failed on run {}", run);
                all_correct = false;
            }
        }
    }

    BenchResult::new("GPU", "Branch Compute", times, all_correct)
}

fn run_gpu_sequential(device: &Device, chain_length: u32, reference: &[f32]) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "sequential_chain");

    let num_chains = DATA_SIZE / chain_length;
    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let chain_buf = device.new_buffer_with_data(
            &chain_length as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = i as f32 * 0.0001;  // Small values
            }
        }

        let start = Instant::now();

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&chain_buf), 0);
        enc.dispatch_threads(
            MTLSize::new(num_chains as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }

        // Verify on every timed run with distributed sampling
        if run >= WARM_UP_RUNS {
            let ptr = data_buf.contents() as *const f32;
            if !verify_correctness_full(ptr, reference) {
                println!("Sequential chain={} failed on run {}", chain_length, run);
                all_correct = false;
            }
        }
    }

    BenchResult::new("GPU", &format!("Sequential (chain={})", chain_length), times, all_correct)
}

// ============================================================================
// Reference Data Generation (untimed, for correctness verification)
// ============================================================================

fn generate_parallel_compute_reference() -> Vec<f32> {
    let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    cpu_parallel_compute(&mut data);
    data
}

fn generate_reduction_reference() -> f32 {
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    cpu_reduce_sum(&data)
}

fn generate_random_access_reference() -> (Vec<f32>, Vec<u32>) {
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    let indices: Vec<u32> = (0..DATA_SIZE)
        .map(|i| ((i as u64 * 1103515245 + 12345) % DATA_SIZE as u64) as u32)
        .collect();
    let mut output = vec![0.0f32; DATA_SIZE as usize];
    cpu_random_access(&data, &indices, &mut output);
    (output, indices)
}

fn generate_branch_reference() -> (Vec<f32>, Vec<u32>) {
    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_branch_compute(&mut data, &categories);
    (data, categories)
}

fn generate_sequential_reference(chain_length: usize) -> Vec<f32> {
    let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32 * 0.0001).collect();
    cpu_sequential_chain(&mut data, chain_length);
    data
}

// ============================================================================
// CPU Benchmark Runners (timing only, no reference generation)
// ============================================================================

fn run_cpu_parallel_compute_timed() -> BenchResult {
    let mut times = Vec::new();

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();

        let start = Instant::now();
        cpu_parallel_compute(&mut data);
        let elapsed = start.elapsed();

        std::hint::black_box(&data);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    BenchResult::new("CPU", "Parallel Compute", times, true)
}

fn run_cpu_reduction_timed() -> BenchResult {
    let mut times = Vec::new();
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let start = Instant::now();
        let sum = cpu_reduce_sum(&data);
        let elapsed = start.elapsed();

        std::hint::black_box(sum);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    BenchResult::new("CPU", "Reduction", times, true)
}

fn run_cpu_random_access_timed(indices: &[u32]) -> BenchResult {
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    let mut times = Vec::new();

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let mut output = vec![0.0f32; DATA_SIZE as usize];

        let start = Instant::now();
        cpu_random_access(&data, indices, &mut output);
        let elapsed = start.elapsed();

        std::hint::black_box(&output);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    BenchResult::new("CPU", "Random Access", times, true)
}

fn run_cpu_branch_timed(categories: &[u32]) -> BenchResult {
    let mut times = Vec::new();

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();

        let start = Instant::now();
        cpu_branch_compute(&mut data, categories);
        let elapsed = start.elapsed();

        std::hint::black_box(&data);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    BenchResult::new("CPU", "Branch Compute", times, true)
}

fn run_cpu_sequential_timed(chain_length: usize) -> BenchResult {
    let mut times = Vec::new();

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32 * 0.0001).collect();

        let start = Instant::now();
        cpu_sequential_chain(&mut data, chain_length);
        let elapsed = start.elapsed();

        std::hint::black_box(&data);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    BenchResult::new("CPU", &format!("Sequential (chain={})", chain_length), times, true)
}

// Legacy functions for backwards compatibility
fn run_cpu_parallel_compute() -> (BenchResult, Vec<f32>) {
    let reference = generate_parallel_compute_reference();
    let result = run_cpu_parallel_compute_timed();
    (result, reference)
}

fn run_cpu_reduction() -> (BenchResult, f32) {
    let reference = generate_reduction_reference();
    let result = run_cpu_reduction_timed();
    (result, reference)
}

fn run_cpu_random_access() -> (BenchResult, Vec<f32>) {
    let (reference, _) = generate_random_access_reference();
    let indices: Vec<u32> = (0..DATA_SIZE)
        .map(|i| ((i as u64 * 1103515245 + 12345) % DATA_SIZE as u64) as u32)
        .collect();
    let result = run_cpu_random_access_timed(&indices);
    (result, reference)
}

fn run_cpu_branch() -> (BenchResult, Vec<f32>, Vec<u32>) {
    let (reference, categories) = generate_branch_reference();
    let result = run_cpu_branch_timed(&categories);
    (result, reference, categories)
}

fn run_cpu_sequential(chain_length: usize) -> (BenchResult, Vec<f32>) {
    let reference = generate_sequential_reference(chain_length);
    let result = run_cpu_sequential_timed(chain_length);
    (result, reference)
}

// Helper for TRUE random ordering - uses system time entropy
// Each run of the benchmark will have different CPU/GPU ordering
fn should_run_gpu_first(workload_name: &str, run_seed: u64) -> bool {
    let mut hasher = DefaultHasher::new();
    workload_name.hash(&mut hasher);
    run_seed.hash(&mut hasher);  // Include runtime seed for true randomness
    hasher.finish() % 2 == 0
}

// ============================================================================
// Main Benchmark
// ============================================================================

#[test]
fn fair_benchmark() {
    // CRITICAL: Ensure release mode - debug builds handicap CPU with bounds checks
    // while GPU Metal shaders are always optimized
    assert!(
        !cfg!(debug_assertions),
        "Benchmark must run in release mode! Use: cargo test --release --test test_fair_benchmark"
    );

    let device = Device::system_default().expect("No Metal device");
    let num_cpus = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    FAIR GPU vs CPU BENCHMARK                                 ║");
    println!("║  Data Size: {} elements | CPU Cores: {} | Warm-up: {} | Trials: {}          ║",
        DATA_SIZE, num_cpus, WARM_UP_RUNS, TIMED_RUNS);
    println!("║  Methodology: Same workload, same data, verified correctness, random order    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let mut all_results: Vec<(BenchResult, BenchResult)> = Vec::new();

    // Phase 1: Generate all reference data BEFORE any timing (thermal fairness)
    println!("\n[Phase 1] Generating reference data (untimed)...");
    let parallel_ref = generate_parallel_compute_reference();
    let reduction_ref = generate_reduction_reference();
    let (random_access_ref, random_indices) = generate_random_access_reference();
    let (branch_ref, branch_categories) = generate_branch_reference();
    let seq100_ref = generate_sequential_reference(100);
    let seq1000_ref = generate_sequential_reference(1000);
    println!("  Reference data ready. Cooling down...");
    std::thread::sleep(std::time::Duration::from_millis(500)); // Brief cooling period

    // Phase 2: Run benchmarks with randomized CPU/GPU order
    // Use system time as seed for TRUE randomness across benchmark runs
    let run_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    println!("\n[Phase 2] Running benchmarks (randomized CPU/GPU order, seed={})...", run_seed);

    // 1. Parallel Compute
    let workload = "Parallel Compute";
    println!("\n[1/6] {} (100 FMA/element)...", workload);
    let (cpu_result, gpu_result) = if should_run_gpu_first(workload, run_seed) {
        println!("  Order: GPU first, then CPU");
        let gpu = run_gpu_parallel_compute(&device, &parallel_ref);
        let cpu = run_cpu_parallel_compute_timed();
        (cpu, gpu)
    } else {
        println!("  Order: CPU first, then GPU");
        let cpu = run_cpu_parallel_compute_timed();
        let gpu = run_gpu_parallel_compute(&device, &parallel_ref);
        (cpu, gpu)
    };
    println!("  CPU: {:.2}ms ± {:.2}ms  GPU: {:.2}ms ± {:.2}ms  Correct: {}",
        cpu_result.mean_ms, cpu_result.std_dev_ms,
        gpu_result.mean_ms, gpu_result.std_dev_ms,
        if gpu_result.correct { "✓" } else { "✗" });
    all_results.push((cpu_result, gpu_result));

    // 2. Reduction
    let workload = "Reduction";
    println!("\n[2/6] {} (sum all elements)...", workload);
    let (cpu_result, gpu_result) = if should_run_gpu_first(workload, run_seed) {
        println!("  Order: GPU first, then CPU");
        let gpu = run_gpu_reduction(&device, reduction_ref);
        let cpu = run_cpu_reduction_timed();
        (cpu, gpu)
    } else {
        println!("  Order: CPU first, then GPU");
        let cpu = run_cpu_reduction_timed();
        let gpu = run_gpu_reduction(&device, reduction_ref);
        (cpu, gpu)
    };
    println!("  CPU: {:.2}ms ± {:.2}ms  GPU: {:.2}ms ± {:.2}ms  Correct: {}",
        cpu_result.mean_ms, cpu_result.std_dev_ms,
        gpu_result.mean_ms, gpu_result.std_dev_ms,
        if gpu_result.correct { "✓" } else { "✗" });
    all_results.push((cpu_result, gpu_result));

    // 3. Random Access
    let workload = "Random Access";
    println!("\n[3/6] {} (10 chases/element)...", workload);
    let (cpu_result, gpu_result) = if should_run_gpu_first(workload, run_seed) {
        println!("  Order: GPU first, then CPU");
        let gpu = run_gpu_random_access(&device, &random_access_ref);
        let cpu = run_cpu_random_access_timed(&random_indices);
        (cpu, gpu)
    } else {
        println!("  Order: CPU first, then GPU");
        let cpu = run_cpu_random_access_timed(&random_indices);
        let gpu = run_gpu_random_access(&device, &random_access_ref);
        (cpu, gpu)
    };
    println!("  CPU: {:.2}ms ± {:.2}ms  GPU: {:.2}ms ± {:.2}ms  Correct: {}",
        cpu_result.mean_ms, cpu_result.std_dev_ms,
        gpu_result.mean_ms, gpu_result.std_dev_ms,
        if gpu_result.correct { "✓" } else { "✗" });
    all_results.push((cpu_result, gpu_result));

    // 4. Branch Compute
    let workload = "Branch Compute";
    println!("\n[4/6] {} (4 branches)...", workload);
    let (cpu_result, gpu_result) = if should_run_gpu_first(workload, run_seed) {
        println!("  Order: GPU first, then CPU");
        let gpu = run_gpu_branch(&device, &branch_ref, &branch_categories);
        let cpu = run_cpu_branch_timed(&branch_categories);
        (cpu, gpu)
    } else {
        println!("  Order: CPU first, then GPU");
        let cpu = run_cpu_branch_timed(&branch_categories);
        let gpu = run_gpu_branch(&device, &branch_ref, &branch_categories);
        (cpu, gpu)
    };
    println!("  CPU: {:.2}ms ± {:.2}ms  GPU: {:.2}ms ± {:.2}ms  Correct: {}",
        cpu_result.mean_ms, cpu_result.std_dev_ms,
        gpu_result.mean_ms, gpu_result.std_dev_ms,
        if gpu_result.correct { "✓" } else { "✗" });
    all_results.push((cpu_result, gpu_result));

    // 5. Sequential Chain (100)
    let workload = "Sequential100";
    println!("\n[5/6] Sequential Chain (chain=100)...");
    let (cpu_result, gpu_result) = if should_run_gpu_first(workload, run_seed) {
        println!("  Order: GPU first, then CPU");
        let gpu = run_gpu_sequential(&device, 100, &seq100_ref);
        let cpu = run_cpu_sequential_timed(100);
        (cpu, gpu)
    } else {
        println!("  Order: CPU first, then GPU");
        let cpu = run_cpu_sequential_timed(100);
        let gpu = run_gpu_sequential(&device, 100, &seq100_ref);
        (cpu, gpu)
    };
    println!("  CPU: {:.2}ms ± {:.2}ms  GPU: {:.2}ms ± {:.2}ms  Correct: {}",
        cpu_result.mean_ms, cpu_result.std_dev_ms,
        gpu_result.mean_ms, gpu_result.std_dev_ms,
        if gpu_result.correct { "✓" } else { "✗" });
    all_results.push((cpu_result, gpu_result));

    // 6. Sequential Chain (1000)
    let workload = "Sequential1000";
    println!("\n[6/6] Sequential Chain (chain=1000)...");
    let (cpu_result, gpu_result) = if should_run_gpu_first(workload, run_seed) {
        println!("  Order: GPU first, then CPU");
        let gpu = run_gpu_sequential(&device, 1000, &seq1000_ref);
        let cpu = run_cpu_sequential_timed(1000);
        (cpu, gpu)
    } else {
        println!("  Order: CPU first, then GPU");
        let cpu = run_cpu_sequential_timed(1000);
        let gpu = run_gpu_sequential(&device, 1000, &seq1000_ref);
        (cpu, gpu)
    };
    println!("  CPU: {:.2}ms ± {:.2}ms  GPU: {:.2}ms ± {:.2}ms  Correct: {}",
        cpu_result.mean_ms, cpu_result.std_dev_ms,
        gpu_result.mean_ms, gpu_result.std_dev_ms,
        if gpu_result.correct { "✓" } else { "✗" });
    all_results.push((cpu_result, gpu_result));

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                    RESULTS (n={} trials)                                   ║", TIMED_RUNS);
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Workload              CPU median    GPU median    Speedup     Correct                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════╣");

    let mut gpu_wins = 0;
    let mut cpu_wins = 0;
    let mut all_correct = true;

    for (cpu, gpu) in &all_results {
        // Use median for speedup calculation (more robust to outliers)
        let speedup = cpu.median_ms / gpu.median_ms;
        if speedup > 1.0 { gpu_wins += 1; } else { cpu_wins += 1; }
        if !gpu.correct { all_correct = false; }

        let correct_mark = if gpu.correct { "✓" } else { "✗" };
        let speedup_str = if speedup > 1.0 {
            format!("GPU {:.1}x", speedup)
        } else {
            format!("CPU {:.1}x", 1.0 / speedup)
        };

        println!("║  {:20} {:>7.2}ms      {:>7.2}ms      {:>10}  {}                       ║",
            gpu.workload,
            cpu.median_ms,
            gpu.median_ms,
            speedup_str, correct_mark);
    }

    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  GPU Wins: {}  |  CPU Wins: {}  |  All Correct: {}                                         ║",
        gpu_wins, cpu_wins, if all_correct { "✓" } else { "✗" });
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════╝");

    // Detailed statistics
    println!("\nDetailed Statistics:");
    println!("  {:20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Workload", "Mean", "Median", "StdDev", "Min", "Max");
    println!("  {}", "-".repeat(70));
    for (cpu, gpu) in &all_results {
        println!("  {} (CPU)", gpu.workload);
        println!("    {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            cpu.mean_ms, cpu.median_ms, cpu.std_dev_ms, cpu.min_ms, cpu.max_ms);
        println!("  {} (GPU)", gpu.workload);
        println!("    {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            gpu.mean_ms, gpu.median_ms, gpu.std_dev_ms, gpu.min_ms, gpu.max_ms);
    }

    // Assertions for CI
    assert!(all_correct, "Some GPU results did not match CPU reference!");
}
