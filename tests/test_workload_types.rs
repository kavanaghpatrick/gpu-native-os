// Comprehensive GPU vs CPU Workload Benchmark
//
// THE QUESTION: Can GPU compete with CPU for ALL workload types?
//
// Workload Categories:
// 1. Embarrassingly Parallel - independent operations (GPU should dominate)
// 2. Sequential/Dependent - each step depends on previous (CPU territory?)
// 3. Branch-Heavy - lots of divergent conditionals (warp divergence hurts GPU)
// 4. Random Memory Access - pointer chasing, irregular patterns
// 5. Reduction - sum all elements (coordination required)
// 6. Prefix Sum (Scan) - each element = sum of all previous
// 7. Search/Filter - find matching elements
// 8. String Processing - variable length, lots of branching

use metal::*;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

const DATA_SIZE: u32 = 10_000_000;  // 10M elements - need enough work to amortize GPU overhead

#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    workload: String,
    time_ms: f64,
    throughput: f64,  // ops/sec or elements/sec
}

// ============================================================================
// GPU Implementations
// ============================================================================

const GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 1. Embarrassingly Parallel - independent FMA operations
kernel void parallel_compute(
    device float* data [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    float val = data[tid];
    for (uint i = 0; i < 100; i++) {
        val = val * 1.001 + 0.001;
    }
    data[tid] = val;
}

// 2. Sequential Dependent - simulated via warp-sequential pattern
// Each thread processes a chain of dependent operations
kernel void sequential_chain(
    device float* data [[buffer(0)]],
    constant uint& chain_length [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Each thread owns a chain of elements
    uint start = tid * chain_length;
    float val = data[start];

    // Sequential dependency within the chain
    for (uint i = 1; i < chain_length; i++) {
        val = val * 1.001 + data[start + i];
        data[start + i] = val;
    }
}

// 3. Branch-Heavy - divergent conditionals
kernel void branch_heavy(
    device float* data [[buffer(0)]],
    device uint* categories [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float val = data[tid];
    uint cat = categories[tid] % 16;  // 16 different branches

    // Divergent branches - worst case for GPU warps
    if (cat == 0) { val = val * 2.0; }
    else if (cat == 1) { val = val + 1.0; }
    else if (cat == 2) { val = val - 1.0; }
    else if (cat == 3) { val = val / 2.0; }
    else if (cat == 4) { val = sqrt(abs(val)); }
    else if (cat == 5) { val = sin(val); }
    else if (cat == 6) { val = cos(val); }
    else if (cat == 7) { val = exp(val * 0.01); }
    else if (cat == 8) { val = log(abs(val) + 1.0); }
    else if (cat == 9) { val = val * val; }
    else if (cat == 10) { val = val * val * val; }
    else if (cat == 11) { val = 1.0 / (val + 1.0); }
    else if (cat == 12) { val = fma(val, 2.0, 1.0); }
    else if (cat == 13) { val = fma(val, val, val); }
    else if (cat == 14) { val = min(val, 100.0); }
    else { val = max(val, -100.0); }

    data[tid] = val;
}

// 4. Random Memory Access - indirect indexing
kernel void random_access(
    device float* data [[buffer(0)]],
    device uint* indices [[buffer(1)]],  // random indices
    device float* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Chase through random indices
    uint idx = indices[tid];
    float val = data[idx];

    // Follow the chain
    for (uint i = 0; i < 10; i++) {
        idx = indices[idx % 1000000];
        val += data[idx % 1000000];
    }

    output[tid] = val;
}

// 5. Reduction - sum all elements (parallel reduction)
kernel void reduce_sum(
    device float* data [[buffer(0)]],
    device atomic_uint* result [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Load into shared memory
    shared[local_tid] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared[local_tid] += shared[local_tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 of each threadgroup adds to global result
    if (local_tid == 0) {
        // Use atomic to accumulate (cast float to uint for atomic)
        uint val_bits = as_type<uint>(shared[0]);
        atomic_fetch_add_explicit(result, val_bits, memory_order_relaxed);
    }
}

// 6. Prefix Sum (Scan) - Hillis-Steele algorithm within threadgroups
kernel void prefix_sum(
    device float* data [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    shared[local_tid] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele parallel prefix sum
    for (uint offset = 1; offset < tg_size; offset <<= 1) {
        float val = shared[local_tid];
        if (local_tid >= offset) {
            val += shared[local_tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[local_tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[tid] = shared[local_tid];
}

// 7. Search/Filter - find elements matching condition
kernel void search_filter(
    device float* data [[buffer(0)]],
    device atomic_uint* match_count [[buffer(1)]],
    device uint* matches [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float val = data[tid];

    // Complex matching condition
    bool match = (val > threshold) &&
                 (fmod(val, 2.0) < 1.0) &&
                 (sin(val) > 0.0);

    if (match) {
        uint slot = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
        if (slot < 100000) {  // Cap output size
            matches[slot] = tid;
        }
    }
}

// 8.5. Warp-optimized Reduction using SIMD shuffle
kernel void reduce_warp_optimized(
    device float* data [[buffer(0)]],
    device atomic_float* result [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each thread loads its value
    float val = data[tid];

    // Warp-level reduction using SIMD shuffle (no shared memory needed within warp)
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);

    // First thread of each warp writes to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp reduces the warp sums
    uint num_warps = tg_size / 32;
    if (simd_group == 0 && simd_lane < num_warps) {
        val = shared[simd_lane];

        // Reduce warp sums
        if (num_warps >= 16) val += simd_shuffle_xor(val, 8);
        if (num_warps >= 8) val += simd_shuffle_xor(val, 4);
        if (num_warps >= 4) val += simd_shuffle_xor(val, 2);
        if (num_warps >= 2) val += simd_shuffle_xor(val, 1);

        // Thread 0 adds to global result
        if (simd_lane == 0) {
            atomic_fetch_add_explicit(result, val, memory_order_relaxed);
        }
    }
}

// 9. Branch-sorted (GPU-optimized) - sort by category first to reduce divergence
kernel void branch_sorted(
    device float* data [[buffer(0)]],
    device uint* sorted_indices [[buffer(1)]],  // indices sorted by category
    device uint* categories [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint idx = sorted_indices[tid];  // Access in sorted order
    float val = data[idx];
    uint cat = categories[idx];

    // Now all threads in a warp have same category = no divergence
    if (cat == 0) { val = val * 2.0; }
    else if (cat == 1) { val = val + 1.0; }
    else if (cat == 2) { val = val - 1.0; }
    else if (cat == 3) { val = val / 2.0; }
    else if (cat == 4) { val = sqrt(abs(val)); }
    else if (cat == 5) { val = sin(val); }
    else if (cat == 6) { val = cos(val); }
    else if (cat == 7) { val = exp(val * 0.01); }
    else if (cat == 8) { val = log(abs(val) + 1.0); }
    else if (cat == 9) { val = val * val; }
    else if (cat == 10) { val = val * val * val; }
    else if (cat == 11) { val = 1.0 / (val + 1.0); }
    else if (cat == 12) { val = fma(val, 2.0, 1.0); }
    else if (cat == 13) { val = fma(val, val, val); }
    else if (cat == 14) { val = min(val, 100.0); }
    else { val = max(val, -100.0); }

    data[idx] = val;
}

// 8. String-like Processing - variable length operations
kernel void string_like(
    device uchar* strings [[buffer(0)]],      // packed string data
    device uint* lengths [[buffer(1)]],        // length of each string
    device uint* offsets [[buffer(2)]],        // offset of each string
    device uint* results [[buffer(3)]],        // hash result per string
    uint tid [[thread_position_in_grid]]
) {
    uint offset = offsets[tid];
    uint length = lengths[tid];

    // Hash the string (variable-length iteration)
    uint hash = 5381;
    for (uint i = 0; i < length; i++) {
        uchar c = strings[offset + i];
        hash = ((hash << 5) + hash) + c;  // djb2 hash
    }

    results[tid] = hash;
}
"#;

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

// ============================================================================
// CPU Implementations
// ============================================================================

fn cpu_parallel_compute(data: &mut [f32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = data.len() / num_threads;

    thread::scope(|s| {
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

fn cpu_sequential_chain(data: &mut [f32], chain_length: usize) {
    // Truly sequential - no parallelism possible
    let num_chains = data.len() / chain_length;

    for chain in 0..num_chains {
        let start = chain * chain_length;
        let mut val = data[start];
        for i in 1..chain_length {
            val = val * 1.001 + data[start + i];
            data[start + i] = val;
        }
    }
}

fn cpu_sequential_chain_parallel(data: &mut [f32], chain_length: usize) {
    // Parallel across chains, sequential within
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let num_chains = data.len() / chain_length;
    let chains_per_thread = num_chains / num_threads;

    thread::scope(|s| {
        for (t, chunk) in data.chunks_mut(chains_per_thread * chain_length).enumerate() {
            s.spawn(move || {
                let local_chains = chunk.len() / chain_length;
                for chain in 0..local_chains {
                    let start = chain * chain_length;
                    let mut val = chunk[start];
                    for i in 1..chain_length {
                        val = val * 1.001 + chunk[start + i];
                        chunk[start + i] = val;
                    }
                }
            });
        }
    });
}

fn cpu_branch_heavy(data: &mut [f32], categories: &[u32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = data.len() / num_threads;

    thread::scope(|s| {
        for (chunk, cats) in data.chunks_mut(chunk_size).zip(categories.chunks(chunk_size)) {
            s.spawn(move || {
                for (val, &cat) in chunk.iter_mut().zip(cats.iter()) {
                    let cat = cat % 16;
                    *val = match cat {
                        0 => *val * 2.0,
                        1 => *val + 1.0,
                        2 => *val - 1.0,
                        3 => *val / 2.0,
                        4 => val.abs().sqrt(),
                        5 => val.sin(),
                        6 => val.cos(),
                        7 => (*val * 0.01).exp(),
                        8 => (val.abs() + 1.0).ln(),
                        9 => *val * *val,
                        10 => *val * *val * *val,
                        11 => 1.0 / (*val + 1.0),
                        12 => *val * 2.0 + 1.0,
                        13 => *val * *val + *val,
                        14 => val.min(100.0),
                        _ => val.max(-100.0),
                    };
                }
            });
        }
    });
}

fn cpu_random_access(data: &[f32], indices: &[u32], output: &mut [f32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = output.len() / num_threads;

    thread::scope(|s| {
        for (i, (out_chunk, idx_chunk)) in output.chunks_mut(chunk_size)
            .zip(indices.chunks(chunk_size))
            .enumerate()
        {
            s.spawn(move || {
                for (out, &start_idx) in out_chunk.iter_mut().zip(idx_chunk.iter()) {
                    let mut idx = start_idx as usize;
                    let mut val = data[idx % data.len()];

                    for _ in 0..10 {
                        idx = indices[idx % indices.len()] as usize;
                        val += data[idx % data.len()];
                    }

                    *out = val;
                }
            });
        }
    });
}

fn cpu_reduce_sum(data: &[f32]) -> f64 {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = data.len() / num_threads;

    let partial_sums: Vec<f64> = thread::scope(|s| {
        let handles: Vec<_> = data.chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || {
                    chunk.iter().map(|&x| x as f64).sum::<f64>()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    partial_sums.iter().sum()
}

fn cpu_prefix_sum(data: &[f32], output: &mut [f32]) {
    // Sequential prefix sum (inherently sequential)
    let mut sum = 0.0f32;
    for (i, &val) in data.iter().enumerate() {
        sum += val;
        output[i] = sum;
    }
}

fn cpu_prefix_sum_parallel(data: &[f32], output: &mut [f32]) {
    // Parallel prefix sum using three-phase algorithm
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = (data.len() + num_threads - 1) / num_threads;  // Round up

    // Phase 1: Local prefix sums - use atomics to avoid borrow issues
    use std::sync::atomic::AtomicU32;
    let actual_chunks = (data.len() + chunk_size - 1) / chunk_size;
    let local_sums: Vec<AtomicU32> = (0..actual_chunks).map(|_| AtomicU32::new(0)).collect();

    thread::scope(|s| {
        for (t, (chunk, out_chunk)) in data.chunks(chunk_size)
            .zip(output.chunks_mut(chunk_size))
            .enumerate()
        {
            let local_sum_ref = &local_sums[t];
            s.spawn(move || {
                let mut sum = 0.0f32;
                for (i, &val) in chunk.iter().enumerate() {
                    sum += val;
                    out_chunk[i] = sum;
                }
                local_sum_ref.store(sum.to_bits(), Ordering::Relaxed);
            });
        }
    });

    // Phase 2: Prefix sum of block sums (sequential, small)
    let mut block_prefix = vec![0.0f32; actual_chunks];
    let mut sum = 0.0f32;
    for i in 0..actual_chunks {
        block_prefix[i] = sum;
        sum += f32::from_bits(local_sums[i].load(Ordering::Relaxed));
    }

    // Phase 3: Add block prefix to each element
    thread::scope(|s| {
        for (t, out_chunk) in output.chunks_mut(chunk_size).enumerate() {
            let prefix = block_prefix[t];
            s.spawn(move || {
                for val in out_chunk.iter_mut() {
                    *val += prefix;
                }
            });
        }
    });
}

fn cpu_search_filter(data: &[f32], threshold: f32) -> Vec<u32> {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = data.len() / num_threads;

    let partial_results: Vec<Vec<u32>> = thread::scope(|s| {
        let handles: Vec<_> = data.chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;
                s.spawn(move || {
                    chunk.iter()
                        .enumerate()
                        .filter(|(_, &val)| {
                            val > threshold &&
                            (val % 2.0) < 1.0 &&
                            val.sin() > 0.0
                        })
                        .map(|(i, _)| (base_idx + i) as u32)
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    partial_results.into_iter().flatten().collect()
}

fn cpu_string_like(strings: &[u8], lengths: &[u32], offsets: &[u32], results: &mut [u32]) {
    let num_threads = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let chunk_size = results.len() / num_threads;

    thread::scope(|s| {
        for (chunk_idx, out_chunk) in results.chunks_mut(chunk_size).enumerate() {
            let base_idx = chunk_idx * chunk_size;
            s.spawn(move || {
                for (i, out) in out_chunk.iter_mut().enumerate() {
                    let idx = base_idx + i;
                    let offset = offsets[idx] as usize;
                    let length = lengths[idx] as usize;

                    // djb2 hash
                    let mut hash = 5381u32;
                    for j in 0..length {
                        let c = strings[offset + j];
                        hash = hash.wrapping_shl(5).wrapping_add(hash).wrapping_add(c as u32);
                    }
                    *out = hash;
                }
            });
        }
    });
}

// ============================================================================
// Benchmark Runners
// ============================================================================

fn run_gpu_parallel_compute(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "parallel_compute");

    // Initialize data
    let mut data_vec: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    let data_buf = device.new_buffer_with_data(
        data_vec.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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

    BenchResult {
        name: "GPU".to_string(),
        workload: "Parallel Compute".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_parallel_compute() -> BenchResult {
    let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();

    let start = Instant::now();
    cpu_parallel_compute(&mut data);
    let elapsed = start.elapsed();

    std::hint::black_box(&data);

    BenchResult {
        name: "CPU".to_string(),
        workload: "Parallel Compute".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_sequential(device: &Device, chain_length: u32) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "sequential_chain");

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

    let chain_buf = device.new_buffer_with_data(
        &chain_length as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let num_chains = DATA_SIZE / chain_length;

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

    BenchResult {
        name: "GPU".to_string(),
        workload: format!("Sequential (chain={})", chain_length),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_sequential(chain_length: usize, parallel: bool) -> BenchResult {
    let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();

    let start = Instant::now();
    if parallel {
        cpu_sequential_chain_parallel(&mut data, chain_length);
    } else {
        cpu_sequential_chain(&mut data, chain_length);
    }
    let elapsed = start.elapsed();

    std::hint::black_box(&data);

    BenchResult {
        name: if parallel { "CPU-MT" } else { "CPU-ST" }.to_string(),
        workload: format!("Sequential (chain={})", chain_length),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_branch_heavy(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "branch_heavy");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cat_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        let cat_ptr = cat_buf.contents() as *mut u32;
        for i in 0..DATA_SIZE {
            *data_ptr.add(i as usize) = i as f32;
            // Random-ish categories to maximize divergence
            *cat_ptr.add(i as usize) = (i * 7 + 13) % 16;
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

    BenchResult {
        name: "GPU".to_string(),
        workload: "Branch Heavy".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_branch_heavy() -> BenchResult {
    let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 16).collect();

    let start = Instant::now();
    cpu_branch_heavy(&mut data, &categories);
    let elapsed = start.elapsed();

    std::hint::black_box(&data);

    BenchResult {
        name: "CPU".to_string(),
        workload: "Branch Heavy".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_random_access(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "random_access");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let idx_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create random-ish indices
    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        let idx_ptr = idx_buf.contents() as *mut u32;
        for i in 0..DATA_SIZE {
            *data_ptr.add(i as usize) = i as f32;
            *idx_ptr.add(i as usize) = (i * 1103515245 + 12345) % DATA_SIZE;
        }
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&idx_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    BenchResult {
        name: "GPU".to_string(),
        workload: "Random Access".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_random_access() -> BenchResult {
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    let indices: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 1103515245 + 12345) % DATA_SIZE).collect();
    let mut output = vec![0.0f32; DATA_SIZE as usize];

    let start = Instant::now();
    cpu_random_access(&data, &indices, &mut output);
    let elapsed = start.elapsed();

    std::hint::black_box(&output);

    BenchResult {
        name: "CPU".to_string(),
        workload: "Random Access".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_reduction(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "reduce_sum");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let result_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);

    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        for i in 0..DATA_SIZE {
            *data_ptr.add(i as usize) = 1.0;  // Sum should be DATA_SIZE
        }
        let result_ptr = result_buf.contents() as *mut u32;
        *result_ptr = 0;
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&result_buf), 0);
    enc.set_threadgroup_memory_length(0, 256 * 4);  // shared memory for 256 floats
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    BenchResult {
        name: "GPU".to_string(),
        workload: "Reduction (sum)".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_reduction() -> BenchResult {
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];

    let start = Instant::now();
    let sum = cpu_reduce_sum(&data);
    let elapsed = start.elapsed();

    std::hint::black_box(sum);

    BenchResult {
        name: "CPU".to_string(),
        workload: "Reduction (sum)".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_reduction_warp_optimized(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "reduce_warp_optimized");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let result_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);

    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        for i in 0..DATA_SIZE {
            *data_ptr.add(i as usize) = 1.0;
        }
        let result_ptr = result_buf.contents() as *mut f32;
        *result_ptr = 0.0;
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&result_buf), 0);
    enc.set_threadgroup_memory_length(0, 256 * 4);  // 8 warps * 4 bytes
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    BenchResult {
        name: "GPU-Warp".to_string(),
        workload: "Reduction (warp shuffle)".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_branch_sorted(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "branch_sorted");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sorted_idx_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let cat_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Sort indices by category - create sorted order
    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        let sorted_ptr = sorted_idx_buf.contents() as *mut u32;
        let cat_ptr = cat_buf.contents() as *mut u32;

        // Create sorted indices (grouped by category)
        let elements_per_cat = DATA_SIZE / 16;
        for cat in 0..16u32 {
            for i in 0..elements_per_cat {
                let tid = cat * elements_per_cat + i;
                let original_idx = (tid * 7 + 13) % DATA_SIZE;  // Scatter original position
                *sorted_ptr.add(tid as usize) = original_idx;
                *cat_ptr.add(original_idx as usize) = cat;
                *data_ptr.add(original_idx as usize) = original_idx as f32;
            }
        }
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&sorted_idx_buf), 0);
    enc.set_buffer(2, Some(&cat_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    BenchResult {
        name: "GPU-Sorted".to_string(),
        workload: "Branch (sorted by category)".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_prefix_sum(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "prefix_sum");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        for i in 0..DATA_SIZE {
            *data_ptr.add(i as usize) = 1.0;
        }
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_threadgroup_memory_length(0, 256 * 4);
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    BenchResult {
        name: "GPU".to_string(),
        workload: "Prefix Sum".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_prefix_sum(parallel: bool) -> BenchResult {
    let data: Vec<f32> = vec![1.0; DATA_SIZE as usize];
    let mut output = vec![0.0f32; DATA_SIZE as usize];

    let start = Instant::now();
    if parallel {
        cpu_prefix_sum_parallel(&data, &mut output);
    } else {
        cpu_prefix_sum(&data, &mut output);
    }
    let elapsed = start.elapsed();

    std::hint::black_box(&output);

    BenchResult {
        name: if parallel { "CPU-MT" } else { "CPU-ST" }.to_string(),
        workload: "Prefix Sum".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_search_filter(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "search_filter");

    let data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
    let matches_buf = device.new_buffer(
        100000 * 4,  // max 100k matches
        MTLResourceOptions::StorageModeShared,
    );
    let threshold: f32 = 500000.0;
    let thresh_buf = device.new_buffer_with_data(
        &threshold as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        let data_ptr = data_buf.contents() as *mut f32;
        for i in 0..DATA_SIZE {
            *data_ptr.add(i as usize) = i as f32;
        }
        let count_ptr = count_buf.contents() as *mut u32;
        *count_ptr = 0;
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&count_buf), 0);
    enc.set_buffer(2, Some(&matches_buf), 0);
    enc.set_buffer(3, Some(&thresh_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    let match_count = unsafe {
        *(count_buf.contents() as *const u32)
    };

    BenchResult {
        name: "GPU".to_string(),
        workload: format!("Search/Filter ({}k matches)", match_count / 1000),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_search_filter() -> BenchResult {
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| i as f32).collect();
    let threshold = 500000.0;

    let start = Instant::now();
    let matches = cpu_search_filter(&data, threshold);
    let elapsed = start.elapsed();

    std::hint::black_box(&matches);

    BenchResult {
        name: "CPU".to_string(),
        workload: format!("Search/Filter ({}k matches)", matches.len() / 1000),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: DATA_SIZE as f64 / elapsed.as_secs_f64(),
    }
}

fn run_gpu_string_like(device: &Device) -> BenchResult {
    let (pipeline, queue) = create_gpu_pipeline(device, "string_like");

    let num_strings = 100000u32;
    let avg_length = 50u32;
    let total_bytes = num_strings * avg_length;

    let strings_buf = device.new_buffer(
        total_bytes as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lengths_buf = device.new_buffer(
        (num_strings * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let offsets_buf = device.new_buffer(
        (num_strings * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let results_buf = device.new_buffer(
        (num_strings * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        let strings_ptr = strings_buf.contents() as *mut u8;
        let lengths_ptr = lengths_buf.contents() as *mut u32;
        let offsets_ptr = offsets_buf.contents() as *mut u32;

        let mut offset = 0u32;
        for i in 0..num_strings {
            let length = 20 + (i % 60);  // Variable length 20-80
            *lengths_ptr.add(i as usize) = length;
            *offsets_ptr.add(i as usize) = offset;

            for j in 0..length {
                *strings_ptr.add((offset + j) as usize) = (65 + (j % 26)) as u8;
            }
            offset += length;
        }
    }

    let start = Instant::now();

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&strings_buf), 0);
    enc.set_buffer(1, Some(&lengths_buf), 0);
    enc.set_buffer(2, Some(&offsets_buf), 0);
    enc.set_buffer(3, Some(&results_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(num_strings as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let elapsed = start.elapsed();

    BenchResult {
        name: "GPU".to_string(),
        workload: "String Processing".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: num_strings as f64 / elapsed.as_secs_f64(),
    }
}

fn run_cpu_string_like() -> BenchResult {
    let num_strings = 100000usize;
    let avg_length = 50usize;

    let mut strings = vec![0u8; num_strings * avg_length];
    let mut lengths = vec![0u32; num_strings];
    let mut offsets = vec![0u32; num_strings];
    let mut results = vec![0u32; num_strings];

    let mut offset = 0u32;
    for i in 0..num_strings {
        let length = 20 + (i % 60) as u32;
        lengths[i] = length;
        offsets[i] = offset;

        for j in 0..length as usize {
            strings[(offset as usize) + j] = (65 + (j % 26)) as u8;
        }
        offset += length;
    }

    let start = Instant::now();
    cpu_string_like(&strings, &lengths, &offsets, &mut results);
    let elapsed = start.elapsed();

    std::hint::black_box(&results);

    BenchResult {
        name: "CPU".to_string(),
        workload: "String Processing".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput: num_strings as f64 / elapsed.as_secs_f64(),
    }
}

// ============================================================================
// Main Benchmark
// ============================================================================

#[test]
fn benchmark_all_workloads() {
    let device = Device::system_default().expect("No Metal device");
    let num_cpus = thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║           COMPREHENSIVE GPU vs CPU WORKLOAD BENCHMARK                    ║");
    println!("║           Data Size: {} elements | CPU Cores: {}                      ║", DATA_SIZE, num_cpus);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    let mut all_results: Vec<(BenchResult, BenchResult, f64)> = Vec::new();

    // 1. Parallel Compute
    println!("\n[1/8] Parallel Compute (independent FMA operations)...");
    let gpu = run_gpu_parallel_compute(&device);
    let cpu = run_cpu_parallel_compute();
    let speedup = gpu.throughput / cpu.throughput;
    println!("  GPU: {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu.time_ms, gpu.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup);
    all_results.push((gpu, cpu, speedup));

    // 2. Sequential Chains
    println!("\n[2/8] Sequential Chains (dependent operations)...");
    for chain_len in [10u32, 100, 1000] {
        let gpu = run_gpu_sequential(&device, chain_len);
        let cpu = run_cpu_sequential(chain_len as usize, true);
        let speedup = gpu.throughput / cpu.throughput;
        println!("  Chain={}: GPU: {:.2}ms  CPU-MT: {:.2}ms  Speedup: {:.1}x",
            chain_len, gpu.time_ms, cpu.time_ms, speedup);
        all_results.push((gpu, cpu, speedup));
    }

    // 3. Branch Heavy
    println!("\n[3/8] Branch Heavy (divergent conditionals)...");
    let gpu = run_gpu_branch_heavy(&device);
    let cpu = run_cpu_branch_heavy();
    let speedup = gpu.throughput / cpu.throughput;
    println!("  GPU (divergent):  {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu.time_ms, gpu.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup);
    all_results.push((gpu.clone(), cpu.clone(), speedup));

    // 3b. Branch Heavy - GPU Optimized (sorted by category)
    let gpu_sorted = run_gpu_branch_sorted(&device);
    let speedup_sorted = gpu_sorted.throughput / cpu.throughput;
    println!("  GPU (sorted):     {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu_sorted.time_ms, gpu_sorted.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup_sorted);
    all_results.push((gpu_sorted, cpu, speedup_sorted));

    // 4. Random Access
    println!("\n[4/8] Random Memory Access (pointer chasing)...");
    let gpu = run_gpu_random_access(&device);
    let cpu = run_cpu_random_access();
    let speedup = gpu.throughput / cpu.throughput;
    println!("  GPU: {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu.time_ms, gpu.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup);
    all_results.push((gpu, cpu, speedup));

    // 5. Reduction
    println!("\n[5/8] Reduction (parallel sum)...");
    let gpu = run_gpu_reduction(&device);
    let cpu = run_cpu_reduction();
    let speedup = gpu.throughput / cpu.throughput;
    println!("  GPU (basic):  {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu.time_ms, gpu.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup);
    all_results.push((gpu.clone(), cpu.clone(), speedup));

    // 5b. Reduction - GPU Warp Optimized
    let gpu_warp = run_gpu_reduction_warp_optimized(&device);
    let speedup_warp = gpu_warp.throughput / cpu.throughput;
    println!("  GPU (warp):   {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu_warp.time_ms, gpu_warp.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup_warp);
    all_results.push((gpu_warp, cpu, speedup_warp));

    // 6. Prefix Sum
    println!("\n[6/8] Prefix Sum (scan - inherently sequential)...");
    let gpu = run_gpu_prefix_sum(&device);
    let cpu_st = run_cpu_prefix_sum(false);
    let cpu_mt = run_cpu_prefix_sum(true);
    let speedup_st = gpu.throughput / cpu_st.throughput;
    let speedup_mt = gpu.throughput / cpu_mt.throughput;
    println!("  GPU: {:.2}ms  CPU-ST: {:.2}ms  CPU-MT: {:.2}ms",
        gpu.time_ms, cpu_st.time_ms, cpu_mt.time_ms);
    println!("  Speedup vs ST: {:.1}x  vs MT: {:.1}x", speedup_st, speedup_mt);
    all_results.push((gpu, cpu_mt, speedup_mt));

    // 7. Search/Filter
    println!("\n[7/8] Search/Filter (conditional output)...");
    let gpu = run_gpu_search_filter(&device);
    let cpu = run_cpu_search_filter();
    let speedup = gpu.throughput / cpu.throughput;
    println!("  GPU: {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu.time_ms, gpu.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup);
    all_results.push((gpu, cpu, speedup));

    // 8. String Processing
    println!("\n[8/8] String Processing (variable length, hashing)...");
    let gpu = run_gpu_string_like(&device);
    let cpu = run_cpu_string_like();
    let speedup = gpu.throughput / cpu.throughput;
    println!("  GPU: {:.2}ms ({:.1}M/s)  CPU: {:.2}ms ({:.1}M/s)  Speedup: {:.1}x",
        gpu.time_ms, gpu.throughput / 1_000_000.0,
        cpu.time_ms, cpu.throughput / 1_000_000.0, speedup);
    all_results.push((gpu, cpu, speedup));

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    let mut gpu_wins = 0;
    let mut cpu_wins = 0;

    for (gpu, cpu, speedup) in &all_results {
        let winner = if *speedup >= 1.0 { "GPU" } else { "CPU" };
        if *speedup >= 1.0 { gpu_wins += 1; } else { cpu_wins += 1; }

        let speedup_str = if *speedup >= 1.0 {
            format!("GPU {:.1}x", speedup)
        } else {
            format!("CPU {:.1}x", 1.0 / speedup)
        };

        println!("║  {:40} {:>12} wins  ║", gpu.workload, speedup_str);
    }

    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║  GPU Wins: {}  |  CPU Wins: {}                                            ║", gpu_wins, cpu_wins);

    let avg_speedup: f64 = all_results.iter().map(|(_, _, s)| *s).sum::<f64>() / all_results.len() as f64;
    let min_speedup = all_results.iter().map(|(_, _, s)| *s).fold(f64::INFINITY, f64::min);
    let max_speedup = all_results.iter().map(|(_, _, s)| *s).fold(0.0f64, f64::max);

    println!("║  Average Speedup: {:.1}x  Min: {:.1}x  Max: {:.1}x                          ║",
        avg_speedup, min_speedup, max_speedup);
    println!("╚══════════════════════════════════════════════════════════════════════════╝");

    // Verdict
    println!("\n═══ VERDICT ═══");
    if cpu_wins == 0 {
        println!("GPU wins ALL workloads. The GPU IS the computer.");
    } else if gpu_wins > cpu_wins {
        println!("GPU wins {}/{} workloads. CPU better for: {}",
            gpu_wins, all_results.len(),
            all_results.iter()
                .filter(|(_, _, s)| *s < 1.0)
                .map(|(g, _, _)| g.workload.as_str())
                .collect::<Vec<_>>()
                .join(", "));
    } else {
        println!("CPU wins {}/{} workloads. GPU only better for highly parallel work.",
            cpu_wins, all_results.len());
    }
}
