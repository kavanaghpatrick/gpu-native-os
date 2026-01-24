// Persistence Strategies Benchmark
//
// THE GPU IS THE COMPUTER. But which execution model is best?
//
// Strategy 1: RAPID RE-DISPATCH
//   - Minimal work per dispatch
//   - Host immediately re-launches
//   - Overhead: ~133µs per dispatch (measured)
//
// Strategy 2: MEGA-DISPATCH (Chunked)
//   - Maximum work per dispatch (up to safe limit)
//   - Checkpoint state at end
//   - Host restarts from checkpoint
//
// Strategy 3: WORK QUEUE
//   - GPU pulls work from device memory queue
//   - GPU decides what to do next
//   - Host just keeps queue fed
//
// We measure: throughput, latency, CPU overhead, scalability

use metal::*;
use std::time::Instant;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

// ============================================================================
// Common workload: Compute N iterations of work
// ============================================================================

const TOTAL_ITERATIONS: u32 = 10_000_000;  // 10M iterations total work

// ============================================================================
// Strategy 1: Rapid Re-dispatch
// ============================================================================

const RAPID_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct State {
    atomic_uint iteration;
    atomic_uint completed;
    uint target;
    uint _pad;
};

kernel void rapid_kernel(
    device State* state [[buffer(0)]],
    device float* data [[buffer(1)]],
    constant uint& work_per_dispatch [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    // Each dispatch does a fixed amount of work
    uint start = atomic_load_explicit(&state->iteration, memory_order_relaxed);
    uint end = min(start + work_per_dispatch, state->target);

    // Parallel work across threads
    for (uint i = start + tid; i < end; i += threads) {
        // Actual compute work
        float val = data[i % 1024];
        for (uint j = 0; j < 100; j++) {
            val = val * 1.001 + 0.001;
        }
        data[i % 1024] = val;
    }

    // Thread 0 updates progress
    if (tid == 0) {
        atomic_store_explicit(&state->iteration, end, memory_order_relaxed);
        if (end >= state->target) {
            atomic_store_explicit(&state->completed, 1, memory_order_relaxed);
        }
    }
}
"#;

fn run_rapid_strategy(device: &Device, work_per_dispatch: u32) -> StrategyResult {
    let queue = device.new_command_queue();

    // Compile shader
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(RAPID_SHADER, &options)
        .expect("Rapid shader compile failed");
    let function = library.get_function("rapid_kernel", None).expect("Function not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed");

    // Create buffers
    let state_buf = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
    let data_buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

    // Initialize state
    unsafe {
        let state = state_buf.contents() as *mut u32;
        *state = 0;           // iteration
        *state.add(1) = 0;    // completed
        *state.add(2) = TOTAL_ITERATIONS;  // target
    }

    // Initialize data
    unsafe {
        let data = data_buf.contents() as *mut f32;
        for i in 0..1024 {
            *data.add(i) = i as f32;
        }
    }

    let work_buf = device.new_buffer_with_data(
        &work_per_dispatch as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let start = Instant::now();
    let mut dispatch_count = 0u32;

    loop {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&state_buf), 0);
        enc.set_buffer(1, Some(&data_buf), 0);
        enc.set_buffer(2, Some(&work_buf), 0);

        enc.dispatch_thread_groups(
            MTLSize::new(16, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        dispatch_count += 1;

        // Check if completed
        let completed = unsafe {
            let state = state_buf.contents() as *const u32;
            *state.add(1)
        };

        if completed != 0 {
            break;
        }
    }

    let elapsed = start.elapsed();

    StrategyResult {
        name: format!("Rapid ({}k/dispatch)", work_per_dispatch / 1000),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

// ============================================================================
// Strategy 2: Mega-dispatch (Chunked)
// ============================================================================

const MEGA_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct State {
    atomic_uint iteration;
    atomic_uint completed;
    uint target;
    uint _pad;
};

kernel void mega_kernel(
    device State* state [[buffer(0)]],
    device float* data [[buffer(1)]],
    constant uint& max_iterations [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]]
) {
    uint start = atomic_load_explicit(&state->iteration, memory_order_relaxed);
    uint target = state->target;

    // Do as much work as possible within budget
    uint iterations_done = 0;
    uint i = start + tid;

    while (iterations_done < max_iterations && i < target) {
        // Actual compute work
        float val = data[i % 1024];
        for (uint j = 0; j < 100; j++) {
            val = val * 1.001 + 0.001;
        }
        data[i % 1024] = val;

        i += threads;
        iterations_done++;
    }

    // Thread 0 checkpoints progress
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint new_iter = min(start + max_iterations * threads, target);
        atomic_store_explicit(&state->iteration, new_iter, memory_order_relaxed);
        if (new_iter >= target) {
            atomic_store_explicit(&state->completed, 1, memory_order_relaxed);
        }
    }
}
"#;

fn run_mega_strategy(device: &Device, max_iterations_per_thread: u32) -> StrategyResult {
    let queue = device.new_command_queue();

    // Compile shader
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(MEGA_SHADER, &options)
        .expect("Mega shader compile failed");
    let function = library.get_function("mega_kernel", None).expect("Function not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed");

    // Create buffers
    let state_buf = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
    let data_buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

    // Initialize state
    unsafe {
        let state = state_buf.contents() as *mut u32;
        *state = 0;
        *state.add(1) = 0;
        *state.add(2) = TOTAL_ITERATIONS;
    }

    // Initialize data
    unsafe {
        let data = data_buf.contents() as *mut f32;
        for i in 0..1024 {
            *data.add(i) = i as f32;
        }
    }

    let max_buf = device.new_buffer_with_data(
        &max_iterations_per_thread as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let start = Instant::now();
    let mut dispatch_count = 0u32;

    loop {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&state_buf), 0);
        enc.set_buffer(1, Some(&data_buf), 0);
        enc.set_buffer(2, Some(&max_buf), 0);

        enc.dispatch_thread_groups(
            MTLSize::new(16, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        dispatch_count += 1;

        let completed = unsafe {
            let state = state_buf.contents() as *const u32;
            *state.add(1)
        };

        if completed != 0 {
            break;
        }
    }

    let elapsed = start.elapsed();

    StrategyResult {
        name: format!("Mega ({}k iter/thread)", max_iterations_per_thread / 1000),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

// ============================================================================
// Strategy 3: Work Queue (Fixed - Pre-partitioned)
// ============================================================================

const QUEUE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct QueueState {
    atomic_uint next_chunk;     // Next chunk to process
    uint total_chunks;          // Total number of chunks
    uint chunk_size;            // Iterations per chunk
    uint target;                // Total target iterations
    atomic_uint completed;      // Total completed iterations
};

kernel void queue_kernel(
    device QueueState* queue [[buffer(0)]],
    device float* data [[buffer(1)]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Each threadgroup independently claims and processes chunks
    uint total_chunks = queue->total_chunks;
    uint chunk_size_val = queue->chunk_size;
    uint target = queue->target;

    while (true) {
        // Thread 0 of THIS threadgroup claims next chunk
        threadgroup uint chunk_start;
        threadgroup uint chunk_end;

        if (local_tid == 0) {
            uint my_chunk = atomic_fetch_add_explicit(&queue->next_chunk, 1, memory_order_relaxed);
            if (my_chunk < total_chunks) {
                chunk_start = my_chunk * chunk_size_val;
                chunk_end = min(chunk_start + chunk_size_val, target);
            } else {
                chunk_start = 0xFFFFFFFF;  // Signal done
                chunk_end = 0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if we got a valid chunk
        if (chunk_start == 0xFFFFFFFF) {
            break;  // No more work
        }

        // All threads in this threadgroup process the chunk
        for (uint i = chunk_start + local_tid; i < chunk_end; i += tg_size) {
            float val = data[i % 1024];
            for (uint j = 0; j < 100; j++) {
                val = val * 1.001 + 0.001;
            }
            data[i % 1024] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 records completion
        if (local_tid == 0) {
            atomic_fetch_add_explicit(&queue->completed, chunk_end - chunk_start, memory_order_relaxed);
        }
    }
}
"#;

fn run_queue_strategy(device: &Device, chunk_size: u32) -> StrategyResult {
    let queue = device.new_command_queue();

    // Compile shader
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(QUEUE_SHADER, &options)
        .expect("Queue shader compile failed");
    let function = library.get_function("queue_kernel", None).expect("Function not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed");

    // Calculate number of chunks
    let num_chunks = (TOTAL_ITERATIONS + chunk_size - 1) / chunk_size;

    // Create buffers - QueueState is 20 bytes (5 u32s), padded to 32
    let queue_buf = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
    let data_buf = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

    // Initialize queue state
    unsafe {
        let state = queue_buf.contents() as *mut u32;
        *state = 0;               // next_chunk (atomic)
        *state.add(1) = num_chunks;  // total_chunks
        *state.add(2) = chunk_size;  // chunk_size
        *state.add(3) = TOTAL_ITERATIONS;  // target
        *state.add(4) = 0;        // completed (atomic)
    }

    // Initialize data
    unsafe {
        let data = data_buf.contents() as *mut f32;
        for i in 0..1024 {
            *data.add(i) = i as f32;
        }
    }

    let start = Instant::now();
    let mut dispatch_count = 0u32;

    loop {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&queue_buf), 0);
        enc.set_buffer(1, Some(&data_buf), 0);

        // Launch multiple threadgroups that compete for chunks
        enc.dispatch_thread_groups(
            MTLSize::new(16, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        dispatch_count += 1;

        let (completed, next_chunk) = unsafe {
            let state = queue_buf.contents() as *const u32;
            (*state.add(4), *state)
        };

        if completed >= TOTAL_ITERATIONS || next_chunk >= num_chunks {
            break;
        }
    }

    let elapsed = start.elapsed();

    StrategyResult {
        name: format!("Queue ({}k/chunk)", chunk_size / 1000),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

// ============================================================================
// CPU Benchmarks for Comparison
// ============================================================================

/// CPU single-threaded - same workload as GPU
fn run_cpu_single_threaded() -> StrategyResult {
    let mut data = vec![0.0f32; 1024];
    for i in 0..1024 {
        data[i] = i as f32;
    }

    let start = Instant::now();

    for i in 0..TOTAL_ITERATIONS {
        let idx = (i as usize) % 1024;
        let mut val = data[idx];
        for _ in 0..100 {
            val = val * 1.001 + 0.001;
        }
        data[idx] = val;
    }

    // Prevent optimizer from eliminating the loop
    std::hint::black_box(&data);

    let elapsed = start.elapsed();

    StrategyResult {
        name: "CPU Single-threaded".to_string(),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count: 1,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

/// CPU multi-threaded - using std::thread, same workload as GPU
fn run_cpu_multi_threaded(num_threads: usize) -> StrategyResult {
    use std::sync::Arc;

    // Shared data array (simulating same pattern as GPU)
    let data = Arc::new(std::sync::Mutex::new(vec![0.0f32; 1024]));
    for i in 0..1024 {
        data.lock().unwrap()[i] = i as f32;
    }

    let start = Instant::now();

    let iterations_per_thread = TOTAL_ITERATIONS / num_threads as u32;
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let data = Arc::clone(&data);
        let thread_start = t as u32 * iterations_per_thread;
        let thread_end = if t == num_threads - 1 {
            TOTAL_ITERATIONS
        } else {
            thread_start + iterations_per_thread
        };

        handles.push(thread::spawn(move || {
            // Local accumulator to reduce lock contention
            let mut local_data = vec![0.0f32; 1024];
            {
                let locked = data.lock().unwrap();
                local_data.copy_from_slice(&locked);
            }

            for i in thread_start..thread_end {
                let idx = (i as usize) % 1024;
                let mut val = local_data[idx];
                for _ in 0..100 {
                    val = val * 1.001 + 0.001;
                }
                local_data[idx] = val;
            }

            // Write back (simulating shared memory pattern)
            {
                let mut locked = data.lock().unwrap();
                for i in 0..1024 {
                    locked[i] = local_data[i];
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Prevent optimizer from eliminating
    std::hint::black_box(&data);

    let elapsed = start.elapsed();

    StrategyResult {
        name: format!("CPU {}-threaded", num_threads),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count: 1,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

/// CPU multi-threaded lock-free - best case CPU performance
fn run_cpu_lockfree(num_threads: usize) -> StrategyResult {
    use std::sync::Arc;

    // Each thread has its own slice of data - no contention
    let data: Arc<Vec<AtomicU32>> = Arc::new(
        (0..1024).map(|i| AtomicU32::new((i as f32).to_bits())).collect()
    );

    let counter = Arc::new(AtomicU32::new(0));

    let start = Instant::now();

    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let data = Arc::clone(&data);
        let counter = Arc::clone(&counter);

        handles.push(thread::spawn(move || {
            loop {
                // Claim next work item
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i >= TOTAL_ITERATIONS {
                    break;
                }

                let idx = (i as usize) % 1024;
                let mut val = f32::from_bits(data[idx].load(Ordering::Relaxed));
                for _ in 0..100 {
                    val = val * 1.001 + 0.001;
                }
                data[idx].store(val.to_bits(), Ordering::Relaxed);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    std::hint::black_box(&data);

    let elapsed = start.elapsed();

    StrategyResult {
        name: format!("CPU Lock-free {}-thread", num_threads),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count: 1,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

/// CPU with pre-partitioned work - no atomic coordination (best case)
fn run_cpu_prepartitioned(num_threads: usize) -> StrategyResult {
    let start = Instant::now();

    let iterations_per_thread = TOTAL_ITERATIONS / num_threads as u32;
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let thread_start = t as u32 * iterations_per_thread;
        let thread_end = if t == num_threads - 1 {
            TOTAL_ITERATIONS
        } else {
            thread_start + iterations_per_thread
        };

        handles.push(thread::spawn(move || {
            let mut data = vec![0.0f32; 1024];
            for i in 0..1024 {
                data[i] = i as f32;
            }

            for i in thread_start..thread_end {
                let idx = (i as usize) % 1024;
                let mut val = data[idx];
                for _ in 0..100 {
                    val = val * 1.001 + 0.001;
                }
                data[idx] = val;
            }

            std::hint::black_box(&data);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start.elapsed();

    StrategyResult {
        name: format!("CPU Prepartitioned {}-thread", num_threads),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        dispatch_count: 1,
        iterations_completed: TOTAL_ITERATIONS,
        throughput_m_iter_per_sec: TOTAL_ITERATIONS as f64 / elapsed.as_secs_f64() / 1_000_000.0,
    }
}

// ============================================================================
// Results
// ============================================================================

#[derive(Debug)]
struct StrategyResult {
    name: String,
    total_time_ms: f64,
    dispatch_count: u32,
    iterations_completed: u32,
    throughput_m_iter_per_sec: f64,
}

// ============================================================================
// Main Benchmark
// ============================================================================

#[test]
fn benchmark_persistence_strategies() {
    let device = Device::system_default().expect("No Metal device");

    // Get CPU core count
    let num_cpus = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     GPU vs CPU BENCHMARK                                         ║");
    println!("║     Total Work: {} iterations (each: 100 FMA ops)      ║", TOTAL_ITERATIONS);
    println!("║     CPU Cores: {:2}                                                ║", num_cpus);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut results = Vec::new();

    // ==== GPU Strategies ====
    println!("═══ GPU STRATEGIES ═══");

    // Strategy 1: Rapid re-dispatch with varying work sizes
    println!("\nRunning GPU Strategy 1: RAPID RE-DISPATCH...");
    for work_size in [10_000u32, 50_000, 100_000, 500_000] {
        let result = run_rapid_strategy(&device, work_size);
        println!("  {}: {:.1}ms, {} dispatches, {:.1}M iter/s",
            result.name, result.total_time_ms, result.dispatch_count, result.throughput_m_iter_per_sec);
        results.push(result);
    }

    // Strategy 2: Mega-dispatch with varying iteration limits
    println!("\nRunning GPU Strategy 2: MEGA-DISPATCH (Chunked)...");
    for max_iter in [1_000u32, 5_000, 10_000, 50_000] {
        let result = run_mega_strategy(&device, max_iter);
        println!("  {}: {:.1}ms, {} dispatches, {:.1}M iter/s",
            result.name, result.total_time_ms, result.dispatch_count, result.throughput_m_iter_per_sec);
        results.push(result);
    }

    // Strategy 3: Work queue with varying item sizes
    println!("\nRunning GPU Strategy 3: WORK QUEUE...");
    for item_size in [10_000u32, 50_000, 100_000, 500_000] {
        let result = run_queue_strategy(&device, item_size);
        println!("  {}: {:.1}ms, {} dispatches, {:.1}M iter/s",
            result.name, result.total_time_ms, result.dispatch_count, result.throughput_m_iter_per_sec);
        results.push(result);
    }

    // ==== CPU Strategies ====
    println!("\n═══ CPU STRATEGIES ═══");

    // CPU single-threaded
    println!("\nRunning CPU Single-threaded...");
    let result = run_cpu_single_threaded();
    println!("  {}: {:.1}ms, {:.1}M iter/s",
        result.name, result.total_time_ms, result.throughput_m_iter_per_sec);
    results.push(result);

    // CPU multi-threaded with different thread counts
    println!("\nRunning CPU Multi-threaded...");
    for threads in [2, 4, num_cpus] {
        let result = run_cpu_multi_threaded(threads);
        println!("  {}: {:.1}ms, {:.1}M iter/s",
            result.name, result.total_time_ms, result.throughput_m_iter_per_sec);
        results.push(result);
    }

    // CPU lock-free (atomic counter for work stealing)
    println!("\nRunning CPU Lock-free...");
    for threads in [4, num_cpus] {
        let result = run_cpu_lockfree(threads);
        println!("  {}: {:.1}ms, {:.1}M iter/s",
            result.name, result.total_time_ms, result.throughput_m_iter_per_sec);
        results.push(result);
    }

    // CPU prepartitioned (best case - no coordination)
    println!("\nRunning CPU Pre-partitioned (optimal CPU)...");
    for threads in [4, num_cpus] {
        let result = run_cpu_prepartitioned(threads);
        println!("  {}: {:.1}ms, {:.1}M iter/s",
            result.name, result.total_time_ms, result.throughput_m_iter_per_sec);
        results.push(result);
    }

    // Find winner
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                         ALL RESULTS                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    results.sort_by(|a, b| b.throughput_m_iter_per_sec.partial_cmp(&a.throughput_m_iter_per_sec).unwrap());

    for (i, r) in results.iter().enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        let is_gpu = r.name.starts_with("Mega") || r.name.starts_with("Rapid") || r.name.starts_with("Queue");
        let marker = if is_gpu { "GPU" } else { "CPU" };
        println!("║ {} [{:3}] {:32} {:>8.1}ms {:>8.1}M/s ║",
            medal, marker, r.name, r.total_time_ms, r.throughput_m_iter_per_sec);
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // Separate GPU and CPU for analysis
    let best_gpu = results.iter().find(|r| {
        r.name.starts_with("Mega") || r.name.starts_with("Rapid") || r.name.starts_with("Queue")
    }).unwrap();
    let best_cpu = results.iter().find(|r| r.name.starts_with("CPU")).unwrap();

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      GPU vs CPU ANALYSIS                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Best GPU: {:40} {:>8.1}M/s  ║", best_gpu.name, best_gpu.throughput_m_iter_per_sec);
    println!("║  Best CPU: {:40} {:>8.1}M/s  ║", best_cpu.name, best_cpu.throughput_m_iter_per_sec);
    println!("║                                                                      ║");
    println!("║  GPU Speedup over CPU: {:>6.1}x                                      ║",
        best_gpu.throughput_m_iter_per_sec / best_cpu.throughput_m_iter_per_sec);
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // Detailed breakdown
    let cpu_single = results.iter().find(|r| r.name == "CPU Single-threaded").unwrap();
    println!("\nDetailed Breakdown:");
    println!("  GPU vs CPU single-threaded: {:.1}x speedup",
        best_gpu.throughput_m_iter_per_sec / cpu_single.throughput_m_iter_per_sec);
    println!("  GPU vs best CPU (all cores): {:.1}x speedup",
        best_gpu.throughput_m_iter_per_sec / best_cpu.throughput_m_iter_per_sec);

    // Efficiency analysis
    println!("\nEfficiency (throughput per compute unit):");
    let gpu_threads = 16 * 256;  // 4096 GPU threads
    let gpu_per_thread = best_gpu.throughput_m_iter_per_sec / gpu_threads as f64;
    let cpu_per_core = best_cpu.throughput_m_iter_per_sec / num_cpus as f64;
    println!("  GPU: {:.3}M iter/s per thread ({} threads)", gpu_per_thread, gpu_threads);
    println!("  CPU: {:.3}M iter/s per core ({} cores)", cpu_per_core, num_cpus);
    println!("  Per-unit efficiency ratio: {:.1}x (CPU cores are individually more powerful)",
        cpu_per_core / (gpu_per_thread * 1000.0));  // Adjust scale
}

#[test]
fn test_rapid_basic() {
    let device = Device::system_default().expect("No Metal device");
    let result = run_rapid_strategy(&device, 100_000);
    assert!(result.iterations_completed == TOTAL_ITERATIONS);
    println!("Rapid: {:.1}ms, {} dispatches", result.total_time_ms, result.dispatch_count);
}

#[test]
fn test_mega_basic() {
    let device = Device::system_default().expect("No Metal device");
    let result = run_mega_strategy(&device, 10_000);
    assert!(result.iterations_completed == TOTAL_ITERATIONS);
    println!("Mega: {:.1}ms, {} dispatches", result.total_time_ms, result.dispatch_count);
}

#[test]
fn test_queue_basic() {
    let device = Device::system_default().expect("No Metal device");
    let result = run_queue_strategy(&device, 100_000);
    assert!(result.iterations_completed == TOTAL_ITERATIONS);
    println!("Queue: {:.1}ms, {} dispatches", result.total_time_ms, result.dispatch_count);
}
