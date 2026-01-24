// Issue #106: GPU Branch Optimization - Persistent Threads with Work Queue Pattern
//
// PRD: Keep GPU threads alive processing work from atomic queue instead of one-shot dispatch.
//
// This demonstrates:
// 1. Traditional dispatch: Launch N threads for N items
// 2. Persistent threads: Threads stay alive, pull work from queue
//
// Benefits: Better load balancing, reduced kernel launch overhead

use metal::*;
use std::time::Instant;

const DATA_SIZE: u32 = 10_000_000;
const WARM_UP_RUNS: usize = 3;
const TIMED_RUNS: usize = 10;
const TOLERANCE: f32 = 0.0001;

// For persistent kernel, use fewer threads that process multiple items
const PERSISTENT_THREADS: u32 = 256 * 128;  // 32K threads
const MAX_ITEMS_PER_THREAD: u32 = (DATA_SIZE + PERSISTENT_THREADS - 1) / PERSISTENT_THREADS;

// ============================================================================
// GPU Shaders
// ============================================================================

const GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Work item structure for queue
struct WorkItem {
    uint data_index;
    uint category;
};

// TRADITIONAL: One thread per data element
kernel void compute_traditional(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;

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

// GRID-STRIDE: Fewer threads, each processes multiple items
kernel void compute_grid_stride(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    constant uint& total_threads [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes multiple elements with stride = total_threads
    for (uint i = tid; i < element_count; i += total_threads) {
        float val = data[i];
        uint cat = categories[i] % 4;

        if (cat == 0) {
            val = val * 2.0f + 1.0f;
        } else if (cat == 1) {
            val = val * 0.5f - 1.0f;
        } else if (cat == 2) {
            val = val * val;
        } else {
            val = 1.0f / (val + 1.0f);
        }

        data[i] = val;
    }
}

// WORK QUEUE: Threads pull work from atomic queue
kernel void compute_work_queue(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    device atomic_uint* queue_head [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread pulls batches of work until queue is empty
    while (true) {
        // Atomically claim a batch of work
        uint batch_start = atomic_fetch_add_explicit(queue_head, batch_size, memory_order_relaxed);

        // Check if we've exhausted the queue
        if (batch_start >= element_count) break;

        // Process this batch
        uint batch_end = min(batch_start + batch_size, element_count);
        for (uint i = batch_start; i < batch_end; i++) {
            float val = data[i];
            uint cat = categories[i] % 4;

            if (cat == 0) {
                val = val * 2.0f + 1.0f;
            } else if (cat == 1) {
                val = val * 0.5f - 1.0f;
            } else if (cat == 2) {
                val = val * val;
            } else {
                val = 1.0f / (val + 1.0f);
            }

            data[i] = val;
        }
    }
}

// WORK QUEUE with variable complexity (simulates imbalanced workload)
kernel void compute_variable_work(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    device const uint* complexity [[buffer(2)]],  // Work amount per element
    device atomic_uint* queue_head [[buffer(3)]],
    constant uint& element_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    while (true) {
        uint idx = atomic_fetch_add_explicit(queue_head, 1, memory_order_relaxed);
        if (idx >= element_count) break;

        float val = data[idx];
        uint cat = categories[idx] % 4;
        uint work = complexity[idx];

        // Variable amount of work per element
        for (uint w = 0; w < work; w++) {
            if (cat == 0) {
                val = val * 1.001f + 0.001f;
            } else if (cat == 1) {
                val = val * 0.999f - 0.001f;
            } else if (cat == 2) {
                val = val * 1.0001f;
            } else {
                val = val * 0.9999f;
            }
        }

        data[idx] = val;
    }
}

// TRADITIONAL for variable work (baseline)
kernel void compute_variable_traditional(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    device const uint* complexity [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;
    uint work = complexity[tid];

    for (uint w = 0; w < work; w++) {
        if (cat == 0) {
            val = val * 1.001f + 0.001f;
        } else if (cat == 1) {
            val = val * 0.999f - 0.001f;
        } else if (cat == 2) {
            val = val * 1.0001f;
        } else {
            val = val * 0.9999f;
        }
    }

    data[tid] = val;
}
"#;

// ============================================================================
// CPU Reference
// ============================================================================

fn cpu_compute(data: &mut [f32], categories: &[u32]) {
    for (val, &cat) in data.iter_mut().zip(categories.iter()) {
        let cat = cat % 4;
        *val = match cat {
            0 => *val * 2.0 + 1.0,
            1 => *val * 0.5 - 1.0,
            2 => *val * *val,
            _ => 1.0 / (*val + 1.0),
        };
    }
}

fn cpu_variable_compute(data: &mut [f32], categories: &[u32], complexity: &[u32]) {
    for ((val, &cat), &work) in data.iter_mut().zip(categories.iter()).zip(complexity.iter()) {
        let cat = cat % 4;
        for _ in 0..work {
            *val = match cat {
                0 => *val * 1.001 + 0.001,
                1 => *val * 0.999 - 0.001,
                2 => *val * 1.0001,
                _ => *val * 0.9999,
            };
        }
    }
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    median_ms: f64,
    std_dev_ms: f64,
    correct: bool,
}

impl BenchResult {
    fn new(name: &str, times: Vec<f64>, correct: bool) -> Self {
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

        BenchResult {
            name: name.to_string(),
            median_ms: median,
            std_dev_ms: std_dev,
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

fn run_traditional(device: &Device, categories: &[u32], reference: &[f32]) -> BenchResult {
    let pipeline = create_pipeline(device, "compute_traditional");
    let queue = device.new_command_queue();

    let cat_buf = device.new_buffer_with_data(
        categories.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
        let data_buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&cat_buf), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
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

            // Verify
            let ptr = data_buf.contents() as *const f32;
            let mut errors = 0;
            for i in 0..reference.len() {
                let diff = (unsafe { *ptr.add(i) } - reference[i]).abs();
                if diff > TOLERANCE { errors += 1; }
            }
            if errors > 0 { all_correct = false; }
        }
    }

    BenchResult::new("Traditional", times, all_correct)
}

fn run_grid_stride(device: &Device, categories: &[u32], reference: &[f32]) -> BenchResult {
    let pipeline = create_pipeline(device, "compute_grid_stride");
    let queue = device.new_command_queue();

    let cat_buf = device.new_buffer_with_data(
        categories.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let threads_buf = device.new_buffer_with_data(
        &PERSISTENT_THREADS as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
        let data_buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&cat_buf), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        enc.set_buffer(3, Some(&threads_buf), 0);
        enc.dispatch_threads(
            MTLSize::new(PERSISTENT_THREADS as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);

            let ptr = data_buf.contents() as *const f32;
            let mut errors = 0;
            for i in 0..reference.len() {
                let diff = (unsafe { *ptr.add(i) } - reference[i]).abs();
                if diff > TOLERANCE { errors += 1; }
            }
            if errors > 0 { all_correct = false; }
        }
    }

    BenchResult::new("Grid-Stride", times, all_correct)
}

fn run_work_queue(device: &Device, categories: &[u32], reference: &[f32]) -> BenchResult {
    let pipeline = create_pipeline(device, "compute_work_queue");
    let queue = device.new_command_queue();

    let cat_buf = device.new_buffer_with_data(
        categories.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let batch_size: u32 = 256;  // Process 256 items per atomic claim
    let batch_buf = device.new_buffer_with_data(
        &batch_size as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut times = Vec::new();
    let mut all_correct = true;

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
        let data_buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let head_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe { *(head_buf.contents() as *mut u32) = 0; }

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&cat_buf), 0);
        enc.set_buffer(2, Some(&head_buf), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&batch_buf), 0);
        enc.dispatch_threads(
            MTLSize::new(PERSISTENT_THREADS as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let elapsed = start.elapsed();

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);

            let ptr = data_buf.contents() as *const f32;
            let mut errors = 0;
            for i in 0..reference.len() {
                let diff = (unsafe { *ptr.add(i) } - reference[i]).abs();
                if diff > TOLERANCE { errors += 1; }
            }
            if errors > 0 { all_correct = false; }
        }
    }

    BenchResult::new("Work Queue", times, all_correct)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_persistent_threads_correctness() {
    let device = Device::system_default().expect("No Metal device");

    println!("\n=== Issue #106: Persistent Threads Correctness Test ===\n");

    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_compute(&mut reference, &categories);

    let traditional = run_traditional(&device, &categories, &reference);
    let grid_stride = run_grid_stride(&device, &categories, &reference);
    let work_queue = run_work_queue(&device, &categories, &reference);

    println!("  Traditional: {}", if traditional.correct { "✓ PASS" } else { "✗ FAIL" });
    println!("  Grid-Stride: {}", if grid_stride.correct { "✓ PASS" } else { "✗ FAIL" });
    println!("  Work Queue:  {}", if work_queue.correct { "✓ PASS" } else { "✗ FAIL" });

    assert!(traditional.correct && grid_stride.correct && work_queue.correct);
    println!("\nAll persistent thread implementations produce correct results!");
}

#[test]
fn bench_persistent_threads() {
    assert!(!cfg!(debug_assertions), "Benchmark must run in release mode!");

    let device = Device::system_default().expect("No Metal device");

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Issue #106: Persistent Threads Benchmark                              ║");
    println!("║  Data Size: {} elements | Trials: {}                                         ║",
        DATA_SIZE, TIMED_RUNS);
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_compute(&mut reference, &categories);

    println!("\nUniform Workload (all items same complexity):\n");

    let traditional = run_traditional(&device, &categories, &reference);
    let grid_stride = run_grid_stride(&device, &categories, &reference);
    let work_queue = run_work_queue(&device, &categories, &reference);

    println!("  Traditional: {:.2}ms (correct: {})", traditional.median_ms, traditional.correct);
    println!("  Grid-Stride: {:.2}ms (correct: {})", grid_stride.median_ms, grid_stride.correct);
    println!("  Work Queue:  {:.2}ms (correct: {})", work_queue.median_ms, work_queue.correct);

    println!("\n╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESULTS SUMMARY                                   ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");

    let results = [&traditional, &grid_stride, &work_queue];
    for result in &results {
        let vs_traditional = if result.name == "Traditional" {
            "---".to_string()
        } else if traditional.median_ms / result.median_ms > 1.0 {
            format!("{:.2}x faster", traditional.median_ms / result.median_ms)
        } else {
            format!("{:.2}x slower", result.median_ms / traditional.median_ms)
        };

        println!("║  {:16} {:>7.2}ms    {:>15}  {}                         ║",
            result.name, result.median_ms, vs_traditional,
            if result.correct { "✓" } else { "✗" });
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 Key Findings:");
    let best = results.iter().min_by(|a, b| a.median_ms.partial_cmp(&b.median_ms).unwrap()).unwrap();
    println!("  • Best: {} ({:.2}ms)", best.name, best.median_ms);

    if grid_stride.median_ms < traditional.median_ms {
        println!("  • Grid-stride is {:.1}% faster than traditional",
            (1.0 - grid_stride.median_ms / traditional.median_ms) * 100.0);
    }

    for result in &results {
        assert!(result.correct, "{} produced incorrect results", result.name);
    }
}
