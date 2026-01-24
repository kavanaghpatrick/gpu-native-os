// Issue #102: GPU Branch Optimization - Predication with select() for Branch-Free Execution
//
// PRD: Replace branchy if/else chains with Metal's select() function to eliminate warp divergence.
//
// This test compares:
// 1. Branchy version (current): Uses if/else chains, causes warp divergence
// 2. Predicated version (new): Uses select() for branch-free execution
//
// Both versions must produce identical results. Predicated version should be faster
// due to elimination of warp divergence.

use metal::*;
use std::time::Instant;

const DATA_SIZE: u32 = 10_000_000;
const WARM_UP_RUNS: usize = 3;
const TIMED_RUNS: usize = 10;
const TOLERANCE: f32 = 0.0001;

// ============================================================================
// GPU Shaders - Branchy vs Predicated
// ============================================================================

const GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// BRANCHY VERSION: Uses if/else chains (causes warp divergence)
// This is the BASELINE we're trying to improve
kernel void branch_compute_branchy(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;

    // DIVERGENT: Different threads in same warp take different paths
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

// PREDICATED VERSION: Uses select() for branch-free execution
// All threads compute ALL paths, then select the correct result
kernel void branch_compute_predicated(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;

    // COMPUTE ALL BRANCHES (no divergence - all threads do same work)
    float result0 = val * 2.0f + 1.0f;      // cat 0
    float result1 = val * 0.5f - 1.0f;      // cat 1
    float result2 = val * val;               // cat 2
    float result3 = 1.0f / (val + 1.0f);    // cat 3

    // SELECT RESULT BRANCHLESSLY using nested select()
    // select(a, b, condition) returns b if condition is true, else a
    float result = select(result1, result0, cat == 0);  // cat 0 or 1
    result = select(result, result2, cat == 2);          // or cat 2
    result = select(result, result3, cat == 3);          // or cat 3

    data[tid] = result;
}

// PREDICATED VERSION 2: Using mix() for smoother blending approach
// This version uses coefficient lookup to avoid even select() chains
kernel void branch_compute_coefficients(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;

    // Precompute all results
    float results[4];
    results[0] = val * 2.0f + 1.0f;
    results[1] = val * 0.5f - 1.0f;
    results[2] = val * val;
    results[3] = 1.0f / (val + 1.0f);

    // Direct array indexing (compiler may optimize this well)
    data[tid] = results[cat];
}

// SIMD VOTING VERSION: Check if warp is uniform, take fast path
kernel void branch_compute_simd_vote(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;

    // Check if entire SIMD group has same category
    uint first_cat = simd_broadcast(cat, 0);
    bool all_same = simd_all(cat == first_cat);

    float result;
    if (all_same) {
        // FAST PATH: All threads in SIMD group have same category
        // No divergence - direct computation
        if (first_cat == 0) {
            result = val * 2.0f + 1.0f;
        } else if (first_cat == 1) {
            result = val * 0.5f - 1.0f;
        } else if (first_cat == 2) {
            result = val * val;
        } else {
            result = 1.0f / (val + 1.0f);
        }
    } else {
        // SLOW PATH: Mixed categories - use predication
        float result0 = val * 2.0f + 1.0f;
        float result1 = val * 0.5f - 1.0f;
        float result2 = val * val;
        float result3 = 1.0f / (val + 1.0f);

        result = select(result1, result0, cat == 0);
        result = select(result, result2, cat == 2);
        result = select(result, result3, cat == 3);
    }

    data[tid] = result;
}
"#;

// ============================================================================
// CPU Reference Implementation
// ============================================================================

fn cpu_branch_compute(data: &mut [f32], categories: &[u32]) {
    let num_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    std::thread::scope(|s| {
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

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    kernel: String,
    times_ms: Vec<f64>,
    mean_ms: f64,
    median_ms: f64,
    std_dev_ms: f64,
    min_ms: f64,
    max_ms: f64,
    correct: bool,
}

impl BenchResult {
    fn new(name: &str, kernel: &str, times: Vec<f64>, correct: bool) -> Self {
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

        let min = *sorted.first().unwrap_or(&0.0);
        let max = *sorted.last().unwrap_or(&0.0);

        BenchResult {
            name: name.to_string(),
            kernel: kernel.to_string(),
            times_ms: times,
            mean_ms: mean,
            median_ms: median,
            std_dev_ms: std_dev,
            min_ms: min,
            max_ms: max,
            correct,
        }
    }
}

fn verify_correctness(gpu_ptr: *const f32, reference: &[f32]) -> (bool, usize) {
    let mut errors = 0;
    for idx in 0..reference.len() {
        let gpu_val = unsafe { *gpu_ptr.add(idx) };
        let cpu_val = reference[idx];
        let diff = (gpu_val - cpu_val).abs();
        let tolerance = TOLERANCE * cpu_val.abs().max(1.0);
        if diff > tolerance {
            if errors < 3 {
                println!("  Mismatch at {}: GPU={:.6} CPU={:.6} diff={:.6}",
                    idx, gpu_val, cpu_val, diff);
            }
            errors += 1;
        }
    }
    (errors == 0, errors)
}

fn create_pipeline(device: &Device, function_name: &str) -> (ComputePipelineState, CommandQueue) {
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

fn run_gpu_benchmark(
    device: &Device,
    kernel_name: &str,
    categories: &[u32],
    reference: &[f32],
) -> BenchResult {
    let (pipeline, queue) = create_pipeline(device, kernel_name);

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
        // Fresh data each run
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = (i as f32) * 0.001;
            }
        }

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

            let ptr = data_buf.contents() as *const f32;
            let (correct, errors) = verify_correctness(ptr, reference);
            if !correct {
                println!("  {} failed on run {} with {} errors", kernel_name, run, errors);
                all_correct = false;
            }
        }
    }

    BenchResult::new("GPU", kernel_name, times, all_correct)
}

fn run_cpu_benchmark(categories: &[u32]) -> BenchResult {
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

    BenchResult::new("CPU", "reference", times, true)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_predication_correctness() {
    let device = Device::system_default().expect("No Metal device");

    // Generate test data with random categories
    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();

    // Generate CPU reference
    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_branch_compute(&mut reference, &categories);

    println!("\n=== Issue #102: Predication Correctness Test ===\n");

    // Test each kernel for correctness
    let kernels = [
        "branch_compute_branchy",
        "branch_compute_predicated",
        "branch_compute_coefficients",
        "branch_compute_simd_vote",
    ];

    for kernel_name in &kernels {
        let (pipeline, queue) = create_pipeline(&device, kernel_name);

        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
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

        unsafe {
            let ptr = data_buf.contents() as *mut f32;
            for i in 0..DATA_SIZE {
                *ptr.add(i as usize) = (i as f32) * 0.001;
            }
        }

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

        let ptr = data_buf.contents() as *const f32;
        let (correct, errors) = verify_correctness(ptr, &reference);

        let status = if correct { "✓ PASS" } else { "✗ FAIL" };
        println!("  {}: {} (errors: {})", kernel_name, status, errors);

        assert!(correct, "{} produced incorrect results", kernel_name);
    }

    println!("\nAll kernels produce correct results!");
}

#[test]
fn bench_predication_vs_branchy() {
    // Must run in release mode for fair comparison
    assert!(
        !cfg!(debug_assertions),
        "Benchmark must run in release mode! Use: cargo test --release --test test_issue_102_predication"
    );

    let device = Device::system_default().expect("No Metal device");
    let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Issue #102: Predication with select() Benchmark                       ║");
    println!("║  Data Size: {} elements | CPU Cores: {} | Trials: {}                          ║",
        DATA_SIZE, num_cpus, TIMED_RUNS);
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Generate categories with random distribution (worst case for branchy version)
    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();

    // Generate CPU reference
    println!("\n[1/5] Generating CPU reference...");
    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_branch_compute(&mut reference, &categories);

    // Run CPU benchmark
    println!("[2/5] Running CPU benchmark...");
    let cpu_result = run_cpu_benchmark(&categories);
    println!("  CPU: {:.2}ms ± {:.2}ms", cpu_result.median_ms, cpu_result.std_dev_ms);

    // Run GPU benchmarks
    println!("[3/5] Running GPU branchy benchmark...");
    let branchy_result = run_gpu_benchmark(&device, "branch_compute_branchy", &categories, &reference);
    println!("  Branchy: {:.2}ms ± {:.2}ms (correct: {})",
        branchy_result.median_ms, branchy_result.std_dev_ms, branchy_result.correct);

    println!("[4/5] Running GPU predicated benchmark...");
    let predicated_result = run_gpu_benchmark(&device, "branch_compute_predicated", &categories, &reference);
    println!("  Predicated: {:.2}ms ± {:.2}ms (correct: {})",
        predicated_result.median_ms, predicated_result.std_dev_ms, predicated_result.correct);

    println!("[5/5] Running GPU coefficient lookup benchmark...");
    let coeffs_result = run_gpu_benchmark(&device, "branch_compute_coefficients", &categories, &reference);
    println!("  Coefficients: {:.2}ms ± {:.2}ms (correct: {})",
        coeffs_result.median_ms, coeffs_result.std_dev_ms, coeffs_result.correct);

    // Also test SIMD voting
    println!("[6/5] Running GPU SIMD voting benchmark...");
    let simd_result = run_gpu_benchmark(&device, "branch_compute_simd_vote", &categories, &reference);
    println!("  SIMD Vote: {:.2}ms ± {:.2}ms (correct: {})",
        simd_result.median_ms, simd_result.std_dev_ms, simd_result.correct);

    // Results summary
    println!("\n╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESULTS SUMMARY                                   ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Implementation         Median (ms)   vs CPU        vs Branchy   Correct      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");

    let results = [
        ("CPU Reference", &cpu_result),
        ("GPU Branchy", &branchy_result),
        ("GPU Predicated", &predicated_result),
        ("GPU Coefficients", &coeffs_result),
        ("GPU SIMD Vote", &simd_result),
    ];

    for (name, result) in &results {
        let vs_cpu = if result.median_ms > 0.0 {
            cpu_result.median_ms / result.median_ms
        } else {
            0.0
        };
        let vs_branchy = if result.median_ms > 0.0 {
            branchy_result.median_ms / result.median_ms
        } else {
            0.0
        };

        let vs_cpu_str = if *name == "CPU Reference" {
            "---".to_string()
        } else if vs_cpu > 1.0 {
            format!("{:.2}x faster", vs_cpu)
        } else {
            format!("{:.2}x slower", 1.0 / vs_cpu)
        };

        let vs_branchy_str = if *name == "CPU Reference" || *name == "GPU Branchy" {
            "---".to_string()
        } else if vs_branchy > 1.0 {
            format!("{:.2}x faster", vs_branchy)
        } else {
            format!("{:.2}x slower", 1.0 / vs_branchy)
        };

        let correct_str = if result.correct { "✓" } else { "✗" };

        println!("║  {:20} {:>8.2}      {:>12}  {:>12}  {}            ║",
            name, result.median_ms, vs_cpu_str, vs_branchy_str, correct_str);
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Key findings
    println!("\n📊 Key Findings:");
    let branchy_vs_cpu = cpu_result.median_ms / branchy_result.median_ms;
    let predicated_vs_branchy = branchy_result.median_ms / predicated_result.median_ms;
    let predicated_vs_cpu = cpu_result.median_ms / predicated_result.median_ms;

    if branchy_vs_cpu < 1.0 {
        println!("  • Branchy GPU is {:.1}x SLOWER than CPU (expected - warp divergence)", 1.0 / branchy_vs_cpu);
    } else {
        println!("  • Branchy GPU is {:.1}x faster than CPU", branchy_vs_cpu);
    }

    if predicated_vs_branchy > 1.0 {
        println!("  • Predicated is {:.1}x faster than Branchy (divergence eliminated!)", predicated_vs_branchy);
    } else {
        println!("  • Predicated is {:.1}x slower than Branchy", 1.0 / predicated_vs_branchy);
    }

    if predicated_vs_cpu > 1.0 {
        println!("  • Predicated GPU is {:.1}x faster than CPU ✓", predicated_vs_cpu);
    } else {
        println!("  • Predicated GPU is still {:.1}x slower than CPU", 1.0 / predicated_vs_cpu);
    }

    // Assert correctness
    assert!(branchy_result.correct, "Branchy kernel produced incorrect results");
    assert!(predicated_result.correct, "Predicated kernel produced incorrect results");
    assert!(coeffs_result.correct, "Coefficients kernel produced incorrect results");
    assert!(simd_result.correct, "SIMD Vote kernel produced incorrect results");
}

#[test]
fn test_predication_uniform_categories() {
    // Test with UNIFORM categories (best case for branchy, but predicated should still work)
    let device = Device::system_default().expect("No Metal device");

    println!("\n=== Testing with Uniform Categories (all category 0) ===\n");

    // All same category - no divergence even for branchy version
    let categories: Vec<u32> = vec![0; DATA_SIZE as usize];

    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_branch_compute(&mut reference, &categories);

    let branchy = run_gpu_benchmark(&device, "branch_compute_branchy", &categories, &reference);
    let predicated = run_gpu_benchmark(&device, "branch_compute_predicated", &categories, &reference);

    println!("  Branchy (uniform):    {:.2}ms (correct: {})", branchy.median_ms, branchy.correct);
    println!("  Predicated (uniform): {:.2}ms (correct: {})", predicated.median_ms, predicated.correct);

    let ratio = branchy.median_ms / predicated.median_ms;
    if ratio > 1.0 {
        println!("  → Predicated is {:.2}x faster even with uniform data", ratio);
    } else {
        println!("  → Branchy is {:.2}x faster with uniform data (expected)", 1.0 / ratio);
    }

    assert!(branchy.correct && predicated.correct, "Correctness failed");
}
