// Issue #105: GPU Branch Optimization - Branch-Free Math with Lookup Tables and Polynomials
//
// PRD: Replace conditional math operations with branch-free alternatives:
//      lookup tables in constant memory, polynomial approximations, and bit manipulation.
//
// This test compares:
// 1. Branchy math with if/else chains
// 2. LUT-based computation using constant memory tables
// 3. Polynomial-based computation using coefficient lookup

use metal::*;
use std::time::Instant;

const DATA_SIZE: u32 = 10_000_000;
const NUM_CATEGORIES: u32 = 4;
const WARM_UP_RUNS: usize = 3;
const TIMED_RUNS: usize = 10;
const TOLERANCE: f32 = 0.0001;

// ============================================================================
// GPU Shaders
// ============================================================================

const GPU_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// BRANCHY: Original if/else chain (baseline)
kernel void math_branchy(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid] % 4;

    float result;
    if (cat == 0) {
        // Linear transform: 2x + 1
        result = val * 2.0f + 1.0f;
    } else if (cat == 1) {
        // Linear transform: 0.5x - 1
        result = val * 0.5f - 1.0f;
    } else if (cat == 2) {
        // Quadratic: x^2
        result = val * val;
    } else {
        // Reciprocal: 1/(x+1)
        result = 1.0f / (val + 1.0f);
    }

    data[tid] = result;
}

// LUT-BASED: Precomputed coefficients for linear/polynomial operations
// Each category has (a, b, c, d) where result = a*x^2 + b*x + c + d/(x+1)
constant float4 TRANSFORM_COEFFS[4] = {
    float4(0.0f, 2.0f, 1.0f, 0.0f),    // cat 0: 2x + 1
    float4(0.0f, 0.5f, -1.0f, 0.0f),   // cat 1: 0.5x - 1
    float4(1.0f, 0.0f, 0.0f, 0.0f),    // cat 2: x^2
    float4(0.0f, 0.0f, 0.0f, 1.0f),    // cat 3: 1/(x+1)
};

kernel void math_lut(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float x = data[tid];
    uint cat = categories[tid] % 4;

    // Load coefficients from constant memory (no branch)
    float4 c = TRANSFORM_COEFFS[cat];

    // Unified formula: a*x^2 + b*x + c + d/(x+1)
    // All threads execute same instructions, just different coefficients
    float result = c.x * x * x + c.y * x + c.z + c.w / (x + 1.0f);

    data[tid] = result;
}

// POLYNOMIAL: Extended LUT with more complex transforms
// Demonstrates how to handle transcendental functions
constant float POLY_COEFFS[4][8] = {
    // cat 0: 2x + 1 (exact)
    {0.0f, 0.0f, 0.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f},
    // cat 1: 0.5x - 1 (exact)
    {0.0f, 0.0f, 0.0f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f},
    // cat 2: x^2 (exact)
    {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    // cat 3: 1/(x+1) - use polynomial approx for x in [0,10]
    // Approximated as: c0 + c1*x + c2*x^2 + c3*x^3 (not used, we use direct formula)
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},  // flag for reciprocal
};

kernel void math_polynomial(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float x = data[tid];
    uint cat = categories[tid] % 4;

    // Horner's method for polynomial evaluation (branch-free)
    float result = 0.0f;

    // Load coefficients
    float a0 = POLY_COEFFS[cat][0];
    float a1 = POLY_COEFFS[cat][1];
    float a2 = POLY_COEFFS[cat][2];
    float a3 = POLY_COEFFS[cat][3];
    float a4 = POLY_COEFFS[cat][4];
    float recip_flag = POLY_COEFFS[cat][7];

    // Polynomial: a4 + a3*x + a2*x^2 + a1*x^3 + a0*x^4
    result = a0;
    result = result * x + a1;
    result = result * x + a2;
    result = result * x + a3;
    result = result * x + a4;

    // Handle reciprocal case using select (no branch)
    float recip_result = 1.0f / (x + 1.0f);
    result = select(result, recip_result, recip_flag > 0.5f);

    data[tid] = result;
}

// ARRAY INDEX: Direct array indexing (compiler may optimize)
kernel void math_array_index(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float x = data[tid];
    uint cat = categories[tid] % 4;

    // Compute all 4 results
    float results[4];
    results[0] = x * 2.0f + 1.0f;
    results[1] = x * 0.5f - 1.0f;
    results[2] = x * x;
    results[3] = 1.0f / (x + 1.0f);

    // Direct index (compiler may optimize to predicated select)
    data[tid] = results[cat];
}

// PURE SELECT: Chain of select operations (guaranteed no branches)
kernel void math_select_chain(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float x = data[tid];
    uint cat = categories[tid] % 4;

    // Compute all results
    float r0 = x * 2.0f + 1.0f;
    float r1 = x * 0.5f - 1.0f;
    float r2 = x * x;
    float r3 = 1.0f / (x + 1.0f);

    // Select chain (guaranteed branch-free)
    float result = select(r1, r0, cat == 0u);
    result = select(result, r2, cat == 2u);
    result = select(result, r3, cat == 3u);

    data[tid] = result;
}
"#;

// ============================================================================
// CPU Reference
// ============================================================================

fn cpu_math(data: &mut [f32], categories: &[u32]) {
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

fn verify_correctness(gpu_ptr: *const f32, reference: &[f32]) -> (bool, usize) {
    let mut errors = 0;
    for idx in 0..reference.len() {
        let gpu_val = unsafe { *gpu_ptr.add(idx) };
        let cpu_val = reference[idx];
        let diff = (gpu_val - cpu_val).abs();
        let tolerance = TOLERANCE * cpu_val.abs().max(1.0);
        if diff > tolerance {
            if errors < 3 {
                println!("  Mismatch at {}: GPU={:.6} CPU={:.6}", idx, gpu_val, cpu_val);
            }
            errors += 1;
        }
    }
    (errors == 0, errors)
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

fn run_gpu_benchmark(
    device: &Device,
    kernel_name: &str,
    categories: &[u32],
    reference: &[f32],
) -> BenchResult {
    let pipeline = create_pipeline(device, kernel_name);
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

            let ptr = data_buf.contents() as *const f32;
            let (correct, errors) = verify_correctness(ptr, reference);
            if !correct {
                println!("  {} failed on run {} with {} errors", kernel_name, run, errors);
                all_correct = false;
            }
        }
    }

    BenchResult::new(kernel_name, times, all_correct)
}

fn run_cpu_benchmark(categories: &[u32]) -> BenchResult {
    let mut times = Vec::new();

    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let mut data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();

        let start = Instant::now();
        cpu_math(&mut data, categories);
        let elapsed = start.elapsed();
        std::hint::black_box(&data);

        if run >= WARM_UP_RUNS {
            times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    BenchResult::new("CPU", times, true)
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_branchfree_math_correctness() {
    let device = Device::system_default().expect("No Metal device");

    println!("\n=== Issue #105: Branch-Free Math Correctness Test ===\n");

    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_math(&mut reference, &categories);

    let kernels = [
        "math_branchy",
        "math_lut",
        "math_polynomial",
        "math_array_index",
        "math_select_chain",
    ];

    for kernel_name in &kernels {
        let result = run_gpu_benchmark(&device, kernel_name, &categories, &reference);
        let status = if result.correct { "✓ PASS" } else { "✗ FAIL" };
        println!("  {}: {}", kernel_name, status);
        assert!(result.correct, "{} produced incorrect results", kernel_name);
    }

    println!("\nAll branch-free math kernels produce correct results!");
}

#[test]
fn bench_branchfree_math() {
    assert!(
        !cfg!(debug_assertions),
        "Benchmark must run in release mode!"
    );

    let device = Device::system_default().expect("No Metal device");

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Issue #105: Branch-Free Math Benchmark                                ║");
    println!("║  Data Size: {} elements | Categories: {} | Trials: {}                         ║",
        DATA_SIZE, NUM_CATEGORIES, TIMED_RUNS);
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let mut reference: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();
    cpu_math(&mut reference, &categories);

    println!("\nRunning benchmarks...\n");

    let cpu_result = run_cpu_benchmark(&categories);
    println!("  CPU: {:.2}ms", cpu_result.median_ms);

    let branchy_result = run_gpu_benchmark(&device, "math_branchy", &categories, &reference);
    println!("  Branchy: {:.2}ms", branchy_result.median_ms);

    let lut_result = run_gpu_benchmark(&device, "math_lut", &categories, &reference);
    println!("  LUT: {:.2}ms", lut_result.median_ms);

    let poly_result = run_gpu_benchmark(&device, "math_polynomial", &categories, &reference);
    println!("  Polynomial: {:.2}ms", poly_result.median_ms);

    let array_result = run_gpu_benchmark(&device, "math_array_index", &categories, &reference);
    println!("  Array Index: {:.2}ms", array_result.median_ms);

    let select_result = run_gpu_benchmark(&device, "math_select_chain", &categories, &reference);
    println!("  Select Chain: {:.2}ms", select_result.median_ms);

    // Results summary
    println!("\n╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESULTS SUMMARY                                   ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");

    let results = [
        &cpu_result,
        &branchy_result,
        &lut_result,
        &poly_result,
        &array_result,
        &select_result,
    ];

    for result in &results {
        let vs_cpu = if result.name == "CPU" {
            "---".to_string()
        } else if cpu_result.median_ms / result.median_ms > 1.0 {
            format!("{:.2}x faster", cpu_result.median_ms / result.median_ms)
        } else {
            format!("{:.2}x slower", result.median_ms / cpu_result.median_ms)
        };

        let vs_branchy = if result.name == "CPU" || result.name == "math_branchy" {
            "---".to_string()
        } else if branchy_result.median_ms / result.median_ms > 1.0 {
            format!("{:.2}x faster", branchy_result.median_ms / result.median_ms)
        } else {
            format!("{:.2}x slower", result.median_ms / branchy_result.median_ms)
        };

        println!("║  {:16} {:>7.2}ms    {:>12}  {:>12}  {}             ║",
            result.name, result.median_ms, vs_cpu, vs_branchy,
            if result.correct { "✓" } else { "✗" });
    }

    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Find best GPU result
    let gpu_results = [&branchy_result, &lut_result, &poly_result, &array_result, &select_result];
    let best = gpu_results.iter().min_by(|a, b| a.median_ms.partial_cmp(&b.median_ms).unwrap()).unwrap();

    println!("\n📊 Key Findings:");
    println!("  • Best GPU: {} ({:.2}ms)", best.name, best.median_ms);

    let improvement = branchy_result.median_ms / best.median_ms;
    if improvement > 1.0 {
        println!("  • {:.1}% improvement over branchy", (improvement - 1.0) * 100.0);
    }

    // Assert correctness
    for result in &results {
        assert!(result.correct, "{} produced incorrect results", result.name);
    }
}
