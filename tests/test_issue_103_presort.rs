// Issue #103: GPU Branch Optimization - Pre-Sort Data by Category for Warp Coherence
//
// PRD: Sort input data by category before GPU processing so threads in same warp
//      handle same category, eliminating divergence even with branchy code.
//
// This test compares:
// 1. Unsorted version: Random category distribution causes warp divergence
// 2. Pre-sorted version: Categories grouped together, no divergence per warp
//
// The sort has a cost, but if divergence savings exceed sort cost, we win.

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

// Count elements per category (histogram)
kernel void count_categories(
    device const uint* categories [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;
    uint cat = categories[tid] % 4;
    atomic_fetch_add_explicit(&counts[cat], 1, memory_order_relaxed);
}

// Compute prefix sum for bin offsets (single thread for simplicity)
kernel void prefix_sum(
    device uint* counts [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    constant uint& num_categories [[buffer(2)]]
) {
    uint offset = 0;
    for (uint i = 0; i < num_categories; i++) {
        offsets[i] = offset;
        offset += counts[i];
    }
}

// Scatter elements into sorted order
kernel void scatter_by_category(
    device const float* input_data [[buffer(0)]],
    device const uint* input_categories [[buffer(1)]],
    device float* output_data [[buffer(2)]],
    device uint* output_categories [[buffer(3)]],
    device atomic_uint* bin_counters [[buffer(4)]],
    device const uint* bin_offsets [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    uint cat = input_categories[tid] % 4;
    float val = input_data[tid];

    // Atomically get slot within this category's bin
    uint slot = atomic_fetch_add_explicit(&bin_counters[cat], 1, memory_order_relaxed);
    uint dest_idx = bin_offsets[cat] + slot;

    output_data[dest_idx] = val;
    output_categories[dest_idx] = cat;
}

// Branch compute on sorted data (each warp processes same category = no divergence)
kernel void branch_compute_sorted(
    device float* data [[buffer(0)]],
    device const uint* categories [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= element_count) return;

    float val = data[tid];
    uint cat = categories[tid];  // Already sorted, no modulo needed

    // BRANCHY but COHERENT: All threads in warp have same category
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

// Branch compute on unsorted data (baseline - causes divergence)
kernel void branch_compute_unsorted(
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
"#;

// ============================================================================
// CPU Reference
// ============================================================================

fn cpu_branch_compute(data: &mut [f32], categories: &[u32]) {
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

fn verify_correctness(gpu_ptr: *const f32, reference: &[f32]) -> bool {
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
    errors == 0
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
// GPU Sort Implementation
// ============================================================================

struct GpuSorter {
    count_pipeline: ComputePipelineState,
    prefix_pipeline: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
    queue: CommandQueue,
}

impl GpuSorter {
    fn new(device: &Device) -> Self {
        GpuSorter {
            count_pipeline: create_pipeline(device, "count_categories"),
            prefix_pipeline: create_pipeline(device, "prefix_sum"),
            scatter_pipeline: create_pipeline(device, "scatter_by_category"),
            queue: device.new_command_queue(),
        }
    }

    fn sort(
        &self,
        device: &Device,
        input_data: &Buffer,
        input_categories: &Buffer,
        output_data: &Buffer,
        output_categories: &Buffer,
    ) -> f64 {
        let start = Instant::now();

        // Allocate temporary buffers
        let counts_buf = device.new_buffer(
            (NUM_CATEGORIES * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let offsets_buf = device.new_buffer(
            (NUM_CATEGORIES * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let bin_counters_buf = device.new_buffer(
            (NUM_CATEGORIES * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let count_buf = device.new_buffer_with_data(
            &DATA_SIZE as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let num_cats_buf = device.new_buffer_with_data(
            &NUM_CATEGORIES as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero out counts
        unsafe {
            let ptr = counts_buf.contents() as *mut u32;
            for i in 0..NUM_CATEGORIES {
                *ptr.add(i as usize) = 0;
            }
            let ptr = bin_counters_buf.contents() as *mut u32;
            for i in 0..NUM_CATEGORIES {
                *ptr.add(i as usize) = 0;
            }
        }

        // Step 1: Count elements per category
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.count_pipeline);
        enc.set_buffer(0, Some(input_categories), 0);
        enc.set_buffer(1, Some(&counts_buf), 0);
        enc.set_buffer(2, Some(&count_buf), 0);
        enc.dispatch_threads(
            MTLSize::new(DATA_SIZE as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Step 2: Prefix sum (single thread)
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.prefix_pipeline);
        enc.set_buffer(0, Some(&counts_buf), 0);
        enc.set_buffer(1, Some(&offsets_buf), 0);
        enc.set_buffer(2, Some(&num_cats_buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Step 3: Scatter into sorted order
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.scatter_pipeline);
        enc.set_buffer(0, Some(input_data), 0);
        enc.set_buffer(1, Some(input_categories), 0);
        enc.set_buffer(2, Some(output_data), 0);
        enc.set_buffer(3, Some(output_categories), 0);
        enc.set_buffer(4, Some(&bin_counters_buf), 0);
        enc.set_buffer(5, Some(&offsets_buf), 0);
        enc.set_buffer(6, Some(&count_buf), 0);
        enc.dispatch_threads(
            MTLSize::new(DATA_SIZE as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        start.elapsed().as_secs_f64() * 1000.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_presort_correctness() {
    let device = Device::system_default().expect("No Metal device");

    // Generate random categories
    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();

    // CPU reference (processes in original order)
    let mut cpu_data = data.clone();
    cpu_branch_compute(&mut cpu_data, &categories);

    // GPU sort + compute
    let sorter = GpuSorter::new(&device);

    let input_data_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_cat_buf = device.new_buffer_with_data(
        categories.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sorted_data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sorted_cat_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    sorter.sort(
        &device,
        &input_data_buf,
        &input_cat_buf,
        &sorted_data_buf,
        &sorted_cat_buf,
    );

    // Verify sorting worked - categories should be grouped
    println!("\n=== Issue #103: Pre-Sort Correctness Test ===\n");

    let sorted_cats = unsafe {
        std::slice::from_raw_parts(
            sorted_cat_buf.contents() as *const u32,
            DATA_SIZE as usize,
        )
    };

    // Check that categories are sorted (all 0s, then all 1s, etc.)
    let mut prev_cat = 0u32;
    let mut category_transitions = 0;
    for &cat in sorted_cats {
        if cat < prev_cat {
            println!("  ERROR: Categories not sorted! Found {} after {}", cat, prev_cat);
            panic!("Sort failed");
        }
        if cat != prev_cat {
            category_transitions += 1;
            prev_cat = cat;
        }
    }
    println!("  Categories sorted correctly ({} transitions)", category_transitions);

    // Copy sorted data BEFORE compute so we can verify against original values
    let sorted_input: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            sorted_data_buf.contents() as *const f32,
            DATA_SIZE as usize,
        ).to_vec()
    };

    // Verify all input values are present (no data loss)
    let input_sum: f64 = data.iter().map(|v| *v as f64).sum();
    let sorted_sum: f64 = sorted_input.iter().map(|v| *v as f64).sum();
    let sum_diff = (input_sum - sorted_sum).abs();
    println!("  Data integrity check: input_sum={:.2}, sorted_sum={:.2}, diff={:.6}",
        input_sum, sorted_sum, sum_diff);
    assert!(sum_diff < 0.01, "Data loss during sort!");

    // Now run branch compute on sorted data
    let compute_pipeline = create_pipeline(&device, "branch_compute_sorted");
    let queue = device.new_command_queue();
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&compute_pipeline);
    enc.set_buffer(0, Some(&sorted_data_buf), 0);
    enc.set_buffer(1, Some(&sorted_cat_buf), 0);
    enc.set_buffer(2, Some(&count_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(DATA_SIZE as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify correctness by checking each element matches expected transformation
    let computed_data = unsafe {
        std::slice::from_raw_parts(
            sorted_data_buf.contents() as *const f32,
            DATA_SIZE as usize,
        )
    };

    let mut errors = 0;
    for i in 0..DATA_SIZE as usize {
        let val = sorted_input[i];  // Original value BEFORE compute (from our copy)
        let cat = sorted_cats[i];
        let expected = match cat {
            0 => val * 2.0 + 1.0,
            1 => val * 0.5 - 1.0,
            2 => val * val,
            _ => 1.0 / (val + 1.0),
        };
        let actual = computed_data[i];
        let diff = (expected - actual).abs();
        let tolerance = TOLERANCE * expected.abs().max(1.0);
        if diff > tolerance {
            if errors < 3 {
                println!("  Mismatch at {}: expected={:.6} actual={:.6} (cat={}, input={})",
                    i, expected, actual, cat, val);
            }
            errors += 1;
        }
    }

    let correct = errors == 0;
    println!("  Sorted compute correctness: {} ({} errors)", if correct { "✓ PASS" } else { "✗ FAIL" }, errors);
    assert!(correct, "Sorted branch compute produced incorrect results");
}

#[test]
fn bench_presort_vs_unsorted() {
    assert!(
        !cfg!(debug_assertions),
        "Benchmark must run in release mode! Use: cargo test --release --test test_issue_103_presort"
    );

    let device = Device::system_default().expect("No Metal device");

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Issue #103: Pre-Sort Data by Category Benchmark                       ║");
    println!("║  Data Size: {} elements | Categories: {} | Trials: {}                         ║",
        DATA_SIZE, NUM_CATEGORIES, TIMED_RUNS);
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Generate random categories (worst case for unsorted)
    let categories: Vec<u32> = (0..DATA_SIZE).map(|i| (i * 7 + 13) % 4).collect();
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();

    // CPU reference
    let mut cpu_data = data.clone();
    cpu_branch_compute(&mut cpu_data, &categories);

    // Pre-sort data once
    let sorter = GpuSorter::new(&device);
    let input_data_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_cat_buf = device.new_buffer_with_data(
        categories.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sorted_data_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sorted_cat_buf = device.new_buffer(
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    println!("\n[1/4] Pre-sorting data by category...");
    let sort_time = sorter.sort(
        &device,
        &input_data_buf,
        &input_cat_buf,
        &sorted_data_buf,
        &sorted_cat_buf,
    );
    println!("  Sort time: {:.2}ms", sort_time);

    // Benchmark unsorted compute
    println!("\n[2/4] Benchmarking UNSORTED compute...");
    let unsorted_pipeline = create_pipeline(&device, "branch_compute_unsorted");
    let queue = device.new_command_queue();
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let mut unsorted_times = Vec::new();
    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        // Fresh data
        let data_buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&unsorted_pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&input_cat_buf), 0);
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
            unsorted_times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }
    let unsorted_result = BenchResult::new("Unsorted", unsorted_times, true);
    println!("  Median: {:.2}ms ± {:.2}ms", unsorted_result.median_ms, unsorted_result.std_dev_ms);

    // Benchmark sorted compute (assumes data is already sorted)
    println!("\n[3/4] Benchmarking SORTED compute...");
    let sorted_pipeline = create_pipeline(&device, "branch_compute_sorted");

    let mut sorted_times = Vec::new();
    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        // Copy sorted data to fresh buffer
        let data_buf = device.new_buffer(
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let src = sorted_data_buf.contents() as *const f32;
            let dst = data_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(src, dst, DATA_SIZE as usize);
        }

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&sorted_pipeline);
        enc.set_buffer(0, Some(&data_buf), 0);
        enc.set_buffer(1, Some(&sorted_cat_buf), 0);
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
            sorted_times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }
    let sorted_result = BenchResult::new("Sorted", sorted_times, true);
    println!("  Median: {:.2}ms ± {:.2}ms", sorted_result.median_ms, sorted_result.std_dev_ms);

    // CPU benchmark
    println!("\n[4/4] Benchmarking CPU...");
    let mut cpu_times = Vec::new();
    for run in 0..(WARM_UP_RUNS + TIMED_RUNS) {
        let mut data_copy = data.clone();
        let start = Instant::now();
        cpu_branch_compute(&mut data_copy, &categories);
        let elapsed = start.elapsed();
        std::hint::black_box(&data_copy);
        if run >= WARM_UP_RUNS {
            cpu_times.push(elapsed.as_secs_f64() * 1000.0);
        }
    }
    let cpu_result = BenchResult::new("CPU", cpu_times, true);
    println!("  Median: {:.2}ms ± {:.2}ms", cpu_result.median_ms, cpu_result.std_dev_ms);

    // Results summary
    println!("\n╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              RESULTS SUMMARY                                   ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════╣");

    let sorted_vs_unsorted = unsorted_result.median_ms / sorted_result.median_ms;
    let sorted_total = sort_time + sorted_result.median_ms;
    let sorted_total_vs_unsorted = unsorted_result.median_ms / sorted_total;

    println!("║  CPU:                    {:>7.2}ms                                            ║", cpu_result.median_ms);
    println!("║  GPU Unsorted:           {:>7.2}ms                                            ║", unsorted_result.median_ms);
    println!("║  GPU Sorted (compute):   {:>7.2}ms  ({:.2}x vs unsorted)                      ║",
        sorted_result.median_ms, sorted_vs_unsorted);
    println!("║  GPU Sort overhead:      {:>7.2}ms                                            ║", sort_time);
    println!("║  GPU Sorted (total):     {:>7.2}ms  ({:.2}x vs unsorted)                      ║",
        sorted_total, sorted_total_vs_unsorted);
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 Key Findings:");
    if sorted_vs_unsorted > 1.0 {
        println!("  • Sorted compute is {:.1}x FASTER than unsorted", sorted_vs_unsorted);
    } else {
        println!("  • Sorted compute is {:.1}x SLOWER than unsorted", 1.0 / sorted_vs_unsorted);
    }

    if sorted_total_vs_unsorted > 1.0 {
        println!("  • Including sort cost, still {:.1}x faster overall ✓", sorted_total_vs_unsorted);
    } else {
        println!("  • Including sort cost, {:.1}x slower overall (sort overhead too high)", 1.0 / sorted_total_vs_unsorted);
    }

    let cpu_vs_sorted = cpu_result.median_ms / sorted_total;
    if cpu_vs_sorted < 1.0 {
        println!("  • CPU is still {:.1}x faster than sorted GPU", 1.0 / cpu_vs_sorted);
    } else {
        println!("  • Sorted GPU is {:.1}x faster than CPU ✓", cpu_vs_sorted);
    }
}

#[test]
fn test_presort_uniform_categories() {
    // Test with uniform categories (should see no difference)
    let device = Device::system_default().expect("No Metal device");

    println!("\n=== Testing with Uniform Categories (all category 0) ===\n");

    let categories: Vec<u32> = vec![0; DATA_SIZE as usize];
    let data: Vec<f32> = (0..DATA_SIZE).map(|i| (i as f32) * 0.001).collect();

    let unsorted_pipeline = create_pipeline(&device, "branch_compute_unsorted");
    let queue = device.new_command_queue();
    let count_buf = device.new_buffer_with_data(
        &DATA_SIZE as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let cat_buf = device.new_buffer_with_data(
        categories.as_ptr() as *const _,
        (DATA_SIZE * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Time unsorted with uniform data (should be fast - no divergence anyway)
    let mut times = Vec::new();
    for _ in 0..5 {
        let data_buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (DATA_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&unsorted_pipeline);
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

        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let median = {
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    println!("  Uniform categories (no divergence): {:.2}ms", median);
    println!("  This is the theoretical best case for branchy code.");
}
