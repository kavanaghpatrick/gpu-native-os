// Automated GPU vs CPU Benchmark
//
// Runs comprehensive benchmarks without user interaction.
// Outputs CSV data for analysis.
//
// Usage:
//   cargo run --release --example benchmark_auto
//   cargo run --release --example benchmark_auto -- --iterations 100
//   cargo run --release --example benchmark_auto -- --csv results.csv

use metal::*;
use std::time::Instant;
use std::env;
use std::fs::File;
use std::io::Write;

// Widget counts to test
const WIDGET_COUNTS: [usize; 7] = [32, 64, 128, 256, 512, 1024, 2048];
const DEFAULT_ITERATIONS: usize = 50;
const WARMUP_ITERATIONS: usize = 10;

// ============================================================================
// Benchmark Data Structures
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BenchWidget {
    x: f32,
    y: f32,
    z_order: u32,
    color: u32,
}

#[derive(Debug, Clone)]
struct BenchResult {
    widget_count: usize,
    mode: String,
    gpu_mean_us: f64,
    gpu_std_us: f64,
    gpu_min_us: f64,
    gpu_max_us: f64,
    cpu_mean_us: f64,
    cpu_std_us: f64,
    cpu_min_us: f64,
    cpu_max_us: f64,
    speedup: f64,
    iterations: usize,
}

// ============================================================================
// CPU Implementations
// ============================================================================

fn cpu_quicksort(widgets: &mut [BenchWidget]) {
    widgets.sort_by_key(|w| w.z_order);
}

fn cpu_hit_test(widgets: &[BenchWidget], test_x: f32, test_y: f32, widget_size: f32) -> Option<usize> {
    let half_size = widget_size / 2.0;
    for (i, w) in widgets.iter().enumerate().rev() {
        if test_x >= w.x - half_size && test_x <= w.x + half_size &&
           test_y >= w.y - half_size && test_y <= w.y + half_size {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// GPU Benchmark Implementation
// ============================================================================

struct GpuBenchmark {
    device: Device,
    command_queue: CommandQueue,
    sort_pipeline: ComputePipelineState,
    hittest_pipeline: ComputePipelineState,
    widgets_buffer: Buffer,
    result_buffer: Buffer,
    max_widgets: usize,
}

impl GpuBenchmark {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device");
        let command_queue = device.new_command_queue();

        // Compile shaders
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            struct Widget {
                float x;
                float y;
                uint z_order;
                uint color;
            };

            // Bitonic sort kernel
            kernel void gpu_sort(
                device Widget* widgets [[buffer(0)]],
                constant uint& count [[buffer(1)]],
                uint tid [[thread_index_in_threadgroup]]
            ) {
                threadgroup Widget tg_widgets[1024];

                // Load to threadgroup memory
                if (tid < count) {
                    tg_widgets[tid] = widgets[tid];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Bitonic sort
                for (uint k = 2; k <= 1024; k *= 2) {
                    for (uint j = k / 2; j > 0; j /= 2) {
                        uint ixj = tid ^ j;
                        if (ixj > tid && tid < count && ixj < count) {
                            bool ascending = ((tid & k) == 0);
                            Widget a = tg_widgets[tid];
                            Widget b = tg_widgets[ixj];

                            bool should_swap = ascending ? (a.z_order > b.z_order) : (a.z_order < b.z_order);
                            if (should_swap) {
                                tg_widgets[tid] = b;
                                tg_widgets[ixj] = a;
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }

                // Write back
                if (tid < count) {
                    widgets[tid] = tg_widgets[tid];
                }
            }

            // Parallel hit test kernel
            kernel void gpu_hittest(
                device const Widget* widgets [[buffer(0)]],
                constant uint& count [[buffer(1)]],
                constant float2& test_point [[buffer(2)]],
                constant float& widget_size [[buffer(3)]],
                device atomic_int* result [[buffer(4)]],
                uint tid [[thread_index_in_threadgroup]]
            ) {
                if (tid >= count) return;

                // Test from back to front (highest z first)
                uint idx = count - 1 - tid;
                Widget w = widgets[idx];

                float half_size = widget_size / 2.0;
                bool hit = test_point.x >= w.x - half_size && test_point.x <= w.x + half_size &&
                          test_point.y >= w.y - half_size && test_point.y <= w.y + half_size;

                if (hit) {
                    // Atomically store highest z-order hit (lowest idx = highest z after sort)
                    atomic_fetch_min_explicit(result, int(idx), memory_order_relaxed);
                }
            }
        "#;

        let options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .expect("Failed to compile shaders");

        let sort_fn = library.get_function("gpu_sort", None).unwrap();
        let hittest_fn = library.get_function("gpu_hittest", None).unwrap();

        let sort_pipeline = device.new_compute_pipeline_state_with_function(&sort_fn).unwrap();
        let hittest_pipeline = device.new_compute_pipeline_state_with_function(&hittest_fn).unwrap();

        let max_widgets = 2048;
        let widgets_buffer = device.new_buffer(
            (max_widgets * std::mem::size_of::<BenchWidget>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let result_buffer = device.new_buffer(
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            device,
            command_queue,
            sort_pipeline,
            hittest_pipeline,
            widgets_buffer,
            result_buffer,
            max_widgets,
        }
    }

    fn benchmark_sort(&self, widgets: &mut [BenchWidget]) -> f64 {
        let count = widgets.len();

        // Copy widgets to GPU buffer
        unsafe {
            let ptr = self.widgets_buffer.contents() as *mut BenchWidget;
            std::ptr::copy_nonoverlapping(widgets.as_ptr(), ptr, count);
        }

        let count_buffer = self.device.new_buffer_with_data(
            &(count as u32) as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let start = Instant::now();

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.sort_pipeline);
        encoder.set_buffer(0, Some(&self.widgets_buffer), 0);
        encoder.set_buffer(1, Some(&count_buffer), 0);

        let thread_count = count.min(1024) as u64;
        encoder.dispatch_threads(
            MTLSize::new(thread_count, 1, 1),
            MTLSize::new(thread_count, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start.elapsed();

        // Copy results back
        unsafe {
            let ptr = self.widgets_buffer.contents() as *const BenchWidget;
            std::ptr::copy_nonoverlapping(ptr, widgets.as_mut_ptr(), count);
        }

        elapsed.as_secs_f64() * 1_000_000.0 // Return microseconds
    }

    fn benchmark_hittest(&self, widgets: &[BenchWidget], test_x: f32, test_y: f32, widget_size: f32) -> (f64, Option<usize>) {
        let count = widgets.len();

        // Copy widgets to GPU buffer
        unsafe {
            let ptr = self.widgets_buffer.contents() as *mut BenchWidget;
            std::ptr::copy_nonoverlapping(widgets.as_ptr(), ptr, count);
        }

        // Reset result to max int
        unsafe {
            let ptr = self.result_buffer.contents() as *mut i32;
            *ptr = i32::MAX;
        }

        let count_buffer = self.device.new_buffer_with_data(
            &(count as u32) as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let test_point: [f32; 2] = [test_x, test_y];
        let test_buffer = self.device.new_buffer_with_data(
            test_point.as_ptr() as *const _,
            std::mem::size_of::<[f32; 2]>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let size_buffer = self.device.new_buffer_with_data(
            &widget_size as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let start = Instant::now();

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.hittest_pipeline);
        encoder.set_buffer(0, Some(&self.widgets_buffer), 0);
        encoder.set_buffer(1, Some(&count_buffer), 0);
        encoder.set_buffer(2, Some(&test_buffer), 0);
        encoder.set_buffer(3, Some(&size_buffer), 0);
        encoder.set_buffer(4, Some(&self.result_buffer), 0);

        let thread_count = count.min(1024) as u64;
        encoder.dispatch_threads(
            MTLSize::new(thread_count, 1, 1),
            MTLSize::new(thread_count, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start.elapsed();

        let result = unsafe {
            let ptr = self.result_buffer.contents() as *const i32;
            *ptr
        };

        let hit = if result == i32::MAX { None } else { Some(result as usize) };

        (elapsed.as_secs_f64() * 1_000_000.0, hit)
    }
}

// ============================================================================
// Statistics
// ============================================================================

fn calculate_stats(times: &[f64]) -> (f64, f64, f64, f64) {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, std, min, max)
}

// ============================================================================
// Widget Generation
// ============================================================================

fn generate_widgets(count: usize, seed: u64) -> Vec<BenchWidget> {
    let mut widgets = Vec::with_capacity(count);
    let mut state = seed;

    for i in 0..count {
        // Simple LCG random
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 33) as f32) / (u32::MAX as f32);

        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = ((state >> 33) as f32) / (u32::MAX as f32);

        widgets.push(BenchWidget {
            x,
            y,
            z_order: i as u32,
            color: 0xFFFFFFFF,
        });
    }

    // Shuffle z_order
    for i in (1..count).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        let tmp = widgets[i].z_order;
        widgets[i].z_order = widgets[j].z_order;
        widgets[j].z_order = tmp;
    }

    widgets
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

fn run_benchmarks(iterations: usize, csv_path: Option<&str>) -> Vec<BenchResult> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         GPU vs CPU AUTOMATED BENCHMARK SUITE                     ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let device = Device::system_default().expect("No Metal device");
    println!("║ GPU: {:58} ║", device.name());
    println!("║ Iterations per config: {:43} ║", iterations);
    println!("║ Warmup iterations: {:47} ║", WARMUP_ITERATIONS);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let gpu = GpuBenchmark::new();
    let mut results = Vec::new();

    // ========== SORT BENCHMARKS ==========
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│                      SORT BENCHMARK                              │");
    println!("│  GPU: Parallel Bitonic Sort (O(n log²n), 1024 threads)          │");
    println!("│  CPU: Quicksort (O(n log n), single-threaded)                   │");
    println!("├──────────┬───────────────────┬───────────────────┬──────────────┤");
    println!("│ Widgets  │ GPU (μs)          │ CPU (μs)          │ Speedup      │");
    println!("├──────────┼───────────────────┼───────────────────┼──────────────┤");

    for &count in &WIDGET_COUNTS {
        if count > 1024 {
            // Skip counts > 1024 for sort (threadgroup limit)
            continue;
        }

        let mut gpu_times = Vec::with_capacity(iterations);
        let mut cpu_times = Vec::with_capacity(iterations);

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let mut widgets = generate_widgets(count, 12345);
            let _ = gpu.benchmark_sort(&mut widgets);

            let mut widgets = generate_widgets(count, 12345);
            let start = Instant::now();
            cpu_quicksort(&mut widgets);
            let _ = start.elapsed();
        }

        // Actual benchmark
        for i in 0..iterations {
            let seed = (i as u64) * 7919 + 42;

            // GPU
            let mut widgets = generate_widgets(count, seed);
            let gpu_time = gpu.benchmark_sort(&mut widgets);
            gpu_times.push(gpu_time);

            // CPU
            let mut widgets = generate_widgets(count, seed);
            let start = Instant::now();
            cpu_quicksort(&mut widgets);
            let cpu_time = start.elapsed().as_secs_f64() * 1_000_000.0;
            cpu_times.push(cpu_time);
        }

        let (gpu_mean, gpu_std, gpu_min, gpu_max) = calculate_stats(&gpu_times);
        let (cpu_mean, cpu_std, cpu_min, cpu_max) = calculate_stats(&cpu_times);
        let speedup = cpu_mean / gpu_mean;

        let speedup_str = if speedup >= 1.0 {
            format!("{:>6.2}x GPU", speedup)
        } else {
            format!("{:>6.2}x CPU", 1.0 / speedup)
        };

        println!("│ {:>8} │ {:>7.1} ± {:>6.1} │ {:>7.1} ± {:>6.1} │ {:>12} │",
            count, gpu_mean, gpu_std, cpu_mean, cpu_std, speedup_str);

        results.push(BenchResult {
            widget_count: count,
            mode: "Sort".to_string(),
            gpu_mean_us: gpu_mean,
            gpu_std_us: gpu_std,
            gpu_min_us: gpu_min,
            gpu_max_us: gpu_max,
            cpu_mean_us: cpu_mean,
            cpu_std_us: cpu_std,
            cpu_min_us: cpu_min,
            cpu_max_us: cpu_max,
            speedup,
            iterations,
        });
    }

    println!("└──────────┴───────────────────┴───────────────────┴──────────────┘");
    println!();

    // ========== HIT TEST BENCHMARKS ==========
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│                    HIT TEST BENCHMARK                            │");
    println!("│  GPU: Parallel test all widgets (1024 threads, atomic min)      │");
    println!("│  CPU: Sequential back-to-front scan                             │");
    println!("├──────────┬───────────────────┬───────────────────┬──────────────┤");
    println!("│ Widgets  │ GPU (μs)          │ CPU (μs)          │ Speedup      │");
    println!("├──────────┼───────────────────┼───────────────────┼──────────────┤");

    let widget_size = 0.05f32;

    for &count in &WIDGET_COUNTS {
        let mut gpu_times = Vec::with_capacity(iterations);
        let mut cpu_times = Vec::with_capacity(iterations);

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let widgets = generate_widgets(count, 12345);
            let _ = gpu.benchmark_hittest(&widgets, 0.5, 0.5, widget_size);
            let _ = cpu_hit_test(&widgets, 0.5, 0.5, widget_size);
        }

        // Actual benchmark
        for i in 0..iterations {
            let seed = (i as u64) * 7919 + 42;
            let widgets = generate_widgets(count, seed);

            // Random test point
            let test_x = ((seed * 13) % 1000) as f32 / 1000.0;
            let test_y = ((seed * 17) % 1000) as f32 / 1000.0;

            // GPU
            let (gpu_time, _) = gpu.benchmark_hittest(&widgets, test_x, test_y, widget_size);
            gpu_times.push(gpu_time);

            // CPU
            let start = Instant::now();
            let _ = cpu_hit_test(&widgets, test_x, test_y, widget_size);
            let cpu_time = start.elapsed().as_secs_f64() * 1_000_000.0;
            cpu_times.push(cpu_time);
        }

        let (gpu_mean, gpu_std, gpu_min, gpu_max) = calculate_stats(&gpu_times);
        let (cpu_mean, cpu_std, cpu_min, cpu_max) = calculate_stats(&cpu_times);
        let speedup = cpu_mean / gpu_mean;

        let speedup_str = if speedup >= 1.0 {
            format!("{:>6.2}x GPU", speedup)
        } else {
            format!("{:>6.2}x CPU", 1.0 / speedup)
        };

        println!("│ {:>8} │ {:>7.1} ± {:>6.1} │ {:>7.1} ± {:>6.1} │ {:>12} │",
            count, gpu_mean, gpu_std, cpu_mean, cpu_std, speedup_str);

        results.push(BenchResult {
            widget_count: count,
            mode: "HitTest".to_string(),
            gpu_mean_us: gpu_mean,
            gpu_std_us: gpu_std,
            gpu_min_us: gpu_min,
            gpu_max_us: gpu_max,
            cpu_mean_us: cpu_mean,
            cpu_std_us: cpu_std,
            cpu_min_us: cpu_min,
            cpu_max_us: cpu_max,
            speedup,
            iterations,
        });
    }

    println!("└──────────┴───────────────────┴───────────────────┴──────────────┘");
    println!();

    // ========== SUMMARY ==========
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│                         SUMMARY                                  │");
    println!("├──────────────────────────────────────────────────────────────────┤");

    let sort_results: Vec<_> = results.iter().filter(|r| r.mode == "Sort").collect();
    let hittest_results: Vec<_> = results.iter().filter(|r| r.mode == "HitTest").collect();

    if let Some(best_sort) = sort_results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap()) {
        println!("│ Best Sort speedup: {:>6.2}x GPU faster at {} widgets {:>11} │",
            best_sort.speedup, best_sort.widget_count, "");
    }

    if let Some(best_hit) = hittest_results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap()) {
        println!("│ Best HitTest speedup: {:>6.2}x GPU faster at {} widgets {:>8} │",
            best_hit.speedup, best_hit.widget_count, "");
    }

    // Find crossover points
    let sort_crossover = sort_results.iter()
        .find(|r| r.speedup >= 1.0)
        .map(|r| r.widget_count);
    let hit_crossover = hittest_results.iter()
        .find(|r| r.speedup >= 1.0)
        .map(|r| r.widget_count);

    if let Some(c) = sort_crossover {
        println!("│ Sort crossover (GPU wins): {} widgets {:>28} │", c, "");
    }
    if let Some(c) = hit_crossover {
        println!("│ HitTest crossover (GPU wins): {} widgets {:>25} │", c, "");
    }

    println!("└──────────────────────────────────────────────────────────────────┘");

    // Write CSV if requested
    if let Some(path) = csv_path {
        write_csv(&results, path).expect("Failed to write CSV");
        println!("\nResults written to: {}", path);
    }

    results
}

fn write_csv(results: &[BenchResult], path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "widget_count,mode,gpu_mean_us,gpu_std_us,gpu_min_us,gpu_max_us,cpu_mean_us,cpu_std_us,cpu_min_us,cpu_max_us,speedup,iterations")?;

    for r in results {
        writeln!(file, "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{}",
            r.widget_count, r.mode,
            r.gpu_mean_us, r.gpu_std_us, r.gpu_min_us, r.gpu_max_us,
            r.cpu_mean_us, r.cpu_std_us, r.cpu_min_us, r.cpu_max_us,
            r.speedup, r.iterations
        )?;
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut iterations = DEFAULT_ITERATIONS;
    let mut csv_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" | "-i" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i].parse().unwrap_or(DEFAULT_ITERATIONS);
                }
            }
            "--csv" | "-c" => {
                i += 1;
                if i < args.len() {
                    csv_path = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                println!("GPU vs CPU Automated Benchmark");
                println!();
                println!("Usage: benchmark_auto [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -i, --iterations N   Number of iterations per config (default: {})", DEFAULT_ITERATIONS);
                println!("  -c, --csv FILE       Write results to CSV file");
                println!("  -h, --help           Show this help");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    run_benchmarks(iterations, csv_path.as_deref());
}
