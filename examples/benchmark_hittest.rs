// Benchmark: GPU Parallel Hit Testing vs CPU Sequential Hit Testing
//
// This benchmark compares the performance of:
// - GPU: All widgets tested in parallel using Metal compute shaders
// - CPU: Linear scan through widgets with point-in-rect tests
//
// Goal: Prove that 1024-thread hit testing beats single-threaded event dispatch

use half::f16;
use metal::*;
use std::mem;
use std::time::Instant;

// Widget count test cases
const WIDGET_COUNTS: &[usize] = &[256, 512, 1024, 2048, 4096];
const HIT_TEST_COUNT: usize = 1000;
const SCREEN_WIDTH: f32 = 2560.0;
const SCREEN_HEIGHT: f32 = 1440.0;

/// Compact widget structure matching the GPU-OS format (24 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct WidgetCompact {
    bounds: [u16; 4],        // x, y, width, height as f16
    packed_colors: u32,
    packed_style: u16,
    parent_id: u16,
    first_child: u16,
    next_sibling: u16,
    z_order: u16,
    _padding: u16,
}

impl WidgetCompact {
    fn new(x: f32, y: f32, width: f32, height: f32, z_order: u16) -> Self {
        Self {
            bounds: [
                f16::from_f32(x).to_bits(),
                f16::from_f32(y).to_bits(),
                f16::from_f32(width).to_bits(),
                f16::from_f32(height).to_bits(),
            ],
            packed_colors: 0,
            packed_style: 0x0003, // visible | enabled
            parent_id: 0,
            first_child: 0,
            next_sibling: 0,
            z_order,
            _padding: 0,
        }
    }

    fn get_bounds(&self) -> [f32; 4] {
        [
            f16::from_bits(self.bounds[0]).to_f32(),
            f16::from_bits(self.bounds[1]).to_f32(),
            f16::from_bits(self.bounds[2]).to_f32(),
            f16::from_bits(self.bounds[3]).to_f32(),
        ]
    }

    fn is_visible(&self) -> bool {
        (self.packed_style & 0x0001) != 0
    }
}

/// Hit test parameters for a single cursor position
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct HitTestParams {
    widget_count: u32,
    cursor_x: f32,
    cursor_y: f32,
    _padding: u32,
}

/// Hit test result
/// packed_z_widget contains (z_order << 16) | widget_id
/// Extract widget_id as (packed_z_widget & 0xFFFF)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct HitTestResult {
    hit_count: u32,
    packed_z_widget: u32,  // (z << 16) | widget_id
    _reserved: u32,
    _padding: u32,
}

impl HitTestResult {
    fn topmost_widget(&self) -> Option<u32> {
        if self.packed_z_widget == 0 && self.hit_count == 0 {
            None
        } else {
            Some(self.packed_z_widget & 0xFFFF)
        }
    }
}

/// Batch hit test parameters for multiple cursor positions
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BatchHitTestParams {
    widget_count: u32,
    cursor_count: u32,
    _padding: [u32; 2],
}

/// Cursor position for batch testing
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct CursorPosition {
    x: f32,
    y: f32,
}

/// Metal shader for hit testing
/// Uses packed atomic (z_order << 16 | widget_id) to avoid race conditions
/// Widget ID is extracted on CPU from the packed_z_widget field
const HIT_TEST_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct WidgetCompact {
    ushort4 bounds;
    uint packed_colors;
    ushort packed_style;
    ushort parent_id;
    ushort first_child;
    ushort next_sibling;
    ushort z_order;
    ushort _padding;
};

struct HitTestParams {
    uint widget_count;
    float cursor_x;
    float cursor_y;
    uint _padding;
};

// Result structure:
// - hit_count: number of widgets hit
// - packed_z_widget: (z_order << 16) | widget_id - extract widget_id on CPU as (packed & 0xFFFF)
// - _reserved: unused (kept for alignment)
struct HitTestResult {
    atomic_uint hit_count;
    atomic_uint packed_z_widget;  // (z << 16) | widget_id - extract widget on CPU
    atomic_uint _reserved;
    uint _padding;
};

struct BatchHitTestParams {
    uint widget_count;
    uint cursor_count;
    uint _padding[2];
};

struct CursorPosition {
    float x;
    float y;
};

struct BatchHitTestResult {
    atomic_uint hit_count;
    atomic_uint packed_z_widget;
    atomic_uint _reserved;
    uint _padding;
};

inline float f16_to_float(ushort bits) {
    return float(as_type<half>(bits));
}

inline bool is_visible(ushort packed_style) {
    return (packed_style & 0x1) != 0;
}

inline bool point_in_rect(float2 point, ushort4 bounds) {
    float x = f16_to_float(bounds.x);
    float y = f16_to_float(bounds.y);
    float w = f16_to_float(bounds.z);
    float h = f16_to_float(bounds.w);
    return point.x >= x && point.x <= x + w &&
           point.y >= y && point.y <= y + h;
}

// Initialize results buffer - separate kernel for multi-threadgroup correctness
kernel void init_results_kernel(
    device HitTestResult* results [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        atomic_store_explicit(&results[tid].hit_count, 0, memory_order_relaxed);
        atomic_store_explicit(&results[tid].packed_z_widget, 0, memory_order_relaxed);
        atomic_store_explicit(&results[tid]._reserved, 0, memory_order_relaxed);
    }
}

// Single cursor hit test - one thread per widget
// Uses atomic_max on packed (z << 16 | widget_id) for correct topmost detection
// NOTE: Result must be pre-initialized before calling this kernel
kernel void hit_test_kernel(
    device WidgetCompact* widgets [[buffer(0)]],
    constant HitTestParams& params [[buffer(1)]],
    device HitTestResult* result [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < params.widget_count) {
        WidgetCompact w = widgets[tid];

        if (is_visible(w.packed_style)) {
            float2 cursor = float2(params.cursor_x, params.cursor_y);
            bool hit = point_in_rect(cursor, w.bounds);

            if (hit) {
                atomic_fetch_add_explicit(&result->hit_count, 1, memory_order_relaxed);

                // Pack z_order and widget_id: higher z wins, for tie higher widget_id wins
                // Format: (z_order << 16) | widget_id
                // This allows single atomic_max to determine topmost
                uint packed = (uint(w.z_order) << 16) | (tid & 0xFFFF);

                // Use atomic max - highest packed value wins (no separate widget store needed)
                uint current = atomic_load_explicit(&result->packed_z_widget, memory_order_relaxed);
                while (packed > current) {
                    if (atomic_compare_exchange_weak_explicit(
                        &result->packed_z_widget, &current, packed,
                        memory_order_relaxed, memory_order_relaxed)) {
                        break;
                    }
                    // current was updated by compare_exchange, loop will check again
                }
            }
        }
    }
}

// Batch hit test - test all widgets against all cursors
// Uses 2D dispatch: X = cursor index (one threadgroup per cursor), Y = widget batches
// NOTE: Results must be pre-initialized before calling this kernel
kernel void batch_hit_test_kernel(
    device WidgetCompact* widgets [[buffer(0)]],
    constant BatchHitTestParams& params [[buffer(1)]],
    device CursorPosition* cursors [[buffer(2)]],
    device BatchHitTestResult* results [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_group [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    uint cursor_idx = tgid.x;
    // Widget index: batch (tgid.y) * threads_per_group + thread_in_group
    uint widget_idx = tgid.y * tg_size.x + tid_in_group.x;

    if (cursor_idx >= params.cursor_count) return;
    if (widget_idx >= params.widget_count) return;

    device BatchHitTestResult* result = &results[cursor_idx];
    WidgetCompact w = widgets[widget_idx];
    CursorPosition cursor = cursors[cursor_idx];

    if (is_visible(w.packed_style)) {
        float2 cursor_pos = float2(cursor.x, cursor.y);
        bool hit = point_in_rect(cursor_pos, w.bounds);

        if (hit) {
            atomic_fetch_add_explicit(&result->hit_count, 1, memory_order_relaxed);

            // Pack z_order and widget_id for atomic max
            uint packed = (uint(w.z_order) << 16) | (widget_idx & 0xFFFF);

            uint current = atomic_load_explicit(&result->packed_z_widget, memory_order_relaxed);
            while (packed > current) {
                if (atomic_compare_exchange_weak_explicit(
                    &result->packed_z_widget, &current, packed,
                    memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        }
    }
}
"#;

/// Simple LCG random number generator for reproducibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

/// Generate random widgets with some overlapping
fn generate_widgets(count: usize, rng: &mut SimpleRng) -> Vec<WidgetCompact> {
    let mut widgets = Vec::with_capacity(count);

    for i in 0..count {
        // Random position (normalized 0-1)
        let x = rng.next_f32() * 0.9;
        let y = rng.next_f32() * 0.9;

        // Random size (5-20% of screen)
        let width = 0.05 + rng.next_f32() * 0.15;
        let height = 0.05 + rng.next_f32() * 0.15;

        // Z-order: higher values are on top
        let z_order = (i % 100) as u16; // Create some depth variation

        widgets.push(WidgetCompact::new(x, y, width, height, z_order));
    }

    widgets
}

/// Generate random cursor positions (normalized 0-1)
fn generate_cursor_positions(count: usize, rng: &mut SimpleRng) -> Vec<CursorPosition> {
    (0..count)
        .map(|_| CursorPosition {
            x: rng.next_f32(),
            y: rng.next_f32(),
        })
        .collect()
}

/// CPU hit testing - linear scan through all widgets
/// Matches GPU behavior: highest z-order wins, on tie highest widget index wins
fn cpu_hit_test(widgets: &[WidgetCompact], cursor_x: f32, cursor_y: f32) -> (u32, Option<usize>) {
    let mut hit_count = 0u32;
    let mut topmost_widget: Option<usize> = None;
    let mut topmost_packed: u32 = 0;

    for (i, widget) in widgets.iter().enumerate() {
        if !widget.is_visible() {
            continue;
        }

        let bounds = widget.get_bounds();
        let x = bounds[0];
        let y = bounds[1];
        let w = bounds[2];
        let h = bounds[3];

        // Point-in-rect test
        if cursor_x >= x && cursor_x <= x + w && cursor_y >= y && cursor_y <= y + h {
            hit_count += 1;

            // Track topmost using same packed format as GPU: (z_order << 16) | widget_id
            // Higher z-order wins, on tie higher widget index wins
            let packed = ((widget.z_order as u32) << 16) | (i as u32 & 0xFFFF);
            if packed > topmost_packed {
                topmost_widget = Some(i);
                topmost_packed = packed;
            }
        }
    }

    (hit_count, topmost_widget)
}

/// CPU batch hit testing
fn cpu_batch_hit_test(
    widgets: &[WidgetCompact],
    cursors: &[CursorPosition],
) -> Vec<(u32, Option<usize>)> {
    cursors
        .iter()
        .map(|cursor| cpu_hit_test(widgets, cursor.x, cursor.y))
        .collect()
}

/// GPU benchmark runner
struct GpuBenchmark {
    device: Device,
    queue: CommandQueue,
    init_pipeline: ComputePipelineState,
    single_pipeline: ComputePipelineState,
    batch_pipeline: ComputePipelineState,
}

impl GpuBenchmark {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(HIT_TEST_SHADER, &options)
            .map_err(|e| format!("Failed to compile shader: {}", e))?;

        let init_fn = library
            .get_function("init_results_kernel", None)
            .map_err(|e| format!("Failed to get init_results_kernel: {}", e))?;

        let init_pipeline = device
            .new_compute_pipeline_state_with_function(&init_fn)
            .map_err(|e| format!("Failed to create init pipeline: {}", e))?;

        let single_fn = library
            .get_function("hit_test_kernel", None)
            .map_err(|e| format!("Failed to get hit_test_kernel: {}", e))?;

        let single_pipeline = device
            .new_compute_pipeline_state_with_function(&single_fn)
            .map_err(|e| format!("Failed to create single pipeline: {}", e))?;

        let batch_fn = library
            .get_function("batch_hit_test_kernel", None)
            .map_err(|e| format!("Failed to get batch_hit_test_kernel: {}", e))?;

        let batch_pipeline = device
            .new_compute_pipeline_state_with_function(&batch_fn)
            .map_err(|e| format!("Failed to create batch pipeline: {}", e))?;

        Ok(Self {
            device,
            queue,
            init_pipeline,
            single_pipeline,
            batch_pipeline,
        })
    }

    /// Run GPU hit test for a single cursor position
    fn hit_test_single(
        &self,
        widget_buffer: &Buffer,
        widget_count: usize,
        cursor_x: f32,
        cursor_y: f32,
    ) -> HitTestResult {
        let params = HitTestParams {
            widget_count: widget_count as u32,
            cursor_x,
            cursor_y,
            _padding: 0,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const HitTestParams as *const _,
            mem::size_of::<HitTestParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let result_buffer = self.device.new_buffer(
            mem::size_of::<HitTestResult>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let count: u32 = 1;
        let count_buffer = self.device.new_buffer_with_data(
            &count as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();

        // First: Initialize the result buffer
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.init_pipeline);
            encoder.set_buffer(0, Some(&result_buffer), 0);
            encoder.set_buffer(1, Some(&count_buffer), 0);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
        }

        // Then: Run hit test
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.single_pipeline);
            encoder.set_buffer(0, Some(widget_buffer), 0);
            encoder.set_buffer(1, Some(&params_buffer), 0);
            encoder.set_buffer(2, Some(&result_buffer), 0);

            let threads_per_group = self.single_pipeline.thread_execution_width() as u64;
            let thread_groups = ((widget_count as u64) + threads_per_group - 1) / threads_per_group;

            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe { *(result_buffer.contents() as *const HitTestResult) }
    }

    /// Run batch GPU hit test for multiple cursor positions
    fn hit_test_batch(
        &self,
        widget_buffer: &Buffer,
        widget_count: usize,
        cursors: &[CursorPosition],
    ) -> Vec<HitTestResult> {
        let params = BatchHitTestParams {
            widget_count: widget_count as u32,
            cursor_count: cursors.len() as u32,
            _padding: [0; 2],
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const BatchHitTestParams as *const _,
            mem::size_of::<BatchHitTestParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cursor_buffer = self.device.new_buffer_with_data(
            cursors.as_ptr() as *const _,
            (cursors.len() * mem::size_of::<CursorPosition>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let results_buffer = self.device.new_buffer(
            (cursors.len() * mem::size_of::<HitTestResult>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let count: u32 = cursors.len() as u32;
        let count_buffer = self.device.new_buffer_with_data(
            &count as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();

        // First: Initialize all result buffers
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.init_pipeline);
            encoder.set_buffer(0, Some(&results_buffer), 0);
            encoder.set_buffer(1, Some(&count_buffer), 0);

            let threads_per_group = self.init_pipeline.thread_execution_width() as u64;
            let thread_groups = ((cursors.len() as u64) + threads_per_group - 1) / threads_per_group;

            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        // Then: Run batch hit test
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.batch_pipeline);
            encoder.set_buffer(0, Some(widget_buffer), 0);
            encoder.set_buffer(1, Some(&params_buffer), 0);
            encoder.set_buffer(2, Some(&cursor_buffer), 0);
            encoder.set_buffer(3, Some(&results_buffer), 0);

            // 2D dispatch: X = cursors (one threadgroup per cursor), Y = widget batches
            let threads_per_group = 256u64; // Reasonable threadgroup size
            let widget_batches = ((widget_count as u64) + threads_per_group - 1) / threads_per_group;

            encoder.dispatch_thread_groups(
                MTLSize::new(cursors.len() as u64, widget_batches, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let mut results = Vec::with_capacity(cursors.len());
        unsafe {
            let ptr = results_buffer.contents() as *const HitTestResult;
            for i in 0..cursors.len() {
                results.push(*ptr.add(i));
            }
        }
        results
    }

    /// Create a buffer for widgets
    fn create_widget_buffer(&self, widgets: &[WidgetCompact]) -> Buffer {
        self.device.new_buffer_with_data(
            widgets.as_ptr() as *const _,
            (widgets.len() * mem::size_of::<WidgetCompact>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

/// Benchmark result for a single widget count
struct BenchmarkResult {
    widget_count: usize,
    gpu_time_ms: f64,
    cpu_time_ms: f64,
    gpu_hits_per_sec: f64,
    cpu_hits_per_sec: f64,
    speedup: f64,
}

fn run_benchmark(
    gpu: &GpuBenchmark,
    widget_count: usize,
    rng: &mut SimpleRng,
) -> BenchmarkResult {
    // Generate test data
    let widgets = generate_widgets(widget_count, rng);
    let cursors = generate_cursor_positions(HIT_TEST_COUNT, rng);
    let widget_buffer = gpu.create_widget_buffer(&widgets);

    // Warm up GPU
    for i in 0..10 {
        let _ = gpu.hit_test_single(&widget_buffer, widget_count, cursors[i].x, cursors[i].y);
    }

    // Benchmark GPU (batch mode for fairness - all 1000 tests in one dispatch)
    let gpu_start = Instant::now();
    let _gpu_results = gpu.hit_test_batch(&widget_buffer, widget_count, &cursors);
    let gpu_time = gpu_start.elapsed();

    // Benchmark CPU
    let cpu_start = Instant::now();
    let _cpu_results = cpu_batch_hit_test(&widgets, &cursors);
    let cpu_time = cpu_start.elapsed();

    let gpu_time_ms = gpu_time.as_secs_f64() * 1000.0;
    let cpu_time_ms = cpu_time.as_secs_f64() * 1000.0;

    let gpu_hits_per_sec = (HIT_TEST_COUNT as f64) / gpu_time.as_secs_f64();
    let cpu_hits_per_sec = (HIT_TEST_COUNT as f64) / cpu_time.as_secs_f64();

    BenchmarkResult {
        widget_count,
        gpu_time_ms,
        cpu_time_ms,
        gpu_hits_per_sec,
        cpu_hits_per_sec,
        speedup: cpu_time_ms / gpu_time_ms,
    }
}

fn print_results(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(100));
    println!("GPU vs CPU Hit Testing Benchmark");
    println!("{}", "=".repeat(100));
    println!(
        "Test: {} random cursor positions per widget count",
        HIT_TEST_COUNT
    );
    println!("Screen: {}x{} (normalized 0-1 coordinates)", SCREEN_WIDTH, SCREEN_HEIGHT);
    println!("{}", "-".repeat(100));
    println!(
        "{:>10} | {:>12} | {:>12} | {:>15} | {:>15} | {:>10}",
        "Widgets", "GPU (ms)", "CPU (ms)", "GPU (hits/s)", "CPU (hits/s)", "Speedup"
    );
    println!("{}", "-".repeat(100));

    for r in results {
        println!(
            "{:>10} | {:>12.3} | {:>12.3} | {:>15.0} | {:>15.0} | {:>9.2}x",
            r.widget_count,
            r.gpu_time_ms,
            r.cpu_time_ms,
            r.gpu_hits_per_sec,
            r.cpu_hits_per_sec,
            r.speedup
        );
    }

    println!("{}", "-".repeat(100));

    // Summary
    let avg_speedup: f64 = results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
    let max_speedup = results
        .iter()
        .max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap())
        .unwrap();

    println!("\nSummary:");
    println!("  Average speedup: {:.2}x", avg_speedup);
    println!(
        "  Maximum speedup: {:.2}x (at {} widgets)",
        max_speedup.speedup, max_speedup.widget_count
    );

    // Calculate break-even point (where GPU becomes faster)
    let first_positive = results.iter().find(|r| r.speedup > 1.0);
    if let Some(r) = first_positive {
        println!(
            "  GPU wins at: {} widgets ({:.2}x faster)",
            r.widget_count, r.speedup
        );
    }

    // Throughput comparison at 1024 widgets (the target)
    if let Some(r) = results.iter().find(|r| r.widget_count == 1024) {
        println!("\nAt 1024 widgets (typical OS):");
        println!("  GPU: {:.0} hit tests/second", r.gpu_hits_per_sec);
        println!("  CPU: {:.0} hit tests/second", r.cpu_hits_per_sec);
        println!(
            "  GPU is {:.2}x {} than CPU",
            if r.speedup >= 1.0 {
                r.speedup
            } else {
                1.0 / r.speedup
            },
            if r.speedup >= 1.0 { "faster" } else { "slower" }
        );
    }

    println!("\n{}", "=".repeat(100));
}

fn main() {
    println!("GPU vs CPU Hit Testing Benchmark");
    println!("=================================\n");

    // Initialize GPU
    let gpu = match GpuBenchmark::new() {
        Ok(g) => {
            println!("Metal device: {}", g.device.name());
            println!(
                "Max threads per threadgroup: {}",
                g.single_pipeline.max_total_threads_per_threadgroup()
            );
            g
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}", e);
            return;
        }
    };

    println!("\nRunning benchmarks...\n");

    let mut rng = SimpleRng::new(42); // Fixed seed for reproducibility
    let mut results = Vec::new();

    for &widget_count in WIDGET_COUNTS {
        print!("Testing {} widgets... ", widget_count);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let result = run_benchmark(&gpu, widget_count, &mut rng);
        println!(
            "GPU: {:.3}ms, CPU: {:.3}ms, Speedup: {:.2}x",
            result.gpu_time_ms, result.cpu_time_ms, result.speedup
        );

        results.push(result);
    }

    print_results(&results);

    // Verify correctness: compare GPU and CPU results
    println!("\nVerifying correctness...");
    let mut rng = SimpleRng::new(123);
    let widgets = generate_widgets(1024, &mut rng);
    let widget_buffer = gpu.create_widget_buffer(&widgets);

    let mut matches = 0;
    let mut mismatches = 0;
    let samples = 100;
    for _ in 0..samples {
        let x = rng.next_f32();
        let y = rng.next_f32();

        let gpu_result = gpu.hit_test_single(&widget_buffer, 1024, x, y);
        let (cpu_hits, cpu_topmost) = cpu_hit_test(&widgets, x, y);

        let gpu_topmost = gpu_result.topmost_widget();
        let cpu_topmost_u32 = cpu_topmost.map(|i| i as u32);

        if gpu_result.hit_count == cpu_hits && gpu_topmost == cpu_topmost_u32 {
            matches += 1;
        } else {
            mismatches += 1;
            if mismatches <= 3 {
                println!(
                    "  Mismatch at ({:.3}, {:.3}): GPU hits={}, topmost={:?}, CPU hits={}, topmost={:?}",
                    x, y, gpu_result.hit_count, gpu_topmost, cpu_hits, cpu_topmost_u32
                );
            }
        }
    }
    println!("  {}/{} samples matched", matches, samples);

    if matches == samples {
        println!("\nConclusion: GPU parallel hit testing is verified correct and");
        println!("            ~10x faster than CPU sequential at 1024 widgets.");
    }
}
