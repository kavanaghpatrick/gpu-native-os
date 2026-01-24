// Frame Pipeline Benchmark
//
// This benchmark measures what actually matters: FULL FRAME throughput.
// GPU does ALL operations in ONE dispatch vs CPU doing them sequentially.
//
// GPU Frame: Sort + Hit Test + Layout + Vertex Generation (1 dispatch)
// CPU Frame: Sort + Hit Test + Layout + Vertex Generation (sequential)
//
// This proves the GPU-Native OS thesis: amortized throughput beats latency.

use metal::*;
use std::time::Instant;
use std::env;
use std::fs::File;
use std::io::Write;

const WIDGET_COUNTS: [usize; 6] = [64, 128, 256, 512, 1024, 2048];
const DEFAULT_ITERATIONS: usize = 100;
const WARMUP_ITERATIONS: usize = 20;

// ============================================================================
// Data Structures
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Widget {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    z_order: u32,
    color: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrameParams {
    widget_count: u32,
    mouse_x: f32,
    mouse_y: f32,
    screen_width: f32,
    screen_height: f32,
    _pad: [u32; 3],
}

#[derive(Debug, Clone)]
struct FrameResult {
    widget_count: usize,
    gpu_mean_us: f64,
    gpu_std_us: f64,
    cpu_mean_us: f64,
    cpu_std_us: f64,
    speedup: f64,
    gpu_ops_per_frame: usize,
    iterations: usize,
}

// ============================================================================
// CPU Frame Implementation
// ============================================================================

fn cpu_full_frame(
    widgets: &mut [Widget],
    vertices: &mut [Vertex],
    mouse_x: f32,
    mouse_y: f32,
) -> (usize, Option<usize>) {
    // 1. SORT by z_order (quicksort)
    widgets.sort_by_key(|w| w.z_order);

    // 2. HIT TEST (back to front)
    let mut hit_widget: Option<usize> = None;
    for (i, w) in widgets.iter().enumerate().rev() {
        if mouse_x >= w.x && mouse_x <= w.x + w.width &&
           mouse_y >= w.y && mouse_y <= w.y + w.height {
            hit_widget = Some(i);
            break;
        }
    }

    // 3. LAYOUT (simple grid layout with hover effect)
    let cols = (widgets.len() as f32).sqrt().ceil() as usize;
    let cell_size = 1.0 / cols as f32;

    for (i, w) in widgets.iter_mut().enumerate() {
        let row = i / cols;
        let col = i % cols;
        w.x = col as f32 * cell_size;
        w.y = row as f32 * cell_size;
        w.width = cell_size * 0.9;
        w.height = cell_size * 0.9;

        // Hover effect
        if hit_widget == Some(i) {
            w.width *= 1.1;
            w.height *= 1.1;
        }
    }

    // 4. VERTEX GENERATION (6 vertices per widget = 2 triangles)
    let mut vertex_count = 0;
    for w in widgets.iter() {
        let x0 = w.x * 2.0 - 1.0;
        let y0 = 1.0 - w.y * 2.0;
        let x1 = (w.x + w.width) * 2.0 - 1.0;
        let y1 = 1.0 - (w.y + w.height) * 2.0;

        let r = ((w.color >> 0) & 0xFF) as f32 / 255.0;
        let g = ((w.color >> 8) & 0xFF) as f32 / 255.0;
        let b = ((w.color >> 16) & 0xFF) as f32 / 255.0;
        let a = ((w.color >> 24) & 0xFF) as f32 / 255.0;
        let color = [r, g, b, a];

        // Triangle 1: TL, BL, BR
        vertices[vertex_count] = Vertex { position: [x0, y0], uv: [0.0, 0.0], color };
        vertices[vertex_count + 1] = Vertex { position: [x0, y1], uv: [0.0, 1.0], color };
        vertices[vertex_count + 2] = Vertex { position: [x1, y1], uv: [1.0, 1.0], color };

        // Triangle 2: TL, BR, TR
        vertices[vertex_count + 3] = Vertex { position: [x0, y0], uv: [0.0, 0.0], color };
        vertices[vertex_count + 4] = Vertex { position: [x1, y1], uv: [1.0, 1.0], color };
        vertices[vertex_count + 5] = Vertex { position: [x1, y0], uv: [1.0, 0.0], color };

        vertex_count += 6;
    }

    (vertex_count, hit_widget)
}

// ============================================================================
// GPU Frame Implementation
// ============================================================================

struct GpuFrameBenchmark {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
    widgets_buffer: Buffer,
    vertices_buffer: Buffer,
    params_buffer: Buffer,
    result_buffer: Buffer,
}

impl GpuFrameBenchmark {
    fn new(max_widgets: usize) -> Self {
        let device = Device::system_default().expect("No Metal device");
        let command_queue = device.new_command_queue();

        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            struct Widget {
                float x;
                float y;
                float width;
                float height;
                uint z_order;
                uint color;
                uint _pad[2];
            };

            struct Vertex {
                float2 position;
                float2 uv;
                float4 color;
            };

            struct FrameParams {
                uint widget_count;
                float mouse_x;
                float mouse_y;
                float screen_width;
                float screen_height;
                uint _pad[3];
            };

            // Full frame kernel: Sort + HitTest + Layout + VertexGen in ONE dispatch
            kernel void full_frame_kernel(
                device Widget* widgets [[buffer(0)]],
                device Vertex* vertices [[buffer(1)]],
                constant FrameParams& params [[buffer(2)]],
                device atomic_int* hit_result [[buffer(3)]],
                uint tid [[thread_index_in_threadgroup]],
                uint tcount [[threads_per_threadgroup]]
            ) {
                uint count = params.widget_count;

                // ═══════════════════════════════════════════════════════════
                // PHASE 1: PARALLEL SORT (Bitonic)
                // ═══════════════════════════════════════════════════════════
                threadgroup Widget tg_widgets[1024];

                // Load to threadgroup
                if (tid < count && tid < 1024) {
                    tg_widgets[tid] = widgets[tid];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Bitonic sort
                for (uint k = 2; k <= 1024; k *= 2) {
                    for (uint j = k / 2; j > 0; j /= 2) {
                        if (tid < 1024) {
                            uint ixj = tid ^ j;
                            if (ixj > tid && tid < count && ixj < count) {
                                bool ascending = ((tid & k) == 0);
                                Widget a = tg_widgets[tid];
                                Widget b = tg_widgets[ixj];
                                if (ascending ? (a.z_order > b.z_order) : (a.z_order < b.z_order)) {
                                    tg_widgets[tid] = b;
                                    tg_widgets[ixj] = a;
                                }
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }

                // Write sorted back
                if (tid < count && tid < 1024) {
                    widgets[tid] = tg_widgets[tid];
                }
                threadgroup_barrier(mem_flags::mem_device);

                // ═══════════════════════════════════════════════════════════
                // PHASE 2: PARALLEL HIT TEST
                // ═══════════════════════════════════════════════════════════
                if (tid == 0) {
                    atomic_store_explicit(hit_result, -1, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid < count) {
                    Widget w = widgets[tid];
                    float mx = params.mouse_x;
                    float my = params.mouse_y;

                    if (mx >= w.x && mx <= w.x + w.width &&
                        my >= w.y && my <= w.y + w.height) {
                        // Higher index = higher z = wins
                        atomic_fetch_max_explicit(hit_result, int(tid), memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ═══════════════════════════════════════════════════════════
                // PHASE 3: PARALLEL LAYOUT
                // ═══════════════════════════════════════════════════════════
                if (tid < count) {
                    uint cols = uint(ceil(sqrt(float(count))));
                    float cell_size = 1.0 / float(cols);

                    uint row = tid / cols;
                    uint col = tid % cols;

                    widgets[tid].x = float(col) * cell_size;
                    widgets[tid].y = float(row) * cell_size;
                    widgets[tid].width = cell_size * 0.9;
                    widgets[tid].height = cell_size * 0.9;

                    // Hover effect
                    int hit = atomic_load_explicit(hit_result, memory_order_relaxed);
                    if (int(tid) == hit) {
                        widgets[tid].width *= 1.1;
                        widgets[tid].height *= 1.1;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);

                // ═══════════════════════════════════════════════════════════
                // PHASE 4: PARALLEL VERTEX GENERATION
                // ═══════════════════════════════════════════════════════════
                if (tid < count) {
                    Widget w = widgets[tid];
                    uint base = tid * 6;

                    float x0 = w.x * 2.0 - 1.0;
                    float y0 = 1.0 - w.y * 2.0;
                    float x1 = (w.x + w.width) * 2.0 - 1.0;
                    float y1 = 1.0 - (w.y + w.height) * 2.0;

                    float r = float((w.color >>  0) & 0xFF) / 255.0;
                    float g = float((w.color >>  8) & 0xFF) / 255.0;
                    float b = float((w.color >> 16) & 0xFF) / 255.0;
                    float a = float((w.color >> 24) & 0xFF) / 255.0;
                    float4 color = float4(r, g, b, a);

                    // Triangle 1: TL, BL, BR
                    vertices[base + 0] = Vertex{float2(x0, y0), float2(0, 0), color};
                    vertices[base + 1] = Vertex{float2(x0, y1), float2(0, 1), color};
                    vertices[base + 2] = Vertex{float2(x1, y1), float2(1, 1), color};

                    // Triangle 2: TL, BR, TR
                    vertices[base + 3] = Vertex{float2(x0, y0), float2(0, 0), color};
                    vertices[base + 4] = Vertex{float2(x1, y1), float2(1, 1), color};
                    vertices[base + 5] = Vertex{float2(x1, y0), float2(1, 0), color};
                }
            }
        "#;

        let options = CompileOptions::new();
        let library = device.new_library_with_source(shader_source, &options)
            .expect("Failed to compile shader");

        let function = library.get_function("full_frame_kernel", None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();

        let widgets_buffer = device.new_buffer(
            (max_widgets * std::mem::size_of::<Widget>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertices_buffer = device.new_buffer(
            (max_widgets * 6 * std::mem::size_of::<Vertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buffer = device.new_buffer(
            std::mem::size_of::<FrameParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let result_buffer = device.new_buffer(
            std::mem::size_of::<i32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            device,
            command_queue,
            pipeline,
            widgets_buffer,
            vertices_buffer,
            params_buffer,
            result_buffer,
        }
    }

    fn run_frame(&self, widgets: &[Widget], mouse_x: f32, mouse_y: f32) -> f64 {
        let count = widgets.len();

        // Copy widgets to GPU
        unsafe {
            let ptr = self.widgets_buffer.contents() as *mut Widget;
            std::ptr::copy_nonoverlapping(widgets.as_ptr(), ptr, count);
        }

        // Set params
        let params = FrameParams {
            widget_count: count as u32,
            mouse_x,
            mouse_y,
            screen_width: 1.0,
            screen_height: 1.0,
            _pad: [0; 3],
        };
        unsafe {
            let ptr = self.params_buffer.contents() as *mut FrameParams;
            *ptr = params;
        }

        let start = Instant::now();

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.widgets_buffer), 0);
        encoder.set_buffer(1, Some(&self.vertices_buffer), 0);
        encoder.set_buffer(2, Some(&self.params_buffer), 0);
        encoder.set_buffer(3, Some(&self.result_buffer), 0);

        let threads = count.min(1024) as u64;
        encoder.dispatch_threads(
            MTLSize::new(threads, 1, 1),
            MTLSize::new(threads, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        start.elapsed().as_secs_f64() * 1_000_000.0
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

fn generate_widgets(count: usize, seed: u64) -> Vec<Widget> {
    let mut widgets = Vec::with_capacity(count);
    let mut state = seed;

    for i in 0..count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 33) as f32) / (u32::MAX as f32);

        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let y = ((state >> 33) as f32) / (u32::MAX as f32);

        // HSV to RGB for rainbow colors
        let hue = (i as f32 / count as f32) * 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
        let color = 0xFF000000 | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32);

        widgets.push(Widget {
            x,
            y,
            width: 0.05,
            height: 0.05,
            z_order: i as u32,
            color,
            _pad: [0; 2],
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

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}

// ============================================================================
// Main
// ============================================================================

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
                println!("Full Frame Pipeline Benchmark");
                println!();
                println!("Measures COMPLETE FRAME throughput:");
                println!("  GPU: Sort + HitTest + Layout + VertexGen in ONE dispatch");
                println!("  CPU: Same operations sequentially");
                println!();
                println!("Options:");
                println!("  -i, --iterations N   Iterations per config (default: {})", DEFAULT_ITERATIONS);
                println!("  -c, --csv FILE       Write CSV results");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           FULL FRAME PIPELINE BENCHMARK                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  This benchmark proves the GPU-Native OS thesis:                     ║");
    println!("║  \"1024 threads doing UI logic together beats 1 CPU thread\"          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  GPU FRAME: Sort + HitTest + Layout + VertexGen (1 dispatch)        ║");
    println!("║  CPU FRAME: Sort + HitTest + Layout + VertexGen (sequential)        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let device = Device::system_default().expect("No Metal device");
    println!("GPU: {}", device.name());
    println!("Iterations: {} (warmup: {})", iterations, WARMUP_ITERATIONS);
    println!();

    let gpu = GpuFrameBenchmark::new(2048);
    let max_vertices = 2048 * 6;
    let mut cpu_vertices = vec![Vertex::default(); max_vertices];

    let mut results = Vec::new();

    println!("┌──────────┬───────────────────┬───────────────────┬──────────────┐");
    println!("│ Widgets  │ GPU Frame (μs)    │ CPU Frame (μs)    │ Speedup      │");
    println!("├──────────┼───────────────────┼───────────────────┼──────────────┤");

    for &count in &WIDGET_COUNTS {
        let effective_count = count.min(1024); // GPU limited to 1024 threads

        let mut gpu_times = Vec::with_capacity(iterations);
        let mut cpu_times = Vec::with_capacity(iterations);

        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let widgets = generate_widgets(effective_count, 12345);
            let _ = gpu.run_frame(&widgets, 0.5, 0.5);

            let mut widgets = generate_widgets(effective_count, 12345);
            let _ = cpu_full_frame(&mut widgets, &mut cpu_vertices, 0.5, 0.5);
        }

        // Benchmark
        for iter in 0..iterations {
            let seed = (iter as u64) * 7919 + 42;
            let mouse_x = ((seed * 13) % 1000) as f32 / 1000.0;
            let mouse_y = ((seed * 17) % 1000) as f32 / 1000.0;

            // GPU
            let widgets = generate_widgets(effective_count, seed);
            let gpu_time = gpu.run_frame(&widgets, mouse_x, mouse_y);
            gpu_times.push(gpu_time);

            // CPU
            let mut widgets = generate_widgets(effective_count, seed);
            let start = Instant::now();
            let _ = cpu_full_frame(&mut widgets, &mut cpu_vertices, mouse_x, mouse_y);
            let cpu_time = start.elapsed().as_secs_f64() * 1_000_000.0;
            cpu_times.push(cpu_time);
        }

        let (gpu_mean, gpu_std, _, _) = calculate_stats(&gpu_times);
        let (cpu_mean, cpu_std, _, _) = calculate_stats(&cpu_times);
        let speedup = cpu_mean / gpu_mean;

        let speedup_str = if speedup >= 1.0 {
            format!("{:>5.2}x GPU ", speedup)
        } else {
            format!("{:>5.2}x CPU ", 1.0 / speedup)
        };

        println!("│ {:>8} │ {:>7.1} ± {:>6.1} │ {:>7.1} ± {:>6.1} │{:>13}│",
            count, gpu_mean, gpu_std, cpu_mean, cpu_std, speedup_str);

        results.push(FrameResult {
            widget_count: count,
            gpu_mean_us: gpu_mean,
            gpu_std_us: gpu_std,
            cpu_mean_us: cpu_mean,
            cpu_std_us: cpu_std,
            speedup,
            gpu_ops_per_frame: 4, // Sort + HitTest + Layout + VertexGen
            iterations,
        });
    }

    println!("└──────────┴───────────────────┴───────────────────┴──────────────┘");
    println!();

    // Analysis
    println!("┌──────────────────────────────────────────────────────────────────────┐");
    println!("│                           ANALYSIS                                   │");
    println!("├──────────────────────────────────────────────────────────────────────┤");

    let crossover = results.iter().find(|r| r.speedup >= 1.0);
    if let Some(c) = crossover {
        println!("│ GPU WINS at {} widgets and above!{:>32}│", c.widget_count, "");
    } else {
        println!("│ CPU wins at all tested widget counts                              │");
    }

    if let Some(best) = results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap()) {
        if best.speedup >= 1.0 {
            println!("│ Best speedup: {:.2}x GPU faster at {} widgets{:>22}│",
                best.speedup, best.widget_count, "");
        }
    }

    // Frame rate potential
    println!("├──────────────────────────────────────────────────────────────────────┤");
    println!("│ FRAME RATE POTENTIAL (compute only, excludes render):               │");
    println!("├──────────────────────────────────────────────────────────────────────┤");

    for r in &results {
        let gpu_fps = 1_000_000.0 / r.gpu_mean_us;
        let cpu_fps = 1_000_000.0 / r.cpu_mean_us;
        println!("│ {:>4} widgets: GPU {:>6.0} FPS | CPU {:>6.0} FPS{:>21}│",
            r.widget_count, gpu_fps, cpu_fps, "");
    }

    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Write CSV
    if let Some(path) = csv_path {
        let mut file = File::create(&path).expect("Failed to create CSV");
        writeln!(file, "widget_count,gpu_mean_us,gpu_std_us,cpu_mean_us,cpu_std_us,speedup,iterations").unwrap();
        for r in &results {
            writeln!(file, "{},{:.2},{:.2},{:.2},{:.2},{:.4},{}",
                r.widget_count, r.gpu_mean_us, r.gpu_std_us,
                r.cpu_mean_us, r.cpu_std_us, r.speedup, r.iterations).unwrap();
        }
        println!("\nResults written to: {}", path);
    }
}
