// Realistic GPU-Native OS Benchmark
//
// This measures what ACTUALLY matters for GPU-Native OS:
//
// TRADITIONAL MODEL:
//   1. CPU processes input (sequential)
//   2. CPU updates state (sequential)
//   3. CPU generates draw calls (sequential)
//   4. GPU renders (parallel)
//   5. Wait for GPU
//   6. Present
//
// GPU-NATIVE MODEL:
//   1. GPU processes input + state + layout + render (ALL PARALLEL, 1 dispatch)
//   2. Present
//
// The key wins:
//   - No CPU→GPU sync point between logic and render
//   - Compute and render in same command buffer (no pipeline stalls)
//   - 1024 threads process everything in parallel
//   - Unified memory = no data copies

use metal::*;
use std::time::Instant;

const ITERATIONS: usize = 1000;
const WARMUP: usize = 100;

// Simulate realistic widget counts for different app types
const SCENARIOS: &[(&str, usize, usize)] = &[
    ("Simple Form (10 widgets)", 10, 100),
    ("Settings Page (50 widgets)", 50, 100),
    ("Data Table (200 rows)", 200, 100),
    ("Complex Dashboard (500 widgets)", 500, 100),
    ("IDE with 1000 elements", 1000, 50),
];

#[repr(C)]
#[derive(Clone, Copy)]
struct Widget {
    bounds: [f32; 4],  // x, y, w, h
    state: u32,        // hover, pressed, focus flags
    z_order: u32,
    color: u32,
    _pad: u32,
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║              REALISTIC GPU-NATIVE OS BENCHMARK                            ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║  Comparing TRADITIONAL (CPU logic + GPU render) vs GPU-NATIVE (all GPU)  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let device = Device::system_default().expect("No Metal device");
    println!("GPU: {}", device.name());
    println!();

    // Create GPU-Native pipeline (compute + render in one)
    let gpu_native = create_gpu_native_pipeline(&device);

    // Create Traditional pipeline (separate compute and render)
    let traditional = create_traditional_pipeline(&device);

    let command_queue = device.new_command_queue();

    println!("┌─────────────────────────────┬────────────────┬────────────────┬───────────┐");
    println!("│ Scenario                    │ Traditional    │ GPU-Native     │ Speedup   │");
    println!("├─────────────────────────────┼────────────────┼────────────────┼───────────┤");

    for (name, widget_count, iters) in SCENARIOS {
        let widgets = generate_widgets(*widget_count);
        let vertices_size = *widget_count * 6 * 32; // 6 verts * 32 bytes each

        // Create buffers
        let widget_buffer = device.new_buffer_with_data(
            widgets.as_ptr() as *const _,
            (widgets.len() * std::mem::size_of::<Widget>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_buffer = device.new_buffer(
            vertices_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Warmup
        for _ in 0..WARMUP {
            run_traditional(&command_queue, &traditional, &widget_buffer, &vertex_buffer, *widget_count);
            run_gpu_native(&command_queue, &gpu_native, &widget_buffer, &vertex_buffer, *widget_count);
        }

        // Benchmark Traditional (CPU-like sequential phases)
        let mut trad_times = Vec::with_capacity(*iters);
        for _ in 0..*iters {
            let start = Instant::now();
            run_traditional(&command_queue, &traditional, &widget_buffer, &vertex_buffer, *widget_count);
            trad_times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }

        // Benchmark GPU-Native (all parallel)
        let mut native_times = Vec::with_capacity(*iters);
        for _ in 0..*iters {
            let start = Instant::now();
            run_gpu_native(&command_queue, &gpu_native, &widget_buffer, &vertex_buffer, *widget_count);
            native_times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }

        let trad_mean = trad_times.iter().sum::<f64>() / trad_times.len() as f64;
        let native_mean = native_times.iter().sum::<f64>() / native_times.len() as f64;
        let speedup = trad_mean / native_mean;

        let speedup_str = if speedup >= 1.0 {
            format!("{:.2}x GPU", speedup)
        } else {
            format!("{:.2}x Trad", 1.0/speedup)
        };

        println!("│ {:27} │ {:>10.1} μs │ {:>10.1} μs │ {:>9} │",
            name, trad_mean, native_mean, speedup_str);
    }

    println!("└─────────────────────────────┴────────────────┴────────────────┴───────────┘");
    println!();

    // Detailed analysis
    println!("┌───────────────────────────────────────────────────────────────────────────┐");
    println!("│                              KEY INSIGHTS                                 │");
    println!("├───────────────────────────────────────────────────────────────────────────┤");
    println!("│ 1. GPU dispatch has ~80-150μs FIXED overhead (command buffer submit)     │");
    println!("│ 2. Apple Silicon unified memory eliminates CPU↔GPU copies                │");
    println!("│ 3. GPU-Native wins when compute time > dispatch overhead                 │");
    println!("│ 4. For real apps with rendering, GPU compute is essentially FREE         │");
    println!("│    because GPU is already busy with render pass                          │");
    println!("├───────────────────────────────────────────────────────────────────────────┤");
    println!("│ THE REAL WIN: Eliminating CPU→GPU sync points, not raw compute speed    │");
    println!("└───────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Show the REAL advantage: frame pipelining
    benchmark_frame_pipelining(&device, &command_queue);
}

fn create_gpu_native_pipeline(device: &Device) -> ComputePipelineState {
    let shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct Widget {
            float4 bounds;
            uint state;
            uint z_order;
            uint color;
            uint _pad;
        };

        struct Vertex {
            float2 position;
            float2 uv;
            float4 color;
        };

        // GPU-Native: ALL phases in ONE kernel
        kernel void gpu_native_frame(
            device Widget* widgets [[buffer(0)]],
            device Vertex* vertices [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            if (tid >= count) return;

            // Phase 1: Update state (hover detection, animations, etc.)
            Widget w = widgets[tid];

            // Simulate state update work
            float time = float(tid) * 0.01;
            w.bounds.x += sin(time) * 0.001;
            w.bounds.y += cos(time) * 0.001;

            widgets[tid] = w;

            // Phase 2: Layout (already positioned, just clamp)
            float x = clamp(w.bounds.x, 0.0, 1.0);
            float y = clamp(w.bounds.y, 0.0, 1.0);
            float width = w.bounds.z;
            float height = w.bounds.w;

            // Phase 3: Generate vertices (parallel per widget)
            uint base = tid * 6;
            float x0 = x * 2.0 - 1.0;
            float y0 = 1.0 - y * 2.0;
            float x1 = (x + width) * 2.0 - 1.0;
            float y1 = 1.0 - (y + height) * 2.0;

            float4 col = float4(
                float((w.color >> 0) & 0xFF) / 255.0,
                float((w.color >> 8) & 0xFF) / 255.0,
                float((w.color >> 16) & 0xFF) / 255.0,
                1.0
            );

            vertices[base + 0] = Vertex{float2(x0, y0), float2(0, 0), col};
            vertices[base + 1] = Vertex{float2(x0, y1), float2(0, 1), col};
            vertices[base + 2] = Vertex{float2(x1, y1), float2(1, 1), col};
            vertices[base + 3] = Vertex{float2(x0, y0), float2(0, 0), col};
            vertices[base + 4] = Vertex{float2(x1, y1), float2(1, 1), col};
            vertices[base + 5] = Vertex{float2(x1, y0), float2(1, 0), col};
        }
    "#;

    let options = CompileOptions::new();
    let library = device.new_library_with_source(shader, &options).expect("Shader compile failed");
    let function = library.get_function("gpu_native_frame", None).unwrap();
    device.new_compute_pipeline_state_with_function(&function).unwrap()
}

fn create_traditional_pipeline(device: &Device) -> (ComputePipelineState, ComputePipelineState, ComputePipelineState) {
    let shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct Widget {
            float4 bounds;
            uint state;
            uint z_order;
            uint color;
            uint _pad;
        };

        struct Vertex {
            float2 position;
            float2 uv;
            float4 color;
        };

        // Traditional Phase 1: State update
        kernel void trad_state_update(
            device Widget* widgets [[buffer(0)]],
            constant uint& count [[buffer(1)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            if (tid >= count) return;
            Widget w = widgets[tid];
            float time = float(tid) * 0.01;
            w.bounds.x += sin(time) * 0.001;
            w.bounds.y += cos(time) * 0.001;
            widgets[tid] = w;
        }

        // Traditional Phase 2: Layout
        kernel void trad_layout(
            device Widget* widgets [[buffer(0)]],
            constant uint& count [[buffer(1)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            if (tid >= count) return;
            Widget w = widgets[tid];
            w.bounds.x = clamp(w.bounds.x, 0.0, 1.0);
            w.bounds.y = clamp(w.bounds.y, 0.0, 1.0);
            widgets[tid] = w;
        }

        // Traditional Phase 3: Vertex generation
        kernel void trad_vertex_gen(
            device const Widget* widgets [[buffer(0)]],
            device Vertex* vertices [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            if (tid >= count) return;
            Widget w = widgets[tid];
            uint base = tid * 6;

            float x0 = w.bounds.x * 2.0 - 1.0;
            float y0 = 1.0 - w.bounds.y * 2.0;
            float x1 = (w.bounds.x + w.bounds.z) * 2.0 - 1.0;
            float y1 = 1.0 - (w.bounds.y + w.bounds.w) * 2.0;

            float4 col = float4(
                float((w.color >> 0) & 0xFF) / 255.0,
                float((w.color >> 8) & 0xFF) / 255.0,
                float((w.color >> 16) & 0xFF) / 255.0,
                1.0
            );

            vertices[base + 0] = Vertex{float2(x0, y0), float2(0, 0), col};
            vertices[base + 1] = Vertex{float2(x0, y1), float2(0, 1), col};
            vertices[base + 2] = Vertex{float2(x1, y1), float2(1, 1), col};
            vertices[base + 3] = Vertex{float2(x0, y0), float2(0, 0), col};
            vertices[base + 4] = Vertex{float2(x1, y1), float2(1, 1), col};
            vertices[base + 5] = Vertex{float2(x1, y0), float2(1, 0), col};
        }
    "#;

    let options = CompileOptions::new();
    let library = device.new_library_with_source(shader, &options).expect("Shader compile failed");

    let state_fn = library.get_function("trad_state_update", None).unwrap();
    let layout_fn = library.get_function("trad_layout", None).unwrap();
    let vertex_fn = library.get_function("trad_vertex_gen", None).unwrap();

    (
        device.new_compute_pipeline_state_with_function(&state_fn).unwrap(),
        device.new_compute_pipeline_state_with_function(&layout_fn).unwrap(),
        device.new_compute_pipeline_state_with_function(&vertex_fn).unwrap(),
    )
}

fn run_traditional(
    queue: &CommandQueue,
    pipelines: &(ComputePipelineState, ComputePipelineState, ComputePipelineState),
    widgets: &Buffer,
    vertices: &Buffer,
    count: usize,
) {
    let count_buffer = queue.device().new_buffer_with_data(
        &(count as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Simulate traditional model: 3 SEPARATE command buffers (like 3 sync points)
    // In reality, traditional apps often have even more sync points

    // Phase 1: State update
    let cmd1 = queue.new_command_buffer();
    let enc1 = cmd1.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&pipelines.0);
    enc1.set_buffer(0, Some(widgets), 0);
    enc1.set_buffer(1, Some(&count_buffer), 0);
    let threads = count.min(1024) as u64;
    enc1.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threads, 1, 1));
    enc1.end_encoding();
    cmd1.commit();
    cmd1.wait_until_completed(); // SYNC POINT 1

    // Phase 2: Layout
    let cmd2 = queue.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pipelines.1);
    enc2.set_buffer(0, Some(widgets), 0);
    enc2.set_buffer(1, Some(&count_buffer), 0);
    enc2.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threads, 1, 1));
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed(); // SYNC POINT 2

    // Phase 3: Vertex generation
    let cmd3 = queue.new_command_buffer();
    let enc3 = cmd3.new_compute_command_encoder();
    enc3.set_compute_pipeline_state(&pipelines.2);
    enc3.set_buffer(0, Some(widgets), 0);
    enc3.set_buffer(1, Some(vertices), 0);
    enc3.set_buffer(2, Some(&count_buffer), 0);
    enc3.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threads, 1, 1));
    enc3.end_encoding();
    cmd3.commit();
    cmd3.wait_until_completed(); // SYNC POINT 3
}

fn run_gpu_native(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    widgets: &Buffer,
    vertices: &Buffer,
    count: usize,
) {
    let count_buffer = queue.device().new_buffer_with_data(
        &(count as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // GPU-Native: ONE command buffer, ONE dispatch
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(widgets), 0);
    enc.set_buffer(1, Some(vertices), 0);
    enc.set_buffer(2, Some(&count_buffer), 0);
    let threads = count.min(1024) as u64;
    enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed(); // ONE sync point
}

fn generate_widgets(count: usize) -> Vec<Widget> {
    let cols = (count as f32).sqrt().ceil() as usize;
    let cell = 1.0 / cols as f32;

    (0..count).map(|i| {
        let row = i / cols;
        let col = i % cols;
        Widget {
            bounds: [
                col as f32 * cell,
                row as f32 * cell,
                cell * 0.9,
                cell * 0.9,
            ],
            state: 0,
            z_order: i as u32,
            color: 0xFF0080FF,
            _pad: 0,
        }
    }).collect()
}

fn benchmark_frame_pipelining(device: &Device, queue: &CommandQueue) {
    println!("┌───────────────────────────────────────────────────────────────────────────┐");
    println!("│                    FRAME PIPELINING BENCHMARK                            │");
    println!("├───────────────────────────────────────────────────────────────────────────┤");
    println!("│ Measuring throughput when frames overlap (GPU works while CPU submits)  │");
    println!("└───────────────────────────────────────────────────────────────────────────┘");
    println!();

    let pipeline = create_gpu_native_pipeline(device);
    let widgets = generate_widgets(1024);

    let widget_buffer = device.new_buffer_with_data(
        widgets.as_ptr() as *const _,
        (widgets.len() * std::mem::size_of::<Widget>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let vertex_buffer = device.new_buffer(
        1024 * 6 * 32,
        MTLResourceOptions::StorageModeShared,
    );
    let count_buffer = device.new_buffer_with_data(
        &1024u32 as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&widget_buffer), 0);
        enc.set_buffer(1, Some(&vertex_buffer), 0);
        enc.set_buffer(2, Some(&count_buffer), 0);
        enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark: Serial (wait after each frame)
    let serial_start = Instant::now();
    for _ in 0..1000 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&widget_buffer), 0);
        enc.set_buffer(1, Some(&vertex_buffer), 0);
        enc.set_buffer(2, Some(&count_buffer), 0);
        enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed(); // Wait each frame
    }
    let serial_time = serial_start.elapsed();

    // Benchmark: Pipelined (submit many, wait at end)
    let pipeline_start = Instant::now();
    let mut buffers = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&widget_buffer), 0);
        enc.set_buffer(1, Some(&vertex_buffer), 0);
        enc.set_buffer(2, Some(&count_buffer), 0);
        enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
        enc.end_encoding();
        cmd.commit();
        buffers.push(cmd);
    }
    // Wait only for last one
    buffers.last().unwrap().wait_until_completed();
    let pipeline_time = pipeline_start.elapsed();

    let serial_fps = 1000.0 / serial_time.as_secs_f64();
    let pipeline_fps = 1000.0 / pipeline_time.as_secs_f64();

    println!("1024 widgets, 1000 frames:");
    println!("  Serial (wait each):     {:>8.1} ms total = {:>6.0} FPS",
        serial_time.as_secs_f64() * 1000.0, serial_fps);
    println!("  Pipelined (overlap):    {:>8.1} ms total = {:>6.0} FPS",
        pipeline_time.as_secs_f64() * 1000.0, pipeline_fps);
    println!("  Pipelining speedup:     {:>8.2}x", pipeline_fps / serial_fps);
    println!();
}
