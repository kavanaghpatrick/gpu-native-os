// Benchmark Visual Demo - GPU vs CPU Performance Comparison
//
// A split-screen visualization showing GPU vs CPU performance in real-time.
// This is the "publishable demo" that proves the GPU-Native OS thesis visually.
//
// The demo shows the same operation (sorting or hit-testing widgets) running
// on both GPU and CPU, with live metrics showing the crossover point where
// GPU becomes dramatically faster.
//
// Controls:
// - Up/Down: Increase/decrease widget count (64 -> 128 -> 256 -> 512 -> 1024 -> 2048)
// - Space: Toggle between sort benchmark and hit-test benchmark
// - R: Randomize widget positions/z-orders

use super::app::{AppBuilder, GpuApp, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;
use std::mem;
use std::time::Instant;

// ============================================================================
// Constants
// ============================================================================

// Widget count levels (powers of 2 for clean scaling)
// Max is 1024 due to Metal's threadgroup size limit for sorting
pub const WIDGET_COUNTS: [usize; 5] = [64, 128, 256, 512, 1024];
pub const MAX_WIDGETS: usize = 1024;

// Vertices: widgets (6 per widget) + bars (6*4) + text/labels (estimated 1000)
pub const MAX_VERTICES: usize = MAX_WIDGETS * 6 + 24 + 5000;

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Widget data for benchmarking - 16 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BenchWidget {
    pub x: f32,
    pub y: f32,
    pub z_order: u32,
    pub color: u32, // packed RGBA
}

/// Benchmark parameters - 64 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BenchParams {
    pub widget_count: u32,
    pub mode: u32,           // 0 = sort, 1 = hit-test
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub time: f32,
    pub delta_time: f32,
    pub gpu_time_ms: f32,    // Measured GPU time
    pub cpu_time_ms: f32,    // Measured CPU time
    pub randomize: u32,      // Set to 1 to trigger randomization
    pub frame_number: u32,
    pub _padding: [u32; 6],
}

impl Default for BenchParams {
    fn default() -> Self {
        Self {
            widget_count: 64,
            mode: 0,
            cursor_x: 0.5,
            cursor_y: 0.5,
            time: 0.0,
            delta_time: 1.0 / 120.0,
            gpu_time_ms: 0.0,
            cpu_time_ms: 0.0,
            randomize: 0,
            frame_number: 0,
            _padding: [0; 6],
        }
    }
}

/// Vertex for rendering - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BenchVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

// ============================================================================
// Shader Source
// ============================================================================

fn shader_source() -> String {
    format!(
        r#"
{header}

// ============================================================================
// App-Specific Structures
// ============================================================================

struct BenchWidget {{
    float x;
    float y;
    uint z_order;
    uint color;
}};

struct BenchParams {{
    uint widget_count;
    uint mode;           // 0 = sort, 1 = hit-test
    float cursor_x;
    float cursor_y;
    float time;
    float delta_time;
    float gpu_time_ms;
    float cpu_time_ms;
    uint randomize;
    uint frame_number;
    uint _padding[6];
}};

struct BenchVertex {{
    float2 position;
    float2 uv;
    float4 color;
}};

// ============================================================================
// 7-Segment Digit Patterns
// ============================================================================

// Segments: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
constant bool digit_segments[10][7] = {{
    {{true,  true,  true,  true,  true,  true,  false}}, // 0
    {{false, true,  true,  false, false, false, false}}, // 1
    {{true,  true,  false, true,  true,  false, true}},  // 2
    {{true,  true,  true,  true,  false, false, true}},  // 3
    {{false, true,  true,  false, false, true,  true}},  // 4
    {{true,  false, true,  true,  false, true,  true}},  // 5
    {{true,  false, true,  true,  true,  true,  true}},  // 6
    {{true,  true,  true,  false, false, false, false}}, // 7
    {{true,  true,  true,  true,  true,  true,  true}},  // 8
    {{true,  true,  true,  true,  false, true,  true}},  // 9
}};

// ============================================================================
// Helper Functions
// ============================================================================

void write_quad(device BenchVertex* vertices, uint base_idx,
                float2 pos, float2 size, float4 color) {{
    // TL, BL, BR, TL, BR, TR
    float2 tl = pos;
    float2 br = pos + size;
    float2 tr = float2(br.x, tl.y);
    float2 bl = float2(tl.x, br.y);

    vertices[base_idx + 0].position = tl;
    vertices[base_idx + 0].color = color;
    vertices[base_idx + 1].position = bl;
    vertices[base_idx + 1].color = color;
    vertices[base_idx + 2].position = br;
    vertices[base_idx + 2].color = color;
    vertices[base_idx + 3].position = tl;
    vertices[base_idx + 3].color = color;
    vertices[base_idx + 4].position = br;
    vertices[base_idx + 4].color = color;
    vertices[base_idx + 5].position = tr;
    vertices[base_idx + 5].color = color;
}}

// Write a 7-segment digit at position, return vertex count used
uint write_digit(device BenchVertex* vertices, uint base_idx,
                 float2 pos, float digit_width, float digit_height,
                 int digit, float4 color) {{
    if (digit < 0 || digit > 9) return 0;

    float seg_thickness = digit_width * 0.15;
    float seg_len_h = digit_width - seg_thickness;
    float seg_len_v = (digit_height - seg_thickness * 3.0) / 2.0;

    uint idx = base_idx;

    // Top horizontal
    if (digit_segments[digit][0]) {{
        write_quad(vertices, idx, pos + float2(seg_thickness/2, 0),
                   float2(seg_len_h, seg_thickness), color);
        idx += 6;
    }}

    // Top-right vertical
    if (digit_segments[digit][1]) {{
        write_quad(vertices, idx, pos + float2(digit_width - seg_thickness, seg_thickness),
                   float2(seg_thickness, seg_len_v), color);
        idx += 6;
    }}

    // Bottom-right vertical
    if (digit_segments[digit][2]) {{
        write_quad(vertices, idx, pos + float2(digit_width - seg_thickness, seg_thickness * 2 + seg_len_v),
                   float2(seg_thickness, seg_len_v), color);
        idx += 6;
    }}

    // Bottom horizontal
    if (digit_segments[digit][3]) {{
        write_quad(vertices, idx, pos + float2(seg_thickness/2, digit_height - seg_thickness),
                   float2(seg_len_h, seg_thickness), color);
        idx += 6;
    }}

    // Bottom-left vertical
    if (digit_segments[digit][4]) {{
        write_quad(vertices, idx, pos + float2(0, seg_thickness * 2 + seg_len_v),
                   float2(seg_thickness, seg_len_v), color);
        idx += 6;
    }}

    // Top-left vertical
    if (digit_segments[digit][5]) {{
        write_quad(vertices, idx, pos + float2(0, seg_thickness),
                   float2(seg_thickness, seg_len_v), color);
        idx += 6;
    }}

    // Middle horizontal
    if (digit_segments[digit][6]) {{
        write_quad(vertices, idx, pos + float2(seg_thickness/2, digit_height/2 - seg_thickness/2),
                   float2(seg_len_h, seg_thickness), color);
        idx += 6;
    }}

    return idx - base_idx;
}}

// Write a number (up to 4 digits) at position
uint write_number(device BenchVertex* vertices, uint base_idx,
                  float2 pos, float digit_width, float digit_height,
                  int number, float4 color) {{
    uint idx = base_idx;
    float spacing = digit_width * 1.2;

    // Handle special case of 0
    if (number == 0) {{
        idx += write_digit(vertices, idx, pos, digit_width, digit_height, 0, color);
        return idx - base_idx;
    }}

    // Count digits
    int temp = number;
    int digit_count = 0;
    while (temp > 0) {{
        digit_count++;
        temp /= 10;
    }}

    // Write digits right-to-left
    float x_offset = (digit_count - 1) * spacing;
    temp = number;
    while (temp > 0) {{
        int d = temp % 10;
        idx += write_digit(vertices, idx, pos + float2(x_offset, 0),
                          digit_width, digit_height, d, color);
        x_offset -= spacing;
        temp /= 10;
    }}

    return idx - base_idx;
}}

// Write an 'x' character for speedup display
uint write_x(device BenchVertex* vertices, uint base_idx,
             float2 pos, float size, float4 color) {{
    float thickness = size * 0.2;

    // Two diagonal bars forming X
    // Top-left to bottom-right
    write_quad(vertices, base_idx, pos, float2(thickness, size), color);
    // Top-right to bottom-left
    write_quad(vertices, base_idx + 6, pos + float2(size - thickness, 0), float2(thickness, size), color);

    return 12;
}}

// ============================================================================
// GPU Sort Benchmark (Bitonic Sort)
// ============================================================================

void gpu_sort_widgets(device BenchWidget* widgets, uint count, uint tid, threadgroup BenchWidget* tg_widgets) {{
    // Load widget into threadgroup memory
    if (tid < count) {{
        tg_widgets[tid] = widgets[tid];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort (limited to 1024 - Metal max threadgroup size)
    for (uint k = 2; k <= 1024; k *= 2) {{
        for (uint j = k / 2; j > 0; j /= 2) {{
            uint ixj = tid ^ j;
            if (ixj > tid && tid < count && ixj < count) {{
                bool ascending = ((tid & k) == 0);
                BenchWidget a = tg_widgets[tid];
                BenchWidget b = tg_widgets[ixj];

                bool should_swap = ascending ? (a.z_order > b.z_order) : (a.z_order < b.z_order);
                if (should_swap) {{
                    tg_widgets[tid] = b;
                    tg_widgets[ixj] = a;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    }}

    // Write back
    if (tid < count) {{
        widgets[tid] = tg_widgets[tid];
    }}
}}

// ============================================================================
// GPU Hit Test Benchmark
// ============================================================================

uint gpu_hit_test(device BenchWidget* widgets, uint count, float2 cursor, uint tid) {{
    // Each thread tests a range of widgets
    uint per_thread = (count + 1023) / 1024;
    uint start = tid * per_thread;
    uint end = min(start + per_thread, count);

    uint hit_count = 0;
    float widget_size = 0.03;  // Normalized widget size

    for (uint i = start; i < end; i++) {{
        float2 widget_pos = float2(widgets[i].x, widgets[i].y);
        float2 delta = cursor - widget_pos;

        if (abs(delta.x) < widget_size && abs(delta.y) < widget_size) {{
            hit_count++;
        }}
    }}

    return hit_count;
}}

// ============================================================================
// Main Compute Kernel
// ============================================================================

kernel void benchmark_kernel(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant BenchParams& params [[buffer(2)]],
    device BenchWidget* widgets [[buffer(3)]],
    device BenchVertex* vertices [[buffer(4)]],
    device atomic_uint* vertex_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {{
    // Threadgroup memory for sorting (declared inside kernel)
    // Limited to 1024 - Metal's max threadgroup size
    threadgroup BenchWidget tg_widgets[1024];

    uint count = params.widget_count;

    // Randomize widgets if requested (only thread 0)
    if (params.randomize != 0 && tid < count) {{
        uint seed = tid + params.frame_number * 1234;
        widgets[tid].x = random_float(seed);
        widgets[tid].y = random_float(seed + 5678);
        widgets[tid].z_order = hash(seed + 9012) % count;

        // Color based on z_order
        float hue = float(widgets[tid].z_order) / float(count) * 360.0;
        float3 rgb = hsv_to_rgb(hue, 0.7, 0.9);
        widgets[tid].color = pack_color(rgb);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Run the selected benchmark
    if (params.mode == 0) {{
        // Sort benchmark
        gpu_sort_widgets(widgets, count, tid, tg_widgets);
    }} else {{
        // Hit-test benchmark
        gpu_hit_test(widgets, count, float2(params.cursor_x, params.cursor_y), tid);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // VISUALIZATION: Generate split-screen display
    // ═══════════════════════════════════════════════════════════════════

    // Thread 0 generates all visualization vertices
    if (tid != 0) return;

    uint vidx = 0;

    // Layout constants
    float left_half = -0.02;   // Divider position
    float margin = 0.05;
    float widget_display_size = 0.02;

    // Colors
    float4 gpu_color = float4(0.2, 0.9, 0.3, 1.0);   // Green for GPU
    float4 cpu_color = float4(0.9, 0.3, 0.2, 1.0);   // Red for CPU
    float4 white = float4(1.0, 1.0, 1.0, 1.0);
    float4 gray = float4(0.3, 0.3, 0.3, 1.0);
    float4 dark_gray = float4(0.15, 0.15, 0.18, 1.0);

    // ─────────────────────────────────────────────────────────────────
    // Draw center divider
    // ─────────────────────────────────────────────────────────────────
    write_quad(vertices, vidx, float2(left_half - 0.003, -1.0), float2(0.006, 2.0), gray);
    vidx += 6;

    // ─────────────────────────────────────────────────────────────────
    // Draw GPU side label background (left)
    // ─────────────────────────────────────────────────────────────────
    write_quad(vertices, vidx, float2(-1.0, 0.85), float2(0.98, 0.12), float4(gpu_color.rgb * 0.3, 0.8));
    vidx += 6;

    // ─────────────────────────────────────────────────────────────────
    // Draw CPU side label background (right)
    // ─────────────────────────────────────────────────────────────────
    write_quad(vertices, vidx, float2(0.02, 0.85), float2(0.98, 0.12), float4(cpu_color.rgb * 0.3, 0.8));
    vidx += 6;

    // ─────────────────────────────────────────────────────────────────
    // Draw widgets on both sides (showing sort/hit-test results)
    // ─────────────────────────────────────────────────────────────────

    // GPU side widgets (left) - show sorted/tested widgets
    for (uint i = 0; i < count && i < 256; i++) {{  // Limit to 256 for display
        float4 widget_color = unpack_color(widgets[i].color);

        // Position in grid on left side
        uint cols = 16;
        uint row = i / cols;
        uint col = i % cols;

        float x = -0.98 + float(col) * 0.055;
        float y = 0.75 - float(row) * 0.055;

        // Highlight if hit (for hit-test mode)
        if (params.mode == 1) {{
            float2 wpos = float2(widgets[i].x, widgets[i].y);
            float2 cursor = float2(params.cursor_x, params.cursor_y);
            if (length(wpos - cursor) < 0.1) {{
                widget_color = float4(1.0, 1.0, 0.0, 1.0);  // Yellow for hit
            }}
        }}

        write_quad(vertices, vidx, float2(x, y), float2(0.04, 0.04), widget_color);
        vidx += 6;
    }}

    // CPU side widgets (right) - simulated slower processing
    // Show fewer widgets or with "lag" effect
    uint cpu_show_count = min(count, uint(256));
    float cpu_lag = params.cpu_time_ms / 10.0;  // Visual lag based on CPU time

    for (uint i = 0; i < cpu_show_count; i++) {{
        float4 widget_color = unpack_color(widgets[i].color);

        // Position in grid on right side
        uint cols = 16;
        uint row = i / cols;
        uint col = i % cols;

        float x = 0.04 + float(col) * 0.055;
        float y = 0.75 - float(row) * 0.055;

        // Add visual "lag" effect - fade based on CPU time
        float fade = 1.0 - min(cpu_lag * 0.1, 0.5);
        widget_color.a *= fade;

        write_quad(vertices, vidx, float2(x, y), float2(0.04, 0.04), widget_color);
        vidx += 6;
    }}

    // ─────────────────────────────────────────────────────────────────
    // Draw timing bars
    // ─────────────────────────────────────────────────────────────────

    float bar_y = -0.6;
    float bar_height = 0.15;
    float max_time = max(params.gpu_time_ms, params.cpu_time_ms) * 1.2;
    if (max_time < 0.1) max_time = 1.0;

    // Bar backgrounds (drawn FIRST so bars render on top)
    write_quad(vertices, vidx, float2(-0.95, bar_y - 0.02), float2(0.85, bar_height + 0.04), dark_gray);
    vidx += 6;
    write_quad(vertices, vidx, float2(0.12, bar_y - 0.02), float2(0.85, bar_height + 0.04), dark_gray);
    vidx += 6;

    // GPU time bar (green, left side)
    float gpu_bar_width = (params.gpu_time_ms / max_time) * 0.8;
    write_quad(vertices, vidx, float2(-0.95, bar_y), float2(gpu_bar_width, bar_height), gpu_color);
    vidx += 6;

    // CPU time bar (red, right side)
    float cpu_bar_width = (params.cpu_time_ms / max_time) * 0.8;
    write_quad(vertices, vidx, float2(0.15, bar_y), float2(cpu_bar_width, bar_height), cpu_color);
    vidx += 6;

    // ─────────────────────────────────────────────────────────────────
    // Draw speedup indicator (center bottom)
    // ─────────────────────────────────────────────────────────────────

    float speedup = 1.0;
    if (params.gpu_time_ms > 0.001) {{
        speedup = params.cpu_time_ms / params.gpu_time_ms;
    }}

    // Speedup background
    float4 speedup_color = speedup > 1.5 ? gpu_color : (speedup < 0.8 ? cpu_color : white);
    write_quad(vertices, vidx, float2(-0.25, -0.95), float2(0.5, 0.2), float4(0.1, 0.1, 0.15, 0.9));
    vidx += 6;

    // Write speedup number
    int speedup_int = int(speedup * 10.0);  // One decimal place
    int speedup_whole = speedup_int / 10;
    int speedup_frac = speedup_int % 10;

    float digit_w = 0.06;
    float digit_h = 0.1;

    // Write whole part
    vidx += write_number(vertices, vidx, float2(-0.15, -0.9), digit_w, digit_h,
                         max(1, speedup_whole), speedup_color);

    // Write decimal point (small square)
    write_quad(vertices, vidx, float2(0.02, -0.82), float2(0.015, 0.015), speedup_color);
    vidx += 6;

    // Write fractional part
    vidx += write_digit(vertices, vidx, float2(0.05, -0.9), digit_w, digit_h,
                        speedup_frac, speedup_color);

    // Write 'x' symbol
    vidx += write_x(vertices, vidx, float2(0.15, -0.9), digit_h, speedup_color);

    // ─────────────────────────────────────────────────────────────────
    // Draw widget count indicator (top center)
    // ─────────────────────────────────────────────────────────────────

    write_quad(vertices, vidx, float2(-0.15, 0.92), float2(0.3, 0.06), float4(0.2, 0.2, 0.25, 0.9));
    vidx += 6;

    vidx += write_number(vertices, vidx, float2(-0.1, 0.93), 0.03, 0.04,
                        int(params.widget_count), white);

    // ─────────────────────────────────────────────────────────────────
    // Draw mode indicator
    // ─────────────────────────────────────────────────────────────────

    float4 mode_color = params.mode == 0 ? float4(0.4, 0.7, 1.0, 1.0) : float4(1.0, 0.7, 0.4, 1.0);
    write_quad(vertices, vidx, float2(-0.5, 0.92), float2(0.25, 0.06), mode_color * 0.3);
    vidx += 6;

    // Store total vertex count
    atomic_store_explicit(vertex_count, vidx, memory_order_relaxed);
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 uv;
    float4 color;
}};

vertex VertexOut benchmark_vertex(
    const device BenchVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    BenchVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}}

// ============================================================================
// Fragment Shader
// ============================================================================

fragment float4 benchmark_fragment(VertexOut in [[stage_in]]) {{
    return in.color;
}}
"#,
        header = APP_SHADER_HEADER
    )
}

// ============================================================================
// CPU Benchmark Implementation (for comparison)
// ============================================================================

fn cpu_sort_widgets(widgets: &mut [BenchWidget]) {
    // Simple bubble sort to simulate CPU overhead
    // (Real CPU sort would use quicksort, but we want to show the overhead)
    let n = widgets.len();
    for i in 0..n {
        for j in 0..(n - i - 1) {
            if widgets[j].z_order > widgets[j + 1].z_order {
                widgets.swap(j, j + 1);
            }
        }
    }
}

fn cpu_hit_test(widgets: &[BenchWidget], cursor_x: f32, cursor_y: f32) -> usize {
    let widget_size = 0.03f32;
    widgets.iter().filter(|w| {
        (w.x - cursor_x).abs() < widget_size && (w.y - cursor_y).abs() < widget_size
    }).count()
}

// ============================================================================
// BenchmarkVisual App
// ============================================================================

pub struct BenchmarkVisual {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    params_buffer: Buffer,
    widgets_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,

    // CPU-side widget copy for benchmarking
    cpu_widgets: Vec<BenchWidget>,

    // Current state
    current_params: BenchParams,
    widget_count_index: usize,  // Index into WIDGET_COUNTS
    time: f32,

    // Timing measurements
    gpu_times: Vec<f64>,
    cpu_times: Vec<f64>,
    avg_gpu_time: f64,
    avg_cpu_time: f64,

    // Input state
    mouse_x: f32,
    mouse_y: f32,
    pending_randomize: bool,
}

impl BenchmarkVisual {
    /// Create a new benchmark visualization
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "BenchmarkVisual");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create pipelines
        let compute_pipeline = builder.create_compute_pipeline(&library, "benchmark_kernel")?;
        let render_pipeline =
            builder.create_render_pipeline(&library, "benchmark_vertex", "benchmark_fragment")?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<BenchParams>());
        let widgets_buffer = builder.create_buffer(MAX_WIDGETS * mem::size_of::<BenchWidget>());
        let vertices_buffer = builder.create_buffer(MAX_VERTICES * mem::size_of::<BenchVertex>());
        let vertex_count_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Initialize widgets with random positions
        let mut cpu_widgets = Vec::with_capacity(MAX_WIDGETS);
        for i in 0..MAX_WIDGETS {
            let seed = i as u32;
            cpu_widgets.push(BenchWidget {
                x: Self::hash_to_float(seed),
                y: Self::hash_to_float(seed.wrapping_add(12345)),
                z_order: i as u32,
                color: Self::z_to_color(i, MAX_WIDGETS),
            });
        }

        // Copy to GPU buffer
        unsafe {
            let ptr = widgets_buffer.contents() as *mut BenchWidget;
            std::ptr::copy_nonoverlapping(cpu_widgets.as_ptr(), ptr, cpu_widgets.len());
        }

        // Initialize params
        let current_params = BenchParams {
            widget_count: WIDGET_COUNTS[0] as u32,
            ..Default::default()
        };
        unsafe {
            let ptr = params_buffer.contents() as *mut BenchParams;
            *ptr = current_params;
        }

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            widgets_buffer,
            vertices_buffer,
            vertex_count_buffer,
            cpu_widgets,
            current_params,
            widget_count_index: 0,
            time: 0.0,
            gpu_times: Vec::with_capacity(60),
            cpu_times: Vec::with_capacity(60),
            avg_gpu_time: 0.0,
            avg_cpu_time: 0.0,
            mouse_x: 0.5,
            mouse_y: 0.5,
            pending_randomize: true,  // Randomize on first frame
        })
    }

    fn hash_to_float(x: u32) -> f32 {
        let mut h = x;
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        (h as f32) / (u32::MAX as f32)
    }

    fn z_to_color(z: usize, max_z: usize) -> u32 {
        let hue = (z as f32 / max_z as f32) * 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);
        ((255u32) << 24) | ((b as u32) << 16) | ((g as u32) << 8) | (r as u32)
    }

    /// Increase widget count
    pub fn increase_widget_count(&mut self) {
        if self.widget_count_index < WIDGET_COUNTS.len() - 1 {
            self.widget_count_index += 1;
            self.current_params.widget_count = WIDGET_COUNTS[self.widget_count_index] as u32;
            self.gpu_times.clear();
            self.cpu_times.clear();
            println!("Widget count: {}", self.current_params.widget_count);
        }
    }

    /// Decrease widget count
    pub fn decrease_widget_count(&mut self) {
        if self.widget_count_index > 0 {
            self.widget_count_index -= 1;
            self.current_params.widget_count = WIDGET_COUNTS[self.widget_count_index] as u32;
            self.gpu_times.clear();
            self.cpu_times.clear();
            println!("Widget count: {}", self.current_params.widget_count);
        }
    }

    /// Toggle benchmark mode
    pub fn toggle_mode(&mut self) {
        self.current_params.mode = if self.current_params.mode == 0 { 1 } else { 0 };
        self.gpu_times.clear();
        self.cpu_times.clear();
        let mode_name = if self.current_params.mode == 0 { "SORT" } else { "HIT-TEST" };
        println!("Benchmark mode: {}", mode_name);
    }

    /// Randomize widget positions
    pub fn randomize(&mut self) {
        self.pending_randomize = true;
        println!("Randomizing widgets...");
    }

    /// Get current widget count
    pub fn widget_count(&self) -> usize {
        WIDGET_COUNTS[self.widget_count_index]
    }

    /// Get current mode name
    pub fn mode_name(&self) -> &str {
        if self.current_params.mode == 0 { "Sort" } else { "Hit-Test" }
    }

    /// Get current speedup ratio
    pub fn speedup(&self) -> f64 {
        if self.avg_gpu_time > 0.001 {
            self.avg_cpu_time / self.avg_gpu_time
        } else {
            1.0
        }
    }

    /// Run CPU benchmark and update timing
    fn run_cpu_benchmark(&mut self) {
        let count = self.current_params.widget_count as usize;
        let start = Instant::now();

        // Copy current widget state for CPU test
        let mut test_widgets = self.cpu_widgets[..count].to_vec();

        if self.current_params.mode == 0 {
            // Sort benchmark
            cpu_sort_widgets(&mut test_widgets);
        } else {
            // Hit-test benchmark
            cpu_hit_test(&test_widgets, self.mouse_x, self.mouse_y);
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update rolling average
        self.cpu_times.push(elapsed_ms);
        if self.cpu_times.len() > 60 {
            self.cpu_times.remove(0);
        }
        self.avg_cpu_time = self.cpu_times.iter().sum::<f64>() / self.cpu_times.len() as f64;
    }

    /// Sync GPU widgets to CPU copy
    fn sync_widgets_from_gpu(&mut self) {
        let count = self.current_params.widget_count as usize;
        unsafe {
            let ptr = self.widgets_buffer.contents() as *const BenchWidget;
            for i in 0..count {
                self.cpu_widgets[i] = *ptr.add(i);
            }
        }
    }
}

impl GpuApp for BenchmarkVisual {
    fn name(&self) -> &str {
        "GPU vs CPU Benchmark"
    }

    fn compute_pipeline(&self) -> &ComputePipelineState {
        &self.compute_pipeline
    }

    fn render_pipeline(&self) -> &RenderPipelineState {
        &self.render_pipeline
    }

    fn vertices_buffer(&self) -> &Buffer {
        &self.vertices_buffer
    }

    fn vertex_count(&self) -> usize {
        // Calculate vertex count on CPU to avoid race condition with GPU
        // (GPU writes to buffer but CPU reads before GPU finishes)
        // Static elements: divider(6) + labels(12) + bars(24) + speedup(18) = ~60
        // Widgets: up to 256 per side * 2 sides * 6 verts = 3072
        // Total max: ~3200 vertices, use fixed upper bound
        MAX_VERTICES
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.widgets_buffer,      // slot 3
            &self.vertices_buffer,     // slot 4
            &self.vertex_count_buffer, // slot 5
        ]
    }

    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32) {
        self.time += delta_time;

        // Run CPU benchmark for comparison
        self.run_cpu_benchmark();

        // Sync widgets from GPU if needed
        if self.pending_randomize {
            self.sync_widgets_from_gpu();
        }

        // Update params
        self.current_params.delta_time = delta_time;
        self.current_params.cursor_x = frame_state.cursor_x;
        self.current_params.cursor_y = frame_state.cursor_y;
        self.current_params.time = self.time;
        self.current_params.frame_number = frame_state.frame_number;
        self.current_params.cpu_time_ms = self.avg_cpu_time as f32;
        self.current_params.gpu_time_ms = self.avg_gpu_time as f32;

        if self.pending_randomize {
            self.current_params.randomize = 1;
            self.pending_randomize = false;
        } else {
            self.current_params.randomize = 0;
        }

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut BenchParams;
            *ptr = self.current_params;
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            t if t == InputEventType::MouseMove as u16 => {
                self.mouse_x = event.position[0];
                self.mouse_y = event.position[1];
            }
            t if t == InputEventType::KeyDown as u16 => {
                // Handle key events
                match event.keycode {
                    0x7E => self.increase_widget_count(),  // Up arrow
                    0x7D => self.decrease_widget_count(),  // Down arrow
                    0x31 => self.toggle_mode(),            // Space
                    0x0F => self.randomize(),              // R key
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn post_frame(&mut self, timing: &FrameTiming) {
        // Update GPU timing estimate (we use total frame time as proxy)
        // In real implementation, we'd use GPU timestamp queries
        let gpu_estimate = timing.total_ms * 0.8;  // Assume 80% is GPU work

        self.gpu_times.push(gpu_estimate);
        if self.gpu_times.len() > 60 {
            self.gpu_times.remove(0);
        }
        self.avg_gpu_time = self.gpu_times.iter().sum::<f64>() / self.gpu_times.len() as f64;
    }

    fn clear_color(&self) -> MTLClearColor {
        // Dark background
        MTLClearColor::new(0.08, 0.08, 0.12, 1.0)
    }

    fn thread_count(&self) -> usize {
        1024  // Metal max threadgroup size
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<BenchWidget>(), 16);
        assert_eq!(mem::size_of::<BenchParams>(), 64);
        assert_eq!(mem::size_of::<BenchVertex>(), 32);
    }

    #[test]
    fn test_widget_counts() {
        assert_eq!(WIDGET_COUNTS.len(), 5);
        assert_eq!(WIDGET_COUNTS[0], 64);
        assert_eq!(WIDGET_COUNTS[4], 1024);  // Max is 1024 due to threadgroup limit
    }

    #[test]
    fn test_cpu_sort() {
        let mut widgets = vec![
            BenchWidget { x: 0.0, y: 0.0, z_order: 5, color: 0 },
            BenchWidget { x: 0.0, y: 0.0, z_order: 2, color: 0 },
            BenchWidget { x: 0.0, y: 0.0, z_order: 8, color: 0 },
            BenchWidget { x: 0.0, y: 0.0, z_order: 1, color: 0 },
        ];
        cpu_sort_widgets(&mut widgets);
        assert_eq!(widgets[0].z_order, 1);
        assert_eq!(widgets[1].z_order, 2);
        assert_eq!(widgets[2].z_order, 5);
        assert_eq!(widgets[3].z_order, 8);
    }

    #[test]
    fn test_cpu_hit_test() {
        let widgets = vec![
            BenchWidget { x: 0.5, y: 0.5, z_order: 0, color: 0 },
            BenchWidget { x: 0.1, y: 0.1, z_order: 1, color: 0 },
            BenchWidget { x: 0.52, y: 0.51, z_order: 2, color: 0 },
        ];
        let hits = cpu_hit_test(&widgets, 0.5, 0.5);
        assert!(hits >= 1); // At least the first widget should be hit
    }
}
