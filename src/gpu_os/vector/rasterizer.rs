//! GPU Vector Rasterizer
//!
//! Renders vector paths using GPU compute for tessellation and fragment shaders for AA.

use super::path::{Path, PathCommand, PathSegment};
use crate::gpu_os::app::{AppBuilder, APP_SHADER_HEADER};
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

const MAX_PATHS: usize = 1024;
const MAX_SEGMENTS: usize = 16384;
const MAX_VERTICES: usize = 262144; // 256K vertices
const TESSELLATION_TOLERANCE: f32 = 0.5; // Pixels

// ============================================================================
// Data Structures
// ============================================================================

/// RGBA color
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Color = Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const BLACK: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const RED: Color = Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Color = Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Color = Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Color = Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const CYAN: Color = Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Color = Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };

    pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as f32 / 255.0,
            g: ((hex >> 8) & 0xFF) as f32 / 255.0,
            b: (hex & 0xFF) as f32 / 255.0,
            a: 1.0,
        }
    }
}

/// Fill rule for determining inside/outside
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FillRule {
    EvenOdd = 0,
    NonZero = 1,
}

/// Gradient stop (position 0.0-1.0 and color)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GradientStop {
    pub position: f32,
    pub color: Color,
}

impl GradientStop {
    pub fn new(position: f32, color: Color) -> Self {
        Self { position, color }
    }
}

/// Linear gradient definition
#[derive(Clone, Debug)]
pub struct LinearGradient {
    /// Start point (in path local coordinates, typically 0.0-1.0)
    pub start: [f32; 2],
    /// End point
    pub end: [f32; 2],
    /// Color stops (should be sorted by position)
    pub stops: Vec<GradientStop>,
}

impl LinearGradient {
    pub fn new(start: [f32; 2], end: [f32; 2], stops: Vec<GradientStop>) -> Self {
        Self { start, end, stops }
    }

    /// Create a simple two-color gradient
    pub fn two_color(start: [f32; 2], end: [f32; 2], color1: Color, color2: Color) -> Self {
        Self {
            start,
            end,
            stops: vec![
                GradientStop::new(0.0, color1),
                GradientStop::new(1.0, color2),
            ],
        }
    }
}

/// Radial gradient definition
#[derive(Clone, Debug)]
pub struct RadialGradient {
    /// Center point
    pub center: [f32; 2],
    /// Radius
    pub radius: f32,
    /// Color stops (should be sorted by position)
    pub stops: Vec<GradientStop>,
}

impl RadialGradient {
    pub fn new(center: [f32; 2], radius: f32, stops: Vec<GradientStop>) -> Self {
        Self { center, radius, stops }
    }

    /// Create a simple two-color radial gradient
    pub fn two_color(center: [f32; 2], radius: f32, inner: Color, outer: Color) -> Self {
        Self {
            center,
            radius,
            stops: vec![
                GradientStop::new(0.0, inner),
                GradientStop::new(1.0, outer),
            ],
        }
    }
}

/// Paint type for fills (Issue #35: Gradient support)
#[derive(Clone, Debug)]
pub enum Paint {
    Solid(Color),
    Linear(LinearGradient),
    Radial(RadialGradient),
}

impl Default for Paint {
    fn default() -> Self {
        Paint::Solid(Color::WHITE)
    }
}

/// Paint type constants for GPU
const PAINT_SOLID: u32 = 0;
const PAINT_LINEAR: u32 = 1;
const PAINT_RADIAL: u32 = 2;

/// Maximum gradient stops
const MAX_GRADIENT_STOPS: usize = 8;
const MAX_GRADIENTS: usize = 256;

/// GPU gradient structure (matches Metal shader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct GpuGradient {
    /// Paint type (0=solid, 1=linear, 2=radial)
    paint_type: u32,
    /// Number of stops
    stop_count: u32,
    /// Padding for 16-byte alignment
    _padding: [u32; 2],
    /// Start/center point for gradient
    point1: [f32; 4], // [x, y, 0, 0] - using float4 for alignment
    /// End point / radius for gradient
    point2: [f32; 4], // [x, y, radius, 0] for radial, [x, y, 0, 0] for linear
    /// Gradient stops (position + RGBA)
    stops: [[f32; 8]; MAX_GRADIENT_STOPS], // [pos, r, g, b, a, pad, pad, pad] * 8
}

/// Path header for GPU (per path metadata)
/// NOTE: Metal float4 requires 16-byte alignment, so we add padding after segment_count
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct PathHeader {
    segment_start: u32,
    segment_count: u32,
    paint_type: u32,      // 0=solid, 1=linear, 2=radial
    gradient_index: u32,  // Index into gradient buffer (if paint_type != 0)
    color: [f32; 4],      // Solid color (used when paint_type == 0)
    bounds: [f32; 4],
}

/// Vector vertex output from tessellation
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct VectorVertex {
    position: [f32; 2],     // Clip space position
    world_pos: [f32; 2],    // Original world position (for gradients)
    color: [f32; 4],        // Solid color or vertex color
    paint_info: [f32; 4],   // [paint_type, gradient_index, 0, 0]
}

/// Tessellation parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct TessParams {
    tolerance: f32,
    screen_width: f32,
    screen_height: f32,
    path_count: u32,
}

/// Vertex count (atomic)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct VertexCount {
    count: u32,
    _padding: [u32; 3],
}

// ============================================================================
// Shader Source
// ============================================================================

fn shader_source() -> String {
    format!(
        r#"
{header}

// ============================================================================
// Vector Rasterizer Structures (Issue #35: Gradient Support)
// ============================================================================

// Paint type constants
constant uint PAINT_SOLID = 0;
constant uint PAINT_LINEAR = 1;
constant uint PAINT_RADIAL = 2;

// Maximum gradient stops
constant uint MAX_GRADIENT_STOPS = 8;

struct PathSegment {{
    uchar cmd;
    uchar _pad[7];       // 8-byte alignment for float2
    float2 points[3];
}};

struct PathHeader {{
    uint segment_start;
    uint segment_count;
    uint paint_type;      // 0=solid, 1=linear, 2=radial
    uint gradient_index;  // Index into gradient buffer
    float4 color;         // Solid color (used when paint_type == 0)
    float4 bounds;
}};

// GPU gradient structure
struct GpuGradient {{
    uint paint_type;
    uint stop_count;
    uint _padding[2];
    float4 point1;        // Start/center point [x, y, 0, 0]
    float4 point2;        // End point [x, y, radius, 0] for radial
    float4 stops[MAX_GRADIENT_STOPS * 2]; // [pos, r, g, b] [a, pad, pad, pad] pairs
}};

struct VectorVertex {{
    float2 position;      // Clip space position
    float2 world_pos;     // Original world position (for gradients)
    float4 color;         // Solid color
    float4 paint_info;    // [paint_type, gradient_index, 0, 0]
}};

struct TessParams {{
    float tolerance;
    float screen_width;
    float screen_height;
    uint path_count;
}};

struct VertexCount {{
    atomic_uint count;
    uint _padding[3];
}};

// Command constants
constant uchar CMD_MOVE_TO = 0;
constant uchar CMD_LINE_TO = 1;
constant uchar CMD_QUAD_TO = 2;
constant uchar CMD_CUBIC_TO = 3;
constant uchar CMD_CLOSE = 4;

// ============================================================================
// Helper Functions
// ============================================================================

// Evaluate quadratic bezier at t
float2 eval_quad(float2 p0, float2 p1, float2 p2, float t) {{
    float t1 = 1.0 - t;
    return t1 * t1 * p0 + 2.0 * t1 * t * p1 + t * t * p2;
}}

// Evaluate cubic bezier at t
float2 eval_cubic(float2 p0, float2 p1, float2 p2, float2 p3, float t) {{
    float t1 = 1.0 - t;
    float t1_2 = t1 * t1;
    float t1_3 = t1_2 * t1;
    float t_2 = t * t;
    float t_3 = t_2 * t;
    return t1_3 * p0 + 3.0 * t1_2 * t * p1 + 3.0 * t1 * t_2 * p2 + t_3 * p3;
}}

// Estimate segments needed for quadratic bezier
uint quad_segments(float2 p0, float2 p1, float2 p2, float tol) {{
    // Deviation from chord
    float2 d = p1 - (p0 + p2) * 0.5;
    float dev = length(d);
    if (dev < tol) return 1;
    return uint(ceil(sqrt(dev / tol)));
}}

// Estimate segments needed for cubic bezier
uint cubic_segments(float2 p0, float2 p1, float2 p2, float2 p3, float tol) {{
    // Maximum deviation from chord
    float2 d1 = p1 - (p0 * 2.0 + p3) / 3.0;
    float2 d2 = p2 - (p0 + p3 * 2.0) / 3.0;
    float dev = max(length(d1), length(d2));
    if (dev < tol) return 1;
    return uint(ceil(pow(dev / tol, 0.333)));
}}

// Write a triangle - inline for simplicity
// Pass vertex_count struct pointer and do atomic inside

// ============================================================================
// Tessellation Kernel
// ============================================================================

// Safety constants
constant uint MAX_SEGMENT_COUNT = 1024u;
constant uint MAX_VERTEX_COUNT = 262144u;

// Macro-like inline function for writing triangles with bounds checking
// Now includes paint_info for gradient support
#define WRITE_TRI(p0, p1, p2) {{ \
    uint base = atomic_fetch_add_explicit(&vertex_count->count, 3, memory_order_relaxed); \
    if (base + 3 <= MAX_VERTEX_COUNT) {{ \
        float2 scale = float2(2.0 / screen_size.x, -2.0 / screen_size.y); \
        float2 bias = float2(-1.0, 1.0); \
        vertices[base + 0] = VectorVertex{{(p0) * scale + bias, (p0), color, paint_info}}; \
        vertices[base + 1] = VectorVertex{{(p1) * scale + bias, (p1), color, paint_info}}; \
        vertices[base + 2] = VectorVertex{{(p2) * scale + bias, (p2), color, paint_info}}; \
    }} \
}}

kernel void tessellate_paths(
    device const PathSegment* segments [[buffer(0)]],
    device const PathHeader* headers [[buffer(1)]],
    device VectorVertex* vertices [[buffer(2)]],
    device VertexCount* vertex_count [[buffer(3)]],
    constant TessParams& params [[buffer(4)]],
    device const GpuGradient* gradients [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= params.path_count) return;

    PathHeader header = headers[tid];

    // Bounds check segment_count to prevent infinite loops
    uint safe_segment_count = min(header.segment_count, MAX_SEGMENT_COUNT);
    if (safe_segment_count == 0) return;

    float4 color = header.color;
    float4 paint_info = float4(float(header.paint_type), float(header.gradient_index), 0.0, 0.0);
    float2 screen_size = float2(params.screen_width, params.screen_height);

    // Validate screen size to prevent division by zero
    if (screen_size.x < 1.0 || screen_size.y < 1.0) return;

    float tol = max(params.tolerance, 0.001); // Ensure minimum tolerance

    // Track current position and start of subpath
    float2 current = float2(0, 0);
    float2 subpath_start = float2(0, 0);

    // Collect line segments for the path (fan triangulation from centroid)
    // For simplicity, we'll use a triangle fan from the path centroid

    // First pass: find centroid
    float2 centroid = float2(0, 0);
    uint point_count = 0;

    for (uint i = 0; i < safe_segment_count; i++) {{
        PathSegment seg = segments[header.segment_start + i];

        if (seg.cmd == CMD_MOVE_TO || seg.cmd == CMD_LINE_TO) {{
            centroid += seg.points[0];
            point_count++;
        }} else if (seg.cmd == CMD_QUAD_TO) {{
            centroid += seg.points[0];
            centroid += seg.points[1];
            point_count += 2;
        }} else if (seg.cmd == CMD_CUBIC_TO) {{
            centroid += seg.points[0];
            centroid += seg.points[1];
            centroid += seg.points[2];
            point_count += 3;
        }}
    }}

    if (point_count > 0) {{
        centroid /= float(point_count);
    }}

    // Second pass: generate triangles (fan from centroid)
    current = float2(0, 0);
    float2 prev = current;

    for (uint i = 0; i < safe_segment_count; i++) {{
        PathSegment seg = segments[header.segment_start + i];

        if (seg.cmd == CMD_MOVE_TO) {{
            current = seg.points[0];
            subpath_start = current;
            prev = current;
        }} else if (seg.cmd == CMD_LINE_TO) {{
            float2 next = seg.points[0];
            WRITE_TRI(centroid, prev, next);
            prev = next;
            current = next;
        }} else if (seg.cmd == CMD_QUAD_TO) {{
            float2 p1 = seg.points[0];
            float2 p2 = seg.points[1];
            uint n = quad_segments(current, p1, p2, tol);
            n = min(n, 16u); // Cap segments

            for (uint j = 1; j <= n; j++) {{
                float t = float(j) / float(n);
                float2 next = eval_quad(current, p1, p2, t);
                WRITE_TRI(centroid, prev, next);
                prev = next;
            }}
            current = p2;
        }} else if (seg.cmd == CMD_CUBIC_TO) {{
            float2 p1 = seg.points[0];
            float2 p2 = seg.points[1];
            float2 p3 = seg.points[2];
            uint n = cubic_segments(current, p1, p2, p3, tol);
            n = min(n, 16u); // Cap segments

            for (uint j = 1; j <= n; j++) {{
                float t = float(j) / float(n);
                float2 next = eval_cubic(current, p1, p2, p3, t);
                WRITE_TRI(centroid, prev, next);
                prev = next;
            }}
            current = p3;
        }} else if (seg.cmd == CMD_CLOSE) {{
            if (length(current - subpath_start) > 0.001) {{
                WRITE_TRI(centroid, prev, subpath_start);
            }}
            current = subpath_start;
            prev = subpath_start;
        }}
    }}
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 world_pos;
    float4 color;
    float4 paint_info;   // [paint_type, gradient_index, 0, 0]
}};

vertex VertexOut vector_vertex(
    const device VectorVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    VectorVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.world_pos = v.world_pos;
    out.color = v.color;
    out.paint_info = v.paint_info;
    return out;
}}

// ============================================================================
// Fragment Shader with Gradient Support
// ============================================================================

// Sample gradient color at position t (0-1)
float4 sample_gradient(device const GpuGradient* gradients, uint idx, float t) {{
    GpuGradient grad = gradients[idx];

    // Clamp t to [0, 1]
    t = saturate(t);

    // Find the two stops to interpolate between
    float4 color = float4(1.0); // Default white

    if (grad.stop_count == 0) return color;

    // Each stop is stored as two float4s: [pos, r, g, b] [a, pad, pad, pad]
    uint stop_idx = 0;
    for (uint i = 0; i < grad.stop_count; i++) {{
        float pos = grad.stops[i * 2].x;
        if (t <= pos) {{
            if (i == 0) {{
                // Before first stop, use first color
                float4 s0 = grad.stops[0];
                float4 s1 = grad.stops[1];
                return float4(s0.y, s0.z, s0.w, s1.x);
            }}
            // Interpolate between stops[i-1] and stops[i]
            float4 prev0 = grad.stops[(i - 1) * 2];
            float4 prev1 = grad.stops[(i - 1) * 2 + 1];
            float4 curr0 = grad.stops[i * 2];
            float4 curr1 = grad.stops[i * 2 + 1];

            float prev_pos = prev0.x;
            float curr_pos = curr0.x;
            float local_t = (t - prev_pos) / max(curr_pos - prev_pos, 0.0001);

            float4 prev_color = float4(prev0.y, prev0.z, prev0.w, prev1.x);
            float4 curr_color = float4(curr0.y, curr0.z, curr0.w, curr1.x);

            return mix(prev_color, curr_color, local_t);
        }}
        stop_idx = i;
    }}

    // After last stop, use last color
    float4 last0 = grad.stops[stop_idx * 2];
    float4 last1 = grad.stops[stop_idx * 2 + 1];
    return float4(last0.y, last0.z, last0.w, last1.x);
}}

fragment float4 vector_fragment(
    VertexOut in [[stage_in]],
    device const GpuGradient* gradients [[buffer(0)]]
) {{
    uint paint_type = uint(in.paint_info.x);
    uint gradient_index = uint(in.paint_info.y);

    if (paint_type == PAINT_SOLID) {{
        return in.color;
    }}
    else if (paint_type == PAINT_LINEAR) {{
        GpuGradient grad = gradients[gradient_index];
        float2 start = grad.point1.xy;
        float2 end = grad.point2.xy;
        float2 dir = end - start;
        float len2 = dot(dir, dir);
        float t = 0.0;
        if (len2 > 0.0001) {{
            t = dot(in.world_pos - start, dir) / len2;
        }}
        return sample_gradient(gradients, gradient_index, t);
    }}
    else if (paint_type == PAINT_RADIAL) {{
        GpuGradient grad = gradients[gradient_index];
        float2 center = grad.point1.xy;
        float radius = grad.point2.z;
        float dist = length(in.world_pos - center);
        float t = dist / max(radius, 0.0001);
        return sample_gradient(gradients, gradient_index, t);
    }}

    return in.color;
}}

"#,
        header = APP_SHADER_HEADER
    )
}

// ============================================================================
// VectorRenderer
// ============================================================================

/// GPU vector renderer with compute tessellation and fragment rendering
pub struct VectorRenderer {
    command_queue: CommandQueue,
    tessellate_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    segments_buffer: Buffer,
    headers_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,
    params_buffer: Buffer,

    // State
    paths: Vec<(Path, Color)>,
    segment_count: usize,
}

impl VectorRenderer {
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "VectorRenderer");

        // Compile shaders
        let library = builder.compile_library(&shader_source())?;
        let tessellate_pipeline = builder.create_compute_pipeline(&library, "tessellate_paths")?;
        let render_pipeline = builder.create_render_pipeline(&library, "vector_vertex", "vector_fragment")?;

        // Create buffers
        let segments_buffer = builder.create_buffer(MAX_SEGMENTS * mem::size_of::<PathSegment>());
        let headers_buffer = builder.create_buffer(MAX_PATHS * mem::size_of::<PathHeader>());
        let vertices_buffer = builder.create_buffer(MAX_VERTICES * mem::size_of::<VectorVertex>());
        let vertex_count_buffer = builder.create_buffer(mem::size_of::<VertexCount>());
        let params_buffer = builder.create_buffer(mem::size_of::<TessParams>());

        let command_queue = device.new_command_queue();

        Ok(Self {
            command_queue,
            tessellate_pipeline,
            render_pipeline,
            segments_buffer,
            headers_buffer,
            vertices_buffer,
            vertex_count_buffer,
            params_buffer,
            paths: Vec::new(),
            segment_count: 0,
        })
    }

    /// Add a filled path
    pub fn fill(&mut self, path: &Path, paint: Paint) {
        let color = match paint {
            Paint::Solid(c) => c,
        };
        self.paths.push((path.clone(), color));
    }

    /// Add a filled path with color directly
    pub fn fill_color(&mut self, path: &Path, color: Color) {
        self.paths.push((path.clone(), color));
    }

    /// Stroke a path with the given color and width
    pub fn stroke(&mut self, path: &Path, color: Color, width: f32) {
        // Expand stroke to filled path
        let stroke_path = Self::expand_stroke(path, width);
        self.paths.push((stroke_path, color));
    }

    /// Stroke a path with paint
    pub fn stroke_paint(&mut self, path: &Path, paint: Paint, width: f32) {
        let color = match paint {
            Paint::Solid(c) => c,
        };
        self.stroke(path, color, width);
    }

    /// Expand a stroke into a filled path
    /// Creates an outline path by offsetting the original path by half the stroke width
    fn expand_stroke(path: &Path, width: f32) -> Path {
        use super::path::PathBuilder;

        let half_width = width / 2.0;
        let mut builder = PathBuilder::new();

        // Collect points from the path
        let mut points: Vec<[f32; 2]> = Vec::new();
        let mut current = [0.0f32, 0.0];
        let mut start = [0.0f32, 0.0];

        for seg in &path.segments {
            match seg.command() {
                PathCommand::MoveTo => {
                    start = seg.points[0];
                    current = start;
                    points.push(current);
                }
                PathCommand::LineTo => {
                    current = seg.points[0];
                    points.push(current);
                }
                PathCommand::QuadTo => {
                    // Flatten quadratic bezier
                    let p0 = current;
                    let p1 = seg.points[0];
                    let p2 = seg.points[1];
                    for i in 1..=8 {
                        let t = i as f32 / 8.0;
                        let mt = 1.0 - t;
                        let x = mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0];
                        let y = mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1];
                        points.push([x, y]);
                    }
                    current = p2;
                }
                PathCommand::CubicTo => {
                    // Flatten cubic bezier
                    let p0 = current;
                    let p1 = seg.points[0];
                    let p2 = seg.points[1];
                    let p3 = seg.points[2];
                    for i in 1..=12 {
                        let t = i as f32 / 12.0;
                        let mt = 1.0 - t;
                        let mt2 = mt * mt;
                        let t2 = t * t;
                        let x = mt2 * mt * p0[0] + 3.0 * mt2 * t * p1[0] + 3.0 * mt * t2 * p2[0] + t2 * t * p3[0];
                        let y = mt2 * mt * p0[1] + 3.0 * mt2 * t * p1[1] + 3.0 * mt * t2 * p2[1] + t2 * t * p3[1];
                        points.push([x, y]);
                    }
                    current = p3;
                }
                PathCommand::Close => {
                    if (current[0] - start[0]).abs() > 0.001 || (current[1] - start[1]).abs() > 0.001 {
                        points.push(start);
                    }
                    current = start;
                }
            }
        }

        if points.len() < 2 {
            return path.clone();
        }

        // Build offset paths (left and right sides of stroke)
        let mut left_points: Vec<[f32; 2]> = Vec::new();
        let mut right_points: Vec<[f32; 2]> = Vec::new();

        for i in 0..points.len() {
            let prev = if i == 0 { points[points.len() - 1] } else { points[i - 1] };
            let curr = points[i];
            let next = if i == points.len() - 1 { points[0] } else { points[i + 1] };

            // Calculate normals for incoming and outgoing edges
            let dx1 = curr[0] - prev[0];
            let dy1 = curr[1] - prev[1];
            let len1 = (dx1 * dx1 + dy1 * dy1).sqrt().max(0.0001);
            let nx1 = -dy1 / len1;
            let ny1 = dx1 / len1;

            let dx2 = next[0] - curr[0];
            let dy2 = next[1] - curr[1];
            let len2 = (dx2 * dx2 + dy2 * dy2).sqrt().max(0.0001);
            let nx2 = -dy2 / len2;
            let ny2 = dx2 / len2;

            // Average normal for miter join
            let nx = (nx1 + nx2) / 2.0;
            let ny = (ny1 + ny2) / 2.0;
            let nlen = (nx * nx + ny * ny).sqrt().max(0.0001);
            let nx = nx / nlen;
            let ny = ny / nlen;

            // Miter length (cap at 2x to avoid spikes)
            let dot = nx1 * nx + ny1 * ny;
            let miter_scale = (1.0 / dot.max(0.5)).min(2.0);

            left_points.push([
                curr[0] + nx * half_width * miter_scale,
                curr[1] + ny * half_width * miter_scale,
            ]);
            right_points.push([
                curr[0] - nx * half_width * miter_scale,
                curr[1] - ny * half_width * miter_scale,
            ]);
        }

        // Build the stroke outline: left side forward, then right side backward
        if let Some(first) = left_points.first() {
            builder.move_to(first[0], first[1]);
        }
        for point in left_points.iter().skip(1) {
            builder.line_to(point[0], point[1]);
        }
        // Connect to right side and go backward
        for point in right_points.iter().rev() {
            builder.line_to(point[0], point[1]);
        }
        builder.close();

        builder.build()
    }

    /// Clear all paths
    pub fn clear(&mut self) {
        self.paths.clear();
        self.segment_count = 0;
    }

    /// Upload paths to GPU and tessellate
    fn tessellate(&mut self, screen_width: f32, screen_height: f32) {
        if self.paths.is_empty() {
            return;
        }

        // Reset vertex count
        unsafe {
            let ptr = self.vertex_count_buffer.contents() as *mut VertexCount;
            (*ptr).count = 0;
        }

        // Upload segments and headers
        let mut segment_offset = 0usize;
        let mut headers = Vec::with_capacity(self.paths.len());

        unsafe {
            let seg_ptr = self.segments_buffer.contents() as *mut PathSegment;

            for (path, color) in &self.paths {
                // Copy segments
                for (i, seg) in path.segments.iter().enumerate() {
                    if segment_offset + i >= MAX_SEGMENTS {
                        break;
                    }
                    *seg_ptr.add(segment_offset + i) = *seg;
                }

                headers.push(PathHeader {
                    segment_start: segment_offset as u32,
                    segment_count: path.segments.len() as u32,
                    _padding: [0; 2],
                    color: [color.r, color.g, color.b, color.a],
                    bounds: path.bounds,
                });

                segment_offset += path.segments.len();
            }

            // Upload headers
            let header_ptr = self.headers_buffer.contents() as *mut PathHeader;
            for (i, h) in headers.iter().enumerate() {
                *header_ptr.add(i) = *h;
            }

            // Upload params
            let params_ptr = self.params_buffer.contents() as *mut TessParams;
            *params_ptr = TessParams {
                tolerance: TESSELLATION_TOLERANCE,
                screen_width,
                screen_height,
                path_count: self.paths.len() as u32,
            };
        }

        self.segment_count = segment_offset;

        // Run tessellation compute
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.tessellate_pipeline);
        encoder.set_buffer(0, Some(&self.segments_buffer), 0);
        encoder.set_buffer(1, Some(&self.headers_buffer), 0);
        encoder.set_buffer(2, Some(&self.vertices_buffer), 0);
        encoder.set_buffer(3, Some(&self.vertex_count_buffer), 0);
        encoder.set_buffer(4, Some(&self.params_buffer), 0);

        let path_count = self.paths.len() as u64;
        let threads_per_group = 64u64;
        let thread_groups = (path_count + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups.max(1), 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Render all paths
    pub fn render(
        &mut self,
        encoder: &RenderCommandEncoderRef,
        screen_width: f32,
        screen_height: f32,
    ) {
        // Tessellate paths
        self.tessellate(screen_width, screen_height);

        // Get vertex count
        let vertex_count = unsafe {
            let ptr = self.vertex_count_buffer.contents() as *const VertexCount;
            (*ptr).count as u64
        };

        if vertex_count == 0 {
            return;
        }

        // Render
        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count);
    }

    /// Get vertex count (for debugging)
    pub fn vertex_count(&self) -> u32 {
        unsafe {
            let ptr = self.vertex_count_buffer.contents() as *const VertexCount;
            (*ptr).count
        }
    }

    /// Get path count (for debugging)
    pub fn path_count(&self) -> usize {
        self.paths.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let c = Color::from_hex(0xFF5500);
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.g - 0.333).abs() < 0.01);
        assert!((c.b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_vertex_size() {
        // VectorVertex: 2 + 2 + 4 floats = 8 * 4 = 32 bytes
        assert_eq!(mem::size_of::<VectorVertex>(), 32);
    }

    #[test]
    fn test_path_header_size() {
        // PathHeader: 2 u32 + 2 u32 padding + 4 f32 + 4 f32 = 8 + 8 + 16 + 16 = 48 bytes
        // Metal float4 requires 16-byte alignment
        assert_eq!(mem::size_of::<PathHeader>(), 48);
    }
}
