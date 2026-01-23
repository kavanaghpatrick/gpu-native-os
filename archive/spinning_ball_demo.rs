use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use std::f32::consts::PI;
use std::mem;
use std::collections::VecDeque;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

// Simple vertex for text/UI rendering
// Note: float4 in Metal requires 16-byte alignment, so we need padding after position
#[repr(C)]
#[derive(Copy, Clone)]
struct TextVertex {
    position: [f32; 2],
    _padding: [f32; 2],  // Padding to align color to 16-byte boundary for Metal
    color: [f32; 4],
}

// Vertex structure - position + normal
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 4],
    normal: [f32; 4],
}

// Uniforms passed to shaders
#[repr(C)]
#[derive(Copy, Clone)]
struct Uniforms {
    model_matrix: [[f32; 4]; 4],
    view_proj_matrix: [[f32; 4]; 4],
    time: f32,
    _padding: [f32; 3],
}

// Generate UV sphere vertices
fn generate_sphere(radius: f32, slices: u32, stacks: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for stack in 0..=stacks {
        let phi = PI * (stack as f32) / (stacks as f32);
        let y = radius * phi.cos();
        let r = radius * phi.sin();

        for slice in 0..=slices {
            let theta = 2.0 * PI * (slice as f32) / (slices as f32);
            let x = r * theta.cos();
            let z = r * theta.sin();

            let nx = x / radius;
            let ny = y / radius;
            let nz = z / radius;

            vertices.push(Vertex {
                position: [x, y, z, 1.0],
                normal: [nx, ny, nz, 0.0],
            });
        }
    }

    // Generate indices with counter-clockwise winding (for front-facing outward)
    for stack in 0..stacks {
        for slice in 0..slices {
            let first = stack * (slices + 1) + slice;
            let second = first + slices + 1;

            // Triangle 1 - CCW winding
            indices.push(first);
            indices.push(first + 1);
            indices.push(second);

            // Triangle 2 - CCW winding
            indices.push(second);
            indices.push(first + 1);
            indices.push(second + 1);
        }
    }

    (vertices, indices)
}

// Matrix helpers
fn mat4_identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_rotation_y(angle: f32) -> [[f32; 4]; 4] {
    let c = angle.cos();
    let s = angle.sin();
    [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_rotation_x(angle: f32) -> [[f32; 4]; 4] {
    let c = angle.cos();
    let s = angle.sin();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_multiply(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

// Orthographic projection (working)
fn mat4_perspective(_fovy: f32, aspect: f32, _near: f32, _far: f32) -> [[f32; 4]; 4] {
    let scale = 0.3;
    [
        [scale / aspect, 0.0, 0.0, 0.0],
        [0.0, scale, 0.0, 0.0],
        [0.0, 0.0, scale, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// Transpose matrix for Metal (column-major)
fn mat4_transpose(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

fn mat4_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize([
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    // Standard view matrix for M * v (post-multiplication)
    [
        [s[0], s[1], s[2], -dot(s, eye)],
        [u[0], u[1], u[2], -dot(u, eye)],
        [-f[0], -f[1], -f[2], dot(f, eye)],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Metal shaders
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float4 position;
    float4 normal;
};

struct Uniforms {
    float4x4 model_matrix;
    float4x4 view_proj_matrix;
    float time;
    float3 _padding;
};

struct VertexOut {
    float4 position [[position]];
    float3 world_normal;
    float3 world_pos;
    float3 local_normal;
};

// ============================================================
// VERTEX SHADER - Transform vertices, all on GPU
// ============================================================
vertex VertexOut vertex_main(
    const device Vertex *vertices [[buffer(0)]],
    constant Uniforms &uniforms [[buffer(1)]],
    uint vid [[vertex_id]])
{
    Vertex v = vertices[vid];
    VertexOut out;

    // Transform position
    float4 world_pos = uniforms.model_matrix * v.position;
    out.position = uniforms.view_proj_matrix * world_pos;
    out.world_pos = world_pos.xyz;

    // Transform normal (use upper-left 3x3 of model matrix)
    float3x3 normal_matrix = float3x3(
        uniforms.model_matrix[0].xyz,
        uniforms.model_matrix[1].xyz,
        uniforms.model_matrix[2].xyz
    );
    out.world_normal = normalize(normal_matrix * v.normal.xyz);
    out.local_normal = v.normal.xyz;

    return out;
}

// ============================================================
// FRAGMENT SHADER - Colorful lighting, all on GPU
// ============================================================
fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    constant Uniforms &uniforms [[buffer(0)]])
{
    float time = uniforms.time;

    // === ANIMATED COLOR based on normal direction ===
    // Cycle through rainbow colors
    float3 n = normalize(in.local_normal);

    // Create base color from normal (gives position-based color variation)
    float3 pos_color = n * 0.5 + 0.5;

    // Animated rainbow cycle
    float r = sin(time * 0.5) * 0.5 + 0.5;
    float g = sin(time * 0.5 + 2.094) * 0.5 + 0.5;
    float b = sin(time * 0.5 + 4.189) * 0.5 + 0.5;
    float3 time_color = float3(r, g, b);

    // Mix position color with time color for dynamic effect
    float3 base_color = mix(pos_color, time_color, 0.6);

    // === LIGHTING ===
    float3 normal = normalize(in.world_normal);

    // Light orbits around the sphere
    float3 light_pos = float3(
        sin(time * 0.7) * 3.0,
        2.0,
        cos(time * 0.7) * 3.0
    );
    float3 light_dir = normalize(light_pos - in.world_pos);

    // Second light from opposite side
    float3 light_pos2 = float3(
        -sin(time * 0.5) * 2.5,
        -1.0,
        -cos(time * 0.5) * 2.5
    );
    float3 light_dir2 = normalize(light_pos2 - in.world_pos);

    // Diffuse lighting
    float diff1 = max(dot(normal, light_dir), 0.0);
    float diff2 = max(dot(normal, light_dir2), 0.0) * 0.5;
    float diffuse = diff1 + diff2;

    // Specular (Blinn-Phong)
    float3 view_dir = normalize(float3(0.0, 0.0, 4.0) - in.world_pos);
    float3 half_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_dir), 0.0), 64.0);

    // Ambient
    float ambient = 0.15;

    // Rim lighting for glow effect
    float rim = 1.0 - max(dot(normal, view_dir), 0.0);
    rim = pow(rim, 4.0) * 0.5;

    // Combine everything
    float3 final_color = base_color * (ambient + diffuse * 0.8) +
                         float3(1.0) * spec * 0.6 +
                         time_color * rim;

    return float4(final_color, 1.0);
}
"#;

// 7-segment display patterns for digits 0-9
// Each segment: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
const DIGIT_SEGMENTS: [[bool; 7]; 10] = [
    [true, true, true, true, true, true, false],    // 0
    [false, true, true, false, false, false, false], // 1
    [true, true, false, true, true, false, true],   // 2
    [true, true, true, true, false, false, true],   // 3
    [false, true, true, false, false, true, true],  // 4
    [true, false, true, true, false, true, true],   // 5
    [true, false, true, true, true, true, true],    // 6
    [true, true, true, false, false, false, false], // 7
    [true, true, true, true, true, true, true],     // 8
    [true, true, true, true, false, true, true],    // 9
];

// Letter patterns using 7-segment style
// Segments: top, top-right, bottom-right, bottom, bottom-left, top-left, middle
fn get_letter_segments(c: char) -> Option<[bool; 7]> {
    match c.to_ascii_uppercase() {
        'A' => Some([true, true, true, false, true, true, true]),
        'B' => Some([false, false, true, true, true, true, true]),  // lowercase b style
        'C' => Some([true, false, false, true, true, true, false]),
        'D' => Some([false, true, true, true, true, false, true]),  // lowercase d style
        'E' => Some([true, false, false, true, true, true, true]),
        'F' => Some([true, false, false, false, true, true, true]),
        'G' => Some([true, false, true, true, true, true, false]),
        'H' => Some([false, true, true, false, true, true, true]),
        'I' => Some([false, false, false, false, true, true, false]),
        'J' => Some([false, true, true, true, false, false, false]),
        'L' => Some([false, false, false, true, true, true, false]),
        'M' => Some([true, true, true, false, true, true, false]),  // approximation
        'N' => Some([false, false, true, false, true, false, true]),  // lowercase n
        'O' => Some([true, true, true, true, true, true, false]),
        'P' => Some([true, true, false, false, true, true, true]),
        'R' => Some([false, false, false, false, true, false, true]),  // lowercase r
        'S' => Some([true, false, true, true, false, true, true]),
        'T' => Some([false, false, false, true, true, true, true]),  // lowercase t style
        'U' => Some([false, true, true, true, true, true, false]),
        'Y' => Some([false, true, true, true, false, true, true]),
        ' ' => Some([false, false, false, false, false, false, false]),
        '-' => Some([false, false, false, false, false, false, true]),
        ':' => None,  // Special case handled separately
        _ => None,
    }
}

fn generate_digit_vertices(digit: u8, x: f32, y: f32, scale: f32, color: [f32; 4]) -> Vec<TextVertex> {
    let mut vertices = Vec::new();
    if digit > 9 { return vertices; }

    let segments = DIGIT_SEGMENTS[digit as usize];
    let w = scale * 0.6;  // width
    let h = scale;        // height
    let t = scale * 0.12; // thickness

    // Segment positions: [x1, y1, x2, y2, x3, y3, x4, y4] for each quad
    let segment_quads: [[f32; 8]; 7] = [
        // Top
        [x + t, y, x + w - t, y, x + w - t, y - t, x + t, y - t],
        // Top-right
        [x + w - t, y - t, x + w, y - t, x + w, y - h/2.0 + t/2.0, x + w - t, y - h/2.0 + t/2.0],
        // Bottom-right
        [x + w - t, y - h/2.0 - t/2.0, x + w, y - h/2.0 - t/2.0, x + w, y - h + t, x + w - t, y - h + t],
        // Bottom
        [x + t, y - h + t, x + w - t, y - h + t, x + w - t, y - h, x + t, y - h],
        // Bottom-left
        [x, y - h/2.0 - t/2.0, x + t, y - h/2.0 - t/2.0, x + t, y - h + t, x, y - h + t],
        // Top-left
        [x, y - t, x + t, y - t, x + t, y - h/2.0 + t/2.0, x, y - h/2.0 + t/2.0],
        // Middle
        [x + t, y - h/2.0 + t/2.0, x + w - t, y - h/2.0 + t/2.0, x + w - t, y - h/2.0 - t/2.0, x + t, y - h/2.0 - t/2.0],
    ];

    for (i, &active) in segments.iter().enumerate() {
        if active {
            let q = segment_quads[i];
            // Two triangles per quad (counter-clockwise winding)
            // Corners: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
            // Triangle 1: 0 -> 3 -> 2
            vertices.push(TextVertex { position: [q[0], q[1]], _padding: [0.0, 0.0], color });
            vertices.push(TextVertex { position: [q[6], q[7]], _padding: [0.0, 0.0], color });
            vertices.push(TextVertex { position: [q[4], q[5]], _padding: [0.0, 0.0], color });

            // Triangle 2: 0 -> 2 -> 1
            vertices.push(TextVertex { position: [q[0], q[1]], _padding: [0.0, 0.0], color });
            vertices.push(TextVertex { position: [q[4], q[5]], _padding: [0.0, 0.0], color });
            vertices.push(TextVertex { position: [q[2], q[3]], _padding: [0.0, 0.0], color });
        }
    }
    vertices
}

fn generate_number_vertices(mut num: u64, x: f32, y: f32, scale: f32, color: [f32; 4]) -> Vec<TextVertex> {
    let mut vertices = Vec::new();
    let mut digits = Vec::new();

    if num == 0 {
        digits.push(0u8);
    } else {
        while num > 0 {
            digits.push((num % 10) as u8);
            num /= 10;
        }
        digits.reverse();
    }

    let spacing = scale * 0.8;
    for (i, &d) in digits.iter().enumerate() {
        let dx = x + i as f32 * spacing;
        vertices.extend(generate_digit_vertices(d, dx, y, scale, color));
    }
    vertices
}

// Generate vertices for a single character (digit or letter)
fn generate_char_vertices(c: char, x: f32, y: f32, scale: f32, color: [f32; 4]) -> Vec<TextVertex> {
    let mut vertices = Vec::new();

    // Check if it's a digit
    if let Some(digit) = c.to_digit(10) {
        return generate_digit_vertices(digit as u8, x, y, scale, color);
    }

    // Check if it's a letter with a segment pattern
    if let Some(segments) = get_letter_segments(c) {
        let w = scale * 0.6;
        let h = scale;
        let t = scale * 0.12;

        let segment_quads: [[f32; 8]; 7] = [
            [x + t, y, x + w - t, y, x + w - t, y - t, x + t, y - t],
            [x + w - t, y - t, x + w, y - t, x + w, y - h/2.0 + t/2.0, x + w - t, y - h/2.0 + t/2.0],
            [x + w - t, y - h/2.0 - t/2.0, x + w, y - h/2.0 - t/2.0, x + w, y - h + t, x + w - t, y - h + t],
            [x + t, y - h + t, x + w - t, y - h + t, x + w - t, y - h, x + t, y - h],
            [x, y - h/2.0 - t/2.0, x + t, y - h/2.0 - t/2.0, x + t, y - h + t, x, y - h + t],
            [x, y - t, x + t, y - t, x + t, y - h/2.0 + t/2.0, x, y - h/2.0 + t/2.0],
            [x + t, y - h/2.0 + t/2.0, x + w - t, y - h/2.0 + t/2.0, x + w - t, y - h/2.0 - t/2.0, x + t, y - h/2.0 - t/2.0],
        ];

        for (i, &active) in segments.iter().enumerate() {
            if active {
                let q = segment_quads[i];
                vertices.push(TextVertex { position: [q[0], q[1]], _padding: [0.0, 0.0], color });
                vertices.push(TextVertex { position: [q[6], q[7]], _padding: [0.0, 0.0], color });
                vertices.push(TextVertex { position: [q[4], q[5]], _padding: [0.0, 0.0], color });
                vertices.push(TextVertex { position: [q[0], q[1]], _padding: [0.0, 0.0], color });
                vertices.push(TextVertex { position: [q[4], q[5]], _padding: [0.0, 0.0], color });
                vertices.push(TextVertex { position: [q[2], q[3]], _padding: [0.0, 0.0], color });
            }
        }
    }

    vertices
}

// Generate vertices for a text string
fn generate_text_vertices(text: &str, x: f32, y: f32, scale: f32, color: [f32; 4]) -> Vec<TextVertex> {
    let mut vertices = Vec::new();
    let spacing = scale * 0.75;

    for (i, c) in text.chars().enumerate() {
        let cx = x + i as f32 * spacing;
        vertices.extend(generate_char_vertices(c, cx, y, scale, color));
    }

    vertices
}

fn generate_label_bar(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Vec<TextVertex> {
    vec![
        TextVertex { position: [x, y], _padding: [0.0, 0.0], color },
        TextVertex { position: [x + width, y], _padding: [0.0, 0.0], color },
        TextVertex { position: [x + width, y - height], _padding: [0.0, 0.0], color },
        TextVertex { position: [x, y], _padding: [0.0, 0.0], color },
        TextVertex { position: [x + width, y - height], _padding: [0.0, 0.0], color },
        TextVertex { position: [x, y - height], _padding: [0.0, 0.0], color },
    ]
}

// ============================================================================
// BAR-GRAPH BASED STATS DISPLAY
// ============================================================================

/// Statistics display configuration
pub struct StatsBarConfig {
    /// Position of the stats panel (top-left corner in clip space)
    pub x: f32,
    pub y: f32,
    /// Maximum width of the bars
    pub bar_max_width: f32,
    /// Height of each bar
    pub bar_height: f32,
    /// Vertical spacing between bars
    pub spacing: f32,
    /// Width of the frame time history graph
    pub history_width: f32,
    /// Height of the frame time history graph
    pub history_height: f32,
    /// Number of frame samples to display in history
    pub history_samples: usize,
}

impl Default for StatsBarConfig {
    fn default() -> Self {
        Self {
            x: -0.95,
            y: 0.95,
            bar_max_width: 0.4,
            bar_height: 0.03,
            spacing: 0.05,
            history_width: 0.5,
            history_height: 0.15,
            history_samples: 60,
        }
    }
}

/// Generate a filled rectangle as two triangles.
/// All coordinates are in clip space (-1 to 1).
///
/// # Arguments
/// * `x` - Left edge x coordinate
/// * `y` - Top edge y coordinate
/// * `width` - Width of rectangle (positive = extends right)
/// * `height` - Height of rectangle (positive = extends downward, subtracted from y)
/// * `color` - RGBA color values (0.0 to 1.0)
///
/// # Returns
/// 6 vertices forming 2 triangles (counter-clockwise winding)
fn generate_rect(x: f32, y: f32, width: f32, height: f32, color: [f32; 4]) -> Vec<TextVertex> {
    vec![
        // Triangle 1: top-left -> bottom-left -> bottom-right
        TextVertex { position: [x, y], _padding: [0.0, 0.0], color },
        TextVertex { position: [x, y - height], _padding: [0.0, 0.0], color },
        TextVertex { position: [x + width, y - height], _padding: [0.0, 0.0], color },
        // Triangle 2: top-left -> bottom-right -> top-right
        TextVertex { position: [x, y], _padding: [0.0, 0.0], color },
        TextVertex { position: [x + width, y - height], _padding: [0.0, 0.0], color },
        TextVertex { position: [x + width, y], _padding: [0.0, 0.0], color },
    ]
}

/// Generate a horizontal bar with background and fill.
///
/// # Arguments
/// * `x`, `y` - Top-left position in clip space
/// * `max_width` - Maximum bar width (background extends to this)
/// * `height` - Bar height
/// * `fill_ratio` - Fill amount from 0.0 to 1.0
/// * `fill_color` - Color for the filled portion
/// * `bg_color` - Color for the background
fn generate_horizontal_bar(
    x: f32,
    y: f32,
    max_width: f32,
    height: f32,
    fill_ratio: f32,
    fill_color: [f32; 4],
    bg_color: [f32; 4],
) -> Vec<TextVertex> {
    let mut vertices = Vec::with_capacity(12);

    // Background bar (darker, full width)
    vertices.extend(generate_rect(x, y, max_width, height, bg_color));

    // Fill bar (colored, proportional width)
    let fill_width = max_width * fill_ratio.clamp(0.0, 1.0);
    if fill_width > 0.001 {
        vertices.extend(generate_rect(x, y, fill_width, height, fill_color));
    }

    vertices
}

/// Generate the frame time history graph.
/// Shows the last N frame times as vertical bars, with color indicating performance.
///
/// # Arguments
/// * `frame_times` - Queue of frame times in milliseconds
/// * `x`, `y` - Top-left position in clip space
/// * `width`, `height` - Dimensions of the graph
/// * `max_frame_time_ms` - Maximum frame time for scaling (values above are clamped)
/// * `num_slots` - Number of vertical bar slots to display
fn generate_frame_history(
    frame_times: &VecDeque<f64>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    max_frame_time_ms: f64,
    num_slots: usize,
) -> Vec<TextVertex> {
    let mut vertices = Vec::new();

    // Semi-transparent background
    let bg_color = [0.1, 0.1, 0.15, 0.8];
    vertices.extend(generate_rect(x, y, width, height, bg_color));

    // Reference lines for target frame times
    let target_60fps = 16.67 / max_frame_time_ms;
    let target_30fps = 33.33 / max_frame_time_ms;

    // 60fps reference line (green)
    if target_60fps < 1.0 {
        let line_y = y - height * (1.0 - target_60fps as f32);
        vertices.extend(generate_rect(x, line_y + 0.001, width, 0.002, [0.0, 0.5, 0.0, 0.6]));
    }

    // 30fps reference line (orange)
    if target_30fps < 1.0 {
        let line_y = y - height * (1.0 - target_30fps as f32);
        vertices.extend(generate_rect(x, line_y + 0.001, width, 0.002, [0.5, 0.3, 0.0, 0.6]));
    }

    if frame_times.is_empty() {
        return vertices;
    }

    // Calculate bar dimensions
    let bar_width = width / num_slots as f32;
    let gap = bar_width * 0.15;
    let actual_bar_width = bar_width - gap;
    let num_samples = frame_times.len();

    // Draw frame time bars (newest samples on the right)
    for (i, &frame_time) in frame_times.iter().enumerate() {
        // Position bar in slot (offset so newest is rightmost)
        let slot = num_slots - num_samples + i;
        let bar_x = x + slot as f32 * bar_width + gap * 0.5;

        // Normalize frame time to bar height
        let normalized = (frame_time / max_frame_time_ms).clamp(0.0, 1.0) as f32;
        let bar_height = height * normalized;

        // Color based on frame time thresholds:
        // - Green: < 16.67ms (60+ FPS)
        // - Yellow/Orange: 16.67-33.33ms (30-60 FPS)
        // - Red: > 33.33ms (< 30 FPS)
        let color = if frame_time < 16.67 {
            [0.2, 0.9, 0.3, 0.9]
        } else if frame_time < 33.33 {
            // Interpolate from yellow to orange
            let t = ((frame_time - 16.67) / 16.67) as f32;
            [0.9, 0.9 - t * 0.5, 0.2, 0.9]
        } else {
            [0.9, 0.2, 0.2, 0.9]
        };

        // Draw bar from bottom up
        let bar_y = y - height + bar_height;
        vertices.extend(generate_rect(bar_x, bar_y, actual_bar_width, bar_height, color));
    }

    vertices
}

/// Generate complete bar-graph based statistics display.
///
/// This function creates a visual statistics overlay with:
/// - **FPS bar**: Horizontal bar where longer = higher FPS (green tones)
/// - **Frame time bar**: Horizontal bar where shorter = better (yellow/orange tones)
/// - **Frame history graph**: Last N frames as vertical bars with color-coded performance
///
/// All coordinates are in clip space (-1 to 1).
///
/// # Arguments
/// * `fps` - Current frames per second (averaged)
/// * `frame_time_ms` - Current frame time in milliseconds
/// * `frame_times` - History of frame times for the graph
/// * `config` - Optional configuration for positioning and sizing
///
/// # Returns
/// A vector of `TextVertex` that can be rendered as triangles using the text/UI pipeline.
///
/// # Example
/// ```rust
/// let vertices = generate_stats_bars(60.0, 16.67, &frame_times, None);
/// // Copy vertices to GPU buffer and render as triangles
/// ```
pub fn generate_stats_bars(
    fps: f64,
    frame_time_ms: f64,
    frame_times: &VecDeque<f64>,
    config: Option<StatsBarConfig>,
) -> Vec<TextVertex> {
    let cfg = config.unwrap_or_default();
    let mut vertices = Vec::new();
    let mut y = cfg.y;

    let label_scale = 0.025;
    let label_offset = cfg.bar_max_width + 0.02;
    let value_offset = cfg.bar_max_width + 0.12;

    // ========================================
    // FPS BAR (Green)
    // ========================================
    // Scale: 0-120 FPS maps to 0-100% bar fill
    let fps_ratio = (fps / 120.0).clamp(0.0, 1.0);

    // Color intensity based on FPS quality
    let fps_color = if fps >= 60.0 {
        [0.2, 0.9, 0.3, 1.0]  // Bright green for 60+ FPS
    } else if fps >= 30.0 {
        [0.7, 0.9, 0.2, 1.0]  // Yellow-green for 30-60 FPS
    } else {
        [0.9, 0.5, 0.2, 1.0]  // Orange for < 30 FPS
    };
    let fps_bg = [0.1, 0.2, 0.1, 0.6];

    vertices.extend(generate_horizontal_bar(
        cfg.x, y,
        cfg.bar_max_width, cfg.bar_height,
        fps_ratio as f32, fps_color, fps_bg,
    ));

    // FPS label and value
    vertices.extend(generate_text_vertices("FPS", cfg.x + label_offset, y, label_scale, fps_color));
    vertices.extend(generate_number_vertices(fps as u64, cfg.x + value_offset, y, label_scale, [1.0, 1.0, 1.0, 1.0]));

    // Target FPS markers (30, 60, 90 FPS)
    let marker_color = [0.4, 0.4, 0.4, 0.7];
    for &target_fps in &[30.0, 60.0, 90.0] {
        let marker_x = cfg.x + cfg.bar_max_width * (target_fps / 120.0) as f32;
        vertices.extend(generate_rect(marker_x, y, 0.003, cfg.bar_height, marker_color));
    }

    y -= cfg.spacing;

    // ========================================
    // FRAME TIME BAR (Yellow/Orange)
    // ========================================
    // Scale: 0-50ms maps to 0-100% bar fill
    // Note: Lower frame time is better, but we show actual value
    let ft_ratio = (frame_time_ms / 50.0).clamp(0.0, 1.0);

    // Color transitions based on quality thresholds
    let ft_color = if frame_time_ms < 16.67 {
        [0.9, 0.9, 0.2, 1.0]  // Bright yellow for < 16.67ms (60+ FPS)
    } else if frame_time_ms < 33.33 {
        [0.9, 0.7, 0.1, 1.0]  // Orange for 16.67-33.33ms (30-60 FPS)
    } else {
        [0.9, 0.3, 0.1, 1.0]  // Red-orange for > 33.33ms (< 30 FPS)
    };
    let ft_bg = [0.2, 0.15, 0.1, 0.6];

    vertices.extend(generate_horizontal_bar(
        cfg.x, y,
        cfg.bar_max_width, cfg.bar_height,
        ft_ratio as f32, ft_color, ft_bg,
    ));

    // MS label and value
    vertices.extend(generate_text_vertices("MS", cfg.x + label_offset, y, label_scale, ft_color));
    vertices.extend(generate_number_vertices(frame_time_ms as u64, cfg.x + value_offset, y, label_scale, [1.0, 1.0, 1.0, 1.0]));

    // Target frame time markers
    let mark_16ms = cfg.x + cfg.bar_max_width * (16.67 / 50.0) as f32;
    let mark_33ms = cfg.x + cfg.bar_max_width * (33.33 / 50.0) as f32;
    vertices.extend(generate_rect(mark_16ms, y, 0.003, cfg.bar_height, [0.0, 0.5, 0.0, 0.8]));
    vertices.extend(generate_rect(mark_33ms, y, 0.003, cfg.bar_height, [0.5, 0.3, 0.0, 0.8]));

    y -= cfg.spacing;

    // ========================================
    // FRAME TIME HISTORY GRAPH
    // ========================================
    let max_display_time = 50.0; // Maximum frame time to display (ms)
    vertices.extend(generate_frame_history(
        frame_times,
        cfg.x, y,
        cfg.history_width, cfg.history_height,
        max_display_time,
        cfg.history_samples,
    ));

    // History label
    vertices.extend(generate_text_vertices("HISTORY", cfg.x + cfg.history_width + 0.02, y, label_scale * 0.8, [0.6, 0.6, 0.6, 1.0]));

    vertices
}

// Simple text/UI shader
const TEXT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextVertex {
    float2 position;
    float2 _padding;  // Match Rust struct padding for 16-byte alignment of color
    float4 color;
};

struct TextOut {
    float4 position [[position]];
    float4 color;
};

vertex TextOut text_vertex(
    const device TextVertex *vertices [[buffer(0)]],
    uint vid [[vertex_id]])
{
    TextOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 text_fragment(TextOut in [[stage_in]])
{
    return in.color;
}
"#;

struct App {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<CommandQueue>,
    render_pipeline: Option<RenderPipelineState>,
    text_pipeline: Option<RenderPipelineState>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    uniform_buffer: Option<Buffer>,
    text_vertex_buffer: Option<Buffer>,
    index_count: u64,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
    frame_times: VecDeque<f64>,
    frame_count: u64,
    depth_texture: Option<Texture>,
    depth_state: Option<DepthStencilState>,
    aspect_ratio: f32,
    gpu_name: String,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            command_queue: None,
            render_pipeline: None,
            text_pipeline: None,
            vertex_buffer: None,
            index_buffer: None,
            uniform_buffer: None,
            text_vertex_buffer: None,
            index_count: 0,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
            frame_times: VecDeque::with_capacity(120),
            frame_count: 0,
            depth_texture: None,
            depth_state: None,
            aspect_ratio: 1.0,
            gpu_name: String::new(),
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("Using GPU: {}", device.name());
        println!("Unified Memory: {}", device.has_unified_memory());

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                    let view = appkit_handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
        self.aspect_ratio = size.width as f32 / size.height as f32;

        // Compile shaders
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &compile_options)
            .expect("Failed to compile shaders");

        // Create render pipeline
        let vertex_fn = library.get_function("vertex_main", None).unwrap();
        let fragment_fn = library.get_function("fragment_main", None).unwrap();

        let pipeline_desc = RenderPipelineDescriptor::new();
        pipeline_desc.set_vertex_function(Some(&vertex_fn));
        pipeline_desc.set_fragment_function(Some(&fragment_fn));

        let color_attachment = pipeline_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        pipeline_desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        let render_pipeline = device
            .new_render_pipeline_state(&pipeline_desc)
            .expect("Failed to create render pipeline");

        // Create text pipeline
        let text_library = device
            .new_library_with_source(TEXT_SHADER, &compile_options)
            .expect("Failed to compile text shaders");

        let text_vertex_fn = text_library.get_function("text_vertex", None).unwrap();
        let text_fragment_fn = text_library.get_function("text_fragment", None).unwrap();

        let text_pipeline_desc = RenderPipelineDescriptor::new();
        text_pipeline_desc.set_vertex_function(Some(&text_vertex_fn));
        text_pipeline_desc.set_fragment_function(Some(&text_fragment_fn));

        let text_color_attachment = text_pipeline_desc.color_attachments().object_at(0).unwrap();
        text_color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        text_color_attachment.set_blending_enabled(true);
        text_color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        text_color_attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let text_pipeline = device
            .new_render_pipeline_state(&text_pipeline_desc)
            .expect("Failed to create text pipeline");

        // Text vertex buffer (enough for stats display)
        let text_vertex_buffer = device.new_buffer(
            (4096 * mem::size_of::<TextVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Depth stencil state
        let depth_desc = DepthStencilDescriptor::new();
        depth_desc.set_depth_compare_function(MTLCompareFunction::Less);
        depth_desc.set_depth_write_enabled(true);
        let depth_state = device.new_depth_stencil_state(&depth_desc);

        // Generate sphere
        let (vertices, indices) = generate_sphere(1.0, 48, 32);
        let index_count = indices.len() as u64;

        println!("Sphere: {} vertices, {} triangles", vertices.len(), index_count / 3);

        let vertex_buffer = device.new_buffer_with_data(
            vertices.as_ptr() as *const _,
            (vertices.len() * mem::size_of::<Vertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let index_buffer = device.new_buffer_with_data(
            indices.as_ptr() as *const _,
            (indices.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let uniform_buffer = device.new_buffer(
            mem::size_of::<Uniforms>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let depth_texture = Self::create_depth_texture(&device, size.width, size.height);

        let command_queue = device.new_command_queue();

        self.gpu_name = device.name().to_string();
        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.render_pipeline = Some(render_pipeline);
        self.text_pipeline = Some(text_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.text_vertex_buffer = Some(text_vertex_buffer);
        self.index_count = index_count;
        self.depth_texture = Some(depth_texture);
        self.depth_state = Some(depth_state);
    }

    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        let desc = TextureDescriptor::new();
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        desc.set_storage_mode(MTLStorageMode::Private);
        desc.set_usage(MTLTextureUsage::RenderTarget);
        device.new_texture(&desc)
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let command_queue = self.command_queue.as_ref().unwrap();
        let render_pipeline = self.render_pipeline.as_ref().unwrap();
        let text_pipeline = self.text_pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let index_buffer = self.index_buffer.as_ref().unwrap();
        let uniform_buffer = self.uniform_buffer.as_ref().unwrap();
        let text_vertex_buffer = self.text_vertex_buffer.as_ref().unwrap();
        let depth_texture = self.depth_texture.as_ref().unwrap();
        let depth_state = self.depth_state.as_ref().unwrap();

        // Track frame time
        let now = std::time::Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f64() * 1000.0;
        self.last_frame_time = now;
        self.frame_count += 1;

        self.frame_times.push_back(frame_time);
        if self.frame_times.len() > 120 {
            self.frame_times.pop_front();
        }

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // Calculate time
        let time = self.start_time.elapsed().as_secs_f32();

        // === Build matrices on CPU (could move to GPU compute shader) ===

        // Model matrix: rotation
        let rot_y = mat4_rotation_y(time * 0.8);
        let rot_x = mat4_rotation_x(time * 0.5);
        let model_matrix = mat4_multiply(rot_y, rot_x);

        // DEBUG: Identity view for testing
        let view_matrix = mat4_identity();

        // Projection matrix
        let proj_matrix = mat4_perspective(
            PI / 4.0,          // 45 degree FOV
            self.aspect_ratio,
            0.1,
            100.0,
        );

        let view_proj = mat4_multiply(proj_matrix, view_matrix);

        // Transpose matrices for Metal (expects column-major)
        let uniforms = Uniforms {
            model_matrix: mat4_transpose(model_matrix),
            view_proj_matrix: mat4_transpose(view_proj),
            time,
            _padding: [0.0; 3],
        };

        unsafe {
            let ptr = uniform_buffer.contents() as *mut Uniforms;
            *ptr = uniforms;
        }

        let command_buffer = command_queue.new_command_buffer();

        // Render pass
        {
            let render_pass_desc = RenderPassDescriptor::new();

            let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(drawable.texture()));
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_clear_color(MTLClearColor::new(0.02, 0.02, 0.04, 1.0));
            color_attachment.set_store_action(MTLStoreAction::Store);

            let depth_attachment = render_pass_desc.depth_attachment().unwrap();
            depth_attachment.set_texture(Some(depth_texture));
            depth_attachment.set_load_action(MTLLoadAction::Clear);
            depth_attachment.set_clear_depth(1.0);
            depth_attachment.set_store_action(MTLStoreAction::DontCare);

            let encoder = command_buffer.new_render_command_encoder(&render_pass_desc);
            encoder.set_render_pipeline_state(render_pipeline);
            encoder.set_depth_stencil_state(depth_state);
            encoder.set_cull_mode(MTLCullMode::Back);
            encoder.set_front_facing_winding(MTLWinding::CounterClockwise);

            encoder.set_vertex_buffer(0, Some(vertex_buffer), 0);
            encoder.set_vertex_buffer(1, Some(uniform_buffer), 0);
            encoder.set_fragment_buffer(0, Some(uniform_buffer), 0);

            encoder.draw_indexed_primitives(
                MTLPrimitiveType::Triangle,
                self.index_count,
                MTLIndexType::UInt32,
                index_buffer,
                0,
            );

            encoder.end_encoding();
        }

        // Render stats overlay
        {
            // Calculate stats
            let avg_frame_time: f64 = if self.frame_times.is_empty() {
                0.0
            } else {
                self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64
            };
            let fps = if avg_frame_time > 0.0 { 1000.0 / avg_frame_time } else { 0.0 };
            let min_frame_time = self.frame_times.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_frame_time = self.frame_times.iter().cloned().fold(0.0, f64::max);

            // Build stats bar vertices using the new bar-graph display
            let text_verts = generate_stats_bars(fps, avg_frame_time, &self.frame_times, None);

            if !text_verts.is_empty() {
                unsafe {
                    let ptr = text_vertex_buffer.contents() as *mut TextVertex;
                    std::ptr::copy_nonoverlapping(text_verts.as_ptr(), ptr, text_verts.len());
                }

                let text_pass_desc = RenderPassDescriptor::new();
                let text_color_attachment = text_pass_desc.color_attachments().object_at(0).unwrap();
                text_color_attachment.set_texture(Some(drawable.texture()));
                text_color_attachment.set_load_action(MTLLoadAction::Load);
                text_color_attachment.set_store_action(MTLStoreAction::Store);

                let text_encoder = command_buffer.new_render_command_encoder(&text_pass_desc);
                text_encoder.set_render_pipeline_state(text_pipeline);
                text_encoder.set_vertex_buffer(0, Some(text_vertex_buffer), 0);
                text_encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, text_verts.len() as u64);
                text_encoder.end_encoding();
            }
        }

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }

    fn resize(&mut self, width: u32, height: u32) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }
        if let Some(device) = &self.device {
            self.depth_texture = Some(Self::create_depth_texture(device, width, height));
        }
        self.aspect_ratio = width as f32 / height as f32;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            .with_title("Metal GPU Spinning Colorful Ball");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
