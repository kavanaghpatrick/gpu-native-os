//! GPU Window Compositor
//!
//! Renders windows with compositing effects:
//! - Z-sorted window rendering
//! - Window shadows
//! - Window chrome (title bar, buttons)
//! - Focus state effects
//!
//! Uses a fragment shader approach for smooth anti-aliased rendering.

use metal::*;
use std::mem;

use super::types::*;

/// Vertex for compositor rendering
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct CompositorVertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
    pub color: [f32; 4],
    /// Window index (for looking up window data)
    pub window_index: u32,
    /// Vertex type: 0=shadow, 1=background, 2=titlebar, 3=button, 4=border
    pub vertex_type: u32,
    pub _pad: [f32; 2],
}

/// Compositor uniforms
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct CompositorUniforms {
    pub screen_width: f32,
    pub screen_height: f32,
    pub window_count: u32,
    pub focused_window: u32,

    /// Shadow parameters
    pub shadow_radius: f32,
    pub shadow_opacity: f32,
    pub shadow_offset_x: f32,
    pub shadow_offset_y: f32,

    /// Chrome colors (focused)
    pub titlebar_color_focused: [f32; 4],
    /// Chrome colors (unfocused)
    pub titlebar_color_unfocused: [f32; 4],

    /// Button colors
    pub close_button_color: [f32; 4],
    pub minimize_button_color: [f32; 4],
    pub maximize_button_color: [f32; 4],

    /// Border color
    pub border_color: [f32; 4],

    /// Window background
    pub window_background: [f32; 4],
}

impl Default for CompositorUniforms {
    fn default() -> Self {
        Self {
            screen_width: 1920.0,
            screen_height: 1080.0,
            window_count: 0,
            focused_window: 0,

            shadow_radius: 20.0,
            shadow_opacity: 0.3,
            shadow_offset_x: 0.0,
            shadow_offset_y: 5.0,

            // macOS-style traffic light colors
            titlebar_color_focused: [0.95, 0.95, 0.95, 1.0],     // Light gray
            titlebar_color_unfocused: [0.90, 0.90, 0.90, 1.0],   // Slightly darker
            close_button_color: [1.0, 0.38, 0.36, 1.0],          // Red
            minimize_button_color: [1.0, 0.74, 0.17, 1.0],       // Yellow
            maximize_button_color: [0.15, 0.78, 0.33, 1.0],      // Green
            border_color: [0.75, 0.75, 0.75, 1.0],               // Light border
            window_background: [1.0, 1.0, 1.0, 1.0],             // White
        }
    }
}

/// Maximum vertices per window (shadow + background + titlebar + 3 buttons + border)
pub const MAX_VERTICES_PER_WINDOW: usize = 64;

/// GPU Window Compositor
pub struct GpuCompositor {
    device: Device,
    command_queue: CommandQueue,

    // Pipelines
    vertex_gen_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    uniforms_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,

    // Configuration
    pub max_windows: usize,
}

impl GpuCompositor {
    /// Create a new compositor
    pub fn new(device: &Device, pixel_format: MTLPixelFormat) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = device.new_library_with_source(COMPOSITOR_METAL, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile compositor shaders: {}", e))?;

        // Compute pipeline for vertex generation
        let vertex_gen_fn = library.get_function("generate_compositor_vertices", None)
            .map_err(|e| format!("Failed to get generate_compositor_vertices: {}", e))?;
        let vertex_gen_pipeline = device.new_compute_pipeline_state_with_function(&vertex_gen_fn)
            .map_err(|e| format!("Failed to create vertex_gen pipeline: {}", e))?;

        // Render pipeline
        let vertex_fn = library.get_function("compositor_vertex", None)
            .map_err(|e| format!("Failed to get compositor_vertex: {}", e))?;
        let fragment_fn = library.get_function("compositor_fragment", None)
            .map_err(|e| format!("Failed to get compositor_fragment: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc.color_attachments().object_at(0).unwrap().set_pixel_format(pixel_format);

        // Enable blending for shadows and anti-aliasing
        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_blending_enabled(true);
        attachment.set_rgb_blend_operation(MTLBlendOperation::Add);
        attachment.set_alpha_blend_operation(MTLBlendOperation::Add);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device.new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        let max_windows = MAX_WINDOWS;
        let max_vertices = max_windows * MAX_VERTICES_PER_WINDOW;

        // Create buffers
        let uniforms_buffer = device.new_buffer(
            mem::size_of::<CompositorUniforms>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vertices_buffer = device.new_buffer(
            (max_vertices * mem::size_of::<CompositorVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vertex_count_buffer = device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device: device.clone(),
            command_queue,
            vertex_gen_pipeline,
            render_pipeline,
            uniforms_buffer,
            vertices_buffer,
            vertex_count_buffer,
            max_windows,
        })
    }

    /// Generate vertices for all windows
    ///
    /// Call this before rendering to update the vertex buffer.
    pub fn generate_vertices(
        &self,
        windows_buffer: &Buffer,
        window_count: u32,
        screen_width: f32,
        screen_height: f32,
        focused_window: u32,
    ) {
        // Update uniforms
        let uniforms = CompositorUniforms {
            screen_width,
            screen_height,
            window_count,
            focused_window,
            ..Default::default()
        };

        let uniforms_ptr = self.uniforms_buffer.contents() as *mut CompositorUniforms;
        unsafe { *uniforms_ptr = uniforms; }

        // Reset vertex count
        let count_ptr = self.vertex_count_buffer.contents() as *mut u32;
        unsafe { *count_ptr = 0; }

        // Execute compute kernel
        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.vertex_gen_pipeline);
        encoder.set_buffer(0, Some(windows_buffer), 0);
        encoder.set_buffer(1, Some(&self.uniforms_buffer), 0);
        encoder.set_buffer(2, Some(&self.vertices_buffer), 0);
        encoder.set_buffer(3, Some(&self.vertex_count_buffer), 0);

        // One thread per window
        let thread_count = window_count.max(1) as u64;
        let threads_per_group = self.vertex_gen_pipeline.thread_execution_width();
        let thread_groups = (thread_count + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Render all windows to a texture
    pub fn render(
        &self,
        encoder: &RenderCommandEncoderRef,
        windows_buffer: &Buffer,
    ) {
        let vertex_count = unsafe {
            *(self.vertex_count_buffer.contents() as *const u32)
        };

        if vertex_count == 0 {
            return;
        }

        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&self.uniforms_buffer), 0);
        encoder.set_vertex_buffer(2, Some(windows_buffer), 0);

        encoder.set_fragment_buffer(0, Some(&self.uniforms_buffer), 0);
        encoder.set_fragment_buffer(1, Some(windows_buffer), 0);

        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
    }

    /// Get the generated vertex count
    pub fn vertex_count(&self) -> u32 {
        unsafe { *(self.vertex_count_buffer.contents() as *const u32) }
    }

    /// Get the vertices buffer for custom rendering
    pub fn vertices_buffer(&self) -> &Buffer {
        &self.vertices_buffer
    }

    /// Update uniforms for dark mode or custom themes
    pub fn set_theme(&self, uniforms: CompositorUniforms) {
        let uniforms_ptr = self.uniforms_buffer.contents() as *mut CompositorUniforms;
        unsafe { *uniforms_ptr = uniforms; }
    }
}

/// Metal shader source for compositor
const COMPOSITOR_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Window flags
constant uint WINDOW_FLAG_VISIBLE = 1 << 0;
constant uint WINDOW_FLAG_FOCUSED = 1 << 1;
constant uint WINDOW_FLAG_MINIMIZED = 1 << 2;
constant uint WINDOW_FLAG_BORDERLESS = 1 << 7;

// Window constants
constant float TITLE_BAR_HEIGHT = 28.0;
constant float BUTTON_SIZE = 12.0;
constant float BUTTON_SPACING = 8.0;
constant float BUTTON_INSET = 8.0;
constant float CORNER_RADIUS = 10.0;

// Vertex types
constant uint VERTEX_SHADOW = 0;
constant uint VERTEX_BACKGROUND = 1;
constant uint VERTEX_TITLEBAR = 2;
constant uint VERTEX_BUTTON_CLOSE = 3;
constant uint VERTEX_BUTTON_MINIMIZE = 4;
constant uint VERTEX_BUTTON_MAXIMIZE = 5;
constant uint VERTEX_BORDER = 6;
constant uint VERTEX_CONTENT = 7;  // Window content area

// Max vertices per window (2 triangles per quad, multiple quads per window)
constant uint MAX_VERTICES_PER_WINDOW = 64;

struct Window {
    float x;
    float y;
    float width;
    float height;
    uint id;
    uint z_order;
    uint app_id;
    uint flags;
    float content_x;
    float content_y;
    float content_width;
    float content_height;
    char title[64];
    float _padding[4];
};

struct CompositorUniforms {
    float screen_width;
    float screen_height;
    uint window_count;
    uint focused_window;

    float shadow_radius;
    float shadow_opacity;
    float shadow_offset_x;
    float shadow_offset_y;

    float4 titlebar_color_focused;
    float4 titlebar_color_unfocused;

    float4 close_button_color;
    float4 minimize_button_color;
    float4 maximize_button_color;

    float4 border_color;
    float4 window_background;
};

struct CompositorVertex {
    float2 position;
    float2 tex_coord;
    float4 color;
    uint window_index;
    uint vertex_type;
    float2 _pad;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 tex_coord;
    uint vertex_type;
    uint window_index;
};

// Helper to add a quad (2 triangles, 6 vertices)
void add_quad(
    device CompositorVertex* vertices,
    device atomic_uint* vertex_count,
    float x, float y, float w, float h,
    float4 color,
    uint window_index,
    uint vertex_type,
    float screen_width,
    float screen_height
) {
    uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);

    // Convert to normalized device coordinates
    float x0 = (x / screen_width) * 2.0 - 1.0;
    float y0 = 1.0 - (y / screen_height) * 2.0;
    float x1 = ((x + w) / screen_width) * 2.0 - 1.0;
    float y1 = 1.0 - ((y + h) / screen_height) * 2.0;

    // Triangle 1
    vertices[base + 0] = CompositorVertex{float2(x0, y0), float2(0, 0), color, window_index, vertex_type, float2(0)};
    vertices[base + 1] = CompositorVertex{float2(x1, y0), float2(1, 0), color, window_index, vertex_type, float2(0)};
    vertices[base + 2] = CompositorVertex{float2(x0, y1), float2(0, 1), color, window_index, vertex_type, float2(0)};

    // Triangle 2
    vertices[base + 3] = CompositorVertex{float2(x1, y0), float2(1, 0), color, window_index, vertex_type, float2(0)};
    vertices[base + 4] = CompositorVertex{float2(x1, y1), float2(1, 1), color, window_index, vertex_type, float2(0)};
    vertices[base + 5] = CompositorVertex{float2(x0, y1), float2(0, 1), color, window_index, vertex_type, float2(0)};
}

// Generate vertices for a single window
kernel void generate_compositor_vertices(
    device const Window* windows [[buffer(0)]],
    device const CompositorUniforms& uniforms [[buffer(1)]],
    device CompositorVertex* vertices [[buffer(2)]],
    device atomic_uint* vertex_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uniforms.window_count) return;

    device const Window& win = windows[tid];

    // Skip invisible/minimized windows
    if ((win.flags & WINDOW_FLAG_VISIBLE) == 0 || (win.flags & WINDOW_FLAG_MINIMIZED) != 0) {
        return;
    }

    bool is_focused = (win.flags & WINDOW_FLAG_FOCUSED) != 0;
    float sw = uniforms.screen_width;
    float sh = uniforms.screen_height;

    // 1. Shadow (slightly larger and offset)
    float shadow_x = win.x + uniforms.shadow_offset_x - uniforms.shadow_radius;
    float shadow_y = win.y + uniforms.shadow_offset_y - uniforms.shadow_radius;
    float shadow_w = win.width + uniforms.shadow_radius * 2.0;
    float shadow_h = win.height + uniforms.shadow_radius * 2.0;
    float4 shadow_color = float4(0.0, 0.0, 0.0, uniforms.shadow_opacity * (is_focused ? 1.0 : 0.5));

    add_quad(vertices, vertex_count, shadow_x, shadow_y, shadow_w, shadow_h,
             shadow_color, tid, VERTEX_SHADOW, sw, sh);

    // 2. Window background
    add_quad(vertices, vertex_count, win.x, win.y, win.width, win.height,
             uniforms.window_background, tid, VERTEX_BACKGROUND, sw, sh);

    // Skip chrome for borderless windows
    if ((win.flags & WINDOW_FLAG_BORDERLESS) != 0) {
        // For borderless, content fills whole window
        add_quad(vertices, vertex_count, win.x, win.y, win.width, win.height,
                 float4(1.0), tid, VERTEX_CONTENT, sw, sh);
        return;
    }

    // 3. Content area (below title bar)
    float content_y = win.y + TITLE_BAR_HEIGHT;
    float content_h = win.height - TITLE_BAR_HEIGHT;
    if (content_h > 0) {
        add_quad(vertices, vertex_count, win.x, content_y, win.width, content_h,
                 float4(1.0), tid, VERTEX_CONTENT, sw, sh);
    }

    // 4. Title bar
    float4 titlebar_color = is_focused ? uniforms.titlebar_color_focused : uniforms.titlebar_color_unfocused;
    add_quad(vertices, vertex_count, win.x, win.y, win.width, TITLE_BAR_HEIGHT,
             titlebar_color, tid, VERTEX_TITLEBAR, sw, sh);

    // 4. Traffic light buttons
    float button_y = win.y + (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0;

    // Close button (red)
    float close_x = win.x + BUTTON_INSET;
    add_quad(vertices, vertex_count, close_x, button_y, BUTTON_SIZE, BUTTON_SIZE,
             uniforms.close_button_color, tid, VERTEX_BUTTON_CLOSE, sw, sh);

    // Minimize button (yellow)
    float min_x = win.x + BUTTON_INSET + BUTTON_SIZE + BUTTON_SPACING;
    add_quad(vertices, vertex_count, min_x, button_y, BUTTON_SIZE, BUTTON_SIZE,
             uniforms.minimize_button_color, tid, VERTEX_BUTTON_MINIMIZE, sw, sh);

    // Maximize button (green)
    float max_x = win.x + BUTTON_INSET + 2.0 * (BUTTON_SIZE + BUTTON_SPACING);
    add_quad(vertices, vertex_count, max_x, button_y, BUTTON_SIZE, BUTTON_SIZE,
             uniforms.maximize_button_color, tid, VERTEX_BUTTON_MAXIMIZE, sw, sh);

    // 5. Border (thin line around window)
    // Top border
    add_quad(vertices, vertex_count, win.x, win.y, win.width, 1.0,
             uniforms.border_color, tid, VERTEX_BORDER, sw, sh);
    // Bottom border
    add_quad(vertices, vertex_count, win.x, win.y + win.height - 1.0, win.width, 1.0,
             uniforms.border_color, tid, VERTEX_BORDER, sw, sh);
    // Left border
    add_quad(vertices, vertex_count, win.x, win.y, 1.0, win.height,
             uniforms.border_color, tid, VERTEX_BORDER, sw, sh);
    // Right border
    add_quad(vertices, vertex_count, win.x + win.width - 1.0, win.y, 1.0, win.height,
             uniforms.border_color, tid, VERTEX_BORDER, sw, sh);
}

// Vertex shader
vertex VertexOut compositor_vertex(
    device const CompositorVertex* vertices [[buffer(0)]],
    device const CompositorUniforms& uniforms [[buffer(1)]],
    device const Window* windows [[buffer(2)]],
    uint vid [[vertex_id]]
) {
    device const CompositorVertex& v = vertices[vid];

    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.color = v.color;
    out.tex_coord = v.tex_coord;
    out.vertex_type = v.vertex_type;
    out.window_index = v.window_index;

    return out;
}

// Smooth shadow function
float shadow_alpha(float2 uv, float radius) {
    // Distance from center (0.5, 0.5)
    float2 center = float2(0.5);
    float2 p = uv - center;

    // Create smooth falloff from edges
    float edge_x = 1.0 - smoothstep(0.0, 0.3, abs(p.x) - 0.2);
    float edge_y = 1.0 - smoothstep(0.0, 0.3, abs(p.y) - 0.2);

    return edge_x * edge_y;
}

// Circle SDF for buttons
float circle_sdf(float2 uv) {
    return length(uv - 0.5) - 0.5;
}

// Fragment shader
fragment float4 compositor_fragment(
    VertexOut in [[stage_in]],
    device const CompositorUniforms& uniforms [[buffer(0)]],
    device const Window* windows [[buffer(1)]]
) {
    float4 color = in.color;

    switch (in.vertex_type) {
        case VERTEX_SHADOW: {
            // Soft shadow with falloff
            float alpha = shadow_alpha(in.tex_coord, uniforms.shadow_radius);
            color.a *= alpha;
            break;
        }

        case VERTEX_BUTTON_CLOSE:
        case VERTEX_BUTTON_MINIMIZE:
        case VERTEX_BUTTON_MAXIMIZE: {
            // Circular buttons with anti-aliasing
            float d = circle_sdf(in.tex_coord);
            float aa = fwidth(d);
            float alpha = 1.0 - smoothstep(-aa, aa, d);
            color.a *= alpha;
            break;
        }

        case VERTEX_CONTENT: {
            // Window content area - light background for now
            // TODO: Sample from window content texture
            color = float4(0.98, 0.98, 0.98, 1.0);
            break;
        }

        case VERTEX_BACKGROUND:
        case VERTEX_TITLEBAR:
        case VERTEX_BORDER:
        default:
            // Use color as-is
            break;
    }

    return color;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositor_creation() {
        let device = Device::system_default().expect("No Metal device");
        let compositor = GpuCompositor::new(&device, MTLPixelFormat::BGRA8Unorm);
        assert!(compositor.is_ok());
    }

    #[test]
    fn test_vertex_generation() {
        let device = Device::system_default().expect("No Metal device");
        let compositor = GpuCompositor::new(&device, MTLPixelFormat::BGRA8Unorm)
            .expect("Failed to create compositor");

        // Create test windows
        let windows_buffer = device.new_buffer(
            (2 * mem::size_of::<Window>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let ptr = windows_buffer.contents() as *mut Window;
        unsafe {
            *ptr = Window::new(1, "Window 1", 100.0, 100.0, 400.0, 300.0);
            (*ptr).z_order = 0;
            *ptr.add(1) = Window::new(2, "Window 2", 200.0, 200.0, 400.0, 300.0);
            (*ptr.add(1)).z_order = 1;
            (*ptr.add(1)).flags |= WINDOW_FLAG_FOCUSED;
        }

        compositor.generate_vertices(&windows_buffer, 2, 1920.0, 1080.0, 2);

        let count = compositor.vertex_count();
        // Each visible window generates: shadow(6) + background(6) + titlebar(6) + 3 buttons(18) + 4 borders(24) = 60 vertices
        // 2 windows = 120 vertices minimum
        assert!(count >= 60, "Expected at least 60 vertices, got {}", count);
    }
}
