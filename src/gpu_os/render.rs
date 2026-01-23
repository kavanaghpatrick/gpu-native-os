// Issue #17: Hybrid Rendering - Compute Logic + Fragment Pixels
//
// Hybrid rendering pipeline: compute shaders for OS logic,
// fragment shaders for pixel output (leveraging Apple's TBDR).

#[allow(unused_imports)]
use super::memory::{DrawArguments, WidgetCompact};
use metal::*;
use std::mem;
use std::time::Instant;

/// Widget vertex for fragment shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct WidgetVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub bounds: [f32; 4],
    pub corner_radius: f32,
    pub border_width: f32,
    pub _padding: [f32; 2],
}

impl WidgetVertex {
    pub const SIZE: usize = 64;
}

const RENDER_SHADER: &str = r#"
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

struct WidgetVertex {
    float2 position;
    float2 uv;
    float4 color;
    float4 bounds;
    float corner_radius;
    float border_width;
    float2 _padding;
};

struct VertexGenParams {
    uint widget_count;
    uint _padding[3];
};

inline float f16_to_float(ushort bits) {
    return float(as_type<half>(bits));
}

inline float4 unpack_color_rgb565(uint packed, uint offset) {
    ushort bits = ushort((packed >> offset) & 0xFFFF);
    float r = float((bits >> 11) & 0x1F) / 31.0;
    float g = float((bits >> 5) & 0x3F) / 63.0;
    float b = float(bits & 0x1F) / 31.0;
    return float4(r, g, b, 1.0);
}

// Compute kernel to generate vertices from widgets
kernel void vertex_gen_kernel(
    device WidgetCompact* widgets [[buffer(0)]],
    device WidgetVertex* vertices [[buffer(1)]],
    device atomic_uint* draw_args [[buffer(2)]],
    constant VertexGenParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.widget_count) return;

    // Initialize draw args on thread 0
    if (tid == 0) {
        atomic_store_explicit(&draw_args[0], params.widget_count * 6, memory_order_relaxed); // vertex_count
        atomic_store_explicit(&draw_args[1], 1, memory_order_relaxed); // instance_count
        atomic_store_explicit(&draw_args[2], 0, memory_order_relaxed); // vertex_start
        atomic_store_explicit(&draw_args[3], 0, memory_order_relaxed); // base_instance
    }

    WidgetCompact w = widgets[tid];

    // Extract bounds (f16 packed)
    float x = f16_to_float(w.bounds.x);
    float y = f16_to_float(w.bounds.y);
    float width = f16_to_float(w.bounds.z);
    float height = f16_to_float(w.bounds.w);

    // Extract color (RGB565) - background is in high 16 bits
    float4 bg_color = unpack_color_rgb565(w.packed_colors, 16);

    // Extract style
    float corner_radius = float((w.packed_style >> 8) & 0xF) * 4.0;
    float border_width = float((w.packed_style >> 12) & 0xF);

    // Generate 6 vertices (2 triangles) for this widget
    uint base = tid * 6;

    float4 bounds = float4(x, y, width, height);

    // Shared vertex data
    WidgetVertex v;
    v.color = bg_color;
    v.bounds = bounds;
    v.corner_radius = corner_radius;
    v.border_width = border_width;
    v._padding = float2(0.0);

    // Triangle 1: top-left, top-right, bottom-left
    v.position = float2(x, y);
    v.uv = float2(0.0, 0.0);
    vertices[base + 0] = v;

    v.position = float2(x + width, y);
    v.uv = float2(1.0, 0.0);
    vertices[base + 1] = v;

    v.position = float2(x, y + height);
    v.uv = float2(0.0, 1.0);
    vertices[base + 2] = v;

    // Triangle 2: top-right, bottom-right, bottom-left
    v.position = float2(x + width, y);
    v.uv = float2(1.0, 0.0);
    vertices[base + 3] = v;

    v.position = float2(x + width, y + height);
    v.uv = float2(1.0, 1.0);
    vertices[base + 4] = v;

    v.position = float2(x, y + height);
    v.uv = float2(0.0, 1.0);
    vertices[base + 5] = v;
}

// Vertex shader for widget rendering
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
    float4 bounds;
    float corner_radius;
    float border_width;
};

vertex VertexOut widget_vertex(
    const device WidgetVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    WidgetVertex v = vertices[vid];
    VertexOut out;
    // Convert normalized coords to clip space
    out.position = float4(v.position.x * 2.0 - 1.0, 1.0 - v.position.y * 2.0, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    out.bounds = v.bounds;
    out.corner_radius = v.corner_radius;
    out.border_width = v.border_width;
    return out;
}

// Fragment shader with rounded corners
fragment float4 widget_fragment(VertexOut in [[stage_in]]) {
    // Use UV-space SDF (0-1 range) for rounded rectangle
    // This works regardless of actual widget size
    float2 uv = in.uv;
    float2 size = in.bounds.zw;  // Widget size for aspect ratio

    // Normalize corner radius relative to widget size
    float r = in.corner_radius / min(size.x, size.y);
    r = min(r, 0.5);  // Cap at 50% to avoid artifacts

    // SDF in UV space (0-1)
    float2 p = abs(uv - 0.5);  // Distance from center
    float2 q = p - float2(0.5 - r);
    float d = length(max(q, 0.0)) - r;

    // Anti-aliased edge (scale factor based on widget size for consistent AA)
    float aa = 2.0 / max(size.x, size.y);
    float edge = 1.0 - smoothstep(-aa, aa, d);

    // Simple color output with alpha for rounded corners
    return float4(in.color.rgb, in.color.a * edge);
}
"#;

/// Vertex generation parameters
#[repr(C)]
struct VertexGenParams {
    widget_count: u32,
    _padding: [u32; 3],
}

/// Renderer that handles the hybrid compute+fragment pipeline
pub struct HybridRenderer {
    vertex_gen_pipeline: ComputePipelineState,
    widget_render_pipeline: RenderPipelineState,
    params_buffer: Buffer,
}

impl HybridRenderer {
    /// Create a new hybrid renderer
    pub fn new(device: &Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(RENDER_SHADER, &options)
            .map_err(|e| format!("Failed to compile render shader: {}", e))?;

        // Create compute pipeline for vertex generation
        let vertex_gen_fn = library
            .get_function("vertex_gen_kernel", None)
            .map_err(|e| format!("Failed to get vertex_gen_kernel: {}", e))?;

        let vertex_gen_pipeline = device
            .new_compute_pipeline_state_with_function(&vertex_gen_fn)
            .map_err(|e| format!("Failed to create vertex gen pipeline: {}", e))?;

        // Create render pipeline for widget rendering
        let vertex_fn = library
            .get_function("widget_vertex", None)
            .map_err(|e| format!("Failed to get widget_vertex: {}", e))?;

        let fragment_fn = library
            .get_function("widget_fragment", None)
            .map_err(|e| format!("Failed to get widget_fragment: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        // Enable alpha blending
        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let widget_render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create widget render pipeline: {}", e))?;

        let params_buffer = device.new_buffer(
            mem::size_of::<VertexGenParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            vertex_gen_pipeline,
            widget_render_pipeline,
            params_buffer,
        })
    }

    /// Generate vertices from widget state (compute pass)
    pub fn generate_vertices(
        &self,
        encoder: &ComputeCommandEncoderRef,
        widgets: &Buffer,
        widget_count: usize,
        output_vertices: &Buffer,
        draw_args: &Buffer,
    ) {
        if widget_count == 0 {
            return;
        }

        // Update params
        unsafe {
            let ptr = self.params_buffer.contents() as *mut VertexGenParams;
            *ptr = VertexGenParams {
                widget_count: widget_count as u32,
                _padding: [0; 3],
            };
        }

        encoder.set_compute_pipeline_state(&self.vertex_gen_pipeline);
        encoder.set_buffer(0, Some(widgets), 0);
        encoder.set_buffer(1, Some(output_vertices), 0);
        encoder.set_buffer(2, Some(draw_args), 0);
        encoder.set_buffer(3, Some(&self.params_buffer), 0);

        let threads = widget_count;
        let max_threads = self.vertex_gen_pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_group = threads.min(max_threads);
        let thread_groups = (threads + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
    }

    /// Generate vertices synchronously (for testing)
    pub fn generate_vertices_sync(
        &self,
        queue: &CommandQueue,
        widgets: &Buffer,
        widget_count: usize,
        output_vertices: &Buffer,
        draw_args: &Buffer,
    ) -> (DrawArguments, f64) {
        let start = Instant::now();

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.generate_vertices(&encoder, widgets, widget_count, output_vertices, draw_args);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Read draw args
        let args = unsafe { *(draw_args.contents() as *const DrawArguments) };

        (args, time_ms)
    }

    /// Render widgets using indirect draw (render pass)
    pub fn render_widgets(
        &self,
        encoder: &RenderCommandEncoderRef,
        vertices: &Buffer,
        draw_args: &Buffer,
    ) {
        encoder.set_render_pipeline_state(&self.widget_render_pipeline);
        encoder.set_vertex_buffer(0, Some(vertices), 0);

        // Use indirect draw for GPU-driven rendering
        encoder.draw_primitives_indirect(MTLPrimitiveType::Triangle, draw_args, 0);
    }

    /// Get the vertex buffer size needed for N widgets
    pub fn vertex_buffer_size(widget_count: usize) -> usize {
        widget_count * 6 * WidgetVertex::SIZE  // 6 vertices per widget
    }
}

/// Full frame renderer combining all rendering stages
pub struct FrameRenderer {
    hybrid: HybridRenderer,
    #[allow(dead_code)]
    text: super::text::TextRenderer,  // For future text overlay integration
    vertex_buffer: Buffer,
}

impl FrameRenderer {
    /// Create a new frame renderer
    pub fn new(device: &Device) -> Result<Self, String> {
        let hybrid = HybridRenderer::new(device)?;
        let text = super::text::TextRenderer::new(device)?;

        // Create vertex buffer for 1024 widgets
        let vertex_buffer = device.new_buffer(
            HybridRenderer::vertex_buffer_size(1024) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            hybrid,
            text,
            vertex_buffer,
        })
    }

    /// Render a complete frame
    pub fn render_frame(
        &self,
        command_buffer: &CommandBufferRef,
        drawable: &MetalDrawableRef,
        memory: &super::memory::GpuMemory,
        _font_atlas: &super::text::FontAtlas,
    ) {
        // Compute pass: generate vertices
        let compute_encoder = command_buffer.new_compute_command_encoder();
        self.hybrid.generate_vertices(
            &compute_encoder,
            &memory.widget_buffer,
            memory.widget_count(),
            &self.vertex_buffer,
            &memory.draw_args_buffer,
        );
        compute_encoder.end_encoding();

        // Render pass: draw widgets
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.1, 1.0));

        let render_encoder = command_buffer.new_render_command_encoder(&render_desc);
        self.hybrid.render_widgets(&render_encoder, &self.vertex_buffer, &memory.draw_args_buffer);
        render_encoder.end_encoding();

        command_buffer.present_drawable(drawable);
    }
}

/// Render performance statistics
#[derive(Debug, Default)]
pub struct RenderStats {
    pub compute_time_ms: f64,
    pub render_time_ms: f64,
    pub total_time_ms: f64,
    pub vertex_count: usize,
}

impl FrameRenderer {
    /// Render frame and collect stats (for testing/profiling)
    pub fn render_frame_with_stats(
        &self,
        queue: &CommandQueue,
        memory: &super::memory::GpuMemory,
        _font_atlas: &super::text::FontAtlas,
    ) -> RenderStats {
        let total_start = Instant::now();

        // Compute pass timing
        let compute_start = Instant::now();
        let command_buffer = queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        self.hybrid.generate_vertices(
            &compute_encoder,
            &memory.widget_buffer,
            memory.widget_count(),
            &self.vertex_buffer,
            &memory.draw_args_buffer,
        );
        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let compute_time_ms = compute_start.elapsed().as_secs_f64() * 1000.0;

        // For stats without actual render pass (no drawable), just measure compute
        // In real usage, render pass would add ~0.1-0.5ms
        let render_time_ms = 0.0;

        let total_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        // Read vertex count from draw args
        let draw_args = unsafe { *(memory.draw_args_buffer.contents() as *const DrawArguments) };

        RenderStats {
            compute_time_ms,
            render_time_ms,
            total_time_ms,
            vertex_count: draw_args.vertex_count as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_vertex_size() {
        assert_eq!(mem::size_of::<WidgetVertex>(), WidgetVertex::SIZE);
    }
}
