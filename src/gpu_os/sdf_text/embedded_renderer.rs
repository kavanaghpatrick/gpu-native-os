// GPU-Native Embedded SDF Text Renderer
//
// Uses pre-baked atlas data from atlas_data.rs
// All rendering is 100% GPU - no CPU work per frame.

use metal::*;
use super::atlas_data::*;

/// SDF text vertex for GPU rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SdfTextVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// Layout parameters for compute shader
#[repr(C)]
#[derive(Copy, Clone)]
struct SdfLayoutParams {
    start_pos: [f32; 2],
    font_size: f32,
    base_size: f32,
    color: [f32; 4],
    char_count: u32,
    screen_width: f32,
    screen_height: f32,
    _padding: u32,
}

const SDF_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SdfTextVertex {
    float2 position;
    float2 uv;
    float4 color;
};

struct GlyphMetric {
    float advance;
    float _pad1;
    float _pad2;
    float _pad3;
    float4 bounds;  // x_min, y_min, x_max, y_max in font units (16-byte aligned)
    uint atlas_x;
    uint atlas_y;
    uint _pad4;
    uint _pad5;
};

struct SdfLayoutParams {
    float2 start_pos;
    float font_size;
    float base_size;
    float4 color;
    uint char_count;
    float screen_width;
    float screen_height;
    uint _padding;
};

// Constants matching atlas_data.rs
constant uint SDF_SIZE = 48;
constant uint ATLAS_WIDTH = 500;
constant uint ATLAS_HEIGHT = 500;
constant float UNITS_PER_EM = 2048.0;
constant float SPREAD = 8.0;

// Compute kernel: generate 6 vertices per character
kernel void sdf_text_layout(
    device SdfTextVertex* vertices [[buffer(0)]],
    constant GlyphMetric* metrics [[buffer(1)]],
    constant uchar* text [[buffer(2)]],
    constant SdfLayoutParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.char_count) return;

    // Get character and map to glyph index (ASCII 32-126)
    uchar c = text[tid];
    if (c < 32 || c > 126) c = 32;
    uint glyph_idx = c - 32;

    GlyphMetric m = metrics[glyph_idx];

    // Calculate scale from font units to pixels
    float scale = params.font_size / UNITS_PER_EM;

    // Calculate cursor X by summing advances of previous characters
    float cursor_x = params.start_pos.x;
    for (uint i = 0; i < tid; i++) {
        uchar prev_c = text[i];
        if (prev_c < 32 || prev_c > 126) prev_c = 32;
        cursor_x += metrics[prev_c - 32].advance * scale;
    }

    // Glyph bounding box in font units
    float glyph_width = m.bounds.z - m.bounds.x;
    float glyph_height = m.bounds.w - m.bounds.y;

    // Position and size in pixels
    float x = cursor_x + m.bounds.x * scale;
    float y = params.start_pos.y - m.bounds.w * scale;  // Y from baseline down
    float w = glyph_width * scale;
    float h = glyph_height * scale;

    // Skip space character (ASCII 32) and empty glyphs
    if (c == 32 || glyph_width <= 0 || glyph_height <= 0) {
        uint base = tid * 6;
        for (uint i = 0; i < 6; i++) {
            vertices[base + i] = SdfTextVertex{float2(0,0), float2(0,0), float4(0,0,0,0)};
        }
        return;
    }

    // UV coordinates in atlas
    float u0 = float(m.atlas_x) / float(ATLAS_WIDTH);
    float v0 = float(m.atlas_y) / float(ATLAS_HEIGHT);
    float u1 = float(m.atlas_x + SDF_SIZE) / float(ATLAS_WIDTH);
    float v1 = float(m.atlas_y + SDF_SIZE) / float(ATLAS_HEIGHT);

    // Generate 6 vertices (2 triangles) for quad
    // UV coords: flip both U and V to correct 180° rotation from atlas generation
    uint base = tid * 6;

    // Triangle 1: TL, BL, BR (with UVs flipped: u1->u0, v1->v0 mapping)
    vertices[base + 0] = SdfTextVertex{float2(x, y), float2(u0, v1), params.color};
    vertices[base + 1] = SdfTextVertex{float2(x, y + h), float2(u0, v0), params.color};
    vertices[base + 2] = SdfTextVertex{float2(x + w, y + h), float2(u1, v0), params.color};

    // Triangle 2: TL, BR, TR
    vertices[base + 3] = SdfTextVertex{float2(x, y), float2(u0, v1), params.color};
    vertices[base + 4] = SdfTextVertex{float2(x + w, y + h), float2(u1, v0), params.color};
    vertices[base + 5] = SdfTextVertex{float2(x + w, y), float2(u1, v1), params.color};
}

// Vertex shader
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut sdf_text_vertex(
    const device SdfTextVertex* vertices [[buffer(0)]],
    constant float2& screen_size [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    SdfTextVertex v = vertices[vid];
    VertexOut out;

    // Convert pixel coords to clip space (-1 to 1)
    out.position = float4(
        (v.position.x / screen_size.x) * 2.0 - 1.0,
        1.0 - (v.position.y / screen_size.y) * 2.0,
        0.0,
        1.0
    );
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

// Fragment shader with SDF anti-aliasing
fragment float4 sdf_text_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    float distance = atlas.sample(s, in.uv).r;

    // SDF: 0.5 = edge, <0.5 = inside, >0.5 = outside
    float edge = 0.5;
    float width = fwidth(distance) * 0.75;
    float alpha = smoothstep(edge - width, edge + width, distance);

    // Invert because our SDF has inside < 0.5
    alpha = 1.0 - alpha;

    return float4(in.color.rgb, in.color.a * alpha);
}
"#;

/// GPU-Native Embedded SDF Text Renderer
pub struct EmbeddedSdfRenderer {
    layout_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,
    atlas_texture: Texture,
    metrics_buffer: Buffer,
    params_buffer: Buffer,
    text_buffer: Buffer,
    screen_size_buffer: Buffer,
    max_chars: usize,
}

impl EmbeddedSdfRenderer {
    /// Create a new embedded SDF renderer
    /// Loads the pre-baked atlas data into GPU memory
    pub fn new(device: &Device, max_chars: usize) -> Result<Self, String> {
        // Compile shaders
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SDF_SHADER, &options)
            .map_err(|e| format!("Failed to compile SDF shader: {}", e))?;

        // Create compute pipeline for text layout
        let layout_fn = library
            .get_function("sdf_text_layout", None)
            .map_err(|e| format!("Failed to get sdf_text_layout: {}", e))?;

        let layout_pipeline = device
            .new_compute_pipeline_state_with_function(&layout_fn)
            .map_err(|e| format!("Failed to create layout pipeline: {}", e))?;

        // Create render pipeline
        let vertex_fn = library
            .get_function("sdf_text_vertex", None)
            .map_err(|e| format!("Failed to get sdf_text_vertex: {}", e))?;

        let fragment_fn = library
            .get_function("sdf_text_fragment", None)
            .map_err(|e| format!("Failed to get sdf_text_fragment: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));

        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        // Create atlas texture from embedded data
        let texture_desc = TextureDescriptor::new();
        texture_desc.set_width(ATLAS_WIDTH as u64);
        texture_desc.set_height(ATLAS_HEIGHT as u64);
        texture_desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        texture_desc.set_texture_type(MTLTextureType::D2);
        texture_desc.set_usage(MTLTextureUsage::ShaderRead);
        texture_desc.set_storage_mode(MTLStorageMode::Shared);

        let atlas_texture = device.new_texture(&texture_desc);

        // Upload embedded atlas data to texture
        let region = MTLRegion::new_2d(0, 0, ATLAS_WIDTH as u64, ATLAS_HEIGHT as u64);
        atlas_texture.replace_region(
            region,
            0,
            ATLAS_DATA.as_ptr() as *const _,
            ATLAS_WIDTH as u64,
        );

        // Create metrics buffer from embedded data
        // Need to convert GlyphMetric to GPU-compatible format
        // Must match Metal struct layout with float4 alignment
        #[repr(C)]
        struct GpuGlyphMetric {
            advance: f32,
            _pad1: f32,
            _pad2: f32,
            _pad3: f32,
            bounds: [f32; 4],  // float4 needs 16-byte alignment
            atlas_x: u32,
            atlas_y: u32,
            _pad4: u32,
            _pad5: u32,
        }

        let gpu_metrics: Vec<GpuGlyphMetric> = GLYPH_METRICS
            .iter()
            .map(|m| GpuGlyphMetric {
                advance: m.advance,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
                bounds: m.bounds,
                atlas_x: m.atlas_x,
                atlas_y: m.atlas_y,
                _pad4: 0,
                _pad5: 0,
            })
            .collect();

        let metrics_buffer = device.new_buffer_with_data(
            gpu_metrics.as_ptr() as *const _,
            (gpu_metrics.len() * std::mem::size_of::<GpuGlyphMetric>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create parameter buffers
        let params_buffer = device.new_buffer(
            std::mem::size_of::<SdfLayoutParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let text_buffer = device.new_buffer(
            max_chars as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let screen_size_buffer = device.new_buffer(
            8, // 2 floats
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            layout_pipeline,
            render_pipeline,
            atlas_texture,
            metrics_buffer,
            params_buffer,
            text_buffer,
            screen_size_buffer,
            max_chars,
        })
    }

    /// Get the atlas texture (for debugging/visualization)
    pub fn atlas_texture(&self) -> &Texture {
        &self.atlas_texture
    }

    /// Measure text width in pixels
    pub fn measure_text(&self, text: &str, font_size: f32) -> f32 {
        let scale = font_size / UNITS_PER_EM;
        let mut width = 0.0;

        for c in text.chars() {
            let glyph_idx = if c as u32 >= 32 && c as u32 <= 126 {
                (c as u32 - 32) as usize
            } else {
                0 // space
            };

            if glyph_idx < GLYPH_METRICS.len() {
                width += GLYPH_METRICS[glyph_idx].advance * scale;
            }
        }

        width
    }

    /// Layout text using GPU compute shader
    /// Returns the number of vertices generated (6 per character)
    /// vertex_offset: byte offset into output_vertices buffer for this text
    pub fn layout_text(
        &self,
        encoder: &ComputeCommandEncoderRef,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: [f32; 4],
        screen_width: f32,
        screen_height: f32,
        output_vertices: &Buffer,
        vertex_offset: u64,
    ) -> usize {
        let char_count = text.len().min(self.max_chars);
        if char_count == 0 {
            return 0;
        }

        // Copy text to buffer
        unsafe {
            let ptr = self.text_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(text.as_ptr(), ptr, char_count);
        }

        // Update params
        unsafe {
            let ptr = self.params_buffer.contents() as *mut SdfLayoutParams;
            *ptr = SdfLayoutParams {
                start_pos: [x, y],
                font_size,
                base_size: UNITS_PER_EM,
                color,
                char_count: char_count as u32,
                screen_width,
                screen_height,
                _padding: 0,
            };
        }

        encoder.set_compute_pipeline_state(&self.layout_pipeline);
        encoder.set_buffer(0, Some(output_vertices), vertex_offset);
        encoder.set_buffer(1, Some(&self.metrics_buffer), 0);
        encoder.set_buffer(2, Some(&self.text_buffer), 0);
        encoder.set_buffer(3, Some(&self.params_buffer), 0);

        // Dispatch one thread per character
        let threads = char_count;
        let max_threads = self.layout_pipeline.max_total_threads_per_threadgroup() as usize;
        let threads_per_group = threads.min(max_threads);
        let thread_groups = (threads + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );

        char_count * 6
    }

    /// Render text vertices
    pub fn render(
        &self,
        encoder: &RenderCommandEncoderRef,
        vertices: &Buffer,
        vertex_count: usize,
        screen_width: f32,
        screen_height: f32,
    ) {
        if vertex_count == 0 {
            return;
        }

        // Update screen size buffer
        unsafe {
            let ptr = self.screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = [screen_width, screen_height];
        }

        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(vertices), 0);
        encoder.set_vertex_buffer(1, Some(&self.screen_size_buffer), 0);
        encoder.set_fragment_texture(0, Some(&self.atlas_texture));
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
    }

    /// Convenience method: layout and render text in one call
    /// Requires both compute and render encoders
    pub fn draw_text(
        &self,
        compute_encoder: &ComputeCommandEncoderRef,
        render_encoder: &RenderCommandEncoderRef,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: [f32; 4],
        screen_width: f32,
        screen_height: f32,
        vertex_buffer: &Buffer,
    ) {
        let vertex_count = self.layout_text(
            compute_encoder,
            text,
            x,
            y,
            font_size,
            color,
            screen_width,
            screen_height,
            vertex_buffer,
            0,  // Start at offset 0 for single text draw
        );

        self.render(
            render_encoder,
            vertex_buffer,
            vertex_count,
            screen_width,
            screen_height,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_text_vertex_size() {
        assert_eq!(std::mem::size_of::<SdfTextVertex>(), 32);
    }

    #[test]
    fn test_atlas_data_loaded() {
        // Verify the embedded atlas data is accessible
        assert_eq!(ATLAS_WIDTH, 500);
        assert_eq!(ATLAS_HEIGHT, 500);
        assert_eq!(GLYPH_METRICS.len(), 95);
        assert_eq!(ATLAS_DATA.len(), 250000);
    }

    #[test]
    fn test_measure_text() {
        // Create a mock measurement without GPU
        let scale = 24.0 / UNITS_PER_EM;
        let hello_width: f32 = "Hello".chars().map(|c| {
            let idx = (c as u32 - 32) as usize;
            GLYPH_METRICS[idx].advance * scale
        }).sum();

        assert!(hello_width > 0.0);
    }
}
