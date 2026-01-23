// Issue #16: Text Rendering - MSDF Font Atlas
//
// GPU text rendering using Multi-channel Signed Distance Fields (MSDF)
// for resolution-independent, sharp text at any size.

use metal::*;
use std::collections::HashMap;
use std::time::Instant;

/// Glyph metrics for a single character
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GlyphMetrics {
    /// Top-left UV coordinate in atlas
    pub atlas_uv: [f32; 2],
    /// Size in UV space
    pub atlas_size: [f32; 2],
    /// Horizontal advance to next character
    pub advance: f32,
    /// Offset from baseline
    pub bearing: [f32; 2],
    /// Glyph size in pixels (at base font size)
    pub size: [f32; 2],
    pub _padding: f32,
}

/// Text vertex for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct TextVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// Font atlas containing MSDF texture and glyph metrics
pub struct FontAtlas {
    texture: Texture,
    metrics: Buffer,
    glyph_map: HashMap<char, GlyphMetrics>,
}

impl FontAtlas {
    /// Load a font atlas from MSDF atlas image and metrics
    pub fn load(_device: &Device, _atlas_path: &str, _metrics_path: &str) -> Result<Self, String> {
        // For now, use create_default instead
        Err("FontAtlas::load not yet implemented - use create_default()".to_string())
    }

    /// Create a default ASCII font atlas (for testing)
    pub fn create_default(device: &Device) -> Result<Self, String> {
        // Create a simple 16x16 atlas texture with placeholders for ASCII chars
        let texture_desc = TextureDescriptor::new();
        texture_desc.set_width(256);
        texture_desc.set_height(256);
        texture_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        texture_desc.set_texture_type(MTLTextureType::D2);
        texture_desc.set_usage(MTLTextureUsage::ShaderRead);
        texture_desc.set_storage_mode(MTLStorageMode::Shared);

        let texture = device.new_texture(&texture_desc);

        // Generate glyph metrics for printable ASCII (32-126)
        let mut glyph_map = HashMap::new();
        let glyph_count = 95; // Printable ASCII characters

        // Base font size is 16px, each glyph is roughly 10x16 in a 16x16 cell
        let cell_size = 16.0;
        let atlas_size = 256.0;
        let chars_per_row = 16;

        for i in 0..glyph_count {
            let c = (32 + i) as u8 as char;
            let col = i % chars_per_row;
            let row = i / chars_per_row;

            let metrics = GlyphMetrics {
                atlas_uv: [
                    (col as f32 * cell_size) / atlas_size,
                    (row as f32 * cell_size) / atlas_size,
                ],
                atlas_size: [cell_size / atlas_size, cell_size / atlas_size],
                advance: 10.0, // Fixed-width for simplicity
                bearing: [0.0, 14.0],
                size: [10.0, 16.0],
                _padding: 0.0,
            };

            glyph_map.insert(c, metrics);
        }

        // Create metrics buffer
        let metrics_data: Vec<GlyphMetrics> = (0..glyph_count)
            .map(|i| {
                let c = (32 + i) as u8 as char;
                *glyph_map.get(&c).unwrap()
            })
            .collect();

        let metrics = device.new_buffer_with_data(
            metrics_data.as_ptr() as *const _,
            (glyph_count * std::mem::size_of::<GlyphMetrics>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            texture,
            metrics,
            glyph_map,
        })
    }

    /// Get the atlas texture for binding
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// Get the metrics buffer for binding
    pub fn metrics_buffer(&self) -> &Buffer {
        &self.metrics
    }

    /// Get metrics for a specific glyph
    pub fn glyph_metrics(&self, codepoint: char) -> Option<GlyphMetrics> {
        self.glyph_map.get(&codepoint).copied()
    }
}

const TEXT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextVertex {
    float2 position;
    float2 uv;
    float4 color;
};

struct GlyphMetrics {
    float2 atlas_uv;
    float2 atlas_size;
    float advance;
    float2 bearing;
    float2 size;
    float _padding;
};

struct TextLayoutParams {
    float2 start_pos;
    float font_size;
    float base_size;
    float4 color;
    uint char_count;
    uint _padding[3];
};

// Compute kernel for text layout - generates 6 vertices per character
kernel void text_layout_kernel(
    device TextVertex* vertices [[buffer(0)]],
    constant GlyphMetrics* metrics [[buffer(1)]],
    constant uchar* text [[buffer(2)]],
    constant TextLayoutParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.char_count) return;

    // Get character and its metrics
    uchar c = text[tid];
    if (c < 32 || c > 126) c = 32; // Use space for non-printable
    uint glyph_idx = c - 32;
    GlyphMetrics m = metrics[glyph_idx];

    // Calculate position with prefix sum (simplified: fixed-width)
    float scale = params.font_size / params.base_size;
    float x = params.start_pos.x + float(tid) * m.advance * scale;
    float y = params.start_pos.y;

    // Glyph quad dimensions
    float w = m.size.x * scale;
    float h = m.size.y * scale;
    float ox = m.bearing.x * scale;
    float oy = (params.base_size - m.bearing.y) * scale;

    // Generate 6 vertices (2 triangles) for this character
    uint base = tid * 6;

    // Triangle 1: top-left, top-right, bottom-left
    vertices[base + 0] = TextVertex{
        float2(x + ox, y + oy),
        float2(m.atlas_uv.x, m.atlas_uv.y),
        params.color
    };
    vertices[base + 1] = TextVertex{
        float2(x + ox + w, y + oy),
        float2(m.atlas_uv.x + m.atlas_size.x, m.atlas_uv.y),
        params.color
    };
    vertices[base + 2] = TextVertex{
        float2(x + ox, y + oy + h),
        float2(m.atlas_uv.x, m.atlas_uv.y + m.atlas_size.y),
        params.color
    };

    // Triangle 2: top-right, bottom-right, bottom-left
    vertices[base + 3] = TextVertex{
        float2(x + ox + w, y + oy),
        float2(m.atlas_uv.x + m.atlas_size.x, m.atlas_uv.y),
        params.color
    };
    vertices[base + 4] = TextVertex{
        float2(x + ox + w, y + oy + h),
        float2(m.atlas_uv.x + m.atlas_size.x, m.atlas_uv.y + m.atlas_size.y),
        params.color
    };
    vertices[base + 5] = TextVertex{
        float2(x + ox, y + oy + h),
        float2(m.atlas_uv.x, m.atlas_uv.y + m.atlas_size.y),
        params.color
    };
}

// Vertex shader for text rendering
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut text_vertex(
    const device TextVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    TextVertex v = vertices[vid];
    VertexOut out;
    // Convert pixel coords to clip space (-1 to 1)
    out.position = float4(v.position.x * 2.0 - 1.0, 1.0 - v.position.y * 2.0, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

// Fragment shader for MSDF text rendering
fragment float4 text_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    float4 sample = atlas.sample(s, in.uv);

    // MSDF: compute signed distance from RGB channels
    float sd = max(min(sample.r, sample.g), min(max(sample.r, sample.g), sample.b));

    // Anti-aliased edge
    float edge = 0.5;
    float width = fwidth(sd) * 0.5;
    float alpha = smoothstep(edge - width, edge + width, sd);

    return float4(in.color.rgb, in.color.a * alpha);
}
"#;

/// Text layout parameters for compute shader
#[repr(C)]
struct TextLayoutParams {
    start_pos: [f32; 2],
    font_size: f32,
    base_size: f32,
    color: [f32; 4],
    char_count: u32,
    _padding: [u32; 3],
}

/// Text layout engine - generates vertices for text strings
pub struct TextRenderer {
    layout_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,
    params_buffer: Buffer,
    text_buffer: Buffer,
}

impl TextRenderer {
    /// Create a new text renderer
    pub fn new(device: &Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(TEXT_SHADER, &options)
            .map_err(|e| format!("Failed to compile text shader: {}", e))?;

        // Create compute pipeline for text layout
        let layout_fn = library
            .get_function("text_layout_kernel", None)
            .map_err(|e| format!("Failed to get text_layout_kernel: {}", e))?;

        let layout_pipeline = device
            .new_compute_pipeline_state_with_function(&layout_fn)
            .map_err(|e| format!("Failed to create text layout pipeline: {}", e))?;

        // Create render pipeline for text rendering
        let vertex_fn = library
            .get_function("text_vertex", None)
            .map_err(|e| format!("Failed to get text_vertex: {}", e))?;

        let fragment_fn = library
            .get_function("text_fragment", None)
            .map_err(|e| format!("Failed to get text_fragment: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        // Enable alpha blending for text
        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create text render pipeline: {}", e))?;

        // Create parameter buffer
        let params_buffer = device.new_buffer(
            std::mem::size_of::<TextLayoutParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create text buffer (max 4KB for text)
        let text_buffer = device.new_buffer(
            4096,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            layout_pipeline,
            render_pipeline,
            params_buffer,
            text_buffer,
        })
    }

    /// Generate vertices for a text string
    /// Returns number of vertices generated (6 per character)
    pub fn layout_text(
        &self,
        encoder: &ComputeCommandEncoderRef,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: [f32; 4],
        atlas: &FontAtlas,
        output_vertices: &Buffer,
    ) -> usize {
        let char_count = text.len().min(4096);
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
            let ptr = self.params_buffer.contents() as *mut TextLayoutParams;
            *ptr = TextLayoutParams {
                start_pos: [x, y],
                font_size,
                base_size: 16.0,
                color,
                char_count: char_count as u32,
                _padding: [0; 3],
            };
        }

        encoder.set_compute_pipeline_state(&self.layout_pipeline);
        encoder.set_buffer(0, Some(output_vertices), 0);
        encoder.set_buffer(1, Some(atlas.metrics_buffer()), 0);
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

        // Return vertex count (6 per character)
        char_count * 6
    }

    /// Layout text synchronously (for testing)
    pub fn layout_text_sync(
        &self,
        queue: &CommandQueue,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: [f32; 4],
        atlas: &FontAtlas,
        output_vertices: &Buffer,
    ) -> (usize, f64) {
        let start = Instant::now();

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let vertex_count = self.layout_text(
            &encoder,
            text,
            x, y,
            font_size,
            color,
            atlas,
            output_vertices,
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let time_ms = start.elapsed().as_secs_f64() * 1000.0;
        (vertex_count, time_ms)
    }

    /// Render text vertices (call during render pass)
    pub fn render(
        &self,
        encoder: &RenderCommandEncoderRef,
        vertices: &Buffer,
        vertex_count: usize,
        atlas: &FontAtlas,
    ) {
        if vertex_count == 0 {
            return;
        }

        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(vertices), 0);
        encoder.set_fragment_texture(0, Some(atlas.texture()));
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
    }
}

/// Text performance benchmark
#[derive(Debug)]
pub struct TextBenchmark {
    pub char_count: usize,
    pub vertex_count: usize,
    pub layout_time_ms: f64,
}

impl TextRenderer {
    /// Benchmark text layout performance
    pub fn benchmark(
        &self,
        queue: &CommandQueue,
        atlas: &FontAtlas,
        char_counts: &[usize],
        iterations: usize,
    ) -> Vec<TextBenchmark> {
        let device = queue.device();
        let mut results = Vec::new();

        for &count in char_counts {
            // Generate test string
            let text: String = (0..count)
                .map(|i| (b'A' + (i % 26) as u8) as char)
                .collect();

            // Create output buffer
            let vertex_buffer = device.new_buffer(
                (count * 6 * std::mem::size_of::<TextVertex>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Warmup
            for _ in 0..3 {
                self.layout_text_sync(
                    queue,
                    &text,
                    0.0, 0.0,
                    16.0,
                    [1.0, 1.0, 1.0, 1.0],
                    atlas,
                    &vertex_buffer,
                );
            }

            // Benchmark
            let mut total_time = 0.0;
            let mut vertex_count = 0;
            for _ in 0..iterations {
                let (vc, time) = self.layout_text_sync(
                    queue,
                    &text,
                    0.0, 0.0,
                    16.0,
                    [1.0, 1.0, 1.0, 1.0],
                    atlas,
                    &vertex_buffer,
                );
                vertex_count = vc;
                total_time += time;
            }

            results.push(TextBenchmark {
                char_count: count,
                vertex_count,
                layout_time_ms: total_time / iterations as f64,
            });
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glyph_metrics_size() {
        assert_eq!(std::mem::size_of::<GlyphMetrics>(), 40);
    }

    #[test]
    fn test_text_vertex_size() {
        assert_eq!(std::mem::size_of::<TextVertex>(), 32);
    }
}
