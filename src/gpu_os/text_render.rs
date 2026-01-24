// GPU Text Rendering - SDF Font System (GPU-Native)
//
// Zero CPU work per frame:
// - All text stored in single GPU buffer
// - Single compute dispatch for layout
// - Single render pass for drawing
//
// Usage:
//   let mut renderer = TextRenderer::new(&device, max_chars)?;
//   renderer.add_text("Hello", 10.0, 10.0, colors::WHITE);
//   renderer.render(&command_buffer, &render_pass_desc, width, height);

use metal::*;
use std::mem;
use super::sdf_text::atlas_data::{GLYPH_METRICS, UNITS_PER_EM, ATLAS_WIDTH, ATLAS_HEIGHT, SDF_SIZE, ATLAS_DATA};

/// Color constants for convenience
pub mod colors {
    pub const WHITE: u32 = 0xFFFFFFFF;
    pub const BLACK: u32 = 0x000000FF;
    pub const RED: u32 = 0xFF0000FF;
    pub const GREEN: u32 = 0x00FF00FF;
    pub const BLUE: u32 = 0x0000FFFF;
    pub const YELLOW: u32 = 0xFFFF00FF;
    pub const CYAN: u32 = 0x00FFFFFF;
    pub const GRAY: u32 = 0x888888FF;
    pub const DARK_GRAY: u32 = 0x444444FF;
    pub const LIGHT_GRAY: u32 = 0xCCCCCCFF;
}

/// Text segment metadata (GPU-side)
#[repr(C)]
#[derive(Copy, Clone)]
struct TextSegment {
    x: f32,
    y: f32,
    font_size: f32,
    start_char: u32,    // Offset into text buffer
    char_count: u32,    // Number of characters
    color: [f32; 4],
    _padding: [f32; 3], // Pad to 48 bytes (multiple of 16)
}

/// SDF text vertex (matches shader)
#[repr(C)]
#[derive(Copy, Clone)]
struct SdfTextVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

/// GPU parameters
#[repr(C)]
#[derive(Copy, Clone)]
struct LayoutParams {
    screen_width: f32,
    screen_height: f32,
    segment_count: u32,
    _padding: u32,
}

/// Text renderer - single GPU buffer, single dispatch
pub struct TextRenderer {
    // Pipelines
    layout_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // GPU buffers
    text_buffer: Buffer,        // Raw character bytes
    segments_buffer: Buffer,    // TextSegment array
    vertices_buffer: Buffer,    // Output vertices
    params_buffer: Buffer,      // LayoutParams
    metrics_buffer: Buffer,     // Glyph metrics
    screen_buffer: Buffer,      // Screen size for vertex shader

    // Atlas
    atlas_texture: Texture,

    // State (CPU-side tracking only)
    text_offset: usize,         // Current write position in text_buffer
    segment_count: usize,       // Number of segments
    total_chars: usize,         // Total characters added
    max_chars: usize,
    max_segments: usize,

    pub font_size: f32,
}

const MAX_SEGMENTS: usize = 256;

impl TextRenderer {
    pub fn new(device: &Device, max_chars: usize) -> Result<Self, String> {
        // Compile shaders
        let library = device
            .new_library_with_source(SDF_BATCH_SHADER, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile SDF shader: {}", e))?;

        let layout_fn = library.get_function("sdf_batch_layout", None)
            .map_err(|e| format!("Missing sdf_batch_layout: {}", e))?;
        let layout_pipeline = device.new_compute_pipeline_state_with_function(&layout_fn)
            .map_err(|e| format!("Failed to create layout pipeline: {}", e))?;

        let vertex_fn = library.get_function("sdf_vertex", None)
            .map_err(|e| format!("Missing sdf_vertex: {}", e))?;
        let fragment_fn = library.get_function("sdf_fragment", None)
            .map_err(|e| format!("Missing sdf_fragment: {}", e))?;

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

        let render_pipeline = device.new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        // Create buffers
        let text_buffer = device.new_buffer(
            max_chars as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let segments_buffer = device.new_buffer(
            (MAX_SEGMENTS * mem::size_of::<TextSegment>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vertices_buffer = device.new_buffer(
            (max_chars * 6 * mem::size_of::<SdfTextVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = device.new_buffer(
            mem::size_of::<LayoutParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create metrics buffer with GPU-compatible layout
        #[repr(C)]
        struct GpuGlyphMetric {
            advance: f32,
            _pad1: f32,
            _pad2: f32,
            _pad3: f32,
            bounds: [f32; 4],
            atlas_x: u32,
            atlas_y: u32,
            _pad4: u32,
            _pad5: u32,
        }

        let gpu_metrics: Vec<GpuGlyphMetric> = GLYPH_METRICS.iter().map(|m| {
            GpuGlyphMetric {
                advance: m.advance,
                _pad1: 0.0, _pad2: 0.0, _pad3: 0.0,
                bounds: m.bounds,
                atlas_x: m.atlas_x,
                atlas_y: m.atlas_y,
                _pad4: 0, _pad5: 0,
            }
        }).collect();

        let metrics_buffer = device.new_buffer_with_data(
            gpu_metrics.as_ptr() as *const _,
            (gpu_metrics.len() * mem::size_of::<GpuGlyphMetric>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let screen_buffer = device.new_buffer(
            8,  // float2
            MTLResourceOptions::StorageModeShared,
        );

        // Create atlas texture
        let texture_desc = TextureDescriptor::new();
        texture_desc.set_width(ATLAS_WIDTH as u64);
        texture_desc.set_height(ATLAS_HEIGHT as u64);
        texture_desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        texture_desc.set_texture_type(MTLTextureType::D2);
        texture_desc.set_usage(MTLTextureUsage::ShaderRead);
        texture_desc.set_storage_mode(MTLStorageMode::Shared);

        let atlas_texture = device.new_texture(&texture_desc);
        atlas_texture.replace_region(
            MTLRegion::new_2d(0, 0, ATLAS_WIDTH as u64, ATLAS_HEIGHT as u64),
            0,
            ATLAS_DATA.as_ptr() as *const _,
            ATLAS_WIDTH as u64,
        );

        Ok(Self {
            layout_pipeline,
            render_pipeline,
            text_buffer,
            segments_buffer,
            vertices_buffer,
            params_buffer,
            metrics_buffer,
            screen_buffer,
            atlas_texture,
            text_offset: 0,
            segment_count: 0,
            total_chars: 0,
            max_chars,
            max_segments: MAX_SEGMENTS,
            font_size: 14.0,
        })
    }

    /// Clear all text (resets buffer offsets)
    pub fn clear(&mut self) {
        self.text_offset = 0;
        self.segment_count = 0;
        self.total_chars = 0;
    }

    /// Add text - writes directly to GPU buffer
    pub fn add_text(&mut self, text: &str, x: f32, y: f32, color: u32) {
        self.add_text_sized(text, x, y, color, self.font_size);
    }

    /// Add text with scale (backwards compatibility)
    pub fn add_text_scaled(&mut self, text: &str, x: f32, y: f32, color: u32, scale: f32) {
        self.add_text_sized(text, x, y, color, 8.0 * scale);
    }

    /// Add text with specific font size
    pub fn add_text_sized(&mut self, text: &str, x: f32, y: f32, color: u32, font_size: f32) {
        if text.is_empty() || self.segment_count >= self.max_segments {
            return;
        }

        let char_count = text.len().min(self.max_chars - self.text_offset);
        if char_count == 0 {
            return;
        }

        // Write text bytes to GPU buffer
        unsafe {
            let ptr = self.text_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                text.as_ptr(),
                ptr.add(self.text_offset),
                char_count,
            );
        }

        // Write segment metadata
        let color_f32 = [
            ((color >> 24) & 0xFF) as f32 / 255.0,
            ((color >> 16) & 0xFF) as f32 / 255.0,
            ((color >> 8) & 0xFF) as f32 / 255.0,
            (color & 0xFF) as f32 / 255.0,
        ];

        let segment = TextSegment {
            x,
            y,
            font_size,
            start_char: self.text_offset as u32,
            char_count: char_count as u32,
            color: color_f32,
            _padding: [0.0; 3],
        };

        unsafe {
            let ptr = self.segments_buffer.contents() as *mut TextSegment;
            *ptr.add(self.segment_count) = segment;
        }

        self.text_offset += char_count;
        self.total_chars += char_count;
        self.segment_count += 1;
    }

    /// Get text width at current font size
    pub fn text_width(&self, text: &str) -> f32 {
        self.text_width_sized(text, self.font_size)
    }

    /// Get text width at specific font size
    pub fn text_width_sized(&self, text: &str, font_size: f32) -> f32 {
        let scale = font_size / UNITS_PER_EM;
        text.chars()
            .filter(|c| c.is_ascii() && *c >= ' ' && *c <= '~')
            .map(|c| {
                let idx = (c as u32 - 32) as usize;
                if idx < GLYPH_METRICS.len() {
                    GLYPH_METRICS[idx].advance * scale
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Get line height
    pub fn line_height(&self) -> f32 {
        self.font_size * 1.5
    }

    /// Render all text - single compute dispatch + single render pass
    pub fn render(
        &mut self,
        command_buffer: &CommandBufferRef,
        render_pass_desc: &RenderPassDescriptorRef,
        screen_width: f32,
        screen_height: f32,
    ) {
        if self.total_chars == 0 {
            return;
        }

        // Update params
        unsafe {
            let ptr = self.params_buffer.contents() as *mut LayoutParams;
            *ptr = LayoutParams {
                screen_width,
                screen_height,
                segment_count: self.segment_count as u32,
                _padding: 0,
            };

            let screen_ptr = self.screen_buffer.contents() as *mut [f32; 2];
            *screen_ptr = [screen_width, screen_height];
        }

        // Single compute dispatch for ALL text layout
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layout_pipeline);
            encoder.set_buffer(0, Some(&self.vertices_buffer), 0);
            encoder.set_buffer(1, Some(&self.text_buffer), 0);
            encoder.set_buffer(2, Some(&self.segments_buffer), 0);
            encoder.set_buffer(3, Some(&self.metrics_buffer), 0);
            encoder.set_buffer(4, Some(&self.params_buffer), 0);

            // One thread per character
            let threads = self.total_chars;
            let tpg = 256;
            let groups = (threads + tpg - 1) / tpg;
            encoder.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(tpg as u64, 1, 1),
            );
            encoder.end_encoding();
        }

        // Single render pass
        {
            let encoder = command_buffer.new_render_command_encoder(render_pass_desc);
            encoder.set_render_pipeline_state(&self.render_pipeline);
            encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);
            encoder.set_vertex_buffer(1, Some(&self.screen_buffer), 0);
            encoder.set_fragment_texture(0, Some(&self.atlas_texture));

            let vertex_count = self.total_chars * 6;
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
            encoder.end_encoding();
        }
    }

    pub fn char_count(&self) -> usize {
        self.total_chars
    }
}

// Keep for backwards compatibility
pub struct BitmapFont;
impl BitmapFont {
    pub fn new(_device: &Device) -> Self { BitmapFont }
    pub fn char_spacing(&self, scale: f32) -> f32 { 8.0 * scale }
    pub fn line_height(&self, scale: f32) -> f32 { 8.0 * scale * 1.5 }
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct TextChar {
    pub x: f32,
    pub y: f32,
    pub char_code: u32,
    pub color: u32,
}

const SDF_BATCH_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextSegment {
    float x;
    float y;
    float font_size;
    uint start_char;
    uint char_count;
    float4 color;
    float3 _padding;
};

struct GlyphMetric {
    float advance;
    float _pad1;
    float _pad2;
    float _pad3;
    float4 bounds;
    uint atlas_x;
    uint atlas_y;
    uint _pad4;
    uint _pad5;
};

struct LayoutParams {
    float screen_width;
    float screen_height;
    uint segment_count;
    uint _padding;
};

struct SdfTextVertex {
    float2 position;
    float2 uv;
    float4 color;
};

constant uint SDF_SIZE = 48;
constant uint ATLAS_WIDTH = 500;
constant uint ATLAS_HEIGHT = 500;
constant float UNITS_PER_EM = 2048.0;

// Find which segment this character belongs to
kernel void sdf_batch_layout(
    device SdfTextVertex* vertices [[buffer(0)]],
    device const uchar* text [[buffer(1)]],
    device const TextSegment* segments [[buffer(2)]],
    device const GlyphMetric* metrics [[buffer(3)]],
    constant LayoutParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Find which segment this global char index belongs to
    uint char_idx = gid;
    uint seg_idx = 0;
    uint local_idx = char_idx;
    uint global_offset = 0;

    for (uint s = 0; s < params.segment_count; s++) {
        if (char_idx < global_offset + segments[s].char_count) {
            seg_idx = s;
            local_idx = char_idx - global_offset;
            break;
        }
        global_offset += segments[s].char_count;
    }

    if (char_idx >= global_offset + segments[seg_idx].char_count) {
        // Out of bounds - write zero vertices
        uint base = gid * 6;
        for (uint i = 0; i < 6; i++) {
            vertices[base + i] = SdfTextVertex{float2(0), float2(0), float4(0)};
        }
        return;
    }

    TextSegment seg = segments[seg_idx];
    float scale = seg.font_size / UNITS_PER_EM;

    // Get character
    uchar c = text[seg.start_char + local_idx];
    if (c < 32 || c > 126) c = 32;
    uint glyph_idx = c - 32;
    GlyphMetric m = metrics[glyph_idx];

    // Calculate cursor X by summing advances of previous chars in this segment
    float cursor_x = seg.x;
    for (uint i = 0; i < local_idx; i++) {
        uchar prev_c = text[seg.start_char + i];
        if (prev_c < 32 || prev_c > 126) prev_c = 32;
        cursor_x += metrics[prev_c - 32].advance * scale;
    }

    // Glyph bounds
    float glyph_width = m.bounds.z - m.bounds.x;
    float glyph_height = m.bounds.w - m.bounds.y;

    // Position
    float x = cursor_x + m.bounds.x * scale;
    float y = seg.y - m.bounds.w * scale;
    float w = glyph_width * scale;
    float h = glyph_height * scale;

    uint base = gid * 6;

    // Skip space or empty glyphs
    if (c == 32 || glyph_width <= 0 || glyph_height <= 0) {
        for (uint i = 0; i < 6; i++) {
            vertices[base + i] = SdfTextVertex{float2(0), float2(0), float4(0)};
        }
        return;
    }

    // UV coordinates
    float u0 = float(m.atlas_x) / float(ATLAS_WIDTH);
    float v0 = float(m.atlas_y) / float(ATLAS_HEIGHT);
    float u1 = float(m.atlas_x + SDF_SIZE) / float(ATLAS_WIDTH);
    float v1 = float(m.atlas_y + SDF_SIZE) / float(ATLAS_HEIGHT);

    // Generate quad (2 triangles)
    vertices[base + 0] = SdfTextVertex{float2(x, y), float2(u0, v1), seg.color};
    vertices[base + 1] = SdfTextVertex{float2(x, y + h), float2(u0, v0), seg.color};
    vertices[base + 2] = SdfTextVertex{float2(x + w, y + h), float2(u1, v0), seg.color};
    vertices[base + 3] = SdfTextVertex{float2(x, y), float2(u0, v1), seg.color};
    vertices[base + 4] = SdfTextVertex{float2(x + w, y + h), float2(u1, v0), seg.color};
    vertices[base + 5] = SdfTextVertex{float2(x + w, y), float2(u1, v1), seg.color};
}

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut sdf_vertex(
    device const SdfTextVertex* vertices [[buffer(0)]],
    constant float2& screen_size [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    SdfTextVertex v = vertices[vid];

    // Convert to clip space
    float2 ndc = (v.position / screen_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    VertexOut out;
    out.position = float4(ndc, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

fragment float4 sdf_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    constexpr sampler samp(mag_filter::linear, min_filter::linear);
    float d = atlas.sample(samp, in.uv).r;

    // SDF threshold with anti-aliasing
    float edge = 0.5;
    float aa = fwidth(d) * 0.75;
    float alpha = smoothstep(edge - aa, edge + aa, d);

    if (alpha < 0.01) discard_fragment();

    return float4(in.color.rgb, in.color.a * alpha);
}
"#;
