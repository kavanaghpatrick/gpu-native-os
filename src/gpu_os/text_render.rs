// GPU Text Rendering - Simple Bitmap Font System
//
// Provides GPU-accelerated text rendering using an 8x8 bitmap font.
// All rendering happens on the GPU via Metal shaders.
//
// Usage:
//   let font = BitmapFont::new(&device);
//   let mut renderer = TextRenderer::new(&device, &font);
//   renderer.add_text("Hello", 10.0, 10.0, 0xFFFFFFFF);
//   renderer.render(&encoder, screen_width, screen_height);

use metal::*;
use std::mem;

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

/// A single character to render
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct TextChar {
    pub x: f32,
    pub y: f32,
    pub char_code: u32,
    pub color: u32, // RGBA packed as 0xRRGGBBAA
}

/// Uniforms passed to the text shader
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TextUniforms {
    pub screen_size: [f32; 2],
    pub scale: f32,
    pub _padding: f32,
}

/// 8x8 bitmap font stored as a GPU texture
pub struct BitmapFont {
    pub texture: Texture,
    pub char_width: f32,
    pub char_height: f32,
    pub chars_per_row: u32,
    pub num_rows: u32,
}

impl BitmapFont {
    /// Create a new bitmap font with the standard ASCII character set (32-127)
    pub fn new(device: &Device) -> Self {
        let char_w = 8usize;
        let char_h = 8usize;
        let chars_per_row = 16usize;
        let num_rows = 6usize;
        let width = (chars_per_row * char_w) as u64;
        let height = (num_rows * char_h) as u64;

        let desc = TextureDescriptor::new();
        desc.set_width(width);
        desc.set_height(height);
        desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_usage(MTLTextureUsage::ShaderRead);
        desc.set_storage_mode(MTLStorageMode::Shared);

        let texture = device.new_texture(&desc);
        let mut data = vec![0u8; (width * height) as usize];

        // Render font data into texture
        let font = get_font_data();
        for ascii in 32u8..128 {
            let idx = (ascii - 32) as usize;
            if idx * 8 + 7 >= font.len() {
                continue;
            }

            let col = idx % chars_per_row;
            let row = idx / chars_per_row;
            let base_x = col * char_w;
            let base_y = row * char_h;

            for py in 0..8 {
                let byte = font[idx * 8 + py];
                for px in 0..8 {
                    let bit = (byte >> (7 - px)) & 1;
                    let x = base_x + px;
                    let y = base_y + py;
                    if x < width as usize && y < height as usize {
                        data[y * width as usize + x] = if bit == 1 { 255 } else { 0 };
                    }
                }
            }
        }

        texture.replace_region(
            MTLRegion::new_2d(0, 0, width, height),
            0,
            data.as_ptr() as *const _,
            width,
        );

        Self {
            texture,
            char_width: char_w as f32,
            char_height: char_h as f32,
            chars_per_row: chars_per_row as u32,
            num_rows: num_rows as u32,
        }
    }

    /// Get the spacing between characters (accounting for scale)
    pub fn char_spacing(&self, scale: f32) -> f32 {
        self.char_width * scale
    }

    /// Get line height (accounting for scale)
    pub fn line_height(&self, scale: f32) -> f32 {
        self.char_height * scale * 1.5 // 1.5x for line spacing
    }
}

/// Text renderer that batches text drawing into a single GPU call
pub struct TextRenderer {
    pipeline: RenderPipelineState,
    chars_buffer: Buffer,
    uniforms_buffer: Buffer,
    chars: Vec<TextChar>,
    max_chars: usize,
    pub scale: f32,
}

impl TextRenderer {
    /// Create a new text renderer
    pub fn new(device: &Device, max_chars: usize) -> Result<Self, String> {
        let library = device
            .new_library_with_source(TEXT_SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile text shaders: {}", e))?;

        let vertex_fn = library
            .get_function("text_vertex", None)
            .map_err(|e| format!("Missing text_vertex function: {}", e))?;
        let fragment_fn = library
            .get_function("text_fragment", None)
            .map_err(|e| format!("Missing text_fragment function: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));

        // Enable alpha blending
        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create text pipeline: {}", e))?;

        let chars_buffer = device.new_buffer(
            (max_chars * mem::size_of::<TextChar>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let uniforms_buffer = device.new_buffer(
            mem::size_of::<TextUniforms>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        Ok(Self {
            pipeline,
            chars_buffer,
            uniforms_buffer,
            chars: Vec::with_capacity(max_chars),
            max_chars,
            scale: 1.5, // Default scale for readability
        })
    }

    /// Clear all queued text
    pub fn clear(&mut self) {
        self.chars.clear();
    }

    /// Add text at the specified position with the given color
    /// Color format: 0xRRGGBBAA
    pub fn add_text(&mut self, text: &str, x: f32, y: f32, color: u32) {
        let spacing = 8.0 * self.scale;
        let mut cx = x;

        for c in text.chars() {
            if self.chars.len() >= self.max_chars {
                break;
            }

            if c.is_ascii() && c >= ' ' && c <= '~' {
                self.chars.push(TextChar {
                    x: cx,
                    y,
                    char_code: c as u32,
                    color,
                });
                cx += spacing;
            }
        }
    }

    /// Add text with a specific scale override
    pub fn add_text_scaled(&mut self, text: &str, x: f32, y: f32, color: u32, scale: f32) {
        let spacing = 8.0 * scale;
        let mut cx = x;

        for c in text.chars() {
            if self.chars.len() >= self.max_chars {
                break;
            }

            if c.is_ascii() && c >= ' ' && c <= '~' {
                self.chars.push(TextChar {
                    x: cx,
                    y,
                    char_code: c as u32,
                    color,
                });
                cx += spacing;
            }
        }
    }

    /// Get the width of a string in pixels (at current scale)
    pub fn text_width(&self, text: &str) -> f32 {
        let char_count = text.chars().filter(|c| c.is_ascii() && *c >= ' ' && *c <= '~').count();
        char_count as f32 * 8.0 * self.scale
    }

    /// Get line height at current scale
    pub fn line_height(&self) -> f32 {
        8.0 * self.scale * 1.5
    }

    /// Render all queued text
    pub fn render(
        &mut self,
        encoder: &RenderCommandEncoderRef,
        font: &BitmapFont,
        screen_width: f32,
        screen_height: f32,
    ) {
        if self.chars.is_empty() {
            return;
        }

        // Upload uniforms
        let uniforms = TextUniforms {
            screen_size: [screen_width, screen_height],
            scale: self.scale,
            _padding: 0.0,
        };
        unsafe {
            let ptr = self.uniforms_buffer.contents() as *mut TextUniforms;
            *ptr = uniforms;
        }

        // Upload character data
        unsafe {
            let ptr = self.chars_buffer.contents() as *mut TextChar;
            for (i, ch) in self.chars.iter().enumerate() {
                *ptr.add(i) = *ch;
            }
        }

        // Set pipeline and buffers
        encoder.set_render_pipeline_state(&self.pipeline);
        encoder.set_vertex_buffer(0, Some(&self.uniforms_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&self.chars_buffer), 0);
        encoder.set_fragment_texture(0, Some(&font.texture));

        // Draw: 6 vertices per character (2 triangles)
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, (self.chars.len() * 6) as u64);
    }

    /// Get the number of characters currently queued
    pub fn char_count(&self) -> usize {
        self.chars.len()
    }
}

/// Returns the 8x8 bitmap font data for ASCII 32-127
fn get_font_data() -> [u8; 768] {
    [
        // 32 ' ' (space)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 33 '!'
        0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00,
        // 34 '"'
        0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 35 '#'
        0x24, 0x24, 0x7E, 0x24, 0x7E, 0x24, 0x24, 0x00,
        // 36 '$'
        0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00,
        // 37 '%'
        0x00, 0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00,
        // 38 '&'
        0x30, 0x48, 0x30, 0x56, 0x88, 0x88, 0x76, 0x00,
        // 39 '''
        0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 40 '('
        0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00,
        // 41 ')'
        0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00,
        // 42 '*'
        0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00,
        // 43 '+'
        0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00,
        // 44 ','
        0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30,
        // 45 '-'
        0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00,
        // 46 '.'
        0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00,
        // 47 '/'
        0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00,
        // 48 '0'
        0x3C, 0x66, 0x6E, 0x7E, 0x76, 0x66, 0x3C, 0x00,
        // 49 '1'
        0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00,
        // 50 '2'
        0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00,
        // 51 '3'
        0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00,
        // 52 '4'
        0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00,
        // 53 '5'
        0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00,
        // 54 '6'
        0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00,
        // 55 '7'
        0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00,
        // 56 '8'
        0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00,
        // 57 '9'
        0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00,
        // 58 ':'
        0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00,
        // 59 ';'
        0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00,
        // 60 '<'
        0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00,
        // 61 '='
        0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00,
        // 62 '>'
        0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00,
        // 63 '?'
        0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00,
        // 64 '@'
        0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00,
        // 65 'A'
        0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00,
        // 66 'B'
        0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00,
        // 67 'C'
        0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00,
        // 68 'D'
        0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00,
        // 69 'E'
        0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00,
        // 70 'F'
        0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00,
        // 71 'G'
        0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00,
        // 72 'H'
        0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00,
        // 73 'I'
        0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
        // 74 'J'
        0x06, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C, 0x00,
        // 75 'K'
        0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00,
        // 76 'L'
        0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00,
        // 77 'M'
        0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00,
        // 78 'N'
        0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00,
        // 79 'O'
        0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00,
        // 80 'P'
        0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00,
        // 81 'Q'
        0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00,
        // 82 'R'
        0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00,
        // 83 'S'
        0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00,
        // 84 'T'
        0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00,
        // 85 'U'
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00,
        // 86 'V'
        0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00,
        // 87 'W'
        0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00,
        // 88 'X'
        0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00,
        // 89 'Y'
        0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00,
        // 90 'Z'
        0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00,
        // 91 '['
        0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00,
        // 92 '\'
        0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00,
        // 93 ']'
        0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00,
        // 94 '^'
        0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 95 '_'
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00,
        // 96 '`'
        0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 97 'a'
        0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00,
        // 98 'b'
        0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00,
        // 99 'c'
        0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00,
        // 100 'd'
        0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00,
        // 101 'e'
        0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00,
        // 102 'f'
        0x1C, 0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x00,
        // 103 'g'
        0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C,
        // 104 'h'
        0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00,
        // 105 'i'
        0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00,
        // 106 'j'
        0x0C, 0x00, 0x1C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38,
        // 107 'k'
        0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00,
        // 108 'l'
        0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
        // 109 'm'
        0x00, 0x00, 0x76, 0x7F, 0x6B, 0x6B, 0x63, 0x00,
        // 110 'n'
        0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00,
        // 111 'o'
        0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00,
        // 112 'p'
        0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60,
        // 113 'q'
        0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06,
        // 114 'r'
        0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00,
        // 115 's'
        0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00,
        // 116 't'
        0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x1C, 0x00,
        // 117 'u'
        0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00,
        // 118 'v'
        0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00,
        // 119 'w'
        0x00, 0x00, 0x63, 0x6B, 0x6B, 0x7F, 0x36, 0x00,
        // 120 'x'
        0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00,
        // 121 'y'
        0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C,
        // 122 'z'
        0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00,
        // 123 '{'
        0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00,
        // 124 '|'
        0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00,
        // 125 '}'
        0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00,
        // 126 '~'
        0x32, 0x4C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 127 DEL (placeholder)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]
}

const TEXT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TextUniforms {
    float2 screen_size;
    float scale;
    float _padding;
};

struct TextChar {
    float x;
    float y;
    uint char_code;
    uint color;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut text_vertex(
    uint vid [[vertex_id]],
    constant TextUniforms& uniforms [[buffer(0)]],
    constant TextChar* chars [[buffer(1)]]
) {
    uint char_idx = vid / 6;
    uint vert_idx = vid % 6;

    TextChar ch = chars[char_idx];

    // Unpack color (RGBA: 0xRRGGBBAA)
    float4 color;
    color.r = float((ch.color >> 24) & 0xFF) / 255.0;
    color.g = float((ch.color >> 16) & 0xFF) / 255.0;
    color.b = float((ch.color >> 8) & 0xFF) / 255.0;
    color.a = float(ch.color & 0xFF) / 255.0;

    // Character cell size (scaled)
    float char_w = 8.0 * uniforms.scale;
    float char_h = 8.0 * uniforms.scale;

    // Quad vertices (two triangles)
    float2 positions[6] = {
        float2(0, 0), float2(char_w, 0), float2(char_w, char_h),
        float2(0, 0), float2(char_w, char_h), float2(0, char_h)
    };

    // UV coordinates within cell (0-1)
    float2 uvs[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1)
    };

    float2 pos = float2(ch.x, ch.y) + positions[vert_idx];

    // Convert to clip space
    float2 ndc = (pos / uniforms.screen_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Calculate UV based on character code
    // Texture is 128x48 with 16x6 character grid
    uint ascii = ch.char_code;
    if (ascii < 32 || ascii > 127) ascii = 32;
    uint idx = ascii - 32;
    uint col = idx % 16;
    uint row = idx / 16;

    float cell_u = 8.0 / 128.0;
    float cell_v = 8.0 / 48.0;

    float2 uv_base = float2(float(col) * cell_u, float(row) * cell_v);
    float2 uv_offset = uvs[vert_idx] * float2(cell_u, cell_v);

    VertexOut out;
    out.position = float4(ndc, 0.0, 1.0);
    out.uv = uv_base + uv_offset;
    out.color = color;
    return out;
}

fragment float4 text_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> font_tex [[texture(0)]]
) {
    constexpr sampler samp(mag_filter::nearest, min_filter::nearest);
    float glyph = font_tex.sample(samp, in.uv).r;

    // Discard transparent pixels
    if (glyph < 0.5) discard_fragment();

    return in.color;
}
"#;
