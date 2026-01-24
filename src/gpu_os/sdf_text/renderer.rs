// Phase 4-5: GPU SDF Rendering & Text Shaping
//
// Renders text using SDF atlas with anti-aliased edges.
// Phase 4-5 will implement fully.

use metal::*;
use super::atlas::SdfAtlas;
use super::font::SdfFont;

/// Text metrics for measurement
#[derive(Debug, Clone, Copy, Default)]
pub struct TextMetrics {
    pub width: f32,
    pub height: f32,
    pub ascent: f32,
    pub descent: f32,
}

/// Instance of a glyph to render
#[derive(Debug, Clone, Copy)]
pub struct GlyphInstance {
    pub x: f32,
    pub y: f32,
    pub codepoint: char,
    pub font_size: f32,
    pub color: [f32; 4],
}

/// SDF Text Renderer
pub struct SdfTextRenderer {
    render_pipeline: Option<RenderPipelineState>,
    vertex_buffer: Option<Buffer>,
    max_glyphs: usize,
}

impl SdfTextRenderer {
    pub fn new(device: &Device, max_glyphs: usize) -> Result<Self, String> {
        // Phase 4 will implement shader compilation
        Ok(Self {
            render_pipeline: None,
            vertex_buffer: None,
            max_glyphs,
        })
    }

    /// Measure text dimensions
    pub fn measure_text(&self, font: &SdfFont, text: &str, font_size: f32) -> TextMetrics {
        let scale = font_size / font.units_per_em() as f32;
        let mut width = 0.0;

        for c in text.chars() {
            if let Some(advance) = font.glyph_advance(c) {
                width += advance * scale;
            }
        }

        let ascent = font.ascender() as f32 * scale;
        let descent = font.descender() as f32 * scale;
        let height = ascent - descent;

        TextMetrics {
            width,
            height,
            ascent,
            descent,
        }
    }

    /// Layout a line of text into glyph instances
    pub fn layout_line(
        &self,
        font: &SdfFont,
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        color: [f32; 4],
    ) -> Vec<GlyphInstance> {
        let scale = font_size / font.units_per_em() as f32;
        let mut instances = Vec::with_capacity(text.len());
        let mut cursor_x = x;

        for c in text.chars() {
            if let Some(advance) = font.glyph_advance(c) {
                instances.push(GlyphInstance {
                    x: cursor_x,
                    y,
                    codepoint: c,
                    font_size,
                    color,
                });
                cursor_x += advance * scale;
            }
        }

        instances
    }

    /// Render text (Phase 4 will implement GPU rendering)
    pub fn render(
        &self,
        _encoder: &RenderCommandEncoderRef,
        _atlas: &SdfAtlas,
        _instances: &[GlyphInstance],
    ) {
        // Phase 4 will implement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_metrics_default() {
        let metrics = TextMetrics::default();
        assert_eq!(metrics.width, 0.0);
        assert_eq!(metrics.height, 0.0);
    }
}
