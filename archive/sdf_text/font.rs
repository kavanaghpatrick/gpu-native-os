// Phase 1: Font Loading with ttf-parser
//
// Loads TrueType/OpenType fonts and extracts glyph outlines.

use std::collections::HashMap;
use std::path::Path;

/// Error type for font operations
#[derive(Debug)]
pub enum FontError {
    IoError(std::io::Error),
    ParseError(String),
    GlyphNotFound(char),
    NoSystemFont,
}

impl std::fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FontError::IoError(e) => write!(f, "IO error: {}", e),
            FontError::ParseError(s) => write!(f, "Parse error: {}", s),
            FontError::GlyphNotFound(c) => write!(f, "Glyph not found: '{}'", c),
            FontError::NoSystemFont => write!(f, "No system font available"),
        }
    }
}

impl std::error::Error for FontError {}

impl From<std::io::Error> for FontError {
    fn from(e: std::io::Error) -> Self {
        FontError::IoError(e)
    }
}

/// Rectangle bounds
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl Rect {
    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f32 {
        self.y_max - self.y_min
    }

    pub fn is_empty(&self) -> bool {
        self.width() <= 0.0 || self.height() <= 0.0
    }
}

/// Path segment for glyph outlines
#[derive(Debug, Clone)]
pub enum PathSegment {
    MoveTo(f32, f32),
    LineTo(f32, f32),
    QuadTo(f32, f32, f32, f32),        // control, end
    CubicTo(f32, f32, f32, f32, f32, f32), // control1, control2, end
    Close,
}

/// Glyph outline containing path segments
#[derive(Debug, Clone)]
pub struct GlyphOutline {
    pub bounds: Rect,
    pub segments: Vec<PathSegment>,
}

impl GlyphOutline {
    pub fn new() -> Self {
        Self {
            bounds: Rect::default(),
            segments: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

impl Default for GlyphOutline {
    fn default() -> Self {
        Self::new()
    }
}

/// Glyph metrics for layout
#[derive(Debug, Clone, Copy, Default)]
pub struct GlyphMetrics {
    pub advance_width: f32,
    pub left_side_bearing: f32,
    pub bounds: Rect,
}

/// SDF Font - loads and provides access to glyph data
pub struct SdfFont {
    font_data: Vec<u8>,
    units_per_em: u16,
    ascender: i16,
    descender: i16,
    line_gap: i16,
    glyph_cache: HashMap<char, (GlyphOutline, GlyphMetrics)>,
}

impl SdfFont {
    /// Load font from raw data
    pub fn load(data: &[u8]) -> Result<Self, FontError> {
        let face = ttf_parser::Face::parse(data, 0)
            .map_err(|e| FontError::ParseError(format!("{:?}", e)))?;

        let units_per_em = face.units_per_em();
        let ascender = face.ascender();
        let descender = face.descender();
        let line_gap = face.line_gap();

        let mut font = Self {
            font_data: data.to_vec(),
            units_per_em,
            ascender,
            descender,
            line_gap,
            glyph_cache: HashMap::new(),
        };

        // Pre-cache ASCII glyphs (32-126)
        font.cache_ascii_glyphs()?;

        Ok(font)
    }

    /// Load font from file path
    pub fn load_file(path: &Path) -> Result<Self, FontError> {
        let data = std::fs::read(path)?;
        Self::load(&data)
    }

    /// Try to load a system font (macOS)
    pub fn load_system_font(name: &str) -> Result<Self, FontError> {
        // Common macOS font paths
        let font_paths = [
            format!("/System/Library/Fonts/{}.ttf", name),
            format!("/System/Library/Fonts/{}.otf", name),
            format!("/Library/Fonts/{}.ttf", name),
            format!("/Library/Fonts/{}.otf", name),
            format!("/System/Library/Fonts/Supplemental/{}.ttf", name),
            format!("/System/Library/Fonts/Supplemental/{}.otf", name),
        ];

        for path in &font_paths {
            if let Ok(data) = std::fs::read(path) {
                if let Ok(font) = Self::load(&data) {
                    return Ok(font);
                }
            }
        }

        // Try default fonts
        let default_fonts = [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ];

        for path in &default_fonts {
            if let Ok(data) = std::fs::read(path) {
                if let Ok(font) = Self::load(&data) {
                    return Ok(font);
                }
            }
        }

        Err(FontError::NoSystemFont)
    }

    /// Load default system font
    pub fn load_default() -> Result<Self, FontError> {
        Self::load_system_font("SF-Pro")
            .or_else(|_| Self::load_system_font("Helvetica"))
            .or_else(|_| Self::load_system_font("Arial"))
    }

    fn face(&self) -> Result<ttf_parser::Face<'_>, FontError> {
        ttf_parser::Face::parse(&self.font_data, 0)
            .map_err(|e| FontError::ParseError(format!("{:?}", e)))
    }

    fn cache_ascii_glyphs(&mut self) -> Result<(), FontError> {
        // Collect all glyph data first to avoid borrow conflicts
        let glyphs: Vec<(char, GlyphOutline, GlyphMetrics)> = {
            let face = self.face()?;
            let mut result = Vec::new();

            for codepoint in 32u8..=126 {
                let c = codepoint as char;
                if let Some(glyph_id) = face.glyph_index(c) {
                    let outline = Self::extract_outline_from_face(&face, glyph_id);
                    let metrics = Self::extract_metrics_from_face(&face, glyph_id);
                    result.push((c, outline, metrics));
                }
            }
            result
        };

        // Now insert into cache
        for (c, outline, metrics) in glyphs {
            self.glyph_cache.insert(c, (outline, metrics));
        }

        Ok(())
    }

    fn extract_outline_from_face(face: &ttf_parser::Face<'_>, glyph_id: ttf_parser::GlyphId) -> GlyphOutline {
        let mut builder = OutlineBuilder::new();

        if face.outline_glyph(glyph_id, &mut builder).is_some() {
            builder.finish()
        } else {
            GlyphOutline::new()
        }
    }

    fn extract_metrics_from_face(face: &ttf_parser::Face<'_>, glyph_id: ttf_parser::GlyphId) -> GlyphMetrics {
        let advance_width = face
            .glyph_hor_advance(glyph_id)
            .unwrap_or(0) as f32;

        let left_side_bearing = face
            .glyph_hor_side_bearing(glyph_id)
            .unwrap_or(0) as f32;

        let bounds = if let Some(bbox) = face.glyph_bounding_box(glyph_id) {
            Rect {
                x_min: bbox.x_min as f32,
                y_min: bbox.y_min as f32,
                x_max: bbox.x_max as f32,
                y_max: bbox.y_max as f32,
            }
        } else {
            Rect::default()
        };

        GlyphMetrics {
            advance_width,
            left_side_bearing,
            bounds,
        }
    }

    /// Get units per em
    pub fn units_per_em(&self) -> u16 {
        self.units_per_em
    }

    /// Get ascender (positive, above baseline)
    pub fn ascender(&self) -> i16 {
        self.ascender
    }

    /// Get descender (negative, below baseline)
    pub fn descender(&self) -> i16 {
        self.descender
    }

    /// Get line gap
    pub fn line_gap(&self) -> i16 {
        self.line_gap
    }

    /// Get line height in font units
    pub fn line_height(&self) -> i16 {
        self.ascender - self.descender + self.line_gap
    }

    /// Get glyph outline for a character
    pub fn glyph_outline(&self, codepoint: char) -> Option<&GlyphOutline> {
        self.glyph_cache.get(&codepoint).map(|(outline, _)| outline)
    }

    /// Get glyph metrics for a character
    pub fn glyph_metrics(&self, codepoint: char) -> Option<&GlyphMetrics> {
        self.glyph_cache.get(&codepoint).map(|(_, metrics)| metrics)
    }

    /// Get glyph advance width in font units
    pub fn glyph_advance(&self, codepoint: char) -> Option<f32> {
        self.glyph_metrics(codepoint).map(|m| m.advance_width)
    }

    /// Check if a glyph exists for this character
    pub fn has_glyph(&self, codepoint: char) -> bool {
        self.glyph_cache.contains_key(&codepoint)
    }

    /// Get number of cached glyphs
    pub fn cached_glyph_count(&self) -> usize {
        self.glyph_cache.len()
    }

    /// Scale font units to pixels
    pub fn scale_to_pixels(&self, font_units: f32, font_size_px: f32) -> f32 {
        font_units * font_size_px / self.units_per_em as f32
    }
}

/// Builder for extracting glyph outlines from ttf-parser
struct OutlineBuilder {
    segments: Vec<PathSegment>,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
    started: bool,
}

impl OutlineBuilder {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
            x_min: f32::MAX,
            y_min: f32::MAX,
            x_max: f32::MIN,
            y_max: f32::MIN,
            started: false,
        }
    }

    fn update_bounds(&mut self, x: f32, y: f32) {
        self.x_min = self.x_min.min(x);
        self.y_min = self.y_min.min(y);
        self.x_max = self.x_max.max(x);
        self.y_max = self.y_max.max(y);
    }

    fn finish(self) -> GlyphOutline {
        let bounds = if self.started {
            Rect {
                x_min: self.x_min,
                y_min: self.y_min,
                x_max: self.x_max,
                y_max: self.y_max,
            }
        } else {
            Rect::default()
        };

        GlyphOutline {
            bounds,
            segments: self.segments,
        }
    }
}

impl ttf_parser::OutlineBuilder for OutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.started = true;
        self.update_bounds(x, y);
        self.segments.push(PathSegment::MoveTo(x, y));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.update_bounds(x, y);
        self.segments.push(PathSegment::LineTo(x, y));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.update_bounds(x1, y1);
        self.update_bounds(x, y);
        self.segments.push(PathSegment::QuadTo(x1, y1, x, y));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.update_bounds(x1, y1);
        self.update_bounds(x2, y2);
        self.update_bounds(x, y);
        self.segments.push(PathSegment::CubicTo(x1, y1, x2, y2, x, y));
    }

    fn close(&mut self) {
        self.segments.push(PathSegment::Close);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_dimensions() {
        let rect = Rect {
            x_min: 10.0,
            y_min: 20.0,
            x_max: 50.0,
            y_max: 80.0,
        };
        assert_eq!(rect.width(), 40.0);
        assert_eq!(rect.height(), 60.0);
        assert!(!rect.is_empty());
    }

    #[test]
    fn test_empty_rect() {
        let rect = Rect::default();
        assert!(rect.is_empty());
    }

    #[test]
    fn test_glyph_outline_default() {
        let outline = GlyphOutline::new();
        assert!(outline.is_empty());
        assert!(outline.segments.is_empty());
    }
}
