// Phase 2: CPU SDF Generation
//
// Generates Signed Distance Field bitmaps from glyph outlines.
// This is a reference implementation - Phase 2 will implement fully.

use super::font::GlyphOutline;

/// SDF bitmap containing signed distance values
#[derive(Debug, Clone)]
pub struct SdfBitmap {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl SdfBitmap {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width * height) as usize],
        }
    }

    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize]
        } else {
            1.0 // Outside is positive (far from edge)
        }
    }

    pub fn set(&mut self, x: u32, y: u32, value: f32) {
        if x < self.width && y < self.height {
            self.data[(y * self.width + x) as usize] = value;
        }
    }

    /// Convert to 8-bit grayscale (for GPU upload)
    /// Maps signed distance to 0-255 range
    pub fn to_u8(&self, spread: f32) -> Vec<u8> {
        self.data
            .iter()
            .map(|&d| {
                // Map [-spread, spread] to [0, 255]
                let normalized = (d / spread + 1.0) * 0.5;
                (normalized.clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect()
    }
}

/// SDF Generator - creates SDF from glyph outlines
pub struct SdfGenerator {
    pub sdf_size: u32,
    pub padding: u32,
    pub spread: f32,
}

impl SdfGenerator {
    pub fn new(sdf_size: u32, padding: u32) -> Self {
        Self {
            sdf_size,
            padding,
            spread: sdf_size as f32 * 0.25, // Default spread
        }
    }

    /// Generate SDF from glyph outline
    /// Phase 2 will implement the full algorithm
    pub fn generate(&self, _outline: &GlyphOutline) -> SdfBitmap {
        // Placeholder - returns empty SDF
        // Phase 2 will implement distance calculation
        SdfBitmap::new(self.sdf_size, self.sdf_size)
    }
}

impl Default for SdfGenerator {
    fn default() -> Self {
        Self::new(64, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_bitmap_creation() {
        let sdf = SdfBitmap::new(32, 32);
        assert_eq!(sdf.width, 32);
        assert_eq!(sdf.height, 32);
        assert_eq!(sdf.data.len(), 1024);
    }

    #[test]
    fn test_sdf_bitmap_get_set() {
        let mut sdf = SdfBitmap::new(16, 16);
        sdf.set(5, 5, -0.5);
        assert_eq!(sdf.get(5, 5), -0.5);
    }

    #[test]
    fn test_sdf_to_u8() {
        let mut sdf = SdfBitmap::new(4, 4);
        sdf.set(0, 0, -1.0);  // Inside
        sdf.set(1, 0, 0.0);   // Edge
        sdf.set(2, 0, 1.0);   // Outside

        let bytes = sdf.to_u8(1.0);
        assert_eq!(bytes[0], 0);    // -1.0 -> 0
        assert_eq!(bytes[1], 127);  // 0.0 -> ~127
        assert_eq!(bytes[2], 255);  // 1.0 -> 255
    }
}
