// Phase 3: SDF Atlas Packing
//
// Packs multiple glyph SDFs into a single texture atlas.
// Phase 3 will implement fully.

use metal::*;
use std::collections::HashMap;
use super::generator::SdfBitmap;

/// Error type for atlas operations
#[derive(Debug)]
pub enum AtlasError {
    AtlasFull,
    TextureCreationFailed(String),
}

impl std::fmt::Display for AtlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtlasError::AtlasFull => write!(f, "Atlas is full"),
            AtlasError::TextureCreationFailed(s) => write!(f, "Texture creation failed: {}", s),
        }
    }
}

impl std::error::Error for AtlasError {}

/// Information about a glyph in the atlas
#[derive(Debug, Clone, Copy, Default)]
pub struct AtlasGlyph {
    /// UV rectangle in atlas (x, y, width, height) in 0-1 range
    pub uv_rect: [f32; 4],
    /// Glyph size in pixels
    pub size: [f32; 2],
    /// Offset from baseline
    pub bearing: [f32; 2],
    /// Horizontal advance
    pub advance: f32,
}

/// SDF Atlas - packed texture with glyph SDFs
pub struct SdfAtlas {
    texture: Option<Texture>,
    width: u32,
    height: u32,
    glyphs: HashMap<char, AtlasGlyph>,
    // Packing state
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    padding: u32,
}

impl SdfAtlas {
    pub fn new(_device: &Device, width: u32, height: u32) -> Self {
        Self {
            texture: None,
            width,
            height,
            glyphs: HashMap::new(),
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            padding: 2,
        }
    }

    /// Add a glyph to the atlas
    /// Phase 3 will implement packing algorithm
    pub fn add_glyph(
        &mut self,
        codepoint: char,
        _sdf: &SdfBitmap,
        size: [f32; 2],
        bearing: [f32; 2],
        advance: f32,
    ) -> Result<(), AtlasError> {
        // Placeholder - just store metadata
        self.glyphs.insert(codepoint, AtlasGlyph {
            uv_rect: [0.0, 0.0, 0.0, 0.0], // Phase 3 will compute real UVs
            size,
            bearing,
            advance,
        });
        Ok(())
    }

    /// Build the atlas texture
    /// Phase 3 will implement
    pub fn build(&mut self, device: &Device) -> Result<(), AtlasError> {
        // Create a simple placeholder texture
        let desc = TextureDescriptor::new();
        desc.set_width(self.width as u64);
        desc.set_height(self.height as u64);
        desc.set_pixel_format(MTLPixelFormat::R8Unorm);
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_usage(MTLTextureUsage::ShaderRead);
        desc.set_storage_mode(MTLStorageMode::Shared);

        self.texture = Some(device.new_texture(&desc));
        Ok(())
    }

    /// Get the atlas texture
    pub fn texture(&self) -> Option<&Texture> {
        self.texture.as_ref()
    }

    /// Get glyph info
    pub fn glyph_info(&self, codepoint: char) -> Option<&AtlasGlyph> {
        self.glyphs.get(&codepoint)
    }

    /// Get number of glyphs in atlas
    pub fn glyph_count(&self) -> usize {
        self.glyphs.len()
    }

    /// Get atlas dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_glyph_default() {
        let glyph = AtlasGlyph::default();
        assert_eq!(glyph.uv_rect, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(glyph.advance, 0.0);
    }
}
