//! Window Content Textures
//!
//! Manages render-to-texture targets for each window.
//! Apps render their content to these textures, which the compositor
//! then samples when rendering windows.

use metal::*;
use std::collections::HashMap;

use super::types::*;

/// Manages textures for window content rendering
pub struct WindowTextures {
    device: Device,
    /// Window ID -> (texture, width, height)
    textures: HashMap<u32, WindowTexture>,
    /// Pixel format for all textures
    pixel_format: MTLPixelFormat,
}

/// A single window's render texture
pub struct WindowTexture {
    pub texture: Texture,
    pub width: u32,
    pub height: u32,
}

impl WindowTextures {
    /// Create a new window texture manager
    pub fn new(device: &Device, pixel_format: MTLPixelFormat) -> Self {
        Self {
            device: device.clone(),
            textures: HashMap::new(),
            pixel_format,
        }
    }

    /// Get or create a texture for a window
    ///
    /// Creates a new texture if one doesn't exist or if size changed.
    pub fn get_or_create(&mut self, window_id: u32, width: u32, height: u32) -> &Texture {
        // Check if we need to create or resize
        let needs_create = match self.textures.get(&window_id) {
            None => true,
            Some(wt) => wt.width != width || wt.height != height,
        };

        if needs_create {
            let texture = self.create_texture(width, height);
            self.textures.insert(window_id, WindowTexture {
                texture,
                width,
                height,
            });
        }

        &self.textures.get(&window_id).unwrap().texture
    }

    /// Get existing texture for a window (if any)
    pub fn get(&self, window_id: u32) -> Option<&Texture> {
        self.textures.get(&window_id).map(|wt| &wt.texture)
    }

    /// Remove texture for a window
    pub fn remove(&mut self, window_id: u32) {
        self.textures.remove(&window_id);
    }

    /// Create a render texture
    fn create_texture(&self, width: u32, height: u32) -> Texture {
        let desc = TextureDescriptor::new();
        desc.set_width(width.max(1) as u64);
        desc.set_height(height.max(1) as u64);
        desc.set_pixel_format(self.pixel_format);
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        desc.set_storage_mode(MTLStorageMode::Private);

        self.device.new_texture(&desc)
    }

    /// Clear a window's texture to a color
    pub fn clear(
        &self,
        command_queue: &CommandQueue,
        window_id: u32,
        color: MTLClearColor,
    ) {
        if let Some(wt) = self.textures.get(&window_id) {
            let cmd = command_queue.new_command_buffer();
            let desc = RenderPassDescriptor::new();

            let attachment = desc.color_attachments().object_at(0).unwrap();
            attachment.set_texture(Some(&wt.texture));
            attachment.set_load_action(MTLLoadAction::Clear);
            attachment.set_store_action(MTLStoreAction::Store);
            attachment.set_clear_color(color);

            let encoder = cmd.new_render_command_encoder(&desc);
            encoder.end_encoding();

            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    /// Create a render pass for rendering to a window's texture
    pub fn begin_render_pass(&self, window_id: u32) -> Option<&RenderPassDescriptor> {
        // Note: RenderPassDescriptor doesn't work well with map() due to lifetime issues
        // Callers should use get() and create their own render pass
        None  // TODO: Implement proper render pass creation
    }

    /// Get texture info for creating a render pass
    pub fn texture_info(&self, window_id: u32) -> Option<(&Texture, u32, u32)> {
        self.textures.get(&window_id).map(|wt| (&wt.texture, wt.width, wt.height))
    }

    /// Get all window IDs that have textures
    pub fn window_ids(&self) -> Vec<u32> {
        self.textures.keys().copied().collect()
    }

    /// Get texture dimensions for a window
    pub fn dimensions(&self, window_id: u32) -> Option<(u32, u32)> {
        self.textures.get(&window_id).map(|wt| (wt.width, wt.height))
    }
}

/// Array of window textures for GPU access
///
/// Used to pass multiple window textures to the compositor shader.
pub struct WindowTextureArray {
    /// Texture array (up to MAX_WINDOWS textures)
    textures: Vec<Texture>,
    /// Mapping from window_id to array index
    id_to_index: HashMap<u32, usize>,
}

impl WindowTextureArray {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            id_to_index: HashMap::new(),
        }
    }

    /// Update the array from WindowTextures
    pub fn update_from(&mut self, window_textures: &WindowTextures, window_ids: &[u32]) {
        self.textures.clear();
        self.id_to_index.clear();

        for (idx, &id) in window_ids.iter().enumerate() {
            if let Some(tex) = window_textures.get(id) {
                self.textures.push(tex.clone());
                self.id_to_index.insert(id, idx);
            }
        }
    }

    /// Get index for a window ID
    pub fn index_for(&self, window_id: u32) -> Option<usize> {
        self.id_to_index.get(&window_id).copied()
    }

    /// Get textures slice for binding
    pub fn textures(&self) -> &[Texture] {
        &self.textures
    }
}

impl Default for WindowTextureArray {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_textures_create() {
        let device = Device::system_default().expect("No Metal device");
        let mut wt = WindowTextures::new(&device, MTLPixelFormat::BGRA8Unorm);

        let tex = wt.get_or_create(1, 400, 300);
        assert_eq!(tex.width(), 400);
        assert_eq!(tex.height(), 300);
    }

    #[test]
    fn test_window_textures_resize() {
        let device = Device::system_default().expect("No Metal device");
        let mut wt = WindowTextures::new(&device, MTLPixelFormat::BGRA8Unorm);

        wt.get_or_create(1, 400, 300);
        assert_eq!(wt.dimensions(1), Some((400, 300)));

        // Resize
        wt.get_or_create(1, 800, 600);
        assert_eq!(wt.dimensions(1), Some((800, 600)));
    }

    #[test]
    fn test_window_textures_remove() {
        let device = Device::system_default().expect("No Metal device");
        let mut wt = WindowTextures::new(&device, MTLPixelFormat::BGRA8Unorm);

        wt.get_or_create(1, 400, 300);
        assert!(wt.get(1).is_some());

        wt.remove(1);
        assert!(wt.get(1).is_none());
    }
}
