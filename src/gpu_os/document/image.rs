//! GPU-Native Image Loading and Atlas Management
//!
//! Uses Metal's native texture loading for GPU-accelerated PNG/JPEG decoding.
//! No CPU-side image processing - everything happens on GPU.

use metal::*;
use metal::foreign_types::ForeignType;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;

/// Maximum atlas dimensions (4096x4096 = 64MB for RGBA8)
pub const ATLAS_WIDTH: u32 = 4096;
pub const ATLAS_HEIGHT: u32 = 4096;

/// Image info stored in GPU buffer for shader access
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageInfo {
    /// Unique image ID
    pub id: u32,
    /// Original image width in pixels
    pub width: u32,
    /// Original image height in pixels
    pub height: u32,
    /// Pixel format (0 = RGBA8, 1 = BGRA8)
    pub format: u32,
    /// X position in atlas
    pub atlas_x: u32,
    /// Y position in atlas
    pub atlas_y: u32,
    /// Width in atlas (may be scaled)
    pub atlas_width: u32,
    /// Height in atlas (may be scaled)
    pub atlas_height: u32,
}

/// Reference to an image from an element
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageRef {
    /// Image ID (index into images array)
    pub image_id: u32,
    /// Specified width (0 = auto/intrinsic)
    pub width: f32,
    /// Specified height (0 = auto/intrinsic)
    pub height: f32,
    /// Object-fit mode: 0=fill, 1=contain, 2=cover, 3=none, 4=scale-down
    pub object_fit: u32,
}

/// Object-fit modes
pub const OBJECT_FIT_FILL: u32 = 0;
pub const OBJECT_FIT_CONTAIN: u32 = 1;
pub const OBJECT_FIT_COVER: u32 = 2;
pub const OBJECT_FIT_NONE: u32 = 3;
pub const OBJECT_FIT_SCALE_DOWN: u32 = 4;

/// Rectangle for atlas packing
#[derive(Clone, Copy, Debug)]
struct AtlasRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// GPU-native image loader using Metal's MTKTextureLoader
/// MTKTextureLoader is optional - RGBA loading works without it
pub struct GpuImageLoader {
    device: Device,
    texture_loader: Option<*mut Object>,
}

impl GpuImageLoader {
    /// Create a new GPU image loader
    /// Note: MTKTextureLoader requires MetalKit framework to be linked.
    /// If not available, file loading will fail but RGBA loading will work.
    pub fn new(device: &Device) -> Result<Self, String> {
        // Try to create MTKTextureLoader via objc (may fail if MetalKit not linked)
        let texture_loader: Option<*mut Object> = unsafe {
            // Try to get the class - this will return null if MetalKit isn't linked
            let cls: *mut Object = objc::runtime::Class::get("MTKTextureLoader")
                .map(|c| c as *const _ as *mut Object)
                .unwrap_or(std::ptr::null_mut());

            if cls.is_null() {
                None
            } else {
                let loader: *mut Object = msg_send![cls, alloc];
                let loader: *mut Object = msg_send![loader, initWithDevice: device.as_ptr()];
                if loader.is_null() {
                    None
                } else {
                    Some(loader)
                }
            }
        };

        Ok(Self {
            device: device.clone(),
            texture_loader,
        })
    }

    /// Check if file loading is available (requires MetalKit)
    pub fn can_load_files(&self) -> bool {
        self.texture_loader.is_some()
    }

    /// Load an image from a file path directly to GPU texture
    /// Returns the texture and image dimensions
    /// Requires MetalKit to be linked (check with can_load_files())
    pub fn load_from_file(&self, path: &Path) -> Result<(Texture, u32, u32), String> {
        let texture_loader = self.texture_loader
            .ok_or("MTKTextureLoader not available - MetalKit may not be linked")?;

        let path_str = path.to_str().ok_or("Invalid path")?;

        // Create NSURL from path
        let url: *mut Object = unsafe {
            let path_nsstring = create_nsstring(path_str);
            let cls = class!(NSURL);
            let url: *mut Object = msg_send![cls, fileURLWithPath: path_nsstring];
            url
        };

        if url.is_null() {
            return Err(format!("Failed to create URL for path: {}", path_str));
        }

        // Load texture synchronously using MTKTextureLoader
        let texture: *mut Object = unsafe {
            let options: *mut Object = std::ptr::null_mut();
            let mut error: *mut Object = std::ptr::null_mut();

            let tex: *mut Object = msg_send![
                texture_loader,
                newTextureWithContentsOfURL: url
                options: options
                error: &mut error
            ];

            if !error.is_null() || tex.is_null() {
                return Err(format!("Failed to load texture from: {}", path_str));
            }
            tex
        };

        // Get texture dimensions
        let width: u64 = unsafe { msg_send![texture, width] };
        let height: u64 = unsafe { msg_send![texture, height] };

        // Wrap in metal-rs Texture
        let texture = unsafe { Texture::from_ptr(texture as *mut _) };

        Ok((texture, width as u32, height as u32))
    }

    /// Load image from raw RGBA data (for programmatic images)
    /// Does not require MetalKit - always available
    pub fn load_from_rgba(&self, data: &[u8], width: u32, height: u32) -> Result<Texture, String> {
        if data.len() != (width * height * 4) as usize {
            return Err("Data size doesn't match dimensions".to_string());
        }

        let desc = TextureDescriptor::new();
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_usage(MTLTextureUsage::ShaderRead);
        desc.set_storage_mode(MTLStorageMode::Shared);

        let texture = self.device.new_texture(&desc);
        texture.replace_region(
            MTLRegion::new_2d(0, 0, width as u64, height as u64),
            0,
            data.as_ptr() as *const _,
            (width * 4) as u64,
        );

        Ok(texture)
    }
}

impl Drop for GpuImageLoader {
    fn drop(&mut self) {
        if let Some(loader) = self.texture_loader {
            if !loader.is_null() {
                unsafe {
                    let _: () = msg_send![loader, release];
                }
            }
        }
    }
}

/// GPU texture atlas for efficient image rendering
pub struct GpuImageAtlas {
    device: Device,
    /// The atlas texture
    pub texture: Texture,
    /// Current packing state (simple row-based for now)
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    /// Loaded images
    images: Vec<ImageInfo>,
    /// Path to image ID mapping
    path_map: HashMap<String, u32>,
    /// Compute pipeline for blitting
    blit_pipeline: ComputePipelineState,
}

impl GpuImageAtlas {
    /// Create a new GPU image atlas
    pub fn new(device: &Device) -> Result<Self, String> {
        // Create atlas texture
        let desc = TextureDescriptor::new();
        desc.set_width(ATLAS_WIDTH as u64);
        desc.set_height(ATLAS_HEIGHT as u64);
        desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        desc.set_storage_mode(MTLStorageMode::Private); // GPU-only for performance

        let texture = device.new_texture(&desc);

        // Create blit compute pipeline for copying images to atlas
        let shader_source = BLIT_SHADER;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_source, &options)
            .map_err(|e| format!("Failed to compile blit shader: {}", e))?;

        let blit_fn = library
            .get_function("blit_to_atlas", None)
            .map_err(|e| format!("Failed to get blit function: {}", e))?;

        let blit_pipeline = device
            .new_compute_pipeline_state_with_function(&blit_fn)
            .map_err(|e| format!("Failed to create blit pipeline: {}", e))?;

        Ok(Self {
            device: device.clone(),
            texture,
            cursor_x: 0,
            cursor_y: 0,
            row_height: 0,
            images: Vec::new(),
            path_map: HashMap::new(),
            blit_pipeline,
        })
    }

    /// Add an image to the atlas
    /// Returns the ImageInfo with atlas coordinates
    pub fn add_image(
        &mut self,
        source_texture: &Texture,
        path: &str,
        command_queue: &CommandQueue,
    ) -> Result<ImageInfo, String> {
        // Check if already loaded
        if let Some(&id) = self.path_map.get(path) {
            return Ok(self.images[id as usize]);
        }

        let width = source_texture.width() as u32;
        let height = source_texture.height() as u32;

        // Simple row-based packing
        if self.cursor_x + width > ATLAS_WIDTH {
            // Move to next row
            self.cursor_x = 0;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }

        if self.cursor_y + height > ATLAS_HEIGHT {
            return Err("Atlas is full".to_string());
        }

        let atlas_x = self.cursor_x;
        let atlas_y = self.cursor_y;

        // Blit source texture to atlas position using GPU compute
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.blit_pipeline);
        encoder.set_texture(0, Some(source_texture));
        encoder.set_texture(1, Some(&self.texture));

        // Pass offset as buffer
        let params: [u32; 4] = [atlas_x, atlas_y, width, height];
        let params_buffer = self.device.new_buffer_with_data(
            params.as_ptr() as *const _,
            16,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(0, Some(&params_buffer), 0);

        let threads_per_group = MTLSize::new(16, 16, 1);
        let thread_groups = MTLSize::new(
            (width as u64 + 15) / 16,
            (height as u64 + 15) / 16,
            1,
        );
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Update packing state
        self.cursor_x += width;
        self.row_height = self.row_height.max(height);

        // Create image info
        let id = self.images.len() as u32;
        let info = ImageInfo {
            id,
            width,
            height,
            format: 0, // RGBA8
            atlas_x,
            atlas_y,
            atlas_width: width,
            atlas_height: height,
        };

        self.images.push(info);
        self.path_map.insert(path.to_string(), id);

        Ok(info)
    }

    /// Get image info by ID
    pub fn get_image(&self, id: u32) -> Option<&ImageInfo> {
        self.images.get(id as usize)
    }

    /// Get image info by path
    pub fn get_image_by_path(&self, path: &str) -> Option<&ImageInfo> {
        self.path_map.get(path).and_then(|&id| self.get_image(id))
    }

    /// Get all images
    pub fn images(&self) -> &[ImageInfo] {
        &self.images
    }

    /// Get atlas texture
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// Calculate UV coordinates for an image
    pub fn get_uvs(&self, image_id: u32) -> Option<([f32; 2], [f32; 2])> {
        self.get_image(image_id).map(|info| {
            let u0 = info.atlas_x as f32 / ATLAS_WIDTH as f32;
            let v0 = info.atlas_y as f32 / ATLAS_HEIGHT as f32;
            let u1 = (info.atlas_x + info.atlas_width) as f32 / ATLAS_WIDTH as f32;
            let v1 = (info.atlas_y + info.atlas_height) as f32 / ATLAS_HEIGHT as f32;
            ([u0, v0], [u1, v1])
        })
    }
}

/// Helper to create NSString from Rust string
unsafe fn create_nsstring(s: &str) -> *mut Object {
    let cls = class!(NSString);
    let nsstring: *mut Object = msg_send![
        cls,
        stringWithUTF8String: CString::new(s).unwrap().as_ptr()
    ];
    nsstring
}

/// Metal shader for blitting images to atlas
const BLIT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BlitParams {
    uint offset_x;
    uint offset_y;
    uint width;
    uint height;
};

kernel void blit_to_atlas(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> dest [[texture(1)]],
    constant BlitParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 color = source.read(gid);
    uint2 dest_pos = uint2(params.offset_x + gid.x, params.offset_y + gid.y);
    dest.write(color, dest_pos);
}
"#;

/// GPU-based image attribute extractor
/// Extracts src, width, height from img tags using GPU compute
pub struct GpuImageAttributeExtractor {
    device: Device,
    command_queue: CommandQueue,
    extract_pipeline: ComputePipelineState,
    image_refs_buffer: Buffer,
    image_count_buffer: Buffer,
    max_images: usize,
}

/// Parsed image element with path reference
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParsedImage {
    /// Element index in the elements array
    pub element_index: u32,
    /// Start of src attribute value in HTML
    pub src_start: u32,
    /// Length of src attribute value
    pub src_length: u32,
    /// Width attribute (0 = not specified)
    pub width: u32,
    /// Height attribute (0 = not specified)
    pub height: u32,
    /// Padding for alignment
    pub _padding: [u32; 3],
}

impl GpuImageAttributeExtractor {
    /// Create a new GPU image attribute extractor
    pub fn new(device: &Device, max_images: usize) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shader
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(IMAGE_ATTRIBUTE_SHADER, &options)
            .map_err(|e| format!("Failed to compile image attribute shader: {}", e))?;

        let extract_fn = library
            .get_function("extract_image_attributes", None)
            .map_err(|e| format!("Failed to get extract_image_attributes: {}", e))?;

        let extract_pipeline = device
            .new_compute_pipeline_state_with_function(&extract_fn)
            .map_err(|e| format!("Failed to create extract pipeline: {}", e))?;

        // Allocate buffers
        let image_refs_buffer = device.new_buffer(
            (max_images * std::mem::size_of::<ParsedImage>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let image_count_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            device: device.clone(),
            command_queue,
            extract_pipeline,
            image_refs_buffer,
            image_count_buffer,
            max_images,
        })
    }

    /// Extract image attributes from parsed elements
    /// Returns parsed images with src path positions and dimensions
    pub fn extract(
        &mut self,
        elements_buffer: &Buffer,
        element_count: u32,
        tokens_buffer: &Buffer,
        html_buffer: &Buffer,
        html_len: u32,
    ) -> Vec<ParsedImage> {
        // Reset count
        unsafe {
            let ptr = self.image_count_buffer.contents() as *mut u32;
            *ptr = 0;
        }

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.extract_pipeline);
        encoder.set_buffer(0, Some(elements_buffer), 0);
        encoder.set_buffer(1, Some(tokens_buffer), 0);
        encoder.set_buffer(2, Some(html_buffer), 0);
        encoder.set_buffer(3, Some(&self.image_refs_buffer), 0);
        encoder.set_buffer(4, Some(&self.image_count_buffer), 0);

        // Pass params
        let params: [u32; 4] = [element_count, html_len, self.max_images as u32, 0];
        let params_buffer = self.device.new_buffer_with_data(
            params.as_ptr() as *const _,
            16,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&params_buffer), 0);

        let threads = MTLSize::new(((element_count + 255) / 256 * 256) as u64, 1, 1);
        let threads_per_group = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, threads_per_group);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let count = unsafe {
            let ptr = self.image_count_buffer.contents() as *const u32;
            (*ptr as usize).min(self.max_images)
        };

        let images: Vec<ParsedImage> = unsafe {
            let ptr = self.image_refs_buffer.contents() as *const ParsedImage;
            (0..count).map(|i| *ptr.add(i)).collect()
        };

        images
    }

    /// Get src path from parsed image using the original HTML
    pub fn get_src<'a>(&self, image: &ParsedImage, html: &'a [u8]) -> &'a [u8] {
        let start = image.src_start as usize;
        let end = start + image.src_length as usize;
        if end <= html.len() {
            &html[start..end]
        } else {
            b""
        }
    }
}

/// Metal shader for extracting image attributes
const IMAGE_ATTRIBUTE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Element type for IMG
constant uint ELEM_IMG = 14;

struct Element {
    uint element_type;
    int parent;
    int first_child;
    int next_sibling;
    uint text_start;
    uint text_length;
    uint token_index;
    uint _padding;
};

struct Token {
    uint token_type;
    uint start;
    uint end;
    uint _padding;
};

struct ParsedImage {
    uint element_index;
    uint src_start;
    uint src_length;
    uint width;
    uint height;
    uint _padding[3];
};

struct Params {
    uint element_count;
    uint html_len;
    uint max_images;
    uint _padding;
};

// Parse integer from ASCII digits in HTML
uint parse_uint(device const uchar* html, uint start, uint len) {
    uint value = 0;
    for (uint i = 0; i < len && i < 10; i++) {
        uchar c = html[start + i];
        if (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
        } else {
            break;
        }
    }
    return value;
}

// Extract attribute value after finding "name="
// Returns start position and sets length via pointer
uint extract_attr_value(device const uchar* html, uint pos, uint tag_end, thread uint* out_len) {
    *out_len = 0;

    // Skip quote if present
    uchar quote = html[pos];
    if (quote == '"' || quote == '\'') {
        pos++;
        // Find closing quote
        uint val_end = pos;
        while (val_end < tag_end && html[val_end] != quote) {
            val_end++;
        }
        *out_len = val_end - pos;
        return pos;
    } else {
        // Unquoted - read until space or >
        uint val_end = pos;
        while (val_end < tag_end && html[val_end] != ' ' &&
               html[val_end] != '>' && html[val_end] != '/') {
            val_end++;
        }
        *out_len = val_end - pos;
        return pos;
    }
}

// Find "src=" attribute and return value position
uint find_src_attr(device const uchar* html, uint tag_start, uint tag_end, thread uint* out_len) {
    *out_len = 0;
    for (uint i = tag_start; i + 4 < tag_end; i++) {
        if (html[i] == 's' && html[i+1] == 'r' && html[i+2] == 'c' && html[i+3] == '=') {
            return extract_attr_value(html, i + 4, tag_end, out_len);
        }
    }
    return 0;
}

// Find "width=" attribute and return value position
uint find_width_attr(device const uchar* html, uint tag_start, uint tag_end, thread uint* out_len) {
    *out_len = 0;
    for (uint i = tag_start; i + 6 < tag_end; i++) {
        if (html[i] == 'w' && html[i+1] == 'i' && html[i+2] == 'd' &&
            html[i+3] == 't' && html[i+4] == 'h' && html[i+5] == '=') {
            return extract_attr_value(html, i + 6, tag_end, out_len);
        }
    }
    return 0;
}

// Find "height=" attribute and return value position
uint find_height_attr(device const uchar* html, uint tag_start, uint tag_end, thread uint* out_len) {
    *out_len = 0;
    for (uint i = tag_start; i + 7 < tag_end; i++) {
        if (html[i] == 'h' && html[i+1] == 'e' && html[i+2] == 'i' &&
            html[i+3] == 'g' && html[i+4] == 'h' && html[i+5] == 't' && html[i+6] == '=') {
            return extract_attr_value(html, i + 7, tag_end, out_len);
        }
    }
    return 0;
}

kernel void extract_image_attributes(
    device const Element* elements [[buffer(0)]],
    device const Token* tokens [[buffer(1)]],
    device const uchar* html [[buffer(2)]],
    device ParsedImage* images [[buffer(3)]],
    device atomic_uint* image_count [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_IMG) return;

    // Get token for this element
    Token token = tokens[elem.token_index];
    uint tag_start = token.start;
    uint tag_end = token.end;

    // Allocate output slot
    uint idx = atomic_fetch_add_explicit(image_count, 1, memory_order_relaxed);
    if (idx >= params.max_images) return;

    ParsedImage img;
    img.element_index = gid;
    img.src_start = 0;
    img.src_length = 0;
    img.width = 0;
    img.height = 0;
    img._padding[0] = 0;
    img._padding[1] = 0;
    img._padding[2] = 0;

    // Extract src attribute
    uint src_len = 0;
    uint src_start = find_src_attr(html, tag_start, tag_end, &src_len);
    if (src_len > 0) {
        img.src_start = src_start;
        img.src_length = src_len;
    }

    // Extract width attribute
    uint width_len = 0;
    uint width_start = find_width_attr(html, tag_start, tag_end, &width_len);
    if (width_len > 0) {
        img.width = parse_uint(html, width_start, width_len);
    }

    // Extract height attribute
    uint height_len = 0;
    uint height_start = find_height_attr(html, tag_start, tag_end, &height_len);
    if (height_len > 0) {
        img.height = parse_uint(html, height_start, height_len);
    }

    images[idx] = img;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_info_size() {
        assert_eq!(std::mem::size_of::<ImageInfo>(), 32);
    }

    #[test]
    fn test_image_ref_size() {
        assert_eq!(std::mem::size_of::<ImageRef>(), 16);
    }

    #[test]
    fn test_parsed_image_size() {
        assert_eq!(std::mem::size_of::<ParsedImage>(), 32);
    }
}
