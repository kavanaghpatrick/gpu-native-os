//! Test suite for Issue #65: Document Image Integration
//!
//! Tests GPU-accelerated image processing in documents:
//! - Image attribute extraction (src, width, height)
//! - Image atlas management
//! - Image vertices in paint pipeline
//! - Layout integration with images

use metal::{Device, MTLResourceOptions};
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuImageAttributeExtractor, GpuImageAtlas,
    GpuImageLoader, ImageInfo, ParsedImage, ELEM_IMG,
};

// ======= IMAGE ATTRIBUTE EXTRACTION =======

#[test]
fn test_image_attribute_extractor_creation() {
    let device = Device::system_default().expect("No Metal device");
    let extractor = GpuImageAttributeExtractor::new(&device, 100);
    assert!(extractor.is_ok(), "Should create image attribute extractor");
}

#[test]
fn test_extract_simple_img_tag() {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");
    let mut extractor = GpuImageAttributeExtractor::new(&device, 100).expect("Failed to create extractor");

    let html = b"<img src=\"test.png\">";
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);

    // Create buffers for GPU extraction
    let elements_buffer = device.new_buffer_with_data(
        elements.as_ptr() as *const _,
        (elements.len() * std::mem::size_of_val(&elements[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let tokens_buffer = device.new_buffer_with_data(
        tokens.as_ptr() as *const _,
        (tokens.len() * std::mem::size_of_val(&tokens[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let html_buffer = device.new_buffer_with_data(
        html.as_ptr() as *const _,
        html.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let images = extractor.extract(
        &elements_buffer,
        elements.len() as u32,
        &tokens_buffer,
        &html_buffer,
        html.len() as u32,
    );

    assert_eq!(images.len(), 1, "Should extract 1 image");

    // Verify src extraction
    let src = extractor.get_src(&images[0], html);
    assert_eq!(src, b"test.png", "Should extract correct src");
}

#[test]
fn test_extract_img_with_dimensions() {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");
    let mut extractor = GpuImageAttributeExtractor::new(&device, 100).expect("Failed to create extractor");

    let html = b"<img src=\"photo.jpg\" width=\"200\" height=\"150\">";
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);

    let elements_buffer = device.new_buffer_with_data(
        elements.as_ptr() as *const _,
        (elements.len() * std::mem::size_of_val(&elements[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tokens_buffer = device.new_buffer_with_data(
        tokens.as_ptr() as *const _,
        (tokens.len() * std::mem::size_of_val(&tokens[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let html_buffer = device.new_buffer_with_data(
        html.as_ptr() as *const _,
        html.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let images = extractor.extract(
        &elements_buffer,
        elements.len() as u32,
        &tokens_buffer,
        &html_buffer,
        html.len() as u32,
    );

    assert_eq!(images.len(), 1, "Should extract 1 image");
    assert_eq!(images[0].width, 200, "Should extract width=200");
    assert_eq!(images[0].height, 150, "Should extract height=150");
}

#[test]
fn test_extract_multiple_images() {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");
    let mut extractor = GpuImageAttributeExtractor::new(&device, 100).expect("Failed to create extractor");

    let html = b"<div><img src=\"a.png\"><img src=\"b.png\"><img src=\"c.png\"></div>";
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);

    let elements_buffer = device.new_buffer_with_data(
        elements.as_ptr() as *const _,
        (elements.len() * std::mem::size_of_val(&elements[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tokens_buffer = device.new_buffer_with_data(
        tokens.as_ptr() as *const _,
        (tokens.len() * std::mem::size_of_val(&tokens[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let html_buffer = device.new_buffer_with_data(
        html.as_ptr() as *const _,
        html.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let images = extractor.extract(
        &elements_buffer,
        elements.len() as u32,
        &tokens_buffer,
        &html_buffer,
        html.len() as u32,
    );

    assert_eq!(images.len(), 3, "Should extract 3 images");
}

#[test]
fn test_extract_img_with_single_quotes() {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");
    let mut extractor = GpuImageAttributeExtractor::new(&device, 100).expect("Failed to create extractor");

    let html = b"<img src='image.png' width='100'>";
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);

    let elements_buffer = device.new_buffer_with_data(
        elements.as_ptr() as *const _,
        (elements.len() * std::mem::size_of_val(&elements[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tokens_buffer = device.new_buffer_with_data(
        tokens.as_ptr() as *const _,
        (tokens.len() * std::mem::size_of_val(&tokens[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let html_buffer = device.new_buffer_with_data(
        html.as_ptr() as *const _,
        html.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let images = extractor.extract(
        &elements_buffer,
        elements.len() as u32,
        &tokens_buffer,
        &html_buffer,
        html.len() as u32,
    );

    assert_eq!(images.len(), 1, "Should extract 1 image");
    let src = extractor.get_src(&images[0], html);
    assert_eq!(src, b"image.png", "Should extract src with single quotes");
    assert_eq!(images[0].width, 100, "Should extract width with single quotes");
}

// ======= IMAGE LOADER =======

#[test]
fn test_image_loader_creation() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device);
    assert!(loader.is_ok(), "Should create image loader");
}

#[test]
fn test_load_from_rgba() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");

    // Create a simple 2x2 RGBA image (red pixel pattern)
    let width = 2u32;
    let height = 2u32;
    let data: Vec<u8> = vec![
        255, 0, 0, 255,   // Red
        0, 255, 0, 255,   // Green
        0, 0, 255, 255,   // Blue
        255, 255, 0, 255, // Yellow
    ];

    let result = loader.load_from_rgba(&data, width, height);
    assert!(result.is_ok(), "Should load RGBA data");

    let texture = result.unwrap();
    assert_eq!(texture.width(), 2);
    assert_eq!(texture.height(), 2);
}

#[test]
fn test_load_rgba_wrong_size() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");

    // Wrong data size (3 bytes instead of 4 per pixel)
    let data: Vec<u8> = vec![255, 0, 0];
    let result = loader.load_from_rgba(&data, 2, 2);

    assert!(result.is_err(), "Should fail with wrong data size");
}

// ======= IMAGE ATLAS =======

#[test]
fn test_atlas_creation() {
    let device = Device::system_default().expect("No Metal device");
    let atlas = GpuImageAtlas::new(&device);
    assert!(atlas.is_ok(), "Should create image atlas");
}

#[test]
fn test_atlas_add_image() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");
    let mut atlas = GpuImageAtlas::new(&device).expect("Failed to create atlas");
    let queue = device.new_command_queue();

    // Create a test texture
    let data: Vec<u8> = vec![255u8; 64 * 64 * 4]; // 64x64 white image
    let texture = loader.load_from_rgba(&data, 64, 64).expect("Failed to load texture");

    let result = atlas.add_image(&texture, "test.png", &queue);
    assert!(result.is_ok(), "Should add image to atlas");

    let info = result.unwrap();
    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);
    assert_eq!(info.atlas_x, 0);
    assert_eq!(info.atlas_y, 0);
}

#[test]
fn test_atlas_get_image_by_path() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");
    let mut atlas = GpuImageAtlas::new(&device).expect("Failed to create atlas");
    let queue = device.new_command_queue();

    let data: Vec<u8> = vec![255u8; 32 * 32 * 4];
    let texture = loader.load_from_rgba(&data, 32, 32).expect("Failed to load texture");
    atlas.add_image(&texture, "lookup.png", &queue).expect("Failed to add");

    let found = atlas.get_image_by_path("lookup.png");
    assert!(found.is_some(), "Should find image by path");
    assert_eq!(found.unwrap().width, 32);

    let not_found = atlas.get_image_by_path("nonexistent.png");
    assert!(not_found.is_none(), "Should not find nonexistent image");
}

#[test]
fn test_atlas_uvs() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");
    let mut atlas = GpuImageAtlas::new(&device).expect("Failed to create atlas");
    let queue = device.new_command_queue();

    let data: Vec<u8> = vec![255u8; 100 * 100 * 4];
    let texture = loader.load_from_rgba(&data, 100, 100).expect("Failed to load texture");
    let info = atlas.add_image(&texture, "uv_test.png", &queue).expect("Failed to add");

    let uvs = atlas.get_uvs(info.id);
    assert!(uvs.is_some(), "Should get UVs");

    let (uv_min, uv_max) = uvs.unwrap();

    // Atlas is 4096x4096, image is 100x100 at (0,0)
    assert!((uv_min[0] - 0.0).abs() < 0.001, "u0 should be ~0");
    assert!((uv_min[1] - 0.0).abs() < 0.001, "v0 should be ~0");
    assert!((uv_max[0] - 100.0/4096.0).abs() < 0.001, "u1 should be ~0.0244");
    assert!((uv_max[1] - 100.0/4096.0).abs() < 0.001, "v1 should be ~0.0244");
}

#[test]
fn test_atlas_packing() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");
    let mut atlas = GpuImageAtlas::new(&device).expect("Failed to create atlas");
    let queue = device.new_command_queue();

    // Add multiple images
    let data: Vec<u8> = vec![255u8; 128 * 128 * 4];
    let tex1 = loader.load_from_rgba(&data, 128, 128).expect("tex1");
    let tex2 = loader.load_from_rgba(&data, 128, 128).expect("tex2");
    let tex3 = loader.load_from_rgba(&data, 128, 128).expect("tex3");

    let info1 = atlas.add_image(&tex1, "img1.png", &queue).expect("add1");
    let info2 = atlas.add_image(&tex2, "img2.png", &queue).expect("add2");
    let info3 = atlas.add_image(&tex3, "img3.png", &queue).expect("add3");

    // Images should be placed side by side in row-based packing
    assert_eq!(info1.atlas_x, 0);
    assert_eq!(info1.atlas_y, 0);

    assert_eq!(info2.atlas_x, 128);
    assert_eq!(info2.atlas_y, 0);

    assert_eq!(info3.atlas_x, 256);
    assert_eq!(info3.atlas_y, 0);
}

#[test]
fn test_atlas_deduplication() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");
    let mut atlas = GpuImageAtlas::new(&device).expect("Failed to create atlas");
    let queue = device.new_command_queue();

    let data: Vec<u8> = vec![255u8; 64 * 64 * 4];
    let texture = loader.load_from_rgba(&data, 64, 64).expect("Failed to load");

    // Add same path twice
    let info1 = atlas.add_image(&texture, "same.png", &queue).expect("add1");
    let info2 = atlas.add_image(&texture, "same.png", &queue).expect("add2");

    // Should return same image info (deduplicated)
    assert_eq!(info1.id, info2.id, "Same path should return same image");
    assert_eq!(atlas.images().len(), 1, "Atlas should only have 1 image");
}

// ======= PARSER IMG ELEMENT =======

#[test]
fn test_parser_creates_img_elements() {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");

    let html = b"<div><img src=\"test.png\"></div>";
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);

    // Find IMG element
    let img_elem = elements.iter().find(|e| e.element_type == ELEM_IMG);
    assert!(img_elem.is_some(), "Should parse img element");
}

// ======= IMAGE INFO STRUCT =======

#[test]
fn test_image_info_size() {
    assert_eq!(std::mem::size_of::<ImageInfo>(), 32, "ImageInfo should be 32 bytes");
}

#[test]
fn test_parsed_image_size() {
    assert_eq!(std::mem::size_of::<ParsedImage>(), 32, "ParsedImage should be 32 bytes");
}

// ======= PERFORMANCE =======

#[test]
fn test_extraction_performance() {
    let device = Device::system_default().expect("No Metal device");
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");
    let mut extractor = GpuImageAttributeExtractor::new(&device, 1000).expect("Failed to create extractor");

    // Generate HTML with many images
    let mut html = String::from("<div>");
    for i in 0..500 {
        html.push_str(&format!("<img src=\"img{}.png\" width=\"{}\" height=\"{}\">", i, i % 200 + 50, i % 150 + 50));
    }
    html.push_str("</div>");
    let html = html.into_bytes();

    let tokens = tokenizer.tokenize(&html);
    let (elements, _) = parser.parse(&tokens, &html);

    let elements_buffer = device.new_buffer_with_data(
        elements.as_ptr() as *const _,
        (elements.len() * std::mem::size_of_val(&elements[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let tokens_buffer = device.new_buffer_with_data(
        tokens.as_ptr() as *const _,
        (tokens.len() * std::mem::size_of_val(&tokens[0])) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let html_buffer = device.new_buffer_with_data(
        html.as_ptr() as *const _,
        html.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    let _ = extractor.extract(
        &elements_buffer,
        elements.len() as u32,
        &tokens_buffer,
        &html_buffer,
        html.len() as u32,
    );

    // Timed run
    let start = std::time::Instant::now();
    let images = extractor.extract(
        &elements_buffer,
        elements.len() as u32,
        &tokens_buffer,
        &html_buffer,
        html.len() as u32,
    );
    let elapsed = start.elapsed();

    println!("Extracted {} images in {:?}", images.len(), elapsed);
    assert_eq!(images.len(), 500, "Should extract 500 images");
    assert!(elapsed.as_millis() < 50, "Extraction should be <50ms, got {:?}", elapsed);
}

#[test]
fn test_atlas_performance() {
    let device = Device::system_default().expect("No Metal device");
    let loader = GpuImageLoader::new(&device).expect("Failed to create loader");
    let mut atlas = GpuImageAtlas::new(&device).expect("Failed to create atlas");
    let queue = device.new_command_queue();

    // Create test textures
    let data: Vec<u8> = vec![255u8; 64 * 64 * 4];
    let textures: Vec<_> = (0..100)
        .map(|_| loader.load_from_rgba(&data, 64, 64).expect("Failed to load"))
        .collect();

    // Warmup
    let _ = atlas.add_image(&textures[0], "warmup.png", &queue);

    // Timed run
    let start = std::time::Instant::now();
    for (i, tex) in textures.iter().enumerate() {
        atlas.add_image(tex, &format!("perf_{}.png", i), &queue).expect("Failed to add");
    }
    let elapsed = start.elapsed();

    println!("Added {} images to atlas in {:?}", textures.len(), elapsed);
    assert!(elapsed.as_millis() < 500, "Atlas additions should be <500ms, got {:?}", elapsed);
}
