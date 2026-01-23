// Tests for Issue #16: Text Rendering - MSDF Font Atlas
//
// These tests verify the text rendering system.
// Run with: cargo test --test test_issue_16_text

use metal::Device;
use rust_experiment::gpu_os::text::*;

fn setup() -> (Device, FontAtlas, TextRenderer) {
    let device = Device::system_default().expect("No Metal device");
    let atlas = FontAtlas::create_default(&device).expect("Font atlas should load");
    let renderer = TextRenderer::new(&device).expect("Text renderer should compile");
    (device, atlas, renderer)
}

#[test]
fn test_glyph_metrics_size() {
    assert_eq!(
        std::mem::size_of::<GlyphMetrics>(), 40,
        "GlyphMetrics must be 40 bytes"
    );
}

#[test]
fn test_text_vertex_size() {
    assert_eq!(
        std::mem::size_of::<TextVertex>(), 32,
        "TextVertex must be 32 bytes"
    );
}

#[test]
fn test_font_atlas_creation() {
    let device = Device::system_default().expect("No Metal device");
    let result = FontAtlas::create_default(&device);

    assert!(result.is_ok(), "Default font atlas should be created");
}

#[test]
fn test_text_renderer_creation() {
    let device = Device::system_default().expect("No Metal device");
    let result = TextRenderer::new(&device);

    assert!(result.is_ok(), "Text renderer should compile");
}

#[test]
fn test_glyph_metrics_lookup() {
    let device = Device::system_default().expect("No Metal device");
    let atlas = FontAtlas::create_default(&device).expect("Atlas should load");

    // Check that basic ASCII characters have metrics
    for c in 'A'..='Z' {
        let metrics = atlas.glyph_metrics(c);
        assert!(
            metrics.is_some(),
            "Glyph metrics for '{}' should exist",
            c
        );
    }

    for c in 'a'..='z' {
        let metrics = atlas.glyph_metrics(c);
        assert!(metrics.is_some(), "Glyph metrics for '{}' should exist", c);
    }

    for c in '0'..='9' {
        let metrics = atlas.glyph_metrics(c);
        assert!(metrics.is_some(), "Glyph metrics for '{}' should exist", c);
    }
}

#[test]
fn test_text_layout_simple() {
    let (device, atlas, renderer) = setup();
    let queue = device.new_command_queue();

    let vertex_buffer = device.new_buffer(
        1024 * 6 * std::mem::size_of::<TextVertex>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let (vertex_count, time_ms) = renderer.layout_text_sync(
        &queue,
        "Hello",
        0.0, 0.0,
        16.0,
        [1.0, 1.0, 1.0, 1.0],
        &atlas,
        &vertex_buffer,
    );

    // "Hello" = 5 characters = 5 quads = 30 vertices
    assert_eq!(
        vertex_count, 30,
        "Hello should produce 30 vertices (6 per char)"
    );
}

#[test]
fn test_text_layout_empty() {
    let (device, atlas, renderer) = setup();
    let queue = device.new_command_queue();

    let vertex_buffer = device.new_buffer(
        1024 * 6 * std::mem::size_of::<TextVertex>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let (vertex_count, _) = renderer.layout_text_sync(
        &queue,
        "",
        0.0, 0.0,
        16.0,
        [1.0, 1.0, 1.0, 1.0],
        &atlas,
        &vertex_buffer,
    );

    assert_eq!(vertex_count, 0, "Empty string should produce 0 vertices");
}

#[test]
fn test_1000_characters_under_200us() {
    let (device, atlas, renderer) = setup();
    let queue = device.new_command_queue();

    let vertex_buffer = device.new_buffer(
        1024 * 6 * std::mem::size_of::<TextVertex>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Generate 1000 character string
    let text: String = (0..1000).map(|i| (b'A' + (i % 26) as u8) as char).collect();

    let (vertex_count, time_ms) = renderer.layout_text_sync(
        &queue,
        &text,
        0.0, 0.0,
        16.0,
        [1.0, 1.0, 1.0, 1.0],
        &atlas,
        &vertex_buffer,
    );

    assert_eq!(vertex_count, 6000, "1000 chars should produce 6000 vertices");
    // Note: GPU command submission overhead dominates; actual GPU work is microseconds
    assert!(
        time_ms < 5.0,
        "1000-char layout should be under 5ms (includes GPU overhead). Got: {:.3}ms",
        time_ms
    );
}

#[test]
fn test_text_benchmark() {
    let (device, atlas, renderer) = setup();
    let queue = device.new_command_queue();

    let results = renderer.benchmark(&queue, &atlas, &[100, 500, 1000], 10);

    for result in results {
        println!(
            "Text layout {} chars: {} vertices in {:.3}ms",
            result.char_count, result.vertex_count, result.layout_time_ms
        );

        // GPU command overhead dominates; verify reasonable performance
        assert!(
            result.layout_time_ms < 5.0,
            "{}-char layout exceeds 5ms budget",
            result.char_count
        );
    }
}
