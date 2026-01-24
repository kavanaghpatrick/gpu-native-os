//! Tests for GPU Runtime Font System Integration
//!
//! Verifies:
//! - BitmapFont creation in GpuRuntime
//! - TextRenderer integration
//! - DocumentApp text rendering
//!
//! Run with: cargo test --test test_gpu_runtime_fonts

use metal::Device;
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer, colors};

// ============================================================================
// BitmapFont Tests
// ============================================================================

#[test]
fn test_bitmap_font_creation() {
    let device = Device::system_default().expect("No Metal device");
    let font = BitmapFont::new(&device);

    assert_eq!(font.char_width, 8.0, "Char width should be 8");
    assert_eq!(font.char_height, 8.0, "Char height should be 8");
    assert_eq!(font.chars_per_row, 16, "16 chars per row");
    assert_eq!(font.num_rows, 6, "6 rows for ASCII 32-127");
}

#[test]
fn test_bitmap_font_spacing() {
    let device = Device::system_default().expect("No Metal device");
    let font = BitmapFont::new(&device);

    // Default scale
    assert_eq!(font.char_spacing(1.0), 8.0);
    assert_eq!(font.char_spacing(2.0), 16.0);

    // Line height includes 1.5x spacing
    assert_eq!(font.line_height(1.0), 12.0);
    assert_eq!(font.line_height(2.0), 24.0);
}

// ============================================================================
// TextRenderer Tests
// ============================================================================

#[test]
fn test_text_renderer_creation() {
    let device = Device::system_default().expect("No Metal device");
    let renderer = TextRenderer::new(&device, 1000);

    assert!(renderer.is_ok(), "TextRenderer should compile");
}

#[test]
fn test_text_renderer_add_text() {
    let device = Device::system_default().expect("No Metal device");
    let mut renderer = TextRenderer::new(&device, 1000).unwrap();

    assert_eq!(renderer.char_count(), 0, "Initially empty");

    renderer.add_text("Hello", 10.0, 20.0, colors::WHITE);
    assert_eq!(renderer.char_count(), 5, "5 chars added");

    renderer.add_text(" World", 50.0, 20.0, colors::RED);
    assert_eq!(renderer.char_count(), 11, "11 total chars");

    renderer.clear();
    assert_eq!(renderer.char_count(), 0, "Cleared");
}

#[test]
fn test_text_renderer_max_chars() {
    let device = Device::system_default().expect("No Metal device");
    let mut renderer = TextRenderer::new(&device, 10).unwrap();

    // Add 20 chars (should be limited to 10)
    renderer.add_text("12345678901234567890", 0.0, 0.0, colors::WHITE);

    // Should not exceed max
    assert!(renderer.char_count() <= 10, "Should respect max_chars limit");
}

#[test]
fn test_color_constants() {
    assert_eq!(colors::WHITE, 0xFFFFFFFF);
    assert_eq!(colors::BLACK, 0x000000FF);
    assert_eq!(colors::RED, 0xFF0000FF);
    assert_eq!(colors::GREEN, 0x00FF00FF);
    assert_eq!(colors::BLUE, 0x0000FFFF);
}

// ============================================================================
// GpuRuntime Font Integration Tests
// ============================================================================

#[test]
fn test_gpu_runtime_has_font() {
    let device = Device::system_default().expect("No Metal device");
    let runtime = GpuRuntime::new(device);

    let font = runtime.font();
    assert_eq!(font.char_width, 8.0, "Runtime font should be initialized");
}

#[test]
fn test_gpu_runtime_has_text_renderer() {
    let device = Device::system_default().expect("No Metal device");
    let runtime = GpuRuntime::new(device);

    let text = runtime.text_renderer();
    assert_eq!(text.char_count(), 0, "Text renderer should be initialized and empty");
}

#[test]
fn test_gpu_runtime_text_renderer_mut() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = GpuRuntime::new(device);

    {
        let text = runtime.text_renderer_mut();
        text.add_text("Test", 0.0, 0.0, colors::WHITE);
        assert_eq!(text.char_count(), 4);
    }

    // Clear via mut access
    runtime.text_renderer_mut().clear();
    assert_eq!(runtime.text_renderer().char_count(), 0);
}

// ============================================================================
// DocumentApp Text Rendering Tests
// ============================================================================

#[test]
fn test_document_app_uses_text_rendering() {
    use rust_experiment::gpu_os::document_app::DocumentApp;
    use rust_experiment::gpu_os::app::GpuApp;

    let device = Device::system_default().expect("No Metal device");
    let app = DocumentApp::new(&device, 800.0, 600.0).expect("DocumentApp creation");

    assert!(app.uses_text_rendering(), "DocumentApp should use text rendering");
}

#[test]
fn test_document_app_render_text() {
    use rust_experiment::gpu_os::document_app::DocumentApp;
    use rust_experiment::gpu_os::app::GpuApp;

    let device = Device::system_default().expect("No Metal device");
    let mut app = DocumentApp::new(&device, 800.0, 600.0).expect("DocumentApp creation");
    let mut text = TextRenderer::new(&device, 1000).unwrap();

    // Call render_text to add status text
    app.render_text(&mut text);

    // Should have added some status text
    assert!(text.char_count() > 0, "render_text should add status text");
}

// ============================================================================
// Performance Tests
// ============================================================================

#[test]
fn test_text_add_performance() {
    let device = Device::system_default().expect("No Metal device");
    let mut renderer = TextRenderer::new(&device, 10000).unwrap();

    let start = std::time::Instant::now();

    // Add 1000 lines of text
    for i in 0..1000 {
        renderer.add_text(&format!("Line {}", i), 0.0, i as f32 * 12.0, colors::WHITE);
    }

    let elapsed = start.elapsed();
    println!("Added 1000 lines in {:?}", elapsed);

    assert!(elapsed.as_millis() < 50, "Adding 1000 lines should take <50ms");
}
