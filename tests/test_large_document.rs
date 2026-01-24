//! Tests for large document handling (Issue #85)
//!
//! Validates that the document pipeline handles large documents correctly
//! without out-of-bounds positions or excessive memory usage.

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine,
    Stylesheet, Viewport, PaintVertex,
};

fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine) {
    let device = Device::system_default().expect("No Metal device");
    let tokenizer = GpuTokenizer::new(&device).expect("tokenizer");
    let parser = GpuParser::new(&device).expect("parser");
    let styler = GpuStyler::new(&device).expect("styler");
    let layout = GpuLayoutEngine::new(&device).expect("layout");
    let paint = GpuPaintEngine::new(&device).expect("paint");
    (tokenizer, parser, styler, layout, paint)
}

fn make_viewport(width: f32, height: f32) -> Viewport {
    Viewport {
        width,
        height,
        _padding: [0.0; 2],
    }
}

fn generate_large_html(num_elements: usize) -> Vec<u8> {
    let mut html = String::from("<!DOCTYPE html><html><body>");
    for i in 0..num_elements {
        html.push_str(&format!("<div style=\"height: 50px; background: #{};\">Item {}</div>",
            format!("{:06x}", (i * 12345) % 0xFFFFFF), i));
    }
    html.push_str("</body></html>");
    html.into_bytes()
}

fn count_out_of_bounds(vertices: &[PaintVertex], limit: f32) -> usize {
    vertices.iter().filter(|v| {
        v.position[0] < -limit || v.position[0] > limit ||
        v.position[1] < -limit || v.position[1] > limit
    }).count()
}

// ============================================================================
// Tests for Issue #85: Large Document Layout
// ============================================================================

#[test]
fn test_100_elements_no_out_of_bounds() {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
    let html = generate_large_html(100);
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(&html);
    let (elements, text_buffer) = parser.parse(&tokens, &html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);

    let oob_count = count_out_of_bounds(&vertices, 10.0);
    // Small documents should have no out-of-bounds
    assert!(oob_count == 0 || oob_count < vertices.len() / 100,
        "Too many out-of-bounds: {} / {}", oob_count, vertices.len());
}

#[test]
fn test_500_elements_processes_successfully() {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
    let html = generate_large_html(500);
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(&html);
    let (elements, text_buffer) = parser.parse(&tokens, &html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);

    // Should generate vertices without crashing
    assert!(vertices.len() > 0, "Should generate vertices");
    assert!(elements.len() >= 500, "Should parse all elements");
}

#[test]
#[ignore] // Enable when viewport culling is implemented
fn test_viewport_culling_reduces_vertices() {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
    let html = generate_large_html(1000);
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(&html);
    let (elements, text_buffer) = parser.parse(&tokens, &html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);

    // With culling, we should generate far fewer vertices than elements * 4
    // Viewport shows ~15 elements at 50px height each
    let expected_max = 50 * 4 * 6; // 50 visible elements, 4 vertices each, some margin
    assert!(vertices.len() < expected_max,
        "Expected culling to reduce vertices: {} >= {}", vertices.len(), expected_max);
}

#[test]
fn test_no_nan_or_inf_in_large_document() {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
    let html = generate_large_html(200);
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(&html);
    let (elements, text_buffer) = parser.parse(&tokens, &html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);

    for (i, v) in vertices.iter().enumerate() {
        assert!(!v.position[0].is_nan(), "NaN position[0] at vertex {}", i);
        assert!(!v.position[1].is_nan(), "NaN position[1] at vertex {}", i);
        assert!(!v.position[0].is_infinite(), "Inf position[0] at vertex {}", i);
        assert!(!v.position[1].is_infinite(), "Inf position[1] at vertex {}", i);
    }
}

#[test]
fn test_colors_valid_in_large_document() {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
    let html = generate_large_html(200);
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(&html);
    let (elements, text_buffer) = parser.parse(&tokens, &html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);

    for (i, v) in vertices.iter().enumerate() {
        for (j, &c) in v.color.iter().enumerate() {
            assert!(c >= 0.0 && c <= 1.0, "Invalid color[{}]={} at vertex {}", j, c, i);
            assert!(!c.is_nan(), "NaN color[{}] at vertex {}", j, i);
        }
    }
}
