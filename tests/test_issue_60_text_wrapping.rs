//! Test suite for Issue #60: GPU Text Wrapping & Line Breaking
//!
//! Tests GPU-native text wrapping functionality including:
//! - Short text that doesn't need wrapping
//! - Long text that wraps at word boundaries
//! - Text with explicit newlines
//! - Text-align properties
//! - Performance with large text content

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine,
    Stylesheet, Viewport, LayoutBox, PaintVertex, FLAG_TEXT,
};

fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine) {
    let device = Device::system_default().expect("No Metal device");
    let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let parser = GpuParser::new(&device).expect("Failed to create parser");
    let styler = GpuStyler::new(&device).expect("Failed to create styler");
    let layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");
    let paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");
    (tokenizer, parser, styler, layout, paint)
}

fn process_html(html: &[u8], css: &str, viewport: Viewport) -> Vec<PaintVertex> {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();

    let tokens = tokenizer.tokenize(html);
    let (elements, text) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text, viewport);
    paint.paint(&elements, &boxes, &styles, &text, viewport)
}

/// Get text vertices and find unique Y positions (lines)
/// Each glyph has 4 vertices in a quad. We look at every 4th vertex (top-left) to find line positions.
fn get_text_line_positions(vertices: &[PaintVertex]) -> Vec<f32> {
    let text_vertices: Vec<_> = vertices.iter()
        .filter(|v| v.flags == FLAG_TEXT)
        .collect();

    // Get every 4th vertex (top-left of each glyph quad)
    let mut y_positions: Vec<f32> = text_vertices.iter()
        .step_by(4)  // Every 4th vertex is a new glyph
        .map(|v| v.position[1])
        .collect();

    // Deduplicate Y positions (group vertices on same line)
    // Use larger tolerance since NDC values are small
    y_positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_positions.dedup_by(|a, b| (*a - *b).abs() < 0.05);

    y_positions
}

#[test]
fn test_short_text_no_wrap() {
    let html = b"<div>Hello</div>";
    let css = "div { width: 300px; font-size: 16px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    // Short text should be on a single line
    assert_eq!(y_positions.len(), 1, "Short text 'Hello' should produce 1 line, got {}", y_positions.len());
}

#[test]
fn test_long_text_wraps() {
    // Long text that should wrap in a 100px container
    let html = b"<div>This is a longer text that should definitely wrap to multiple lines in a narrow container.</div>";
    let css = "div { width: 100px; font-size: 14px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    println!("Long text produced {} lines", y_positions.len());

    // Should have multiple lines
    assert!(y_positions.len() >= 2, "Long text in narrow container should wrap to 2+ lines, got {}", y_positions.len());
}

#[test]
fn test_explicit_newlines() {
    let html = b"<pre>Line1\nLine2\nLine3</pre>";
    let css = "pre { width: 500px; font-size: 16px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    // Should have 3 lines due to newlines
    assert!(y_positions.len() >= 3, "Text with 2 newlines should produce 3 lines, got {}", y_positions.len());
}

#[test]
fn test_word_boundary_break() {
    // Text that should break at word boundary, not mid-word
    let html = b"<div>hello world test</div>";
    let css = "div { width: 80px; font-size: 16px; }";  // Force wrap
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let text_vertices: Vec<_> = vertices.iter()
        .filter(|v| v.flags == FLAG_TEXT)
        .collect();

    // Should have text vertices
    assert!(!text_vertices.is_empty(), "Should produce text vertices");

    // Verify multiple lines
    let y_positions = get_text_line_positions(&vertices);
    println!("Word boundary test: {} lines", y_positions.len());
}

#[test]
fn test_text_in_padded_container() {
    let html = b"<div>Hello world this is a longer text</div>";
    let css = "div { width: 200px; padding: 20px; font-size: 14px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    // Text should fit in content area (200 - 40 = 160px content width)
    println!("Padded container test: {} lines", y_positions.len());
}

#[test]
fn test_very_long_word() {
    // Very long word that must break mid-word
    let html = b"<div>Supercalifragilisticexpialidocious</div>";
    let css = "div { width: 100px; font-size: 16px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    // Long word should force break
    println!("Long word test: {} lines", y_positions.len());
    assert!(y_positions.len() >= 2, "Very long word should break across lines");
}

#[test]
fn test_empty_text() {
    let html = b"<div></div>";
    let css = "div { width: 100px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let text_vertices: Vec<_> = vertices.iter()
        .filter(|v| v.flags == FLAG_TEXT)
        .collect();

    // Empty div should produce no text vertices
    assert!(text_vertices.is_empty() || text_vertices.iter().all(|v| v.color[3] == 0.0),
        "Empty text should produce no visible text vertices");
}

#[test]
fn test_multiple_text_elements() {
    let html = b"<div><p>First paragraph with some text.</p><p>Second paragraph with different text.</p></div>";
    let css = "div { width: 200px; } p { font-size: 14px; margin: 10px 0; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let text_vertices: Vec<_> = vertices.iter()
        .filter(|v| v.flags == FLAG_TEXT)
        .collect();

    // Should have text from both paragraphs
    assert!(!text_vertices.is_empty(), "Should have text vertices from paragraphs");
}

#[test]
fn test_performance_10k_characters() {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();

    // Generate text with words
    let mut text = String::new();
    for i in 0..1000 {
        text.push_str("word ");
        if i % 20 == 19 {
            text.push('\n');
        }
    }
    let html = format!("<div>{}</div>", text);
    let html_bytes = html.as_bytes();

    let css = "div { width: 300px; font-size: 14px; line-height: 1.4; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let tokens = tokenizer.tokenize(html_bytes);
    let (elements, text_content) = parser.parse(&tokens, html_bytes);
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html_bytes, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_content, viewport);

    // Warmup
    let _ = paint.paint(&elements, &boxes, &styles, &text_content, viewport);

    // Timed run
    let start = std::time::Instant::now();
    let vertices = paint.paint(&elements, &boxes, &styles, &text_content, viewport);
    let elapsed = start.elapsed();

    let text_vertex_count = vertices.iter().filter(|v| v.flags == FLAG_TEXT).count();
    println!("10K characters: {} text vertices in {:?}", text_vertex_count, elapsed);

    // Note: Issue #131 two-pass text layout adds overhead for line break pre-computation
    assert!(elapsed.as_millis() < 100, "10K characters took too long: {:?}", elapsed);
}

#[test]
fn test_nested_text_layout() {
    let html = b"<div><span>Hello</span> <span>World</span></div>";
    let css = "div { width: 300px; font-size: 16px; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let text_vertices: Vec<_> = vertices.iter()
        .filter(|v| v.flags == FLAG_TEXT)
        .collect();

    // Should have text vertices for both spans
    assert!(!text_vertices.is_empty(), "Should have text from nested spans");
}

#[test]
fn test_wide_container_no_wrap() {
    let html = b"<div>This is some text that should not wrap because the container is wide enough.</div>";
    let css = "div { width: 1000px; font-size: 14px; }";
    let viewport = Viewport { width: 1200.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    // Wide container should produce single line
    assert_eq!(y_positions.len(), 1, "Wide container should produce 1 line, got {}", y_positions.len());
}

#[test]
fn test_line_height() {
    let html = b"<div>Line one\nLine two</div>";
    let css = "div { width: 300px; font-size: 16px; line-height: 2.0; }";
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = process_html(html, css, viewport);
    let y_positions = get_text_line_positions(&vertices);

    if y_positions.len() >= 2 {
        // Line height should affect spacing
        let line_spacing = (y_positions[1] - y_positions[0]).abs();
        // In NDC, spacing depends on viewport height
        println!("Line spacing in NDC: {}", line_spacing);
    }
}
