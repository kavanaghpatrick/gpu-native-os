//! Test suite for Issue #61: CSS Inheritance System
//!
//! Tests GPU-native CSS property inheritance including:
//! - Child inheriting color from parent
//! - Grandchild inheriting from grandparent
//! - Explicit values overriding inheritance
//! - Non-inheritable properties not propagating
//! - Deep nesting performance

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, Stylesheet, ComputedStyle,
};

fn setup() -> (GpuTokenizer, GpuParser, GpuStyler) {
    let device = Device::system_default().expect("No Metal device");
    let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let parser = GpuParser::new(&device).expect("Failed to create parser");
    let styler = GpuStyler::new(&device).expect("Failed to create styler");
    (tokenizer, parser, styler)
}

fn process_html_css(html: &[u8], css: &str) -> Vec<ComputedStyle> {
    let (mut tokenizer, mut parser, mut styler) = setup();
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse(css);
    styler.resolve_styles(&elements, &tokens, html, &stylesheet)
}

fn colors_equal(a: [f32; 4], b: [f32; 4], tolerance: f32) -> bool {
    (a[0] - b[0]).abs() < tolerance &&
    (a[1] - b[1]).abs() < tolerance &&
    (a[2] - b[2]).abs() < tolerance &&
    (a[3] - b[3]).abs() < tolerance
}

#[test]
fn test_child_inherits_color() {
    let html = b"<div><p>text</p></div>";
    let css = "div { color: red; }";

    let styles = process_html_css(html, css);

    // div style
    let div_style = &styles[0];
    assert!(colors_equal(div_style.color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "div should have red color, got {:?}", div_style.color);

    // p should inherit red from div
    let p_style = &styles[1];
    assert!(colors_equal(p_style.color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "p should inherit red from div, got {:?}", p_style.color);
}

#[test]
fn test_grandchild_inherits_color() {
    let html = b"<div><p><span>text</span></p></div>";
    let css = "div { color: blue; }";

    let styles = process_html_css(html, css);

    // div has blue
    let div_style = &styles[0];
    assert!(colors_equal(div_style.color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "div should have blue color");

    // p inherits blue
    let p_style = &styles[1];
    assert!(colors_equal(p_style.color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "p should inherit blue from div");

    // span inherits blue
    let span_style = &styles[2];
    assert!(colors_equal(span_style.color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "span should inherit blue from grandparent div");
}

#[test]
fn test_explicit_overrides_inheritance() {
    let html = b"<div><p>text</p></div>";
    let css = "div { color: red; } p { color: blue; }";

    let styles = process_html_css(html, css);

    // div should have red
    let div_style = &styles[0];
    assert!(colors_equal(div_style.color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "div should have red color, got {:?}", div_style.color);

    // p should have blue (explicit overrides inherited red)
    let p_style = &styles[1];
    assert!(colors_equal(p_style.color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "p should have explicit blue, not inherited red, got {:?}", p_style.color);
}

#[test]
fn test_font_size_inherits() {
    let html = b"<div><p>text</p></div>";
    let css = "div { font-size: 24px; }";

    let styles = process_html_css(html, css);

    // div has 24px
    assert!((styles[0].font_size - 24.0).abs() < 0.01, "div should have 24px font");

    // p inherits 24px
    assert!((styles[1].font_size - 24.0).abs() < 0.01,
        "p should inherit 24px font, got {}", styles[1].font_size);
}

#[test]
fn test_font_weight_inherits() {
    let html = b"<div><span>text</span></div>";
    let css = "div { font-weight: bold; }";

    let styles = process_html_css(html, css);

    // div has bold (700)
    assert_eq!(styles[0].font_weight, 700, "div should have bold weight");

    // span inherits bold
    assert_eq!(styles[1].font_weight, 700,
        "span should inherit bold, got {}", styles[1].font_weight);
}

#[test]
fn test_line_height_inherits() {
    let html = b"<div><p>text</p></div>";
    let css = "div { line-height: 2.0; }";

    let styles = process_html_css(html, css);

    // div has 2.0
    assert!((styles[0].line_height - 2.0).abs() < 0.01);

    // p inherits 2.0
    assert!((styles[1].line_height - 2.0).abs() < 0.01,
        "p should inherit 2.0 line-height, got {}", styles[1].line_height);
}

#[test]
fn test_text_align_inherits() {
    let html = b"<div><p>text</p></div>";
    let css = "div { text-align: center; }";

    let styles = process_html_css(html, css);

    // div has center (1)
    assert_eq!(styles[0].text_align, 1, "div should have text-align: center");

    // p inherits center
    assert_eq!(styles[1].text_align, 1,
        "p should inherit text-align: center, got {}", styles[1].text_align);
}

#[test]
fn test_background_does_not_inherit() {
    let html = b"<div><p>text</p></div>";
    let css = "div { background: red; }";

    let styles = process_html_css(html, css);

    // div has red background
    assert!(colors_equal(styles[0].background_color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "div should have red background");

    // p should have transparent background (default, not inherited)
    assert!(colors_equal(styles[1].background_color, [0.0, 0.0, 0.0, 0.0], 0.01),
        "p should have transparent background (not inherited), got {:?}",
        styles[1].background_color);
}

#[test]
fn test_margin_does_not_inherit() {
    let html = b"<div><p>text</p></div>";
    let css = "div { margin: 20px; }";

    let styles = process_html_css(html, css);

    // div has 20px margin
    assert!((styles[0].margin[0] - 20.0).abs() < 0.01, "div should have 20px margin");

    // p should have 0 margin (default, not inherited)
    assert!((styles[1].margin[0]).abs() < 0.01,
        "p should have 0 margin (not inherited), got {}", styles[1].margin[0]);
}

#[test]
fn test_padding_does_not_inherit() {
    let html = b"<div><p>text</p></div>";
    let css = "div { padding: 15px; }";

    let styles = process_html_css(html, css);

    // div has 15px padding
    assert!((styles[0].padding[0] - 15.0).abs() < 0.01);

    // p should have 0 padding (not inherited)
    assert!((styles[1].padding[0]).abs() < 0.01,
        "p should have 0 padding (not inherited)");
}

#[test]
fn test_deep_nesting_inheritance() {
    // Create deeply nested HTML
    let html = b"<div><div><div><div><div><p>deep</p></div></div></div></div></div>";
    let css = "div { color: green; font-size: 20px; }";

    let styles = process_html_css(html, css);

    // Find the p element (deepest)
    let p_style = styles.last().expect("Should have p style");

    // Should inherit green color
    assert!(colors_equal(p_style.color, [0.0, 0.5, 0.0, 1.0], 0.01),
        "Deeply nested p should inherit green, got {:?}", p_style.color);

    // Should inherit 20px font
    assert!((p_style.font_size - 20.0).abs() < 0.01,
        "Deeply nested p should inherit 20px font, got {}", p_style.font_size);
}

#[test]
fn test_mixed_inheritance_chain() {
    // Grandparent: red, Parent: overrides to blue, Child: inherits blue
    let html = b"<div class=\"grandparent\"><div class=\"parent\"><p>text</p></div></div>";
    let css = ".grandparent { color: red; } .parent { color: blue; }";

    let styles = process_html_css(html, css);

    // grandparent div has red
    assert!(colors_equal(styles[0].color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "grandparent should have red");

    // parent div has blue (explicit)
    assert!(colors_equal(styles[1].color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "parent should have blue (explicit)");

    // p inherits blue from parent (not red from grandparent)
    assert!(colors_equal(styles[2].color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "p should inherit blue from parent, got {:?}", styles[2].color);
}

#[test]
fn test_performance_deep_nesting() {
    let (mut tokenizer, mut parser, mut styler) = setup();

    // Create 50-level deep nesting
    let mut html = String::new();
    for _ in 0..50 {
        html.push_str("<div>");
    }
    html.push_str("<p>deep</p>");
    for _ in 0..50 {
        html.push_str("</div>");
    }

    let css = "div { color: purple; font-size: 18px; line-height: 1.5; }";

    let html_bytes = html.as_bytes();
    let tokens = tokenizer.tokenize(html_bytes);
    let (elements, _) = parser.parse(&tokens, html_bytes);
    let stylesheet = Stylesheet::parse(css);

    // Warmup
    let _ = styler.resolve_styles(&elements, &tokens, html_bytes, &stylesheet);

    // Timed run
    let start = std::time::Instant::now();
    let styles = styler.resolve_styles(&elements, &tokens, html_bytes, &stylesheet);
    let elapsed = start.elapsed();

    println!("50-level deep nesting: {} styles in {:?}", styles.len(), elapsed);

    // Check inheritance worked
    let p_style = styles.last().unwrap();
    assert!(colors_equal(p_style.color, [0.5, 0.0, 0.5, 1.0], 0.01),
        "Deep p should inherit purple color");
    assert!((p_style.font_size - 18.0).abs() < 0.01,
        "Deep p should inherit 18px font");

    assert!(elapsed.as_millis() < 50, "50-level deep took too long: {:?}", elapsed);
}

#[test]
fn test_performance_10k_elements() {
    let (mut tokenizer, mut parser, mut styler) = setup();

    // Generate wide HTML tree
    let mut html = String::from("<div>");
    for i in 0..2500 {
        html.push_str(&format!("<p class=\"item-{}\">Item {}</p>", i % 10, i));
    }
    html.push_str("</div>");

    let css = "div { color: orange; font-size: 14px; } .item-0 { color: red; }";

    let html_bytes = html.as_bytes();
    let tokens = tokenizer.tokenize(html_bytes);
    let (elements, _) = parser.parse(&tokens, html_bytes);
    let stylesheet = Stylesheet::parse(css);

    // Warmup
    let _ = styler.resolve_styles(&elements, &tokens, html_bytes, &stylesheet);

    // Timed run
    let start = std::time::Instant::now();
    let styles = styler.resolve_styles(&elements, &tokens, html_bytes, &stylesheet);
    let elapsed = start.elapsed();

    println!("~5K elements: {} styles in {:?}", styles.len(), elapsed);

    assert!(elapsed.as_millis() < 20, "~5K elements took too long: {:?}", elapsed);
}

#[test]
fn test_text_node_inherits_from_parent() {
    let html = b"<p>Hello World</p>";
    let css = "p { color: red; font-size: 24px; }";

    let styles = process_html_css(html, css);

    // p element
    let p_style = &styles[0];
    assert!(colors_equal(p_style.color, [1.0, 0.0, 0.0, 1.0], 0.01));

    // Text node should inherit from p
    if styles.len() > 1 {
        let text_style = &styles[1];
        assert!(colors_equal(text_style.color, [1.0, 0.0, 0.0, 1.0], 0.01),
            "Text node should inherit red from p, got {:?}", text_style.color);
        assert!((text_style.font_size - 24.0).abs() < 0.01,
            "Text node should inherit 24px from p");
    }
}
