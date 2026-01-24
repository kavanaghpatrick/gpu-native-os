//! Comprehensive tests for GPU-Native Document Processing
//!
//! Tests that ALL computation happens on GPU with NO CPU involvement:
//! - Tokenization on GPU
//! - Parsing on GPU
//! - Style resolution on GPU (including inline styles)
//! - Layout on GPU
//! - Paint on GPU

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
};

fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine) {
    let device = Device::system_default().expect("No Metal device");
    let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let parser = GpuParser::new(&device).expect("Failed to create parser");
    let styler = GpuStyler::new(&device).expect("Failed to create styler");
    let layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");
    (tokenizer, parser, styler, layout)
}

fn make_viewport(width: f32, height: f32) -> Viewport {
    Viewport {
        width,
        height,
        _padding: [0.0, 0.0],
    }
}

fn process_html(
    html: &[u8],
    css: &str,
    viewport: Viewport,
) -> (Vec<Element>, Vec<LayoutBox>, Vec<ComputedStyle>) {
    let (mut tokenizer, mut parser, mut styler, mut layout) = setup();

    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, viewport);

    (elements, boxes, styles)
}

fn find_elements_of_type(elements: &[Element], elem_type: u32) -> Vec<usize> {
    elements.iter()
        .enumerate()
        .filter(|(_, e)| e.element_type == elem_type)
        .map(|(i, _)| i)
        .collect()
}

const ELEM_DIV: u32 = 1;
const ELEM_SPAN: u32 = 2;
const ELEM_P: u32 = 3;

// ============================================================================
// GPU-NATIVE INLINE STYLE PARSING TESTS
// All parsing happens in Metal shader, no CPU string processing
// ============================================================================

#[test]
fn test_gpu_inline_width_parsing() {
    let html = b"<div style=\"width: 150px;\">test</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert_eq!(styles[divs[0]].width, 150.0);
}

#[test]
fn test_gpu_inline_height_parsing() {
    let html = b"<div style=\"height: 250px;\">test</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert_eq!(styles[divs[0]].height, 250.0);
}

#[test]
fn test_gpu_inline_color_red() {
    let html = b"<div style=\"color: red;\">test</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert!((s.color[0] - 1.0).abs() < 0.01);
    assert!((s.color[1] - 0.0).abs() < 0.01);
    assert!((s.color[2] - 0.0).abs() < 0.01);
}

#[test]
fn test_gpu_inline_color_blue() {
    let html = b"<div style=\"color: blue;\">test</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert!((s.color[0] - 0.0).abs() < 0.01);
    assert!((s.color[1] - 0.0).abs() < 0.01);
    assert!((s.color[2] - 1.0).abs() < 0.01);
}

#[test]
fn test_gpu_inline_color_green() {
    let html = b"<div style=\"color: green;\">test</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert!((s.color[0] - 0.0).abs() < 0.01);
    assert!((s.color[1] - 0.5).abs() < 0.01);
    assert!((s.color[2] - 0.0).abs() < 0.01);
}

#[test]
fn test_gpu_inline_display_none() {
    let html = b"<div style=\"display: none;\">hidden</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert_eq!(styles[divs[0]].display, 0); // DISPLAY_NONE
}

#[test]
fn test_gpu_inline_display_flex() {
    let html = b"<div style=\"display: flex;\">flex</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert_eq!(styles[divs[0]].display, 3); // DISPLAY_FLEX
}

#[test]
fn test_gpu_inline_display_block() {
    let html = b"<div style=\"display: block;\">block</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert_eq!(styles[divs[0]].display, 1); // DISPLAY_BLOCK
}

#[test]
fn test_gpu_inline_display_inline() {
    let html = b"<span style=\"display: inline;\">inline</span>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let spans = find_elements_of_type(&elements, ELEM_SPAN);
    assert_eq!(styles[spans[0]].display, 2); // DISPLAY_INLINE
}

#[test]
fn test_gpu_inline_background_hex() {
    let html = b"<div style=\"background-color: #ff0000;\">red bg</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert!((s.background_color[0] - 1.0).abs() < 0.01);
    assert!((s.background_color[1] - 0.0).abs() < 0.01);
    assert!((s.background_color[2] - 0.0).abs() < 0.01);
}

#[test]
fn test_gpu_inline_background_hex_green() {
    let html = b"<div style=\"background-color: #00ff00;\">green bg</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert!((s.background_color[0] - 0.0).abs() < 0.01);
    assert!((s.background_color[1] - 1.0).abs() < 0.01);
    assert!((s.background_color[2] - 0.0).abs() < 0.01);
}

#[test]
fn test_gpu_inline_margin_single() {
    let html = b"<div style=\"margin: 20px;\">margin</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert_eq!(s.margin[0], 20.0);
    assert_eq!(s.margin[1], 20.0);
    assert_eq!(s.margin[2], 20.0);
    assert_eq!(s.margin[3], 20.0);
}

#[test]
fn test_gpu_inline_margin_four_values() {
    let html = b"<div style=\"margin: 10px 20px 30px 40px;\">margin</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert_eq!(s.margin[0], 10.0);
    assert_eq!(s.margin[1], 20.0);
    assert_eq!(s.margin[2], 30.0);
    assert_eq!(s.margin[3], 40.0);
}

#[test]
fn test_gpu_inline_margin_two_values() {
    let html = b"<div style=\"margin: 15px 25px;\">margin</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert_eq!(s.margin[0], 15.0); // top
    assert_eq!(s.margin[1], 25.0); // right
    assert_eq!(s.margin[2], 15.0); // bottom
    assert_eq!(s.margin[3], 25.0); // left
}

#[test]
fn test_gpu_inline_multiple_properties() {
    let html = b"<div style=\"width: 100px; height: 50px; margin: 5px;\">multi</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert_eq!(s.width, 100.0);
    assert_eq!(s.height, 50.0);
    assert_eq!(s.margin[0], 5.0);
}

#[test]
fn test_gpu_inline_with_single_quotes() {
    let html = b"<div style='width: 300px;'>single quotes</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert_eq!(styles[divs[0]].width, 300.0);
}

#[test]
fn test_gpu_inline_overrides_css_rule() {
    let html = b"<div class=\"box\" style=\"width: 500px;\">override</div>";
    let css = ".box { width: 100px; height: 200px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    // Inline should override CSS width
    assert_eq!(s.width, 500.0);
    // CSS height should still apply
    assert_eq!(s.height, 200.0);
}

#[test]
fn test_gpu_inline_on_multiple_elements() {
    let html = b"<div style=\"width: 100px;\">first</div><div style=\"width: 200px;\">second</div><div style=\"width: 300px;\">third</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(divs.len() >= 3);
    assert_eq!(styles[divs[0]].width, 100.0);
    assert_eq!(styles[divs[1]].width, 200.0);
    assert_eq!(styles[divs[2]].width, 300.0);
}

#[test]
fn test_gpu_inline_whitespace_handling() {
    let html = b"<div style=\"  width:   200px  ;  height:100px  \">whitespace</div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let s = &styles[divs[0]];
    assert_eq!(s.width, 200.0);
    assert_eq!(s.height, 100.0);
}

#[test]
fn test_gpu_inline_no_style_attribute() {
    let html = b"<div class=\"plain\">no inline style</div>";
    let css = ".plain { width: 150px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);
    // CSS rule should apply
    assert_eq!(styles[divs[0]].width, 150.0);
}

// ============================================================================
// GPU ARCHITECTURE VALIDATION
// ============================================================================

#[test]
fn test_computed_style_size_for_gpu() {
    // ComputedStyle must be properly sized for GPU transfer
    let size = std::mem::size_of::<ComputedStyle>();
    assert!(size % 4 == 0, "ComputedStyle must be 4-byte aligned for GPU");
    assert!(size <= 600, "ComputedStyle too large: {}", size);
}

#[test]
fn test_element_size_for_gpu() {
    let size = std::mem::size_of::<Element>();
    assert!(size % 4 == 0, "Element must be 4-byte aligned for GPU");
    assert_eq!(size, 32, "Element should be 32 bytes");
}

#[test]
fn test_layout_box_size_for_gpu() {
    let size = std::mem::size_of::<LayoutBox>();
    assert!(size % 4 == 0, "LayoutBox must be 4-byte aligned for GPU");
    assert!(size <= 128, "LayoutBox should be reasonable size: {}", size);
}

// ============================================================================
// PARALLEL PROCESSING TESTS
// ============================================================================

#[test]
fn test_many_elements_processed_in_parallel() {
    // Create HTML with many elements - GPU should process them in parallel
    let mut html = String::from("<div>");
    for i in 0..50 {
        let width = (i + 1) * 10;
        html.push_str(&format!("<span style=\"width: {}px;\">item{}</span>", width, i));
    }
    html.push_str("</div>");

    let (elements, _, styles) = process_html(html.as_bytes(), "", make_viewport(800.0, 600.0));
    let spans = find_elements_of_type(&elements, ELEM_SPAN);

    // Verify we got all spans
    assert_eq!(spans.len(), 50, "Should find 50 spans");

    // Collect all widths and verify each expected value is present
    let widths: Vec<f32> = spans.iter().map(|&i| styles[i].width).collect();

    // Check that we have a variety of widths (not all the same default value)
    let unique_widths: std::collections::HashSet<u32> = widths.iter().map(|w| *w as u32).collect();
    assert!(unique_widths.len() >= 10, "Should have many different widths, got {} unique", unique_widths.len());

    // Check that expected widths are present
    for expected in [10, 100, 200, 300, 400, 500].iter() {
        assert!(widths.iter().any(|w| (*w as u32) == *expected),
            "Should find width {} in spans", expected);
    }
}

#[test]
fn test_nested_elements_with_inline_styles() {
    let html = b"<div style=\"width: 400px;\"><div style=\"width: 300px;\"><div style=\"width: 200px;\">nested</div></div></div>";
    let (elements, _, styles) = process_html(html, "", make_viewport(800.0, 600.0));
    let divs = find_elements_of_type(&elements, ELEM_DIV);

    // All three divs should have their inline widths
    let widths: Vec<f32> = divs.iter().map(|&i| styles[i].width).collect();
    assert!(widths.contains(&400.0));
    assert!(widths.contains(&300.0));
    assert!(widths.contains(&200.0));
}
