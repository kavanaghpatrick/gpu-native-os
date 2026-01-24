//! Test suite for Issue #69: GPU Gradient Rendering
//!
//! Tests GPU-accelerated CSS gradients:
//! - linear-gradient() parsing
//! - radial-gradient() parsing
//! - Color stops with positions
//! - Direction keywords (to right, to bottom, etc.)
//! - Angle-based gradients

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
    GRADIENT_NONE, GRADIENT_LINEAR, GRADIENT_RADIAL,
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

// ======= GRADIENT TYPE DETECTION =======

#[test]
fn test_no_gradient_default() {
    let html = b"<div>no gradient</div>";
    let css = "div { width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_type, GRADIENT_NONE, "Default should have no gradient");
}

#[test]
fn test_linear_gradient_basic() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: linear-gradient(red, blue); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_type, GRADIENT_LINEAR, "Should detect linear gradient");
    assert!(style.gradient_stop_count >= 2, "Should have at least 2 color stops");
}

#[test]
fn test_radial_gradient_basic() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: radial-gradient(red, blue); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_type, GRADIENT_RADIAL, "Should detect radial gradient");
}

// ======= LINEAR GRADIENT DIRECTIONS =======

#[test]
fn test_linear_gradient_to_right() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: linear-gradient(to right, red, blue); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_type, GRADIENT_LINEAR);
    assert!((style.gradient_angle - 90.0).abs() < 1.0, "to right should be 90 degrees, got {}", style.gradient_angle);
}

#[test]
fn test_linear_gradient_to_bottom() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: linear-gradient(to bottom, red, blue); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_type, GRADIENT_LINEAR);
    assert!((style.gradient_angle - 180.0).abs() < 1.0, "to bottom should be 180 degrees, got {}", style.gradient_angle);
}

#[test]
fn test_linear_gradient_angle() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: linear-gradient(45deg, red, blue); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_type, GRADIENT_LINEAR);
    assert!((style.gradient_angle - 45.0).abs() < 1.0, "Angle should be 45 degrees, got {}", style.gradient_angle);
}

// ======= COLOR STOPS =======

#[test]
fn test_gradient_color_stops() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: linear-gradient(red, green, blue); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_stop_count, 3, "Should have 3 color stops");
}

#[test]
fn test_gradient_color_stop_positions() {
    let html = b"<div class=\"grad\">gradient</div>";
    let css = ".grad { background: linear-gradient(red 0%, blue 100%); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.gradient_stop_count, 2);
    assert!((style.gradient_stop_positions[0] - 0.0).abs() < 0.01, "First stop at 0%");
    assert!((style.gradient_stop_positions[1] - 1.0).abs() < 0.01, "Second stop at 100%");
}

// ======= GRADIENT CONSTANTS =======

#[test]
fn test_gradient_constants() {
    assert_eq!(GRADIENT_NONE, 0);
    assert_eq!(GRADIENT_LINEAR, 1);
    assert_eq!(GRADIENT_RADIAL, 2);
}

// ======= COMPUTED STYLE SIZE =======

#[test]
fn test_computed_style_size() {
    let size = std::mem::size_of::<ComputedStyle>();
    // Allow for expanded struct with gradient properties
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}
