//! Test suite for Issue #67: GPU Overflow & Clipping System
//!
//! Tests GPU-accelerated CSS overflow handling:
//! - overflow: visible (default - content can overflow container)
//! - overflow: hidden (clip content at container bounds)
//! - overflow: scroll (always show scrollbars)
//! - overflow: auto (show scrollbars only when needed)
//! - overflow-x, overflow-y (independent axis control)
//! - Scissor rect generation for GPU clipping

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
    OVERFLOW_VISIBLE, OVERFLOW_HIDDEN, OVERFLOW_SCROLL, OVERFLOW_AUTO,
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

// Helper to find element by type
fn find_elements_of_type(elements: &[Element], elem_type: u32) -> Vec<usize> {
    elements.iter()
        .enumerate()
        .filter(|(_, e)| e.element_type == elem_type)
        .map(|(i, _)| i)
        .collect()
}

const ELEM_DIV: u32 = 1;

// ======= OVERFLOW PROPERTY PARSING =======

#[test]
fn test_overflow_visible_default() {
    let html = b"<div>test</div>";
    let css = "div { width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(!divs.is_empty(), "Should find div");

    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_VISIBLE, "Default overflow-x should be visible");
    assert_eq!(style.overflow_y, OVERFLOW_VISIBLE, "Default overflow-y should be visible");
}

#[test]
fn test_overflow_hidden() {
    let html = b"<div class=\"clip\">hidden</div>";
    let css = ".clip { overflow: hidden; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_HIDDEN, "Should parse overflow: hidden for x");
    assert_eq!(style.overflow_y, OVERFLOW_HIDDEN, "Should parse overflow: hidden for y");
}

#[test]
fn test_overflow_scroll() {
    let html = b"<div class=\"scroll\">scrollable</div>";
    let css = ".scroll { overflow: scroll; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_SCROLL, "Should parse overflow: scroll for x");
    assert_eq!(style.overflow_y, OVERFLOW_SCROLL, "Should parse overflow: scroll for y");
}

#[test]
fn test_overflow_auto() {
    let html = b"<div class=\"auto\">auto overflow</div>";
    let css = ".auto { overflow: auto; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_AUTO, "Should parse overflow: auto for x");
    assert_eq!(style.overflow_y, OVERFLOW_AUTO, "Should parse overflow: auto for y");
}

// ======= INDEPENDENT AXIS CONTROL =======

#[test]
fn test_overflow_x_only() {
    let html = b"<div class=\"ox\">overflow-x</div>";
    let css = ".ox { overflow-x: hidden; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_HIDDEN, "overflow-x should be hidden");
    assert_eq!(style.overflow_y, OVERFLOW_VISIBLE, "overflow-y should remain visible");
}

#[test]
fn test_overflow_y_only() {
    let html = b"<div class=\"oy\">overflow-y</div>";
    let css = ".oy { overflow-y: scroll; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_VISIBLE, "overflow-x should remain visible");
    assert_eq!(style.overflow_y, OVERFLOW_SCROLL, "overflow-y should be scroll");
}

#[test]
fn test_overflow_xy_different() {
    let html = b"<div class=\"mixed\">mixed overflow</div>";
    let css = ".mixed { overflow-x: scroll; overflow-y: hidden; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_SCROLL, "overflow-x should be scroll");
    assert_eq!(style.overflow_y, OVERFLOW_HIDDEN, "overflow-y should be hidden");
}

// ======= SHORTHAND OVERRIDE =======

#[test]
fn test_overflow_shorthand_overrides_individual() {
    let html = b"<div class=\"over\">override</div>";
    let css = ".over { overflow-x: scroll; overflow: hidden; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    // shorthand 'overflow: hidden' should override both
    assert_eq!(style.overflow_x, OVERFLOW_HIDDEN, "overflow shorthand should override x");
    assert_eq!(style.overflow_y, OVERFLOW_HIDDEN, "overflow shorthand should override y");
}

// ======= COMPUTED STYLE STRUCT SIZE =======

#[test]
fn test_computed_style_size() {
    // ComputedStyle should maintain GPU-friendly alignment
    let size = std::mem::size_of::<ComputedStyle>();
    // Allow for expanded struct with overflow, shadow, and gradient properties
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}

// ======= CLIPPING BEHAVIOR (Style Parsing) =======

#[test]
fn test_nested_overflow_containers() {
    let html = b"<div class=\"outer\"><div class=\"inner\">nested</div></div>";
    let css = ".outer { overflow: hidden; width: 200px; height: 100px; }
               .inner { overflow: scroll; width: 300px; height: 150px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(divs.len() >= 2, "Should find 2 divs");

    // Verify both containers have correct overflow
    let outer_style = &styles[divs[0]];
    assert_eq!(outer_style.overflow_x, OVERFLOW_HIDDEN);
    assert_eq!(outer_style.overflow_y, OVERFLOW_HIDDEN);

    let inner_style = &styles[divs[1]];
    assert_eq!(inner_style.overflow_x, OVERFLOW_SCROLL);
    assert_eq!(inner_style.overflow_y, OVERFLOW_SCROLL);
}

#[test]
fn test_overflow_with_border_radius() {
    // Overflow hidden with border-radius creates rounded clipping
    let html = b"<div class=\"rounded\">rounded clip</div>";
    let css = ".rounded { overflow: hidden; border-radius: 10px; width: 100px; height: 100px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_HIDDEN);
    assert_eq!(style.border_radius, 10.0);
    // Note: Actual rounded clipping is a rendering concern
}

#[test]
fn test_text_overflow_clip() {
    // text-overflow: clip is distinct from overflow: hidden
    let html = b"<div class=\"textclip\">long text</div>";
    let css = ".textclip { overflow: hidden; width: 50px; white-space: nowrap; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.overflow_x, OVERFLOW_HIDDEN);
    // Note: text-overflow: clip is default, ellipsis requires separate property
}

// ======= OVERFLOW CONSTANTS =======

#[test]
fn test_overflow_constants() {
    assert_eq!(OVERFLOW_VISIBLE, 0);
    assert_eq!(OVERFLOW_HIDDEN, 1);
    assert_eq!(OVERFLOW_SCROLL, 2);
    assert_eq!(OVERFLOW_AUTO, 3);
}
