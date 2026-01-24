//! Test suite for Issue #72: GPU Inline Style Parsing
//!
//! Tests GPU-accelerated inline style attribute parsing:
//! - style="..." attribute on HTML elements
//! - Inline styles override stylesheet rules
//! - Multiple properties in inline styles

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
    DISPLAY_NONE, DISPLAY_FLEX,
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
    let (elements, text_buffer) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);

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

// ======= BASIC INLINE STYLES =======

#[test]
fn test_inline_style_width() {
    let html = b"<div style=\"width: 200px;\">styled</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.width, 200.0, "Inline width should be 200px");
}

#[test]
fn test_inline_style_height() {
    let html = b"<div style=\"height: 100px;\">styled</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.height, 100.0, "Inline height should be 100px");
}

#[test]
fn test_inline_style_color() {
    let html = b"<div style=\"color: red;\">styled</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Red color
    assert!((style.color[0] - 1.0).abs() < 0.01, "Red should be 1.0");
    assert!((style.color[1] - 0.0).abs() < 0.01, "Green should be 0.0");
    assert!((style.color[2] - 0.0).abs() < 0.01, "Blue should be 0.0");
}

#[test]
fn test_inline_style_display() {
    let html = b"<div style=\"display: none;\">hidden</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.display, DISPLAY_NONE, "Inline display should be none");
}

// ======= MULTIPLE PROPERTIES =======

#[test]
fn test_inline_multiple_properties() {
    let html = b"<div style=\"width: 300px; height: 150px; display: flex;\">multi</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.width, 300.0, "Width should be 300px");
    assert_eq!(style.height, 150.0, "Height should be 150px");
    assert_eq!(style.display, DISPLAY_FLEX, "Display should be flex");
}

// ======= INLINE STYLES OVERRIDE CSS =======

#[test]
fn test_inline_overrides_css() {
    let html = b"<div class=\"box\" style=\"width: 500px;\">override</div>";
    let css = ".box { width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Inline style overrides CSS
    assert_eq!(style.width, 500.0, "Inline width should override CSS width");
    // CSS height still applies
    assert_eq!(style.height, 50.0, "CSS height should still apply");
}

// ======= INLINE STYLE ON MULTIPLE ELEMENTS =======

#[test]
fn test_multiple_elements_inline_styles() {
    let html = b"<div style=\"width: 100px;\">first</div><div style=\"width: 200px;\">second</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(divs.len() >= 2, "Should find 2 divs");

    assert_eq!(styles[divs[0]].width, 100.0, "First div width should be 100px");
    assert_eq!(styles[divs[1]].width, 200.0, "Second div width should be 200px");
}

// ======= COMPLEX INLINE VALUES =======

#[test]
fn test_inline_margin() {
    let html = b"<div style=\"margin: 10px 20px 30px 40px;\">margin</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.margin[0], 10.0, "Top margin");
    assert_eq!(style.margin[1], 20.0, "Right margin");
    assert_eq!(style.margin[2], 30.0, "Bottom margin");
    assert_eq!(style.margin[3], 40.0, "Left margin");
}

#[test]
fn test_inline_background_color() {
    let html = b"<div style=\"background-color: #ff0000;\">bg</div>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Red background
    assert!((style.background_color[0] - 1.0).abs() < 0.01, "Red should be 1.0");
}

// ======= COMPUTED STYLE SIZE =======

#[test]
fn test_computed_style_size() {
    let size = std::mem::size_of::<ComputedStyle>();
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}
