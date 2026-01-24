//! Test suite for Issue #68: GPU Box Shadow Rendering
//!
//! Tests GPU-accelerated CSS box-shadow:
//! - Simple box-shadow (offset-x, offset-y, color)
//! - Blur radius
//! - Spread radius
//! - Inset shadows
//! - Multiple shadows

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

// ======= BOX-SHADOW PARSING =======

#[test]
fn test_no_shadow_default() {
    let html = b"<div>no shadow</div>";
    let css = "div { width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Default: no shadow (shadow_count = 0)
    assert_eq!(style.shadow_count, 0, "Default should have no shadows");
}

#[test]
fn test_simple_box_shadow() {
    let html = b"<div class=\"shadow\">shadow</div>";
    let css = ".shadow { box-shadow: 5px 10px black; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 1, "Should have 1 shadow");
    assert_eq!(style.shadow_offset_x[0], 5.0, "Offset X should be 5px");
    assert_eq!(style.shadow_offset_y[0], 10.0, "Offset Y should be 10px");
    assert_eq!(style.shadow_blur[0], 0.0, "Default blur should be 0");
    assert_eq!(style.shadow_spread[0], 0.0, "Default spread should be 0");
}

#[test]
fn test_box_shadow_with_blur() {
    let html = b"<div class=\"shadow\">shadow</div>";
    let css = ".shadow { box-shadow: 5px 10px 15px black; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 1);
    assert_eq!(style.shadow_offset_x[0], 5.0);
    assert_eq!(style.shadow_offset_y[0], 10.0);
    assert_eq!(style.shadow_blur[0], 15.0, "Blur should be 15px");
}

#[test]
fn test_box_shadow_with_spread() {
    let html = b"<div class=\"shadow\">shadow</div>";
    let css = ".shadow { box-shadow: 5px 10px 15px 20px black; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 1);
    assert_eq!(style.shadow_blur[0], 15.0);
    assert_eq!(style.shadow_spread[0], 20.0, "Spread should be 20px");
}

#[test]
fn test_box_shadow_inset() {
    let html = b"<div class=\"shadow\">inset</div>";
    let css = ".shadow { box-shadow: inset 5px 10px 15px black; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 1);
    assert_eq!(style.shadow_inset[0], 1, "Should be inset shadow");
}

#[test]
fn test_box_shadow_color_rgb() {
    let html = b"<div class=\"shadow\">colored</div>";
    let css = ".shadow { box-shadow: 5px 10px rgb(255, 0, 0); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 1);
    // Red color
    assert!((style.shadow_color[0] - 1.0).abs() < 0.01, "Red should be 1.0");
    assert!((style.shadow_color[1] - 0.0).abs() < 0.01, "Green should be 0.0");
    assert!((style.shadow_color[2] - 0.0).abs() < 0.01, "Blue should be 0.0");
}

#[test]
fn test_box_shadow_color_rgba() {
    let html = b"<div class=\"shadow\">transparent</div>";
    let css = ".shadow { box-shadow: 5px 10px rgba(0, 0, 0, 0.5); }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 1);
    assert!((style.shadow_color[3] - 0.5).abs() < 0.01, "Alpha should be 0.5");
}

// ======= MULTIPLE SHADOWS =======

#[test]
fn test_multiple_shadows() {
    let html = b"<div class=\"shadow\">multi</div>";
    let css = ".shadow { box-shadow: 2px 2px black, 5px 5px red; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 2, "Should have 2 shadows");
    assert_eq!(style.shadow_offset_x[0], 2.0);
    assert_eq!(style.shadow_offset_y[0], 2.0);
    assert_eq!(style.shadow_offset_x[1], 5.0);
    assert_eq!(style.shadow_offset_y[1], 5.0);
}

// ======= BOX-SHADOW: NONE =======

#[test]
fn test_box_shadow_none() {
    let html = b"<div class=\"shadow\">none</div>";
    let css = ".shadow { box-shadow: none; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.shadow_count, 0, "box-shadow: none should have 0 shadows");
}

// ======= COMPUTED STYLE SIZE =======

#[test]
fn test_computed_style_size() {
    let size = std::mem::size_of::<ComputedStyle>();
    // Allow for expanded struct with shadow and gradient properties
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}
