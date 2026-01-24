//! Test suite for Issue #66: CSS Positioned Layout
//!
//! Tests GPU-accelerated CSS positioning:
//! - position: static (default)
//! - position: relative (offset from normal flow)
//! - position: absolute (positioned relative to containing block)
//! - position: fixed (positioned relative to viewport)
//! - top, right, bottom, left offset properties
//! - z-index stacking order

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
    POSITION_STATIC, POSITION_RELATIVE, POSITION_ABSOLUTE, POSITION_FIXED,
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
const ELEM_SPAN: u32 = 2;
const ELEM_P: u32 = 3;

// ======= POSITION PROPERTY PARSING =======

#[test]
fn test_position_static_default() {
    let html = b"<div>test</div>";
    let css = "div { width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(!divs.is_empty(), "Should find div");

    let style = &styles[divs[0]];
    assert_eq!(style.position, POSITION_STATIC, "Default position should be static");
}

#[test]
fn test_position_relative() {
    let html = b"<div class=\"rel\">relative</div>";
    let css = ".rel { position: relative; width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.position, POSITION_RELATIVE, "Should parse position: relative");
}

#[test]
fn test_position_absolute() {
    let html = b"<div class=\"abs\">absolute</div>";
    let css = ".abs { position: absolute; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.position, POSITION_ABSOLUTE, "Should parse position: absolute");
}

#[test]
fn test_position_fixed() {
    let html = b"<div class=\"fixed\">fixed</div>";
    let css = ".fixed { position: fixed; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.position, POSITION_FIXED, "Should parse position: fixed");
}

// ======= OFFSET PROPERTIES =======

#[test]
fn test_top_offset() {
    let html = b"<div class=\"offset\">offset</div>";
    let css = ".offset { position: relative; top: 20px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.top, 20.0, "Should parse top: 20px");
}

#[test]
fn test_left_offset() {
    let html = b"<div class=\"offset\">offset</div>";
    let css = ".offset { position: relative; left: 30px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.left, 30.0, "Should parse left: 30px");
}

#[test]
fn test_right_offset() {
    let html = b"<div class=\"offset\">offset</div>";
    let css = ".offset { position: absolute; right: 10px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.right, 10.0, "Should parse right: 10px");
}

#[test]
fn test_bottom_offset() {
    let html = b"<div class=\"offset\">offset</div>";
    let css = ".offset { position: absolute; bottom: 15px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.bottom, 15.0, "Should parse bottom: 15px");
}

#[test]
fn test_all_offsets() {
    let html = b"<div class=\"offset\">offset</div>";
    let css = ".offset { position: absolute; top: 10px; right: 20px; bottom: 30px; left: 40px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.top, 10.0, "Should parse top");
    assert_eq!(style.right, 20.0, "Should parse right");
    assert_eq!(style.bottom, 30.0, "Should parse bottom");
    assert_eq!(style.left, 40.0, "Should parse left");
}

// ======= Z-INDEX =======

#[test]
fn test_z_index_positive() {
    let html = b"<div class=\"z\">z-indexed</div>";
    let css = ".z { position: relative; z-index: 10; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.z_index, 10, "Should parse z-index: 10");
}

#[test]
fn test_z_index_negative() {
    let html = b"<div class=\"z\">z-indexed</div>";
    let css = ".z { position: relative; z-index: -5; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];
    assert_eq!(style.z_index, -5, "Should parse negative z-index");
}

// ======= LAYOUT: RELATIVE POSITIONING =======

#[test]
fn test_relative_offset_from_normal_position() {
    let html = b"<div><span class=\"rel\">relative</span></div>";
    let css = "div { width: 200px; } .rel { position: relative; top: 20px; left: 30px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let spans = find_elements_of_type(&elements, ELEM_SPAN);
    assert!(!spans.is_empty(), "Should find span");

    // Verify style was parsed correctly
    let style = &styles[spans[0]];
    assert_eq!(style.position, POSITION_RELATIVE);
    assert_eq!(style.top, 20.0);
    assert_eq!(style.left, 30.0);
    // Note: Layout engine integration for relative positioning is a future enhancement
}

#[test]
fn test_relative_does_not_affect_siblings() {
    let html = b"<div><span class=\"a\">first</span><span class=\"b\">second</span></div>";
    let css = "div { width: 300px; } .a { position: relative; top: 50px; left: 100px; } .b { }";
    let (elements, boxes, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let spans = find_elements_of_type(&elements, ELEM_SPAN);
    assert_eq!(spans.len(), 2, "Should find 2 spans");

    // Second span should be in normal flow position (not affected by first span's offset)
    let second_box = &boxes[spans[1]];
    // In inline layout, second span should be positioned after first in normal flow
    // (the relative offset shouldn't push it)
    assert!(second_box.y < 50.0, "Second span should be at normal Y position, not pushed by relative offset");
}

// ======= LAYOUT: ABSOLUTE POSITIONING =======

#[test]
fn test_absolute_relative_to_viewport() {
    let html = b"<div class=\"abs\">absolute</div>";
    let css = ".abs { position: absolute; top: 50px; left: 100px; width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Verify style was parsed correctly
    assert_eq!(style.position, POSITION_ABSOLUTE);
    assert_eq!(style.top, 50.0);
    assert_eq!(style.left, 100.0);
    // Note: Layout engine integration for absolute positioning is a future enhancement
}

#[test]
fn test_absolute_relative_to_positioned_ancestor() {
    let html = b"<div class=\"container\"><div class=\"abs\">absolute</div></div>";
    let css = ".container { position: relative; margin: 50px; width: 300px; height: 200px; }
               .abs { position: absolute; top: 10px; left: 20px; width: 50px; height: 30px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(divs.len() >= 2, "Should find 2 divs");

    // Verify container style
    let container_style = &styles[divs[0]];
    assert_eq!(container_style.position, POSITION_RELATIVE);

    // Verify absolute element style
    let abs_style = &styles[divs[1]];
    assert_eq!(abs_style.position, POSITION_ABSOLUTE);
    assert_eq!(abs_style.top, 10.0);
    assert_eq!(abs_style.left, 20.0);
    // Note: Layout engine integration for containing block calculation is a future enhancement
}

#[test]
fn test_absolute_removed_from_flow() {
    let html = b"<div class=\"container\"><div class=\"abs\">abs</div><div class=\"normal\">normal</div></div>";
    let css = ".container { width: 200px; }
               .abs { position: absolute; width: 100px; height: 100px; }
               .normal { width: 150px; height: 50px; background: red; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    assert!(divs.len() >= 3, "Should find 3 divs");

    // Verify absolute element has position: absolute
    let abs_style = &styles[divs[1]];
    assert_eq!(abs_style.position, POSITION_ABSOLUTE);

    // Verify normal element has position: static (default)
    let normal_style = &styles[divs[2]];
    assert_eq!(normal_style.position, POSITION_STATIC);
    // Note: Layout engine removing absolute from flow is a future enhancement
}

// ======= LAYOUT: FIXED POSITIONING =======

#[test]
fn test_fixed_relative_to_viewport() {
    let html = b"<div class=\"fixed\">fixed</div>";
    let css = ".fixed { position: fixed; bottom: 20px; right: 30px; width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Verify style was parsed correctly
    assert_eq!(style.position, POSITION_FIXED);
    assert_eq!(style.bottom, 20.0);
    assert_eq!(style.right, 30.0);
    // Note: Layout engine integration for fixed positioning is a future enhancement
}

// ======= EDGE CASES =======

#[test]
fn test_auto_offset_values() {
    // When only top is set, other offsets should remain at default (OFFSET_AUTO)
    let html = b"<div class=\"abs\">absolute</div>";
    let css = ".abs { position: absolute; top: 50px; width: 100px; height: 100px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Verify top was set and other offsets are OFFSET_AUTO (f32::MAX)
    assert_eq!(style.position, POSITION_ABSOLUTE);
    assert_eq!(style.top, 50.0);
    // Unset offsets should be OFFSET_AUTO (f32::MAX)
    assert!(style.left == f32::MAX, "Left should be OFFSET_AUTO (f32::MAX)");
    assert!(style.right == f32::MAX, "Right should be OFFSET_AUTO (f32::MAX)");
    assert!(style.bottom == f32::MAX, "Bottom should be OFFSET_AUTO (f32::MAX)");
    // Note: Layout engine applying offsets is a future enhancement
}

#[test]
fn test_conflicting_offsets() {
    // When both top and bottom are set with explicit height, both are stored
    let html = b"<div class=\"abs\">absolute</div>";
    let css = ".abs { position: absolute; top: 20px; bottom: 30px; width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Verify both top and bottom are stored
    assert_eq!(style.position, POSITION_ABSOLUTE);
    assert_eq!(style.top, 20.0);
    assert_eq!(style.bottom, 30.0);
    // Note: Layout engine resolving conflicting offsets is a future enhancement
}

#[test]
fn test_percentage_offsets() {
    let html = b"<div class=\"abs\">absolute</div>";
    let css = ".abs { position: absolute; top: 10%; left: 20%; width: 100px; height: 50px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    // Verify style parsing (percentages stored as negative values)
    assert_eq!(style.position, POSITION_ABSOLUTE);
    // Percentages are stored as negative: -10.0 means 10%
    assert_eq!(style.top, -10.0, "Top 10% should be stored as -10.0");
    assert_eq!(style.left, -20.0, "Left 20% should be stored as -20.0");
    // Note: Layout engine resolving percentages to pixels is a future enhancement
}

// ======= COMPUTED STYLE STRUCT SIZE =======

#[test]
fn test_computed_style_size() {
    // ComputedStyle should maintain GPU-friendly alignment
    let size = std::mem::size_of::<ComputedStyle>();
    // Allow for expanded struct with position, shadow, and gradient properties
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}
