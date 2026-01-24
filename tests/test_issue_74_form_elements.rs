//! Test suite for Issue #74: GPU Form Elements Display
//!
//! Tests GPU-accelerated rendering of HTML form elements:
//! - <input> (text, button, checkbox, radio)
//! - <button>
//! - <textarea>
//! - <select>
//! - Basic form structure

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
};

const ELEM_INPUT: u32 = 24;
const ELEM_BUTTON: u32 = 25;
const ELEM_TEXTAREA: u32 = 26;
const ELEM_SELECT: u32 = 27;

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

// ======= INPUT ELEMENT =======

#[test]
fn test_input_element_parsing() {
    let html = b"<input type=\"text\" />";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let inputs = find_elements_of_type(&elements, ELEM_INPUT);
    assert!(!inputs.is_empty(), "Should find input element");
}

#[test]
fn test_input_button_parsing() {
    let html = b"<input type=\"button\" value=\"Click\" />";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let inputs = find_elements_of_type(&elements, ELEM_INPUT);
    assert!(!inputs.is_empty(), "Should find input button");
}

// ======= BUTTON ELEMENT =======

#[test]
fn test_button_element_parsing() {
    let html = b"<button>Submit</button>";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let buttons = find_elements_of_type(&elements, ELEM_BUTTON);
    assert!(!buttons.is_empty(), "Should find button element");
}

// ======= TEXTAREA ELEMENT =======

#[test]
fn test_textarea_element_parsing() {
    let html = b"<textarea>text content</textarea>";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let textareas = find_elements_of_type(&elements, ELEM_TEXTAREA);
    assert!(!textareas.is_empty(), "Should find textarea element");
}

// ======= SELECT ELEMENT =======

#[test]
fn test_select_element_parsing() {
    let html = b"<select><option>A</option><option>B</option></select>";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let selects = find_elements_of_type(&elements, ELEM_SELECT);
    assert!(!selects.is_empty(), "Should find select element");
}

// ======= FORM STRUCTURE =======

#[test]
fn test_form_with_multiple_inputs() {
    let html = b"<form><input /><input /><button>Go</button></form>";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let inputs = find_elements_of_type(&elements, ELEM_INPUT);
    let buttons = find_elements_of_type(&elements, ELEM_BUTTON);
    assert_eq!(inputs.len(), 2, "Should find 2 inputs");
    assert_eq!(buttons.len(), 1, "Should find 1 button");
}

// ======= FORM ELEMENT STYLING =======

#[test]
fn test_input_styling() {
    let html = b"<input style=\"width: 200px; height: 30px;\" />";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let inputs = find_elements_of_type(&elements, ELEM_INPUT);
    let style = &styles[inputs[0]];
    assert_eq!(style.width, 200.0, "Input width should be 200px");
    assert_eq!(style.height, 30.0, "Input height should be 30px");
}

#[test]
fn test_button_styling() {
    let html = b"<button style=\"width: 100px; background-color: blue;\">Click</button>";
    let css = "";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let buttons = find_elements_of_type(&elements, ELEM_BUTTON);
    let style = &styles[buttons[0]];
    assert_eq!(style.width, 100.0, "Button width should be 100px");
    // Blue background
    assert!((style.background_color[2] - 1.0).abs() < 0.01, "Blue channel should be 1.0");
}

// ======= COMPUTED STYLE =======

#[test]
fn test_computed_style_size() {
    let size = std::mem::size_of::<ComputedStyle>();
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}
