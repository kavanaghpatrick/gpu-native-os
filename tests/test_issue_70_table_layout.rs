//! Test suite for Issue #70: GPU Table Layout
//!
//! Tests GPU-accelerated CSS table layout:
//! - display: table, table-row, table-cell
//! - Table structure parsing (table, tr, td, th)
//! - Basic table styling (border-collapse, border-spacing)

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, ComputedStyle, LayoutBox, Element,
    DISPLAY_TABLE, DISPLAY_TABLE_ROW, DISPLAY_TABLE_CELL,
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
const ELEM_TABLE: u32 = 17;
const ELEM_TR: u32 = 18;
const ELEM_TD: u32 = 19;
const ELEM_TH: u32 = 20;

// ======= DISPLAY TABLE PROPERTIES =======

#[test]
fn test_display_table() {
    let html = b"<div class=\"tbl\">table</div>";
    let css = ".tbl { display: table; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.display, DISPLAY_TABLE, "Should parse display: table");
}

#[test]
fn test_display_table_row() {
    let html = b"<div class=\"row\">row</div>";
    let css = ".row { display: table-row; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.display, DISPLAY_TABLE_ROW, "Should parse display: table-row");
}

#[test]
fn test_display_table_cell() {
    let html = b"<div class=\"cell\">cell</div>";
    let css = ".cell { display: table-cell; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let divs = find_elements_of_type(&elements, ELEM_DIV);
    let style = &styles[divs[0]];

    assert_eq!(style.display, DISPLAY_TABLE_CELL, "Should parse display: table-cell");
}

// ======= TABLE ELEMENT PARSING =======

#[test]
fn test_table_element_parsing() {
    let html = b"<table><tr><td>cell</td></tr></table>";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let tables = find_elements_of_type(&elements, ELEM_TABLE);
    let rows = find_elements_of_type(&elements, ELEM_TR);
    let cells = find_elements_of_type(&elements, ELEM_TD);

    assert!(!tables.is_empty(), "Should find table element");
    assert!(!rows.is_empty(), "Should find tr element");
    assert!(!cells.is_empty(), "Should find td element");
}

#[test]
fn test_th_element_parsing() {
    let html = b"<table><tr><th>header</th></tr></table>";
    let css = "";
    let (elements, _, _) = process_html(html, css, make_viewport(800.0, 600.0));

    let headers = find_elements_of_type(&elements, ELEM_TH);
    assert!(!headers.is_empty(), "Should find th element");
}

// ======= TABLE STYLING =======

#[test]
fn test_border_collapse() {
    let html = b"<table class=\"collapsed\"><tr><td>cell</td></tr></table>";
    let css = ".collapsed { border-collapse: collapse; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let tables = find_elements_of_type(&elements, ELEM_TABLE);
    let style = &styles[tables[0]];

    assert_eq!(style.border_collapse, 1, "Should parse border-collapse: collapse");
}

#[test]
fn test_border_spacing() {
    let html = b"<table class=\"spaced\"><tr><td>cell</td></tr></table>";
    let css = ".spaced { border-spacing: 10px; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let tables = find_elements_of_type(&elements, ELEM_TABLE);
    let style = &styles[tables[0]];

    assert_eq!(style.border_spacing, 10.0, "Should parse border-spacing: 10px");
}

// ======= TABLE DEFAULT STYLES =======

#[test]
fn test_table_styled_display() {
    // Explicit CSS styles for table elements
    let html = b"<table><tr><td>cell</td></tr></table>";
    let css = "table { display: table; } tr { display: table-row; } td { display: table-cell; }";
    let (elements, _, styles) = process_html(html, css, make_viewport(800.0, 600.0));

    let tables = find_elements_of_type(&elements, ELEM_TABLE);
    let rows = find_elements_of_type(&elements, ELEM_TR);
    let cells = find_elements_of_type(&elements, ELEM_TD);

    // Tables with explicit CSS should have correct display types
    assert_eq!(styles[tables[0]].display, DISPLAY_TABLE, "table should have display: table");
    assert_eq!(styles[rows[0]].display, DISPLAY_TABLE_ROW, "tr should have display: table-row");
    assert_eq!(styles[cells[0]].display, DISPLAY_TABLE_CELL, "td should have display: table-cell");
}

// ======= DISPLAY CONSTANTS =======

#[test]
fn test_display_table_constants() {
    assert_eq!(DISPLAY_TABLE, 5);
    assert_eq!(DISPLAY_TABLE_ROW, 6);
    assert_eq!(DISPLAY_TABLE_CELL, 7);
}

// ======= COMPUTED STYLE SIZE =======

#[test]
fn test_computed_style_size() {
    let size = std::mem::size_of::<ComputedStyle>();
    assert!(size % 4 == 0 && size <= 600, "ComputedStyle should have reasonable size: {}", size);
}
