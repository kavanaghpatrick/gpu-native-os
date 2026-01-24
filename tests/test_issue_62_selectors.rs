//! Test suite for Issue #62: Advanced CSS Selectors
//!
//! Tests GPU-native advanced CSS selector support including:
//! - Descendant selectors (E F)
//! - Child selectors (E > F)
//! - Sibling selectors (E + F, E ~ F)
//! - Attribute selectors
//! - Pseudo-classes

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, Stylesheet, ComputedStyle, Element,
    ELEM_DIV, ELEM_P, ELEM_SPAN, ELEM_UL, ELEM_LI, ELEM_TEXT,
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

fn process_html_css_with_elements(html: &[u8], css: &str) -> (Vec<ComputedStyle>, Vec<Element>) {
    let (mut tokenizer, mut parser, mut styler) = setup();
    let tokens = tokenizer.tokenize(html);
    let (elements, _) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    (styles, elements)
}

/// Find indices of all elements of a given type
fn find_elements_of_type(elements: &[Element], elem_type: u32) -> Vec<usize> {
    elements.iter()
        .enumerate()
        .filter(|(_, e)| e.element_type == elem_type)
        .map(|(i, _)| i)
        .collect()
}

fn colors_equal(a: [f32; 4], b: [f32; 4], tolerance: f32) -> bool {
    (a[0] - b[0]).abs() < tolerance &&
    (a[1] - b[1]).abs() < tolerance &&
    (a[2] - b[2]).abs() < tolerance &&
    (a[3] - b[3]).abs() < tolerance
}

// ======= DESCENDANT SELECTORS =======

#[test]
fn test_descendant_selector() {
    let html = b"<div><p>text</p></div>";
    let css = "div p { color: red; }";

    let styles = process_html_css(html, css);

    // p should be red
    assert!(colors_equal(styles[1].color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "div p should match, got {:?}", styles[1].color);
}

#[test]
fn test_descendant_selector_deep() {
    let html = b"<div><section><p>text</p></section></div>";
    let css = "div p { color: blue; }";

    let styles = process_html_css(html, css);

    // p should be blue (div is grandparent)
    let p_style = styles.iter().find(|s| (s.color[2] - 1.0).abs() < 0.01);
    assert!(p_style.is_some(), "div p should match deeply nested p");
}

#[test]
fn test_descendant_no_match() {
    let html = b"<section><p>text</p></section>";
    let css = "div p { color: red; }";

    let styles = process_html_css(html, css);

    // p should be default black (no div parent)
    assert!(colors_equal(styles[1].color, [0.0, 0.0, 0.0, 1.0], 0.01),
        "p without div parent should not match");
}

// ======= CHILD SELECTORS =======

#[test]
fn test_child_selector() {
    let html = b"<div><p>text</p></div>";
    let css = "div > p { color: green; }";

    let styles = process_html_css(html, css);

    // p is direct child of div
    assert!(colors_equal(styles[1].color, [0.0, 0.5, 0.0, 1.0], 0.01),
        "div > p should match direct child");
}

#[test]
fn test_child_selector_no_match() {
    let html = b"<div><section><p>text</p></section></div>";
    let css = "div > p { color: green; }";

    let styles = process_html_css(html, css);

    // p is grandchild, not direct child
    // p should be default black
    let p_idx = 2; // div > section > p
    assert!(colors_equal(styles[p_idx].color, [0.0, 0.0, 0.0, 1.0], 0.01),
        "div > p should not match grandchild, got {:?}", styles[p_idx].color);
}

// ======= PSEUDO-CLASSES =======

#[test]
fn test_first_child() {
    let html = b"<ul><li>first</li><li>second</li><li>third</li></ul>";
    let css = "li:first-child { color: red; }";

    let (styles, elements) = process_html_css_with_elements(html, css);
    let li_indices = find_elements_of_type(&elements, ELEM_LI);
    assert!(li_indices.len() >= 2, "Should have at least 2 li elements");

    // First li should be red
    let first_li = &styles[li_indices[0]];
    assert!(colors_equal(first_li.color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "li:first-child should be red, got {:?}", first_li.color);

    // Second li should be default
    let second_li = &styles[li_indices[1]];
    assert!(colors_equal(second_li.color, [0.0, 0.0, 0.0, 1.0], 0.01),
        "Second li should not match :first-child, got {:?}", second_li.color);
}

#[test]
fn test_last_child() {
    let html = b"<ul><li>first</li><li>second</li><li>third</li></ul>";
    let css = "li:last-child { color: blue; }";

    let (styles, elements) = process_html_css_with_elements(html, css);
    let li_indices = find_elements_of_type(&elements, ELEM_LI);
    assert!(li_indices.len() >= 3, "Should have at least 3 li elements");

    // Last li should be blue
    let last_li = &styles[li_indices[2]];
    assert!(colors_equal(last_li.color, [0.0, 0.0, 1.0, 1.0], 0.01),
        "li:last-child should be blue, got {:?}", last_li.color);

    // First li should be default
    let first_li = &styles[li_indices[0]];
    assert!(colors_equal(first_li.color, [0.0, 0.0, 0.0, 1.0], 0.01),
        "First li should not match :last-child, got {:?}", first_li.color);
}

#[test]
fn test_nth_child_even() {
    let html = b"<ul><li>1</li><li>2</li><li>3</li><li>4</li></ul>";
    let css = "li:nth-child(even) { color: orange; }";

    let (styles, elements) = process_html_css_with_elements(html, css);
    let li_indices = find_elements_of_type(&elements, ELEM_LI);
    assert!(li_indices.len() >= 4, "Should have at least 4 li elements");

    // Even li's (2, 4) should be orange (0-indexed: 1, 3)
    let li2 = &styles[li_indices[1]];
    let li4 = &styles[li_indices[3]];

    assert!(colors_equal(li2.color, [1.0, 0.65, 0.0, 1.0], 0.05),
        "li:nth-child(even) should match 2nd li, got {:?}", li2.color);
    assert!(colors_equal(li4.color, [1.0, 0.65, 0.0, 1.0], 0.05),
        "li:nth-child(even) should match 4th li, got {:?}", li4.color);
}

#[test]
fn test_nth_child_odd() {
    let html = b"<ul><li>1</li><li>2</li><li>3</li></ul>";
    let css = "li:nth-child(odd) { color: purple; }";

    let (styles, elements) = process_html_css_with_elements(html, css);
    let li_indices = find_elements_of_type(&elements, ELEM_LI);
    assert!(li_indices.len() >= 3, "Should have at least 3 li elements");

    // Odd li's (1, 3) should be purple (0-indexed: 0, 2)
    let li1 = &styles[li_indices[0]];
    let li3 = &styles[li_indices[2]];

    assert!(colors_equal(li1.color, [0.5, 0.0, 0.5, 1.0], 0.01),
        "li:nth-child(odd) should match 1st li, got {:?}", li1.color);
    assert!(colors_equal(li3.color, [0.5, 0.0, 0.5, 1.0], 0.01),
        "li:nth-child(odd) should match 3rd li, got {:?}", li3.color);
}

#[test]
fn test_only_child() {
    let html = b"<div><p>only</p></div><div><p>one</p><p>two</p></div>";
    let css = "p:only-child { color: cyan; }";

    let styles = process_html_css(html, css);

    // First p is only child
    let first_p = &styles[1];
    assert!(colors_equal(first_p.color, [0.0, 1.0, 1.0, 1.0], 0.01),
        "p:only-child should match, got {:?}", first_p.color);
}

#[test]
fn test_empty() {
    let html = b"<div></div><div><p>content</p></div>";
    let css = "div:empty { background: yellow; }";

    let styles = process_html_css(html, css);

    // First div is empty
    assert!(colors_equal(styles[0].background_color, [1.0, 1.0, 0.0, 1.0], 0.01),
        "div:empty should match empty div");
}

// ======= COMBINED SELECTORS =======

#[test]
fn test_tag_and_class() {
    let html = b"<p class=\"highlight\">text</p><span class=\"highlight\">other</span>";
    let css = "p.highlight { color: red; }";

    let (styles, elements) = process_html_css_with_elements(html, css);
    let p_indices = find_elements_of_type(&elements, ELEM_P);
    let span_indices = find_elements_of_type(&elements, ELEM_SPAN);

    assert!(!p_indices.is_empty(), "Should have at least 1 p element");
    assert!(!span_indices.is_empty(), "Should have at least 1 span element");

    // p.highlight should be red
    assert!(colors_equal(styles[p_indices[0]].color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "p.highlight should be red, got {:?}", styles[p_indices[0]].color);

    // span.highlight should be default (not p)
    assert!(colors_equal(styles[span_indices[0]].color, [0.0, 0.0, 0.0, 1.0], 0.01),
        "span.highlight should not match p.highlight, got {:?}", styles[span_indices[0]].color);
}

#[test]
fn test_complex_descendant_with_class() {
    let html = b"<div class=\"container\"><ul><li>item</li></ul></div>";
    let css = ".container li { color: red; }";

    let styles = process_html_css(html, css);

    // li inside .container should be red
    let li_idx = 2; // div > ul > li
    assert!(colors_equal(styles[li_idx].color, [1.0, 0.0, 0.0, 1.0], 0.01),
        ".container li should be red");
}

// ======= SPECIFICITY =======

#[test]
fn test_specificity_tag_vs_class() {
    let html = b"<p class=\"special\">text</p>";
    let css = "p { color: blue; } .special { color: red; }";

    let styles = process_html_css(html, css);

    // Class has higher specificity
    assert!(colors_equal(styles[0].color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "Class should override tag, got {:?}", styles[0].color);
}

#[test]
fn test_specificity_descendant_adds_up() {
    let html = b"<div><p>text</p></div>";
    let css = "p { color: blue; } div p { color: red; }";

    let styles = process_html_css(html, css);

    // "div p" has higher specificity than just "p"
    assert!(colors_equal(styles[1].color, [1.0, 0.0, 0.0, 1.0], 0.01),
        "div p should override p, got {:?}", styles[1].color);
}

// ======= PERFORMANCE =======

#[test]
fn test_performance_complex_selectors() {
    let (mut tokenizer, mut parser, mut styler) = setup();

    // Generate HTML with nested structure
    let mut html = String::from("<div class=\"container\">");
    for i in 0..100 {
        html.push_str(&format!("<ul><li class=\"item-{}\">Item {}</li></ul>", i % 5, i));
    }
    html.push_str("</div>");

    // Complex selectors
    let css = r#"
        .container li { color: blue; }
        .container li:first-child { color: red; }
        ul > li { font-size: 14px; }
        .item-0 { font-weight: bold; }
    "#;

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

    println!("Complex selectors on {} elements: {:?}", styles.len(), elapsed);

    assert!(elapsed.as_millis() < 50, "Complex selectors took too long: {:?}", elapsed);
}
