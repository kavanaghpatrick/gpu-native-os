//! Tests for vertex initialization (Issue #86)
//!
//! Validates that all allocated vertex slots are properly written
//! during the paint pass - no uninitialized vertices should exist.

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine,
    Stylesheet, Viewport, PaintVertex,
    FLAG_BACKGROUND, FLAG_BORDER, FLAG_TEXT, FLAG_IMAGE,
};

fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine) {
    let device = Device::system_default().expect("No Metal device");
    let tokenizer = GpuTokenizer::new(&device).expect("tokenizer");
    let parser = GpuParser::new(&device).expect("parser");
    let styler = GpuStyler::new(&device).expect("styler");
    let layout = GpuLayoutEngine::new(&device).expect("layout");
    let paint = GpuPaintEngine::new(&device).expect("paint");
    (tokenizer, parser, styler, layout, paint)
}

fn make_viewport(width: f32, height: f32) -> Viewport {
    Viewport {
        width,
        height,
        _padding: [0.0; 2],
    }
}

fn process_html(html: &[u8]) -> Vec<PaintVertex> {
    let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(html);
    let (elements, text_buffer) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    paint.paint(&elements, &boxes, &styles, &text_buffer, viewport)
}

fn count_uninitialized(vertices: &[PaintVertex]) -> usize {
    vertices.iter().filter(|v| v.flags == 0).count()
}

fn count_by_flag(vertices: &[PaintVertex]) -> (usize, usize, usize, usize, usize) {
    let mut bg = 0;
    let mut border = 0;
    let mut text = 0;
    let mut image = 0;
    let mut uninit = 0;

    for v in vertices {
        match v.flags {
            f if f == FLAG_BACKGROUND => bg += 1,
            f if f == FLAG_BORDER => border += 1,
            f if f == FLAG_TEXT => text += 1,
            f if f == FLAG_IMAGE => image += 1,
            0 => uninit += 1,
            _ => {}
        }
    }
    (bg, border, text, image, uninit)
}

// ============================================================================
// Tests for Issue #86: Uninitialized Vertices
// ============================================================================

#[test]
fn test_simple_html_no_uninitialized() {
    let html = b"<html><body><div>Hello World</div></body></html>";
    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "Simple HTML should have no uninitialized vertices");
}

#[test]
fn test_styled_html_no_uninitialized() {
    let html = br#"<html><head><style>
        div { background: red; padding: 10px; margin: 5px; }
    </style></head><body>
        <div>Box 1</div>
        <div>Box 2</div>
        <div>Box 3</div>
    </body></html>"#;

    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "Styled HTML should have no uninitialized vertices");
}

#[test]
fn test_inline_styles_no_uninitialized() {
    let html = br#"<html><body>
        <div style="width: 100px; height: 50px; background: blue;">A</div>
        <div style="display: none;">Hidden</div>
        <div style="width: 200px; background: green;">B</div>
    </body></html>"#;

    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "Inline styles should have no uninitialized vertices");
}

#[test]
fn test_nested_elements_no_uninitialized() {
    let html = br#"<html><body>
        <div><div><div><span>Deep</span></div></div></div>
    </body></html>"#;

    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "Nested elements should have no uninitialized vertices");
}

#[test]
fn test_mixed_content_no_uninitialized() {
    let html = br#"<html><body>
        <h1>Title</h1>
        <p>Paragraph with <strong>bold</strong> and <em>italic</em>.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        <div style="border: 2px solid black; padding: 10px;">
            <span>Inside border</span>
        </div>
    </body></html>"#;

    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "Mixed content should have no uninitialized vertices");
}

#[test]
fn test_empty_elements_no_uninitialized() {
    let html = br#"<html><body>
        <div></div>
        <span></span>
        <p></p>
        <div>   </div>
    </body></html>"#;

    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "Empty elements should have no uninitialized vertices");
}

#[test]
fn test_display_none_no_uninitialized() {
    let html = br#"<html><body>
        <div style="display: none;">Hidden 1</div>
        <div>Visible</div>
        <div style="display: none;"><span>Hidden nested</span></div>
    </body></html>"#;

    let vertices = process_html(html);
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "display:none elements should have no uninitialized vertices");
}

#[test]
fn test_stress_100_elements_no_uninitialized() {
    let mut html = String::from("<html><body>");
    for i in 0..100 {
        html.push_str(&format!("<div style=\"background: #{:06x};\">Item {}</div>",
            (i * 12345) % 0xFFFFFF, i));
    }
    html.push_str("</body></html>");

    let vertices = process_html(html.as_bytes());
    let uninit = count_uninitialized(&vertices);
    assert_eq!(uninit, 0, "100 elements should have no uninitialized vertices, found {}", uninit);
}

#[test]
fn test_all_vertices_have_valid_flags() {
    let html = br#"<html><body>
        <div style="background: red; border: 1px solid black;">
            <span>Text content</span>
        </div>
    </body></html>"#;

    let vertices = process_html(html);
    let (bg, border, text, image, uninit) = count_by_flag(&vertices);

    println!("Vertex breakdown: bg={}, border={}, text={}, image={}, uninit={}",
        bg, border, text, image, uninit);

    assert_eq!(uninit, 0, "All vertices should have valid flags");
    assert!(bg + border + text + image + uninit == vertices.len(),
        "All vertices should be accounted for");
}

#[test]
#[ignore] // Enable when investigating specific patterns
fn test_identify_uninitialized_pattern() {
    // This test helps identify which element patterns cause uninitialized vertices
    let test_cases = [
        ("basic", "<div>text</div>"),
        ("styled", "<div style=\"background: red;\">text</div>"),
        ("border", "<div style=\"border: 1px solid black;\">text</div>"),
        ("nested", "<div><span>text</span></div>"),
        ("empty", "<div></div>"),
        ("whitespace", "<div>   </div>"),
        ("hidden", "<div style=\"display: none;\">text</div>"),
    ];

    for (name, html) in test_cases {
        let full_html = format!("<html><body>{}</body></html>", html);
        let vertices = process_html(full_html.as_bytes());
        let uninit = count_uninitialized(&vertices);
        println!("{}: {} vertices, {} uninitialized", name, vertices.len(), uninit);

        if uninit > 0 {
            println!("  Uninitialized vertices at indices:");
            for (i, v) in vertices.iter().enumerate() {
                if v.flags == 0 {
                    println!("    [{}] pos=({:.2}, {:.2}) color=({:.2}, {:.2}, {:.2}, {:.2})",
                        i, v.position[0], v.position[1],
                        v.color[0], v.color[1], v.color[2], v.color[3]);
                }
            }
        }
    }
}
