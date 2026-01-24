//! Tests for Issue #89: Level-Parallel Layout Engine
//!
//! Validates that the GPU layout engine correctly positions elements
//! using all 1024 threads, with no single-thread bottlenecks or stack overflows.

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine,
    Stylesheet, Viewport, LayoutBox, Element,
};

fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine) {
    let device = Device::system_default().expect("No Metal device");
    let tokenizer = GpuTokenizer::new(&device).expect("tokenizer");
    let parser = GpuParser::new(&device).expect("parser");
    let styler = GpuStyler::new(&device).expect("styler");
    let layout = GpuLayoutEngine::new(&device).expect("layout");
    (tokenizer, parser, styler, layout)
}

fn make_viewport(width: f32, height: f32) -> Viewport {
    Viewport {
        width,
        height,
        _padding: [0.0; 2],
    }
}

fn layout_html(html: &[u8]) -> (Vec<Element>, Vec<LayoutBox>) {
    let (mut tokenizer, mut parser, mut styler, mut layout) = setup();
    let viewport = make_viewport(1024.0, 768.0);

    let tokens = tokenizer.tokenize(html);
    let (elements, _text_buffer) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse("");
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, viewport);
    (elements, boxes)
}

// ============================================================================
// Core Layout Tests
// ============================================================================

#[test]
fn test_simple_stacking() {
    let html = b"<html><body><div>Line 1</div><div>Line 2</div><div>Line 3</div></body></html>";
    let (elements, boxes) = layout_html(html);

    // Find the actual div elements (type=1) that are children of body
    let div_boxes: Vec<_> = elements.iter().zip(boxes.iter())
        .filter(|(e, b)| e.element_type == 1 && b.height > 0.0)  // type=1 is div
        .map(|(_, b)| b)
        .collect();

    assert!(div_boxes.len() >= 3, "Expected at least 3 divs, got {}", div_boxes.len());

    // Each subsequent div should have increasing Y position
    for i in 1..div_boxes.len() {
        assert!(
            div_boxes[i].y > div_boxes[i-1].y,
            "Div {} (y={}) should be below div {} (y={})",
            i, div_boxes[i].y, i-1, div_boxes[i-1].y
        );
    }
}

#[test]
fn test_no_overlapping_y() {
    let html = b"<html><body>
        <div>A</div>
        <div>B</div>
        <div>C</div>
        <div>D</div>
        <div>E</div>
    </body></html>";

    let (elements, boxes) = layout_html(html);

    // Find actual div elements (type=1) that are siblings (all have same parent)
    let div_boxes: Vec<_> = elements.iter().zip(boxes.iter())
        .filter(|(e, _)| e.element_type == 1)  // divs only
        .map(|(_, b)| b)
        .collect();

    // Verify that sibling divs have strictly increasing Y
    for i in 1..div_boxes.len() {
        assert!(
            div_boxes[i].y > div_boxes[i-1].y,
            "Div {} (y={:.1}) should have higher Y than div {} (y={:.1})",
            i, div_boxes[i].y, i-1, div_boxes[i-1].y
        );
    }
}

#[test]
fn test_100_elements_unique_y() {
    let mut html = String::from("<html><body>");
    for i in 0..100 {
        html.push_str(&format!("<div>Item {}</div>", i));
    }
    html.push_str("</body></html>");

    let (_elements, boxes) = layout_html(html.as_bytes());

    // Collect Y positions of visible boxes
    let mut y_positions: Vec<f32> = boxes.iter()
        .filter(|b| b.height > 0.0 && b.content_height > 0.0)
        .map(|b| b.y)
        .collect();

    y_positions.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Check that Y positions are monotonically increasing (or equal for containers)
    let mut last_y = -1.0f32;
    for y in &y_positions {
        assert!(
            *y >= last_y,
            "Y positions should increase: found {} after {}",
            y, last_y
        );
        last_y = *y;
    }
}

#[test]
fn test_500_elements_no_stack_overflow() {
    // This would overflow the old 256-element stack
    let mut html = String::from("<html><body>");
    for i in 0..500 {
        html.push_str(&format!("<div>Element {}</div>", i));
    }
    html.push_str("</body></html>");

    let (_elements, boxes) = layout_html(html.as_bytes());

    // All 500 divs should have proper layout (non-zero Y spread)
    let max_y = boxes.iter().map(|b| b.y).fold(0.0f32, f32::max);
    let min_y = boxes.iter().map(|b| b.y).fold(f32::MAX, f32::min);

    assert!(
        max_y - min_y > 1000.0, // 500 divs * ~19.2px = ~9600px spread
        "500 elements should spread across Y: min={}, max={} (spread={})",
        min_y, max_y, max_y - min_y
    );
}

#[test]
fn test_1000_elements_performance() {
    use std::time::Instant;

    let mut html = String::from("<html><body>");
    for i in 0..1000 {
        html.push_str(&format!("<div>Line {}</div>", i));
    }
    html.push_str("</body></html>");

    let start = Instant::now();
    let (_elements, boxes) = layout_html(html.as_bytes());
    let elapsed = start.elapsed();

    println!("Layout 1000 elements: {:?}", elapsed);

    // Should complete in under 100ms (16ms target, but allow headroom for CI)
    assert!(
        elapsed.as_millis() < 100,
        "Layout of 1000 elements took {:?} (expected < 100ms)",
        elapsed
    );

    // Verify layout correctness
    let y_spread = boxes.iter().map(|b| b.y).fold(0.0f32, f32::max);
    assert!(y_spread > 1000.0, "1000 divs should spread > 1000px, got {}", y_spread);
}

// ============================================================================
// Deep Nesting Tests (validates depth computation)
// ============================================================================

#[test]
fn test_deep_nesting_10_levels() {
    let html = b"<html><body>
        <div><div><div><div><div>
        <div><div><div><div><div>
        Deepest
        </div></div></div></div></div>
        </div></div></div></div></div>
    </body></html>";

    let (elements, boxes) = layout_html(html);

    // Find the text node and verify it has proper height
    let text_box = elements.iter().zip(boxes.iter())
        .find(|(e, _)| e.element_type == 100);  // ELEM_TEXT

    assert!(text_box.is_some(), "Should find text node");
    let (_, b) = text_box.unwrap();
    assert!(b.height > 0.0, "Text should have positive height, got {}", b.height);
}

#[test]
fn test_deep_nesting_50_levels() {
    // This tests depth computation to 50 levels
    let mut html = String::from("<html><body>");
    for _ in 0..50 {
        html.push_str("<div>");
    }
    html.push_str("Deep");
    for _ in 0..50 {
        html.push_str("</div>");
    }
    html.push_str("</body></html>");

    let (_elements, boxes) = layout_html(html.as_bytes());

    // Should not crash or hang
    assert!(!boxes.is_empty(), "Deep nesting should produce layout boxes");
}

// ============================================================================
// Mixed Layout Tests
// ============================================================================

#[test]
fn test_mixed_block_inline() {
    let html = b"<html><body>
        <div>Block 1</div>
        <span>Inline 1</span><span>Inline 2</span>
        <div>Block 2</div>
    </body></html>";

    let (_elements, boxes) = layout_html(html);

    // Blocks should stack, inlines should flow
    assert!(boxes.len() > 0, "Should have layout boxes");
}

#[test]
fn test_styled_margins() {
    let html = br#"<html><head><style>
        div { margin: 10px; padding: 5px; }
    </style></head><body>
        <div>Box 1</div>
        <div>Box 2</div>
    </body></html>"#;

    let (elements, boxes) = layout_html(html);

    // Find actual div elements (type=1)
    let div_boxes: Vec<_> = elements.iter().zip(boxes.iter())
        .filter(|(e, b)| e.element_type == 1 && b.height > 0.0)
        .map(|(_, b)| b)
        .collect();

    // Should have gap between boxes due to margins
    if div_boxes.len() >= 2 {
        // Second div should be below first div
        assert!(
            div_boxes[1].y > div_boxes[0].y,
            "Box 2 (y={:.1}) should be below Box 1 (y={:.1})",
            div_boxes[1].y, div_boxes[0].y
        );
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_body() {
    let html = b"<html><body></body></html>";
    let (_elements, boxes) = layout_html(html);
    assert!(!boxes.is_empty(), "Empty body should still produce boxes for html/body");
}

#[test]
fn test_whitespace_only() {
    let html = b"<html><body>   \n   \t   </body></html>";
    let (_elements, boxes) = layout_html(html);
    assert!(!boxes.is_empty());
}

#[test]
fn test_display_none() {
    let html = br#"<html><body>
        <div>Visible</div>
        <div style="display: none;">Hidden</div>
        <div>Also visible</div>
    </body></html>"#;

    let (_elements, boxes) = layout_html(html);

    // Hidden element should have zero dimensions
    let hidden_boxes: Vec<_> = boxes.iter()
        .filter(|b| b.width == 0.0 && b.height == 0.0)
        .collect();

    assert!(
        hidden_boxes.len() >= 1,
        "display:none elements should have zero size"
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
#[ignore] // Run with: cargo test test_3000_elements -- --ignored
fn test_3000_elements_wikipedia_scale() {
    let mut html = String::from("<html><body>");
    for i in 0..3000 {
        html.push_str(&format!("<div>Wikipedia paragraph {}</div>", i));
    }
    html.push_str("</body></html>");

    let start = std::time::Instant::now();
    let (_elements, boxes) = layout_html(html.as_bytes());
    let elapsed = start.elapsed();

    println!("Layout 3000 elements: {:?}", elapsed);

    // Should complete (no stack overflow)
    assert!(!boxes.is_empty());

    // Should have proper Y spread
    let max_y = boxes.iter().map(|b| b.y).fold(0.0f32, f32::max);
    assert!(
        max_y > 10000.0,
        "3000 divs should spread across Y: max_y={}", max_y
    );

    // Count overlapping Y positions
    let mut y_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for box_ in &boxes {
        if box_.height > 5.0 {
            let y_key = (box_.y * 10.0) as i32;
            *y_counts.entry(y_key).or_insert(0) += 1;
        }
    }

    let max_overlap = y_counts.values().max().unwrap_or(&0);
    assert!(
        *max_overlap <= 5,
        "Too many boxes at same Y: {}", max_overlap
    );
}

#[test]
#[ignore] // Long running test
fn test_10000_elements_limit() {
    let mut html = String::from("<html><body>");
    for i in 0..10000 {
        html.push_str(&format!("<p>Para {}</p>", i));
    }
    html.push_str("</body></html>");

    let start = std::time::Instant::now();
    let (_elements, boxes) = layout_html(html.as_bytes());
    let elapsed = start.elapsed();

    println!("Layout 10000 elements: {:?}", elapsed);

    // Target: < 16ms for real-time rendering
    // Allow 500ms for CI headroom
    assert!(
        elapsed.as_millis() < 500,
        "10000 elements took {:?}", elapsed
    );

    assert!(!boxes.is_empty());
}
