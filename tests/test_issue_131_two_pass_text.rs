//! Issue #131: O(1) Two-Pass Text Line Layout
//!
//! Tests for two-pass text layout that eliminates O(word_length) lookback.

use metal::*;
use std::time::Instant;

// Import from the main crate
use rust_experiment::gpu_os::document::{
    Element,
    ComputedStyle,
    LayoutBox, Viewport,
    GpuPaintEngine, PaintVertex, FLAG_TEXT,
    LineInfo, TextLineDataHeader, MAX_LINES_PER_ELEMENT,
};

const ELEM_TEXT: u32 = 100;

/// Helper to create a text element
fn create_text_element(text_start: u32, text_length: u32) -> Element {
    Element {
        element_type: ELEM_TEXT,
        parent: -1,
        first_child: -1,
        next_sibling: -1,
        text_start,
        text_length,
        token_index: 0,
        prev_sibling: -1,
    }
}

/// Helper to create default style for text
fn create_text_style(font_size: f32, line_height: f32) -> ComputedStyle {
    ComputedStyle {
        display: 2,  // DISPLAY_INLINE
        color: [0.0, 0.0, 0.0, 1.0],  // Black text
        font_size,
        line_height,
        opacity: 1.0,
        ..Default::default()
    }
}

/// Helper to create layout box
fn create_layout_box(width: f32, height: f32) -> LayoutBox {
    LayoutBox {
        x: 0.0,
        y: 0.0,
        width,
        height,
        content_x: 0.0,
        content_y: 0.0,
        content_width: width,
        content_height: height,
        ..Default::default()
    }
}

#[test]
fn test_gpu_struct_alignment() {
    // Verify structs are GPU-friendly
    assert_eq!(std::mem::size_of::<LineInfo>(), 16, "LineInfo must be 16 bytes");
    assert_eq!(std::mem::align_of::<LineInfo>(), 4, "LineInfo must be 4-byte aligned");
    assert_eq!(std::mem::size_of::<TextLineDataHeader>(), 16, "TextLineDataHeader must be 16 bytes");

    println!("GPU struct alignment test:");
    println!("  LineInfo: {} bytes, {}-byte aligned",
        std::mem::size_of::<LineInfo>(),
        std::mem::align_of::<LineInfo>());
    println!("  TextLineDataHeader: {} bytes",
        std::mem::size_of::<TextLineDataHeader>());
    println!("  MAX_LINES_PER_ELEMENT: {}", MAX_LINES_PER_ELEMENT);
}

#[test]
fn test_memory_overhead() {
    let line_info_size = std::mem::size_of::<LineInfo>();
    let header_size = std::mem::size_of::<TextLineDataHeader>();
    let per_element_size = header_size + MAX_LINES_PER_ELEMENT * line_info_size;

    println!("\nMemory overhead test:");
    println!("  LineInfo size: {} bytes", line_info_size);
    println!("  Header size: {} bytes", header_size);
    println!("  Max lines per element: {}", MAX_LINES_PER_ELEMENT);
    println!("  Per-element overhead: {} bytes ({:.1} KB)",
        per_element_size, per_element_size as f64 / 1024.0);

    // Should be reasonable overhead
    assert!(per_element_size < 2048, "Per-element overhead should be < 2KB");

    println!("\n  For 1000 text elements: {:.1} MB",
        1000.0 * per_element_size as f64 / (1024.0 * 1024.0));
}

#[test]
fn test_two_pass_short_text() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // Enable two-pass text (should be default)
    assert!(paint.use_two_pass_text, "Two-pass text should be enabled by default");

    let text = b"Hello";
    let elements = vec![create_text_element(0, text.len() as u32)];
    let styles = vec![create_text_style(16.0, 1.2)];
    let boxes = vec![create_layout_box(200.0, 50.0)];
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    // First test with single-pass (should work)
    paint.use_two_pass_text = false;
    let single_pass_vertices = paint.paint(&elements, &boxes, &styles, text, viewport);
    let single_pass_text: Vec<_> = single_pass_vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();
    println!("\nShort text test (SINGLE-PASS):");
    println!("  Total vertices: {}", single_pass_vertices.len());
    println!("  Text vertices: {}", single_pass_text.len());

    // Now test with two-pass
    paint.use_two_pass_text = true;
    let vertices = paint.paint(&elements, &boxes, &styles, text, viewport);
    let text_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();

    println!("\nShort text test (TWO-PASS):");
    println!("  Text: \"{}\" ({} chars)", String::from_utf8_lossy(text), text.len());
    println!("  Total vertices: {}", vertices.len());
    println!("  Text vertices: {}", text_vertices.len());

    // Print first few vertices for debug
    for (i, v) in vertices.iter().take(10).enumerate() {
        println!("  V{}: pos=({:.4}, {:.4}), flags={}, color=[{:.1},{:.1},{:.1},{:.1}]",
            i, v.position[0], v.position[1], v.flags, v.color[0], v.color[1], v.color[2], v.color[3]);
    }

    // Should have text vertices (5 chars = 5 * 4 = 20 vertices)
    assert!(text_vertices.len() >= 20, "Expected >= 20 text vertices, got {}", text_vertices.len());
}

#[test]
fn test_two_pass_text_wrapping() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // Text that should wrap in a narrow container
    let text = b"Hello world this is a test of text wrapping";
    let elements = vec![create_text_element(0, text.len() as u32)];
    let styles = vec![create_text_style(16.0, 1.2)];
    // Narrow container forces wrapping
    let boxes = vec![create_layout_box(100.0, 200.0)];
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = paint.paint(&elements, &boxes, &styles, text, viewport);

    let text_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();

    println!("\nText wrapping test:");
    println!("  Text: \"{}\"", String::from_utf8_lossy(text));
    println!("  Container width: 100px");
    println!("  Text vertices: {}", text_vertices.len());

    // Check that vertices exist and are positioned correctly
    assert!(!text_vertices.is_empty(), "Should have text vertices");

    // With wrapping, we should see different Y positions in the vertices
    let unique_y: std::collections::HashSet<i32> = text_vertices.iter()
        .map(|v| (v.position[1] * 1000.0) as i32)  // Convert to int for comparison
        .collect();

    println!("  Unique Y positions: {}", unique_y.len());
    // With wrapping, we should have multiple Y positions (multiple lines)
    assert!(unique_y.len() >= 2, "Should have multiple Y positions (lines) when wrapping");
}

#[test]
fn test_two_pass_vs_single_pass_correctness() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    let text = b"Hello world";
    let elements = vec![create_text_element(0, text.len() as u32)];
    let styles = vec![create_text_style(16.0, 1.2)];
    let boxes = vec![create_layout_box(200.0, 50.0)];
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    // Test with two-pass enabled
    paint.use_two_pass_text = true;
    let two_pass_vertices = paint.paint(&elements, &boxes, &styles, text, viewport);

    // Test with single-pass (old method)
    paint.use_two_pass_text = false;
    let single_pass_vertices = paint.paint(&elements, &boxes, &styles, text, viewport);

    let two_pass_text: Vec<_> = two_pass_vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();
    let single_pass_text: Vec<_> = single_pass_vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();

    println!("\nTwo-pass vs single-pass correctness:");
    println!("  Two-pass text vertices: {}", two_pass_text.len());
    println!("  Single-pass text vertices: {}", single_pass_text.len());

    assert_eq!(two_pass_text.len(), single_pass_text.len(),
        "Both methods should produce same vertex count");

    // Compare first few vertices (positions should be similar)
    for i in 0..two_pass_text.len().min(8) {
        let tp = two_pass_text[i];
        let sp = single_pass_text[i];
        let diff_x = (tp.position[0] - sp.position[0]).abs();
        let diff_y = (tp.position[1] - sp.position[1]).abs();

        if i < 4 {
            println!("  Vertex {}: two_pass=({:.4}, {:.4}), single=({:.4}, {:.4})",
                i, tp.position[0], tp.position[1], sp.position[0], sp.position[1]);
        }

        // Allow small floating point differences
        assert!(diff_x < 0.01 && diff_y < 0.01,
            "Vertex {} positions differ too much: two_pass=({}, {}), single=({}, {})",
            i, tp.position[0], tp.position[1], sp.position[0], sp.position[1]);
    }

    // Re-enable two-pass
    paint.use_two_pass_text = true;
}

#[test]
fn test_two_pass_long_text_performance() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // 10K characters with spaces (will wrap many times)
    let text: Vec<u8> = (0..10000).map(|i| {
        if i % 10 == 9 { b' ' } else { b'a' }
    }).collect();

    let elements = vec![create_text_element(0, text.len() as u32)];
    let styles = vec![create_text_style(14.0, 1.2)];
    // Narrow container forces many wraps
    let boxes = vec![create_layout_box(200.0, 5000.0)];
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let iterations = 10;

    // Warmup
    let _ = paint.paint(&elements, &boxes, &styles, &text, viewport);

    // Benchmark two-pass
    paint.use_two_pass_text = true;
    let two_pass_start = Instant::now();
    for _ in 0..iterations {
        let _ = paint.paint(&elements, &boxes, &styles, &text, viewport);
    }
    let two_pass_time = two_pass_start.elapsed();

    // Benchmark single-pass
    paint.use_two_pass_text = false;
    let single_pass_start = Instant::now();
    for _ in 0..iterations {
        let _ = paint.paint(&elements, &boxes, &styles, &text, viewport);
    }
    let single_pass_time = single_pass_start.elapsed();

    let speedup = single_pass_time.as_secs_f64() / two_pass_time.as_secs_f64();

    println!("\n=== Performance: 10K chars, narrow container ===");
    println!("Two-pass:    {:.2}ms ({} iterations)",
        two_pass_time.as_secs_f64() * 1000.0, iterations);
    println!("Single-pass: {:.2}ms ({} iterations)",
        single_pass_time.as_secs_f64() * 1000.0, iterations);
    println!("Ratio:       {:.2}x", speedup);

    // Note: On GPU, the benefit is eliminating SIMD divergence,
    // not raw throughput. The speedup may vary.
    println!("\nNote: GPU benefit is from eliminating O(word_length) lookback");
    println!("      and SIMD divergence, which may not show in total time");
    println!("      but improves worst-case performance consistency.");

    // Re-enable two-pass
    paint.use_two_pass_text = true;
}

#[test]
fn test_two_pass_many_elements() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // Create 100 text elements, each with text that may wrap
    let base_text = b"word1 word2 word3 word4 word5 ";
    let mut full_text = Vec::new();
    let mut elements = Vec::new();
    let mut styles = Vec::new();
    let mut boxes = Vec::new();

    for i in 0..100 {
        let start = full_text.len() as u32;
        full_text.extend_from_slice(base_text);

        elements.push(create_text_element(start, base_text.len() as u32));
        styles.push(create_text_style(14.0, 1.2));
        boxes.push(LayoutBox {
            x: 0.0,
            y: (i as f32) * 30.0,
            width: 100.0,
            height: 25.0,
            content_x: 0.0,
            content_y: (i as f32) * 30.0,
            content_width: 100.0,
            content_height: 25.0,
            ..Default::default()
        });
    }

    let viewport = Viewport { width: 800.0, height: 3000.0, _padding: [0.0; 2] };

    // Warmup
    let _ = paint.paint(&elements, &boxes, &styles, &full_text, viewport);

    let start = Instant::now();
    let vertices = paint.paint(&elements, &boxes, &styles, &full_text, viewport);
    let elapsed = start.elapsed();

    let text_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();

    println!("\nMany elements test:");
    println!("  Elements: {}", elements.len());
    println!("  Total text: {} chars", full_text.len());
    println!("  Text vertices: {}", text_vertices.len());
    println!("  Time: {:?}", elapsed);

    assert!(!text_vertices.is_empty(), "Should have text vertices");
    assert!(elapsed.as_millis() < 100, "100 elements should paint in < 100ms");
}

#[test]
fn test_two_pass_newlines() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // Text with explicit newlines
    let text = b"Line one\nLine two\nLine three";
    let elements = vec![create_text_element(0, text.len() as u32)];
    let styles = vec![create_text_style(16.0, 1.2)];
    let boxes = vec![create_layout_box(500.0, 100.0)];  // Wide container, no forced wrap
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = paint.paint(&elements, &boxes, &styles, text, viewport);
    let text_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();

    // Count unique Y positions (should be 3 lines)
    let unique_y: std::collections::HashSet<i32> = text_vertices.iter()
        .filter(|v| v.color[3] > 0.0)  // Only visible characters
        .map(|v| (v.position[1] * 1000.0) as i32)
        .collect();

    println!("\nNewlines test:");
    println!("  Text: \"{}\"", String::from_utf8_lossy(text).replace('\n', "\\n"));
    println!("  Unique Y positions: {}", unique_y.len());

    // Should have exactly 3 different Y positions for 3 lines
    assert!(unique_y.len() >= 3, "Should have at least 3 Y positions for 3 lines");
}

#[test]
fn test_two_pass_empty_text() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // Empty text element
    let elements = vec![create_text_element(0, 0)];
    let styles = vec![create_text_style(16.0, 1.2)];
    let boxes = vec![create_layout_box(200.0, 50.0)];
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    // Should not crash
    let vertices = paint.paint(&elements, &boxes, &styles, &[], viewport);

    println!("\nEmpty text test:");
    println!("  Total vertices: {}", vertices.len());
    // Empty text should produce no text vertices (or very few)
}

#[test]
fn test_two_pass_long_word() {
    let device = Device::system_default().expect("No Metal device");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    // Very long word that can't fit on one line
    let text = "x".repeat(100);
    let text_bytes = text.as_bytes();
    let elements = vec![create_text_element(0, text_bytes.len() as u32)];
    let styles = vec![create_text_style(16.0, 1.2)];
    // Very narrow container
    let boxes = vec![create_layout_box(50.0, 500.0)];
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let vertices = paint.paint(&elements, &boxes, &styles, text_bytes, viewport);
    let text_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();

    // Should have vertices for all 100 characters
    assert!(text_vertices.len() >= 100 * 4, "Should have vertices for all characters");

    // Count lines (unique Y positions)
    let unique_y: std::collections::HashSet<i32> = text_vertices.iter()
        .map(|v| (v.position[1] * 1000.0) as i32)
        .collect();

    println!("\nLong word test:");
    println!("  Word length: {} chars", text.len());
    println!("  Container width: 50px");
    println!("  Lines detected: {}", unique_y.len());

    // Long word must break across multiple lines
    assert!(unique_y.len() >= 2, "Long word should wrap to multiple lines");
}
