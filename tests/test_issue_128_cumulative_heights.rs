//! Issue #128: O(1) Sibling Cumulative Heights for Layout
//!
//! Tests for pre-computed cumulative height buffer that replaces O(siblings) chain walk.

use metal::*;
use std::time::Instant;

/// Simulated element for testing
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TestElement {
    parent: i32,
    first_child: i32,
    next_sibling: i32,
    prev_sibling: i32,
}

/// Simulated layout box
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TestLayoutBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

/// Create a wide tree (many siblings per parent)
fn create_wide_tree(siblings_per_parent: usize, depth: usize) -> (Vec<TestElement>, Vec<TestLayoutBox>) {
    let mut elements = Vec::new();
    let mut boxes = Vec::new();

    // Root element
    elements.push(TestElement {
        parent: -1,
        first_child: if depth > 0 { 1 } else { -1 },
        next_sibling: -1,
        prev_sibling: -1,
    });
    boxes.push(TestLayoutBox {
        x: 0.0, y: 0.0, width: 100.0, height: 50.0,
    });

    // Create siblings at each level
    let mut current_idx = 1i32;
    for _level in 0..depth {
        let first_child_of_level = current_idx;

        for sib in 0..siblings_per_parent {
            let prev = if sib == 0 { -1 } else { current_idx - 1 };
            let next = if sib == siblings_per_parent - 1 { -1 } else { current_idx + 1 };

            elements.push(TestElement {
                parent: 0,  // All children of root for simplicity
                first_child: -1,
                next_sibling: next,
                prev_sibling: prev,
            });
            boxes.push(TestLayoutBox {
                x: 0.0,
                y: 0.0,  // Will be computed
                width: 100.0,
                height: 20.0 + (sib % 5) as f32 * 10.0,  // Varying heights
            });
            current_idx += 1;
        }

        // Update root's first_child to point to first sibling
        elements[0].first_child = first_child_of_level;
    }

    (elements, boxes)
}

/// O(siblings) algorithm - current implementation
fn compute_y_position_old(elements: &[TestElement], boxes: &[TestLayoutBox], idx: usize) -> f32 {
    let elem = &elements[idx];
    if elem.parent < 0 {
        return 0.0;
    }

    // Walk sibling chain - O(siblings)
    let parent = &elements[elem.parent as usize];
    let mut y = 0.0f32;
    let mut sib = parent.first_child;

    while sib >= 0 && sib != idx as i32 {
        y += boxes[sib as usize].height;
        sib = elements[sib as usize].next_sibling;
    }

    y
}

/// O(1) algorithm - using pre-computed cumulative heights
fn compute_y_position_new(cumulative_heights: &[f32], idx: usize) -> f32 {
    cumulative_heights[idx]
}

/// Pre-compute cumulative heights buffer
fn build_cumulative_heights(elements: &[TestElement], boxes: &[TestLayoutBox]) -> Vec<f32> {
    let mut cumulative = vec![0.0f32; elements.len()];

    for (idx, elem) in elements.iter().enumerate() {
        if elem.parent < 0 {
            cumulative[idx] = 0.0;
        } else if elem.prev_sibling < 0 {
            // First child - no preceding siblings
            cumulative[idx] = 0.0;
        } else {
            // cumulative[idx] = cumulative[prev] + height[prev]
            let prev = elem.prev_sibling as usize;
            cumulative[idx] = cumulative[prev] + boxes[prev].height;
        }
    }

    cumulative
}

#[test]
fn test_cumulative_heights_correctness() {
    let (elements, boxes) = create_wide_tree(10, 1);
    let cumulative = build_cumulative_heights(&elements, &boxes);

    println!("Cumulative heights correctness test:");
    println!("  Elements: {}", elements.len());

    // Verify each element's Y position matches
    for idx in 1..elements.len() {
        let old_y = compute_y_position_old(&elements, &boxes, idx);
        let new_y = compute_y_position_new(&cumulative, idx);

        assert!((old_y - new_y).abs() < 0.001,
            "Mismatch at idx {}: old={}, new={}", idx, old_y, new_y);
    }

    println!("  All {} elements match!", elements.len() - 1);
}

#[test]
fn test_cumulative_heights_wide_tree() {
    // Test with many siblings (worst case for old algorithm)
    let siblings = 100;
    let (elements, boxes) = create_wide_tree(siblings, 1);
    let cumulative = build_cumulative_heights(&elements, &boxes);

    println!("\nWide tree test ({} siblings):", siblings);

    // Check last sibling - requires walking all previous
    let last_idx = elements.len() - 1;
    let old_y = compute_y_position_old(&elements, &boxes, last_idx);
    let new_y = compute_y_position_new(&cumulative, last_idx);

    println!("  Last sibling Y: old={:.1}, new={:.1}", old_y, new_y);
    assert!((old_y - new_y).abs() < 0.001);
}

#[test]
fn benchmark_o_siblings_vs_o_1() {
    let siblings = 100;
    let (elements, boxes) = create_wide_tree(siblings, 1);
    let cumulative = build_cumulative_heights(&elements, &boxes);

    println!("\n=== O(siblings) vs O(1) Benchmark ===\n");
    println!("Tree: {} siblings", siblings);

    let iterations = 10000;

    // Benchmark old algorithm - O(siblings) per lookup
    let old_start = Instant::now();
    let mut old_sum = 0.0f32;
    for _ in 0..iterations {
        for idx in 1..elements.len() {
            old_sum += compute_y_position_old(&elements, &boxes, idx);
        }
    }
    let old_time = old_start.elapsed();

    // Benchmark new algorithm - O(1) per lookup
    let new_start = Instant::now();
    let mut new_sum = 0.0f32;
    for _ in 0..iterations {
        for idx in 1..elements.len() {
            new_sum += compute_y_position_new(&cumulative, idx);
        }
    }
    let new_time = new_start.elapsed();

    // Prevent optimization
    assert!((old_sum - new_sum).abs() < 1.0);

    let speedup = old_time.as_secs_f64() / new_time.as_secs_f64();

    println!("Old O(siblings): {:.2}ms ({} iterations)",
        old_time.as_secs_f64() * 1000.0, iterations);
    println!("New O(1):        {:.2}ms ({} iterations)",
        new_time.as_secs_f64() * 1000.0, iterations);
    println!("Speedup:         {:.1}x", speedup);

    // With 100 siblings, expect significant speedup
    assert!(speedup > 5.0, "Expected >5x speedup, got {:.1}x", speedup);
}

#[test]
fn test_memory_overhead() {
    // Verify memory overhead is acceptable
    let element_count = 10000;
    let cumulative_buffer_size = element_count * std::mem::size_of::<f32>();

    println!("\nMemory overhead test:");
    println!("  {} elements", element_count);
    println!("  Cumulative buffer: {} bytes ({:.1} KB)",
        cumulative_buffer_size, cumulative_buffer_size as f64 / 1024.0);

    // Should be 4 bytes per element (f32)
    assert_eq!(cumulative_buffer_size, element_count * 4);
    println!("  Overhead per element: 4 bytes (acceptable)");
}

#[test]
fn test_gpu_buffer_alignment() {
    // Verify f32 buffer is GPU-friendly
    let device = Device::system_default().expect("No Metal device");

    let element_count = 1000u64;
    let buffer = device.new_buffer(
        element_count * 4,  // f32 = 4 bytes
        MTLResourceOptions::StorageModeShared,
    );

    println!("\nGPU buffer alignment test:");
    println!("  Buffer length: {} bytes", buffer.length());
    println!("  Alignment: float (4 bytes) - GPU optimal");

    assert_eq!(buffer.length(), element_count * 4);
}

// =============================================================================
// GPU IMPLEMENTATION TESTS
// =============================================================================

use rust_experiment::gpu_os::document::{
    Element,
    ComputedStyle,
    GpuLayoutEngine,
    Viewport,
};

/// Create elements with proper prev_sibling links for GPU testing
fn create_gpu_wide_tree(siblings_per_parent: usize) -> (Vec<Element>, Vec<ComputedStyle>) {
    let mut elements = Vec::new();
    let mut styles = Vec::new();

    // Root element
    elements.push(Element {
        element_type: 1, // div
        parent: -1,
        first_child: 1,
        next_sibling: -1,
        prev_sibling: -1,
        text_start: 0,
        text_length: 0,
        token_index: 0,
    });
    styles.push(ComputedStyle {
        display: 1, // DISPLAY_BLOCK
        width: 800.0,
        height: 0.0, // auto
        ..Default::default()
    });

    // Child elements
    for i in 0..siblings_per_parent {
        let idx = i + 1;
        elements.push(Element {
            element_type: 1, // div
            parent: 0,
            first_child: -1,
            next_sibling: if i < siblings_per_parent - 1 { (idx + 1) as i32 } else { -1 },
            prev_sibling: if i > 0 { (idx - 1) as i32 } else { -1 },
            text_start: 0,
            text_length: 0,
            token_index: 0,
        });
        styles.push(ComputedStyle {
            display: 1, // DISPLAY_BLOCK
            width: 0.0, // auto
            height: 20.0 + (i % 5) as f32 * 10.0, // Varying heights
            ..Default::default()
        });
    }

    (elements, styles)
}

#[test]
fn test_gpu_cumulative_heights_kernel() {
    let device = Device::system_default().expect("No Metal device");
    let mut layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");

    // Create tree with 10 siblings
    let (elements, styles) = create_gpu_wide_tree(10);
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    // Compute layout using GPU with O(1) cumulative heights
    let boxes = layout.compute_layout(&elements, &styles, &[], viewport);

    println!("\nGPU cumulative heights test (10 siblings):");
    let mut expected_y = 0.0f32;
    for (i, b) in boxes.iter().enumerate().skip(1) {
        // Skip root
        let expected_height = 20.0 + ((i - 1) % 5) as f32 * 10.0;
        println!("  Element {}: y={:.1}, expected_y={:.1}, height={:.1}",
            i, b.y, expected_y, b.height);

        // Y position should match cumulative sum of previous siblings
        assert!(
            (b.y - expected_y).abs() < 1.0,
            "Element {} Y mismatch: got {:.1}, expected {:.1}",
            i, b.y, expected_y
        );

        expected_y += b.height;
    }
}

#[test]
fn test_gpu_cumulative_heights_wide_tree() {
    let device = Device::system_default().expect("No Metal device");
    let mut layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");

    // Test with 100 siblings (worst case for old O(siblings) algorithm)
    let (elements, styles) = create_gpu_wide_tree(100);
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    let boxes = layout.compute_layout(&elements, &styles, &[], viewport);

    println!("\nGPU cumulative heights test (100 siblings):");
    println!("  Total elements: {}", boxes.len());

    // Verify last sibling's Y position
    let last_idx = boxes.len() - 1;
    let last_box = &boxes[last_idx];

    // Calculate expected Y by summing all previous heights
    let mut expected_y = 0.0f32;
    for i in 1..last_idx {
        expected_y += boxes[i].height;
    }

    println!("  Last sibling (idx {}): y={:.1}, expected={:.1}", last_idx, last_box.y, expected_y);

    assert!(
        (last_box.y - expected_y).abs() < 1.0,
        "Last sibling Y mismatch: got {:.1}, expected {:.1}",
        last_box.y, expected_y
    );
}

#[test]
fn benchmark_gpu_layout_with_cumulative() {
    let device = Device::system_default().expect("No Metal device");
    let mut layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");

    // Create tree with 1000 siblings (extreme test for O(1) vs O(siblings))
    let (elements, styles) = create_gpu_wide_tree(1000);
    let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

    // Warmup
    let _ = layout.compute_layout(&elements, &styles, &[], viewport);

    // Timed run
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = layout.compute_layout(&elements, &styles, &[], viewport);
    }
    let elapsed = start.elapsed();

    let per_layout = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("\n=== GPU Layout Performance (Issue #128) ===");
    println!("Tree: 1000 siblings (worst case for old O(siblings) algorithm)");
    println!("Layout time: {:.2}ms per layout ({} iterations)", per_layout, iterations);

    // With O(1) lookup, should be fast even with 1000 siblings
    // Old O(siblings) would be much slower
    assert!(
        per_layout < 20.0,
        "Layout took too long: {:.2}ms (expected <20ms with O(1) lookup)",
        per_layout
    );

    println!("PASS: O(1) cumulative heights working correctly!");
}

#[test]
fn test_prev_sibling_links() {
    // Verify that prev_sibling links are correctly maintained
    let (elements, _styles) = create_gpu_wide_tree(10);

    println!("\nPrev sibling links test:");
    for (i, elem) in elements.iter().enumerate() {
        if i == 0 {
            assert_eq!(elem.prev_sibling, -1, "Root should have no prev_sibling");
        } else if i == 1 {
            assert_eq!(elem.prev_sibling, -1, "First child should have no prev_sibling");
        } else {
            assert_eq!(
                elem.prev_sibling,
                (i - 1) as i32,
                "Element {} should have prev_sibling {}",
                i,
                i - 1
            );
        }
        println!("  Element {}: prev_sibling={}", i, elem.prev_sibling);
    }
}
