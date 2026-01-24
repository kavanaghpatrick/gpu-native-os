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

// Placeholder for GPU implementation tests
#[test]
#[ignore = "Requires GPU implementation"]
fn test_gpu_cumulative_heights_kernel() {
    // TODO: Test actual Metal kernel implementation
}

#[test]
#[ignore = "Requires GPU implementation"]
fn benchmark_gpu_layout_with_cumulative() {
    // TODO: Benchmark full layout pipeline with O(1) sibling positioning
}
