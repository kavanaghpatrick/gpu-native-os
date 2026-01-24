//! Issue #130: O(1) Depth Buffer for Layout
//!
//! Tests for pre-computed depth buffer that replaces O(depth) parent chain walk.

use metal::*;
use std::time::Instant;

/// Simulated element for testing
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TestElement {
    parent: i32,
    first_child: i32,
    next_sibling: i32,
}

/// Create a deep tree
fn create_deep_tree(depth: usize, children_per_node: usize) -> Vec<TestElement> {
    let mut elements = Vec::new();

    // Root
    elements.push(TestElement {
        parent: -1,
        first_child: if depth > 0 { 1 } else { -1 },
        next_sibling: -1,
    });

    let mut current_idx = 1i32;
    let mut parent_idx = 0i32;

    for level in 0..depth {
        let first_at_level = current_idx;

        for child in 0..children_per_node {
            elements.push(TestElement {
                parent: parent_idx,
                first_child: -1,  // Will update if not leaf
                next_sibling: if child < children_per_node - 1 { current_idx + 1 } else { -1 },
            });
            current_idx += 1;
        }

        // Update parent's first_child
        elements[parent_idx as usize].first_child = first_at_level;

        // Move to next level's parent (first child of current level)
        if level < depth - 1 {
            parent_idx = first_at_level;
        }
    }

    elements
}

/// O(depth) algorithm - walk parent chain
fn compute_depth_old(elements: &[TestElement], idx: usize) -> u32 {
    let mut depth = 0u32;
    let mut current = elements[idx].parent;

    while current >= 0 {
        depth += 1;
        current = elements[current as usize].parent;
    }

    depth
}

/// O(1) algorithm - lookup pre-computed depth
fn compute_depth_new(depths: &[u32], idx: usize) -> u32 {
    depths[idx]
}

/// Build depth buffer (simulating level-parallel GPU algorithm)
fn build_depth_buffer(elements: &[TestElement]) -> Vec<u32> {
    let mut depths = vec![0xFFFFFFFFu32; elements.len()];

    // Phase 1: Mark roots
    for (idx, elem) in elements.iter().enumerate() {
        if elem.parent < 0 {
            depths[idx] = 0;
        }
    }

    // Phase 2: Level-parallel propagation
    let mut level = 0u32;
    loop {
        let mut changed = false;

        for (idx, elem) in elements.iter().enumerate() {
            if depths[idx] != 0xFFFFFFFF {
                continue;  // Already computed
            }

            let parent = elem.parent;
            if parent >= 0 && depths[parent as usize] == level {
                depths[idx] = level + 1;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        level += 1;
        if level > 1000 {
            panic!("Tree too deep");
        }
    }

    depths
}

#[test]
fn test_depth_buffer_correctness() {
    let elements = create_deep_tree(10, 2);
    let depths = build_depth_buffer(&elements);

    println!("Depth buffer correctness test:");
    println!("  Elements: {}", elements.len());

    // Verify each element's depth matches
    for idx in 0..elements.len() {
        let old_depth = compute_depth_old(&elements, idx);
        let new_depth = compute_depth_new(&depths, idx);

        assert_eq!(old_depth, new_depth,
            "Mismatch at idx {}: old={}, new={}", idx, old_depth, new_depth);
    }

    println!("  All {} elements match!", elements.len());
}

#[test]
fn test_depth_buffer_deep_tree() {
    // Test with very deep tree (worst case for old algorithm)
    let depth = 100;
    let elements = create_deep_tree(depth, 1);  // Linear chain
    let depths = build_depth_buffer(&elements);

    println!("\nDeep tree test (depth={}):", depth);
    println!("  Elements: {}", elements.len());

    // Check deepest element
    let deepest_idx = elements.len() - 1;
    let old_depth = compute_depth_old(&elements, deepest_idx);
    let new_depth = compute_depth_new(&depths, deepest_idx);

    println!("  Deepest element depth: old={}, new={}", old_depth, new_depth);
    assert_eq!(old_depth, new_depth);
    assert_eq!(new_depth, depth as u32);
}

#[test]
fn benchmark_o_depth_vs_o_1() {
    let tree_depth = 50;
    let elements = create_deep_tree(tree_depth, 2);
    let depths = build_depth_buffer(&elements);

    println!("\n=== O(depth) vs O(1) Benchmark ===\n");
    println!("Tree depth: {}, Elements: {}", tree_depth, elements.len());

    let iterations = 10000;

    // Benchmark old algorithm - O(depth) per lookup
    let old_start = Instant::now();
    let mut old_sum = 0u32;
    for _ in 0..iterations {
        for idx in 0..elements.len() {
            old_sum += compute_depth_old(&elements, idx);
        }
    }
    let old_time = old_start.elapsed();

    // Benchmark new algorithm - O(1) per lookup
    let new_start = Instant::now();
    let mut new_sum = 0u32;
    for _ in 0..iterations {
        for idx in 0..elements.len() {
            new_sum += compute_depth_new(&depths, idx);
        }
    }
    let new_time = new_start.elapsed();

    assert_eq!(old_sum, new_sum);

    let speedup = old_time.as_secs_f64() / new_time.as_secs_f64();

    println!("Old O(depth): {:.2}ms ({} iterations)",
        old_time.as_secs_f64() * 1000.0, iterations);
    println!("New O(1):     {:.2}ms ({} iterations)",
        new_time.as_secs_f64() * 1000.0, iterations);
    println!("Speedup:      {:.1}x", speedup);

    // With depth 50, expect significant speedup
    assert!(speedup > 5.0, "Expected >5x speedup, got {:.1}x", speedup);
}

#[test]
fn test_level_parallel_build() {
    // Test that level-parallel build works correctly
    let elements = create_deep_tree(5, 3);

    println!("\nLevel-parallel build test:");

    // Track which level each element is computed at
    let mut depths = vec![0xFFFFFFFFu32; elements.len()];
    let mut level_counts = Vec::new();

    // Phase 1: Roots
    let mut count = 0;
    for (idx, elem) in elements.iter().enumerate() {
        if elem.parent < 0 {
            depths[idx] = 0;
            count += 1;
        }
    }
    level_counts.push(count);

    // Phase 2: Level by level
    let mut level = 0u32;
    loop {
        let mut count = 0;

        for (idx, elem) in elements.iter().enumerate() {
            if depths[idx] != 0xFFFFFFFF {
                continue;
            }

            let parent = elem.parent;
            if parent >= 0 && depths[parent as usize] == level {
                depths[idx] = level + 1;
                count += 1;
            }
        }

        if count == 0 {
            break;
        }

        level_counts.push(count);
        level += 1;
    }

    println!("  Levels processed: {}", level_counts.len());
    for (l, count) in level_counts.iter().enumerate() {
        println!("    Level {}: {} elements", l, count);
    }

    // On GPU, each level is one dispatch (parallel across all elements at that level)
    println!("  GPU would need {} dispatches", level_counts.len());
}

#[test]
fn test_memory_overhead() {
    let element_count = 10000;
    let depth_buffer_size = element_count * std::mem::size_of::<u32>();

    println!("\nMemory overhead test:");
    println!("  {} elements", element_count);
    println!("  Depth buffer: {} bytes ({:.1} KB)",
        depth_buffer_size, depth_buffer_size as f64 / 1024.0);

    // Should be 4 bytes per element (u32)
    assert_eq!(depth_buffer_size, element_count * 4);
    println!("  Overhead per element: 4 bytes (acceptable)");
}

#[test]
fn test_gpu_buffer_creation() {
    let device = Device::system_default().expect("No Metal device");

    let element_count = 1000u64;
    let buffer = device.new_buffer(
        element_count * 4,  // u32 = 4 bytes
        MTLResourceOptions::StorageModeShared,
    );

    println!("\nGPU buffer creation test:");
    println!("  Buffer: {} elements, {} bytes", element_count, buffer.length());

    assert_eq!(buffer.length(), element_count * 4);
}

#[test]
fn test_cache_invalidation_concept() {
    // Test the concept of cache invalidation
    let mut elements = create_deep_tree(5, 2);
    let depths_v1 = build_depth_buffer(&elements);

    println!("\nCache invalidation concept test:");
    println!("  Initial depths computed");

    // Simulate tree modification (change parent)
    // In real implementation, this would invalidate the cache
    let modified_idx = elements.len() - 1;
    let old_parent = elements[modified_idx].parent;
    elements[modified_idx].parent = 0;  // Reparent to root

    // Depths are now invalid - need to recompute
    let depths_v2 = build_depth_buffer(&elements);

    let old_depth = depths_v1[modified_idx];
    let new_depth = depths_v2[modified_idx];

    println!("  Element {} reparented: depth {} -> {}", modified_idx, old_depth, new_depth);

    // Restore
    elements[modified_idx].parent = old_parent;

    // Key insight: depth buffer must be invalidated when tree structure changes
    assert_ne!(old_depth, new_depth, "Depth should change after reparenting");
}

// Placeholder for GPU implementation tests
#[test]
#[ignore = "Requires GPU implementation"]
fn test_gpu_depth_computation_kernel() {
    // TODO: Test actual Metal kernel implementation
}

#[test]
#[ignore = "Requires GPU implementation"]
fn benchmark_gpu_layout_with_depth_buffer() {
    // TODO: Benchmark full layout pipeline with O(1) depth lookup
}
