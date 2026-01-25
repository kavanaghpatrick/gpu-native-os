# Issue #141: GPU-Driven Depth Propagation - Eliminate CPU-GPU Ping-Pong Loop

## Problem Statement

The depth propagation algorithm in `src/gpu_os/document/layout.rs` (lines 432-461) uses a CPU-driven loop that ping-pongs between CPU and GPU for each tree level:

```rust
let mut level = 0u32;
loop {
    // CPU writes level to buffer
    unsafe { *(self.current_level_buffer.contents() as *mut u32) = level; }
    unsafe { *(self.changed_buffer.contents() as *mut u32) = 0; }

    // GPU executes one level
    let cmd = self.command_queue.new_command_buffer();
    // ... dispatch propagate_depths kernel ...
    cmd.commit();
    cmd.wait_until_completed();  // BLOCKS HERE

    // CPU reads result
    let changed = unsafe { *(self.changed_buffer.contents() as *const u32) };
    if changed == 0 { break; }
    level += 1;
}
```

**Impact:** For a tree with 30 levels of depth, this creates:
- 30× command buffer allocations
- 30× `wait_until_completed()` calls (~500-1000µs each)
- 30× CPU-GPU coherency stalls
- **Total: 15-30ms just for depth propagation**

## Solution

Replace CPU-driven loop with GPU-driven termination using Metal SharedEvent or persistent kernel with atomic termination flag.

### Approach A: GPU-Driven Loop with SharedEvent

Use Metal's `MTLSharedEvent` to signal completion from GPU without CPU polling:

```rust
// GPU kernel runs until no more changes
loop {
    dispatch_depths_kernel()
    if atomic_load(changed) == 0 {
        signal_shared_event(completion_value)
        break
    }
    atomic_store(changed, 0)
    level += 1
}
```

### Approach B: Single-Pass Parallel BFS

Use parallel breadth-first traversal where each thread processes one node and propagates to children:

```metal
kernel void propagate_depths_parallel(
    device Element* elements,
    device uint* depths,
    device atomic_uint* frontier_count,
    uint tid [[thread_position_in_grid]]
) {
    // Each thread checks if it's in current frontier
    // If so, propagates depth to all children
    // Children added to next frontier atomically
}
```

## Requirements

### Functional Requirements
1. Depth values identical to CPU-driven loop
2. Works for trees of any depth (tested up to 1000 levels)
3. Handles disconnected subtrees correctly
4. No CPU involvement during propagation

### Performance Requirements
1. **Target:** <1ms for 65K elements regardless of tree depth
2. **Eliminate:** All `wait_until_completed()` calls in depth loop
3. **Single dispatch:** One command buffer for entire propagation

### Non-Functional Requirements
1. Compatible with batched layout (Issue #140)
2. Works with async completion callbacks
3. Clear debugging/profiling support

## Technical Design

### Metal Kernel: Iterative Depth Propagation

```metal
// src/gpu_os/document/layout.metal

struct DepthPropagationState {
    atomic_uint changed;
    atomic_uint current_level;
    atomic_uint max_depth;
    uint element_count;
};

kernel void propagate_depths_gpu_driven(
    device Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    device DepthPropagationState* state [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    // GPU-driven loop - all threads participate
    uint level = 0;

    while (level < 1000) {  // Max depth safeguard
        // Barrier: all threads sync before checking level
        threadgroup_barrier(mem_flags::mem_device);

        // Thread 0 manages state
        if (tid == 0) {
            atomic_store_explicit(&state->changed, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Each thread processes elements at current level
        for (uint i = tid; i < state->element_count; i += threads_per_grid) {
            if (depths[i] == level) {
                // Propagate to children
                uint first_child = elements[i].first_child;
                uint child_count = elements[i].child_count;

                for (uint c = 0; c < child_count; c++) {
                    uint child_idx = first_child + c;
                    depths[child_idx] = level + 1;
                    atomic_store_explicit(&state->changed, 1, memory_order_relaxed);
                }
            }
        }

        // Barrier: wait for all propagation
        threadgroup_barrier(mem_flags::mem_device);

        // Check termination
        if (atomic_load_explicit(&state->changed, memory_order_relaxed) == 0) {
            // Thread 0 records max depth
            if (tid == 0) {
                atomic_store_explicit(&state->max_depth, level, memory_order_relaxed);
            }
            break;
        }

        level += 1;
    }
}
```

### Alternative: Level-Parallel with Indirect Dispatch

```metal
// Kernel that processes one level, writes indirect dispatch args for next level

kernel void propagate_depths_level(
    device Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    device uint* current_level [[buffer(2)]],
    device atomic_uint* next_level_count [[buffer(3)]],
    device uint* next_level_elements [[buffer(4)]],  // Elements to process next
    uint tid [[thread_position_in_grid]]
) {
    uint level = *current_level;
    Element elem = elements[tid];

    if (depths[tid] == level) {
        // Propagate to children
        for (uint c = 0; c < elem.child_count; c++) {
            uint child_idx = elem.first_child + c;
            depths[child_idx] = level + 1;

            // Add child to next level's work list
            uint slot = atomic_fetch_add_explicit(next_level_count, 1, memory_order_relaxed);
            next_level_elements[slot] = child_idx;
        }
    }
}

// Use MTLIndirectCommandBuffer for GPU-driven dispatch chain
```

### Rust Implementation

```rust
// src/gpu_os/document/layout.rs

impl LayoutEngine {
    pub fn propagate_depths_gpu_driven(&mut self, element_count: usize) {
        // Initialize state
        unsafe {
            let state_ptr = self.depth_state_buffer.contents() as *mut DepthPropagationState;
            (*state_ptr).changed = 0;
            (*state_ptr).current_level = 0;
            (*state_ptr).max_depth = 0;
            (*state_ptr).element_count = element_count as u32;
        }

        // Initialize root depths
        unsafe {
            let depths_ptr = self.depth_buffer.contents() as *mut u32;
            // Root elements (parent_index == 0) get depth 0
            for i in 0..element_count {
                let elem_ptr = self.element_buffer.contents() as *const Element;
                if (*elem_ptr.add(i)).parent_index == 0 {
                    *depths_ptr.add(i) = 0;
                } else {
                    *depths_ptr.add(i) = u32::MAX; // Unvisited
                }
            }
        }

        // Single dispatch - GPU handles iteration
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.propagate_depths_gpu_driven_pipeline);
        encoder.set_buffer(0, Some(&self.element_buffer), 0);
        encoder.set_buffer(1, Some(&self.depth_buffer), 0);
        encoder.set_buffer(2, Some(&self.depth_state_buffer), 0);

        // Dispatch enough threads to cover all elements
        let threads = MTLSize::new(element_count as u64, 1, 1);
        let threadgroup = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();

        // NO wait_until_completed() here!
        // Caller handles sync via batched command buffer
    }

    pub fn encode_depth_propagation(
        &self,
        command_buffer: &CommandBufferRef,
        element_count: usize
    ) {
        // For use in batched layout (Issue #140)
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("depth_propagation_gpu_driven");

        encoder.set_compute_pipeline_state(&self.propagate_depths_gpu_driven_pipeline);
        encoder.set_buffer(0, Some(&self.element_buffer), 0);
        encoder.set_buffer(1, Some(&self.depth_buffer), 0);
        encoder.set_buffer(2, Some(&self.depth_state_buffer), 0);

        let threads = MTLSize::new(element_count as u64, 1, 1);
        let threadgroup = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        // Memory barrier added by caller
    }
}
```

## Pseudocode

```
function propagate_depths_gpu_driven(elements, depths, state):
    # Initialize
    state.changed = 0
    state.level = 0

    # Mark roots
    for each element where parent == 0:
        depths[element] = 0

    # GPU-driven loop (runs entirely on GPU)
    while state.level < MAX_DEPTH:
        barrier()  # Sync all threads

        # Reset changed flag
        if thread_id == 0:
            state.changed = 0
        barrier()

        # Each thread processes elements at current level
        for element in my_elements:
            if depths[element] == state.level:
                for child in element.children:
                    depths[child] = state.level + 1
                    atomic_set(state.changed, 1)

        barrier()  # Wait for all propagation

        # Check termination
        if state.changed == 0:
            if thread_id == 0:
                state.max_depth = state.level
            break

        state.level += 1
```

## Test Plan

### Unit Tests

```rust
// tests/test_issue_141_gpu_depth_propagation.rs

#[test]
fn test_gpu_depth_matches_cpu_depth() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Various tree structures
    let test_cases = vec![
        create_linear_tree(100),      // Deep linear chain
        create_binary_tree(10),       // Balanced binary tree (2^10 = 1024 nodes)
        create_wide_tree(1000, 3),    // Wide tree (1000 children, 3 levels)
        create_random_tree(5000),     // Random structure
    ];

    for (i, elements) in test_cases.iter().enumerate() {
        // CPU reference implementation
        let cpu_depths = compute_depths_cpu(&elements);

        // GPU implementation
        let gpu_depths = layout_engine.propagate_depths_gpu_driven(&elements);

        assert_eq!(
            cpu_depths, gpu_depths,
            "Depth mismatch in test case {}", i
        );
    }
}

#[test]
fn test_deep_tree_performance() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Very deep tree (1000 levels, 1 element per level)
    let elements = create_linear_tree(1000);

    let start = std::time::Instant::now();
    for _ in 0..100 {
        layout_engine.propagate_depths_gpu_driven(&elements);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_millis() as f64 / 100.0;
    println!("Average depth propagation time: {}ms", avg_ms);

    // Should be <1ms regardless of tree depth
    assert!(avg_ms < 1.0, "Depth propagation too slow: {}ms", avg_ms);
}

#[test]
fn test_wide_tree_performance() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Wide tree (65K elements, 3 levels)
    let elements = create_wide_tree(65000, 3);

    let start = std::time::Instant::now();
    for _ in 0..100 {
        layout_engine.propagate_depths_gpu_driven(&elements);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_millis() as f64 / 100.0;
    println!("Average depth propagation time: {}ms", avg_ms);

    assert!(avg_ms < 1.0, "Depth propagation too slow: {}ms", avg_ms);
}

#[test]
fn test_no_cpu_gpu_roundtrips() {
    // Instrument to verify no wait_until_completed() in depth propagation
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    let elements = create_random_tree(1000);

    // This should complete without blocking
    let cmd = layout_engine.encode_depth_propagation_to_command_buffer(&elements);

    // Verify command buffer is uncommitted (no sync happened)
    // We're testing the API shape, not runtime behavior
    assert!(!cmd.is_committed());
}

#[test]
fn test_disconnected_subtrees() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Multiple root elements (disconnected subtrees)
    let elements = create_forest(10, 100); // 10 trees, 100 elements each

    let depths = layout_engine.propagate_depths_gpu_driven(&elements);

    // Verify each subtree has correct depths
    for tree_idx in 0..10 {
        let root = tree_idx * 100;
        assert_eq!(depths[root], 0, "Root {} should have depth 0", root);

        // Check children have depth 1, grandchildren depth 2, etc.
        verify_depth_invariants(&elements, &depths, root);
    }
}
```

### Visual Verification Tests

```rust
// tests/test_issue_141_visual.rs

#[test]
fn visual_test_depth_coloring() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);
    let mut renderer = TestRenderer::new(&device, 800, 600);

    // Create nested boxes
    let document = r#"
        <div id="l0" style="padding: 10px; background: hsl(0, 50%, 90%);">
            <div id="l1" style="padding: 10px; background: hsl(30, 50%, 85%);">
                <div id="l2" style="padding: 10px; background: hsl(60, 50%, 80%);">
                    <div id="l3" style="padding: 10px; background: hsl(90, 50%, 75%);">
                        <div id="l4" style="padding: 10px; background: hsl(120, 50%, 70%);">
                            <div id="l5" style="padding: 10px; background: hsl(150, 50%, 65%);">
                                Level 5 (depth=5)
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    "#;

    let (elements, styles) = parse_html_css(document);
    let layout = layout_engine.compute_layout_batched(&elements, &styles);

    // Color each element by depth
    for (i, elem) in elements.iter().enumerate() {
        let depth = layout.depths[i];
        let hue = depth as f32 * 30.0;
        renderer.set_element_color(i, hsl_to_rgb(hue, 0.5, 0.9 - depth as f32 * 0.05));
    }

    renderer.render_layout(&layout, &elements);
    renderer.save_to_file("tests/visual_output/depth_coloring.png");

    // Verify nested structure is visible
    let baseline = image::open("tests/visual_baselines/depth_coloring.png").unwrap();
    let actual = image::open("tests/visual_output/depth_coloring.png").unwrap();

    let diff = image_diff(&baseline, &actual);
    assert!(diff < 0.001, "Visual difference: {}", diff);
}

#[test]
fn visual_test_tree_visualization() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);
    let mut renderer = TestRenderer::new(&device, 1200, 800);

    // Create tree structure and visualize it
    let elements = create_binary_tree(6); // 63 nodes
    let depths = layout_engine.propagate_depths_gpu_driven(&elements);

    // Render tree with nodes colored by depth
    renderer.render_tree_diagram(&elements, &depths);
    renderer.save_to_file("tests/visual_output/tree_depths.png");

    // Verify max depth
    let max_depth = depths.iter().max().unwrap();
    assert_eq!(*max_depth, 5, "Binary tree depth should be 5");
}
```

## Success Metrics

1. **No CPU-GPU ping-pong:** Zero `wait_until_completed()` calls during depth propagation
2. **Constant time:** <1ms for any tree depth (up to 1000 levels)
3. **Correctness:** 100% depth value match with CPU reference
4. **Integration:** Works seamlessly with batched layout (Issue #140)

## Dependencies

- Issue #140: Batch Layout Passes (this is a sub-component)

## Files to Modify

1. `src/gpu_os/document/layout.rs` - Rust implementation
2. `src/gpu_os/document/layout.metal` - GPU kernel
3. `tests/test_issue_141_gpu_depth_propagation.rs` - Unit tests
4. `tests/test_issue_141_visual.rs` - Visual tests
