# Issue #140: Batch Layout Passes - Eliminate 7 Synchronous GPU Waits

## Problem Statement

The layout engine in `src/gpu_os/document/layout.rs` performs 7 sequential GPU operations, each with a blocking `wait_until_completed()` call. This serializes GPU work and adds ~700-1400µs of CPU overhead per layout pass.

**Current flow (lines 372-590):**
```
Pass 1: compute_intrinsic_sizes → wait_until_completed() [~100-200µs]
Pass 2: compute_block_layout → wait_until_completed() [~100-200µs]
Pass 3: init_depths → wait_until_completed() [~100-200µs]
Pass 4: propagate_depths (loop) → wait_until_completed() × N [~500-1000µs × depth]
Pass 5: sum_heights → wait_until_completed() [~100-200µs]
Pass 6: position_siblings → wait_until_completed() [~100-200µs]
Pass 7: finalize_level → wait_until_completed() [~100-200µs]
```

**Total overhead:** 700-1400µs just from dispatch overhead, not counting GPU execution time.

## Solution

Batch independent passes into a single command buffer with multiple compute encoders. Use memory barriers between dependent passes instead of CPU synchronization.

**Target flow:**
```
CommandBuffer {
  Encoder 1: compute_intrinsic_sizes
  Barrier (buffer scope)
  Encoder 2: compute_block_layout
  Barrier (buffer scope)
  Encoder 3: init_depths + propagate_depths (GPU-driven loop)
  Barrier (buffer scope)
  Encoder 4: sum_heights
  Barrier (buffer scope)
  Encoder 5: position_siblings
  Barrier (buffer scope)
  Encoder 6: finalize_level
}
commit()
// Single wait or async callback
```

## Requirements

### Functional Requirements
1. Layout produces identical results to current implementation
2. All 7 passes execute in correct dependency order
3. Memory barriers ensure GPU coherency between passes
4. Support both sync and async completion modes

### Performance Requirements
1. Reduce CPU overhead from 7 waits to 1 wait (or async callback)
2. Target: <100µs CPU overhead for layout dispatch
3. GPU execution time unchanged or improved (better occupancy)

### Non-Functional Requirements
1. No changes to public API
2. Backwards compatible with existing layout tests
3. Clear error handling if any pass fails

## Technical Design

### Metal Command Buffer Batching

```rust
// src/gpu_os/document/layout.rs

pub fn compute_layout_batched(&mut self, element_count: usize) -> LayoutResult {
    let command_buffer = self.command_queue.new_command_buffer();
    command_buffer.set_label("batched_layout");

    // Pass 1: Intrinsic sizes
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("intrinsic_sizes");
        encoder.set_compute_pipeline_state(&self.intrinsic_sizes_pipeline);
        self.bind_layout_buffers(encoder);
        let threads = MTLSize::new(element_count as u64, 1, 1);
        let threadgroup = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, threadgroup);
        encoder.end_encoding();
    }

    // Memory barrier between passes
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);
        encoder.end_encoding();
    }

    // Pass 2: Block layout
    {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_label("block_layout");
        encoder.set_compute_pipeline_state(&self.block_layout_pipeline);
        self.bind_layout_buffers(encoder);
        encoder.dispatch_threads(threads, threadgroup);
        encoder.end_encoding();
    }

    // ... continue for all passes ...

    // Pass 4: GPU-driven depth propagation (see Issue #141)
    self.encode_depth_propagation(command_buffer, element_count);

    // Final passes...

    command_buffer.commit();

    // Option A: Sync wait (for compatibility)
    if self.sync_mode {
        command_buffer.wait_until_completed();
        self.read_layout_results()
    } else {
        // Option B: Async callback
        let result_buffer = self.layout_buffer.clone();
        command_buffer.add_completed_handler(block::ConcreteBlock::new(move |_| {
            // Signal completion via channel or atomic
        }));
        LayoutResult::Pending
    }
}

fn bind_layout_buffers(&self, encoder: &ComputeCommandEncoderRef) {
    encoder.set_buffer(0, Some(&self.element_buffer), 0);
    encoder.set_buffer(1, Some(&self.style_buffer), 0);
    encoder.set_buffer(2, Some(&self.layout_buffer), 0);
    encoder.set_buffer(3, Some(&self.depth_buffer), 0);
    encoder.set_buffer(4, Some(&self.element_count_buffer), 0);
}
```

### Metal Shader Memory Barriers

```metal
// src/gpu_os/document/layout.metal

// Add explicit memory barriers in shaders that write data read by next pass
kernel void compute_intrinsic_sizes(
    device Element* elements [[buffer(0)]],
    device LayoutBox* layout [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // ... compute intrinsic sizes ...

    // Ensure writes are visible to subsequent passes
    threadgroup_barrier(mem_flags::mem_device);
}
```

## Pseudocode

```
function compute_layout_batched(elements, styles):
    cmd_buffer = queue.new_command_buffer()

    # Phase 1: Size computation (independent per element)
    encode_pass(cmd_buffer, "intrinsic_sizes", elements.count)
    encode_barrier(cmd_buffer)

    # Phase 2: Block layout (depends on intrinsic sizes)
    encode_pass(cmd_buffer, "block_layout", elements.count)
    encode_barrier(cmd_buffer)

    # Phase 3: Depth propagation (level-by-level, GPU-driven)
    encode_depth_propagation_gpu_driven(cmd_buffer, elements.count)
    encode_barrier(cmd_buffer)

    # Phase 4: Height summation (depends on depths)
    encode_pass(cmd_buffer, "sum_heights", elements.count)
    encode_barrier(cmd_buffer)

    # Phase 5: Sibling positioning (depends on heights)
    encode_pass(cmd_buffer, "position_siblings", elements.count)
    encode_barrier(cmd_buffer)

    # Phase 6: Finalization
    encode_pass(cmd_buffer, "finalize_level", elements.count)

    cmd_buffer.commit()

    if sync_mode:
        cmd_buffer.wait_until_completed()
        return read_results()
    else:
        cmd_buffer.add_completed_handler(on_complete)
        return PENDING
```

## Test Plan

### Unit Tests

```rust
// tests/test_issue_140_batched_layout.rs

#[test]
fn test_batched_layout_matches_sequential() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Create test document
    let elements = create_test_elements(1000);
    let styles = create_test_styles();

    // Run sequential (old) layout
    let sequential_result = layout_engine.compute_layout_sequential(&elements, &styles);

    // Run batched (new) layout
    let batched_result = layout_engine.compute_layout_batched(&elements, &styles);

    // Compare all layout boxes
    for i in 0..elements.len() {
        assert_eq!(
            sequential_result.boxes[i],
            batched_result.boxes[i],
            "Layout mismatch at element {}", i
        );
    }
}

#[test]
fn test_batched_layout_performance() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Large document
    let elements = create_test_elements(10_000);
    let styles = create_test_styles();

    // Warmup
    for _ in 0..5 {
        layout_engine.compute_layout_batched(&elements, &styles);
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..100 {
        layout_engine.compute_layout_batched(&elements, &styles);
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() / 100;
    println!("Average layout time: {}µs", avg_us);

    // CPU overhead should be <100µs (dispatch only, not GPU execution)
    // Total time depends on GPU, but dispatch overhead is what we're measuring
    assert!(avg_us < 5000, "Layout too slow: {}µs", avg_us);
}

#[test]
fn test_batched_layout_async_completion() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);
    layout_engine.set_sync_mode(false);

    let elements = create_test_elements(1000);
    let styles = create_test_styles();

    let (tx, rx) = std::sync::mpsc::channel();

    layout_engine.compute_layout_batched_async(&elements, &styles, move |result| {
        tx.send(result).unwrap();
    });

    // Should complete within reasonable time
    let result = rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
    assert!(result.is_ok());
}

#[test]
fn test_memory_barriers_prevent_race() {
    // Test that barriers correctly synchronize passes
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);

    // Document with deep nesting (exercises depth propagation)
    let elements = create_deeply_nested_elements(100, 20); // 100 elements, 20 levels
    let styles = create_test_styles();

    // Run many times to catch race conditions
    for i in 0..1000 {
        let result = layout_engine.compute_layout_batched(&elements, &styles);

        // Verify depths are monotonically increasing parent->child
        for j in 1..elements.len() {
            let parent_depth = result.depths[elements[j].parent_index];
            let child_depth = result.depths[j];
            assert!(
                child_depth == parent_depth + 1 || elements[j].parent_index == 0,
                "Depth invariant violated at iteration {}, element {}", i, j
            );
        }
    }
}
```

### Visual Verification Tests

```rust
// tests/test_issue_140_visual.rs

#[test]
fn visual_test_batched_layout_complex_document() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);
    let mut renderer = TestRenderer::new(&device, 1024, 768);

    // Complex document with flexbox, nested elements, text
    let document = r#"
        <div style="display: flex; flex-direction: column; padding: 20px;">
            <header style="height: 60px; background: #333;">Header</header>
            <main style="display: flex; flex: 1;">
                <nav style="width: 200px; background: #eee;">
                    <ul>
                        <li>Item 1</li>
                        <li>Item 2</li>
                        <li>Item 3</li>
                    </ul>
                </nav>
                <article style="flex: 1; padding: 20px;">
                    <h1>Title</h1>
                    <p>Lorem ipsum dolor sit amet...</p>
                    <p>Second paragraph with more text...</p>
                </article>
            </main>
            <footer style="height: 40px; background: #333;">Footer</footer>
        </div>
    "#;

    let (elements, styles) = parse_html_css(document);
    let layout = layout_engine.compute_layout_batched(&elements, &styles);

    renderer.render_layout(&layout, &elements);
    renderer.save_to_file("tests/visual_output/batched_layout_complex.png");

    // Compare against baseline
    let baseline = image::open("tests/visual_baselines/batched_layout_complex.png").unwrap();
    let actual = image::open("tests/visual_output/batched_layout_complex.png").unwrap();

    let diff = image_diff(&baseline, &actual);
    assert!(diff < 0.001, "Visual difference too large: {}", diff);
}

#[test]
fn visual_test_batched_layout_stress() {
    let device = Device::system_default().unwrap();
    let mut layout_engine = LayoutEngine::new(&device);
    let mut renderer = TestRenderer::new(&device, 1920, 1080);

    // 10,000 elements grid
    let elements = create_grid_elements(100, 100); // 100x100 grid
    let styles = create_grid_styles();

    let layout = layout_engine.compute_layout_batched(&elements, &styles);

    renderer.render_layout(&layout, &elements);
    renderer.save_to_file("tests/visual_output/batched_layout_stress.png");

    // Verify grid alignment
    for row in 0..100 {
        for col in 0..100 {
            let idx = row * 100 + col;
            let box_ = &layout.boxes[idx];

            // All boxes in same row should have same y
            if col > 0 {
                let prev_box = &layout.boxes[idx - 1];
                assert_eq!(box_.y, prev_box.y, "Row {} misaligned", row);
            }

            // All boxes in same column should have same x
            if row > 0 {
                let above_box = &layout.boxes[idx - 100];
                assert_eq!(box_.x, above_box.x, "Column {} misaligned", col);
            }
        }
    }
}
```

## Success Metrics

1. **CPU overhead reduction:** 7 waits → 1 wait (or async)
2. **Dispatch time:** <100µs for layout dispatch (excluding GPU execution)
3. **Test pass rate:** 100% of existing layout tests pass
4. **Visual regression:** <0.1% pixel difference from baseline

## Dependencies

- Issue #141: GPU-Driven Depth Propagation (for the loop elimination)

## Files to Modify

1. `src/gpu_os/document/layout.rs` - Main batching logic
2. `src/gpu_os/document/layout.metal` - Add memory barriers
3. `tests/test_issue_140_batched_layout.rs` - New test file
4. `tests/test_issue_140_visual.rs` - Visual tests

## Rollout Plan

1. Implement batched layout alongside existing sequential
2. Add feature flag to switch between them
3. Run all tests with both modes
4. Benchmark performance difference
5. Remove sequential implementation once validated
