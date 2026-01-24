# PRD: Level-Parallel Layout Engine

## Problem

**Current layout.metal violates GPU-native principles:**
```metal
kernel void layout_children_sequential(...) {
    if (gid != 0) return;  // 1023 threads idle
    int stack[256];        // Overflows on large DOMs
}
```

**Result:** Wikipedia (3000+ elements) text all overlaps at y=0.

## Solution (5-Line Algorithm)

```
1. depth[i] = count ancestors           // all threads
2. For L = max..0: sum child heights    // threads at depth L
3. For L = 0..max: accumulate sibling Y // threads at depth L
4. absolute = parent.content + relative // all threads
```

## Core Kernels

### 1. Compute Depths (All Threads)

```metal
kernel void layout_compute_depths(
    device const Element* elements,
    device uint* depths,
    device atomic_uint* max_depth,
    constant uint& count,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint d = 0;
    int parent = elements[gid].parent;
    while (parent >= 0 && d < 256) {
        d++;
        parent = elements[parent].parent;
    }
    depths[gid] = d;
    atomic_fetch_max_explicit(max_depth, d, memory_order_relaxed);
}
```

### 2. Sum Child Heights (Per Level, Bottom-Up)

```metal
kernel void layout_sum_heights(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes,
    device const uint* depths,
    constant uint& level,
    constant uint& count,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count || depths[gid] != level) return;
    if (styles[gid].height > 0) return;  // Explicit height

    float h = 0;
    int child = elements[gid].first_child;
    while (child >= 0) {
        h += boxes[child].height + styles[child].margin[0] + styles[child].margin[2];
        child = elements[child].next_sibling;
    }

    boxes[gid].height = h + styles[gid].padding[0] + styles[gid].padding[2];
    boxes[gid].content_height = h;
}
```

### 3. Position Siblings (Per Level, Top-Down)

```metal
kernel void layout_position_siblings(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes,
    device const uint* depths,
    constant uint& level,
    constant uint& count,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count || depths[gid] != level) return;

    // Count preceding siblings' heights
    float y = styles[gid].margin[0];
    int parent = elements[gid].parent;
    if (parent >= 0) {
        int sib = elements[parent].first_child;
        while (sib >= 0 && sib != int(gid)) {
            y += boxes[sib].height + styles[sib].margin[0] + styles[sib].margin[2];
            sib = elements[sib].next_sibling;
        }
    }

    boxes[gid].y = y;
    boxes[gid].x = styles[gid].margin[3];
}
```

### 4. Finalize Absolute (All Threads)

```metal
kernel void layout_finalize(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes,
    constant uint& count,
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    int parent = elements[gid].parent;
    if (parent >= 0) {
        boxes[gid].x += boxes[parent].content_x;
        boxes[gid].y += boxes[parent].content_y;
    }

    boxes[gid].content_x = boxes[gid].x + styles[gid].border_width[3] + styles[gid].padding[3];
    boxes[gid].content_y = boxes[gid].y + styles[gid].border_width[0] + styles[gid].padding[0];
}
```

## Rust Dispatch

```rust
pub fn compute_layout(&mut self, elements: &[Element], styles: &[ComputedStyle], viewport: Viewport) -> Vec<LayoutBox> {
    // 1. Depths (single dispatch)
    self.dispatch_depths(element_count);

    // 2. Read max_depth
    let max_depth = self.read_max_depth();

    // 3. Heights bottom-up (max_depth dispatches)
    for level in (0..=max_depth).rev() {
        self.dispatch_heights(level, element_count);
    }

    // 4. Positions top-down (max_depth dispatches)
    for level in 0..=max_depth {
        self.dispatch_positions(level, element_count);
    }

    // 5. Finalize (single dispatch)
    self.dispatch_finalize(element_count);

    self.read_boxes()
}
```

## Data Structures

```rust
// Add to GpuLayoutEngine
depth_buffer: Buffer,      // uint per element
max_depth_buffer: Buffer,  // single atomic_uint
```

## Tests

See `tests/test_issue_89_layout.rs`:
- `test_simple_stacking` - 3 divs stack correctly
- `test_500_elements_no_stack_overflow` - exceeds old 256 limit
- `test_1000_elements_performance` - completes in <100ms
- `test_3000_elements_wikipedia_scale` - Wikipedia-size DOM works

## Success Criteria

1. Wikipedia renders without text overlap
2. All 1024 threads participate in every pass
3. No fixed-size stacks
4. <16ms for 10K elements

## Files

- `src/gpu_os/document/layout.metal` - New kernels
- `src/gpu_os/document/layout.rs` - New dispatch logic
- `tests/test_issue_89_layout.rs` - Tests

## Non-Goals (V1)

- Margin collapsing
- Inline layout
- Flex layout optimization
- Float/positioned elements
