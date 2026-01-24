# Issue #106: GPU Layout Engine

## Summary
Implement a complete GPU-native layout engine that computes element positions and sizes using Metal compute shaders, handling block, inline, and flex layouts.

## Motivation
Layout computation is the core of browser rendering. Current implementation has issues with:
- Vertical whitespace (margins not collapsing)
- Percentage resolution
- Proper block formatting context
- Inline layout

A complete rewrite using GPU compute can achieve 5-20x speedup while fixing correctness issues.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU Layout Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Width Resolution (Top-Down)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ For each depth level (root → leaves):                    │   │
│  │   - Resolve width from parent or specified value         │   │
│  │   - Handle min-width, max-width constraints              │   │
│  │   - Calculate available width for children               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Phase 2: Content Sizing (Bottom-Up)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ For each depth level (leaves → root):                    │   │
│  │   - Measure text content                                 │   │
│  │   - Sum children heights (block) or widths (inline)      │   │
│  │   - Collapse margins                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Phase 3: Height Resolution                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ For each depth level (root → leaves):                    │   │
│  │   - Apply specified height or use content height         │   │
│  │   - Handle min-height, max-height constraints            │   │
│  │   - Calculate overflow                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Phase 4: Position Computation                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ For each depth level (root → leaves):                    │   │
│  │   - Calculate x,y relative to parent                     │   │
│  │   - Handle position: relative/absolute/fixed             │   │
│  │   - Apply margin offsets                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  Output:                     ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ GPULayoutBox[node_count]                                 │   │
│  │ - position (x, y)                                        │   │
│  │ - size (width, height)                                   │   │
│  │ - content_box, padding_box, border_box                   │   │
│  │ - scroll_size                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

```metal
// Layout output per node
struct GPULayoutBox {
    // Border box (includes border + padding + content)
    float2 position;        // Relative to parent's content box
    float2 size;

    // Content box
    float2 content_position;  // Relative to border box origin
    float2 content_size;

    // Scroll dimensions (for overflow: scroll)
    float2 scroll_size;

    // For absolute positioning
    float2 absolute_position;  // Relative to viewport or positioned ancestor

    // Flags
    uint flags;             // Out-of-flow, has overflow, etc.
};

// Margin collapse state
struct MarginState {
    float pending_margin_top;
    float pending_margin_bottom;
    bool margin_collapse_through;
};

// Formatting context
constant uint FC_BLOCK = 0;
constant uint FC_INLINE = 1;
constant uint FC_FLEX = 2;
constant uint FC_GRID = 3;

// Layout flags
constant uint LAYOUT_OUT_OF_FLOW = 1;
constant uint LAYOUT_NEW_BFC = 2;
constant uint LAYOUT_HAS_OVERFLOW = 4;
```

## Metal Kernel Implementation

```metal
// src/gpu_os/document/layout_engine.metal

#include <metal_stdlib>
using namespace metal;

//=============================================================================
// Phase 1: Width Resolution (Top-Down)
//=============================================================================

kernel void resolve_widths(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    device uint* node_depths [[buffer(3)]],
    constant uint& node_count [[buffer(4)]],
    constant uint& current_depth [[buffer(5)]],
    constant float2& viewport_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;
    if (node_depths[gid] != current_depth) return;

    GPUNode node = nodes[gid];
    GPUComputedStyle style = styles[gid];

    // Skip display:none
    if (style.display == DISPLAY_NONE) {
        layout[gid].size = float2(0, 0);
        layout[gid].flags |= LAYOUT_OUT_OF_FLOW;
        return;
    }

    // Get containing block width
    float containing_width = viewport_size.x;
    if (node.parent_idx != INVALID_IDX) {
        containing_width = layout[node.parent_idx].content_size.x;
    }

    // Resolve width
    float width;
    float padding_h = style.padding.y + style.padding.w;
    float border_h = style.border_width.y + style.border_width.w;
    float margin_h = style.margin.y + style.margin.w;

    if (isnan(style.width)) {
        // Auto width
        if (is_block_level(style)) {
            // Block elements stretch to fill
            width = containing_width - margin_h;
        } else {
            // Inline elements shrink-to-fit (computed in Phase 2)
            width = NAN;
        }
    } else {
        width = style.width + padding_h + border_h;
    }

    // Apply min/max constraints
    if (!isnan(width)) {
        if (style.min_width > 0 && width < style.min_width) {
            width = style.min_width;
        }
        if (style.max_width != INFINITY && width > style.max_width) {
            width = style.max_width;
        }
    }

    layout[gid].size.x = width;
    layout[gid].content_size.x = width - padding_h - border_h;
}

//=============================================================================
// Phase 2: Content Sizing (Bottom-Up)
//=============================================================================

kernel void measure_content(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    device uint* node_depths [[buffer(3)]],
    device TextMeasurement* text_measurements [[buffer(4)]],
    constant uint& node_count [[buffer(5)]],
    constant uint& current_depth [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;
    if (node_depths[gid] != current_depth) return;

    GPUNode node = nodes[gid];
    GPUComputedStyle style = styles[gid];

    if (style.display == DISPLAY_NONE) return;

    float content_height = 0;
    float content_width = 0;

    // Sum children sizes based on formatting context
    uint fc = get_formatting_context(style);

    if (fc == FC_BLOCK) {
        // Block formatting: stack children vertically
        float prev_margin_bottom = 0;

        uint child = node.first_child_idx;
        while (child != INVALID_IDX) {
            GPUComputedStyle child_style = styles[child];
            GPULayoutBox child_layout = layout[child];

            // Skip out-of-flow elements
            if (child_layout.flags & LAYOUT_OUT_OF_FLOW) {
                child = nodes[child].next_sibling_idx;
                continue;
            }

            // Margin collapsing
            float margin_top = child_style.margin.x;
            float collapsed_margin = max(prev_margin_bottom, margin_top);

            content_height += collapsed_margin;
            content_height += child_layout.size.y;

            prev_margin_bottom = child_style.margin.z;
            child = nodes[child].next_sibling_idx;
        }

    } else if (fc == FC_INLINE) {
        // Inline formatting: flow children horizontally with wrapping
        float line_height = 0;
        float line_width = 0;
        float max_width = layout[gid].content_size.x;

        uint child = node.first_child_idx;
        while (child != INVALID_IDX) {
            GPULayoutBox child_layout = layout[child];

            if (line_width + child_layout.size.x > max_width && line_width > 0) {
                // Wrap to next line
                content_height += line_height;
                line_width = 0;
                line_height = 0;
            }

            line_width += child_layout.size.x;
            line_height = max(line_height, child_layout.size.y);
            content_width = max(content_width, line_width);

            child = nodes[child].next_sibling_idx;
        }

        content_height += line_height;

    } else if (fc == FC_FLEX) {
        // Flexbox layout
        if (style.flex_direction == FLEX_ROW) {
            // Row: sum widths, max height
            uint child = node.first_child_idx;
            while (child != INVALID_IDX) {
                GPULayoutBox child_layout = layout[child];
                content_width += child_layout.size.x + style.gap;
                content_height = max(content_height, child_layout.size.y);
                child = nodes[child].next_sibling_idx;
            }
            content_width -= style.gap; // Remove trailing gap
        } else {
            // Column: max width, sum heights
            uint child = node.first_child_idx;
            while (child != INVALID_IDX) {
                GPULayoutBox child_layout = layout[child];
                content_width = max(content_width, child_layout.size.x);
                content_height += child_layout.size.y + style.gap;
                child = nodes[child].next_sibling_idx;
            }
            content_height -= style.gap;
        }
    }

    // Handle text nodes
    if (node.element_type == ELEM_TEXT) {
        TextMeasurement tm = text_measurements[gid];
        content_width = tm.width;
        content_height = tm.height;
    }

    // Store intrinsic content size
    layout[gid].scroll_size = float2(content_width, content_height);

    // If width was auto, set it now
    if (isnan(layout[gid].size.x)) {
        float padding_h = style.padding.y + style.padding.w;
        float border_h = style.border_width.y + style.border_width.w;
        layout[gid].size.x = content_width + padding_h + border_h;
        layout[gid].content_size.x = content_width;
    }
}

//=============================================================================
// Phase 3: Height Resolution
//=============================================================================

kernel void resolve_heights(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    device uint* node_depths [[buffer(3)]],
    constant uint& node_count [[buffer(4)]],
    constant uint& current_depth [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;
    if (node_depths[gid] != current_depth) return;

    GPUNode node = nodes[gid];
    GPUComputedStyle style = styles[gid];

    if (style.display == DISPLAY_NONE) return;

    float padding_v = style.padding.x + style.padding.z;
    float border_v = style.border_width.x + style.border_width.z;

    float height;
    if (isnan(style.height)) {
        // Auto height: use content height
        height = layout[gid].scroll_size.y + padding_v + border_v;
    } else {
        height = style.height + padding_v + border_v;
    }

    // Apply min/max constraints
    if (style.min_height > 0 && height < style.min_height) {
        height = style.min_height;
    }
    if (style.max_height != INFINITY && height > style.max_height) {
        height = style.max_height;
    }

    layout[gid].size.y = height;
    layout[gid].content_size.y = height - padding_v - border_v;

    // Check for overflow
    if (layout[gid].scroll_size.y > layout[gid].content_size.y) {
        layout[gid].flags |= LAYOUT_HAS_OVERFLOW;
    }
}

//=============================================================================
// Phase 4: Position Computation
//=============================================================================

kernel void compute_positions(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    device uint* node_depths [[buffer(3)]],
    constant uint& node_count [[buffer(4)]],
    constant uint& current_depth [[buffer(5)]],
    constant float2& viewport_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;
    if (node_depths[gid] != current_depth) return;

    GPUNode node = nodes[gid];
    GPUComputedStyle style = styles[gid];

    if (style.display == DISPLAY_NONE) return;

    // Get parent position and style
    float2 parent_content_pos = float2(0, 0);
    if (node.parent_idx != INVALID_IDX) {
        GPULayoutBox parent_layout = layout[node.parent_idx];
        parent_content_pos = parent_layout.position + parent_layout.content_position;
    }

    float2 pos = parent_content_pos;

    // Add margin offset
    pos.x += style.margin.w;
    pos.y += style.margin.x;

    // Add offset from previous siblings (block layout)
    if (node.prev_sibling_idx != INVALID_IDX) {
        uint prev = node.prev_sibling_idx;

        // Find previous in-flow sibling
        while (prev != INVALID_IDX && (layout[prev].flags & LAYOUT_OUT_OF_FLOW)) {
            prev = nodes[prev].prev_sibling_idx;
        }

        if (prev != INVALID_IDX) {
            GPULayoutBox prev_layout = layout[prev];
            GPUComputedStyle prev_style = styles[prev];

            // Block layout: stack vertically
            pos.y = prev_layout.position.y + prev_layout.size.y;

            // Margin collapsing between siblings
            float prev_margin_bottom = prev_style.margin.z;
            float this_margin_top = style.margin.x;
            float collapsed = max(prev_margin_bottom, this_margin_top);
            pos.y += collapsed - prev_margin_bottom; // Adjust for collapse
        }
    }

    // Handle position: relative
    if (style.position == POSITION_RELATIVE) {
        if (!isnan(style.top)) pos.y += style.top;
        if (!isnan(style.left)) pos.x += style.left;
        if (!isnan(style.bottom)) pos.y -= style.bottom;
        if (!isnan(style.right_)) pos.x -= style.right_;
    }

    // Handle position: absolute/fixed
    if (style.position == POSITION_ABSOLUTE || style.position == POSITION_FIXED) {
        layout[gid].flags |= LAYOUT_OUT_OF_FLOW;

        float2 containing_size = viewport_size;
        float2 containing_pos = float2(0, 0);

        if (style.position == POSITION_ABSOLUTE) {
            // Find positioned ancestor
            uint ancestor = node.parent_idx;
            while (ancestor != INVALID_IDX) {
                GPUComputedStyle anc_style = styles[ancestor];
                if (anc_style.position != POSITION_STATIC) {
                    containing_pos = layout[ancestor].position;
                    containing_size = layout[ancestor].size;
                    break;
                }
                ancestor = nodes[ancestor].parent_idx;
            }
        }

        // Calculate position from inset properties
        if (!isnan(style.left)) {
            pos.x = containing_pos.x + style.left;
        } else if (!isnan(style.right_)) {
            pos.x = containing_pos.x + containing_size.x - layout[gid].size.x - style.right_;
        }

        if (!isnan(style.top)) {
            pos.y = containing_pos.y + style.top;
        } else if (!isnan(style.bottom)) {
            pos.y = containing_pos.y + containing_size.y - layout[gid].size.y - style.bottom;
        }

        layout[gid].absolute_position = pos;
    }

    layout[gid].position = pos;

    // Content box position (offset by border + padding)
    layout[gid].content_position = float2(
        style.border_width.w + style.padding.w,
        style.border_width.x + style.padding.x
    );
}

//=============================================================================
// Flex Layout (specialized kernel)
//=============================================================================

kernel void layout_flex_container(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    device uint* flex_containers [[buffer(3)]],
    constant uint& container_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= container_count) return;

    uint container_idx = flex_containers[gid];
    GPUNode container = nodes[container_idx];
    GPUComputedStyle style = styles[container_idx];
    GPULayoutBox container_layout = layout[container_idx];

    bool is_row = style.flex_direction == FLEX_ROW;
    float main_size = is_row ? container_layout.content_size.x : container_layout.content_size.y;
    float cross_size = is_row ? container_layout.content_size.y : container_layout.content_size.x;

    // Phase 1: Calculate total flex-basis and flex factors
    float total_basis = 0;
    float total_grow = 0;
    float total_shrink = 0;
    uint child_count = 0;

    uint child = container.first_child_idx;
    while (child != INVALID_IDX) {
        GPUComputedStyle child_style = styles[child];
        GPULayoutBox child_layout = layout[child];

        float basis = isnan(child_style.flex_basis)
            ? (is_row ? child_layout.size.x : child_layout.size.y)
            : child_style.flex_basis;

        total_basis += basis;
        total_grow += child_style.flex_grow;
        total_shrink += child_style.flex_shrink;
        child_count++;

        child = nodes[child].next_sibling_idx;
    }

    // Account for gaps
    float total_gap = style.gap * (child_count > 1 ? child_count - 1 : 0);
    float available_space = main_size - total_basis - total_gap;

    // Phase 2: Distribute space
    float main_pos = 0;

    // Handle justify-content
    if (available_space > 0 && total_grow == 0) {
        switch (style.justify_content) {
            case JUSTIFY_CENTER:
                main_pos = available_space / 2;
                break;
            case JUSTIFY_END:
                main_pos = available_space;
                break;
            case JUSTIFY_SPACE_BETWEEN:
                // Handled per-item
                break;
            case JUSTIFY_SPACE_AROUND:
                main_pos = available_space / (child_count * 2);
                break;
        }
    }

    float space_between = 0;
    if (style.justify_content == JUSTIFY_SPACE_BETWEEN && child_count > 1) {
        space_between = available_space / (child_count - 1);
    }

    // Phase 3: Position children
    child = container.first_child_idx;
    while (child != INVALID_IDX) {
        GPUComputedStyle child_style = styles[child];
        GPULayoutBox child_layout = layout[child];

        // Calculate final main size
        float basis = isnan(child_style.flex_basis)
            ? (is_row ? child_layout.size.x : child_layout.size.y)
            : child_style.flex_basis;

        float flex_size = basis;
        if (available_space > 0 && total_grow > 0) {
            flex_size += available_space * (child_style.flex_grow / total_grow);
        } else if (available_space < 0 && total_shrink > 0) {
            flex_size += available_space * (child_style.flex_shrink / total_shrink);
        }

        // Calculate cross position based on align-items
        float cross_pos = 0;
        float child_cross_size = is_row ? child_layout.size.y : child_layout.size.x;

        switch (style.align_items) {
            case ALIGN_CENTER:
                cross_pos = (cross_size - child_cross_size) / 2;
                break;
            case ALIGN_END:
                cross_pos = cross_size - child_cross_size;
                break;
            case ALIGN_STRETCH:
                child_cross_size = cross_size;
                break;
            // ALIGN_START is default (cross_pos = 0)
        }

        // Update child layout
        float2 pos = container_layout.position + container_layout.content_position;
        if (is_row) {
            layout[child].position = float2(pos.x + main_pos, pos.y + cross_pos);
            layout[child].size.x = flex_size;
            if (style.align_items == ALIGN_STRETCH) {
                layout[child].size.y = child_cross_size;
            }
        } else {
            layout[child].position = float2(pos.x + cross_pos, pos.y + main_pos);
            layout[child].size.y = flex_size;
            if (style.align_items == ALIGN_STRETCH) {
                layout[child].size.x = child_cross_size;
            }
        }

        main_pos += flex_size + style.gap + space_between;
        child = nodes[child].next_sibling_idx;
    }
}
```

## Rust Integration

```rust
pub struct GPULayoutEngine {
    device: Device,
    resolve_widths_pipeline: ComputePipelineState,
    measure_content_pipeline: ComputePipelineState,
    resolve_heights_pipeline: ComputePipelineState,
    compute_positions_pipeline: ComputePipelineState,
    flex_layout_pipeline: ComputePipelineState,
    layout_buffer: Buffer,
}

impl GPULayoutEngine {
    pub fn layout(
        &self,
        command_buffer: &CommandBufferRef,
        gpu_dom: &GPUDom,
        computed_styles: &Buffer,
        viewport_size: (f32, f32),
    ) {
        let max_depth = gpu_dom.max_depth;
        let encoder = command_buffer.new_compute_command_encoder();

        // Phase 1: Resolve widths (top-down)
        for depth in 0..=max_depth {
            self.dispatch_kernel(
                &encoder,
                &self.resolve_widths_pipeline,
                gpu_dom,
                computed_styles,
                depth,
                viewport_size,
            );
            encoder.memory_barrier_with_resources(&[&self.layout_buffer]);
        }

        // Phase 2: Measure content (bottom-up)
        for depth in (0..=max_depth).rev() {
            self.dispatch_kernel(
                &encoder,
                &self.measure_content_pipeline,
                gpu_dom,
                computed_styles,
                depth,
                viewport_size,
            );
            encoder.memory_barrier_with_resources(&[&self.layout_buffer]);
        }

        // Phase 3: Resolve heights (top-down for percentage, bottom-up already done)
        for depth in 0..=max_depth {
            self.dispatch_kernel(
                &encoder,
                &self.resolve_heights_pipeline,
                gpu_dom,
                computed_styles,
                depth,
                viewport_size,
            );
            encoder.memory_barrier_with_resources(&[&self.layout_buffer]);
        }

        // Phase 4: Compute positions (top-down)
        for depth in 0..=max_depth {
            self.dispatch_kernel(
                &encoder,
                &self.compute_positions_pipeline,
                gpu_dom,
                computed_styles,
                depth,
                viewport_size,
            );
            encoder.memory_barrier_with_resources(&[&self.layout_buffer]);
        }

        // Phase 5: Flex container layout
        if gpu_dom.flex_container_count > 0 {
            encoder.set_compute_pipeline_state(&self.flex_layout_pipeline);
            // ... set buffers
            let threads = MTLSize::new(gpu_dom.flex_container_count as u64, 1, 1);
            encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
        }

        encoder.end_encoding();
    }
}
```

## Benchmarks

```rust
fn bench_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_engine");

    let test_pages = vec![
        ("simple", "test_pages/simple.html"),
        ("nested", "test_pages/deeply_nested.html"),
        ("wikipedia", "test_pages/wikipedia.html"),
        ("flexbox", "test_pages/flexbox_heavy.html"),
    ];

    for (name, path) in test_pages {
        // CPU baseline
        group.bench_function(format!("cpu_{}", name), |b| {
            b.iter(|| cpu_layout(&doc, &styles, viewport))
        });

        // GPU implementation
        group.bench_function(format!("gpu_{}", name), |b| {
            b.iter(|| {
                let cmd = queue.new_command_buffer();
                engine.layout(&cmd, &gpu_dom, &styles_buffer, viewport);
                cmd.commit();
                cmd.wait_until_completed();
            })
        });
    }
}
```

### Expected Results

| Page | Nodes | Depth | CPU Time | GPU Time | Speedup |
|------|-------|-------|----------|----------|---------|
| Simple | 50 | 5 | 0.5ms | 0.1ms | 5x |
| Nested | 500 | 20 | 10ms | 0.5ms | 20x |
| Wikipedia | 5000 | 15 | 50ms | 3ms | 17x |
| Flexbox | 1000 | 8 | 15ms | 0.8ms | 19x |

## Tests

```rust
#[test]
fn test_block_layout() {
    let html = "<div style='width:100px'><p>A</p><p>B</p></div>";
    let layout = compute_layout(html, (800, 600));

    let div = &layout[0];
    assert_eq!(div.size.x, 100.0);

    let p1 = &layout[1];
    let p2 = &layout[2];

    // Paragraphs should stack vertically
    assert!(p2.position.y > p1.position.y + p1.size.y - 1.0);
}

#[test]
fn test_margin_collapse() {
    let html = r#"
        <p style='margin-bottom:20px'>A</p>
        <p style='margin-top:30px'>B</p>
    "#;
    let layout = compute_layout(html, (800, 600));

    let p1 = &layout[0];
    let p2 = &layout[1];

    // Gap should be max(20, 30) = 30, not 50
    let gap = p2.position.y - (p1.position.y + p1.size.y);
    assert!((gap - 30.0).abs() < 1.0);
}

#[test]
fn test_percentage_width() {
    let html = r#"
        <div style='width:200px'>
            <div style='width:50%'></div>
        </div>
    "#;
    let layout = compute_layout(html, (800, 600));

    assert_eq!(layout[1].size.x, 100.0);
}

#[test]
fn test_flexbox_row() {
    let html = r#"
        <div style='display:flex; width:300px'>
            <div style='flex:1'>A</div>
            <div style='flex:2'>B</div>
        </div>
    "#;
    let layout = compute_layout(html, (800, 600));

    // Flex items should divide space 1:2
    assert!((layout[1].size.x - 100.0).abs() < 1.0);
    assert!((layout[2].size.x - 200.0).abs() < 1.0);
}

#[test]
fn test_absolute_positioning() {
    let html = r#"
        <div style='position:relative; width:200px; height:200px'>
            <div style='position:absolute; top:10px; left:10px; width:50px; height:50px'></div>
        </div>
    "#;
    let layout = compute_layout(html, (800, 600));

    let abs = &layout[1];
    assert_eq!(abs.position.x, layout[0].position.x + 10.0);
    assert_eq!(abs.position.y, layout[0].position.y + 10.0);
}
```

## Acceptance Criteria

- [ ] Block layout produces correct stacking
- [ ] Margin collapsing works correctly
- [ ] Percentage values resolve correctly
- [ ] Flexbox row and column layouts work
- [ ] Absolute/fixed positioning works
- [ ] Min/max constraints are applied
- [ ] GPU and CPU produce identical results
- [ ] Performance meets targets (5-20x speedup)

## Dependencies

- Issue #105: GPU Cascade Resolution

## Blocks

- Issue #107: GPU Text Measurement
- Issue #109: Benchmarking Framework
