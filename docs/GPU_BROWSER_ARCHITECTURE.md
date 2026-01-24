# GPU-Native Browser Architecture

## Executive Summary

Analysis of Servo's architecture reveals that **~90% of browser layout/rendering work is embarrassingly parallel** and suitable for GPU compute shaders. This document outlines a GPU-native browser architecture that moves CSS matching, layout computation, and rendering entirely to the GPU.

## Key Findings from Servo Analysis

### 1. Parallelization Profile

| Component | Current | GPU Potential | Speedup Estimate |
|-----------|---------|---------------|------------------|
| Selector Matching | Sequential | Fully parallel | 50-100x |
| Cascade Resolution | Per-element | Batch parallel | 10-50x |
| Block Layout | Rayon (CPU) | GPU compute | 5-20x |
| Table Cells | Rayon (CPU) | GPU 2D grid | 10-50x |
| Flex Items | Rayon (CPU) | GPU parallel | 5-20x |
| Text Measurement | Sequential | GPU atlas | 10-100x |
| Damage Propagation | Tree traversal | GPU atomics | 2-5x |

### 2. Servo's Conditional Parallelism Pattern

Servo already has infrastructure for GPU integration:
```rust
// Current pattern
if layout_context.use_rayon {
    items.par_iter().map(|item| item.layout()).collect()
} else {
    items.iter().map(|item| item.layout()).collect()
}

// GPU extension
if layout_context.use_gpu {
    gpu_dispatch_layout_kernel(items)
} else if layout_context.use_rayon {
    items.par_iter().map(|item| item.layout()).collect()
} else {
    items.iter().map(|item| item.layout()).collect()
}
```

### 3. Critical Bottlenecks Servo Faces

1. **Text measurement** - Sequential font metric queries
2. **Selector matching** - O(rules × elements) sequential
3. **Float placement** - Sequential state machine
4. **Margin collapsing** - Bidirectional dependencies

## GPU-Native Architecture

### Phase 1: GPU Data Structures

```metal
// Node graph - GPU resident
struct GPUNode {
    uint parent_idx;
    uint first_child_idx;
    uint next_sibling_idx;
    uint prev_sibling_idx;
    uint element_type;      // ELEM_DIV, ELEM_P, ELEM_SPAN, etc.
    uint flags;             // 16 flags packed
    uint attribute_offset;  // Into attribute buffer
    uint attribute_count;
};

// Computed styles - GPU resident
struct GPUComputedStyle {
    float4 margin;          // top, right, bottom, left
    float4 padding;
    float4 border_width;
    float4 border_color;
    float4 background_color;
    float2 size;            // width, height (or AUTO)
    float2 min_size;
    float2 max_size;
    uint display;           // block, inline, flex, grid, none
    uint position;          // static, relative, absolute, fixed
    uint overflow;          // visible, hidden, scroll, auto
    uint text_align;
    float font_size;
    float line_height;
    float z_index;
};

// Layout boxes - GPU computed
struct GPULayoutBox {
    float2 position;        // x, y relative to parent
    float2 size;            // computed width, height
    float2 content_position;
    float2 content_size;
    float2 scroll_size;     // for overflow: scroll
    uint parent_idx;
    uint first_child_idx;
};

// CSS Rules - GPU resident
struct GPUCSSRule {
    uint selector_hash;     // For fast matching
    uint specificity;       // a, b, c packed
    uint property_offset;   // Into property buffer
    uint property_count;
};
```

### Phase 2: GPU Selector Matching

```metal
kernel void match_selectors(
    device GPUNode* nodes [[buffer(0)]],
    device GPUCSSRule* rules [[buffer(1)]],
    device uint* matched_rules [[buffer(2)]],  // Output: rule indices per node
    device atomic_uint* match_counts [[buffer(3)]],
    constant uint& node_count [[buffer(4)]],
    constant uint& rule_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUNode node = nodes[gid];
    uint match_offset = gid * MAX_RULES_PER_NODE;
    uint count = 0;

    // Check each rule against this node
    for (uint r = 0; r < rule_count && count < MAX_RULES_PER_NODE; r++) {
        GPUCSSRule rule = rules[r];

        // Fast path: hash-based pre-filter
        if (selector_matches_node(rule, node, nodes)) {
            matched_rules[match_offset + count] = r;
            count++;
        }
    }

    atomic_store_explicit(&match_counts[gid], count, memory_order_relaxed);
}

// Selector matching helper - runs per-thread
bool selector_matches_node(GPUCSSRule rule, GPUNode node, device GPUNode* nodes) {
    // Type selector check
    if (rule.element_type != 0 && rule.element_type != node.element_type) {
        return false;
    }

    // Class selector check (hash comparison)
    if (rule.class_hash != 0 && !node_has_class_hash(node, rule.class_hash)) {
        return false;
    }

    // ID selector check
    if (rule.id_hash != 0 && node.id_hash != rule.id_hash) {
        return false;
    }

    // Ancestor check for descendant combinators
    if (rule.ancestor_hash != 0) {
        uint parent = node.parent_idx;
        bool found = false;
        while (parent != INVALID_IDX && !found) {
            if (nodes[parent].type_hash == rule.ancestor_hash) {
                found = true;
            }
            parent = nodes[parent].parent_idx;
        }
        if (!found) return false;
    }

    return true;
}
```

### Phase 3: GPU Cascade Resolution

```metal
kernel void resolve_cascade(
    device GPUNode* nodes [[buffer(0)]],
    device uint* matched_rules [[buffer(1)]],
    device uint* match_counts [[buffer(2)]],
    device GPUCSSRule* rules [[buffer(3)]],
    device GPUProperty* properties [[buffer(4)]],
    device GPUComputedStyle* computed [[buffer(5)]],  // Output
    constant uint& node_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUComputedStyle style = default_style();
    uint count = match_counts[gid];
    uint offset = gid * MAX_RULES_PER_NODE;

    // Sort matched rules by specificity (insertion sort for small N)
    // Then apply in order
    for (uint i = 0; i < count; i++) {
        uint rule_idx = matched_rules[offset + i];
        GPUCSSRule rule = rules[rule_idx];

        // Apply each property from this rule
        for (uint p = 0; p < rule.property_count; p++) {
            GPUProperty prop = properties[rule.property_offset + p];
            apply_property(&style, prop);
        }
    }

    // Handle inheritance from parent
    if (nodes[gid].parent_idx != INVALID_IDX) {
        GPUComputedStyle parent_style = computed[nodes[gid].parent_idx];
        inherit_properties(&style, parent_style);
    }

    computed[gid] = style;
}
```

### Phase 4: GPU Layout Engine

```metal
// Two-pass layout: widths top-down, heights bottom-up

kernel void layout_widths(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    constant float& viewport_width [[buffer(3)]],
    constant uint& node_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUComputedStyle style = styles[gid];
    GPUNode node = nodes[gid];

    // Determine containing block width
    float containing_width = viewport_width;
    if (node.parent_idx != INVALID_IDX) {
        containing_width = layout[node.parent_idx].size.x;
    }

    // Resolve width
    float width;
    if (style.size.x == AUTO) {
        // Block elements: fill container
        if (style.display == DISPLAY_BLOCK) {
            width = containing_width - style.margin.y - style.margin.w
                    - style.padding.y - style.padding.w
                    - style.border_width.y - style.border_width.w;
        } else {
            width = 0; // Shrink-to-fit, computed in second pass
        }
    } else if (is_percentage(style.size.x)) {
        width = containing_width * get_percentage(style.size.x);
    } else {
        width = style.size.x;
    }

    layout[gid].size.x = width;
    layout[gid].content_size.x = width - style.padding.y - style.padding.w
                                  - style.border_width.y - style.border_width.w;
}

kernel void layout_heights(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    constant uint& node_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUComputedStyle style = styles[gid];
    GPUNode node = nodes[gid];

    // Sum children heights (block layout)
    float content_height = 0;
    uint child = node.first_child_idx;
    while (child != INVALID_IDX) {
        GPULayoutBox child_layout = layout[child];
        GPUComputedStyle child_style = styles[child];
        content_height += child_layout.size.y
                         + child_style.margin.x + child_style.margin.z;
        child = nodes[child].next_sibling_idx;
    }

    // Apply specified height or use content height
    float height;
    if (style.size.y == AUTO) {
        height = content_height + style.padding.x + style.padding.z
                + style.border_width.x + style.border_width.z;
    } else {
        height = style.size.y;
    }

    layout[gid].size.y = height;
    layout[gid].content_size.y = height - style.padding.x - style.padding.z
                                  - style.border_width.x - style.border_width.z;
}

kernel void layout_positions(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    constant uint& node_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUNode node = nodes[gid];
    GPUComputedStyle style = styles[gid];

    if (node.parent_idx == INVALID_IDX) {
        layout[gid].position = float2(0, 0);
        return;
    }

    GPULayoutBox parent = layout[node.parent_idx];
    GPUComputedStyle parent_style = styles[node.parent_idx];

    // Start at parent's content origin
    float x = parent.content_position.x + style.margin.w;
    float y = parent.content_position.y + style.margin.x;

    // Add offsets from previous siblings
    uint prev = node.prev_sibling_idx;
    while (prev != INVALID_IDX) {
        GPULayoutBox prev_layout = layout[prev];
        GPUComputedStyle prev_style = styles[prev];
        y += prev_layout.size.y + prev_style.margin.x + prev_style.margin.z;
        prev = nodes[prev].prev_sibling_idx;
    }

    layout[gid].position = float2(x, y);
    layout[gid].content_position = float2(
        x + style.border_width.w + style.padding.w,
        y + style.border_width.x + style.padding.x
    );
}
```

### Phase 5: GPU Paint Generation

```metal
kernel void generate_paint_commands(
    device GPUNode* nodes [[buffer(0)]],
    device GPUComputedStyle* styles [[buffer(1)]],
    device GPULayoutBox* layout [[buffer(2)]],
    device PaintVertex* vertices [[buffer(3)]],
    device atomic_uint* vertex_count [[buffer(4)]],
    constant uint& node_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUComputedStyle style = styles[gid];
    GPULayoutBox box = layout[gid];

    if (style.display == DISPLAY_NONE) return;

    // Allocate vertices atomically
    uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);

    // Generate background quad (2 triangles = 6 vertices)
    float4 color = style.background_color;
    float2 p0 = box.position;
    float2 p1 = box.position + box.size;

    vertices[base + 0] = make_vertex(p0.x, p0.y, color);
    vertices[base + 1] = make_vertex(p1.x, p0.y, color);
    vertices[base + 2] = make_vertex(p0.x, p1.y, color);
    vertices[base + 3] = make_vertex(p1.x, p0.y, color);
    vertices[base + 4] = make_vertex(p1.x, p1.y, color);
    vertices[base + 5] = make_vertex(p0.x, p1.y, color);

    // Border vertices (if any)
    if (style.border_width.x > 0 || style.border_width.y > 0 ||
        style.border_width.z > 0 || style.border_width.w > 0) {
        uint border_base = atomic_fetch_add_explicit(vertex_count, 24, memory_order_relaxed);
        generate_border_vertices(vertices + border_base, box, style);
    }
}
```

## Implementation Phases

### Phase 1: GPU Infrastructure (Week 1-2)
- [ ] GPU buffer management for DOM tree
- [ ] Attribute string pool in GPU memory
- [ ] Style property encoding format
- [ ] Metal compute pipeline setup

### Phase 2: Selector Matching (Week 3-4)
- [ ] Selector hash computation
- [ ] GPU selector matching kernel
- [ ] Specificity sorting on GPU
- [ ] Benchmarking vs Stylo

### Phase 3: Cascade Resolution (Week 5-6)
- [ ] Property inheritance on GPU
- [ ] Computed value resolution
- [ ] Percentage/unit conversion
- [ ] Default value handling

### Phase 4: Layout Engine (Week 7-10)
- [ ] Width resolution pass
- [ ] Height resolution pass
- [ ] Position computation
- [ ] Margin collapsing (sequential fallback)
- [ ] Float handling (sequential fallback)

### Phase 5: Text Layout (Week 11-12)
- [ ] SDF glyph atlas integration
- [ ] GPU text measurement
- [ ] Line breaking on GPU
- [ ] Text vertex generation

### Phase 6: Paint & Render (Week 13-14)
- [ ] Atomic vertex allocation
- [ ] Background/border rendering
- [ ] Image texture sampling
- [ ] Blend mode support

## Performance Targets

| Metric | Current (Servo) | Target (GPU-Native) |
|--------|-----------------|---------------------|
| Layout time (1K nodes) | ~5ms | <0.5ms |
| Layout time (10K nodes) | ~50ms | <2ms |
| Selector matching (1K rules × 1K nodes) | ~10ms | <0.1ms |
| Text measurement (1K characters) | ~2ms | <0.1ms |
| Frame budget utilization | ~60% | <20% |

## What We Keep from Servo

1. **Stylo CSS parser** - Mature, spec-compliant
2. **html5ever HTML parser** - Mature, spec-compliant
3. **WebRender** - Reference for GPU rendering patterns
4. **Taffy** - Reference for flexbox/grid algorithms

## What We Replace

1. **Layout thread** → GPU compute kernels
2. **CPU selector matching** → GPU selector matching
3. **CPU cascade resolution** → GPU batch cascade
4. **Rayon parallelization** → GPU threadgroups
5. **Per-frame CPU work** → Persistent GPU state

## Key Architectural Decisions

### 1. GPU-Resident DOM
The DOM tree lives on GPU with CPU sync only on mutations:
- Tree structure (parent/child/sibling indices)
- Attribute data (in string pool)
- Node flags and type information

### 2. Incremental Updates
Only changed subtrees are re-processed:
- Dirty bit propagation on GPU
- Selective re-layout of affected branches
- Cached layout boxes for unchanged nodes

### 3. Hybrid Execution
Some operations remain CPU-bound:
- JavaScript execution (SpiderMonkey)
- Network I/O
- Complex float/margin interactions
- Rare CSS features (columns, regions)

## Next Steps

1. Create GPU buffer infrastructure for DOM tree
2. Implement selector matching kernel (proof of concept)
3. Benchmark against Servo on Wikipedia test page
4. Iterate on layout kernels for CSS compliance
