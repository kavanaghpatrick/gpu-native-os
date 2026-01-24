# Issue #105: GPU Cascade Resolution

## Summary
Implement GPU compute kernel for CSS cascade resolution - applying matched rules in specificity order to produce computed styles per element.

## Motivation
After selector matching (Issue #104), each element has a list of matching rules sorted by specificity. The cascade resolution phase applies these rules in order, handling inheritance and computing final property values. This is another O(nodes × properties) operation ideal for GPU parallelization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Cascade Resolution                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ MatchResults │  │ GPUProperty  │  │  GPUNode     │       │
│  │   Buffer     │  │   Buffer     │  │   Tree       │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│              ┌────────────▼─────────────────┐               │
│              │  resolve_cascade Kernel       │               │
│              │  (1 thread per node)          │               │
│              │                               │               │
│              │  For each matched rule:       │               │
│              │    Apply properties           │               │
│              │  Then inherit from parent     │               │
│              └────────────┬─────────────────┘               │
│                           │                                  │
│  Output:                  ▼                                  │
│  ┌──────────────────────────────────────────┐               │
│  │ GPUComputedStyle[node_count]             │               │
│  │ (all CSS properties resolved to values)  │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Structures

```metal
// Computed style - all properties resolved to final values
struct GPUComputedStyle {
    // Display and positioning
    uint display;           // block, inline, flex, grid, none
    uint position;          // static, relative, absolute, fixed
    float z_index;

    // Box model (all in pixels)
    float4 margin;          // top, right, bottom, left
    float4 padding;
    float4 border_width;

    // Sizing
    float width;            // NaN = auto
    float height;
    float min_width;
    float max_width;        // INFINITY = none
    float min_height;
    float max_height;

    // Colors (RGBA, 0-1 range)
    float4 color;
    float4 background_color;
    float4 border_color;

    // Text
    float font_size;
    float line_height;
    uint font_weight;       // 100-900
    uint text_align;        // left, center, right, justify

    // Flexbox
    uint flex_direction;    // row, column
    uint justify_content;
    uint align_items;
    float flex_grow;
    float flex_shrink;
    float flex_basis;       // NaN = auto
    float gap;

    // Visual
    float opacity;
    uint overflow_x;
    uint overflow_y;
    uint visibility;        // visible, hidden

    // Positioning offsets
    float top;              // NaN = auto
    float right_;
    float bottom;
    float left;

    // Flags
    uint flags;             // Bit flags for inherited, etc.
};

// Default/initial values
constant GPUComputedStyle DEFAULT_STYLE = {
    .display = DISPLAY_INLINE,  // Default for most elements
    .position = POSITION_STATIC,
    .z_index = 0,
    .margin = float4(0),
    .padding = float4(0),
    .border_width = float4(0),
    .width = NAN,               // auto
    .height = NAN,
    .min_width = 0,
    .max_width = INFINITY,
    .min_height = 0,
    .max_height = INFINITY,
    .color = float4(0, 0, 0, 1),       // Black
    .background_color = float4(0, 0, 0, 0), // Transparent
    .border_color = float4(0, 0, 0, 1),
    .font_size = 16.0,
    .line_height = 1.2,
    .font_weight = 400,
    .text_align = TEXT_ALIGN_LEFT,
    .flex_direction = FLEX_ROW,
    .justify_content = JUSTIFY_START,
    .align_items = ALIGN_STRETCH,
    .flex_grow = 0,
    .flex_shrink = 1,
    .flex_basis = NAN,
    .gap = 0,
    .opacity = 1.0,
    .overflow_x = OVERFLOW_VISIBLE,
    .overflow_y = OVERFLOW_VISIBLE,
    .visibility = VISIBILITY_VISIBLE,
    .top = NAN,
    .right_ = NAN,
    .bottom = NAN,
    .left = NAN,
    .flags = 0
};

// Properties that inherit by default
constant uint INHERITED_PROPERTIES = (
    (1 << PROP_COLOR) |
    (1 << PROP_FONT_SIZE) |
    (1 << PROP_FONT_WEIGHT) |
    (1 << PROP_LINE_HEIGHT) |
    (1 << PROP_TEXT_ALIGN) |
    (1 << PROP_VISIBILITY)
);
```

## Metal Kernel Implementation

```metal
// src/gpu_os/document/cascade.metal

#include <metal_stdlib>
using namespace metal;

// Apply a single property to computed style
void apply_property(
    thread GPUComputedStyle& style,
    GPUProperty prop,
    float containing_block_width,
    float containing_block_height,
    float parent_font_size
) {
    switch (prop.property_id) {
        case PROP_DISPLAY:
            style.display = prop.value;
            break;

        case PROP_POSITION:
            style.position = prop.value;
            break;

        case PROP_WIDTH:
            if (prop.value_type == VALUE_LENGTH) {
                style.width = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.width = containing_block_width * as_type<float>(prop.value);
            } else if (prop.value == KW_AUTO) {
                style.width = NAN;
            }
            break;

        case PROP_HEIGHT:
            if (prop.value_type == VALUE_LENGTH) {
                style.height = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.height = containing_block_height * as_type<float>(prop.value);
            } else if (prop.value == KW_AUTO) {
                style.height = NAN;
            }
            break;

        case PROP_MARGIN_TOP:
            if (prop.value_type == VALUE_LENGTH) {
                style.margin.x = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.margin.x = containing_block_width * as_type<float>(prop.value);
            }
            break;

        case PROP_MARGIN_RIGHT:
            if (prop.value_type == VALUE_LENGTH) {
                style.margin.y = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.margin.y = containing_block_width * as_type<float>(prop.value);
            }
            break;

        case PROP_MARGIN_BOTTOM:
            if (prop.value_type == VALUE_LENGTH) {
                style.margin.z = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.margin.z = containing_block_width * as_type<float>(prop.value);
            }
            break;

        case PROP_MARGIN_LEFT:
            if (prop.value_type == VALUE_LENGTH) {
                style.margin.w = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.margin.w = containing_block_width * as_type<float>(prop.value);
            }
            break;

        // Padding (similar to margin)
        case PROP_PADDING_TOP:
            style.padding.x = resolve_length_percentage(prop, containing_block_width);
            break;
        case PROP_PADDING_RIGHT:
            style.padding.y = resolve_length_percentage(prop, containing_block_width);
            break;
        case PROP_PADDING_BOTTOM:
            style.padding.z = resolve_length_percentage(prop, containing_block_width);
            break;
        case PROP_PADDING_LEFT:
            style.padding.w = resolve_length_percentage(prop, containing_block_width);
            break;

        // Colors
        case PROP_COLOR:
            style.color = get_color(prop);
            break;

        case PROP_BACKGROUND_COLOR:
            style.background_color = get_color(prop);
            break;

        case PROP_BORDER_COLOR:
            style.border_color = get_color(prop);
            break;

        // Border widths
        case PROP_BORDER_TOP_WIDTH:
            style.border_width.x = as_type<float>(prop.value);
            break;
        case PROP_BORDER_RIGHT_WIDTH:
            style.border_width.y = as_type<float>(prop.value);
            break;
        case PROP_BORDER_BOTTOM_WIDTH:
            style.border_width.z = as_type<float>(prop.value);
            break;
        case PROP_BORDER_LEFT_WIDTH:
            style.border_width.w = as_type<float>(prop.value);
            break;

        // Text properties
        case PROP_FONT_SIZE:
            if (prop.value_type == VALUE_LENGTH) {
                style.font_size = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.font_size = parent_font_size * as_type<float>(prop.value);
            }
            break;

        case PROP_LINE_HEIGHT:
            if (prop.value_type == VALUE_LENGTH) {
                style.line_height = as_type<float>(prop.value);
            } else if (prop.value_type == VALUE_PERCENTAGE) {
                style.line_height = style.font_size * as_type<float>(prop.value);
            } else {
                // Unitless number
                style.line_height = style.font_size * as_type<float>(prop.value);
            }
            break;

        case PROP_FONT_WEIGHT:
            if (prop.value == KW_BOLD) {
                style.font_weight = 700;
            } else if (prop.value == KW_NORMAL) {
                style.font_weight = 400;
            } else {
                style.font_weight = prop.value;
            }
            break;

        case PROP_TEXT_ALIGN:
            style.text_align = prop.value;
            break;

        // Flexbox
        case PROP_FLEX_DIRECTION:
            style.flex_direction = prop.value;
            break;

        case PROP_JUSTIFY_CONTENT:
            style.justify_content = prop.value;
            break;

        case PROP_ALIGN_ITEMS:
            style.align_items = prop.value;
            break;

        case PROP_FLEX_GROW:
            style.flex_grow = as_type<float>(prop.value);
            break;

        case PROP_FLEX_SHRINK:
            style.flex_shrink = as_type<float>(prop.value);
            break;

        case PROP_GAP:
            style.gap = as_type<float>(prop.value);
            break;

        // Positioning
        case PROP_TOP:
            style.top = resolve_length_percentage_auto(prop, containing_block_height);
            break;
        case PROP_RIGHT:
            style.right_ = resolve_length_percentage_auto(prop, containing_block_width);
            break;
        case PROP_BOTTOM:
            style.bottom = resolve_length_percentage_auto(prop, containing_block_height);
            break;
        case PROP_LEFT:
            style.left = resolve_length_percentage_auto(prop, containing_block_width);
            break;

        // Visual
        case PROP_OPACITY:
            style.opacity = as_type<float>(prop.value);
            break;

        case PROP_OVERFLOW:
            style.overflow_x = prop.value;
            style.overflow_y = prop.value;
            break;

        case PROP_VISIBILITY:
            style.visibility = prop.value;
            break;

        case PROP_Z_INDEX:
            style.z_index = as_type<float>(prop.value);
            break;
    }
}

// Inherit specified properties from parent
void inherit_from_parent(
    thread GPUComputedStyle& style,
    GPUComputedStyle parent
) {
    // Inherit color
    if (isnan(as_type<float>(style.flags & FLAG_COLOR_SET))) {
        style.color = parent.color;
    }

    // Inherit font-size
    if (style.font_size == 0) {
        style.font_size = parent.font_size;
    }

    // Inherit font-weight
    if (style.font_weight == 0) {
        style.font_weight = parent.font_weight;
    }

    // Inherit line-height
    if (style.line_height == 0) {
        style.line_height = parent.line_height;
    }

    // Inherit text-align
    if (style.text_align == 0) {
        style.text_align = parent.text_align;
    }

    // Inherit visibility
    if (style.visibility == 0) {
        style.visibility = parent.visibility;
    }
}

// Apply user-agent defaults based on element type
GPUComputedStyle get_ua_defaults(ushort element_type) {
    GPUComputedStyle style = DEFAULT_STYLE;

    switch (element_type) {
        case ELEM_DIV:
        case ELEM_P:
        case ELEM_H1:
        case ELEM_H2:
        case ELEM_H3:
        case ELEM_H4:
        case ELEM_H5:
        case ELEM_H6:
        case ELEM_UL:
        case ELEM_OL:
        case ELEM_LI:
        case ELEM_HEADER:
        case ELEM_FOOTER:
        case ELEM_ARTICLE:
        case ELEM_SECTION:
        case ELEM_NAV:
        case ELEM_MAIN:
        case ELEM_ASIDE:
        case ELEM_FORM:
            style.display = DISPLAY_BLOCK;
            break;

        case ELEM_SPAN:
        case ELEM_A:
        case ELEM_STRONG:
        case ELEM_EM:
        case ELEM_CODE:
            style.display = DISPLAY_INLINE;
            break;

        case ELEM_H1:
            style.font_size = 32.0;
            style.font_weight = 700;
            style.margin = float4(21.44, 0, 21.44, 0);
            break;

        case ELEM_H2:
            style.font_size = 24.0;
            style.font_weight = 700;
            style.margin = float4(19.92, 0, 19.92, 0);
            break;

        case ELEM_H3:
            style.font_size = 18.72;
            style.font_weight = 700;
            style.margin = float4(18.72, 0, 18.72, 0);
            break;

        case ELEM_P:
            style.margin = float4(16, 0, 16, 0);
            break;

        case ELEM_UL:
        case ELEM_OL:
            style.margin = float4(16, 0, 16, 0);
            style.padding.w = 40; // padding-left
            break;

        case ELEM_A:
            style.color = float4(0, 0, 0.8, 1); // Blue
            break;

        case ELEM_STRONG:
            style.font_weight = 700;
            break;

        case ELEM_EM:
            // Would need font-style: italic
            break;

        case ELEM_TABLE:
            style.display = DISPLAY_TABLE;
            break;

        case ELEM_TR:
            style.display = DISPLAY_TABLE_ROW;
            break;

        case ELEM_TD:
        case ELEM_TH:
            style.display = DISPLAY_TABLE_CELL;
            break;
    }

    return style;
}

// Main cascade resolution kernel
kernel void resolve_cascade(
    device GPUNode* nodes [[buffer(0)]],
    device MatchResult* matched_rules [[buffer(1)]],
    device uint* match_counts [[buffer(2)]],
    device GPURule* rules [[buffer(3)]],
    device GPUProperty* properties [[buffer(4)]],
    device GPUComputedStyle* computed [[buffer(5)]],
    constant uint& node_count [[buffer(6)]],
    constant float2& viewport_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUNode node = nodes[gid];

    // Start with UA defaults for this element type
    GPUComputedStyle style = get_ua_defaults(node.element_type);

    // Get containing block dimensions
    float containing_width = viewport_size.x;
    float containing_height = viewport_size.y;
    float parent_font_size = 16.0;

    if (node.parent_idx != INVALID_IDX) {
        // Note: This creates a data dependency - parent must be computed first
        // We handle this with multiple kernel passes or topological ordering
        GPUComputedStyle parent = computed[node.parent_idx];
        if (!isnan(parent.width)) {
            containing_width = parent.width;
        }
        if (!isnan(parent.height)) {
            containing_height = parent.height;
        }
        parent_font_size = parent.font_size;
    }

    // Apply matched rules in specificity order (already sorted)
    uint count = match_counts[gid];
    uint offset = gid * MAX_MATCHES_PER_NODE;

    for (uint i = 0; i < count; i++) {
        MatchResult match = matched_rules[offset + i];
        GPURule rule = rules[match.rule_index];

        // Apply each property from this rule
        for (uint p = 0; p < rule.property_count; p++) {
            GPUProperty prop = properties[rule.property_offset + p];
            apply_property(style, prop, containing_width, containing_height, parent_font_size);
        }
    }

    // Inherit from parent for inherited properties
    if (node.parent_idx != INVALID_IDX) {
        GPUComputedStyle parent = computed[node.parent_idx];
        inherit_from_parent(style, parent);
    }

    computed[gid] = style;
}

// Multi-pass cascade for handling parent dependencies
// Pass 1: Process root and depth-0 elements
// Pass 2: Process depth-1 elements
// ... etc.
kernel void resolve_cascade_by_depth(
    device GPUNode* nodes [[buffer(0)]],
    device MatchResult* matched_rules [[buffer(1)]],
    device uint* match_counts [[buffer(2)]],
    device GPURule* rules [[buffer(3)]],
    device GPUProperty* properties [[buffer(4)]],
    device GPUComputedStyle* computed [[buffer(5)]],
    device uint* node_depths [[buffer(6)]],
    constant uint& node_count [[buffer(7)]],
    constant uint& current_depth [[buffer(8)]],
    constant float2& viewport_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;
    if (node_depths[gid] != current_depth) return;

    // ... same logic as above, but parent is guaranteed to be computed
}
```

## Rust Integration

```rust
// src/gpu_os/document/cascade.rs

pub struct GPUCascadeResolver {
    device: Device,
    pipeline: ComputePipelineState,
    depth_pipeline: ComputePipelineState,
    computed_styles_buffer: Buffer,
}

impl GPUCascadeResolver {
    pub fn resolve(
        &self,
        command_buffer: &CommandBufferRef,
        gpu_dom: &GPUDom,
        matcher: &GPUSelectorMatcher,
        gpu_css: &GPUCSS,
        viewport_size: (f32, f32),
    ) {
        // Compute max depth
        let max_depth = gpu_dom.compute_max_depth();

        let encoder = command_buffer.new_compute_command_encoder();

        // Process each depth level
        for depth in 0..=max_depth {
            encoder.set_compute_pipeline_state(&self.depth_pipeline);
            encoder.set_buffer(0, Some(&gpu_dom.node_buffer), 0);
            encoder.set_buffer(1, Some(&matcher.matched_rules_buffer), 0);
            encoder.set_buffer(2, Some(&matcher.match_counts_buffer), 0);
            encoder.set_buffer(3, Some(&gpu_css.rule_buffer), 0);
            encoder.set_buffer(4, Some(&gpu_css.property_buffer), 0);
            encoder.set_buffer(5, Some(&self.computed_styles_buffer), 0);
            encoder.set_buffer(6, Some(&gpu_dom.depth_buffer), 0);
            encoder.set_bytes(7, 4, &gpu_dom.node_count as *const _ as *const _);
            encoder.set_bytes(8, 4, &depth as *const _ as *const _);
            encoder.set_bytes(9, 8, &viewport_size as *const _ as *const _);

            let threads = MTLSize::new(gpu_dom.node_count as u64, 1, 1);
            let threadgroup = MTLSize::new(256, 1, 1);
            encoder.dispatch_threads(threads, threadgroup);

            // Barrier between depth levels
            encoder.memory_barrier_with_resources(&[&self.computed_styles_buffer]);
        }

        encoder.end_encoding();
    }

    pub fn get_computed_style(&self, node_idx: u32) -> GPUComputedStyle {
        let styles = unsafe {
            std::slice::from_raw_parts(
                self.computed_styles_buffer.contents() as *const GPUComputedStyle,
                self.computed_styles_buffer.length() as usize / std::mem::size_of::<GPUComputedStyle>(),
            )
        };
        styles[node_idx as usize]
    }
}
```

## Benchmarks

```rust
fn bench_cascade_resolution(c: &mut Criterion) {
    let device = Device::system_default().unwrap();

    let test_cases = vec![
        ("small", "test_pages/small.html", "test_css/simple.css"),
        ("wikipedia", "test_pages/wikipedia.html", "test_css/wikipedia.css"),
    ];

    let mut group = c.benchmark_group("cascade_resolution");

    for (name, html_path, css_path) in test_cases {
        // Setup...

        group.bench_with_input(
            BenchmarkId::new("cpu", name),
            &(),
            |b, _| b.iter(|| cpu_resolve_cascade(&doc, &matched))
        );

        group.bench_with_input(
            BenchmarkId::new("gpu", name),
            &(),
            |b, _| {
                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    resolver.resolve(&cmd, &gpu_dom, &matcher, &gpu_css, viewport);
                    cmd.commit();
                    cmd.wait_until_completed();
                })
            }
        );
    }
}
```

### Expected Results

| Page | Nodes | Matched Rules | CPU Time | GPU Time | Speedup |
|------|-------|---------------|----------|----------|---------|
| Small | 100 | 500 | 1ms | 0.1ms | 10x |
| Wikipedia | 5000 | 25000 | 50ms | 1ms | 50x |

## Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_display_block() {
        let html = "<div></div>";
        let css = "div { display: block; }";
        let styles = resolve(&html, &css);

        assert_eq!(styles[0].display, DISPLAY_BLOCK);
    }

    #[test]
    fn test_inheritance() {
        let html = "<div><span></span></div>";
        let css = "div { color: red; }";
        let styles = resolve(&html, &css);

        // div has explicit color
        assert_eq!(styles[0].color, Color::RED);
        // span inherits color
        assert_eq!(styles[1].color, Color::RED);
    }

    #[test]
    fn test_specificity_override() {
        let html = r#"<div class="foo" id="bar"></div>"#;
        let css = r#"
            div { color: red; }
            .foo { color: blue; }
            #bar { color: green; }
        "#;
        let styles = resolve(&html, &css);

        // #bar wins (highest specificity)
        assert_eq!(styles[0].color, Color::GREEN);
    }

    #[test]
    fn test_ua_defaults() {
        let html = "<h1>Test</h1>";
        let css = "";  // No author styles
        let styles = resolve(&html, &css);

        assert_eq!(styles[0].display, DISPLAY_BLOCK);
        assert_eq!(styles[0].font_size, 32.0);
        assert_eq!(styles[0].font_weight, 700);
    }

    #[test]
    fn test_percentage_resolution() {
        let html = "<div><span></span></div>";
        let css = "div { width: 100px; } span { width: 50%; }";
        let styles = resolve(&html, &css);

        assert_eq!(styles[0].width, 100.0);
        assert_eq!(styles[1].width, 50.0); // 50% of 100px
    }

    #[test]
    fn test_important() {
        let html = "<div></div>";
        let css = r#"
            div { color: red !important; }
            div { color: blue; }
        "#;
        let styles = resolve(&html, &css);

        // !important wins
        assert_eq!(styles[0].color, Color::RED);
    }
}
```

## Acceptance Criteria

- [ ] All CSS properties correctly applied
- [ ] Inheritance works for inherited properties
- [ ] Specificity ordering respected
- [ ] !important declarations win
- [ ] Percentage values resolved against containing block
- [ ] UA defaults applied for all common elements
- [ ] Multi-pass depth handling correct
- [ ] GPU matches CPU reference implementation

## Dependencies

- Issue #102: HTML5 Parser
- Issue #103: CSS Parser
- Issue #104: GPU Selector Matching

## Blocks

- Issue #106: GPU Layout Engine
