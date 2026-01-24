# Issue #99: Block Layout Rewrite (CSS 2.1 Compliant)

## Summary
Complete rewrite of block layout to implement CSS 2.1 Visual Formatting Model correctly. This is the critical fix for the whitespace issues.

## Problem

Current layout has fundamental issues:
1. Heights calculated incorrectly (150,000px instead of ~5,000px)
2. Margin collapsing not implemented correctly
3. Out-of-flow elements included in height calculations
4. Text height calculated before parent width is known

## CSS 2.1 Block Layout Algorithm

### Width Calculation (Section 10.3.3)

For block-level elements in normal flow:
```
margin-left + border-left + padding-left + width + padding-right + border-right + margin-right = containing block width

If width is auto:
    width = containing_block_width - margins - borders - padding

If margins are auto:
    If width is constrained, auto margins split remaining space
```

### Height Calculation (Section 10.6.3)

For block-level elements with `height: auto`:
```
If element has only block-level children:
    height = distance from top border-edge of topmost block-level child
             to bottom border-edge of bottommost block-level child
             + padding-top + padding-bottom
             + collapsed margins (if applicable)

If element establishes a new BFC:
    height includes floating descendants
```

### Margin Collapsing (Section 8.3.1)

Adjacent vertical margins collapse into a single margin:
```
collapsed_margin = max(margin1, margin2)  // Both positive
                 = min(margin1, margin2)  // Both negative
                 = margin1 + margin2      // Mixed signs
```

Margins collapse when:
- Top margin of box with top margin of first in-flow child
- Bottom margin of box with bottom margin of last in-flow child (if height:auto)
- Bottom margin of box with top margin of next in-flow sibling
- Top and bottom margins of empty box

## Data Structures

```rust
#[repr(C)]
pub struct LayoutBox {
    // Border box position (relative to containing block)
    pub x: f32,
    pub y: f32,

    // Border box dimensions
    pub width: f32,
    pub height: f32,

    // Content box (computed)
    pub content_x: f32,
    pub content_y: f32,
    pub content_width: f32,
    pub content_height: f32,

    // Margin info (for collapsing)
    pub margin_top: f32,           // Actual margin after collapsing
    pub margin_bottom: f32,
    pub margin_top_collapsed: f32,  // Collapsed margin propagated up
    pub margin_bottom_collapsed: f32,

    // Flags
    pub is_empty: u8,              // For margin collapsing
    pub margins_collapsed_through: u8,  // Top and bottom collapsed together
    pub _padding: [u8; 6],
}
```

## GPU Kernels

```metal
// =============================================================================
// PASS 1: Compute containing block widths (top-down by level)
// =============================================================================
kernel void compute_containing_block_widths(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device const ElementFormattingInfo* fc_info [[buffer(2)]],
    device LayoutBox* boxes [[buffer(3)]],
    device const uint* depths [[buffer(4)]],
    constant uint& current_level [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    constant float& viewport_width [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = styles[gid];
    LayoutBox box;

    // Initialize box
    box.x = 0;
    box.y = 0;
    box.margin_top = style.margin_top;
    box.margin_bottom = style.margin_bottom;
    box.margin_top_collapsed = style.margin_top;
    box.margin_bottom_collapsed = style.margin_bottom;
    box.is_empty = 0;
    box.margins_collapsed_through = 0;

    // Skip hidden elements
    if (style.display == DISPLAY_NONE) {
        box.width = 0;
        box.height = 0;
        boxes[gid] = box;
        return;
    }

    // Get containing block width
    float cb_width = viewport_width;
    if (elem.parent >= 0) {
        cb_width = boxes[elem.parent].content_width;
    }

    // Calculate width
    float margin_h = style.margin_left + style.margin_right;
    float border_h = style.border_left_width + style.border_right_width;
    float padding_h = style.padding_left + style.padding_right;

    if (style.width > 0) {
        // Explicit width
        box.width = style.width + border_h + padding_h;
    } else {
        // Auto width - fill containing block
        if (fc_info[gid].participates_in == FC_BLOCK) {
            box.width = cb_width - margin_h;
        } else {
            // Inline elements - shrink to fit (handled later)
            box.width = 0;
        }
    }

    // Content box width
    box.content_width = box.width - border_h - padding_h;
    if (box.content_width < 0) box.content_width = 0;

    // X position (left margin)
    box.x = style.margin_left;
    box.content_x = box.x + style.border_left_width + style.padding_left;

    boxes[gid] = box;
}

// =============================================================================
// PASS 2: Calculate text heights (now that widths are known)
// =============================================================================
kernel void compute_text_heights(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint8_t* text_buffer [[buffer(3)]],
    constant uint& element_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT) return;
    if (elem.text_length == 0) return;

    ComputedStyleCompact style = styles[gid];
    if (style.display == DISPLAY_NONE) return;

    // Get container width from parent
    float container_width = 10000.0;  // Fallback
    if (elem.parent >= 0) {
        container_width = boxes[elem.parent].content_width;
    }
    if (container_width <= 0) container_width = 10000.0;

    // Check for whitespace-only text
    bool has_content = false;
    for (uint i = 0; i < elem.text_length; i++) {
        uint8_t c = text_buffer[elem.text_start + i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            has_content = true;
            break;
        }
    }

    if (!has_content) {
        boxes[gid].height = 0;
        boxes[gid].content_height = 0;
        boxes[gid].is_empty = 1;
        return;
    }

    // Calculate wrapped text height
    float font_size = style.font_size > 0 ? style.font_size : 16.0;
    float line_height = style.line_height > 0 ? style.line_height : 1.2;
    float line_height_px = font_size * line_height;

    uint line_count = 1;
    float current_width = 0.0;

    for (uint i = 0; i < elem.text_length; i++) {
        uint8_t c = text_buffer[elem.text_start + i];

        if (c == '\n') {
            line_count++;
            current_width = 0;
            continue;
        }

        // Glyph advance (approximate)
        float advance;
        if (c == ' ') advance = font_size * 0.3;
        else if (c == '\t') advance = font_size * 1.2;
        else advance = font_size * 0.6;

        current_width += advance;

        // Line wrap
        if (current_width > container_width) {
            line_count++;
            current_width = advance;
        }
    }

    float height = float(line_count) * line_height_px;
    boxes[gid].height = height;
    boxes[gid].content_height = height;
    boxes[gid].width = container_width;
    boxes[gid].content_width = container_width;
}

// =============================================================================
// PASS 3: Compute block heights (bottom-up by level)
// CSS 2.1 Section 10.6.3 with proper margin collapsing
// =============================================================================
kernel void compute_block_heights(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device const ElementFormattingInfo* fc_info [[buffer(2)]],
    device LayoutBox* boxes [[buffer(3)]],
    device const uint* depths [[buffer(4)]],
    constant uint& current_level [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = styles[gid];

    // Skip hidden, text nodes, and elements with explicit height
    if (style.display == DISPLAY_NONE) return;
    if (elem.element_type == ELEM_TEXT) return;
    if (style.height > 0) {
        boxes[gid].height = style.height +
            style.padding_top + style.padding_bottom +
            style.border_top_width + style.border_bottom_width;
        boxes[gid].content_height = style.height;
        return;
    }

    // No children = empty box
    if (elem.first_child < 0) {
        boxes[gid].height = style.padding_top + style.padding_bottom +
                          style.border_top_width + style.border_bottom_width;
        boxes[gid].content_height = 0;
        boxes[gid].is_empty = 1;
        return;
    }

    // Collect in-flow children heights with margin collapsing
    float content_height = 0;
    float prev_margin_bottom = 0;
    bool is_first = true;
    bool has_content = false;

    // Track margins for collapsing with parent
    float first_child_margin_top = 0;
    float last_child_margin_bottom = 0;

    int child = elem.first_child;
    while (child >= 0) {
        ComputedStyleCompact child_style = styles[child];
        ElementFormattingInfo child_fc = fc_info[child];
        LayoutBox child_box = boxes[child];

        // Skip non-participating elements (hidden, out-of-flow)
        if (child_style.display == DISPLAY_NONE ||
            child_fc.participates_in == 0) {
            child = elements[child].next_sibling;
            continue;
        }

        float child_margin_top = child_box.margin_top_collapsed;
        float child_margin_bottom = child_box.margin_bottom_collapsed;

        // Empty children - margins collapse through
        if (child_box.is_empty && child_box.margins_collapsed_through) {
            float collapsed = collapse_margins(child_margin_top, child_margin_bottom);
            if (is_first) {
                first_child_margin_top = collapsed;
            }
            prev_margin_bottom = collapse_margins(prev_margin_bottom, collapsed);
            child = elements[child].next_sibling;
            continue;
        }

        // Non-empty child
        has_content = true;

        if (is_first) {
            first_child_margin_top = child_margin_top;
            is_first = false;
        } else {
            // Collapse adjacent margins
            float collapsed = collapse_margins(prev_margin_bottom, child_margin_top);
            content_height += collapsed;
        }

        content_height += child_box.height;
        prev_margin_bottom = child_margin_bottom;
        last_child_margin_bottom = child_margin_bottom;

        child = elements[child].next_sibling;
    }

    // Determine if this box establishes a BFC (affects margin collapsing)
    bool establishes_bfc = fc_info[gid].is_bfc_root;
    bool has_top_border_padding = (style.border_top_width > 0 || style.padding_top > 0);
    bool has_bottom_border_padding = (style.border_bottom_width > 0 || style.padding_bottom > 0);

    // Margin collapsing with first child
    if (!establishes_bfc && !has_top_border_padding && has_content) {
        // Parent's top margin collapses with first child's top margin
        boxes[gid].margin_top_collapsed = collapse_margins(style.margin_top, first_child_margin_top);
    } else {
        boxes[gid].margin_top_collapsed = style.margin_top;
        if (has_content && !is_first) {
            content_height += first_child_margin_top;
        }
    }

    // Margin collapsing with last child
    if (!establishes_bfc && !has_bottom_border_padding && has_content) {
        // Parent's bottom margin collapses with last child's bottom margin
        boxes[gid].margin_bottom_collapsed = collapse_margins(style.margin_bottom, last_child_margin_bottom);
    } else {
        boxes[gid].margin_bottom_collapsed = style.margin_bottom;
        if (has_content) {
            content_height += last_child_margin_bottom;
        }
    }

    // Empty box handling
    if (!has_content) {
        boxes[gid].is_empty = 1;
        if (!has_top_border_padding && !has_bottom_border_padding) {
            boxes[gid].margins_collapsed_through = 1;
        }
    }

    // Final height
    boxes[gid].content_height = content_height;
    boxes[gid].height = content_height +
        style.padding_top + style.padding_bottom +
        style.border_top_width + style.border_bottom_width;
}

// Helper: Collapse two margins according to CSS 2.1
float collapse_margins(float m1, float m2) {
    if (m1 >= 0 && m2 >= 0) return max(m1, m2);
    if (m1 < 0 && m2 < 0) return min(m1, m2);  // More negative
    return m1 + m2;  // Mixed signs
}

// =============================================================================
// PASS 4: Position children (top-down by level)
// =============================================================================
kernel void position_children(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device const ElementFormattingInfo* fc_info [[buffer(2)]],
    device LayoutBox* boxes [[buffer(3)]],
    device const uint* depths [[buffer(4)]],
    constant uint& current_level [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = styles[gid];
    ElementFormattingInfo info = fc_info[gid];

    if (style.display == DISPLAY_NONE) return;

    // Out-of-flow elements positioned separately
    if (info.participates_in == 0) {
        // TODO: Absolute/fixed positioning
        return;
    }

    // Get parent info
    if (elem.parent < 0) {
        // Root element
        boxes[gid].y = style.margin_top;
        return;
    }

    ComputedStyleCompact parent_style = styles[elem.parent];
    ElementFormattingInfo parent_fc = fc_info[elem.parent];

    // Calculate Y position by walking siblings
    float y = 0;
    float prev_margin_bottom = 0;
    bool found_visible_sibling = false;

    int sib = elements[elem.parent].first_child;
    while (sib >= 0 && sib != int(gid)) {
        ComputedStyleCompact sib_style = styles[sib];
        ElementFormattingInfo sib_fc = fc_info[sib];
        LayoutBox sib_box = boxes[sib];

        // Skip non-participating siblings
        if (sib_style.display == DISPLAY_NONE || sib_fc.participates_in == 0) {
            sib = elements[sib].next_sibling;
            continue;
        }

        // Empty siblings with collapsed-through margins
        if (sib_box.is_empty && sib_box.margins_collapsed_through) {
            prev_margin_bottom = collapse_margins(
                prev_margin_bottom,
                collapse_margins(sib_box.margin_top_collapsed, sib_box.margin_bottom_collapsed)
            );
            sib = elements[sib].next_sibling;
            continue;
        }

        // Non-empty sibling
        if (!found_visible_sibling) {
            y += sib_box.margin_top_collapsed;
            found_visible_sibling = true;
        } else {
            y += collapse_margins(prev_margin_bottom, sib_box.margin_top_collapsed);
        }

        y += sib_box.height;
        prev_margin_bottom = sib_box.margin_bottom_collapsed;

        sib = elements[sib].next_sibling;
    }

    // Add margin between last sibling and this element
    LayoutBox my_box = boxes[gid];
    if (found_visible_sibling) {
        y += collapse_margins(prev_margin_bottom, my_box.margin_top_collapsed);
    } else {
        // First child - check for margin collapsing with parent
        bool parent_has_top_border_padding =
            parent_style.border_top_width > 0 || parent_style.padding_top > 0;

        if (!parent_fc.is_bfc_root && !parent_has_top_border_padding) {
            // Margin collapsed with parent - don't add it
            y = 0;
        } else {
            y = my_box.margin_top_collapsed;
        }
    }

    boxes[gid].y = y;
    boxes[gid].content_y = y + style.border_top_width + style.padding_top;
}

// =============================================================================
// PASS 5: Convert to absolute positions (top-down by level)
// =============================================================================
kernel void finalize_positions(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    constant uint& current_level [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = styles[gid];

    if (elem.parent < 0) return;  // Root already absolute

    LayoutBox parent_box = boxes[elem.parent];

    // Add parent's content box position
    boxes[gid].x += parent_box.content_x;
    boxes[gid].y += parent_box.content_y;
    boxes[gid].content_x += parent_box.content_x;
    boxes[gid].content_y += parent_box.content_y;
}
```

## Pseudocode

```
FUNCTION compute_block_layout(elements, styles, fc_info, text_buffer, viewport):
    // Compute tree depths
    depths = compute_depths(elements)
    max_depth = max(depths)

    // Initialize boxes
    boxes = allocate(element_count, sizeof(LayoutBox))

    // PASS 1: Widths (top-down)
    FOR level = 0 TO max_depth:
        gpu_dispatch(compute_containing_block_widths,
            elements, styles, fc_info, boxes, depths, level, viewport.width)

    // PASS 2: Text heights
    gpu_dispatch(compute_text_heights,
        elements, styles, boxes, text_buffer)

    // PASS 3: Block heights (bottom-up)
    FOR level = max_depth DOWNTO 0:
        gpu_dispatch(compute_block_heights,
            elements, styles, fc_info, boxes, depths, level)

    // PASS 4: Positions (top-down)
    FOR level = 0 TO max_depth:
        gpu_dispatch(position_children,
            elements, styles, fc_info, boxes, depths, level)

    // PASS 5: Absolute positions (top-down)
    FOR level = 1 TO max_depth:  // Skip root
        gpu_dispatch(finalize_positions,
            elements, styles, boxes, depths, level)

    RETURN boxes
```

## Tests

### Test 1: Simple block stacking
```rust
#[test]
fn test_simple_block_stacking() {
    let html = r#"
        <div style="width: 100px; height: 50px"></div>
        <div style="width: 100px; height: 50px"></div>
    "#;
    let boxes = compute_layout(html);

    assert_eq!(boxes[0].y, 0.0);
    assert_eq!(boxes[0].height, 50.0);
    assert_eq!(boxes[1].y, 50.0);  // Immediately after first
    assert_eq!(boxes[1].height, 50.0);
}
```

### Test 2: Margin collapsing between siblings
```rust
#[test]
fn test_sibling_margin_collapsing() {
    let html = r#"
        <div style="margin-bottom: 20px; height: 50px"></div>
        <div style="margin-top: 30px; height: 50px"></div>
    "#;
    let boxes = compute_layout(html);

    // Collapsed margin should be max(20, 30) = 30
    assert_eq!(boxes[0].y, 0.0);
    assert_eq!(boxes[0].height, 50.0);
    assert_eq!(boxes[1].y, 80.0);  // 50 + 30 (collapsed margin)
}
```

### Test 3: Parent-child margin collapsing
```rust
#[test]
fn test_parent_child_margin_collapsing() {
    let html = r#"
        <div style="margin-top: 20px">
            <div style="margin-top: 30px; height: 50px"></div>
        </div>
    "#;
    let boxes = compute_layout(html);

    let parent = find_element_by_depth(&boxes, 0);
    let child = find_element_by_depth(&boxes, 1);

    // Parent and child top margins collapse: max(20, 30) = 30
    assert_eq!(parent.y, 30.0);  // Collapsed margin
    assert_eq!(child.y, 30.0);   // Same as parent (no additional margin)
}
```

### Test 4: Border prevents margin collapsing
```rust
#[test]
fn test_border_prevents_collapsing() {
    let html = r#"
        <div style="margin-top: 20px; border-top: 1px solid black">
            <div style="margin-top: 30px; height: 50px"></div>
        </div>
    "#;
    let boxes = compute_layout(html);

    let parent = &boxes[0];
    let child = &boxes[1];

    // Border prevents collapsing - both margins apply
    assert_eq!(parent.y, 20.0);          // Parent's margin
    assert_eq!(child.y, 20.0 + 1.0 + 30.0);  // Parent y + border + child margin
}
```

### Test 5: Auto height from children
```rust
#[test]
fn test_auto_height_from_children() {
    let html = r#"
        <div>
            <div style="height: 50px"></div>
            <div style="height: 30px"></div>
        </div>
    "#;
    let boxes = compute_layout(html);

    let parent = &boxes[0];
    assert_eq!(parent.content_height, 80.0);  // 50 + 30
}
```

### Test 6: Out-of-flow excluded from height
```rust
#[test]
fn test_absolute_excluded_from_height() {
    let html = r#"
        <div style="position: relative">
            <div style="height: 50px"></div>
            <div style="position: absolute; height: 1000px"></div>
            <div style="height: 30px"></div>
        </div>
    "#;
    let boxes = compute_layout(html);

    let parent = &boxes[0];
    // Height should NOT include the absolute element (1000px)
    assert_eq!(parent.content_height, 80.0);  // 50 + 30 only
}
```

### Test 7: Empty element margin collapsing
```rust
#[test]
fn test_empty_element_margin_collapsing() {
    let html = r#"
        <div style="margin-bottom: 20px; height: 50px"></div>
        <div style="margin-top: 10px; margin-bottom: 15px"></div>
        <div style="margin-top: 25px; height: 30px"></div>
    "#;
    let boxes = compute_layout(html);

    // Empty middle element: its margins collapse through
    // Collapsed margin = max(20, 10, 15, 25) = 25
    assert_eq!(boxes[2].y, 75.0);  // 50 + 25
}
```

### Test 8: Wikipedia layout
```rust
#[test]
fn test_wikipedia_layout() {
    let html = include_str!("../testdata/wikipedia.html");
    let css = include_str!("../testdata/wikipedia.css");

    let result = process_document(html, css);

    // Root should have reasonable height (not 150,000px)
    let root = &result.boxes[0];
    assert!(root.height < 50000.0, "Root height {} is unreasonable", root.height);

    // No overlapping visible text elements
    for i in 0..result.elements.len() {
        for j in i+1..result.elements.len() {
            if result.styles[i].display == DISPLAY_NONE { continue; }
            if result.styles[j].display == DISPLAY_NONE { continue; }
            if result.elements[i].element_type != ELEM_TEXT { continue; }
            if result.elements[j].element_type != ELEM_TEXT { continue; }

            let box_i = &result.boxes[i];
            let box_j = &result.boxes[j];

            // Check for significant overlap (same Y position within tolerance)
            let y_overlap = (box_i.y - box_j.y).abs() < 1.0;
            let both_have_height = box_i.height > 0.0 && box_j.height > 0.0;

            assert!(!y_overlap || !both_have_height,
                "Text elements {} and {} overlap at y={}", i, j, box_i.y);
        }
    }
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/block_layout.rs` | Create | Block layout implementation |
| `src/gpu_os/document/block_layout.metal` | Create | GPU kernels |
| `src/gpu_os/document/layout.rs` | Modify | Integrate new layout |
| `src/gpu_os/document/layout.metal` | Replace | New kernels |
| `tests/test_issue_99_block_layout.rs` | Create | Tests |

## Acceptance Criteria

1. [ ] Widths computed top-down correctly
2. [ ] Heights computed bottom-up correctly
3. [ ] Text heights computed with correct container width
4. [ ] Sibling margin collapsing (max of adjacent)
5. [ ] Parent-child margin collapsing (top and bottom)
6. [ ] Border/padding prevents margin collapsing
7. [ ] Empty element margin collapsing
8. [ ] Out-of-flow elements excluded from height
9. [ ] Wikipedia layout renders with reasonable heights
10. [ ] No text overlapping
11. [ ] All tests pass
