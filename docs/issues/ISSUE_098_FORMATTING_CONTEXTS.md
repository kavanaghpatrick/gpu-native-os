# Issue #98: Formatting Context Construction

## Summary
Build Block Formatting Contexts (BFC) and Inline Formatting Contexts (IFC) from computed styles. This determines how elements are laid out relative to each other.

## Problem
CSS layout depends on "formatting contexts" - regions where specific layout rules apply. A Block Formatting Context lays out children vertically; an Inline Formatting Context lays them out horizontally in line boxes.

## CSS 2.1 Formatting Context Rules

### Block Formatting Context (BFC)

A new BFC is established by:
- Root element (`<html>`)
- Floats (`float: left/right`)
- Absolutely positioned elements (`position: absolute/fixed`)
- Inline-blocks (`display: inline-block`)
- Table cells (`display: table-cell`)
- Table captions (`display: table-caption`)
- Elements with `overflow` other than `visible`
- Flex items (direct children of `display: flex`)
- Grid items (direct children of `display: grid`)

In a BFC:
- Boxes are laid out vertically from top of containing block
- Vertical margins between adjacent boxes collapse
- Each box's left edge touches the containing block's left edge

### Inline Formatting Context (IFC)

An IFC is established when a block container contains only inline-level content.

In an IFC:
- Boxes are laid out horizontally in "line boxes"
- Line boxes are stacked vertically
- Horizontal margins, borders, padding between inline boxes are respected
- Vertical alignment controlled by `vertical-align`

## Data Structures

```rust
pub const FC_BLOCK: u8 = 1;
pub const FC_INLINE: u8 = 2;

#[repr(C)]
pub struct FormattingContext {
    pub context_type: u8,           // FC_BLOCK or FC_INLINE
    pub establishes_bfc: u8,        // Does this element establish a new BFC?
    pub _padding: [u8; 2],
    pub container_element: u32,     // Element that establishes this context
    pub first_child_fc: i32,        // First nested formatting context
    pub next_sibling_fc: i32,       // Next sibling context
}

#[repr(C)]
pub struct ElementFormattingInfo {
    pub formatting_context: u32,    // Index of containing FC
    pub fc_child_index: u32,        // Index within FC's children
    pub is_bfc_root: u8,            // Does this element establish a BFC?
    pub is_ifc_root: u8,            // Does this element establish an IFC?
    pub participates_in: u8,        // FC_BLOCK or FC_INLINE
    pub _padding: u8,
}

#[repr(C)]
pub struct InlineItem {
    pub element_index: u32,         // Source element
    pub item_type: u8,              // TEXT, INLINE_BOX, LINE_BREAK
    pub _padding: [u8; 3],
    pub text_start: u32,            // For text items
    pub text_length: u32,
}

pub const INLINE_TEXT: u8 = 1;
pub const INLINE_BOX: u8 = 2;
pub const INLINE_LINE_BREAK: u8 = 3;
```

## GPU Kernels

```metal
// Pass 1: Determine formatting context participation (parallel per-element)
kernel void determine_fc_participation(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device ElementFormattingInfo* fc_info [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = styles[gid];
    ElementFormattingInfo info;

    info.formatting_context = 0;  // Will be set in next pass
    info.fc_child_index = 0;
    info._padding = 0;

    // Check if this element establishes a BFC
    info.is_bfc_root = 0;
    if (gid == 0) {
        // Root element always establishes BFC
        info.is_bfc_root = 1;
    } else if (style.float_ != FLOAT_NONE) {
        info.is_bfc_root = 1;
    } else if (style.position == POSITION_ABSOLUTE || style.position == POSITION_FIXED) {
        info.is_bfc_root = 1;
    } else if (style.display == DISPLAY_INLINE_BLOCK) {
        info.is_bfc_root = 1;
    } else if (style.display == DISPLAY_TABLE_CELL) {
        info.is_bfc_root = 1;
    } else if (style.overflow_x != OVERFLOW_VISIBLE || style.overflow_y != OVERFLOW_VISIBLE) {
        info.is_bfc_root = 1;
    } else if (style.display == DISPLAY_FLEX || style.display == DISPLAY_GRID) {
        info.is_bfc_root = 1;
    }

    // Check if this element establishes an IFC
    // (Block container with only inline-level children)
    info.is_ifc_root = 0;
    if (is_block_container(style.display) && elem.first_child >= 0) {
        // Check if all children are inline-level
        bool all_inline = true;
        int child = elem.first_child;
        while (child >= 0) {
            ComputedStyleCompact child_style = styles[child];
            if (is_block_level(child_style.display)) {
                all_inline = false;
                break;
            }
            child = elements[child].next_sibling;
        }
        if (all_inline) {
            info.is_ifc_root = 1;
        }
    }

    // Determine what type of formatting this element participates in
    if (is_block_level(style.display)) {
        info.participates_in = FC_BLOCK;
    } else if (is_inline_level(style.display)) {
        info.participates_in = FC_INLINE;
    } else {
        info.participates_in = FC_BLOCK;  // Default
    }

    // Out-of-flow elements don't participate in normal flow
    if (style.position == POSITION_ABSOLUTE || style.position == POSITION_FIXED) {
        info.participates_in = 0;  // No participation
    }
    if (style.float_ != FLOAT_NONE) {
        info.participates_in = 0;  // Floats handled separately
    }

    fc_info[gid] = info;
}

// Helper functions
bool is_block_level(uint8_t display) {
    return display == DISPLAY_BLOCK ||
           display == DISPLAY_LIST_ITEM ||
           display == DISPLAY_TABLE ||
           display == DISPLAY_FLEX ||
           display == DISPLAY_GRID;
}

bool is_inline_level(uint8_t display) {
    return display == DISPLAY_INLINE ||
           display == DISPLAY_INLINE_BLOCK ||
           display == DISPLAY_INLINE_FLEX;
}

bool is_block_container(uint8_t display) {
    return display == DISPLAY_BLOCK ||
           display == DISPLAY_LIST_ITEM ||
           display == DISPLAY_INLINE_BLOCK ||
           display == DISPLAY_TABLE_CELL;
}

// Pass 2: Assign formatting contexts (top-down by level)
kernel void assign_formatting_contexts(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device ElementFormattingInfo* fc_info [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    device atomic_uint* fc_counter [[buffer(4)]],
    device FormattingContext* contexts [[buffer(5)]],
    constant uint& current_level [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = styles[gid];
    ElementFormattingInfo info = fc_info[gid];

    // Skip hidden elements
    if (style.display == DISPLAY_NONE) return;

    // Get parent's formatting context
    uint parent_fc = 0;
    if (elem.parent >= 0) {
        parent_fc = fc_info[elem.parent].formatting_context;

        // If parent establishes IFC, we're in that IFC
        if (fc_info[elem.parent].is_ifc_root) {
            parent_fc = elem.parent;  // Parent is the IFC root
        }
    }

    // If we establish a new BFC, create one
    if (info.is_bfc_root) {
        uint new_fc = atomic_fetch_add_explicit(fc_counter, 1, memory_order_relaxed);
        contexts[new_fc].context_type = FC_BLOCK;
        contexts[new_fc].establishes_bfc = 1;
        contexts[new_fc].container_element = gid;
        contexts[new_fc].first_child_fc = -1;
        contexts[new_fc].next_sibling_fc = -1;

        fc_info[gid].formatting_context = new_fc;
    } else {
        fc_info[gid].formatting_context = parent_fc;
    }

    // Count position within parent's children
    if (elem.parent >= 0) {
        uint idx = 0;
        int sib = elements[elem.parent].first_child;
        while (sib >= 0 && sib != int(gid)) {
            ComputedStyleCompact sib_style = styles[sib];
            ElementFormattingInfo sib_info = fc_info[sib];

            // Only count in-flow siblings
            if (sib_style.display != DISPLAY_NONE &&
                sib_info.participates_in == info.participates_in) {
                idx++;
            }
            sib = elements[sib].next_sibling;
        }
        fc_info[gid].fc_child_index = idx;
    }
}

// Pass 3: Build inline item list for IFC roots (parallel per IFC root)
kernel void build_inline_items(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyleCompact* styles [[buffer(1)]],
    device const ElementFormattingInfo* fc_info [[buffer(2)]],
    device const uint8_t* text_buffer [[buffer(3)]],
    device InlineItem* inline_items [[buffer(4)]],
    device atomic_uint* item_count [[buffer(5)]],
    device uint* ifc_item_starts [[buffer(6)]],  // Start index per IFC root
    constant uint& element_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (fc_info[gid].is_ifc_root == 0) return;

    // This element is an IFC root - collect all inline items
    uint item_start = atomic_fetch_add_explicit(item_count, 0, memory_order_relaxed);
    ifc_item_starts[gid] = item_start;

    // Walk all descendants, collecting inline items
    collect_inline_items_recursive(
        elements, styles, fc_info, text_buffer,
        inline_items, item_count, gid
    );
}

void collect_inline_items_recursive(
    device const Element* elements,
    device const ComputedStyleCompact* styles,
    device const ElementFormattingInfo* fc_info,
    device const uint8_t* text_buffer,
    device InlineItem* items,
    device atomic_uint* item_count,
    uint elem_idx
) {
    Element elem = elements[elem_idx];
    ComputedStyleCompact style = styles[elem_idx];

    if (style.display == DISPLAY_NONE) return;

    // Text node
    if (elem.element_type == ELEM_TEXT && elem.text_length > 0) {
        uint slot = atomic_fetch_add_explicit(item_count, 1, memory_order_relaxed);
        items[slot].element_index = elem_idx;
        items[slot].item_type = INLINE_TEXT;
        items[slot].text_start = elem.text_start;
        items[slot].text_length = elem.text_length;
        return;
    }

    // Inline element (like <span>)
    if (is_inline_level(style.display)) {
        // Start of inline box
        uint slot = atomic_fetch_add_explicit(item_count, 1, memory_order_relaxed);
        items[slot].element_index = elem_idx;
        items[slot].item_type = INLINE_BOX;
        items[slot].text_start = 0;
        items[slot].text_length = 0;
    }

    // Process children
    int child = elem.first_child;
    while (child >= 0) {
        // Stop at block-level children (they establish their own context)
        if (is_block_level(styles[child].display)) {
            break;
        }
        collect_inline_items_recursive(elements, styles, fc_info, text_buffer, items, item_count, child);
        child = elements[child].next_sibling;
    }

    // BR element creates line break
    if (elem.element_type == ELEM_BR) {
        uint slot = atomic_fetch_add_explicit(item_count, 1, memory_order_relaxed);
        items[slot].element_index = elem_idx;
        items[slot].item_type = INLINE_LINE_BREAK;
    }
}
```

## Pseudocode

```
FUNCTION build_formatting_contexts(elements, computed_styles):
    // Pass 1: Determine FC participation for each element
    fc_info = gpu_dispatch(determine_fc_participation, elements, computed_styles)

    // Pass 2: Assign formatting contexts (top-down)
    contexts = allocate_contexts(element_count)
    fc_counter = 0

    depths = compute_depths(elements)
    max_depth = max(depths)

    FOR level = 0 TO max_depth:
        gpu_dispatch(assign_formatting_contexts,
            elements, computed_styles, fc_info, depths, fc_counter, contexts, level)

    // Pass 3: Build inline item lists for IFC roots
    inline_items = allocate_items(estimated_count)
    item_count = 0
    ifc_item_starts = zeros(element_count)

    gpu_dispatch(build_inline_items,
        elements, computed_styles, fc_info, text_buffer, inline_items, item_count, ifc_item_starts)

    RETURN FormattingContextResult {
        fc_info,
        contexts,
        inline_items,
        ifc_item_starts
    }
```

## Tests

### Test 1: Root establishes BFC
```rust
#[test]
fn test_root_establishes_bfc() {
    let html = "<html><body><div>text</div></body></html>";
    let result = build_formatting_contexts(html);

    assert_eq!(result.fc_info[0].is_bfc_root, 1);  // html
}
```

### Test 2: overflow:hidden establishes BFC
```rust
#[test]
fn test_overflow_establishes_bfc() {
    let html = r#"<div style="overflow: hidden"><p>text</p></div>"#;
    let result = build_formatting_contexts(html);

    let div_idx = find_element_by_tag(&result.elements, "div");
    assert_eq!(result.fc_info[div_idx].is_bfc_root, 1);
}
```

### Test 3: position:absolute establishes BFC
```rust
#[test]
fn test_absolute_establishes_bfc() {
    let html = r#"<div style="position: absolute"><p>text</p></div>"#;
    let result = build_formatting_contexts(html);

    let div_idx = find_element_by_tag(&result.elements, "div");
    assert_eq!(result.fc_info[div_idx].is_bfc_root, 1);
}
```

### Test 4: Block with inline children creates IFC
```rust
#[test]
fn test_ifc_creation() {
    let html = r#"<p>Hello <span>world</span>!</p>"#;
    let result = build_formatting_contexts(html);

    let p_idx = find_element_by_tag(&result.elements, "p");
    assert_eq!(result.fc_info[p_idx].is_ifc_root, 1);
}
```

### Test 5: Block children prevent IFC
```rust
#[test]
fn test_block_prevents_ifc() {
    let html = r#"<div>text<div>block</div>more text</div>"#;
    let result = build_formatting_contexts(html);

    let outer_div = find_first_element_by_tag(&result.elements, "div");
    assert_eq!(result.fc_info[outer_div].is_ifc_root, 0);  // Has block child
}
```

### Test 6: Inline items collection
```rust
#[test]
fn test_inline_items() {
    let html = r#"<p>Hello <span>world</span>!</p>"#;
    let result = build_formatting_contexts(html);

    let p_idx = find_element_by_tag(&result.elements, "p");
    let items = get_inline_items(&result, p_idx);

    // Should have: "Hello " (text), <span> (box), "world" (text), "!" (text)
    assert_eq!(items.len(), 4);
    assert_eq!(items[0].item_type, INLINE_TEXT);
    assert_eq!(items[1].item_type, INLINE_BOX);
    assert_eq!(items[2].item_type, INLINE_TEXT);
    assert_eq!(items[3].item_type, INLINE_TEXT);
}
```

### Test 7: Out-of-flow elements don't participate
```rust
#[test]
fn test_out_of_flow() {
    let html = r#"
        <div>
            <span style="position: absolute">abs</span>
            <span>normal</span>
        </div>
    "#;
    let result = build_formatting_contexts(html);

    let abs_span = find_element_by_style(&result, "position", "absolute");
    let normal_span = find_element_by_content(&result, "normal");

    assert_eq!(result.fc_info[abs_span].participates_in, 0);  // No participation
    assert_eq!(result.fc_info[normal_span].participates_in, FC_INLINE);
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/formatting_context.rs` | Create | FC implementation |
| `src/gpu_os/document/formatting_context.metal` | Create | GPU kernels |
| `tests/test_issue_98_formatting_context.rs` | Create | Tests |

## Acceptance Criteria

1. [ ] Identify BFC-establishing elements
2. [ ] Identify IFC-establishing elements
3. [ ] Build FC tree structure
4. [ ] Assign elements to formatting contexts
5. [ ] Track participation type (block/inline)
6. [ ] Collect inline items for IFC
7. [ ] Handle out-of-flow elements correctly
8. [ ] All tests pass
