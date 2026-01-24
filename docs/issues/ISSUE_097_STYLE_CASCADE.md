# Issue #97: GPU Style Cascade

## Summary
Apply CSS cascade algorithm to compute final styles for each element. Process matched rules in specificity order, handle inheritance, and resolve computed values.

## Problem
After selector matching, each element has a list of matched rules. We need to combine these rules according to CSS cascade rules to produce final computed styles.

## Solution

### CSS Cascade Order (CSS 2.1 Section 6.4)

1. User agent declarations (lowest priority)
2. User normal declarations
3. Author normal declarations
4. Author important declarations
5. User important declarations (highest priority)

Within each origin, sort by:
1. Specificity (higher wins)
2. Source order (later wins)

### Inheritance

Some properties inherit by default (color, font-size, etc.), others don't (margin, padding, etc.).

```rust
// Properties that inherit by default
const INHERITS: &[u16] = &[
    PROP_COLOR,
    PROP_FONT_FAMILY,
    PROP_FONT_SIZE,
    PROP_FONT_STYLE,
    PROP_FONT_WEIGHT,
    PROP_LINE_HEIGHT,
    PROP_TEXT_ALIGN,
    PROP_TEXT_DECORATION,
    PROP_TEXT_TRANSFORM,
    PROP_VISIBILITY,
    PROP_WHITE_SPACE,
    PROP_WORD_SPACING,
    PROP_LETTER_SPACING,
    PROP_DIRECTION,
];
```

### Data Structures

```rust
#[repr(C)]
pub struct ComputedStyleCompact {
    // Display & Position (4 bytes)
    pub display: u8,          // NONE, BLOCK, INLINE, FLEX, etc.
    pub position: u8,         // STATIC, RELATIVE, ABSOLUTE, FIXED
    pub float_: u8,           // NONE, LEFT, RIGHT
    pub clear: u8,            // NONE, LEFT, RIGHT, BOTH

    // Box model - margins (16 bytes)
    pub margin_top: f32,
    pub margin_right: f32,
    pub margin_bottom: f32,
    pub margin_left: f32,

    // Box model - padding (16 bytes)
    pub padding_top: f32,
    pub padding_right: f32,
    pub padding_bottom: f32,
    pub padding_left: f32,

    // Box model - border widths (16 bytes)
    pub border_top_width: f32,
    pub border_right_width: f32,
    pub border_bottom_width: f32,
    pub border_left_width: f32,

    // Dimensions (16 bytes)
    pub width: f32,           // 0 = auto
    pub height: f32,          // 0 = auto
    pub min_width: f32,
    pub min_height: f32,

    // Typography (16 bytes)
    pub font_size: f32,
    pub line_height: f32,     // Multiplier, e.g., 1.2
    pub font_weight: u16,     // 100-900
    pub text_align: u8,       // LEFT, CENTER, RIGHT, JUSTIFY
    pub white_space: u8,      // NORMAL, NOWRAP, PRE, etc.

    // Colors (16 bytes)
    pub color: u32,           // RGBA packed
    pub background_color: u32;
    pub border_color: u32;
    pub _color_pad: u32;

    // Positioning offsets (16 bytes)
    pub top: f32,             // NaN = auto
    pub right: f32,
    pub bottom: f32,
    pub left: f32,

    // Overflow & visibility (4 bytes)
    pub overflow_x: u8,       // VISIBLE, HIDDEN, SCROLL, AUTO
    pub overflow_y: u8,
    pub visibility: u8;       // VISIBLE, HIDDEN, COLLAPSE
    pub z_index_set: u8;      // Boolean: is z-index explicitly set?

    pub z_index: i32;         // Stacking order

    // Total: 120 bytes (aligned to 128)
    pub _padding: [u8; 8];
}
```

### GPU Kernels

```metal
// Pass 1: Apply matched rules to compute styles (parallel per-element)
kernel void apply_cascade(
    device const Element* elements [[buffer(0)]],
    device const MatchedRule* matched_rules [[buffer(1)]],
    device const uint* match_offsets [[buffer(2)]],
    device const uint* match_counts [[buffer(3)]],
    device const CSSRule* css_rules [[buffer(4)]],
    device const CSSProperty* properties [[buffer(5)]],
    device const uint8_t* css_buffer [[buffer(6)]],
    device ComputedStyleCompact* computed [[buffer(7)]],
    constant uint& element_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    // Start with initial values
    ComputedStyleCompact style = default_style();

    // Apply user-agent styles based on element type
    Element elem = elements[gid];
    apply_user_agent_style(elem.element_type, style);

    // Apply matched rules in order (lowest specificity first)
    // Rules are pre-sorted by specificity ascending
    uint match_start = match_offsets[gid];
    uint match_count = match_counts[gid];

    for (uint i = 0; i < match_count; i++) {
        MatchedRule match = matched_rules[match_start + i];
        CSSRule rule = css_rules[match.rule_index];

        // Apply each property in the rule
        for (uint p = 0; p < rule.property_count; p++) {
            CSSProperty prop = properties[rule.property_start + p];
            apply_property(style, prop, css_buffer);
        }
    }

    // Apply inline styles (highest priority)
    // (Inline styles were parsed separately and stored with element)
    if (elem.inline_style_start > 0) {
        apply_inline_styles(style, elements, elem, css_buffer);
    }

    computed[gid] = style;
}

// Helper: Apply a single CSS property
void apply_property(
    thread ComputedStyleCompact& style,
    CSSProperty prop,
    device const uint8_t* css_buffer
) {
    switch (prop.property_id) {
        case PROP_DISPLAY:
            style.display = parse_display_value(css_buffer, prop.value_start, prop.value_length);
            break;

        case PROP_POSITION:
            style.position = parse_position_value(css_buffer, prop.value_start, prop.value_length);
            break;

        case PROP_MARGIN_TOP:
            style.margin_top = prop.numeric_value;
            break;

        case PROP_MARGIN_RIGHT:
            style.margin_right = prop.numeric_value;
            break;

        case PROP_MARGIN_BOTTOM:
            style.margin_bottom = prop.numeric_value;
            break;

        case PROP_MARGIN_LEFT:
            style.margin_left = prop.numeric_value;
            break;

        case PROP_MARGIN:
            // Shorthand: apply to all sides
            style.margin_top = prop.numeric_value;
            style.margin_right = prop.numeric_value;
            style.margin_bottom = prop.numeric_value;
            style.margin_left = prop.numeric_value;
            break;

        case PROP_PADDING_TOP:
            style.padding_top = prop.numeric_value;
            break;

        // ... similar for other padding, border properties

        case PROP_WIDTH:
            style.width = prop.numeric_value;
            break;

        case PROP_HEIGHT:
            style.height = prop.numeric_value;
            break;

        case PROP_FONT_SIZE:
            style.font_size = prop.numeric_value;
            break;

        case PROP_LINE_HEIGHT:
            style.line_height = prop.numeric_value;
            break;

        case PROP_COLOR:
            style.color = parse_color(css_buffer, prop.value_start, prop.value_length);
            break;

        case PROP_BACKGROUND_COLOR:
            style.background_color = parse_color(css_buffer, prop.value_start, prop.value_length);
            break;

        case PROP_OVERFLOW:
            uint8_t ov = parse_overflow_value(css_buffer, prop.value_start, prop.value_length);
            style.overflow_x = ov;
            style.overflow_y = ov;
            break;

        case PROP_VISIBILITY:
            style.visibility = parse_visibility_value(css_buffer, prop.value_start, prop.value_length);
            break;

        case PROP_Z_INDEX:
            style.z_index = int(prop.numeric_value);
            style.z_index_set = 1;
            break;

        // ... handle remaining properties
    }
}

// Helper: Parse display value from CSS text
uint8_t parse_display_value(device const uint8_t* css, uint start, uint len) {
    // Check for common values
    if (string_equals(css, start, len, "none")) return DISPLAY_NONE;
    if (string_equals(css, start, len, "block")) return DISPLAY_BLOCK;
    if (string_equals(css, start, len, "inline")) return DISPLAY_INLINE;
    if (string_equals(css, start, len, "inline-block")) return DISPLAY_INLINE_BLOCK;
    if (string_equals(css, start, len, "flex")) return DISPLAY_FLEX;
    if (string_equals(css, start, len, "inline-flex")) return DISPLAY_INLINE_FLEX;
    if (string_equals(css, start, len, "grid")) return DISPLAY_GRID;
    if (string_equals(css, start, len, "table")) return DISPLAY_TABLE;
    if (string_equals(css, start, len, "table-row")) return DISPLAY_TABLE_ROW;
    if (string_equals(css, start, len, "table-cell")) return DISPLAY_TABLE_CELL;
    if (string_equals(css, start, len, "list-item")) return DISPLAY_LIST_ITEM;
    return DISPLAY_BLOCK; // Default
}

// Pass 2: Inheritance (top-down by level)
kernel void apply_inheritance(
    device const Element* elements [[buffer(0)]],
    device ComputedStyleCompact* computed [[buffer(1)]],
    device const uint* depths [[buffer(2)]],
    constant uint& current_level [[buffer(3)]],
    constant uint& element_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    if (elem.parent < 0) return;  // Root has no parent

    ComputedStyleCompact parent_style = computed[elem.parent];
    ComputedStyleCompact my_style = computed[gid];

    // Inherit specified properties if not explicitly set
    // (In practice, track which properties were set via a bitmask)

    // These properties always inherit
    if (my_style.font_size == 0) {
        my_style.font_size = parent_style.font_size;
    }
    if (my_style.line_height == 0) {
        my_style.line_height = parent_style.line_height;
    }
    if (my_style.color == 0) {
        my_style.color = parent_style.color;
    }
    // font-weight, text-align, etc.

    computed[gid] = my_style;
}

// Pass 3: Resolve relative units (em, %) to pixels
kernel void resolve_units(
    device const Element* elements [[buffer(0)]],
    device ComputedStyleCompact* computed [[buffer(1)]],
    device const uint* depths [[buffer(2)]],
    constant uint& current_level [[buffer(3)]],
    constant uint& element_count [[buffer(4)]],
    constant float& viewport_width [[buffer(5)]],
    constant float& viewport_height [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyleCompact style = computed[gid];

    float parent_font_size = 16.0;  // Default
    float parent_width = viewport_width;

    if (elem.parent >= 0) {
        ComputedStyleCompact parent = computed[elem.parent];
        parent_font_size = parent.font_size;
        parent_width = parent.width > 0 ? parent.width : viewport_width;
    }

    // Resolve font-size first (needed for em units)
    // If font-size was set in em or %, convert to px
    // (This requires tracking the unit type during cascade - simplified here)

    // Resolve percentage widths
    if (style.width < 0) {  // Negative = percentage
        style.width = parent_width * (-style.width / 100.0);
    }

    // Resolve percentage margins
    // Note: In CSS, percentage margins are relative to containing block WIDTH
    if (style.margin_top < 0) {
        style.margin_top = parent_width * (-style.margin_top / 100.0);
    }
    // ... similar for other margins, padding

    computed[gid] = style;
}
```

## Pseudocode

```
FUNCTION apply_cascade(elements, matched_rules, css_rules, css_properties):
    // Pass 1: Apply matched rules
    computed_styles = gpu_dispatch(
        apply_cascade_kernel,
        elements, matched_rules, css_rules, css_properties
    )

    // Pass 2: Inheritance (top-down by level)
    depths = compute_depths(elements)
    max_depth = max(depths)

    FOR level = 0 TO max_depth:
        gpu_dispatch(
            apply_inheritance,
            elements, computed_styles, depths, level
        )

    // Pass 3: Resolve relative units (top-down by level)
    FOR level = 0 TO max_depth:
        gpu_dispatch(
            resolve_units,
            elements, computed_styles, depths, level
        )

    RETURN computed_styles
```

## Tests

### Test 1: Basic cascade
```rust
#[test]
fn test_basic_cascade() {
    let html = r#"<div class="red"></div>"#;
    let css = ".red { color: red; }";

    let result = process_document(html, css);
    let style = &result.computed_styles[0];

    assert_eq!(style.color, rgba(255, 0, 0, 255));
}
```

### Test 2: Specificity override
```rust
#[test]
fn test_specificity_override() {
    let html = r#"<div id="main" class="blue"></div>"#;
    let css = r#"
        .blue { color: blue; }
        #main { color: red; }
    "#;

    let result = process_document(html, css);
    let style = &result.computed_styles[0];

    // #main (specificity 100) beats .blue (specificity 10)
    assert_eq!(style.color, rgba(255, 0, 0, 255));
}
```

### Test 3: Source order
```rust
#[test]
fn test_source_order() {
    let html = r#"<div class="test"></div>"#;
    let css = r#"
        .test { color: red; }
        .test { color: blue; }
    "#;

    let result = process_document(html, css);
    let style = &result.computed_styles[0];

    // Later rule wins with same specificity
    assert_eq!(style.color, rgba(0, 0, 255, 255));
}
```

### Test 4: Inheritance
```rust
#[test]
fn test_inheritance() {
    let html = r#"
        <div class="parent">
            <span class="child"></span>
        </div>
    "#;
    let css = ".parent { color: red; font-size: 20px; }";

    let result = process_document(html, css);
    let parent_style = &result.computed_styles[0];
    let child_style = &result.computed_styles[1];

    // color and font-size inherit
    assert_eq!(child_style.color, parent_style.color);
    assert_eq!(child_style.font_size, parent_style.font_size);
}
```

### Test 5: Non-inherited properties
```rust
#[test]
fn test_non_inheritance() {
    let html = r#"
        <div class="parent">
            <span class="child"></span>
        </div>
    "#;
    let css = ".parent { margin: 20px; padding: 10px; }";

    let result = process_document(html, css);
    let parent_style = &result.computed_styles[0];
    let child_style = &result.computed_styles[1];

    // margin and padding don't inherit
    assert_eq!(parent_style.margin_top, 20.0);
    assert_eq!(child_style.margin_top, 0.0);  // Default, not inherited
}
```

### Test 6: display:none from CSS
```rust
#[test]
fn test_display_none() {
    let html = r#"<div class="hidden">content</div>"#;
    let css = ".hidden { display: none; }";

    let result = process_document(html, css);
    let style = &result.computed_styles[0];

    assert_eq!(style.display, DISPLAY_NONE);
}
```

### Test 7: Wikipedia hidden elements
```rust
#[test]
fn test_wikipedia_hidden_elements() {
    let html = include_str!("../testdata/wikipedia.html");
    let css = include_str!("../testdata/wikipedia.css");

    let result = process_document(html, css);

    // Find elements with class "mw-hidden"
    let hidden_elements = find_elements_by_class(&result.elements, "mw-hidden");

    for elem_idx in hidden_elements {
        let style = &result.computed_styles[elem_idx];
        assert_eq!(style.display, DISPLAY_NONE,
            "Element with class 'mw-hidden' should have display:none");
    }
}
```

### Test 8: User agent styles
```rust
#[test]
fn test_user_agent_styles() {
    let html = r#"<h1>Title</h1><p>Text</p>"#;
    let css = "";  // No author CSS

    let result = process_document(html, css);

    // h1 should have larger font-size from UA stylesheet
    let h1_style = &result.computed_styles[0];
    let p_style = &result.computed_styles[1];

    assert!(h1_style.font_size > p_style.font_size);
    assert_eq!(h1_style.display, DISPLAY_BLOCK);
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/cascade.rs` | Create | Cascade implementation |
| `src/gpu_os/document/cascade.metal` | Create | GPU kernels |
| `src/gpu_os/document/computed_style.rs` | Create | ComputedStyleCompact struct |
| `tests/test_issue_97_cascade.rs` | Create | Tests |

## Acceptance Criteria

1. [ ] Apply matched rules in specificity order
2. [ ] Source order tie-breaking
3. [ ] User agent styles for HTML elements
4. [ ] Inheritance for inheritable properties
5. [ ] Non-inheritance for non-inheritable properties
6. [ ] Parse common display values
7. [ ] Parse position values
8. [ ] Parse color values
9. [ ] Resolve percentage units
10. [ ] Wikipedia .mw-hidden elements have display:none
11. [ ] All tests pass
