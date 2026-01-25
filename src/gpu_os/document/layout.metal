#include <metal_stdlib>
using namespace metal;

// Display types (from style)
constant uint DISPLAY_NONE = 0;
constant uint DISPLAY_BLOCK = 1;
constant uint DISPLAY_INLINE = 2;
constant uint DISPLAY_FLEX = 3;
constant uint DISPLAY_INLINE_BLOCK = 4;

// Flex direction
constant uint FLEX_ROW = 0;
constant uint FLEX_COLUMN = 1;

// Justify content
constant uint JUSTIFY_START = 0;
constant uint JUSTIFY_CENTER = 1;
constant uint JUSTIFY_END = 2;
constant uint JUSTIFY_SPACE_BETWEEN = 3;
constant uint JUSTIFY_SPACE_AROUND = 4;

// Align items
constant uint ALIGN_START = 0;
constant uint ALIGN_CENTER = 1;
constant uint ALIGN_END = 2;
constant uint ALIGN_STRETCH = 3;

// Element type for text
constant uint ELEM_TEXT = 100;

// Limits
constant uint MAX_CHILDREN = 128;
constant uint THREAD_COUNT = 1024;

struct Element {
    uint element_type;
    int parent;
    int first_child;
    int next_sibling;
    int prev_sibling;    // Issue #128: Enable O(1) cumulative height lookup
    uint text_start;
    uint text_length;
    uint token_index;
};

struct ComputedStyle {
    // Display
    uint display;

    // Box model
    float width;
    float height;
    float margin[4];     // top, right, bottom, left
    float padding[4];

    // Flexbox
    uint flex_direction;
    uint justify_content;
    uint align_items;
    float flex_grow;
    float flex_shrink;

    // Typography
    float color[4];
    float font_size;
    float line_height;
    uint font_weight;
    uint text_align;

    // Visual
    float background_color[4];
    float border_width[4];
    float border_color[4];
    float border_radius;
    float opacity;

    // CSS Positioning
    uint position;       // POSITION_STATIC, RELATIVE, ABSOLUTE, FIXED
    float top;           // Offset from top
    float right_;        // Offset from right
    float bottom;        // Offset from bottom
    float left;          // Offset from left
    int z_index;         // Stacking order

    // Inheritance tracking
    uint properties_set;

    // CSS Overflow
    uint overflow_x;
    uint overflow_y;

    // Box Shadow (up to 4 shadows)
    uint shadow_count;
    float shadow_offset_x[4];
    float shadow_offset_y[4];
    float shadow_blur[4];
    float shadow_spread[4];
    float shadow_color[16];
    uint shadow_inset[4];

    // Gradients (up to 8 color stops)
    uint gradient_type;
    float gradient_angle;
    uint gradient_stop_count;
    float gradient_stop_colors[32];
    float gradient_stop_positions[8];

    // Table layout
    uint border_collapse;
    float border_spacing;
    float _padding[2];
};

// CSS Position values
constant uint POSITION_STATIC = 0;
constant uint POSITION_RELATIVE = 1;
constant uint POSITION_ABSOLUTE = 2;
constant uint POSITION_FIXED = 3;

// Special value for "auto" offset
constant float OFFSET_AUTO = 3.4028235e+38;

// CSS Overflow values
constant uint OVERFLOW_VISIBLE = 0;
constant uint OVERFLOW_HIDDEN = 1;
constant uint OVERFLOW_SCROLL = 2;
constant uint OVERFLOW_AUTO = 3;

struct LayoutBox {
    // Position (relative to parent content box)
    float x;
    float y;

    // Dimensions (border box)
    float width;
    float height;

    // Content box (absolute position after finalize)
    float content_x;
    float content_y;
    float content_width;
    float content_height;

    // Scroll dimensions
    float scroll_width;
    float scroll_height;

    float _padding[6];
};

struct Viewport {
    float width;
    float height;
    float _padding[2];
};

// Calculate glyph advance for a character
float glyph_advance(uint char_code, float font_size) {
    if (char_code == ' ') return font_size * 0.3;
    if (char_code == '\t') return font_size * 1.2;
    if (char_code == '\n' || char_code == '\r') return 0.0;
    return font_size * 0.6;
}

// Calculate intrinsic content width for text (unwrapped)
float text_width(uint text_length, float font_size) {
    // Approximation: average character width is ~0.6 * font_size
    return float(text_length) * font_size * 0.6;
}

// Calculate intrinsic content height for single line of text
float text_height(float font_size, float line_height) {
    return font_size * line_height;
}

// Calculate multi-line text height based on container width and text content
float text_height_wrapped(
    device const uint8_t* text,
    uint text_start,
    uint text_length,
    float font_size,
    float line_height,
    float container_width
) {
    if (text_length == 0) return 0.0;
    if (container_width <= 0) container_width = 10000.0;

    // First pass: check if text is whitespace-only
    bool has_visible_content = false;
    for (uint i = 0; i < text_length; i++) {
        uint c = text[text_start + i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            has_visible_content = true;
            break;
        }
    }

    // Whitespace-only text nodes get zero height (collapse whitespace)
    if (!has_visible_content) return 0.0;

    float line_height_px = font_size * line_height;
    uint line_count = 1;
    float current_width = 0.0;
    bool line_has_content = false;

    for (uint i = 0; i < text_length; i++) {
        uint char_code = text[text_start + i];

        // Newline forces new line only if current line has content
        if (char_code == '\n') {
            if (line_has_content) {
                line_count++;
                line_has_content = false;
            }
            current_width = 0.0;
            continue;
        }

        // Skip other whitespace at line start
        if (!line_has_content && (char_code == ' ' || char_code == '\t')) {
            continue;
        }

        line_has_content = true;
        float advance = glyph_advance(char_code, font_size);
        current_width += advance;

        // Check for wrap
        if (current_width > container_width && i > 0) {
            line_count++;
            current_width = advance;
        }
    }

    // Only count the last line if it has content
    if (!line_has_content && line_count > 0) {
        line_count--;
    }

    return max(float(line_count), 1.0) * line_height_px;
}

// Pass 1: Compute intrinsic sizes (bottom-up)
// This pass computes min/max content sizes for each element
kernel void compute_intrinsic_sizes(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant Viewport& viewport [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    LayoutBox box;
    box.x = 0;
    box.y = 0;
    box.width = 0;
    box.height = 0;
    box.content_x = 0;
    box.content_y = 0;
    box.content_width = 0;
    box.content_height = 0;
    box.scroll_width = 0;
    box.scroll_height = 0;
    for (int i = 0; i < 6; i++) box._padding[i] = 0;

    // Hidden elements get zero size
    if (style.display == DISPLAY_NONE) {
        boxes[gid] = box;
        return;
    }

    // Calculate padding and border totals
    float padding_h = style.padding[1] + style.padding[3];
    float padding_v = style.padding[0] + style.padding[2];
    float border_h = style.border_width[1] + style.border_width[3];
    float border_v = style.border_width[0] + style.border_width[2];

    // Text nodes: intrinsic size based on content
    if (elem.element_type == ELEM_TEXT) {
        float font_size = style.font_size > 0 ? style.font_size : 16.0;
        float line_height = style.line_height > 0 ? style.line_height : 1.2;
        float tw = text_width(elem.text_length, font_size);
        float th = text_height(font_size, line_height);
        box.content_width = tw;
        box.content_height = th;
        box.width = tw;
        box.height = th;
        boxes[gid] = box;
        return;
    }

    // For container elements, use explicit size if set
    if (style.width > 0) {
        box.width = style.width;
        box.content_width = style.width - padding_h - border_h;
    }

    if (style.height > 0) {
        box.height = style.height;
        box.content_height = style.height - padding_v - border_v;
    }

    boxes[gid] = box;
}

// Pass 2: Compute layout (top-down for blocks, iterative for flex)
// This is the main layout pass
kernel void compute_block_layout(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant Viewport& viewport [[buffer(4)]],
    device const uint8_t* text [[buffer(5)]],  // Text buffer for wrapped height calculation
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    // Skip hidden elements
    if (style.display == DISPLAY_NONE) return;

    // Get parent's layout
    float parent_content_width = viewport.width;
    float parent_content_x = 0;
    float parent_content_y = 0;

    if (elem.parent >= 0) {
        LayoutBox parent_box = boxes[elem.parent];
        ComputedStyle parent_style = styles[elem.parent];
        parent_content_width = parent_box.content_width;
        parent_content_x = parent_box.content_x;
        parent_content_y = parent_box.content_y;

        // If parent has no explicit width, use viewport
        if (parent_content_width <= 0) {
            parent_content_width = viewport.width -
                parent_style.padding[1] - parent_style.padding[3] -
                parent_style.border_width[1] - parent_style.border_width[3] -
                parent_style.margin[1] - parent_style.margin[3];
        }
    }

    LayoutBox box = boxes[gid];

    // Calculate padding and border totals
    float padding_h = style.padding[1] + style.padding[3];
    float padding_v = style.padding[0] + style.padding[2];
    float border_h = style.border_width[1] + style.border_width[3];
    float border_v = style.border_width[0] + style.border_width[2];
    float margin_h = style.margin[1] + style.margin[3];

    // Block layout: width fills parent minus margins
    if (style.display == DISPLAY_BLOCK && style.width <= 0) {
        box.width = parent_content_width - margin_h;
        box.content_width = box.width - padding_h - border_h;
    }

    // NOTE: Text height calculation moved to propagate_widths_and_text kernel
    // which runs top-down by level AFTER depths are computed

    // Position: margin from parent edge
    box.x = style.margin[3];  // Left margin

    // Y position depends on previous siblings (handled in sequential pass)
    // For now, just set based on parent
    box.y = style.margin[0];  // Top margin

    // Content box position
    box.content_x = box.x + style.border_width[3] + style.padding[3];
    box.content_y = box.y + style.border_width[0] + style.padding[0];

    boxes[gid] = box;
}

// Helper: Layout one parent's children and update parent height
void layout_parent_children(
    device const Element* elements,
    device const ComputedStyle* styles,
    device LayoutBox* boxes,
    uint parent_idx
) {
    Element elem = elements[parent_idx];
    ComputedStyle style = styles[parent_idx];

    if (style.display == DISPLAY_NONE) return;
    if (elem.first_child < 0) return;

    LayoutBox parent_box = boxes[parent_idx];
    float child_y = 0;
    float child_x = 0;

    // Flex layout handling
    bool is_flex = (style.display == DISPLAY_FLEX);
    bool is_row = (style.flex_direction == FLEX_ROW);

    // Collect children
    int children[MAX_CHILDREN];
    uint child_count = 0;
    float total_size = 0;
    float total_flex_grow = 0;

    int child_idx = elem.first_child;
    while (child_idx >= 0 && child_count < MAX_CHILDREN) {
        children[child_count] = child_idx;
        LayoutBox child_box = boxes[child_idx];
        ComputedStyle child_style = styles[child_idx];

        if (is_flex) {
            if (is_row) {
                total_size += child_box.width + child_style.margin[1] + child_style.margin[3];
            } else {
                total_size += child_box.height + child_style.margin[0] + child_style.margin[2];
            }
            total_flex_grow += child_style.flex_grow;
        }

        child_count++;
        child_idx = elements[child_idx].next_sibling;
    }

    // Calculate flex distribution
    float available = is_row ? parent_box.content_width : parent_box.content_height;
    float remaining = available - total_size;
    float gap = 0;
    float start_offset = 0;

    if (is_flex && child_count > 0) {
        switch (style.justify_content) {
            case JUSTIFY_CENTER:
                start_offset = remaining / 2;
                break;
            case JUSTIFY_END:
                start_offset = remaining;
                break;
            case JUSTIFY_SPACE_BETWEEN:
                if (child_count > 1) gap = remaining / float(child_count - 1);
                break;
            case JUSTIFY_SPACE_AROUND:
                gap = remaining / float(child_count);
                start_offset = gap / 2;
                break;
            default:
                break;
        }
    }

    // Position each child
    if (is_row) {
        child_x = start_offset;
    } else {
        child_y = start_offset;
    }

    for (uint c = 0; c < child_count; c++) {
        int idx = children[c];
        LayoutBox child_box = boxes[idx];
        ComputedStyle child_style = styles[idx];

        if (is_flex) {
            if (is_row) {
                child_box.x = child_x + child_style.margin[3];
                switch (style.align_items) {
                    case ALIGN_CENTER:
                        child_box.y = (parent_box.content_height - child_box.height) / 2;
                        break;
                    case ALIGN_END:
                        child_box.y = parent_box.content_height - child_box.height - child_style.margin[2];
                        break;
                    case ALIGN_STRETCH:
                        child_box.height = parent_box.content_height - child_style.margin[0] - child_style.margin[2];
                        child_box.y = child_style.margin[0];
                        break;
                    default:
                        child_box.y = child_style.margin[0];
                        break;
                }
                if (total_flex_grow > 0 && child_style.flex_grow > 0 && remaining > 0) {
                    child_box.width += remaining * (child_style.flex_grow / total_flex_grow);
                }
                child_x += child_box.width + child_style.margin[1] + child_style.margin[3] + gap;
            } else {
                child_box.y = child_y + child_style.margin[0];
                switch (style.align_items) {
                    case ALIGN_CENTER:
                        child_box.x = (parent_box.content_width - child_box.width) / 2;
                        break;
                    case ALIGN_END:
                        child_box.x = parent_box.content_width - child_box.width - child_style.margin[1];
                        break;
                    case ALIGN_STRETCH:
                        child_box.width = parent_box.content_width - child_style.margin[1] - child_style.margin[3];
                        child_box.x = child_style.margin[3];
                        break;
                    default:
                        child_box.x = child_style.margin[3];
                        break;
                }
                if (total_flex_grow > 0 && child_style.flex_grow > 0 && remaining > 0) {
                    child_box.height += remaining * (child_style.flex_grow / total_flex_grow);
                }
                child_y += child_box.height + child_style.margin[0] + child_style.margin[2] + gap;
            }
        } else {
            // Block layout: stack vertically
            child_box.x = child_style.margin[3];
            child_box.y = child_y + child_style.margin[0];
            child_y += child_box.height + child_style.margin[0] + child_style.margin[2];
        }

        // Update content box
        child_box.content_x = child_box.x + child_style.border_width[3] + child_style.padding[3];
        child_box.content_y = child_box.y + child_style.border_width[0] + child_style.padding[0];
        child_box.content_width = child_box.width -
            child_style.padding[1] - child_style.padding[3] -
            child_style.border_width[1] - child_style.border_width[3];
        child_box.content_height = child_box.height -
            child_style.padding[0] - child_style.padding[2] -
            child_style.border_width[0] - child_style.border_width[2];

        boxes[idx] = child_box;
    }

    // Update parent height to contain children (if auto)
    ComputedStyle parent_style = styles[parent_idx];
    if (parent_style.height <= 0 && child_count > 0) {
        float content_height;
        if (is_flex && is_row) {
            content_height = 0;
            for (uint c = 0; c < child_count; c++) {
                LayoutBox child_box = boxes[children[c]];
                ComputedStyle child_style = styles[children[c]];
                float h = child_box.height + child_style.margin[0] + child_style.margin[2];
                if (h > content_height) content_height = h;
            }
        } else {
            content_height = child_y;
        }
        parent_box.content_height = content_height;
        parent_box.height = content_height +
            parent_style.padding[0] + parent_style.padding[2] +
            parent_style.border_width[0] + parent_style.border_width[2];
        boxes[parent_idx] = parent_box;
    }
}

// =============================================================================
// LEVEL-PARALLEL LAYOUT (Issue #89)
// All 1024 threads participate in every pass - no single-thread bottlenecks
// =============================================================================

// =============================================================================
// O(1) DEPTH BUFFER (Issue #130)
// Pre-compute depths using level-parallel algorithm instead of O(depth) walks
// =============================================================================

// Phase 1: Initialize depths - mark roots as depth 0, others as uncomputed
kernel void init_depths(
    device const Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    if (elements[gid].parent < 0) {
        depths[gid] = 0;  // Root node
    } else {
        depths[gid] = 0xFFFFFFFF;  // Not yet computed
    }
}

// Phase 2: Propagate depths level by level
// Each dispatch processes all elements whose parent is at current_level
kernel void propagate_depths(
    device const Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    constant uint& current_level [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    device atomic_uint* changed [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != 0xFFFFFFFF) return;  // Already computed

    int parent = elements[gid].parent;
    if (parent < 0) return;

    uint parent_depth = depths[parent];
    if (parent_depth == current_level) {
        // Parent is at current level, so we're at current_level + 1
        depths[gid] = current_level + 1;
        atomic_fetch_add_explicit(changed, 1, memory_order_relaxed);
    }
}

// Legacy kernel: Compute tree depth for each element using O(depth) parent walk
// Kept for fallback/compatibility - use init_depths + propagate_depths for O(1)
kernel void layout_compute_depths(
    device const Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    device atomic_uint* max_depth [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    // Walk up to root counting depth
    uint d = 0;
    int parent = elements[gid].parent;
    while (parent >= 0 && d < 256) {
        d++;
        parent = elements[parent].parent;
    }
    depths[gid] = d;
    atomic_fetch_max_explicit(max_depth, d, memory_order_relaxed);
}

// Pass 3b: Sum child heights for elements at specific level (LEVEL-PARALLEL, bottom-up)
kernel void layout_sum_heights(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    constant uint& current_level [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;  // Only process this level

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    // Skip text nodes - they already have intrinsic heights from pass 1
    if (elem.element_type == ELEM_TEXT) return;

    // Skip hidden elements
    if (style.display == DISPLAY_NONE) return;

    // If explicit height, keep it
    if (style.height > 0) return;

    // Skip leaf elements that already have height (e.g., from intrinsic sizing)
    if (elem.first_child < 0 && boxes[gid].height > 0) return;

    // Sum heights of children WITH MARGIN COLLAPSING (CSS spec)
    // Adjacent vertical margins collapse into the larger one
    float total_h = 0;
    float prev_margin_bottom = 0;
    bool is_first_child = true;

    int child = elem.first_child;
    while (child >= 0) {
        ComputedStyle child_style = styles[child];

        // FIX: Skip out-of-flow elements (absolute/fixed positioned)
        // These don't contribute to parent's height calculation
        if (child_style.display == DISPLAY_NONE ||
            child_style.position == POSITION_ABSOLUTE ||
            child_style.position == POSITION_FIXED) {
            child = elements[child].next_sibling;
            continue;
        }

        {
            float child_height = boxes[child].height;
            float child_margin_top = child_style.margin[0];
            float child_margin_bottom = child_style.margin[2];

            // Empty blocks: their margins collapse together
            // (top and bottom margin become one collapsed margin)
            if (child_height <= 0.1) {
                // For empty element, just track the max margin for collapsing
                prev_margin_bottom = max(prev_margin_bottom, max(child_margin_top, child_margin_bottom));
            } else {
                // MARGIN COLLAPSING: use max of adjacent margins, not sum
                if (is_first_child) {
                    // First child: its top margin might collapse with parent (handled elsewhere)
                    // For now, just use its margin
                    total_h += child_margin_top;
                } else {
                    // Collapse previous sibling's bottom margin with this one's top margin
                    float collapsed_margin = max(prev_margin_bottom, child_margin_top);
                    total_h += collapsed_margin;
                }

                total_h += child_height;
                prev_margin_bottom = child_margin_bottom;
                is_first_child = false;
            }
        }
        child = elements[child].next_sibling;
    }

    // Add last child's bottom margin (if any children)
    if (!is_first_child) {
        total_h += prev_margin_bottom;
    }

    // Add padding and border
    float padding_v = style.padding[0] + style.padding[2];
    float border_v = style.border_width[0] + style.border_width[2];

    boxes[gid].height = total_h + padding_v + border_v;
    boxes[gid].content_height = total_h;
}

// =============================================================================
// Issue #128: O(1) Sibling Cumulative Heights
// Pre-compute cumulative Y offsets for O(1) lookup in positioning pass
// =============================================================================

// Cumulative info struct: stores Y offset and margin info for collapsing
struct CumulativeInfo {
    float y_offset;           // Cumulative Y position (sum of preceding siblings' heights + margins)
    float prev_margin_bottom; // Previous sibling's bottom margin (for margin collapsing)
    uint flags;               // Bit 0: has_visible_sibling
    float _padding;
};

// Pass 3b-2: Compute cumulative heights (PARENT-PARALLEL)
// Each thread processes one parent's children sequentially to compute cumulative heights.
// This is parallel across parents (which is the common case) while being sequential
// within sibling chains (which have data dependencies).
kernel void compute_cumulative_heights(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device const LayoutBox* boxes [[buffer(2)]],
    device CumulativeInfo* cumulative [[buffer(3)]],
    device const uint* depths [[buffer(4)]],
    constant uint& current_level [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    // Only process elements at PARENT level (one level above current)
    // Each thread handles one parent's children
    if (current_level == 0) {
        // For level 0, only process elements with parent == -1 (roots)
        if (depths[gid] != 0) return;
        // Roots have no siblings, just initialize
        cumulative[gid].y_offset = 0;
        cumulative[gid].prev_margin_bottom = 0;
        cumulative[gid].flags = 0;
        return;
    }

    // Check if this element is a parent of elements at current_level
    // (i.e., this element is at current_level - 1)
    if (depths[gid] != current_level - 1) return;

    Element parent_elem = elements[gid];
    int first_child = parent_elem.first_child;

    if (first_child < 0) return;  // No children

    // Process this parent's children sequentially
    float y = 0;
    float prev_margin_bottom = 0;
    bool found_visible_sibling = false;

    int child = first_child;
    while (child >= 0) {
        ComputedStyle child_style = styles[child];

        // Skip out-of-flow elements
        if (child_style.display == DISPLAY_NONE ||
            child_style.position == POSITION_ABSOLUTE ||
            child_style.position == POSITION_FIXED) {
            // Still need to set cumulative for this element
            cumulative[child].y_offset = y;
            cumulative[child].prev_margin_bottom = prev_margin_bottom;
            cumulative[child].flags = found_visible_sibling ? 1u : 0u;
            child = elements[child].next_sibling;
            continue;
        }

        // Store cumulative position for this child
        cumulative[child].y_offset = y;
        cumulative[child].prev_margin_bottom = prev_margin_bottom;
        cumulative[child].flags = found_visible_sibling ? 1u : 0u;

        // Now accumulate this child's contribution for the NEXT sibling
        float child_height = boxes[child].height;
        float child_margin_top = child_style.margin[0];
        float child_margin_bottom = child_style.margin[2];

        // Handle empty elements
        if (child_height <= 0.1) {
            prev_margin_bottom = max(prev_margin_bottom, max(child_margin_top, child_margin_bottom));
        } else {
            if (!found_visible_sibling) {
                y += child_margin_top;
            } else {
                y += max(prev_margin_bottom, child_margin_top);
            }
            y += child_height;
            prev_margin_bottom = child_margin_bottom;
            found_visible_sibling = true;
        }

        child = elements[child].next_sibling;
    }
}

// Pass 3c: Position siblings at specific level (LEVEL-PARALLEL, top-down)
// Issue #128: Now uses O(1) cumulative height lookup
kernel void layout_position_siblings(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    device const CumulativeInfo* cumulative [[buffer(4)]],  // Issue #128: O(1) lookup
    constant uint& current_level [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;  // Only process this level

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    // Skip hidden elements and out-of-flow elements
    if (style.display == DISPLAY_NONE) return;
    if (style.position == POSITION_ABSOLUTE || style.position == POSITION_FIXED) return;

    // Check if parent is flex - if so, handle flex layout differently
    if (elem.parent >= 0) {
        ComputedStyle parent_style = styles[elem.parent];
        if (parent_style.display == DISPLAY_FLEX) {
            // Flex children are positioned by the flex algorithm, not block flow
            // For flex row: items are positioned horizontally
            // For flex column: items are positioned vertically (like block flow but with flex features)
            bool is_row = (parent_style.flex_direction == FLEX_ROW);

            if (is_row) {
                // Flex row: X position is cumulative, Y depends on align-items
                CumulativeInfo info = cumulative[gid];
                float x = info.y_offset;  // Reuse cumulative for X in flex row

                // Add horizontal margins
                x += style.margin[3];

                boxes[gid].x = x;
                boxes[gid].y = style.margin[0];  // Will be adjusted by align-items later

                // Update content box position
                boxes[gid].content_x = boxes[gid].x + style.border_width[3] + style.padding[3];
                boxes[gid].content_y = boxes[gid].y + style.border_width[0] + style.padding[0];
                return;
            }
            // Flex column falls through to block-like Y positioning
        }
    }

    // Block flow: Y position from cumulative heights
    CumulativeInfo info = cumulative[gid];
    float y = info.y_offset;

    // Add this element's top margin with margin collapsing
    float my_margin_top = style.margin[0];
    bool found_visible_sibling = (info.flags & 1) != 0;

    if (found_visible_sibling) {
        y += max(info.prev_margin_bottom, my_margin_top);
    } else {
        y += my_margin_top;  // First visible child
    }

    boxes[gid].y = y;
    boxes[gid].x = style.margin[3];  // Left margin

    // Update content box position (relative)
    boxes[gid].content_x = boxes[gid].x + style.border_width[3] + style.padding[3];
    boxes[gid].content_y = boxes[gid].y + style.border_width[0] + style.padding[0];
}

// Pass 3d: Finalize absolute positions (ALL THREADS, top-down by level)
kernel void layout_finalize_level(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    constant uint& current_level [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    if (elem.parent >= 0) {
        LayoutBox parent_box = boxes[elem.parent];
        boxes[gid].x += parent_box.content_x;
        boxes[gid].y += parent_box.content_y;
        boxes[gid].content_x += parent_box.content_x;
        boxes[gid].content_y += parent_box.content_y;
    }
}

// Pass NEW: Propagate widths and calculate text heights (top-down by level)
// This pass fixes the whitespace issue by calculating text heights AFTER parent widths are known
kernel void propagate_widths_and_text(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    constant uint& current_level [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    constant Viewport& viewport [[buffer(6)]],
    device const uint8_t* text [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    // Skip hidden elements
    if (style.display == DISPLAY_NONE) return;

    // Get parent's content width (which is correct because parent was processed in previous level)
    float parent_content_width = viewport.width;
    if (elem.parent >= 0) {
        parent_content_width = boxes[elem.parent].content_width;
        if (parent_content_width <= 0) {
            parent_content_width = viewport.width;
        }
    }

    // For block elements without explicit width, set width from parent
    if ((style.display == DISPLAY_BLOCK || elem.element_type == ELEM_TEXT) && style.width <= 0) {
        float margin_h = style.margin[1] + style.margin[3];
        float padding_h = style.padding[1] + style.padding[3];
        float border_h = style.border_width[1] + style.border_width[3];

        boxes[gid].width = parent_content_width - margin_h;
        boxes[gid].content_width = boxes[gid].width - padding_h - border_h;
    }

    // TEXT ELEMENTS: Calculate wrapped height now that we have correct parent width
    if (elem.element_type == ELEM_TEXT && elem.text_length > 0) {
        float font_size = style.font_size > 0 ? style.font_size : 16.0;
        float line_height_mult = style.line_height > 0 ? style.line_height : 1.2;

        float container_width = parent_content_width > 0 ? parent_content_width : viewport.width;

        float wrapped_height = text_height_wrapped(
            text,
            elem.text_start,
            elem.text_length,
            font_size,
            line_height_mult,
            container_width
        );

        boxes[gid].height = wrapped_height;
        boxes[gid].content_height = wrapped_height;
        boxes[gid].width = container_width;
        boxes[gid].content_width = container_width;
    }
}

// Legacy: Keep old sequential version for compatibility during transition
kernel void layout_children_sequential(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant Viewport& viewport [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    // Post-order traversal using explicit stack
    int stack[256];
    int visited[256];
    int stack_top = 0;

    for (int i = int(element_count) - 1; i >= 0; i--) {
        if (elements[i].parent < 0) {
            stack[stack_top] = i;
            visited[stack_top] = 0;
            stack_top++;
        }
    }

    while (stack_top > 0) {
        stack_top--;
        int idx = stack[stack_top];
        int is_visited = visited[stack_top];

        if (is_visited) {
            layout_parent_children(elements, styles, boxes, idx);
        } else {
            stack[stack_top] = idx;
            visited[stack_top] = 1;
            stack_top++;

            Element elem = elements[idx];
            int children[MAX_CHILDREN];
            int child_count = 0;

            int child_idx = elem.first_child;
            while (child_idx >= 0 && child_count < MAX_CHILDREN) {
                children[child_count++] = child_idx;
                child_idx = elements[child_idx].next_sibling;
            }

            for (int c = child_count - 1; c >= 0; c--) {
                if (stack_top < 256) {
                    stack[stack_top] = children[c];
                    visited[stack_top] = 0;
                    stack_top++;
                }
            }
        }
    }
}

// Pass 4: Convert relative positions to absolute (top-down)
kernel void finalize_positions(
    device const Element* elements [[buffer(0)]],
    device LayoutBox* boxes [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Only thread 0 runs this for tree traversal
    if (gid != 0) return;

    for (uint i = 0; i < element_count; i++) {
        Element elem = elements[i];

        if (elem.parent >= 0) {
            LayoutBox parent_box = boxes[elem.parent];
            LayoutBox box = boxes[i];

            // Add parent's content box position to get absolute position
            box.x += parent_box.content_x;
            box.y += parent_box.content_y;
            box.content_x += parent_box.content_x;
            box.content_y += parent_box.content_y;

            boxes[i] = box;
        }
    }
}
