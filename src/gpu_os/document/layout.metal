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
    uint text_start;
    uint text_length;
    uint token_index;
    uint _padding;
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

    float _padding[2];
};

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

// Calculate intrinsic content width for text
float text_width(uint text_length, float font_size) {
    // Approximation: average character width is ~0.6 * font_size
    return float(text_length) * font_size * 0.6;
}

// Calculate intrinsic content height for text
float text_height(float font_size, float line_height) {
    return font_size * line_height;
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
        float tw = text_width(elem.text_length, style.font_size);
        float th = text_height(style.font_size, style.line_height);
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

// Pass 3: Layout children sequentially (handles sibling positioning)
// This must run on CPU or single-threaded for correctness
kernel void layout_children_sequential(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant Viewport& viewport [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Only thread 0 runs this
    if (gid != 0) return;

    for (uint i = 0; i < element_count; i++) {
        Element elem = elements[i];
        ComputedStyle style = styles[i];

        if (style.display == DISPLAY_NONE) continue;
        if (elem.first_child < 0) continue;

        LayoutBox parent_box = boxes[i];
        float child_y = 0;
        float child_x = 0;

        // Flex layout handling
        bool is_flex = (style.display == DISPLAY_FLEX);
        bool is_row = (style.flex_direction == FLEX_ROW);

        // Collect children for flex layout
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
                default: // JUSTIFY_START
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

                    // Align items (cross-axis)
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
                        default: // ALIGN_START
                            child_box.y = child_style.margin[0];
                            break;
                    }

                    // Distribute flex-grow
                    if (total_flex_grow > 0 && child_style.flex_grow > 0 && remaining > 0) {
                        float extra = remaining * (child_style.flex_grow / total_flex_grow);
                        child_box.width += extra;
                    }

                    child_x += child_box.width + child_style.margin[1] + child_style.margin[3] + gap;
                } else {
                    child_box.y = child_y + child_style.margin[0];

                    // Align items (cross-axis)
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
                        default: // ALIGN_START
                            child_box.x = child_style.margin[3];
                            break;
                    }

                    // Distribute flex-grow
                    if (total_flex_grow > 0 && child_style.flex_grow > 0 && remaining > 0) {
                        float extra = remaining * (child_style.flex_grow / total_flex_grow);
                        child_box.height += extra;
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
        ComputedStyle parent_style = styles[i];
        if (parent_style.height <= 0 && child_count > 0) {
            float content_height;
            if (is_flex && is_row) {
                // Row flex: height is max child height
                content_height = 0;
                for (uint c = 0; c < child_count; c++) {
                    LayoutBox child_box = boxes[children[c]];
                    ComputedStyle child_style = styles[children[c]];
                    float h = child_box.height + child_style.margin[0] + child_style.margin[2];
                    if (h > content_height) content_height = h;
                }
            } else {
                // Column or block: height is sum
                content_height = child_y;
            }
            parent_box.content_height = content_height;
            parent_box.height = content_height +
                parent_style.padding[0] + parent_style.padding[2] +
                parent_style.border_width[0] + parent_style.border_width[2];
            boxes[i] = parent_box;
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
