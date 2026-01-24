#include <metal_stdlib>
using namespace metal;

// Display values (from style)
constant uint DISPLAY_NONE = 0;

// Pointer events values
constant uint POINTER_EVENTS_AUTO = 0;
constant uint POINTER_EVENTS_NONE = 1;

// Visibility values
constant uint VISIBILITY_VISIBLE = 0;
constant uint VISIBILITY_HIDDEN = 1;

struct LayoutBox {
    float x;
    float y;
    float width;
    float height;
    uint element_index;
    uint _padding[3];
};

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
    uint display;
    float width;
    float height;
    float margin[4];
    float padding[4];
    uint flex_direction;
    uint justify_content;
    uint align_items;
    float flex_grow;
    float flex_shrink;
    float color[4];
    float font_size;
    float line_height;
    uint font_weight;
    uint text_align;
    float background_color[4];
    float border_width[4];
    float border_color[4];
    float border_radius;
    float opacity;
    // CSS Positioning
    uint position;
    float top;
    float right_;
    float bottom;
    float left;
    int z_index;
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

struct HitTestParams {
    float mouse_x;
    float mouse_y;
    float scroll_x;
    float scroll_y;
    uint element_count;
    uint _padding[3];
};

struct HitResult {
    uint element_id;       // Element index that was hit
    int depth;             // Depth in tree (higher = more nested)
    uint element_type;     // Element type for quick checking
    uint _padding;
};

/// GPU hit test kernel
/// Tests all elements in parallel to find the one under the cursor
/// Uses atomic operations to find the deepest element (innermost wins)
kernel void hit_test(
    device const LayoutBox* boxes [[buffer(0)]],
    device const Element* elements [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    constant HitTestParams& params [[buffer(3)]],
    device atomic_uint* hit_element_id [[buffer(4)]],
    device atomic_int* hit_depth [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) return;

    LayoutBox box = boxes[gid];
    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    // Skip invisible elements
    if (style.display == DISPLAY_NONE) return;
    if (style.opacity < 0.01) return;

    // Skip elements with zero size
    if (box.width <= 0 || box.height <= 0) return;

    // Calculate adjusted position (account for scroll)
    float pos_x = params.mouse_x + params.scroll_x;
    float pos_y = params.mouse_y + params.scroll_y;

    // Check if point is inside box
    if (pos_x >= box.x && pos_x <= box.x + box.width &&
        pos_y >= box.y && pos_y <= box.y + box.height) {

        // Calculate depth by counting parents
        int depth = 0;
        int parent = elem.parent;
        while (parent >= 0 && depth < 100) {  // Max depth guard
            depth++;
            parent = elements[parent].parent;
        }

        // Atomic compare to find deepest element (innermost wins)
        int old_depth;
        do {
            old_depth = atomic_load_explicit(hit_depth, memory_order_relaxed);
            if (depth <= old_depth) return;  // Shallower depth, skip
        } while (!atomic_compare_exchange_weak_explicit(
            hit_depth, &old_depth, depth,
            memory_order_relaxed, memory_order_relaxed));

        // Won the race - store our element ID
        atomic_store_explicit(hit_element_id, gid, memory_order_relaxed);
    }
}

/// Hit test returning all elements under cursor
/// Each thread checks if its element contains the point
/// Results written to a boolean array
kernel void hit_test_all(
    device const LayoutBox* boxes [[buffer(0)]],
    device const Element* elements [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    constant HitTestParams& params [[buffer(3)]],
    device uint* hit_flags [[buffer(4)]],  // 1 if hit, 0 otherwise
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.element_count) return;

    LayoutBox box = boxes[gid];
    ComputedStyle style = styles[gid];

    // Default to not hit
    hit_flags[gid] = 0;

    // Skip invisible elements
    if (style.display == DISPLAY_NONE) return;
    if (style.opacity < 0.01) return;

    // Skip elements with zero size
    if (box.width <= 0 || box.height <= 0) return;

    // Calculate adjusted position
    float pos_x = params.mouse_x + params.scroll_x;
    float pos_y = params.mouse_y + params.scroll_y;

    // Check if point is inside box
    if (pos_x >= box.x && pos_x <= box.x + box.width &&
        pos_y >= box.y && pos_y <= box.y + box.height) {
        hit_flags[gid] = 1;
    }
}
