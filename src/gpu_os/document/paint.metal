#include <metal_stdlib>
using namespace metal;

// Element type for text
constant uint ELEM_TEXT = 100;

// Vertex flags
constant uint FLAG_BACKGROUND = 1;
constant uint FLAG_BORDER = 2;
constant uint FLAG_TEXT = 4;

// Limits
constant uint MAX_VERTICES_PER_ELEMENT = 64;  // Background(4) + Border(16) + rounded corners
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

struct LayoutBox {
    float x;
    float y;
    float width;
    float height;
    float content_x;
    float content_y;
    float content_width;
    float content_height;
    float scroll_width;
    float scroll_height;
    float _padding[6];
};

struct ComputedStyle {
    // Display
    uint display;

    // Box model
    float width;
    float height;
    float margin[4];
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

struct PaintVertex {
    float2 position;      // 8 bytes
    float2 tex_coord;     // 8 bytes
    float4 color;         // 16 bytes
    uint flags;           // 4 bytes
    uint _pad1;           // 4 bytes
    uint _pad2;           // 4 bytes
    uint _pad3;           // 4 bytes
};  // Total: 48 bytes (no uint3 alignment issue)

struct PaintCommand {
    uint element_index;
    uint vertex_start;
    uint vertex_count;
    uint texture_id;
};

struct Viewport {
    float width;
    float height;
    float2 _padding;
};

// Atomic counter for vertex allocation
struct VertexCounter {
    atomic_uint count;
};

// Count vertices needed per element (first pass)
kernel void count_vertices(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device uint* vertex_counts [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    uint count = 0;

    // Hidden elements get no vertices
    if (style.display == 0) {  // DISPLAY_NONE
        vertex_counts[gid] = 0;
        return;
    }

    // Background (4 vertices for quad, or more for rounded corners)
    if (style.background_color[3] > 0) {
        if (style.border_radius > 0) {
            // Rounded rectangle: center quad + 4 corner arcs (8 segments each)
            count += 4 + 4 * 8 * 3;  // ~100 vertices for smooth corners
        } else {
            count += 4;  // Simple quad
        }
    }

    // Border (4 trapezoids, 4 vertices each = 16)
    if (style.border_width[0] > 0 || style.border_width[1] > 0 ||
        style.border_width[2] > 0 || style.border_width[3] > 0) {
        count += 16;
    }

    // Text (4 vertices per character)
    if (elem.element_type == ELEM_TEXT) {
        count += elem.text_length * 4;
    }

    vertex_counts[gid] = count;
}

// Compute prefix sum for vertex offsets (single-threaded for simplicity)
kernel void compute_offsets(
    device uint* vertex_counts [[buffer(0)]],
    device uint* vertex_offsets [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Only thread 0 computes prefix sum
    if (gid != 0) return;

    uint offset = 0;
    for (uint i = 0; i < element_count; i++) {
        vertex_offsets[i] = offset;
        offset += vertex_counts[i];
    }
}

// Generate background vertices
kernel void generate_background_vertices(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint* vertex_offsets [[buffer(3)]],
    device PaintVertex* vertices [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    constant Viewport& viewport [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    ComputedStyle style = styles[gid];

    // Skip if no background
    if (style.background_color[3] <= 0) return;
    if (style.display == 0) return;

    LayoutBox box = boxes[gid];
    uint offset = vertex_offsets[gid];

    // Normalize positions to NDC [-1, 1]
    float2 scale = float2(2.0 / viewport.width, -2.0 / viewport.height);
    float2 bias = float2(-1.0, 1.0);

    float4 color = float4(
        style.background_color[0],
        style.background_color[1],
        style.background_color[2],
        style.background_color[3] * style.opacity
    );

    // Simple quad (no border radius for now)
    float left = box.x;
    float right = box.x + box.width;
    float top = box.y;
    float bottom = box.y + box.height;

    // Top-left
    vertices[offset + 0].position = float2(left, top) * scale + bias;
    vertices[offset + 0].tex_coord = float2(0, 0);
    vertices[offset + 0].color = color;
    vertices[offset + 0].flags = FLAG_BACKGROUND;

    // Top-right
    vertices[offset + 1].position = float2(right, top) * scale + bias;
    vertices[offset + 1].tex_coord = float2(1, 0);
    vertices[offset + 1].color = color;
    vertices[offset + 1].flags = FLAG_BACKGROUND;

    // Bottom-right
    vertices[offset + 2].position = float2(right, bottom) * scale + bias;
    vertices[offset + 2].tex_coord = float2(1, 1);
    vertices[offset + 2].color = color;
    vertices[offset + 2].flags = FLAG_BACKGROUND;

    // Bottom-left
    vertices[offset + 3].position = float2(left, bottom) * scale + bias;
    vertices[offset + 3].tex_coord = float2(0, 1);
    vertices[offset + 3].color = color;
    vertices[offset + 3].flags = FLAG_BACKGROUND;
}

// Generate border vertices
kernel void generate_border_vertices(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint* vertex_offsets [[buffer(3)]],
    device const uint* vertex_counts [[buffer(4)]],
    device PaintVertex* vertices [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    constant Viewport& viewport [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    ComputedStyle style = styles[gid];

    // Skip if no border
    bool has_border = style.border_width[0] > 0 || style.border_width[1] > 0 ||
                      style.border_width[2] > 0 || style.border_width[3] > 0;
    if (!has_border) return;
    if (style.display == 0) return;

    LayoutBox box = boxes[gid];

    // Offset after background vertices
    uint background_count = (style.background_color[3] > 0) ? 4 : 0;
    uint offset = vertex_offsets[gid] + background_count;

    float2 scale = float2(2.0 / viewport.width, -2.0 / viewport.height);
    float2 bias = float2(-1.0, 1.0);

    float4 color = float4(
        style.border_color[0],
        style.border_color[1],
        style.border_color[2],
        style.border_color[3] * style.opacity
    );

    float left = box.x;
    float right = box.x + box.width;
    float top = box.y;
    float bottom = box.y + box.height;

    float bt = style.border_width[0];  // top
    float br = style.border_width[1];  // right
    float bb = style.border_width[2];  // bottom
    float bl = style.border_width[3];  // left

    // Top border (trapezoid)
    if (bt > 0) {
        vertices[offset + 0].position = float2(left, top) * scale + bias;
        vertices[offset + 1].position = float2(right, top) * scale + bias;
        vertices[offset + 2].position = float2(right - br, top + bt) * scale + bias;
        vertices[offset + 3].position = float2(left + bl, top + bt) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
    offset += 4;

    // Right border
    if (br > 0) {
        vertices[offset + 0].position = float2(right, top) * scale + bias;
        vertices[offset + 1].position = float2(right, bottom) * scale + bias;
        vertices[offset + 2].position = float2(right - br, bottom - bb) * scale + bias;
        vertices[offset + 3].position = float2(right - br, top + bt) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
    offset += 4;

    // Bottom border
    if (bb > 0) {
        vertices[offset + 0].position = float2(right, bottom) * scale + bias;
        vertices[offset + 1].position = float2(left, bottom) * scale + bias;
        vertices[offset + 2].position = float2(left + bl, bottom - bb) * scale + bias;
        vertices[offset + 3].position = float2(right - br, bottom - bb) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
    offset += 4;

    // Left border
    if (bl > 0) {
        vertices[offset + 0].position = float2(left, bottom) * scale + bias;
        vertices[offset + 1].position = float2(left, top) * scale + bias;
        vertices[offset + 2].position = float2(left + bl, top + bt) * scale + bias;
        vertices[offset + 3].position = float2(left + bl, bottom - bb) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
}

// Generate text vertices (simplified - assumes monospace font)
kernel void generate_text_vertices(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint8_t* text_buffer [[buffer(3)]],
    device const uint* vertex_offsets [[buffer(4)]],
    device const uint* vertex_counts [[buffer(5)]],
    device PaintVertex* vertices [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    constant Viewport& viewport [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];

    // Skip non-text elements
    if (elem.element_type != ELEM_TEXT) return;
    if (elem.text_length == 0) return;

    ComputedStyle style = styles[gid];
    if (style.display == 0) return;

    LayoutBox box = boxes[gid];

    // Calculate offset (after background and border)
    uint background_count = (style.background_color[3] > 0) ? 4 : 0;
    bool has_border = style.border_width[0] > 0 || style.border_width[1] > 0 ||
                      style.border_width[2] > 0 || style.border_width[3] > 0;
    uint border_count = has_border ? 16 : 0;
    uint offset = vertex_offsets[gid] + background_count + border_count;

    float2 scale = float2(2.0 / viewport.width, -2.0 / viewport.height);
    float2 bias = float2(-1.0, 1.0);

    float4 color = float4(
        style.color[0],
        style.color[1],
        style.color[2],
        style.color[3] * style.opacity
    );

    // Glyph dimensions (approximate monospace)
    float glyph_width = style.font_size * 0.6;
    float glyph_height = style.font_size;

    float x = box.x;
    float y = box.y;

    // Generate quad for each character
    for (uint i = 0; i < elem.text_length; i++) {
        uint char_code = text_buffer[elem.text_start + i];

        // Skip whitespace for now (but still advance position)
        if (char_code != ' ' && char_code != '\n' && char_code != '\t') {
            uint v = offset + i * 4;

            // Top-left
            vertices[v + 0].position = float2(x, y) * scale + bias;
            vertices[v + 0].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16) / 16.0);
            vertices[v + 0].color = color;
            vertices[v + 0].flags = FLAG_TEXT;

            // Top-right
            vertices[v + 1].position = float2(x + glyph_width, y) * scale + bias;
            vertices[v + 1].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16) / 16.0);
            vertices[v + 1].color = color;
            vertices[v + 1].flags = FLAG_TEXT;

            // Bottom-right
            vertices[v + 2].position = float2(x + glyph_width, y + glyph_height) * scale + bias;
            vertices[v + 2].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16 + 1) / 16.0);
            vertices[v + 2].color = color;
            vertices[v + 2].flags = FLAG_TEXT;

            // Bottom-left
            vertices[v + 3].position = float2(x, y + glyph_height) * scale + bias;
            vertices[v + 3].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16 + 1) / 16.0);
            vertices[v + 3].color = color;
            vertices[v + 3].flags = FLAG_TEXT;
        }

        x += glyph_width;
    }
}
