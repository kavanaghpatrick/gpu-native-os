#include <metal_stdlib>
using namespace metal;

// Element types
constant uint ELEM_IMG = 14;
constant uint ELEM_TEXT = 100;

// Vertex flags
constant uint FLAG_BACKGROUND = 1;
constant uint FLAG_BORDER = 2;
constant uint FLAG_TEXT = 4;
constant uint FLAG_IMAGE = 8;

// Limits
constant uint MAX_VERTICES_PER_ELEMENT = 64;  // Background(4) + Border(16) + rounded corners
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

    // Background (4 vertices for quad)
    // Note: rounded corners not yet implemented, so always count 4
    if (style.background_color[3] > 0) {
        count += 4;  // Simple quad (even with border_radius, we render as simple quad for now)
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

    // Image (4 vertices for quad)
    if (elem.element_type == ELEM_IMG) {
        count += 4;
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

    // Helper to write a border quad - ALWAYS write all 4 vertices to avoid gaps
    // Use zero-alpha color for borders with width 0 so they're invisible but don't leave garbage

    // Top border (trapezoid) - always write, use zero-alpha if bt == 0
    {
        float4 top_color = (bt > 0) ? color : float4(0, 0, 0, 0);
        vertices[offset + 0].position = float2(left, top) * scale + bias;
        vertices[offset + 1].position = float2(right, top) * scale + bias;
        vertices[offset + 2].position = float2(right - br, top + bt) * scale + bias;
        vertices[offset + 3].position = float2(left + bl, top + bt) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = top_color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
    offset += 4;

    // Right border - always write, use zero-alpha if br == 0
    {
        float4 right_color = (br > 0) ? color : float4(0, 0, 0, 0);
        vertices[offset + 0].position = float2(right, top) * scale + bias;
        vertices[offset + 1].position = float2(right, bottom) * scale + bias;
        vertices[offset + 2].position = float2(right - br, bottom - bb) * scale + bias;
        vertices[offset + 3].position = float2(right - br, top + bt) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = right_color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
    offset += 4;

    // Bottom border - always write, use zero-alpha if bb == 0
    {
        float4 bottom_color = (bb > 0) ? color : float4(0, 0, 0, 0);
        vertices[offset + 0].position = float2(right, bottom) * scale + bias;
        vertices[offset + 1].position = float2(left, bottom) * scale + bias;
        vertices[offset + 2].position = float2(left + bl, bottom - bb) * scale + bias;
        vertices[offset + 3].position = float2(right - br, bottom - bb) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = bottom_color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
    offset += 4;

    // Left border - always write, use zero-alpha if bl == 0
    {
        float4 left_color = (bl > 0) ? color : float4(0, 0, 0, 0);
        vertices[offset + 0].position = float2(left, bottom) * scale + bias;
        vertices[offset + 1].position = float2(left, top) * scale + bias;
        vertices[offset + 2].position = float2(left + bl, top + bt) * scale + bias;
        vertices[offset + 3].position = float2(left + bl, bottom - bb) * scale + bias;
        for (int i = 0; i < 4; i++) {
            vertices[offset + i].tex_coord = float2(0, 0);
            vertices[offset + i].color = left_color;
            vertices[offset + i].flags = FLAG_BORDER;
        }
    }
}

// LineBox structure for multi-line text
struct LineBox {
    uint element_index;
    uint char_start;
    uint char_end;
    float width;
    float x;
    float y;
    float _padding[2];
};

// Calculate glyph advance width
float glyph_advance(uint char_code, float font_size) {
    if (char_code == ' ') return font_size * 0.3;
    if (char_code == '\t') return font_size * 1.2;
    if (char_code == '\n' || char_code == '\r') return 0.0;
    return font_size * 0.6;
}

// Generate text vertices with multi-line support
// If line_count > 0, uses LineBox data for positioning
// Otherwise falls back to single-line rendering
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

    float font_size = style.font_size > 0 ? style.font_size : 16.0;
    float line_height = style.line_height > 0 ? style.line_height : 1.2;
    float line_height_px = font_size * line_height;
    float container_width = box.content_width > 0 ? box.content_width : 10000.0;

    // Simple single-pass text wrapping directly in the paint kernel
    // Use content box position (accounts for padding/border)
    float x = box.content_x;
    float y = box.content_y;
    float line_start_x = box.content_x;

    // Track word boundaries for wrapping
    float word_start_x = x;
    uint word_start_char = 0;
    bool in_word = false;

    for (uint i = 0; i < elem.text_length; i++) {
        uint char_code = text_buffer[elem.text_start + i];
        float advance = glyph_advance(char_code, font_size);

        // Handle newlines - don't skip, we need to write placeholder vertices
        // The position update happens after vertex generation
        bool is_newline = (char_code == '\n');
        if (is_newline) {
            // Position will be reset after writing placeholder vertices
            in_word = false;
        }

        // Check for word boundaries
        bool is_space = (char_code == ' ' || char_code == '\t');
        if (is_space && in_word) {
            in_word = false;
        } else if (!is_space && !in_word) {
            in_word = true;
            word_start_x = x;
            word_start_char = i;
        }

        // Check if we need to wrap
        float current_line_width = x - line_start_x + advance;
        if (current_line_width > container_width && x > line_start_x) {
            // Try to wrap at word boundary
            if (in_word && word_start_x > line_start_x) {
                // Move current word to next line
                // We need to adjust vertices already written for this word
                float word_offset_x = word_start_x - line_start_x;

                // Move to next line
                x = line_start_x;
                y += line_height_px;

                // Reposition word vertices (shift back to start of new line)
                // Use character indices since vertices are allocated per-character
                for (uint ci = word_start_char; ci < i; ci++) {
                    uint v = offset + ci * 4;
                    for (int vv = 0; vv < 4; vv++) {
                        // Adjust X position
                        float old_x_ndc = vertices[v + vv].position.x;
                        float old_x = (old_x_ndc - bias.x) / scale.x;
                        float new_x = old_x - word_offset_x;
                        vertices[v + vv].position.x = new_x * scale.x + bias.x;

                        // Adjust Y position
                        float old_y_ndc = vertices[v + vv].position.y;
                        float old_y = (old_y_ndc - bias.y) / scale.y;
                        float new_y = old_y + line_height_px;
                        vertices[v + vv].position.y = new_y * scale.y + bias.y;
                    }
                }

                // Continue from where word ended
                x = line_start_x + (x - word_start_x);
                word_start_x = line_start_x;
            } else {
                // No word boundary - force break
                x = line_start_x;
                y += line_height_px;
            }
        }

        // Generate quad for ALL characters to avoid gaps in vertex buffer
        // Use zero-alpha for whitespace so they're invisible but don't leave garbage
        {
            uint v = offset + i * 4;  // Use 'i' (character index) not vertex_idx to maintain alignment
            float glyph_width = advance;
            float glyph_height = font_size;

            // Determine if this character should be visible
            bool is_visible = !is_space && char_code != '\r' && char_code != '\n';
            float4 vertex_color = is_visible ? color : float4(0, 0, 0, 0);  // Zero-alpha for whitespace

            // Top-left
            vertices[v + 0].position = float2(x, y) * scale + bias;
            vertices[v + 0].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16) / 16.0);
            vertices[v + 0].color = vertex_color;
            vertices[v + 0].flags = FLAG_TEXT;

            // Top-right
            vertices[v + 1].position = float2(x + glyph_width, y) * scale + bias;
            vertices[v + 1].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16) / 16.0);
            vertices[v + 1].color = vertex_color;
            vertices[v + 1].flags = FLAG_TEXT;

            // Bottom-right
            vertices[v + 2].position = float2(x + glyph_width, y + glyph_height) * scale + bias;
            vertices[v + 2].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16 + 1) / 16.0);
            vertices[v + 2].color = vertex_color;
            vertices[v + 2].flags = FLAG_TEXT;

            // Bottom-left
            vertices[v + 3].position = float2(x, y + glyph_height) * scale + bias;
            vertices[v + 3].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16 + 1) / 16.0);
            vertices[v + 3].color = vertex_color;
            vertices[v + 3].flags = FLAG_TEXT;
        }

        // Update position - special handling for newlines
        if (is_newline) {
            x = line_start_x;
            y += line_height_px;
        } else {
            x += advance;
        }
    }
}

// ============================================================================
// Issue #131: O(1) Two-Pass Text Line Layout - Pass 2
// ============================================================================

// Per-element line info header (16 bytes)
struct TextLineDataHeader {
    uint line_count;
    uint _padding[3];
};

// Pre-computed line information (16 bytes)
struct LineInfoPaint {
    uint char_start;
    uint char_end;
    float y_offset;
    float width;
};

constant uint MAX_LINES_PER_ELEMENT_PAINT = 64;

// Pass 2: Generate text vertices using pre-computed line data - O(1) per character
// Each thread processes one text element using LineInfo from Pass 1
kernel void generate_text_vertices_fast(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint8_t* text_buffer [[buffer(3)]],
    device const uint* vertex_offsets [[buffer(4)]],
    device const uint* vertex_counts [[buffer(5)]],
    device PaintVertex* vertices [[buffer(6)]],
    device const TextLineDataHeader* line_headers [[buffer(7)]],
    device const LineInfoPaint* line_info [[buffer(8)]],
    constant uint& element_count [[buffer(9)]],
    constant Viewport& viewport [[buffer(10)]],
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

    float font_size = style.font_size > 0 ? style.font_size : 16.0;

    // Get line data for this element
    uint line_count = line_headers[gid].line_count;
    uint line_base = gid * MAX_LINES_PER_ELEMENT_PAINT;

    // Handle edge case: no lines computed (fall back to single line)
    if (line_count == 0) {
        // Simple fallback: render all text on one line
        float x = box.content_x;
        float y = box.content_y;

        for (uint i = 0; i < elem.text_length; i++) {
            uint char_code = text_buffer[elem.text_start + i];
            float advance = glyph_advance(char_code, font_size);

            uint v = offset + i * 4;
            bool is_visible = (char_code != ' ' && char_code != '\t' &&
                              char_code != '\n' && char_code != '\r');
            float4 vertex_color = is_visible ? color : float4(0, 0, 0, 0);

            // Generate quad
            vertices[v + 0].position = float2(x, y) * scale + bias;
            vertices[v + 0].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16) / 16.0);
            vertices[v + 0].color = vertex_color;
            vertices[v + 0].flags = FLAG_TEXT;

            vertices[v + 1].position = float2(x + advance, y) * scale + bias;
            vertices[v + 1].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16) / 16.0);
            vertices[v + 1].color = vertex_color;
            vertices[v + 1].flags = FLAG_TEXT;

            vertices[v + 2].position = float2(x + advance, y + font_size) * scale + bias;
            vertices[v + 2].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16 + 1) / 16.0);
            vertices[v + 2].color = vertex_color;
            vertices[v + 2].flags = FLAG_TEXT;

            vertices[v + 3].position = float2(x, y + font_size) * scale + bias;
            vertices[v + 3].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16 + 1) / 16.0);
            vertices[v + 3].color = vertex_color;
            vertices[v + 3].flags = FLAG_TEXT;

            x += advance;
        }
        return;
    }

    // Process each character with O(1) position lookup using pre-computed line data
    uint current_line = 0;
    float x = box.content_x;

    for (uint i = 0; i < elem.text_length; i++) {
        // O(1) lookup: Find which line this character is on
        // Characters are sequential, so we just advance when we cross line boundaries
        while (current_line < line_count - 1 &&
               i >= line_info[line_base + current_line].char_end) {
            current_line++;
            x = box.content_x;  // Reset X for new line
        }

        // O(1): Y position directly from pre-computed line data
        float y = box.content_y + line_info[line_base + current_line].y_offset;

        uint char_code = text_buffer[elem.text_start + i];
        float advance = glyph_advance(char_code, font_size);

        // Generate vertex quad - NO LOOKBACK NEEDED!
        uint v = offset + i * 4;
        bool is_visible = (char_code != ' ' && char_code != '\t' &&
                          char_code != '\n' && char_code != '\r');
        float4 vertex_color = is_visible ? color : float4(0, 0, 0, 0);

        // Top-left
        vertices[v + 0].position = float2(x, y) * scale + bias;
        vertices[v + 0].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16) / 16.0);
        vertices[v + 0].color = vertex_color;
        vertices[v + 0].flags = FLAG_TEXT;

        // Top-right
        vertices[v + 1].position = float2(x + advance, y) * scale + bias;
        vertices[v + 1].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16) / 16.0);
        vertices[v + 1].color = vertex_color;
        vertices[v + 1].flags = FLAG_TEXT;

        // Bottom-right
        vertices[v + 2].position = float2(x + advance, y + font_size) * scale + bias;
        vertices[v + 2].tex_coord = float2(float(char_code % 16 + 1) / 16.0, float(char_code / 16 + 1) / 16.0);
        vertices[v + 2].color = vertex_color;
        vertices[v + 2].flags = FLAG_TEXT;

        // Bottom-left
        vertices[v + 3].position = float2(x, y + font_size) * scale + bias;
        vertices[v + 3].tex_coord = float2(float(char_code % 16) / 16.0, float(char_code / 16 + 1) / 16.0);
        vertices[v + 3].color = vertex_color;
        vertices[v + 3].flags = FLAG_TEXT;

        x += advance;
    }
}

// ============================================================================
// Image Rendering
// ============================================================================

// ImageInfo structure for GPU image rendering
struct ImageInfo {
    uint id;
    uint width;
    uint height;
    uint format;
    uint atlas_x;
    uint atlas_y;
    uint atlas_width;
    uint atlas_height;
};

// ImageRef structure linking elements to images
struct ImageRef {
    uint image_id;
    float width;
    float height;
    uint object_fit;
};

// Atlas dimensions (must match Rust side)
constant float ATLAS_WIDTH = 4096.0;
constant float ATLAS_HEIGHT = 4096.0;

// Generate image vertices
kernel void generate_image_vertices(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint* vertex_offsets [[buffer(3)]],
    device const ImageInfo* images [[buffer(4)]],
    device const uint* element_to_image [[buffer(5)]],  // Maps element index to image index
    device PaintVertex* vertices [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    constant Viewport& viewport [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];

    // Skip non-image elements
    if (elem.element_type != ELEM_IMG) return;

    ComputedStyle style = styles[gid];
    if (style.display == 0) return;

    LayoutBox box = boxes[gid];

    // Get image mapping
    uint image_idx = element_to_image[gid];
    if (image_idx == 0xFFFFFFFF) return;  // No image loaded

    ImageInfo img = images[image_idx];

    // Calculate offset (after background, border, text)
    uint background_count = (style.background_color[3] > 0) ? 4 : 0;
    bool has_border = style.border_width[0] > 0 || style.border_width[1] > 0 ||
                      style.border_width[2] > 0 || style.border_width[3] > 0;
    uint border_count = has_border ? 16 : 0;
    uint offset = vertex_offsets[gid] + background_count + border_count;

    float2 scale = float2(2.0 / viewport.width, -2.0 / viewport.height);
    float2 bias = float2(-1.0, 1.0);

    // Calculate UV coordinates from atlas position
    float u0 = float(img.atlas_x) / ATLAS_WIDTH;
    float v0 = float(img.atlas_y) / ATLAS_HEIGHT;
    float u1 = float(img.atlas_x + img.atlas_width) / ATLAS_WIDTH;
    float v1 = float(img.atlas_y + img.atlas_height) / ATLAS_HEIGHT;

    // Use layout box for positioning
    float left = box.content_x;
    float right = box.content_x + box.content_width;
    float top = box.content_y;
    float bottom = box.content_y + box.content_height;

    // White color (texture will be sampled)
    float4 color = float4(1.0, 1.0, 1.0, style.opacity);

    // Generate quad
    // Top-left
    vertices[offset + 0].position = float2(left, top) * scale + bias;
    vertices[offset + 0].tex_coord = float2(u0, v0);
    vertices[offset + 0].color = color;
    vertices[offset + 0].flags = FLAG_IMAGE;

    // Top-right
    vertices[offset + 1].position = float2(right, top) * scale + bias;
    vertices[offset + 1].tex_coord = float2(u1, v0);
    vertices[offset + 1].color = color;
    vertices[offset + 1].flags = FLAG_IMAGE;

    // Bottom-right
    vertices[offset + 2].position = float2(right, bottom) * scale + bias;
    vertices[offset + 2].tex_coord = float2(u1, v1);
    vertices[offset + 2].color = color;
    vertices[offset + 2].flags = FLAG_IMAGE;

    // Bottom-left
    vertices[offset + 3].position = float2(left, bottom) * scale + bias;
    vertices[offset + 3].tex_coord = float2(u0, v1);
    vertices[offset + 3].color = color;
    vertices[offset + 3].flags = FLAG_IMAGE;
}
