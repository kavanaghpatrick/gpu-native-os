#include <metal_stdlib>
using namespace metal;

// Element types
constant uint ELEM_TEXT = 100;

// White-space values
constant uint WHITE_SPACE_NORMAL = 0;
constant uint WHITE_SPACE_NOWRAP = 1;
constant uint WHITE_SPACE_PRE = 2;
constant uint WHITE_SPACE_PRE_WRAP = 3;
constant uint WHITE_SPACE_PRE_LINE = 4;

// Text-align values
constant uint TEXT_ALIGN_LEFT = 0;
constant uint TEXT_ALIGN_CENTER = 1;
constant uint TEXT_ALIGN_RIGHT = 2;
constant uint TEXT_ALIGN_JUSTIFY = 3;

// Break types
constant uint BREAK_NONE = 0;
constant uint BREAK_SPACE = 1;
constant uint BREAK_HYPHEN = 2;
constant uint BREAK_NEWLINE = 3;

// Maximum lines per text element
constant uint MAX_LINES_PER_ELEMENT = 1024;

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
    float _padding[2];
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

struct BreakOpportunity {
    uint char_index;
    uint break_type;
    float cumulative_width;
    float _padding;
};

struct LineBox {
    uint element_index;
    uint char_start;
    uint char_end;
    float width;
    float x;
    float y;
    float _padding[2];
};

// Calculate glyph width (approximation - monospace 0.6 factor)
float glyph_advance(uint char_code, float font_size) {
    // Space is narrower
    if (char_code == ' ') return font_size * 0.3;
    // Tab is 4 spaces
    if (char_code == '\t') return font_size * 1.2;
    // Newline has no width
    if (char_code == '\n' || char_code == '\r') return 0.0;
    // Regular character
    return font_size * 0.6;
}

// Check if character is a break opportunity
uint get_break_type(uint char_code) {
    if (char_code == ' ' || char_code == '\t') return BREAK_SPACE;
    if (char_code == '-') return BREAK_HYPHEN;
    if (char_code == '\n') return BREAK_NEWLINE;
    return BREAK_NONE;
}

// Pass 1: Find break opportunities and measure text
// Each thread processes one text element
kernel void find_break_opportunities(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device const uint8_t* text [[buffer(2)]],
    device BreakOpportunity* breaks [[buffer(3)]],
    constant uint& element_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT) return;
    if (elem.text_length == 0) return;

    ComputedStyle style = styles[gid];
    float font_size = style.font_size > 0 ? style.font_size : 16.0;

    float cumulative = 0.0;

    // Process each character in this element's text
    for (uint i = 0; i < elem.text_length; i++) {
        uint global_idx = elem.text_start + i;
        uint char_code = text[global_idx];

        float advance = glyph_advance(char_code, font_size);
        cumulative += advance;

        // Store break opportunity
        breaks[global_idx].char_index = i;
        breaks[global_idx].break_type = get_break_type(char_code);
        breaks[global_idx].cumulative_width = cumulative;
    }
}

// Pass 2: Compute line breaks based on container width
// Single threaded for correct line ordering
kernel void compute_line_breaks(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const BreakOpportunity* breaks [[buffer(3)]],
    device LineBox* lines [[buffer(4)]],
    device atomic_uint* line_count [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;  // Single thread

    for (uint elem_idx = 0; elem_idx < element_count; elem_idx++) {
        Element elem = elements[elem_idx];
        if (elem.element_type != ELEM_TEXT) continue;
        if (elem.text_length == 0) continue;

        ComputedStyle style = styles[elem_idx];
        LayoutBox box = boxes[elem_idx];

        float container_width = box.content_width;
        if (container_width <= 0) container_width = 10000.0;  // Effectively unlimited

        float font_size = style.font_size > 0 ? style.font_size : 16.0;
        float line_height = style.line_height > 0 ? style.line_height : 1.2;
        float line_height_px = font_size * line_height;

        uint line_start = 0;
        float line_start_width = 0.0;
        uint last_break_point = 0;
        float last_break_width = 0.0;
        float current_y = 0.0;
        uint lines_for_element = 0;

        for (uint i = 0; i < elem.text_length; i++) {
            uint global_idx = elem.text_start + i;
            BreakOpportunity brk = breaks[global_idx];

            float line_width = brk.cumulative_width - line_start_width;

            // Check for forced newline
            if (brk.break_type == BREAK_NEWLINE) {
                // Create line
                uint line_idx = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
                lines[line_idx].element_index = elem_idx;
                lines[line_idx].char_start = line_start;
                lines[line_idx].char_end = i;
                lines[line_idx].width = line_width;
                lines[line_idx].x = 0.0;  // Will be adjusted by position_lines
                lines[line_idx].y = current_y;

                current_y += line_height_px;
                line_start = i + 1;
                line_start_width = brk.cumulative_width;
                last_break_point = i + 1;
                last_break_width = brk.cumulative_width;
                lines_for_element++;
                continue;
            }

            // Track break opportunity
            if (brk.break_type == BREAK_SPACE || brk.break_type == BREAK_HYPHEN) {
                last_break_point = i + 1;  // Break after this char
                last_break_width = brk.cumulative_width;
            }

            // Check if line overflows
            if (line_width > container_width && i > line_start) {
                // Break at last opportunity, or force break if no opportunity
                uint break_at = (last_break_point > line_start) ? last_break_point : i;
                float break_width = (last_break_point > line_start) ?
                    (last_break_width - line_start_width) : line_width;

                // Create line
                uint line_idx = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
                lines[line_idx].element_index = elem_idx;
                lines[line_idx].char_start = line_start;
                lines[line_idx].char_end = break_at;
                lines[line_idx].width = break_width;
                lines[line_idx].x = 0.0;
                lines[line_idx].y = current_y;

                current_y += line_height_px;
                line_start = break_at;
                line_start_width = breaks[elem.text_start + break_at - 1].cumulative_width;
                if (break_at < elem.text_length) {
                    line_start_width = breaks[elem.text_start + break_at - 1].cumulative_width;
                }
                last_break_point = break_at;
                last_break_width = line_start_width;
                lines_for_element++;
            }
        }

        // Final line (remaining text)
        if (line_start < elem.text_length) {
            float final_width = breaks[elem.text_start + elem.text_length - 1].cumulative_width - line_start_width;

            uint line_idx = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
            lines[line_idx].element_index = elem_idx;
            lines[line_idx].char_start = line_start;
            lines[line_idx].char_end = elem.text_length;
            lines[line_idx].width = final_width;
            lines[line_idx].x = 0.0;
            lines[line_idx].y = current_y;

            lines_for_element++;
            current_y += line_height_px;
        }

        // Update layout box height for this text element
        box.content_height = current_y;
        box.height = current_y;  // For text elements, height = content height
        boxes[elem_idx] = box;
    }
}

// Pass 3: Position lines based on text-align
kernel void position_lines(
    device const LayoutBox* boxes [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LineBox* lines [[buffer(2)]],
    device const uint* line_count_ptr [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint line_count = *line_count_ptr;
    if (gid >= line_count) return;

    LineBox line = lines[gid];
    uint elem_idx = line.element_index;

    LayoutBox box = boxes[elem_idx];
    ComputedStyle style = styles[elem_idx];

    float container_width = box.content_width;
    float line_width = line.width;
    float space = container_width - line_width;

    // Apply text-align
    switch (style.text_align) {
        case TEXT_ALIGN_CENTER:
            line.x = space / 2.0;
            break;
        case TEXT_ALIGN_RIGHT:
            line.x = space;
            break;
        case TEXT_ALIGN_JUSTIFY:
            // Justify would need word spacing adjustment
            // For now, treat as left
            line.x = 0.0;
            break;
        default:  // TEXT_ALIGN_LEFT
            line.x = 0.0;
            break;
    }

    lines[gid] = line;
}

// Measure text advances (for use by layout engine)
kernel void measure_text_advances(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device const uint8_t* text [[buffer(2)]],
    device float* advances [[buffer(3)]],  // One per character
    constant uint& element_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT) return;
    if (elem.text_length == 0) return;

    ComputedStyle style = styles[gid];
    float font_size = style.font_size > 0 ? style.font_size : 16.0;

    // Measure each glyph
    for (uint i = 0; i < elem.text_length; i++) {
        uint char_code = text[elem.text_start + i];
        advances[elem.text_start + i] = glyph_advance(char_code, font_size);
    }
}
