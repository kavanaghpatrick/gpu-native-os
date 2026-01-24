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
constant uint MAX_LINES_PER_ELEMENT = 64;

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

// ============================================================================
// Glyph Metrics (uploaded from font at load time)
// ============================================================================

struct GlyphMetrics {
    float advance;          // Horizontal advance width
    float bearing_x;        // Left side bearing
    float bearing_y;        // Top bearing (baseline to top)
    float width;            // Glyph bbox width
    float height;           // Glyph bbox height
    ushort atlas_x;         // Atlas position X
    ushort atlas_y;         // Atlas position Y
    ushort atlas_w;         // Atlas size W
    ushort atlas_h;         // Atlas size H
};

// Positioned glyph output
struct PositionedGlyph {
    float x;                // Screen X position
    float y;                // Screen Y position
    uint glyph_id;          // Index into atlas
    uint color;             // Packed RGBA
    float scale;            // Font size / base size
    uint line_index;        // Which line
    float _padding[2];
};

// Calculate glyph width using real metrics from buffer
float glyph_advance_from_metrics(
    device const GlyphMetrics* metrics,
    uint char_code,
    float font_size,
    float base_size
) {
    // Map ASCII to glyph index (32-127 -> 0-95)
    if (char_code < 32 || char_code >= 128) {
        // Non-printable: newline, tab handled specially
        if (char_code == '\n' || char_code == '\r') return 0.0;
        if (char_code == '\t') return metrics[0].advance * 4.0 * (font_size / base_size);
        return 0.0;
    }

    uint glyph_id = char_code - 32;
    float scale = font_size / base_size;
    return metrics[glyph_id].advance * scale;
}

// Fallback: Calculate glyph width (monospace approximation)
// Used when metrics buffer not available
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

// ============================================================================
// Issue #131: O(1) Two-Pass Text Line Layout
// ============================================================================

// Per-element line info header (16 bytes)
struct TextLineDataHeader {
    uint line_count;
    uint _padding[3];
};

// Pre-computed line information (16 bytes)
struct LineInfo {
    uint char_start;
    uint char_end;
    float y_offset;
    float width;
};

// Pass 1: Compute line breaks and store LineInfo for each line
// One thread per text element - computes all lines for that element
kernel void compute_line_breaks_two_pass(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint8_t* text [[buffer(3)]],
    device TextLineDataHeader* line_headers [[buffer(4)]],
    device LineInfo* line_info [[buffer(5)]],  // Flat array: element_idx * MAX_LINES_PER_ELEMENT + line_idx
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT || elem.text_length == 0) {
        line_headers[gid].line_count = 0;
        return;
    }

    ComputedStyle style = styles[gid];
    LayoutBox box = boxes[gid];

    float font_size = style.font_size > 0 ? style.font_size : 16.0;
    float line_height = (style.line_height > 0 ? style.line_height : 1.2) * font_size;
    float container_width = box.content_width > 0 ? box.content_width : 10000.0;

    uint line_count = 0;
    uint line_start = 0;
    float x = 0.0;
    float y = 0.0;
    uint word_start = 0;
    float word_start_x = 0.0;
    bool in_word = false;

    // Base offset for this element's lines in the flat array
    uint line_base = gid * MAX_LINES_PER_ELEMENT;

    for (uint i = 0; i < elem.text_length && line_count < MAX_LINES_PER_ELEMENT; i++) {
        uint c = text[elem.text_start + i];
        float advance = glyph_advance(c, font_size);

        bool is_space = (c == ' ' || c == '\t');
        bool is_newline = (c == '\n');

        // Track word boundaries
        if (is_space && in_word) {
            in_word = false;
        } else if (!is_space && !is_newline && !in_word) {
            in_word = true;
            word_start = i;
            word_start_x = x;
        }

        // Check for line break
        bool needs_wrap = (x + advance > container_width && x > 0.0);

        if (is_newline || needs_wrap) {
            // Record this line
            uint line_idx = line_base + line_count;
            if (needs_wrap && in_word && word_start > line_start) {
                // Break at word boundary
                line_info[line_idx].char_start = line_start;
                line_info[line_idx].char_end = word_start;
                line_info[line_idx].y_offset = y;
                line_info[line_idx].width = word_start_x;
            } else {
                // Break at current position
                line_info[line_idx].char_start = line_start;
                line_info[line_idx].char_end = i;
                line_info[line_idx].y_offset = y;
                line_info[line_idx].width = x;
            }
            line_count++;

            // Start new line
            y += line_height;
            if (needs_wrap && in_word && word_start > line_start) {
                // Word moves to new line - recalculate x from word start
                line_start = word_start;
                x = x - word_start_x;
                word_start_x = 0.0;
            } else {
                line_start = i + 1;
                x = 0.0;
            }

            if (is_newline) {
                x = 0.0;
                in_word = false;
                continue;
            }
        }

        x += advance;
    }

    // Record final line
    if (line_start < elem.text_length && line_count < MAX_LINES_PER_ELEMENT) {
        uint line_idx = line_base + line_count;
        line_info[line_idx].char_start = line_start;
        line_info[line_idx].char_end = elem.text_length;
        line_info[line_idx].y_offset = y;
        line_info[line_idx].width = x;
        line_count++;
    }

    line_headers[gid].line_count = line_count;
}

// ============================================================================
// Issue #90: GPU-Native Text Layout Kernels
// ============================================================================

// Kernel 1: Character to Glyph with Real Metrics (Parallel - one thread per char)
kernel void char_to_glyph(
    device const uint8_t* text [[buffer(0)]],
    device const GlyphMetrics* metrics [[buffer(1)]],
    device float* advances [[buffer(2)]],
    device uint* glyph_ids [[buffer(3)]],
    constant uint& char_count [[buffer(4)]],
    constant float& font_size [[buffer(5)]],
    constant float& base_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    uint8_t c = text[gid];

    // Map ASCII to glyph ID
    uint glyph_id = (c >= 32 && c < 128) ? (c - 32) : 0;
    glyph_ids[gid] = glyph_id;

    // Get advance from metrics
    float scale = font_size / base_size;
    if (c == '\n' || c == '\r') {
        advances[gid] = 0.0;
    } else if (c == '\t') {
        advances[gid] = metrics[0].advance * 4.0 * scale;  // 4 spaces
    } else {
        advances[gid] = metrics[glyph_id].advance * scale;
    }
}

// Kernel 2a: Parallel Prefix Sum - Up-sweep phase
kernel void prefix_sum_up(
    device float* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = (gid + 1) * stride * 2 - 1;
    if (idx < n) {
        data[idx] += data[idx - stride];
    }
}

// Kernel 2b: Parallel Prefix Sum - Down-sweep phase
kernel void prefix_sum_down(
    device float* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = (gid + 1) * stride * 2 - 1;
    if (idx < n) {
        float t = data[idx - stride];
        data[idx - stride] = data[idx];
        data[idx] += t;
    }
}

// Kernel 2c: Simple sequential prefix sum for small arrays (< 1024)
kernel void prefix_sum_sequential(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    float sum = 0.0;
    for (uint i = 0; i < n; i++) {
        output[i] = sum;
        sum += input[i];
    }
    // Store final sum at position n (if buffer is large enough)
    if (n > 0) {
        output[n] = sum;
    }
}

// Kernel 3: Find Break Opportunities (Parallel - one thread per char)
kernel void find_breaks_parallel(
    device const uint8_t* text [[buffer(0)]],
    device uint* is_break [[buffer(1)]],
    device uint* break_type [[buffer(2)]],
    constant uint& char_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    uint8_t c = text[gid];

    // Break AFTER spaces, newlines, hyphens
    if (c == ' ' || c == '\t') {
        is_break[gid] = 1;
        break_type[gid] = BREAK_SPACE;
    } else if (c == '\n') {
        is_break[gid] = 1;
        break_type[gid] = BREAK_NEWLINE;
    } else if (c == '-' && gid > 0) {
        is_break[gid] = 1;
        break_type[gid] = BREAK_HYPHEN;
    } else {
        is_break[gid] = 0;
        break_type[gid] = BREAK_NONE;
    }
}

// Kernel 4: Assign Line Indices (Parallel per text element)
// Each thread processes one text element's worth of characters
kernel void assign_lines_parallel(
    device const float* cumulative_widths [[buffer(0)]],
    device const uint* is_break [[buffer(1)]],
    device const Element* elements [[buffer(2)]],
    device const LayoutBox* boxes [[buffer(3)]],
    device uint* line_indices [[buffer(4)]],
    device LineBox* lines [[buffer(5)]],
    device atomic_uint* line_count [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    constant float& line_height_px [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT) return;
    if (elem.text_length == 0) return;

    LayoutBox box = boxes[gid];
    float container_width = box.content_width > 0 ? box.content_width : 10000.0;

    uint start = elem.text_start;
    uint end = start + elem.text_length;

    uint line_start_char = 0;
    float line_start_width = cumulative_widths[start];
    uint last_break_char = 0;
    float last_break_width = line_start_width;
    uint current_line = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
    float current_y = 0.0;

    // First line
    LineBox first_line;
    first_line.element_index = gid;
    first_line.char_start = 0;
    first_line.y = 0.0;
    first_line.x = 0.0;

    for (uint i = start; i < end; i++) {
        uint local_i = i - start;
        float width_so_far = cumulative_widths[i + 1] - line_start_width;

        // Track break opportunities
        if (is_break[i]) {
            last_break_char = local_i + 1;
            last_break_width = cumulative_widths[i + 1];
        }

        // Check for newline (forced break)
        // We detect this by checking if advance at this position is 0 and it's a break
        // Actually, check break_type would be better but we don't have it here
        // For now, rely on width overflow logic

        // Check if line overflows
        if (width_so_far > container_width && local_i > line_start_char) {
            // Finish current line
            uint break_at = (last_break_char > line_start_char) ? last_break_char : local_i;

            // Store line info
            lines[current_line].element_index = gid;
            lines[current_line].char_start = line_start_char;
            lines[current_line].char_end = break_at;
            lines[current_line].width = (last_break_char > line_start_char) ?
                (last_break_width - line_start_width) : width_so_far;
            lines[current_line].y = current_y;
            lines[current_line].x = 0.0;

            // Start new line
            current_y += line_height_px;
            current_line = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
            line_start_char = break_at;
            line_start_width = cumulative_widths[start + break_at];
            last_break_char = break_at;
            last_break_width = line_start_width;
        }

        // Assign line index to this character
        line_indices[i] = current_line;
    }

    // Final line
    if (line_start_char < elem.text_length) {
        lines[current_line].element_index = gid;
        lines[current_line].char_start = line_start_char;
        lines[current_line].char_end = elem.text_length;
        lines[current_line].width = cumulative_widths[end] - line_start_width;
        lines[current_line].y = current_y;
        lines[current_line].x = 0.0;
    }
}

// Kernel 5: Position Glyphs (Fully Parallel - one thread per character)
kernel void position_glyphs(
    device const float* cumulative_widths [[buffer(0)]],
    device const uint* line_indices [[buffer(1)]],
    device const LineBox* lines [[buffer(2)]],
    device const Element* elements [[buffer(3)]],
    device const LayoutBox* boxes [[buffer(4)]],
    device const uint* glyph_ids [[buffer(5)]],
    device const ComputedStyle* styles [[buffer(6)]],
    device PositionedGlyph* output [[buffer(7)]],
    constant uint& char_count [[buffer(8)]],
    constant float& base_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    // Find which element this character belongs to
    // This is O(1) since we can binary search or use a lookup table
    // For simplicity, scan elements (could optimize with element_start buffer)

    uint line_idx = line_indices[gid];
    LineBox line = lines[line_idx];
    uint elem_idx = line.element_index;

    Element elem = elements[elem_idx];
    LayoutBox box = boxes[elem_idx];
    ComputedStyle style = styles[elem_idx];

    uint local_char = gid - elem.text_start;

    // X position = cumulative width - line start width + container offset + text-align offset
    float line_start_cumulative = cumulative_widths[elem.text_start + line.char_start];
    float char_cumulative = cumulative_widths[gid];
    float x = box.content_x + line.x + (char_cumulative - line_start_cumulative);

    // Y position = container Y + line Y offset
    float y = box.content_y + line.y;

    // Pack color from style
    uint color = (uint(style.color[0] * 255) << 24) |
                 (uint(style.color[1] * 255) << 16) |
                 (uint(style.color[2] * 255) << 8) |
                 (uint(style.color[3] * 255));

    output[gid].x = x;
    output[gid].y = y;
    output[gid].glyph_id = glyph_ids[gid];
    output[gid].color = color;
    output[gid].scale = style.font_size / base_size;
    output[gid].line_index = line_idx;
}

// Kernel 6: Generate Text Vertices from Positioned Glyphs (Parallel)
kernel void generate_text_vertices(
    device const PositionedGlyph* glyphs [[buffer(0)]],
    device const GlyphMetrics* metrics [[buffer(1)]],
    device const uint8_t* text [[buffer(2)]],
    device float* vertices [[buffer(3)]],  // 6 vertices per char, each vertex = 8 floats
    constant uint& char_count [[buffer(4)]],
    constant float& atlas_width [[buffer(5)]],
    constant float& atlas_height [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    // Skip non-printable characters
    uint8_t c = text[gid];
    if (c < 32 || c == '\n' || c == '\r' || c == '\t') {
        // Write degenerate triangles (zero area)
        uint base = gid * 6 * 8;  // 6 vertices × 8 floats each
        for (uint i = 0; i < 48; i++) {
            vertices[base + i] = 0.0;
        }
        return;
    }

    PositionedGlyph g = glyphs[gid];
    GlyphMetrics m = metrics[g.glyph_id];

    // Calculate quad corners
    float x0 = g.x + m.bearing_x * g.scale;
    float y0 = g.y + m.bearing_y * g.scale;  // Baseline to top
    float x1 = x0 + m.width * g.scale;
    float y1 = y0 + m.height * g.scale;

    // UV coordinates from atlas
    float u0 = float(m.atlas_x) / atlas_width;
    float v0 = float(m.atlas_y) / atlas_height;
    float u1 = u0 + float(m.atlas_w) / atlas_width;
    float v1 = v0 + float(m.atlas_h) / atlas_height;

    // Unpack color
    float r = float((g.color >> 24) & 0xFF) / 255.0;
    float gc = float((g.color >> 16) & 0xFF) / 255.0;
    float b = float((g.color >> 8) & 0xFF) / 255.0;
    float a = float(g.color & 0xFF) / 255.0;

    // Each vertex: x, y, u, v, r, g, b, a
    uint base = gid * 6 * 8;

    // Triangle 1: top-left, top-right, bottom-right
    // Vertex 0: top-left
    vertices[base + 0] = x0; vertices[base + 1] = y0;
    vertices[base + 2] = u0; vertices[base + 3] = v0;
    vertices[base + 4] = r; vertices[base + 5] = gc; vertices[base + 6] = b; vertices[base + 7] = a;

    // Vertex 1: top-right
    vertices[base + 8] = x1; vertices[base + 9] = y0;
    vertices[base + 10] = u1; vertices[base + 11] = v0;
    vertices[base + 12] = r; vertices[base + 13] = gc; vertices[base + 14] = b; vertices[base + 15] = a;

    // Vertex 2: bottom-right
    vertices[base + 16] = x1; vertices[base + 17] = y1;
    vertices[base + 18] = u1; vertices[base + 19] = v1;
    vertices[base + 20] = r; vertices[base + 21] = gc; vertices[base + 22] = b; vertices[base + 23] = a;

    // Triangle 2: top-left, bottom-right, bottom-left
    // Vertex 3: top-left
    vertices[base + 24] = x0; vertices[base + 25] = y0;
    vertices[base + 26] = u0; vertices[base + 27] = v0;
    vertices[base + 28] = r; vertices[base + 29] = gc; vertices[base + 30] = b; vertices[base + 31] = a;

    // Vertex 4: bottom-right
    vertices[base + 32] = x1; vertices[base + 33] = y1;
    vertices[base + 34] = u1; vertices[base + 35] = v1;
    vertices[base + 36] = r; vertices[base + 37] = gc; vertices[base + 38] = b; vertices[base + 39] = a;

    // Vertex 5: bottom-left
    vertices[base + 40] = x0; vertices[base + 41] = y1;
    vertices[base + 42] = u0; vertices[base + 43] = v1;
    vertices[base + 44] = r; vertices[base + 45] = gc; vertices[base + 46] = b; vertices[base + 47] = a;
}
