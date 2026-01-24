#include <metal_stdlib>
using namespace metal;

// Element types (from parser)
constant uint ELEM_TEXT = 100;

// Selector types
constant uint SEL_UNIVERSAL = 0;
constant uint SEL_TAG = 1;
constant uint SEL_CLASS = 2;
constant uint SEL_ID = 3;
constant uint SEL_ATTRIBUTE = 4;
constant uint SEL_PSEUDO = 5;
constant uint SEL_TAG_CLASS = 6;  // Combined tag.class (e.g., p.highlight)
constant uint SEL_TAG_ID = 7;     // Combined tag#id (e.g., div#main)

// Combinator types
constant uint COMB_NONE = 0;
constant uint COMB_DESCENDANT = 1;  // E F (space)
constant uint COMB_CHILD = 2;       // E > F
constant uint COMB_ADJACENT = 3;    // E + F
constant uint COMB_SIBLING = 4;     // E ~ F

// Attribute operators
constant uint ATTR_EXISTS = 0;
constant uint ATTR_EQUALS = 1;
constant uint ATTR_CONTAINS = 2;
constant uint ATTR_STARTS = 3;
constant uint ATTR_ENDS = 4;
constant uint ATTR_WORD = 5;
constant uint ATTR_LANG = 6;

// Pseudo-class types
constant uint PSEUDO_FIRST_CHILD = 1;
constant uint PSEUDO_LAST_CHILD = 2;
constant uint PSEUDO_NTH_CHILD = 3;
constant uint PSEUDO_FIRST_OF_TYPE = 4;
constant uint PSEUDO_LAST_OF_TYPE = 5;
constant uint PSEUDO_ONLY_CHILD = 6;
constant uint PSEUDO_EMPTY = 7;
constant uint PSEUDO_ROOT = 8;

// Display values
constant uint DISPLAY_NONE = 0;
constant uint DISPLAY_BLOCK = 1;
constant uint DISPLAY_INLINE = 2;
constant uint DISPLAY_FLEX = 3;
constant uint DISPLAY_INLINE_BLOCK = 4;
constant uint DISPLAY_TABLE = 5;
constant uint DISPLAY_TABLE_ROW = 6;
constant uint DISPLAY_TABLE_CELL = 7;

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

// Text align
constant uint TEXT_LEFT = 0;
constant uint TEXT_CENTER = 1;
constant uint TEXT_RIGHT = 2;

// CSS Position values
constant uint POSITION_STATIC = 0;
constant uint POSITION_RELATIVE = 1;
constant uint POSITION_ABSOLUTE = 2;
constant uint POSITION_FIXED = 3;

// Special value for "auto" offset
constant float OFFSET_AUTO = 3.4028235e+38;  // f32::MAX

// CSS Overflow values
constant uint OVERFLOW_VISIBLE = 0;
constant uint OVERFLOW_HIDDEN = 1;
constant uint OVERFLOW_SCROLL = 2;
constant uint OVERFLOW_AUTO = 3;

// Property IDs
constant uint PROP_DISPLAY = 0;
constant uint PROP_WIDTH = 1;
constant uint PROP_HEIGHT = 2;
constant uint PROP_MARGIN = 3;
constant uint PROP_PADDING = 4;
constant uint PROP_COLOR = 5;
constant uint PROP_BACKGROUND = 6;
constant uint PROP_FONT_SIZE = 7;
constant uint PROP_LINE_HEIGHT = 8;
constant uint PROP_FONT_WEIGHT = 9;
constant uint PROP_TEXT_ALIGN = 10;
constant uint PROP_FLEX_DIRECTION = 11;
constant uint PROP_JUSTIFY_CONTENT = 12;
constant uint PROP_ALIGN_ITEMS = 13;
constant uint PROP_FLEX_GROW = 14;
constant uint PROP_FLEX_SHRINK = 15;
constant uint PROP_BORDER_WIDTH = 16;
constant uint PROP_BORDER_COLOR = 17;
constant uint PROP_BORDER_RADIUS = 18;
constant uint PROP_OPACITY = 19;
constant uint PROP_POSITION = 20;
constant uint PROP_TOP = 21;
constant uint PROP_RIGHT = 22;
constant uint PROP_BOTTOM = 23;
constant uint PROP_LEFT = 24;
constant uint PROP_Z_INDEX = 25;
constant uint PROP_OVERFLOW = 26;
constant uint PROP_OVERFLOW_X = 27;
constant uint PROP_OVERFLOW_Y = 28;
constant uint PROP_BOX_SHADOW = 29;
constant uint PROP_BOX_SHADOW_COLOR = 30;
constant uint PROP_BOX_SHADOW_INSET = 31;
constant uint PROP_GRADIENT_TYPE = 32;
constant uint PROP_GRADIENT_ANGLE = 33;
constant uint PROP_GRADIENT_STOP = 34;
constant uint PROP_BORDER_COLLAPSE = 35;
constant uint PROP_BORDER_SPACING = 36;

// Gradient types
constant uint GRADIENT_NONE = 0;
constant uint GRADIENT_LINEAR = 1;
constant uint GRADIENT_RADIAL = 2;

// Property set bitmask flags (for tracking which properties were explicitly set)
constant uint PROP_SET_COLOR = (1u << 0);
constant uint PROP_SET_FONT_SIZE = (1u << 1);
constant uint PROP_SET_LINE_HEIGHT = (1u << 2);
constant uint PROP_SET_FONT_WEIGHT = (1u << 3);
constant uint PROP_SET_TEXT_ALIGN = (1u << 4);

// Limits
constant uint MAX_MATCHES = 32;
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

struct Token {
    uint token_type;
    uint start;
    uint end;
    uint _padding;
};

struct Selector {
    uint selector_type;     // TAG, CLASS, ID, ATTRIBUTE, PSEUDO
    uint hash;              // Hash of tag/class/id name
    uint specificity;
    uint style_start;       // Start index in style_defs
    uint style_count;       // Number of StyleDefs for this selector
    uint combinator;        // NONE, DESCENDANT, CHILD, ADJACENT, SIBLING
    int next_part;          // Index of next selector part (-1 if none)
    uint pseudo_type;       // For pseudo-classes
    uint attr_name_hash;    // For attribute selectors
    uint attr_op;           // ATTR_EXISTS, ATTR_EQUALS, etc.
    uint attr_value_hash;   // For attribute value matching
    int nth_a;              // For :nth-child(an+b)
    int nth_b;
    uint _padding[3];
};

struct StyleDef {
    uint property_id;
    float values[4];
};

struct InlineStyleIndex {
    uint start;
    uint count;
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
    float color[4];      // RGBA
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
    float top;           // Offset from top (OFFSET_AUTO = auto)
    float right_;        // Offset from right (right is reserved in Metal)
    float bottom;        // Offset from bottom
    float left;          // Offset from left
    int z_index;         // Stacking order

    // Inheritance tracking - bitmask of explicitly set properties
    uint properties_set;

    // CSS Overflow
    uint overflow_x;     // OVERFLOW_VISIBLE, HIDDEN, SCROLL, AUTO
    uint overflow_y;     // OVERFLOW_VISIBLE, HIDDEN, SCROLL, AUTO

    // Box Shadow (up to 4 shadows)
    uint shadow_count;           // Number of shadows (0-4)
    float shadow_offset_x[4];    // Horizontal offset per shadow
    float shadow_offset_y[4];    // Vertical offset per shadow
    float shadow_blur[4];        // Blur radius per shadow
    float shadow_spread[4];      // Spread radius per shadow
    float shadow_color[16];      // RGBA color per shadow (4 floats * 4 shadows)
    uint shadow_inset[4];        // 1 = inset, 0 = outset

    // Gradients (up to 8 color stops)
    uint gradient_type;          // GRADIENT_NONE, LINEAR, RADIAL
    float gradient_angle;        // Angle in degrees (for linear gradient)
    uint gradient_stop_count;    // Number of color stops (0-8)
    float gradient_stop_colors[32]; // RGBA color per stop (4 floats * 8 stops)
    float gradient_stop_positions[8]; // Position 0.0-1.0 per stop

    // Table layout
    uint border_collapse;        // 0 = separate, 1 = collapse
    float border_spacing;        // Spacing between cells
    float _padding[2];           // Pad to maintain 16-byte alignment
};

// Hash function (djb2)
uint hash_string(device const uint8_t* str, uint start, uint end) {
    uint hash = 5381;
    for (uint i = start; i < end; i++) {
        uint8_t c = str[i];
        if (c >= 'A' && c <= 'Z') c += 32;  // lowercase
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

// Hash element type to match tag selector
uint element_type_hash(uint elem_type) {
    // Pre-computed hashes for common elements (djb2 hash)
    // These must match the hashes used when parsing CSS
    switch (elem_type) {
        case 1: return 193489480;   // div
        case 2: return 2090731639;  // span
        case 3: return 177685;      // p
        case 4: return 177670;      // a
        case 5: return 5863390;     // h1
        case 6: return 5863391;     // h2
        case 7: return 5863392;     // h3
        case 11: return 5863878;    // ul
        case 12: return 5863680;    // ol
        case 13: return 5863578;    // li
        case 14: return 193495042;  // img
        case 15: return 5863257;    // br
        case 16: return 5863455;    // hr
        case 17: return 275315341;  // table
        case 18: return 5863851;    // tr
        case 19: return 5863837;    // td
        case 20: return 5863841;    // th
        case 23: return 2090263929; // form
        case 24: return 262752949;  // input
        case 25: return 4110376321; // button
        case 30: return 193500106;  // nav
        case 31: return 30546094;   // header
        case 32: return 4259622356; // footer
        case 33: return 2090499946; // main
        case 34: return 2332218074; // section
        case 35: return 1089149929; // article
        case 40: return 479447426;  // strong
        case 41: return 5863351;    // em
        case 42: return 177671;     // b
        case 43: return 177678;     // i
        case 38: return 2090155648; // code
        case 37: return 193502828;  // pre
        case 101: return 2090341082;// html
        case 102: return 2090324343;// head
        case 103: return 2090119731;// body
        default: return 0;
    }
}

// Check if element has a specific class
bool element_has_class(
    device const Element* elements,
    device const uint8_t* html,
    device const Token* tokens,
    uint elem_idx,
    uint class_hash
) {
    Element elem = elements[elem_idx];
    if (elem.element_type == ELEM_TEXT) return false;

    Token token = tokens[elem.token_index];

    // Search for class="..." or class='...'
    for (uint i = token.start; i < token.end - 7; i++) {
        if (html[i] == 'c' && html[i+1] == 'l' && html[i+2] == 'a' &&
            html[i+3] == 's' && html[i+4] == 's' && html[i+5] == '=') {
            i += 6;
            char quote = html[i];
            if (quote != '"' && quote != '\'') continue;
            i++;

            // Parse space-separated class names
            while (i < token.end && html[i] != quote) {
                // Skip whitespace
                while (i < token.end && html[i] == ' ') i++;
                if (i >= token.end || html[i] == quote) break;

                // Find end of class name
                uint class_start = i;
                while (i < token.end && html[i] != quote && html[i] != ' ') {
                    i++;
                }

                // Hash and compare
                if (hash_string(html, class_start, i) == class_hash) {
                    return true;
                }
            }
            break;
        }
    }
    return false;
}

// Check if element has a specific ID
bool element_has_id(
    device const Element* elements,
    device const uint8_t* html,
    device const Token* tokens,
    uint elem_idx,
    uint id_hash
) {
    Element elem = elements[elem_idx];
    if (elem.element_type == ELEM_TEXT) return false;

    Token token = tokens[elem.token_index];

    // Search for id="..." or id='...'
    for (uint i = token.start; i < token.end - 4; i++) {
        if (html[i] == 'i' && html[i+1] == 'd' && html[i+2] == '=') {
            i += 3;
            char quote = html[i];
            if (quote != '"' && quote != '\'') continue;
            i++;

            uint id_start = i;
            while (i < token.end && html[i] != quote) {
                i++;
            }

            if (hash_string(html, id_start, i) == id_hash) {
                return true;
            }
            break;
        }
    }
    return false;
}

// Check if element has a specific attribute
bool element_has_attribute(
    device const Element* elements,
    device const uint8_t* html,
    device const Token* tokens,
    uint elem_idx,
    uint attr_name_hash
) {
    Element elem = elements[elem_idx];
    if (elem.element_type == ELEM_TEXT) return false;

    Token token = tokens[elem.token_index];

    // Simple attribute search - look for attr= or attr (boolean)
    // This is a simplified implementation
    for (uint i = token.start; i < token.end; i++) {
        // Skip until we find a space (start of attributes)
        if (html[i] == ' ') {
            while (i < token.end && html[i] == ' ') i++;

            uint attr_start = i;
            // Find end of attribute name
            while (i < token.end && html[i] != '=' && html[i] != ' ' && html[i] != '>') {
                i++;
            }

            if (hash_string(html, attr_start, i) == attr_name_hash) {
                return true;
            }
        }
    }
    return false;
}

// Get attribute value hash (simplified)
uint get_attribute_value_hash(
    device const Element* elements,
    device const uint8_t* html,
    device const Token* tokens,
    uint elem_idx,
    uint attr_name_hash
) {
    Element elem = elements[elem_idx];
    if (elem.element_type == ELEM_TEXT) return 0;

    Token token = tokens[elem.token_index];

    for (uint i = token.start; i < token.end; i++) {
        if (html[i] == ' ') {
            while (i < token.end && html[i] == ' ') i++;

            uint attr_start = i;
            while (i < token.end && html[i] != '=' && html[i] != ' ' && html[i] != '>') {
                i++;
            }

            if (hash_string(html, attr_start, i) == attr_name_hash && html[i] == '=') {
                i++; // Skip =
                char quote = html[i];
                if (quote == '"' || quote == '\'') {
                    i++;
                    uint val_start = i;
                    while (i < token.end && html[i] != quote) {
                        i++;
                    }
                    return hash_string(html, val_start, i);
                }
            }
        }
    }
    return 0;
}

// Helper: compare html substring with constant string
bool inline_val_equals(device const uint8_t* html, uint start, uint end, constant char* str, uint len) {
    if (end - start != len) return false;
    for (uint i = 0; i < len; i++) {
        char c = html[start + i];
        if (c >= 'A' && c <= 'Z') c += 32;  // lowercase
        if (c != str[i]) return false;
    }
    return true;
}

// Helper: parse hex digit
float hex_digit_value(uint8_t c) {
    if (c >= '0' && c <= '9') return float(c - '0');
    if (c >= 'a' && c <= 'f') return float(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return float(c - 'A' + 10);
    return 0;
}

// Parse inline style attribute directly on GPU
void parse_inline_style_gpu(
    device const uint8_t* html,
    uint start,
    uint end,
    thread ComputedStyle* style
) {
    // Find style=" or style=' in the tag
    uint i = start;
    while (i + 6 < end) {
        if (html[i] == 's' && html[i+1] == 't' && html[i+2] == 'y' &&
            html[i+3] == 'l' && html[i+4] == 'e' && html[i+5] == '=') {
            i += 6;
            char quote = html[i];
            if (quote == '"' || quote == '\'') {
                i++;
                uint style_start = i;
                while (i < end && html[i] != quote) i++;
                uint style_end = i;

                // Parse CSS declarations within the style attribute
                uint j = style_start;
                while (j < style_end) {
                    // Skip whitespace and semicolons
                    while (j < style_end && (html[j] == ' ' || html[j] == ';' || html[j] == '\n' || html[j] == '\t')) j++;
                    if (j >= style_end) break;

                    // Get property name
                    uint prop_start = j;
                    while (j < style_end && html[j] != ':' && html[j] != ';') j++;
                    uint prop_end = j;
                    if (j < style_end && html[j] == ':') j++;

                    // Skip whitespace
                    while (j < style_end && html[j] == ' ') j++;

                    // Get value
                    uint val_start = j;
                    while (j < style_end && html[j] != ';') j++;
                    uint val_end = j;

                    // Trim trailing whitespace from value
                    while (val_end > val_start && (html[val_end-1] == ' ' || html[val_end-1] == '\n' || html[val_end-1] == '\t')) val_end--;
                    // Trim trailing whitespace from property name
                    while (prop_end > prop_start && (html[prop_end-1] == ' ' || html[prop_end-1] == '\t')) prop_end--;

                    // Parse property by computing hash of property name
                    uint prop_hash = hash_string(html, prop_start, prop_end);

                    // Parse numeric value (px)
                    float px_val = 0;
                    for (uint k = val_start; k < val_end && html[k] >= '0' && html[k] <= '9'; k++) {
                        px_val = px_val * 10 + (html[k] - '0');
                    }

                    // width: hash = 279163045
                    if (prop_hash == 279163045) {
                        style->width = px_val;
                        style->properties_set |= (1 << 1);
                    }
                    // height: hash = 30836958
                    else if (prop_hash == 30836958) {
                        style->height = px_val;
                        style->properties_set |= (1 << 2);
                    }
                    // color: hash = 255668804
                    else if (prop_hash == 255668804) {
                        uint val_hash = hash_string(html, val_start, val_end);
                        // red = 193504576, blue = 2090117005, green = 260512342
                        if (val_hash == 193504576) {
                            style->color[0] = 1.0; style->color[1] = 0.0; style->color[2] = 0.0; style->color[3] = 1.0;
                        } else if (val_hash == 2090117005) {
                            style->color[0] = 0.0; style->color[1] = 0.0; style->color[2] = 1.0; style->color[3] = 1.0;
                        } else if (val_hash == 260512342) {
                            style->color[0] = 0.0; style->color[1] = 0.5; style->color[2] = 0.0; style->color[3] = 1.0;
                        } else if (val_hash == 254362690) { // black
                            style->color[0] = 0.0; style->color[1] = 0.0; style->color[2] = 0.0; style->color[3] = 1.0;
                        } else if (val_hash == 279132550) { // white
                            style->color[0] = 1.0; style->color[1] = 1.0; style->color[2] = 1.0; style->color[3] = 1.0;
                        }
                        style->properties_set |= (1 << 5);
                    }
                    // display: hash = 315443099
                    else if (prop_hash == 315443099) {
                        uint val_hash = hash_string(html, val_start, val_end);
                        // none = 2090551285, block = 254377936, inline = 80755812, flex = 2090260244
                        if (val_hash == 2090551285) {
                            style->display = DISPLAY_NONE;
                        } else if (val_hash == 254377936) {
                            style->display = DISPLAY_BLOCK;
                        } else if (val_hash == 80755812) {
                            style->display = DISPLAY_INLINE;
                        } else if (val_hash == 2090260244) {
                            style->display = DISPLAY_FLEX;
                        }
                        style->properties_set |= (1 << 0);
                    }
                    // background-color: hash = 2007711153
                    else if (prop_hash == 2007711153) {
                        uint val_hash = hash_string(html, val_start, val_end);
                        if (val_hash == 193504576) { // red
                            style->background_color[0] = 1.0; style->background_color[1] = 0.0;
                            style->background_color[2] = 0.0; style->background_color[3] = 1.0;
                        } else if (val_hash == 2090117005) { // blue
                            style->background_color[0] = 0.0; style->background_color[1] = 0.0;
                            style->background_color[2] = 1.0; style->background_color[3] = 1.0;
                        } else if (html[val_start] == '#' && val_end - val_start >= 7) {
                            float r = hex_digit_value(html[val_start+1]) * 16 + hex_digit_value(html[val_start+2]);
                            float g = hex_digit_value(html[val_start+3]) * 16 + hex_digit_value(html[val_start+4]);
                            float b = hex_digit_value(html[val_start+5]) * 16 + hex_digit_value(html[val_start+6]);
                            style->background_color[0] = r / 255.0;
                            style->background_color[1] = g / 255.0;
                            style->background_color[2] = b / 255.0;
                            style->background_color[3] = 1.0;
                        }
                        style->properties_set |= (1 << 6);
                    }
                    // margin: hash = 222093699
                    else if (prop_hash == 222093699) {
                        float vals[4] = {0, 0, 0, 0};
                        int val_count = 0;
                        uint k = val_start;
                        while (k < val_end && val_count < 4) {
                            while (k < val_end && html[k] == ' ') k++;
                            if (k >= val_end) break;
                            float v = 0;
                            while (k < val_end && html[k] >= '0' && html[k] <= '9') {
                                v = v * 10 + (html[k] - '0');
                                k++;
                            }
                            while (k < val_end && html[k] != ' ' && !(html[k] >= '0' && html[k] <= '9')) k++;
                            vals[val_count++] = v;
                        }
                        if (val_count == 1) {
                            style->margin[0] = style->margin[1] = style->margin[2] = style->margin[3] = vals[0];
                        } else if (val_count == 2) {
                            style->margin[0] = style->margin[2] = vals[0];
                            style->margin[1] = style->margin[3] = vals[1];
                        } else if (val_count == 3) {
                            style->margin[0] = vals[0];
                            style->margin[1] = style->margin[3] = vals[1];
                            style->margin[2] = vals[2];
                        } else if (val_count == 4) {
                            style->margin[0] = vals[0]; style->margin[1] = vals[1];
                            style->margin[2] = vals[2]; style->margin[3] = vals[3];
                        }
                        style->properties_set |= (1 << 3);
                    }
                }
                return;
            }
        }
        i++;
    }
}

// Count child index of element (1-based)
int get_child_index(
    device const Element* elements,
    uint elem_idx,
    uint element_count
) {
    Element elem = elements[elem_idx];
    if (elem.parent < 0) return 1;

    int parent_idx = elem.parent;
    Element parent = elements[parent_idx];

    int index = 1;
    int child = parent.first_child;
    while (child >= 0 && uint(child) < element_count) {
        if (uint(child) == elem_idx) {
            return index;
        }
        // Skip text nodes for child indexing
        if (elements[child].element_type != ELEM_TEXT) {
            index++;
        }
        child = elements[child].next_sibling;
    }
    return index;
}

// Count total children of parent (excluding text nodes)
int get_sibling_count(
    device const Element* elements,
    uint elem_idx,
    uint element_count
) {
    Element elem = elements[elem_idx];
    if (elem.parent < 0) return 1;

    Element parent = elements[elem.parent];

    int count = 0;
    int child = parent.first_child;
    while (child >= 0 && uint(child) < element_count) {
        if (elements[child].element_type != ELEM_TEXT) {
            count++;
        }
        child = elements[child].next_sibling;
    }
    return count;
}

// Check if element matches nth-child formula (an+b)
bool matches_nth_child(int child_index, int a, int b) {
    if (a == 0) {
        return child_index == b;
    }
    int diff = child_index - b;
    if (a > 0) {
        return diff >= 0 && diff % a == 0;
    } else {
        return diff <= 0 && (-diff) % (-a) == 0;
    }
}

// Check pseudo-class match
bool check_pseudo_class(
    device const Element* elements,
    uint elem_idx,
    uint element_count,
    uint pseudo_type,
    int nth_a,
    int nth_b
) {
    Element elem = elements[elem_idx];

    switch (pseudo_type) {
        case PSEUDO_ROOT:
            return elem.parent < 0;

        case PSEUDO_EMPTY:
            return elem.first_child < 0;

        case PSEUDO_FIRST_CHILD: {
            if (elem.parent < 0) return true;
            Element parent = elements[elem.parent];
            int first = parent.first_child;
            // Skip text nodes
            while (first >= 0 && elements[first].element_type == ELEM_TEXT) {
                first = elements[first].next_sibling;
            }
            return first >= 0 && uint(first) == elem_idx;
        }

        case PSEUDO_LAST_CHILD: {
            if (elem.parent < 0) return true;
            // Check if no non-text sibling after
            int sibling = elem.next_sibling;
            while (sibling >= 0) {
                if (elements[sibling].element_type != ELEM_TEXT) {
                    return false;
                }
                sibling = elements[sibling].next_sibling;
            }
            return true;
        }

        case PSEUDO_ONLY_CHILD: {
            if (elem.parent < 0) return true;
            int count = get_sibling_count(elements, elem_idx, element_count);
            return count == 1;
        }

        case PSEUDO_NTH_CHILD: {
            int index = get_child_index(elements, elem_idx, element_count);
            return matches_nth_child(index, nth_a, nth_b);
        }

        case PSEUDO_FIRST_OF_TYPE: {
            if (elem.parent < 0) return true;
            Element parent = elements[elem.parent];
            int child = parent.first_child;
            while (child >= 0) {
                if (elements[child].element_type == elem.element_type) {
                    return uint(child) == elem_idx;
                }
                child = elements[child].next_sibling;
            }
            return true;
        }

        case PSEUDO_LAST_OF_TYPE: {
            int sibling = elem.next_sibling;
            while (sibling >= 0) {
                if (elements[sibling].element_type == elem.element_type) {
                    return false;
                }
                sibling = elements[sibling].next_sibling;
            }
            return true;
        }

        default:
            return true;  // Unknown pseudo-class - no match requirement
    }
}

// Initialize default style
ComputedStyle default_style() {
    ComputedStyle s;
    s.display = DISPLAY_BLOCK;
    s.width = 0;  // auto
    s.height = 0;
    s.margin[0] = 0; s.margin[1] = 0; s.margin[2] = 0; s.margin[3] = 0;
    s.padding[0] = 0; s.padding[1] = 0; s.padding[2] = 0; s.padding[3] = 0;
    s.flex_direction = FLEX_ROW;
    s.justify_content = JUSTIFY_START;
    s.align_items = ALIGN_STRETCH;
    s.flex_grow = 0;
    s.flex_shrink = 1;
    s.color[0] = 0; s.color[1] = 0; s.color[2] = 0; s.color[3] = 1;  // Black
    s.font_size = 16;
    s.line_height = 1.2;
    s.font_weight = 400;
    s.text_align = TEXT_LEFT;
    s.background_color[0] = 0; s.background_color[1] = 0;
    s.background_color[2] = 0; s.background_color[3] = 0;  // Transparent
    s.border_width[0] = 0; s.border_width[1] = 0;
    s.border_width[2] = 0; s.border_width[3] = 0;
    s.border_color[0] = 0; s.border_color[1] = 0;
    s.border_color[2] = 0; s.border_color[3] = 1;  // Black
    s.border_radius = 0;
    s.opacity = 1;
    // CSS Positioning defaults
    s.position = POSITION_STATIC;
    s.top = OFFSET_AUTO;
    s.right_ = OFFSET_AUTO;
    s.bottom = OFFSET_AUTO;
    s.left = OFFSET_AUTO;
    s.z_index = 0;
    s.properties_set = 0;  // No properties explicitly set
    // CSS Overflow defaults
    s.overflow_x = OVERFLOW_VISIBLE;
    s.overflow_y = OVERFLOW_VISIBLE;
    // Box Shadow defaults
    s.shadow_count = 0;
    for (int i = 0; i < 4; i++) {
        s.shadow_offset_x[i] = 0;
        s.shadow_offset_y[i] = 0;
        s.shadow_blur[i] = 0;
        s.shadow_spread[i] = 0;
        s.shadow_inset[i] = 0;
    }
    for (int i = 0; i < 16; i++) {
        s.shadow_color[i] = 0;
    }
    // Gradient defaults
    s.gradient_type = GRADIENT_NONE;
    s.gradient_angle = 180.0;  // Default: to bottom
    s.gradient_stop_count = 0;
    for (int i = 0; i < 32; i++) {
        s.gradient_stop_colors[i] = 0;
    }
    for (int i = 0; i < 8; i++) {
        s.gradient_stop_positions[i] = 0;
    }
    // Table layout defaults
    s.border_collapse = 0;  // separate
    s.border_spacing = 0;
    s._padding[0] = 0;
    s._padding[1] = 0;
    return s;
}

// Apply a single style definition
void apply_style_def(thread ComputedStyle* style, StyleDef def) {
    switch (def.property_id) {
        case PROP_DISPLAY:
            style->display = uint(def.values[0]);
            break;
        case PROP_WIDTH:
            style->width = def.values[0];
            break;
        case PROP_HEIGHT:
            style->height = def.values[0];
            break;
        case PROP_MARGIN:
            style->margin[0] = def.values[0];
            style->margin[1] = def.values[1];
            style->margin[2] = def.values[2];
            style->margin[3] = def.values[3];
            break;
        case PROP_PADDING:
            style->padding[0] = def.values[0];
            style->padding[1] = def.values[1];
            style->padding[2] = def.values[2];
            style->padding[3] = def.values[3];
            break;
        case PROP_COLOR:
            style->color[0] = def.values[0];
            style->color[1] = def.values[1];
            style->color[2] = def.values[2];
            style->color[3] = def.values[3];
            style->properties_set |= PROP_SET_COLOR;
            break;
        case PROP_BACKGROUND:
            style->background_color[0] = def.values[0];
            style->background_color[1] = def.values[1];
            style->background_color[2] = def.values[2];
            style->background_color[3] = def.values[3];
            break;
        case PROP_FONT_SIZE:
            style->font_size = def.values[0];
            style->properties_set |= PROP_SET_FONT_SIZE;
            break;
        case PROP_LINE_HEIGHT:
            style->line_height = def.values[0];
            style->properties_set |= PROP_SET_LINE_HEIGHT;
            break;
        case PROP_FONT_WEIGHT:
            style->font_weight = uint(def.values[0]);
            style->properties_set |= PROP_SET_FONT_WEIGHT;
            break;
        case PROP_TEXT_ALIGN:
            style->text_align = uint(def.values[0]);
            style->properties_set |= PROP_SET_TEXT_ALIGN;
            break;
        case PROP_FLEX_DIRECTION:
            style->flex_direction = uint(def.values[0]);
            break;
        case PROP_JUSTIFY_CONTENT:
            style->justify_content = uint(def.values[0]);
            break;
        case PROP_ALIGN_ITEMS:
            style->align_items = uint(def.values[0]);
            break;
        case PROP_FLEX_GROW:
            style->flex_grow = def.values[0];
            break;
        case PROP_FLEX_SHRINK:
            style->flex_shrink = def.values[0];
            break;
        case PROP_BORDER_WIDTH:
            style->border_width[0] = def.values[0];
            style->border_width[1] = def.values[1];
            style->border_width[2] = def.values[2];
            style->border_width[3] = def.values[3];
            break;
        case PROP_BORDER_COLOR:
            style->border_color[0] = def.values[0];
            style->border_color[1] = def.values[1];
            style->border_color[2] = def.values[2];
            style->border_color[3] = def.values[3];
            break;
        case PROP_BORDER_RADIUS:
            style->border_radius = def.values[0];
            break;
        case PROP_OPACITY:
            style->opacity = def.values[0];
            break;
        case PROP_POSITION:
            style->position = uint(def.values[0]);
            break;
        case PROP_TOP:
            style->top = def.values[0];
            break;
        case PROP_RIGHT:
            style->right_ = def.values[0];
            break;
        case PROP_BOTTOM:
            style->bottom = def.values[0];
            break;
        case PROP_LEFT:
            style->left = def.values[0];
            break;
        case PROP_Z_INDEX:
            style->z_index = int(def.values[0]);
            break;
        case PROP_OVERFLOW:
            // Shorthand sets both x and y
            style->overflow_x = uint(def.values[0]);
            style->overflow_y = uint(def.values[1]);
            break;
        case PROP_OVERFLOW_X:
            style->overflow_x = uint(def.values[0]);
            break;
        case PROP_OVERFLOW_Y:
            style->overflow_y = uint(def.values[0]);
            break;
        case PROP_BOX_SHADOW: {
            // Decode shadow index from offset_x (encoded as idx * 1000 + real_offset)
            float encoded_x = def.values[0];
            uint idx = uint(encoded_x / 1000.0);
            if (idx < 4) {
                style->shadow_offset_x[idx] = encoded_x - float(idx * 1000);
                style->shadow_offset_y[idx] = def.values[1];
                style->shadow_blur[idx] = def.values[2];
                style->shadow_spread[idx] = def.values[3];
                if (idx >= style->shadow_count) {
                    style->shadow_count = idx + 1;
                }
            }
            break;
        }
        case PROP_BOX_SHADOW_COLOR: {
            // Decode shadow index from red channel (encoded as idx * 10 + real_red)
            float encoded_r = def.values[0];
            uint idx = uint(encoded_r / 10.0);
            if (idx < 4) {
                uint base = idx * 4;
                style->shadow_color[base + 0] = encoded_r - float(idx * 10);
                style->shadow_color[base + 1] = def.values[1];
                style->shadow_color[base + 2] = def.values[2];
                style->shadow_color[base + 3] = def.values[3];
            }
            break;
        }
        case PROP_BOX_SHADOW_INSET: {
            uint idx = uint(def.values[0]);
            if (idx < 4) {
                style->shadow_inset[idx] = uint(def.values[1]);
            }
            break;
        }
        case PROP_GRADIENT_TYPE:
            style->gradient_type = uint(def.values[0]);
            break;
        case PROP_GRADIENT_ANGLE:
            style->gradient_angle = def.values[0];
            break;
        case PROP_GRADIENT_STOP: {
            uint idx = uint(def.values[0]);
            if (idx < 8) {
                style->gradient_stop_positions[idx] = def.values[1];
                // Decode colors from packed format
                float rg = def.values[2];
                float ba = def.values[3];
                float r = fmod(rg, 256.0);
                float g = floor(rg / 256.0);
                float b = fmod(ba, 256.0);
                float a = floor(ba / 256.0);
                uint base = idx * 4;
                style->gradient_stop_colors[base + 0] = r;
                style->gradient_stop_colors[base + 1] = g;
                style->gradient_stop_colors[base + 2] = b;
                style->gradient_stop_colors[base + 3] = a;
                if (idx >= style->gradient_stop_count) {
                    style->gradient_stop_count = idx + 1;
                }
            }
            break;
        }
        case PROP_BORDER_COLLAPSE:
            style->border_collapse = uint(def.values[0]);
            break;
        case PROP_BORDER_SPACING:
            style->border_spacing = def.values[0];
            break;
    }
}

// Inherit inheritable properties from parent (only if not explicitly set)
// Thread-local style from thread-local parent
void inherit_styles_local(thread ComputedStyle* style, thread const ComputedStyle* parent) {
    // Only inherit if property was not explicitly set by CSS
    if (!(style->properties_set & PROP_SET_COLOR)) {
        style->color[0] = parent->color[0];
        style->color[1] = parent->color[1];
        style->color[2] = parent->color[2];
        style->color[3] = parent->color[3];
    }
    if (!(style->properties_set & PROP_SET_FONT_SIZE)) {
        style->font_size = parent->font_size;
    }
    if (!(style->properties_set & PROP_SET_LINE_HEIGHT)) {
        style->line_height = parent->line_height;
    }
    if (!(style->properties_set & PROP_SET_FONT_WEIGHT)) {
        style->font_weight = parent->font_weight;
    }
    if (!(style->properties_set & PROP_SET_TEXT_ALIGN)) {
        style->text_align = parent->text_align;
    }
}

// Inherit inheritable properties from parent (device style from device parent)
// Only inherits properties that were not explicitly set by CSS
void inherit_styles_device(device ComputedStyle* style, device const ComputedStyle* parent) {
    uint props_set = style->properties_set;

    // Only inherit if property was not explicitly set by CSS
    if (!(props_set & PROP_SET_COLOR)) {
        style->color[0] = parent->color[0];
        style->color[1] = parent->color[1];
        style->color[2] = parent->color[2];
        style->color[3] = parent->color[3];
    }
    if (!(props_set & PROP_SET_FONT_SIZE)) {
        style->font_size = parent->font_size;
    }
    if (!(props_set & PROP_SET_LINE_HEIGHT)) {
        style->line_height = parent->line_height;
    }
    if (!(props_set & PROP_SET_FONT_WEIGHT)) {
        style->font_weight = parent->font_weight;
    }
    if (!(props_set & PROP_SET_TEXT_ALIGN)) {
        style->text_align = parent->text_align;
    }
}

// Check if a simple selector part matches an element
bool matches_simple_selector(
    device const Element* elements,
    device const uint8_t* html,
    device const Token* tokens,
    device const Selector* selectors,
    uint elem_idx,
    uint selector_idx,
    uint element_count
) {
    Element elem = elements[elem_idx];
    Selector sel = selectors[selector_idx];

    // Check selector type
    bool type_match = false;
    switch (sel.selector_type) {
        case SEL_UNIVERSAL:
            type_match = true;
            break;
        case SEL_TAG:
            type_match = (element_type_hash(elem.element_type) == sel.hash);
            break;
        case SEL_CLASS:
            type_match = element_has_class(elements, html, tokens, elem_idx, sel.hash);
            break;
        case SEL_ID:
            type_match = element_has_id(elements, html, tokens, elem_idx, sel.hash);
            break;
        case SEL_TAG_CLASS:
            // Combined tag.class - must match both tag AND class
            // hash contains tag hash, attr_name_hash contains class hash
            type_match = (element_type_hash(elem.element_type) == sel.hash) &&
                         element_has_class(elements, html, tokens, elem_idx, sel.attr_name_hash);
            break;
        case SEL_TAG_ID:
            // Combined tag#id - must match both tag AND id
            // hash contains tag hash, attr_name_hash contains id hash
            type_match = (element_type_hash(elem.element_type) == sel.hash) &&
                         element_has_id(elements, html, tokens, elem_idx, sel.attr_name_hash);
            break;
        case SEL_ATTRIBUTE:
            if (sel.attr_op == ATTR_EXISTS) {
                type_match = element_has_attribute(elements, html, tokens, elem_idx, sel.attr_name_hash);
            } else if (sel.attr_op == ATTR_EQUALS) {
                uint val_hash = get_attribute_value_hash(elements, html, tokens, elem_idx, sel.attr_name_hash);
                type_match = (val_hash == sel.attr_value_hash);
            } else {
                // Other attr ops - simplified, just check existence for now
                type_match = element_has_attribute(elements, html, tokens, elem_idx, sel.attr_name_hash);
            }
            break;
        default:
            type_match = true;  // For pseudo-only selectors
    }

    if (!type_match) return false;

    // Check pseudo-class if present
    if (sel.pseudo_type != 0) {
        if (!check_pseudo_class(elements, elem_idx, element_count, sel.pseudo_type, sel.nth_a, sel.nth_b)) {
            return false;
        }
    }

    return true;
}

// Check if a complex selector (with combinators) matches an element
// Returns true if the entire selector chain matches
bool matches_complex_selector(
    device const Element* elements,
    device const uint8_t* html,
    device const Token* tokens,
    device const Selector* selectors,
    uint elem_idx,
    uint first_selector_idx,
    uint element_count
) {
    // Work backwards through the selector chain
    // The first selector matches the target element
    // Then follow combinators to check ancestors/siblings

    int current_elem = int(elem_idx);
    int current_sel = int(first_selector_idx);

    while (current_sel >= 0) {
        Selector sel = selectors[current_sel];

        // Check if current element matches this selector part
        if (!matches_simple_selector(elements, html, tokens, selectors, uint(current_elem), uint(current_sel), element_count)) {
            return false;
        }

        // Move to next selector part (if any)
        if (sel.next_part < 0) {
            // No more parts - full match
            return true;
        }

        // Get the next selector part and its combinator requirement
        Selector next_sel = selectors[sel.next_part];

        // Based on combinator, find the right element to check
        switch (sel.combinator) {
            case COMB_DESCENDANT: {
                // Need to find any ancestor that matches next_sel
                Element child_elem = elements[current_elem];
                int ancestor = child_elem.parent;
                bool found = false;
                while (ancestor >= 0) {
                    // Try matching next selector on this ancestor
                    if (matches_simple_selector(elements, html, tokens, selectors, uint(ancestor), uint(sel.next_part), element_count)) {
                        current_elem = ancestor;
                        found = true;
                        break;
                    }
                    ancestor = elements[ancestor].parent;
                }
                if (!found) return false;
                break;
            }

            case COMB_CHILD: {
                // Next part must match direct parent
                Element child_elem = elements[current_elem];
                if (child_elem.parent < 0) return false;
                current_elem = child_elem.parent;
                break;
            }

            case COMB_ADJACENT: {
                // Next part must match immediate previous sibling
                // Need to find previous sibling - walk from parent's first child
                Element child_elem = elements[current_elem];
                if (child_elem.parent < 0) return false;
                Element parent = elements[child_elem.parent];
                int prev_sibling = -1;
                int child = parent.first_child;
                while (child >= 0 && child != current_elem) {
                    if (elements[child].element_type != ELEM_TEXT) {
                        prev_sibling = child;
                    }
                    child = elements[child].next_sibling;
                }
                if (prev_sibling < 0) return false;
                current_elem = prev_sibling;
                break;
            }

            case COMB_SIBLING: {
                // Next part must match any previous sibling
                Element child_elem = elements[current_elem];
                if (child_elem.parent < 0) return false;
                Element parent = elements[child_elem.parent];
                bool found = false;
                int child = parent.first_child;
                while (child >= 0 && child != current_elem) {
                    if (elements[child].element_type != ELEM_TEXT) {
                        if (matches_simple_selector(elements, html, tokens, selectors, uint(child), uint(sel.next_part), element_count)) {
                            current_elem = child;
                            found = true;
                            break;
                        }
                    }
                    child = elements[child].next_sibling;
                }
                if (!found) return false;
                break;
            }

            default:
                // No combinator - shouldn't happen for chained selectors
                break;
        }

        current_sel = sel.next_part;
    }

    return true;
}

// Main style resolution kernel
kernel void resolve_styles(
    device const Element* elements [[buffer(0)]],
    device const uint8_t* html [[buffer(1)]],
    device const Token* tokens [[buffer(2)]],
    device const Selector* selectors [[buffer(3)]],
    device const StyleDef* style_defs [[buffer(4)]],
    device ComputedStyle* computed [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    constant uint& selector_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];

    // Initialize with defaults
    ComputedStyle style = default_style();

    // Text nodes inherit from parent, no direct styling
    if (elem.element_type == ELEM_TEXT) {
        if (elem.parent >= 0) {
            // For text nodes, just copy parent style
            // (Parent should already be computed due to tree order)
            style = computed[elem.parent];
        }
        computed[gid] = style;
        return;
    }

    // Track matched selectors
    uint matches[MAX_MATCHES];
    uint match_specs[MAX_MATCHES];
    uint match_count = 0;

    // Match all selectors against this element
    for (uint s = 0; s < selector_count && match_count < MAX_MATCHES; s++) {
        Selector sel = selectors[s];

        // Only process "root" selectors (those with style_count > 0)
        if (sel.style_count == 0) continue;

        bool matched = matches_complex_selector(elements, html, tokens, selectors, gid, s, element_count);

        if (matched) {
            matches[match_count] = s;
            match_specs[match_count] = sel.specificity;
            match_count++;
        }
    }

    // Sort by specificity (insertion sort - small array)
    for (uint i = 1; i < match_count; i++) {
        uint key_idx = matches[i];
        uint key_spec = match_specs[i];
        int j = int(i) - 1;
        while (j >= 0 && match_specs[j] > key_spec) {
            matches[j + 1] = matches[j];
            match_specs[j + 1] = match_specs[j];
            j--;
        }
        matches[j + 1] = key_idx;
        match_specs[j + 1] = key_spec;
    }

    // Apply styles in order (lowest specificity first, so higher overwrites)
    for (uint m = 0; m < match_count; m++) {
        Selector sel = selectors[matches[m]];
        for (uint d = 0; d < sel.style_count; d++) {
            apply_style_def(&style, style_defs[sel.style_start + d]);
        }
    }

    // Apply inline styles directly from HTML (highest specificity - override all CSS rules)
    // GPU-native: parse style="..." attribute directly on GPU
    Token tok = tokens[elem.token_index];
    parse_inline_style_gpu(html, tok.start, tok.end, &style);

    // HEURISTIC: Hide elements with common hidden class patterns
    // This compensates for not loading external CSS files
    // Look for class="...hidden..." or class="...noprint..." in the tag
    {
        uint pos = tok.start;
        uint end = tok.end < tok.start + 500 ? tok.end : tok.start + 500; // limit scan
        while (pos < end) {
            // Look for "class=" or "class ="
            if (pos + 6 < end &&
                html[pos] == 'c' && html[pos+1] == 'l' && html[pos+2] == 'a' &&
                html[pos+3] == 's' && html[pos+4] == 's') {
                pos += 5;
                while (pos < end && (html[pos] == ' ' || html[pos] == '=')) pos++;
                if (pos < end && (html[pos] == '"' || html[pos] == '\'')) {
                    uint quote = html[pos];
                    pos++;
                    uint class_start = pos;
                    while (pos < end && html[pos] != quote) pos++;
                    uint class_end = pos;

                    // Scan class value for hidden patterns
                    for (uint i = class_start; i + 5 < class_end; i++) {
                        // "hidden"
                        if (html[i] == 'h' && html[i+1] == 'i' && html[i+2] == 'd' &&
                            html[i+3] == 'd' && html[i+4] == 'e' && html[i+5] == 'n') {
                            style.display = DISPLAY_NONE;
                            break;
                        }
                        // "noprint"
                        if (html[i] == 'n' && html[i+1] == 'o' && html[i+2] == 'p' &&
                            html[i+3] == 'r' && html[i+4] == 'i' && html[i+5] == 'n') {
                            style.display = DISPLAY_NONE;
                            break;
                        }
                        // "collapsed" (Wikipedia sidebar)
                        if (i + 8 < class_end &&
                            html[i] == 'c' && html[i+1] == 'o' && html[i+2] == 'l' &&
                            html[i+3] == 'l' && html[i+4] == 'a' && html[i+5] == 'p' &&
                            html[i+6] == 's' && html[i+7] == 'e' && html[i+8] == 'd') {
                            style.display = DISPLAY_NONE;
                            break;
                        }
                    }
                }
                break; // Done with class attribute
            }
            pos++;
        }
    }

    // HEURISTIC: Hide common invisible elements (script, style, template, noscript)
    if (elem.element_type == 12 ||  // script
        elem.element_type == 13 ||  // style
        elem.element_type == 28 ||  // template (if we have it)
        elem.element_type == 29) {  // noscript (if we have it)
        style.display = DISPLAY_NONE;
    }

    // Note: Inheritance is handled by apply_inheritance kernel after this pass
    // This ensures explicit selector values are not overwritten

    computed[gid] = style;
}

// Second pass for inheritance (ensures parents are computed first)
kernel void apply_inheritance(
    device const Element* elements [[buffer(0)]],
    device ComputedStyle* computed [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];

    if (elem.parent >= 0) {
        inherit_styles_device(&computed[gid], &computed[elem.parent]);
    }
}
