#include <metal_stdlib>
using namespace metal;

// Element types (from parser)
constant uint ELEM_TEXT = 100;

// Selector types
constant uint SEL_UNIVERSAL = 0;
constant uint SEL_TAG = 1;
constant uint SEL_CLASS = 2;
constant uint SEL_ID = 3;

// Display values
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

// Text align
constant uint TEXT_LEFT = 0;
constant uint TEXT_CENTER = 1;
constant uint TEXT_RIGHT = 2;

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

// Limits
constant uint MAX_MATCHES = 32;
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

struct Token {
    uint token_type;
    uint start;
    uint end;
    uint _padding;
};

struct Selector {
    uint selector_type;
    uint hash;
    uint specificity;
    uint style_start;    // Start index in style_defs
    uint style_count;    // Number of StyleDefs for this selector
    uint _padding[3];
};

struct StyleDef {
    uint property_id;
    float values[4];
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

    float _padding[2];
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
    s._padding[0] = 0; s._padding[1] = 0;
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
            break;
        case PROP_BACKGROUND:
            style->background_color[0] = def.values[0];
            style->background_color[1] = def.values[1];
            style->background_color[2] = def.values[2];
            style->background_color[3] = def.values[3];
            break;
        case PROP_FONT_SIZE:
            style->font_size = def.values[0];
            break;
        case PROP_LINE_HEIGHT:
            style->line_height = def.values[0];
            break;
        case PROP_FONT_WEIGHT:
            style->font_weight = uint(def.values[0]);
            break;
        case PROP_TEXT_ALIGN:
            style->text_align = uint(def.values[0]);
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
    }
}

// Inherit inheritable properties from parent (thread-local style from thread-local parent)
void inherit_styles_local(thread ComputedStyle* style, thread const ComputedStyle* parent) {
    style->color[0] = parent->color[0];
    style->color[1] = parent->color[1];
    style->color[2] = parent->color[2];
    style->color[3] = parent->color[3];
    style->font_size = parent->font_size;
    style->line_height = parent->line_height;
    style->font_weight = parent->font_weight;
    style->text_align = parent->text_align;
}

// Inherit inheritable properties from parent (device style from device parent)
void inherit_styles_device(device ComputedStyle* style, device const ComputedStyle* parent) {
    style->color[0] = parent->color[0];
    style->color[1] = parent->color[1];
    style->color[2] = parent->color[2];
    style->color[3] = parent->color[3];
    style->font_size = parent->font_size;
    style->line_height = parent->line_height;
    style->font_weight = parent->font_weight;
    style->text_align = parent->text_align;
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
        bool matched = false;

        switch (sel.selector_type) {
            case SEL_UNIVERSAL:
                matched = true;
                break;
            case SEL_TAG:
                matched = (element_type_hash(elem.element_type) == sel.hash);
                break;
            case SEL_CLASS:
                matched = element_has_class(elements, html, tokens, gid, sel.hash);
                break;
            case SEL_ID:
                matched = element_has_id(elements, html, tokens, gid, sel.hash);
                break;
        }

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
