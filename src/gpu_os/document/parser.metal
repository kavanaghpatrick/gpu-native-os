#include <metal_stdlib>
using namespace metal;

// Token types (must match tokenizer)
constant uint TOKEN_NONE = 0;
constant uint TOKEN_TAG_OPEN = 1;
constant uint TOKEN_TAG_CLOSE = 2;
constant uint TOKEN_TAG_SELF = 3;
constant uint TOKEN_TEXT = 4;
constant uint TOKEN_COMMENT = 5;
constant uint TOKEN_DOCTYPE = 6;

// Element types
constant uint ELEM_UNKNOWN = 0;
constant uint ELEM_DIV = 1;
constant uint ELEM_SPAN = 2;
constant uint ELEM_P = 3;
constant uint ELEM_A = 4;
constant uint ELEM_H1 = 5;
constant uint ELEM_H2 = 6;
constant uint ELEM_H3 = 7;
constant uint ELEM_H4 = 8;
constant uint ELEM_H5 = 9;
constant uint ELEM_H6 = 10;
constant uint ELEM_UL = 11;
constant uint ELEM_OL = 12;
constant uint ELEM_LI = 13;
constant uint ELEM_IMG = 14;
constant uint ELEM_BR = 15;
constant uint ELEM_HR = 16;
constant uint ELEM_TABLE = 17;
constant uint ELEM_TR = 18;
constant uint ELEM_TD = 19;
constant uint ELEM_TH = 20;
constant uint ELEM_THEAD = 21;
constant uint ELEM_TBODY = 22;
constant uint ELEM_FORM = 23;
constant uint ELEM_INPUT = 24;
constant uint ELEM_BUTTON = 25;
constant uint ELEM_TEXTAREA = 26;
constant uint ELEM_SELECT = 27;
constant uint ELEM_OPTION = 28;
constant uint ELEM_LABEL = 29;
constant uint ELEM_NAV = 30;
constant uint ELEM_HEADER = 31;
constant uint ELEM_FOOTER = 32;
constant uint ELEM_MAIN = 33;
constant uint ELEM_SECTION = 34;
constant uint ELEM_ARTICLE = 35;
constant uint ELEM_ASIDE = 36;
constant uint ELEM_PRE = 37;
constant uint ELEM_CODE = 38;
constant uint ELEM_BLOCKQUOTE = 39;
constant uint ELEM_STRONG = 40;
constant uint ELEM_EM = 41;
constant uint ELEM_B = 42;
constant uint ELEM_I = 43;
constant uint ELEM_U = 44;
constant uint ELEM_SMALL = 45;
constant uint ELEM_SUB = 46;
constant uint ELEM_SUP = 47;
constant uint ELEM_TEXT = 100;
constant uint ELEM_HTML = 101;
constant uint ELEM_HEAD = 102;
constant uint ELEM_BODY = 103;
constant uint ELEM_TITLE = 104;
constant uint ELEM_META = 105;
constant uint ELEM_LINK = 106;
constant uint ELEM_STYLE = 107;
constant uint ELEM_SCRIPT = 108;

// Limits
constant uint THREAD_COUNT = 1024;
constant uint MAX_ELEMENTS = 16384;
constant uint MAX_TEXT_SIZE = 512 * 1024;
constant uint MAX_STACK_DEPTH = 256;

struct Token {
    uint token_type;
    uint start;
    uint end;
    uint _padding;
};

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

// Helper: Compare tag name (case insensitive)
bool tag_equals(device const uint8_t* html, uint start, uint len, constant char* name, uint name_len) {
    if (len != name_len) return false;
    for (uint i = 0; i < len; i++) {
        char c = html[start + i];
        // Convert to lowercase
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c != name[i]) return false;
    }
    return true;
}

// Parse tag name from token and return element type
uint parse_tag_type(device const uint8_t* html, uint start, uint end) {
    // Skip '<' and optional '/'
    uint i = start + 1;
    if (i < end && html[i] == '/') i++;
    if (i < end && html[i] == '!') return ELEM_UNKNOWN;  // Skip DOCTYPE, comments

    uint name_start = i;
    while (i < end && html[i] != ' ' && html[i] != '>' && html[i] != '/' && html[i] != '\n' && html[i] != '\t') {
        i++;
    }
    uint name_len = i - name_start;

    if (name_len == 0) return ELEM_UNKNOWN;

    // Match by length first for efficiency
    switch (name_len) {
        case 1:
            if (tag_equals(html, name_start, name_len, "p", 1)) return ELEM_P;
            if (tag_equals(html, name_start, name_len, "a", 1)) return ELEM_A;
            if (tag_equals(html, name_start, name_len, "b", 1)) return ELEM_B;
            if (tag_equals(html, name_start, name_len, "i", 1)) return ELEM_I;
            if (tag_equals(html, name_start, name_len, "u", 1)) return ELEM_U;
            break;
        case 2:
            if (tag_equals(html, name_start, name_len, "h1", 2)) return ELEM_H1;
            if (tag_equals(html, name_start, name_len, "h2", 2)) return ELEM_H2;
            if (tag_equals(html, name_start, name_len, "h3", 2)) return ELEM_H3;
            if (tag_equals(html, name_start, name_len, "h4", 2)) return ELEM_H4;
            if (tag_equals(html, name_start, name_len, "h5", 2)) return ELEM_H5;
            if (tag_equals(html, name_start, name_len, "h6", 2)) return ELEM_H6;
            if (tag_equals(html, name_start, name_len, "ul", 2)) return ELEM_UL;
            if (tag_equals(html, name_start, name_len, "ol", 2)) return ELEM_OL;
            if (tag_equals(html, name_start, name_len, "li", 2)) return ELEM_LI;
            if (tag_equals(html, name_start, name_len, "br", 2)) return ELEM_BR;
            if (tag_equals(html, name_start, name_len, "hr", 2)) return ELEM_HR;
            if (tag_equals(html, name_start, name_len, "tr", 2)) return ELEM_TR;
            if (tag_equals(html, name_start, name_len, "td", 2)) return ELEM_TD;
            if (tag_equals(html, name_start, name_len, "th", 2)) return ELEM_TH;
            if (tag_equals(html, name_start, name_len, "em", 2)) return ELEM_EM;
            break;
        case 3:
            if (tag_equals(html, name_start, name_len, "div", 3)) return ELEM_DIV;
            if (tag_equals(html, name_start, name_len, "img", 3)) return ELEM_IMG;
            if (tag_equals(html, name_start, name_len, "nav", 3)) return ELEM_NAV;
            if (tag_equals(html, name_start, name_len, "pre", 3)) return ELEM_PRE;
            if (tag_equals(html, name_start, name_len, "sub", 3)) return ELEM_SUB;
            if (tag_equals(html, name_start, name_len, "sup", 3)) return ELEM_SUP;
            break;
        case 4:
            if (tag_equals(html, name_start, name_len, "span", 4)) return ELEM_SPAN;
            if (tag_equals(html, name_start, name_len, "html", 4)) return ELEM_HTML;
            if (tag_equals(html, name_start, name_len, "head", 4)) return ELEM_HEAD;
            if (tag_equals(html, name_start, name_len, "body", 4)) return ELEM_BODY;
            if (tag_equals(html, name_start, name_len, "form", 4)) return ELEM_FORM;
            if (tag_equals(html, name_start, name_len, "code", 4)) return ELEM_CODE;
            if (tag_equals(html, name_start, name_len, "main", 4)) return ELEM_MAIN;
            if (tag_equals(html, name_start, name_len, "meta", 4)) return ELEM_META;
            if (tag_equals(html, name_start, name_len, "link", 4)) return ELEM_LINK;
            break;
        case 5:
            if (tag_equals(html, name_start, name_len, "table", 5)) return ELEM_TABLE;
            if (tag_equals(html, name_start, name_len, "thead", 5)) return ELEM_THEAD;
            if (tag_equals(html, name_start, name_len, "tbody", 5)) return ELEM_TBODY;
            if (tag_equals(html, name_start, name_len, "input", 5)) return ELEM_INPUT;
            if (tag_equals(html, name_start, name_len, "label", 5)) return ELEM_LABEL;
            if (tag_equals(html, name_start, name_len, "title", 5)) return ELEM_TITLE;
            if (tag_equals(html, name_start, name_len, "style", 5)) return ELEM_STYLE;
            if (tag_equals(html, name_start, name_len, "small", 5)) return ELEM_SMALL;
            if (tag_equals(html, name_start, name_len, "aside", 5)) return ELEM_ASIDE;
            break;
        case 6:
            if (tag_equals(html, name_start, name_len, "button", 6)) return ELEM_BUTTON;
            if (tag_equals(html, name_start, name_len, "select", 6)) return ELEM_SELECT;
            if (tag_equals(html, name_start, name_len, "option", 6)) return ELEM_OPTION;
            if (tag_equals(html, name_start, name_len, "header", 6)) return ELEM_HEADER;
            if (tag_equals(html, name_start, name_len, "footer", 6)) return ELEM_FOOTER;
            if (tag_equals(html, name_start, name_len, "strong", 6)) return ELEM_STRONG;
            if (tag_equals(html, name_start, name_len, "script", 6)) return ELEM_SCRIPT;
            break;
        case 7:
            if (tag_equals(html, name_start, name_len, "section", 7)) return ELEM_SECTION;
            if (tag_equals(html, name_start, name_len, "article", 7)) return ELEM_ARTICLE;
            break;
        case 8:
            if (tag_equals(html, name_start, name_len, "textarea", 8)) return ELEM_TEXTAREA;
            break;
        case 10:
            if (tag_equals(html, name_start, name_len, "blockquote", 10)) return ELEM_BLOCKQUOTE;
            break;
    }

    return ELEM_DIV;  // Default unknown tags to div-like behavior
}

// Pass 2A: Allocate element slots for tokens
kernel void parse_allocate_elements(
    device const Token* tokens [[buffer(0)]],
    device int* token_to_element [[buffer(1)]],
    device atomic_uint* element_count [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    uint chunk = (token_count + tcount - 1) / tcount;
    uint start = tid * chunk;
    uint end = min(start + chunk, token_count);

    for (uint i = start; i < end; i++) {
        Token t = tokens[i];

        // These token types create elements
        bool creates_element =
            t.token_type == TOKEN_TAG_OPEN ||
            t.token_type == TOKEN_TAG_SELF ||
            t.token_type == TOKEN_TEXT;

        if (creates_element) {
            uint slot = atomic_fetch_add_explicit(element_count, 1, memory_order_relaxed);
            if (slot < MAX_ELEMENTS) {
                token_to_element[i] = int(slot);
            } else {
                token_to_element[i] = -1;  // Overflow
            }
        } else {
            token_to_element[i] = -1;
        }
    }
}

// Pass 2B: Build tree structure (sequential)
// Optimized: track last_child per element to avoid O(n) sibling walks
kernel void parse_build_tree(
    device const Token* tokens [[buffer(0)]],
    device const uint8_t* html [[buffer(1)]],
    device const int* token_to_element [[buffer(2)]],
    device Element* elements [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    threadgroup int* parent_stack [[threadgroup(0)]],    // Stack for parent indices
    threadgroup int* last_child [[threadgroup(1)]],      // Last child per parent (by stack position)
    uint tid [[thread_index_in_threadgroup]]
) {
    // Only thread 0 does tree building
    if (tid != 0) return;

    int stack_ptr = 0;
    int current_parent = -1;
    parent_stack[0] = -1;
    last_child[0] = -1;  // No last child at root level initially

    for (uint i = 0; i < token_count; i++) {
        Token t = tokens[i];
        int elem_idx = token_to_element[i];

        if (elem_idx < 0) {
            // Close tag or non-element token
            if (t.token_type == TOKEN_TAG_CLOSE && stack_ptr > 0) {
                current_parent = parent_stack[stack_ptr];
                stack_ptr--;
            }
            continue;
        }

        // Initialize element
        elements[elem_idx].parent = current_parent;
        elements[elem_idx].first_child = -1;
        elements[elem_idx].next_sibling = -1;
        elements[elem_idx].text_start = 0;
        elements[elem_idx].text_length = 0;
        elements[elem_idx].token_index = i;

        // Link to parent's child list (O(1) using last_child tracking)
        if (current_parent >= 0) {
            int last = last_child[stack_ptr];
            if (last < 0) {
                // First child
                elements[current_parent].first_child = elem_idx;
            } else {
                // Link as sibling of last child
                elements[last].next_sibling = elem_idx;
            }
            last_child[stack_ptr] = elem_idx;  // Update last child
        } else {
            // Root level element
            int last = last_child[0];
            if (last >= 0) {
                elements[last].next_sibling = elem_idx;
            }
            last_child[0] = elem_idx;
        }

        if (t.token_type == TOKEN_TAG_OPEN) {
            elements[elem_idx].element_type = parse_tag_type(html, t.start, t.end);

            // Push onto stack
            if (stack_ptr < MAX_STACK_DEPTH - 1) {
                stack_ptr++;
                parent_stack[stack_ptr] = current_parent;
                last_child[stack_ptr] = -1;  // New parent has no children yet
                current_parent = elem_idx;
            }

        } else if (t.token_type == TOKEN_TAG_SELF) {
            elements[elem_idx].element_type = parse_tag_type(html, t.start, t.end);
            // Self-closing: don't push onto stack

        } else if (t.token_type == TOKEN_TEXT) {
            elements[elem_idx].element_type = ELEM_TEXT;
            // Text node: don't push onto stack
        }
    }
}

// Pass 2C: Extract text content (parallel)
kernel void parse_extract_text(
    device const Token* tokens [[buffer(0)]],
    device const uint8_t* html [[buffer(1)]],
    device Element* elements [[buffer(2)]],
    device char* text_buffer [[buffer(3)]],
    device atomic_uint* text_offset [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    uint chunk = (element_count + tcount - 1) / tcount;
    uint start = tid * chunk;
    uint end = min(start + chunk, element_count);

    for (uint i = start; i < end; i++) {
        if (elements[i].element_type != ELEM_TEXT) continue;

        Token t = tokens[elements[i].token_index];
        uint len = t.end - t.start;

        if (len == 0) continue;

        // Allocate space
        uint offset = atomic_fetch_add_explicit(text_offset, len, memory_order_relaxed);
        if (offset + len > MAX_TEXT_SIZE) continue;  // Overflow protection

        // Copy text
        for (uint j = 0; j < len; j++) {
            text_buffer[offset + j] = html[t.start + j];
        }

        elements[i].text_start = offset;
        elements[i].text_length = len;
    }
}
