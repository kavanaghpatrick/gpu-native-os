#include <metal_stdlib>
using namespace metal;

// Token types
constant uint TOKEN_NONE = 0;
constant uint TOKEN_TAG_OPEN = 1;      // <div, <p, etc.
constant uint TOKEN_TAG_CLOSE = 2;     // </div>, </p>, etc.
constant uint TOKEN_TAG_SELF = 3;      // <br/>, <img/>, etc.
constant uint TOKEN_TEXT = 4;          // Text content between tags
constant uint TOKEN_COMMENT = 5;       // <!-- comment -->
constant uint TOKEN_DOCTYPE = 6;       // <!DOCTYPE html>

// Boundary markers
constant uint BOUNDARY_NONE = 0;
constant uint BOUNDARY_TAG_START = 1;
constant uint BOUNDARY_TEXT_START = 2;

// Limits
constant uint THREAD_COUNT = 1024;
constant uint MAX_TOKENS = 65536;

struct Token {
    uint token_type;
    uint start;
    uint end;
    uint _padding;
};

// Helper: Find end of tag (closing >)
uint find_tag_end(device const uint8_t* html, uint start, uint length) {
    bool in_string = false;
    char string_char = 0;

    for (uint i = start + 1; i < length; i++) {
        char c = html[i];

        if (!in_string && (c == '"' || c == '\'')) {
            in_string = true;
            string_char = c;
        } else if (in_string && c == string_char) {
            in_string = false;
        } else if (!in_string && c == '>') {
            return i + 1;  // Include the '>'
        }
    }
    return length;
}

// Helper: Find end of text (next '<')
uint find_text_end(device const uint8_t* html, uint start, uint length) {
    for (uint i = start; i < length; i++) {
        if (html[i] == '<') return i;
    }
    return length;
}

// Helper: Find end of comment (-->)
uint find_comment_end(device const uint8_t* html, uint start, uint length) {
    for (uint i = start + 4; i < length - 2; i++) {
        if (html[i] == '-' && html[i+1] == '-' && html[i+2] == '>') {
            return i + 3;
        }
    }
    return length;
}

// Helper: Check if position starts a comment
bool is_comment_start(device const uint8_t* html, uint pos, uint length) {
    return pos + 3 < length &&
           html[pos] == '<' &&
           html[pos+1] == '!' &&
           html[pos+2] == '-' &&
           html[pos+3] == '-';
}

// Helper: Check if position starts DOCTYPE
bool is_doctype_start(device const uint8_t* html, uint pos, uint length) {
    return pos + 8 < length &&
           html[pos] == '<' &&
           html[pos+1] == '!' &&
           (html[pos+2] == 'D' || html[pos+2] == 'd');
}

// Pass 1A: Detect token boundaries
kernel void tokenize_boundaries(
    device const uint8_t* html [[buffer(0)]],
    device uint* boundaries [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    // Each thread processes a chunk
    uint chunk_size = (length + tcount - 1) / tcount;
    uint start = tid * chunk_size;
    uint end = min(start + chunk_size, length);

    if (start >= length) return;

    // State tracking
    bool in_tag = false;
    bool in_string = false;
    bool in_comment = false;
    char string_char = 0;

    // Scan backwards to determine initial state for this chunk
    // (simplified - full implementation would need cross-chunk sync)
    if (start > 0) {
        uint scan_start = (start > 256) ? start - 256 : 0;
        for (uint i = scan_start; i < start; i++) {
            char c = html[i];
            if (in_comment) {
                if (c == '>' && i >= 2 && html[i-1] == '-' && html[i-2] == '-') {
                    in_comment = false;
                }
            } else if (in_string) {
                if (c == string_char) in_string = false;
            } else if (in_tag) {
                if (c == '"' || c == '\'') {
                    in_string = true;
                    string_char = c;
                } else if (c == '>') {
                    in_tag = false;
                }
            } else {
                if (c == '<') {
                    if (is_comment_start(html, i, length)) {
                        in_comment = true;
                    } else {
                        in_tag = true;
                    }
                }
            }
        }
    }

    // Process this chunk
    for (uint i = start; i < end; i++) {
        char c = html[i];

        // Handle comment state
        if (in_comment) {
            if (c == '>' && i >= 2 && html[i-1] == '-' && html[i-2] == '-') {
                in_comment = false;
                // Mark next char as potential text start
                if (i + 1 < length && html[i + 1] != '<') {
                    boundaries[i + 1] = BOUNDARY_TEXT_START;
                }
            }
            continue;
        }

        // Handle string state (inside tag attributes)
        if (in_tag && in_string) {
            if (c == string_char) {
                in_string = false;
            }
            continue;
        }

        if (in_tag && (c == '"' || c == '\'')) {
            in_string = true;
            string_char = c;
            continue;
        }

        // Detect boundaries
        if (c == '<' && !in_tag) {
            boundaries[i] = BOUNDARY_TAG_START;
            if (is_comment_start(html, i, length)) {
                in_comment = true;
            } else {
                in_tag = true;
            }
        } else if (c == '>' && in_tag) {
            in_tag = false;
            // Mark next char as potential text start
            if (i + 1 < length && html[i + 1] != '<' &&
                html[i + 1] != ' ' && html[i + 1] != '\n' &&
                html[i + 1] != '\r' && html[i + 1] != '\t') {
                boundaries[i + 1] = BOUNDARY_TEXT_START;
            }
            // Also check for text with leading whitespace
            if (i + 1 < length && html[i + 1] != '<') {
                // Find first non-whitespace
                for (uint j = i + 1; j < length && j < i + 100; j++) {
                    char nc = html[j];
                    if (nc == '<') break;
                    if (nc != ' ' && nc != '\n' && nc != '\r' && nc != '\t') {
                        boundaries[i + 1] = BOUNDARY_TEXT_START;
                        break;
                    }
                }
            }
        }
    }

    // Special case: mark start of document if it begins with text
    if (tid == 0 && length > 0 && html[0] != '<') {
        boundaries[0] = BOUNDARY_TEXT_START;
    }
}

// Pass 1B: Extract tokens from boundaries
kernel void tokenize_extract(
    device const uint8_t* html [[buffer(0)]],
    device const uint* boundaries [[buffer(1)]],
    device Token* tokens [[buffer(2)]],
    device atomic_uint* token_count [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    uint chunk_size = (length + tcount - 1) / tcount;
    uint start = tid * chunk_size;
    uint end = min(start + chunk_size, length);

    if (start >= length) return;

    for (uint i = start; i < end; i++) {
        uint boundary = boundaries[i];
        if (boundary == BOUNDARY_NONE) continue;

        uint token_type = TOKEN_NONE;
        uint token_end = i;

        if (boundary == BOUNDARY_TAG_START) {
            // Determine tag type
            if (is_comment_start(html, i, length)) {
                token_type = TOKEN_COMMENT;
                token_end = find_comment_end(html, i, length);
            } else if (is_doctype_start(html, i, length)) {
                token_type = TOKEN_DOCTYPE;
                token_end = find_tag_end(html, i, length);
            } else if (i + 1 < length && html[i + 1] == '/') {
                token_type = TOKEN_TAG_CLOSE;
                token_end = find_tag_end(html, i, length);
            } else {
                token_end = find_tag_end(html, i, length);
                // Check for self-closing
                if (token_end > 1 && html[token_end - 2] == '/') {
                    token_type = TOKEN_TAG_SELF;
                } else {
                    token_type = TOKEN_TAG_OPEN;
                }
            }
        } else if (boundary == BOUNDARY_TEXT_START) {
            token_type = TOKEN_TEXT;
            token_end = find_text_end(html, i, length);

            // Skip empty text tokens
            if (token_end == i) {
                token_type = TOKEN_NONE;
            }

            // Skip whitespace-only text tokens (optional - keep for now)
            // bool all_whitespace = true;
            // for (uint j = i; j < token_end; j++) {
            //     char c = html[j];
            //     if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
            //         all_whitespace = false;
            //         break;
            //     }
            // }
            // if (all_whitespace) token_type = TOKEN_NONE;
        }

        if (token_type != TOKEN_NONE && token_end > i) {
            uint slot = atomic_fetch_add_explicit(token_count, 1, memory_order_relaxed);
            if (slot < MAX_TOKENS) {
                tokens[slot].token_type = token_type;
                tokens[slot].start = i;
                tokens[slot].end = token_end;
                tokens[slot]._padding = 0;
            }
        }
    }
}
