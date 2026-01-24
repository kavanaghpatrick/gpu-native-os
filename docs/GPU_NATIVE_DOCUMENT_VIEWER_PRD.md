# GPU-Native Document Viewer PRD

## Overview

A document viewer where the **entire rendering pipeline runs on the GPU** - from raw bytes to pixels. The CPU's role is minimal: accept network connections, DMA bytes to unified memory, and present the final framebuffer.

**Thesis**: By processing documents with 1024 parallel threads at every stage, we can achieve 10-100x faster rendering than traditional sequential browsers.

## Goals

| Goal | Metric | Target |
|------|--------|--------|
| Parse performance | HTML tokens/second | 10M+ |
| Style performance | Element×Selector matches/second | 100M+ |
| Layout performance | Elements laid out/second | 1M+ |
| End-to-end latency | Bytes received → pixels displayed | <1ms for 100KB doc |
| CPU utilization | % of work on CPU | <5% |

## Non-Goals (V1)

- JavaScript execution (future: WASM compute shaders)
- Full CSS spec compliance (focus on Flexbox subset)
- Network protocol handling (assume pre-decrypted HTTP/1.1)
- Incremental updates (full re-render each frame)
- Accessibility (screen reader support)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU (Minimal)                            │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Network I/O │───▶│ Input Events │───▶│ Present Display  │   │
│  └─────────────┘    └──────────────┘    └──────────────────┘   │
│         │                   │                                    │
│         ▼                   ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              UNIFIED MEMORY (Apple Silicon)              │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │    │
│  │  │ Raw Bytes│ │ Document │ │  Layout  │ │  Vertices  │  │    │
│  │  │  Buffer  │ │  Buffer  │ │  Buffer  │ │   Buffer   │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GPU COMPUTE PIPELINE                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PASS 1: Tokenize          (1024 threads)                 │   │
│  │   Raw bytes → Token stream                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PASS 2: Parse             (1024 threads)                 │   │
│  │   Tokens → Element array (flat tree)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PASS 3: Style             (1024 threads)                 │   │
│  │   Elements × Selectors → Computed styles                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PASS 4: Layout            (1024 threads)                 │   │
│  │   Elements + Styles → Positions/Sizes                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PASS 5: Paint             (1024 threads)                 │   │
│  │   Layout boxes → Vertex geometry                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ RENDER PASS: Rasterize                                   │   │
│  │   Vertices → Pixels                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### GPU Buffer Layout

```rust
// All buffers use #[repr(C)] for GPU compatibility

/// Raw document bytes (input)
struct RawBuffer {
    data: [u8; MAX_DOCUMENT_SIZE],  // 1MB max
    length: u32,
}

/// Token produced by tokenizer
#[repr(C)]
struct Token {
    token_type: u32,      // TAG_OPEN, TAG_CLOSE, TEXT, ATTR_NAME, ATTR_VALUE
    start: u32,           // Byte offset in raw buffer
    end: u32,             // Byte offset end
    _padding: u32,
}

/// Element in flat tree representation
#[repr(C)]
struct Element {
    element_type: u32,    // DIV, P, SPAN, IMG, etc. (enum)
    parent: i32,          // Index of parent (-1 for root)
    first_child: i32,     // Index of first child (-1 if none)
    next_sibling: i32,    // Index of next sibling (-1 if none)
    text_start: u32,      // Start of text content in text buffer
    text_length: u32,     // Length of text content
    attr_start: u32,      // Start of attributes in attr buffer
    attr_count: u32,      // Number of attributes
}

/// Computed style for an element
#[repr(C)]
struct ComputedStyle {
    // Box model
    display: u32,         // BLOCK, FLEX, INLINE, NONE
    width: f32,           // Specified width (0 = auto)
    height: f32,          // Specified height (0 = auto)
    margin: [f32; 4],     // top, right, bottom, left
    padding: [f32; 4],

    // Flex
    flex_direction: u32,  // ROW, COLUMN
    justify_content: u32, // START, CENTER, END, SPACE_BETWEEN
    align_items: u32,     // START, CENTER, END, STRETCH
    flex_grow: f32,
    flex_shrink: f32,

    // Visual
    background_color: [f32; 4],  // RGBA
    color: [f32; 4],             // Text color
    font_size: f32,
    line_height: f32,

    // Border
    border_width: [f32; 4],
    border_color: [f32; 4],
    border_radius: f32,
}

/// Layout result for an element
#[repr(C)]
struct LayoutBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    content_x: f32,       // Content area (inside padding)
    content_y: f32,
    content_width: f32,
    content_height: f32,
    scroll_x: f32,
    scroll_y: f32,
}

/// Vertex for rendering
#[repr(C)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    element_id: u32,      // For hit testing
    vertex_type: u32,     // BACKGROUND, BORDER, TEXT
    _padding: [u32; 2],
}
```

### Constants

```rust
const MAX_DOCUMENT_SIZE: usize = 1024 * 1024;  // 1MB
const MAX_TOKENS: usize = 65536;
const MAX_ELEMENTS: usize = 16384;
const MAX_TEXT_SIZE: usize = 512 * 1024;       // 512KB
const MAX_STYLES: usize = 1024;
const MAX_VERTICES: usize = MAX_ELEMENTS * 24; // 4 quads per element max
const THREAD_COUNT: usize = 1024;
```

---

## Pass 1: Tokenizer

### Purpose
Convert raw HTML bytes into a stream of tokens.

### Algorithm
Parallel scan with chunk assignment. Each thread processes a portion of the input, handling boundary conditions through a two-phase approach.

### Pseudocode

```metal
// Token types
constant uint TOKEN_TAG_OPEN = 1;      // <tag
constant uint TOKEN_TAG_CLOSE = 2;     // </tag>
constant uint TOKEN_TAG_SELF = 3;      // <tag/>
constant uint TOKEN_TEXT = 4;          // text content
constant uint TOKEN_ATTR_NAME = 5;     // attribute name
constant uint TOKEN_ATTR_VALUE = 6;    // attribute value

kernel void tokenize_pass1_boundaries(
    device uint8_t* html [[buffer(0)]],
    device uint* boundary_flags [[buffer(1)]],  // 1 if char starts a token
    constant uint& length [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]]
) {
    // Phase 1: Mark token boundaries
    // Each thread scans its chunk and marks where tokens start

    uint chunk_size = (length + tcount - 1) / tcount;
    uint start = tid * chunk_size;
    uint end = min(start + chunk_size, length);

    bool in_tag = false;  // Will be corrected in phase 2
    bool in_string = false;
    char string_char = 0;

    for (uint i = start; i < end; i++) {
        char c = html[i];

        if (in_string) {
            if (c == string_char) {
                in_string = false;
            }
            continue;
        }

        if (c == '"' || c == '\'') {
            in_string = true;
            string_char = c;
            continue;
        }

        if (c == '<') {
            boundary_flags[i] = TOKEN_TAG_OPEN;
            in_tag = true;
        } else if (c == '>' && in_tag) {
            in_tag = false;
            // Mark next char as potential text start
            if (i + 1 < length) {
                boundary_flags[i + 1] |= TOKEN_TEXT;
            }
        }
    }
}

kernel void tokenize_pass2_extract(
    device uint8_t* html [[buffer(0)]],
    device uint* boundary_flags [[buffer(1)]],
    device Token* tokens [[buffer(2)]],
    device atomic_uint* token_count [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Phase 2: Each thread processes boundaries it owns
    // Uses atomic counter to allocate token slots

    uint chunk_size = (length + THREAD_COUNT - 1) / THREAD_COUNT;
    uint start = tid * chunk_size;
    uint end = min(start + chunk_size, length);

    for (uint i = start; i < end; i++) {
        if (boundary_flags[i] != 0) {
            // Find token end
            uint token_end = find_token_end(html, i, length, boundary_flags[i]);

            // Allocate slot
            uint slot = atomic_fetch_add_explicit(token_count, 1, memory_order_relaxed);

            // Write token
            tokens[slot].token_type = boundary_flags[i];
            tokens[slot].start = i;
            tokens[slot].end = token_end;
        }
    }
}

// Helper: Find where current token ends
uint find_token_end(device uint8_t* html, uint start, uint length, uint token_type) {
    if (token_type == TOKEN_TAG_OPEN) {
        // Find closing >
        for (uint i = start; i < length; i++) {
            if (html[i] == '>') return i + 1;
        }
    } else if (token_type == TOKEN_TEXT) {
        // Find next <
        for (uint i = start; i < length; i++) {
            if (html[i] == '<') return i;
        }
    }
    return length;
}
```

### Tests

```rust
#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    #[test]
    fn test_simple_tag() {
        let html = b"<div>hello</div>";
        let tokens = gpu_tokenize(html);

        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].token_type, TOKEN_TAG_OPEN);
        assert_eq!(&html[tokens[0].start..tokens[0].end], b"<div>");
        assert_eq!(tokens[1].token_type, TOKEN_TEXT);
        assert_eq!(&html[tokens[1].start..tokens[1].end], b"hello");
        assert_eq!(tokens[2].token_type, TOKEN_TAG_CLOSE);
    }

    #[test]
    fn test_nested_tags() {
        let html = b"<div><p>text</p></div>";
        let tokens = gpu_tokenize(html);

        assert_eq!(tokens.len(), 5);
        // <div>, <p>, text, </p>, </div>
    }

    #[test]
    fn test_attributes() {
        let html = b"<div class=\"foo\" id='bar'>content</div>";
        let tokens = gpu_tokenize(html);

        // Should handle quoted strings correctly
        assert!(tokens.iter().any(|t| t.token_type == TOKEN_ATTR_NAME));
        assert!(tokens.iter().any(|t| t.token_type == TOKEN_ATTR_VALUE));
    }

    #[test]
    fn test_self_closing() {
        let html = b"<br/><img src=\"x\"/>";
        let tokens = gpu_tokenize(html);

        assert!(tokens.iter().all(|t| t.token_type == TOKEN_TAG_SELF));
    }

    #[test]
    fn test_whitespace_text() {
        let html = b"<div>  hello  world  </div>";
        let tokens = gpu_tokenize(html);

        let text_token = tokens.iter().find(|t| t.token_type == TOKEN_TEXT).unwrap();
        assert_eq!(&html[text_token.start..text_token.end], b"  hello  world  ");
    }

    #[test]
    fn test_chunk_boundary() {
        // Create HTML where a tag spans the chunk boundary
        // With 1024 threads and 2048 byte input, each thread gets 2 bytes
        let mut html = vec![b' '; 1023];
        html.extend_from_slice(b"<div>test</div>");  // Tag starts at boundary

        let tokens = gpu_tokenize(&html);
        assert!(tokens.iter().any(|t| t.token_type == TOKEN_TAG_OPEN));
    }

    #[test]
    fn test_performance_1mb() {
        let html = generate_html_document(1024 * 1024);  // 1MB

        let start = std::time::Instant::now();
        let tokens = gpu_tokenize(&html);
        let elapsed = start.elapsed();

        println!("Tokenized 1MB in {:?}", elapsed);
        assert!(elapsed.as_millis() < 10, "Should tokenize 1MB in <10ms");
    }
}
```

---

## Pass 2: Parser

### Purpose
Convert token stream into flat element array representing the DOM tree.

### Algorithm
Two-phase parallel parsing:
1. Count elements and allocate slots
2. Build parent-child relationships

### Pseudocode

```metal
// Element types (subset)
constant uint ELEM_ROOT = 0;
constant uint ELEM_DIV = 1;
constant uint ELEM_P = 2;
constant uint ELEM_SPAN = 3;
constant uint ELEM_H1 = 4;
constant uint ELEM_H2 = 5;
constant uint ELEM_H3 = 6;
constant uint ELEM_A = 7;
constant uint ELEM_IMG = 8;
constant uint ELEM_UL = 9;
constant uint ELEM_LI = 10;
constant uint ELEM_TEXT = 100;  // Pseudo-element for text nodes

kernel void parse_pass1_identify(
    device Token* tokens [[buffer(0)]],
    device uint* token_to_element [[buffer(1)]],  // Maps token index to element index
    device atomic_uint* element_count [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Each thread processes tokens it owns
    uint chunk = (token_count + THREAD_COUNT - 1) / THREAD_COUNT;
    uint start = tid * chunk;
    uint end = min(start + chunk, token_count);

    for (uint i = start; i < end; i++) {
        Token t = tokens[i];

        if (t.token_type == TOKEN_TAG_OPEN || t.token_type == TOKEN_TAG_SELF) {
            // Allocate element slot
            uint slot = atomic_fetch_add_explicit(element_count, 1, memory_order_relaxed);
            token_to_element[i] = slot;
        } else if (t.token_type == TOKEN_TEXT) {
            // Text nodes also become elements
            uint slot = atomic_fetch_add_explicit(element_count, 1, memory_order_relaxed);
            token_to_element[i] = slot;
        } else {
            token_to_element[i] = UINT_MAX;  // Not an element-creating token
        }
    }
}

kernel void parse_pass2_build_tree(
    device uint8_t* html [[buffer(0)]],
    device Token* tokens [[buffer(1)]],
    device uint* token_to_element [[buffer(2)]],
    device Element* elements [[buffer(3)]],
    device char* text_buffer [[buffer(4)]],
    device atomic_uint* text_offset [[buffer(5)]],
    constant uint& token_count [[buffer(6)]],
    threadgroup int* stack [[threadgroup(0)]],        // Parent stack (shared)
    threadgroup atomic_int* stack_ptr [[threadgroup(1)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Sequential tree building (one thread does this)
    // Could be parallelized with parallel prefix for nesting depth

    if (tid != 0) return;

    int current_parent = -1;  // Root
    atomic_store_explicit(stack_ptr, 0, memory_order_relaxed);

    for (uint i = 0; i < token_count; i++) {
        Token t = tokens[i];
        uint elem_idx = token_to_element[i];

        if (elem_idx == UINT_MAX) continue;

        if (t.token_type == TOKEN_TAG_OPEN) {
            // Create element
            elements[elem_idx].element_type = parse_tag_name(html, t.start, t.end);
            elements[elem_idx].parent = current_parent;
            elements[elem_idx].first_child = -1;
            elements[elem_idx].next_sibling = -1;

            // Link to parent's children
            if (current_parent >= 0) {
                link_child(elements, current_parent, elem_idx);
            }

            // Push onto stack
            int sp = atomic_fetch_add_explicit(stack_ptr, 1, memory_order_relaxed);
            stack[sp] = current_parent;
            current_parent = elem_idx;

        } else if (t.token_type == TOKEN_TAG_CLOSE) {
            // Pop stack
            int sp = atomic_fetch_sub_explicit(stack_ptr, 1, memory_order_relaxed) - 1;
            current_parent = stack[sp];

        } else if (t.token_type == TOKEN_TEXT) {
            // Create text node
            elements[elem_idx].element_type = ELEM_TEXT;
            elements[elem_idx].parent = current_parent;
            elements[elem_idx].first_child = -1;
            elements[elem_idx].next_sibling = -1;

            // Copy text content
            uint len = t.end - t.start;
            uint offset = atomic_fetch_add_explicit(text_offset, len, memory_order_relaxed);
            elements[elem_idx].text_start = offset;
            elements[elem_idx].text_length = len;

            for (uint j = 0; j < len; j++) {
                text_buffer[offset + j] = html[t.start + j];
            }

            // Link to parent
            if (current_parent >= 0) {
                link_child(elements, current_parent, elem_idx);
            }
        }
    }
}

// Helper: Parse tag name from "<tagname" to element type
uint parse_tag_name(device uint8_t* html, uint start, uint end) {
    // Skip '<' and any '/'
    uint i = start + 1;
    if (html[i] == '/') i++;

    // Extract tag name
    uint name_start = i;
    while (i < end && html[i] != ' ' && html[i] != '>' && html[i] != '/') {
        i++;
    }

    // Match against known tags
    uint len = i - name_start;

    if (len == 3 && html[name_start] == 'd' && html[name_start+1] == 'i' && html[name_start+2] == 'v') {
        return ELEM_DIV;
    }
    if (len == 1 && html[name_start] == 'p') {
        return ELEM_P;
    }
    if (len == 4 && html[name_start] == 's' && html[name_start+1] == 'p') {
        return ELEM_SPAN;
    }
    // ... more tag matching

    return ELEM_DIV;  // Default to div
}

// Helper: Link child to parent's child list
void link_child(device Element* elements, int parent, uint child) {
    if (elements[parent].first_child < 0) {
        elements[parent].first_child = child;
    } else {
        // Find last sibling
        int sibling = elements[parent].first_child;
        while (elements[sibling].next_sibling >= 0) {
            sibling = elements[sibling].next_sibling;
        }
        elements[sibling].next_sibling = child;
    }
}
```

### Tests

```rust
#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn test_simple_tree() {
        let html = b"<div><p>hello</p></div>";
        let (elements, text) = gpu_parse(html);

        assert_eq!(elements.len(), 3);  // div, p, text

        // Check tree structure
        assert_eq!(elements[0].element_type, ELEM_DIV);
        assert_eq!(elements[0].parent, -1);  // Root
        assert_eq!(elements[0].first_child, 1);  // p

        assert_eq!(elements[1].element_type, ELEM_P);
        assert_eq!(elements[1].parent, 0);  // div
        assert_eq!(elements[1].first_child, 2);  // text

        assert_eq!(elements[2].element_type, ELEM_TEXT);
        assert_eq!(elements[2].parent, 1);  // p
        assert_eq!(&text[elements[2].text_start..][..elements[2].text_length], b"hello");
    }

    #[test]
    fn test_siblings() {
        let html = b"<ul><li>one</li><li>two</li><li>three</li></ul>";
        let (elements, _) = gpu_parse(html);

        // ul has first_child pointing to first li
        let ul = &elements[0];
        assert_eq!(ul.element_type, ELEM_UL);

        // Follow sibling chain
        let mut li_count = 0;
        let mut current = ul.first_child;
        while current >= 0 {
            assert_eq!(elements[current as usize].element_type, ELEM_LI);
            li_count += 1;
            current = elements[current as usize].next_sibling;
        }
        assert_eq!(li_count, 3);
    }

    #[test]
    fn test_deep_nesting() {
        let html = b"<div><div><div><div><p>deep</p></div></div></div></div>";
        let (elements, _) = gpu_parse(html);

        // Verify parent chain
        let text_elem = elements.iter().find(|e| e.element_type == ELEM_TEXT).unwrap();
        let mut depth = 0;
        let mut current = text_elem.parent;
        while current >= 0 {
            depth += 1;
            current = elements[current as usize].parent;
        }
        assert_eq!(depth, 5);  // p -> div -> div -> div -> div
    }

    #[test]
    fn test_mixed_content() {
        let html = b"<p>Hello <b>bold</b> world</p>";
        let (elements, text) = gpu_parse(html);

        // Should have: p, text("Hello "), b, text("bold"), text(" world")
        let text_nodes: Vec<_> = elements.iter()
            .filter(|e| e.element_type == ELEM_TEXT)
            .collect();

        assert_eq!(text_nodes.len(), 3);
    }

    #[test]
    fn test_self_closing() {
        let html = b"<div><br/><img src='x'/></div>";
        let (elements, _) = gpu_parse(html);

        // br and img should have no children
        let br = elements.iter().find(|e| e.element_type == ELEM_BR);
        let img = elements.iter().find(|e| e.element_type == ELEM_IMG);

        assert!(br.map_or(true, |e| e.first_child < 0));
        assert!(img.map_or(true, |e| e.first_child < 0));
    }

    #[test]
    fn test_performance_1000_elements() {
        let html = generate_nested_html(1000);

        let start = std::time::Instant::now();
        let (elements, _) = gpu_parse(&html);
        let elapsed = start.elapsed();

        assert_eq!(elements.len(), 1000);
        println!("Parsed 1000 elements in {:?}", elapsed);
        assert!(elapsed.as_millis() < 5, "Should parse 1000 elements in <5ms");
    }
}
```

---

## Pass 3: Style Resolution

### Purpose
Match CSS selectors to elements and compute final styles.

### Algorithm
Parallel selector matching: each thread handles one element and checks all selectors.

### Pseudocode

```metal
// Selector types (simplified)
constant uint SEL_TAG = 1;        // div, p, etc.
constant uint SEL_CLASS = 2;     // .classname
constant uint SEL_ID = 3;        // #id
constant uint SEL_UNIVERSAL = 4; // *

struct Selector {
    uint type;
    uint hash;           // Hash of tag/class/id name for fast comparison
    uint specificity;    // For cascade ordering
    uint style_index;    // Index into style definitions
};

struct StyleDef {
    uint property;       // Which CSS property
    float values[4];     // Property values (up to 4 floats)
};

kernel void style_match(
    device Element* elements [[buffer(0)]],
    device uint8_t* html [[buffer(1)]],           // For class/id extraction
    device Selector* selectors [[buffer(2)]],
    device StyleDef* style_defs [[buffer(3)]],
    device ComputedStyle* computed [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    constant uint& selector_count [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid >= element_count) return;

    Element elem = elements[tid];

    // Initialize with defaults
    ComputedStyle style = default_style();

    // Track matched selectors for specificity ordering
    threadgroup MatchedSelector matches[64];  // Per-thread local
    uint match_count = 0;

    // Check all selectors against this element
    for (uint s = 0; s < selector_count; s++) {
        Selector sel = selectors[s];

        bool matched = false;

        switch (sel.type) {
            case SEL_TAG:
                matched = (hash_element_type(elem.element_type) == sel.hash);
                break;
            case SEL_CLASS:
                matched = element_has_class(elements, html, tid, sel.hash);
                break;
            case SEL_ID:
                matched = element_has_id(elements, html, tid, sel.hash);
                break;
            case SEL_UNIVERSAL:
                matched = true;
                break;
        }

        if (matched && match_count < 64) {
            matches[match_count].selector_index = s;
            matches[match_count].specificity = sel.specificity;
            match_count++;
        }
    }

    // Sort matches by specificity (simple bubble sort for small N)
    sort_by_specificity(matches, match_count);

    // Apply styles in specificity order
    for (uint m = 0; m < match_count; m++) {
        uint sel_idx = matches[m].selector_index;
        uint style_idx = selectors[sel_idx].style_index;
        apply_style_def(&style, style_defs[style_idx]);
    }

    // Inherit from parent where applicable
    if (elem.parent >= 0) {
        inherit_styles(&style, computed[elem.parent]);
    }

    computed[tid] = style;
}

// Default style values
ComputedStyle default_style() {
    ComputedStyle s;
    s.display = DISPLAY_BLOCK;
    s.width = 0;  // auto
    s.height = 0; // auto
    s.margin = {0, 0, 0, 0};
    s.padding = {0, 0, 0, 0};
    s.flex_direction = FLEX_ROW;
    s.justify_content = JUSTIFY_START;
    s.align_items = ALIGN_STRETCH;
    s.flex_grow = 0;
    s.flex_shrink = 1;
    s.background_color = {1, 1, 1, 0};  // Transparent
    s.color = {0, 0, 0, 1};             // Black
    s.font_size = 16;
    s.line_height = 1.2;
    s.border_width = {0, 0, 0, 0};
    s.border_color = {0, 0, 0, 1};
    s.border_radius = 0;
    return s;
}

// Apply a style definition to computed style
void apply_style_def(thread ComputedStyle* style, StyleDef def) {
    switch (def.property) {
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
            style->margin = {def.values[0], def.values[1], def.values[2], def.values[3]};
            break;
        case PROP_PADDING:
            style->padding = {def.values[0], def.values[1], def.values[2], def.values[3]};
            break;
        case PROP_BACKGROUND:
            style->background_color = {def.values[0], def.values[1], def.values[2], def.values[3]};
            break;
        case PROP_COLOR:
            style->color = {def.values[0], def.values[1], def.values[2], def.values[3]};
            break;
        case PROP_FONT_SIZE:
            style->font_size = def.values[0];
            break;
        case PROP_FLEX_DIRECTION:
            style->flex_direction = uint(def.values[0]);
            break;
        // ... more properties
    }
}

// Inherit inheritable properties from parent
void inherit_styles(thread ComputedStyle* style, ComputedStyle parent) {
    // These properties inherit by default
    style->color = parent.color;
    style->font_size = parent.font_size;
    style->line_height = parent.line_height;
}
```

### Tests

```rust
#[cfg(test)]
mod style_tests {
    use super::*;

    #[test]
    fn test_tag_selector() {
        let html = b"<div><p>text</p></div>";
        let css = "p { color: red; }";

        let (elements, styles) = gpu_style(html, css);

        let p_style = &styles[1];  // p element
        assert_eq!(p_style.color, [1.0, 0.0, 0.0, 1.0]);  // Red
    }

    #[test]
    fn test_class_selector() {
        let html = b"<div class=\"highlight\">text</div>";
        let css = ".highlight { background: yellow; }";

        let (elements, styles) = gpu_style(html, css);

        let div_style = &styles[0];
        assert_eq!(div_style.background_color, [1.0, 1.0, 0.0, 1.0]);  // Yellow
    }

    #[test]
    fn test_specificity() {
        let html = b"<p class=\"special\">text</p>";
        let css = "p { color: blue; } .special { color: red; }";

        let (_, styles) = gpu_style(html, css);

        // .special has higher specificity than p
        assert_eq!(styles[0].color, [1.0, 0.0, 0.0, 1.0]);  // Red
    }

    #[test]
    fn test_inheritance() {
        let html = b"<div style=\"color: green\"><p>text</p></div>";

        let (_, styles) = gpu_style(html, "");

        // p should inherit color from div
        assert_eq!(styles[1].color, styles[0].color);
    }

    #[test]
    fn test_cascade_order() {
        let html = b"<p>text</p>";
        let css = "p { margin: 10px; } p { margin: 20px; }";  // Later wins

        let (_, styles) = gpu_style(html, css);

        assert_eq!(styles[0].margin, [20.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn test_performance_1000_elements_100_selectors() {
        let html = generate_html_with_classes(1000);
        let css = generate_css_selectors(100);

        let start = std::time::Instant::now();
        let (_, styles) = gpu_style(&html, &css);
        let elapsed = start.elapsed();

        println!("Styled 1000 elements with 100 selectors in {:?}", elapsed);
        // 1000 * 100 = 100K selector checks
        assert!(elapsed.as_millis() < 5, "Should complete in <5ms");
    }
}
```

---

## Pass 4: Layout

### Purpose
Compute positions and sizes for all elements.

### Algorithm
Multi-pass parallel layout:
1. Intrinsic sizes (parallel)
2. Tree-based constraint propagation (parallel prefix)
3. Final positioning (parallel)

### Pseudocode

```metal
kernel void layout_pass1_intrinsic(
    device Element* elements [[buffer(0)]],
    device ComputedStyle* styles [[buffer(1)]],
    device char* text_buffer [[buffer(2)]],
    device LayoutBox* boxes [[buffer(3)]],
    device FontMetrics* font [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    constant float2& viewport [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid >= element_count) return;

    Element elem = elements[tid];
    ComputedStyle style = styles[tid];

    LayoutBox box;
    box.x = 0;
    box.y = 0;
    box.scroll_x = 0;
    box.scroll_y = 0;

    // Compute intrinsic size based on content
    if (elem.element_type == ELEM_TEXT) {
        // Measure text
        float text_width = 0;
        float text_height = style.font_size * style.line_height;

        for (uint i = 0; i < elem.text_length; i++) {
            char c = text_buffer[elem.text_start + i];
            text_width += font->char_widths[c] * style.font_size;
        }

        box.content_width = text_width;
        box.content_height = text_height;
    } else {
        // Non-text elements: use specified size or 0 (will be computed from children)
        box.content_width = style.width;   // 0 = auto
        box.content_height = style.height; // 0 = auto
    }

    // Add padding to get box size
    box.width = box.content_width + style.padding[1] + style.padding[3];
    box.height = box.content_height + style.padding[0] + style.padding[2];

    boxes[tid] = box;
}

kernel void layout_pass2_flex(
    device Element* elements [[buffer(0)]],
    device ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant float2& viewport [[buffer(4)]],
    threadgroup float* shared_sizes [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid >= element_count) return;

    Element elem = elements[tid];
    ComputedStyle style = styles[tid];

    // Only process flex containers
    if (style.display != DISPLAY_FLEX) return;

    // Gather children sizes
    float total_size = 0;
    float total_grow = 0;
    int child = elem.first_child;

    while (child >= 0) {
        ComputedStyle child_style = styles[child];
        LayoutBox child_box = boxes[child];

        if (style.flex_direction == FLEX_ROW) {
            total_size += child_box.width + child_style.margin[1] + child_style.margin[3];
        } else {
            total_size += child_box.height + child_style.margin[0] + child_style.margin[2];
        }
        total_grow += child_style.flex_grow;

        child = elements[child].next_sibling;
    }

    // Determine container size
    float container_size;
    if (style.flex_direction == FLEX_ROW) {
        container_size = (style.width > 0) ? style.width : viewport.x;
    } else {
        container_size = (style.height > 0) ? style.height : viewport.y;
    }

    // Distribute remaining space
    float remaining = container_size - total_size;
    float grow_unit = (total_grow > 0) ? remaining / total_grow : 0;

    // Position children
    float offset = 0;

    // Handle justify-content
    if (style.justify_content == JUSTIFY_CENTER) {
        offset = remaining / 2;
    } else if (style.justify_content == JUSTIFY_END) {
        offset = remaining;
    } else if (style.justify_content == JUSTIFY_SPACE_BETWEEN && total_grow == 0) {
        // Will add spacing between items
    }

    child = elem.first_child;
    while (child >= 0) {
        ComputedStyle child_style = styles[child];
        LayoutBox* child_box = &boxes[child];

        // Add margin
        if (style.flex_direction == FLEX_ROW) {
            offset += child_style.margin[3];  // left margin
            child_box->x = boxes[tid].content_x + offset;
            child_box->y = boxes[tid].content_y + child_style.margin[0];

            // Apply flex-grow
            if (child_style.flex_grow > 0) {
                child_box->width += grow_unit * child_style.flex_grow;
            }

            offset += child_box->width + child_style.margin[1];  // right margin
        } else {
            offset += child_style.margin[0];  // top margin
            child_box->x = boxes[tid].content_x + child_style.margin[3];
            child_box->y = boxes[tid].content_y + offset;

            if (child_style.flex_grow > 0) {
                child_box->height += grow_unit * child_style.flex_grow;
            }

            offset += child_box->height + child_style.margin[2];  // bottom margin
        }

        // Handle align-items
        if (style.flex_direction == FLEX_ROW) {
            float cross_space = boxes[tid].content_height - child_box->height;
            switch (style.align_items) {
                case ALIGN_CENTER:
                    child_box->y += cross_space / 2;
                    break;
                case ALIGN_END:
                    child_box->y += cross_space;
                    break;
                case ALIGN_STRETCH:
                    child_box->height = boxes[tid].content_height;
                    break;
            }
        }

        // Update content area
        child_box->content_x = child_box->x + child_style.padding[3];
        child_box->content_y = child_box->y + child_style.padding[0];
        child_box->content_width = child_box->width - child_style.padding[1] - child_style.padding[3];
        child_box->content_height = child_box->height - child_style.padding[0] - child_style.padding[2];

        child = elements[child].next_sibling;
    }
}

kernel void layout_pass3_block(
    device Element* elements [[buffer(0)]],
    device ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    constant float2& viewport [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid >= element_count) return;

    Element elem = elements[tid];
    ComputedStyle style = styles[tid];

    // Skip flex items (handled in pass2)
    if (elem.parent >= 0 && styles[elem.parent].display == DISPLAY_FLEX) return;

    // Block layout: stack children vertically
    if (style.display == DISPLAY_BLOCK) {
        float y_offset = 0;
        int child = elem.first_child;

        while (child >= 0) {
            ComputedStyle child_style = styles[child];
            LayoutBox* child_box = &boxes[child];

            child_box->x = boxes[tid].content_x + child_style.margin[3];
            child_box->y = boxes[tid].content_y + y_offset + child_style.margin[0];

            // Block elements take full width by default
            if (child_style.width == 0) {
                child_box->width = boxes[tid].content_width - child_style.margin[1] - child_style.margin[3];
            }

            y_offset += child_box->height + child_style.margin[0] + child_style.margin[2];

            child = elements[child].next_sibling;
        }

        // Update container height if auto
        if (style.height == 0) {
            boxes[tid].height = y_offset + style.padding[0] + style.padding[2];
            boxes[tid].content_height = y_offset;
        }
    }
}
```

### Tests

```rust
#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn test_simple_block() {
        let html = b"<div style=\"width:100px\"><p>text</p></div>";
        let boxes = gpu_layout(html, (800.0, 600.0));

        assert_eq!(boxes[0].width, 100.0);
        assert_eq!(boxes[1].width, 100.0);  // p takes parent width
    }

    #[test]
    fn test_flexbox_row() {
        let html = b"<div style=\"display:flex;width:300px\">
            <div style=\"width:100px\">a</div>
            <div style=\"width:100px\">b</div>
            <div style=\"width:100px\">c</div>
        </div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Children should be laid out horizontally
        assert_eq!(boxes[1].x, 0.0);
        assert_eq!(boxes[2].x, 100.0);
        assert_eq!(boxes[3].x, 200.0);
    }

    #[test]
    fn test_flex_grow() {
        let html = b"<div style=\"display:flex;width:400px\">
            <div style=\"width:100px\">fixed</div>
            <div style=\"flex-grow:1\">grow</div>
        </div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Growing element should take remaining space
        assert_eq!(boxes[1].width, 100.0);
        assert_eq!(boxes[2].width, 300.0);
    }

    #[test]
    fn test_justify_content_center() {
        let html = b"<div style=\"display:flex;width:400px;justify-content:center\">
            <div style=\"width:100px\">a</div>
        </div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Child should be centered (150px from each side)
        assert_eq!(boxes[1].x, 150.0);
    }

    #[test]
    fn test_align_items_center() {
        let html = b"<div style=\"display:flex;height:200px;align-items:center\">
            <div style=\"height:50px\">a</div>
        </div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Child should be vertically centered
        assert_eq!(boxes[1].y, 75.0);
    }

    #[test]
    fn test_nested_flex() {
        let html = b"<div style=\"display:flex;width:600px\">
            <div style=\"display:flex;flex-direction:column;width:300px\">
                <div style=\"height:50px\">a</div>
                <div style=\"height:50px\">b</div>
            </div>
            <div style=\"width:300px\">c</div>
        </div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Verify nested layout
        assert_eq!(boxes[2].y, 0.0);   // First child of column
        assert_eq!(boxes[3].y, 50.0);  // Second child of column
        assert_eq!(boxes[4].x, 300.0); // Second child of row
    }

    #[test]
    fn test_margin_collapse() {
        // Note: GPU layout may simplify margin collapsing rules
        let html = b"<div><p style=\"margin:20px\">a</p><p style=\"margin:20px\">b</p></div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Adjacent margins - behavior depends on implementation
        // Simple version: no collapse, p2.y = p1.y + p1.height + 20 + 20
    }

    #[test]
    fn test_text_wrapping() {
        let html = b"<div style=\"width:100px\"><p>This is a long text that should wrap</p></div>";

        let boxes = gpu_layout(html, (800.0, 600.0));

        // Text should wrap, increasing height
        assert!(boxes[1].height > 16.0);  // More than one line
    }

    #[test]
    fn test_performance_1000_elements() {
        let html = generate_complex_layout(1000);

        let start = std::time::Instant::now();
        let boxes = gpu_layout(&html, (800.0, 600.0));
        let elapsed = start.elapsed();

        println!("Layout 1000 elements in {:?}", elapsed);
        assert!(elapsed.as_millis() < 10, "Should layout in <10ms");
    }
}
```

---

## Pass 5: Paint

### Purpose
Generate vertex geometry for rendering.

### Algorithm
Each thread generates vertices for one element: background, border, and text quads.

### Pseudocode

```metal
kernel void paint(
    device Element* elements [[buffer(0)]],
    device ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device char* text_buffer [[buffer(3)]],
    device Vertex* vertices [[buffer(4)]],
    device atomic_uint* vertex_count [[buffer(5)]],
    device FontAtlas* font [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    constant float2& viewport [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid >= element_count) return;

    Element elem = elements[tid];
    ComputedStyle style = styles[tid];
    LayoutBox box = boxes[tid];

    // Skip invisible elements
    if (style.display == DISPLAY_NONE) return;
    if (box.width <= 0 || box.height <= 0) return;

    // Convert to normalized device coordinates
    float2 scale = float2(2.0 / viewport.x, -2.0 / viewport.y);
    float2 offset = float2(-1.0, 1.0);

    // === Background ===
    if (style.background_color.a > 0) {
        uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);

        float4 color = style.background_color;
        float x0 = box.x * scale.x + offset.x;
        float y0 = box.y * scale.y + offset.y;
        float x1 = (box.x + box.width) * scale.x + offset.x;
        float y1 = (box.y + box.height) * scale.y + offset.y;

        // Two triangles for quad
        vertices[base + 0] = {{x0, y0}, {0, 0}, color, tid, VERTEX_BACKGROUND};
        vertices[base + 1] = {{x1, y0}, {1, 0}, color, tid, VERTEX_BACKGROUND};
        vertices[base + 2] = {{x1, y1}, {1, 1}, color, tid, VERTEX_BACKGROUND};
        vertices[base + 3] = {{x0, y0}, {0, 0}, color, tid, VERTEX_BACKGROUND};
        vertices[base + 4] = {{x1, y1}, {1, 1}, color, tid, VERTEX_BACKGROUND};
        vertices[base + 5] = {{x0, y1}, {0, 1}, color, tid, VERTEX_BACKGROUND};
    }

    // === Border ===
    if (style.border_width[0] > 0 || style.border_width[1] > 0 ||
        style.border_width[2] > 0 || style.border_width[3] > 0) {

        float4 bc = style.border_color;

        // Top border
        if (style.border_width[0] > 0) {
            uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);
            emit_rect(vertices, base, box.x, box.y, box.width, style.border_width[0], bc, tid, scale, offset);
        }

        // Right border
        if (style.border_width[1] > 0) {
            uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);
            float bx = box.x + box.width - style.border_width[1];
            emit_rect(vertices, base, bx, box.y, style.border_width[1], box.height, bc, tid, scale, offset);
        }

        // Bottom border
        if (style.border_width[2] > 0) {
            uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);
            float by = box.y + box.height - style.border_width[2];
            emit_rect(vertices, base, box.x, by, box.width, style.border_width[2], bc, tid, scale, offset);
        }

        // Left border
        if (style.border_width[3] > 0) {
            uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);
            emit_rect(vertices, base, box.x, box.y, style.border_width[3], box.height, bc, tid, scale, offset);
        }
    }

    // === Text ===
    if (elem.element_type == ELEM_TEXT && elem.text_length > 0) {
        float x = box.content_x;
        float y = box.content_y;
        float4 color = style.color;

        for (uint i = 0; i < elem.text_length; i++) {
            char c = text_buffer[elem.text_start + i];
            GlyphInfo glyph = font->glyphs[c];

            uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);

            float gx = x + glyph.bearing_x * style.font_size;
            float gy = y + glyph.bearing_y * style.font_size;
            float gw = glyph.width * style.font_size;
            float gh = glyph.height * style.font_size;

            // Emit textured quad
            float nx0 = gx * scale.x + offset.x;
            float ny0 = gy * scale.y + offset.y;
            float nx1 = (gx + gw) * scale.x + offset.x;
            float ny1 = (gy + gh) * scale.y + offset.y;

            vertices[base + 0] = {{nx0, ny0}, {glyph.u0, glyph.v0}, color, tid, VERTEX_TEXT};
            vertices[base + 1] = {{nx1, ny0}, {glyph.u1, glyph.v0}, color, tid, VERTEX_TEXT};
            vertices[base + 2] = {{nx1, ny1}, {glyph.u1, glyph.v1}, color, tid, VERTEX_TEXT};
            vertices[base + 3] = {{nx0, ny0}, {glyph.u0, glyph.v0}, color, tid, VERTEX_TEXT};
            vertices[base + 4] = {{nx1, ny1}, {glyph.u1, glyph.v1}, color, tid, VERTEX_TEXT};
            vertices[base + 5] = {{nx0, ny1}, {glyph.u0, glyph.v1}, color, tid, VERTEX_TEXT};

            x += glyph.advance * style.font_size;
        }
    }
}

void emit_rect(device Vertex* vertices, uint base,
               float x, float y, float w, float h, float4 color, uint elem_id,
               float2 scale, float2 offset) {
    float x0 = x * scale.x + offset.x;
    float y0 = y * scale.y + offset.y;
    float x1 = (x + w) * scale.x + offset.x;
    float y1 = (y + h) * scale.y + offset.y;

    vertices[base + 0] = {{x0, y0}, {0, 0}, color, elem_id, VERTEX_BACKGROUND};
    vertices[base + 1] = {{x1, y0}, {1, 0}, color, elem_id, VERTEX_BACKGROUND};
    vertices[base + 2] = {{x1, y1}, {1, 1}, color, elem_id, VERTEX_BACKGROUND};
    vertices[base + 3] = {{x0, y0}, {0, 0}, color, elem_id, VERTEX_BACKGROUND};
    vertices[base + 4] = {{x1, y1}, {1, 1}, color, elem_id, VERTEX_BACKGROUND};
    vertices[base + 5] = {{x0, y1}, {0, 1}, color, elem_id, VERTEX_BACKGROUND};
}
```

### Tests

```rust
#[cfg(test)]
mod paint_tests {
    use super::*;

    #[test]
    fn test_background_vertices() {
        let html = b"<div style=\"width:100px;height:50px;background:red\"></div>";
        let vertices = gpu_paint(html, (800.0, 600.0));

        // Should have 6 vertices for one quad
        assert_eq!(vertices.len(), 6);

        // Check color
        for v in &vertices {
            assert_eq!(v.color, [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn test_border_vertices() {
        let html = b"<div style=\"width:100px;height:50px;border:2px solid black\"></div>";
        let vertices = gpu_paint(html, (800.0, 600.0));

        // 4 borders × 6 vertices = 24 vertices
        assert_eq!(vertices.len(), 24);
    }

    #[test]
    fn test_text_vertices() {
        let html = b"<p>Hello</p>";
        let vertices = gpu_paint(html, (800.0, 600.0));

        // 5 characters × 6 vertices = 30 vertices
        let text_verts: Vec<_> = vertices.iter()
            .filter(|v| v.vertex_type == VERTEX_TEXT)
            .collect();
        assert_eq!(text_verts.len(), 30);
    }

    #[test]
    fn test_ndc_coordinates() {
        let html = b"<div style=\"width:800px;height:600px;background:blue\"></div>";
        let vertices = gpu_paint(html, (800.0, 600.0));

        // Full-screen quad should span [-1, 1]
        let min_x = vertices.iter().map(|v| v.position[0]).fold(f32::INFINITY, f32::min);
        let max_x = vertices.iter().map(|v| v.position[0]).fold(f32::NEG_INFINITY, f32::max);

        assert!((min_x - (-1.0)).abs() < 0.01);
        assert!((max_x - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_element_id_for_hit_testing() {
        let html = b"<div id=\"a\"><div id=\"b\">text</div></div>";
        let vertices = gpu_paint(html, (800.0, 600.0));

        // Each vertex should have element_id set
        for v in &vertices {
            assert!(v.element_id < 10);  // Reasonable element count
        }
    }

    #[test]
    fn test_display_none_no_vertices() {
        let html = b"<div style=\"display:none;background:red\">hidden</div>";
        let vertices = gpu_paint(html, (800.0, 600.0));

        // No vertices should be generated
        assert_eq!(vertices.len(), 0);
    }

    #[test]
    fn test_performance_1000_elements() {
        let html = generate_styled_html(1000);

        let start = std::time::Instant::now();
        let vertices = gpu_paint(&html, (800.0, 600.0));
        let elapsed = start.elapsed();

        println!("Paint 1000 elements -> {} vertices in {:?}", vertices.len(), elapsed);
        assert!(elapsed.as_millis() < 5, "Should paint in <5ms");
    }
}
```

---

## Render Pass

### Purpose
Rasterize vertices to framebuffer.

### Algorithm
Standard Metal render pipeline with vertex/fragment shaders.

### Shaders

```metal
struct VertexIn {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
    float4 color [[attribute(2)]];
    uint element_id [[attribute(3)]];
    uint vertex_type [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
    uint element_id;
    uint vertex_type;
};

vertex VertexOut vertex_main(
    VertexIn in [[stage_in]]
) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.element_id = in.element_id;
    out.vertex_type = in.vertex_type;
    return out;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> font_atlas [[texture(0)]],
    sampler font_sampler [[sampler(0)]]
) {
    if (in.vertex_type == VERTEX_TEXT) {
        // Sample font atlas
        float alpha = font_atlas.sample(font_sampler, in.uv).r;
        return float4(in.color.rgb, in.color.a * alpha);
    } else {
        // Solid color
        return in.color;
    }
}
```

---

## Integration: GpuDocumentViewer

### Rust Implementation

```rust
pub struct GpuDocumentViewer {
    device: Device,
    command_queue: CommandQueue,

    // Pipelines
    tokenize_pipeline: ComputePipelineState,
    parse_pipeline: ComputePipelineState,
    style_pipeline: ComputePipelineState,
    layout_pipeline: ComputePipelineState,
    paint_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    raw_buffer: Buffer,           // Input HTML
    token_buffer: Buffer,         // Tokenizer output
    element_buffer: Buffer,       // Parser output
    text_buffer: Buffer,          // Text content
    style_buffer: Buffer,         // Computed styles
    layout_buffer: Buffer,        // Layout boxes
    vertex_buffer: Buffer,        // Paint output

    // Font
    font_atlas: Texture,
    font_metrics: Buffer,

    // State
    document_loaded: bool,
    viewport: (f32, f32),
}

impl GpuDocumentViewer {
    pub fn new(device: Device) -> Self {
        // Compile shaders, create pipelines, allocate buffers
        // ...
    }

    pub fn load_document(&mut self, html: &[u8]) {
        // Copy to GPU buffer
        let raw_ptr = self.raw_buffer.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(html.as_ptr(), raw_ptr, html.len());
        }

        // Run pipeline
        self.run_pipeline();
        self.document_loaded = true;
    }

    fn run_pipeline(&mut self) {
        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1: Tokenize
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.tokenize_pipeline);
            encoder.set_buffer(0, Some(&self.raw_buffer), 0);
            encoder.set_buffer(1, Some(&self.token_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
            encoder.end_encoding();
        }

        // Pass 2: Parse
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.parse_pipeline);
            encoder.set_buffer(0, Some(&self.token_buffer), 0);
            encoder.set_buffer(1, Some(&self.element_buffer), 0);
            encoder.set_buffer(2, Some(&self.text_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
            encoder.end_encoding();
        }

        // Pass 3: Style
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.style_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
            encoder.end_encoding();
        }

        // Pass 4: Layout
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.layout_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
            encoder.end_encoding();
        }

        // Pass 5: Paint
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.paint_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.text_buffer), 0);
            encoder.set_buffer(4, Some(&self.vertex_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(1024, 1, 1));
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    pub fn render(&self, drawable: &MetalDrawableRef) {
        let command_buffer = self.command_queue.new_command_buffer();

        let render_desc = RenderPassDescriptor::new();
        let color = render_desc.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(drawable.texture()));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(1.0, 1.0, 1.0, 1.0));

        let encoder = command_buffer.new_render_command_encoder(&render_desc);
        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);
        encoder.set_fragment_texture(0, Some(&self.font_atlas));

        // Get actual vertex count from atomic counter
        let vertex_count = self.get_vertex_count();
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count);
        encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}

impl GpuApp for GpuDocumentViewer {
    fn name(&self) -> &str { "Document Viewer" }
    fn compute_pipeline(&self) -> &ComputePipelineState { &self.paint_pipeline }
    fn render_pipeline(&self) -> &RenderPipelineState { &self.render_pipeline }
    fn vertices_buffer(&self) -> &Buffer { &self.vertex_buffer }
    fn vertex_count(&self) -> usize { self.get_vertex_count() }
    // ... etc
}
```

---

## Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Tokenize 1MB | <10ms | `benchmark_tokenize` |
| Parse 10K elements | <5ms | `benchmark_parse` |
| Style 10K elements | <5ms | `benchmark_style` |
| Layout 10K elements | <10ms | `benchmark_layout` |
| Paint 10K elements | <5ms | `benchmark_paint` |
| **Total pipeline** | **<35ms** | `benchmark_full` |
| Memory footprint | <50MB | Instruments |
| GPU utilization | >80% | Metal System Trace |

---

## Test Plan

### Unit Tests
- [ ] Tokenizer handles all HTML5 tag types
- [ ] Parser builds correct tree structure
- [ ] Style matching respects specificity
- [ ] Layout computes correct positions
- [ ] Paint generates valid vertices

### Integration Tests
- [ ] End-to-end: HTML → pixels
- [ ] Resize handling
- [ ] Scroll handling
- [ ] Font rendering quality

### Performance Tests
- [ ] 1KB document in <1ms
- [ ] 100KB document in <10ms
- [ ] 1MB document in <100ms
- [ ] 60 FPS during scroll

### Comparison Tests
- [ ] Visual diff against Chrome rendering
- [ ] Layout matches CSS spec test suite (subset)

---

## Milestones

### M1: Tokenizer + Parser (Week 1-2)
- [ ] Implement tokenizer compute shader
- [ ] Implement parser compute shader
- [ ] Unit tests passing
- [ ] Benchmark: 10M tokens/sec

### M2: Style + Layout (Week 3-4)
- [ ] Implement style matching
- [ ] Implement flexbox layout
- [ ] Implement block layout
- [ ] Unit tests passing

### M3: Paint + Render (Week 5-6)
- [ ] Implement vertex generation
- [ ] Implement font atlas
- [ ] Implement render pipeline
- [ ] Visual output working

### M4: Integration + Polish (Week 7-8)
- [ ] GpuApp integration
- [ ] Scroll support
- [ ] Window resize
- [ ] Performance optimization
- [ ] Documentation

---

## Open Questions

1. **Text wrapping**: Implement in layout pass or defer to paint?
2. **Incremental updates**: Full re-render or dirty tracking?
3. **CSS parsing**: On GPU or pre-process on CPU?
4. **Image support**: Decode on GPU or load pre-decoded?
5. **Accessibility**: How to expose structure to screen readers?

---

## References

- [CSS Flexible Box Layout](https://www.w3.org/TR/css-flexbox-1/)
- [HTML5 Tokenization](https://html.spec.whatwg.org/multipage/parsing.html#tokenization)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Parallel HTML Parsing Research](https://research.google/pubs/pub41345/)
