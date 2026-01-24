# Issue #100: GPU Inline Layout

## Summary
Implement inline formatting context layout: break text into words, measure word widths, pack words into line boxes, and position inline elements.

## Problem
Text currently rendered as single block with approximate wrapping. Need proper line-box-based layout for correct text flow and inline element positioning.

## CSS 2.1 Inline Formatting Model

### Line Boxes

In an inline formatting context, inline-level boxes are laid out horizontally in "line boxes":

```
┌──────────────────────────────────────────────────────────────┐
│ Line Box 1                                                   │
│ ┌─────┐ ┌─────┐ ┌─────────────┐ ┌─────┐                     │
│ │word │ │word │ │ inline-box  │ │word │                     │
│ └─────┘ └─────┘ └─────────────┘ └─────┘                     │
│                                        baseline ────────────│
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ Line Box 2                                                   │
│ ┌──────────────┐ ┌─────┐ ┌─────┐                            │
│ │ longer-word  │ │word │ │word │                            │
│ └──────────────┘ └─────┘ └─────┘                            │
└──────────────────────────────────────────────────────────────┘
```

### Line Box Properties

- Width: exactly fits content (from leftmost to rightmost inline box)
- Height: from topmost box top to bottommost box bottom
- Baseline: determined by strut and tallest inline element

### Word Breaking

1. Soft wrap opportunities at:
   - Spaces (U+0020)
   - After hyphens
   - Before/after CJK characters
   - As specified by `word-break` property

2. `white-space` property controls:
   - `normal`: collapse whitespace, wrap at soft wrap opportunities
   - `nowrap`: collapse whitespace, no wrapping
   - `pre`: preserve whitespace, no wrapping
   - `pre-wrap`: preserve whitespace, wrap
   - `pre-line`: collapse whitespace, preserve newlines, wrap

## Data Structures

```rust
#[repr(C)]
pub struct Word {
    pub element_index: u32,       // Source text element
    pub text_start: u32,          // Start in text buffer
    pub text_length: u32,         // Length in bytes
    pub width: f32,               // Measured width in pixels
    pub is_whitespace: u8,        // Is this a whitespace word?
    pub is_newline: u8,           // Is this a newline?
    pub _padding: [u8; 2],
}

#[repr(C)]
pub struct LineBox {
    pub y: f32,                   // Y position in containing block
    pub height: f32,              // Line height
    pub baseline: f32,            // Baseline Y offset from line top
    pub width: f32,               // Actual content width
    pub first_item: u32,          // First word/inline-box index
    pub item_count: u32,          // Number of items
    pub left_offset: f32,         // For text-align
    pub _padding: f32,
}

#[repr(C)]
pub struct InlinePosition {
    pub x: f32,                   // X position in line box
    pub y: f32,                   // Y position (absolute)
    pub width: f32,
    pub height: f32,
    pub line_index: u32,          // Which line box
    pub _padding: [u32; 3],
}

#[repr(C)]
pub struct GlyphPosition {
    pub x: f32,                   // Absolute X
    pub y: f32,                   // Absolute Y (baseline)
    pub char_code: u32,           // Unicode codepoint
    pub font_size: f32,           // For rendering
    pub color: u32,               // RGBA packed
    pub _padding: [u32; 3],
}
```

## GPU Kernels

```metal
// =============================================================================
// PASS 1: Break text into words (parallel per-character)
// =============================================================================

// First, mark word boundaries
kernel void mark_word_boundaries(
    device const uint8_t* text [[buffer(0)]],
    device const Element* elements [[buffer(1)]],
    device const ComputedStyleCompact* styles [[buffer(2)]],
    device uint8_t* is_word_start [[buffer(3)]],
    device uint8_t* word_types [[buffer(4)]],  // 0=text, 1=whitespace, 2=newline
    constant uint& text_length [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= text_length) return;

    uint8_t c = text[gid];
    uint8_t prev_c = gid > 0 ? text[gid - 1] : ' ';

    bool is_whitespace = (c == ' ' || c == '\t');
    bool is_newline = (c == '\n');
    bool prev_whitespace = (prev_c == ' ' || prev_c == '\t' || prev_c == '\n');

    // Word starts when transitioning from whitespace to non-whitespace
    // or at text start
    bool is_start = false;
    uint8_t word_type = 0;

    if (is_newline) {
        is_start = true;
        word_type = 2;  // Newline
    } else if (is_whitespace && !prev_whitespace) {
        is_start = true;
        word_type = 1;  // Whitespace
    } else if (!is_whitespace && (prev_whitespace || gid == 0)) {
        is_start = true;
        word_type = 0;  // Text
    }

    is_word_start[gid] = is_start ? 1 : 0;
    word_types[gid] = word_type;
}

// Then prefix sum and build word array
kernel void build_words(
    device const uint8_t* text [[buffer(0)]],
    device const uint8_t* is_word_start [[buffer(1)]],
    device const uint8_t* word_types [[buffer(2)]],
    device const uint* word_indices [[buffer(3)]],  // From prefix sum
    device Word* words [[buffer(4)]],
    device const ComputedStyleCompact* styles [[buffer(5)]],
    device const Element* elements [[buffer(6)]],
    constant uint& text_length [[buffer(7)]],
    constant uint& ifc_element [[buffer(8)]],  // IFC root element
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= text_length) return;
    if (is_word_start[gid] == 0) return;

    uint word_idx = word_indices[gid];

    // Find word end
    uint end = gid + 1;
    while (end < text_length && is_word_start[end] == 0) {
        end++;
    }

    // Get font size from IFC root or default
    float font_size = 16.0;
    if (ifc_element < 0xFFFFFFFF) {
        font_size = styles[ifc_element].font_size;
        if (font_size <= 0) font_size = 16.0;
    }

    // Measure word width
    float width = 0;
    for (uint i = gid; i < end; i++) {
        uint8_t c = text[i];
        if (c == ' ') width += font_size * 0.3;
        else if (c == '\t') width += font_size * 1.2;
        else if (c == '\n') width += 0;
        else width += font_size * 0.6;
    }

    Word word;
    word.element_index = ifc_element;
    word.text_start = gid;
    word.text_length = end - gid;
    word.width = width;
    word.is_whitespace = (word_types[gid] == 1) ? 1 : 0;
    word.is_newline = (word_types[gid] == 2) ? 1 : 0;

    words[word_idx] = word;
}

// =============================================================================
// PASS 2: Pack words into line boxes (sequential per IFC)
// =============================================================================
kernel void pack_lines(
    device const Word* words [[buffer(0)]],
    device LineBox* lines [[buffer(1)]],
    device uint* word_to_line [[buffer(2)]],
    device atomic_uint* line_count [[buffer(3)]],
    constant uint& word_count [[buffer(4)]],
    constant float& container_width [[buffer(5)]],
    constant float& line_height [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // Single thread packs lines (sequential algorithm)
    if (gid != 0) return;

    float current_x = 0;
    float current_y = 0;
    uint current_line = 0;
    uint line_first_word = 0;
    uint line_word_count = 0;

    for (uint i = 0; i < word_count; i++) {
        Word word = words[i];

        // Newline forces new line
        if (word.is_newline) {
            // Finish current line
            if (line_word_count > 0) {
                LineBox line;
                line.y = current_y;
                line.height = line_height;
                line.baseline = line_height * 0.8;  // Approximate
                line.width = current_x;
                line.first_item = line_first_word;
                line.item_count = line_word_count;
                line.left_offset = 0;
                lines[current_line] = line;
                current_line++;
            }

            // Start new line
            current_x = 0;
            current_y += line_height;
            line_first_word = i + 1;
            line_word_count = 0;
            word_to_line[i] = current_line;
            continue;
        }

        // Check if word fits on current line
        bool fits = (current_x + word.width <= container_width) || (line_word_count == 0);

        if (!fits) {
            // Finish current line (exclude trailing whitespace)
            LineBox line;
            line.y = current_y;
            line.height = line_height;
            line.baseline = line_height * 0.8;
            line.width = current_x;
            line.first_item = line_first_word;
            line.item_count = line_word_count;
            line.left_offset = 0;
            lines[current_line] = line;
            current_line++;

            // Start new line
            current_x = 0;
            current_y += line_height;
            line_first_word = i;
            line_word_count = 0;
        }

        // Skip leading whitespace on new line
        if (line_word_count == 0 && word.is_whitespace) {
            word_to_line[i] = current_line;
            continue;
        }

        // Add word to current line
        word_to_line[i] = current_line;
        current_x += word.width;
        line_word_count++;
    }

    // Finish last line
    if (line_word_count > 0) {
        LineBox line;
        line.y = current_y;
        line.height = line_height;
        line.baseline = line_height * 0.8;
        line.width = current_x;
        line.first_item = line_first_word;
        line.item_count = line_word_count;
        line.left_offset = 0;
        lines[current_line] = line;
        current_line++;
    }

    atomic_store_explicit(line_count, current_line, memory_order_relaxed);
}

// =============================================================================
// PASS 3: Apply text-align (parallel per line)
// =============================================================================
kernel void apply_text_align(
    device LineBox* lines [[buffer(0)]],
    constant uint& line_count [[buffer(1)]],
    constant float& container_width [[buffer(2)]],
    constant uint& text_align [[buffer(3)]],  // LEFT, CENTER, RIGHT, JUSTIFY
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= line_count) return;

    LineBox line = lines[gid];
    float remaining = container_width - line.width;

    switch (text_align) {
        case TEXT_ALIGN_CENTER:
            line.left_offset = remaining / 2;
            break;
        case TEXT_ALIGN_RIGHT:
            line.left_offset = remaining;
            break;
        case TEXT_ALIGN_JUSTIFY:
            // TODO: Distribute space between words
            line.left_offset = 0;
            break;
        default:  // LEFT
            line.left_offset = 0;
            break;
    }

    lines[gid] = line;
}

// =============================================================================
// PASS 4: Position words (parallel per word)
// =============================================================================
kernel void position_words(
    device const Word* words [[buffer(0)]],
    device const LineBox* lines [[buffer(1)]],
    device const uint* word_to_line [[buffer(2)]],
    device InlinePosition* positions [[buffer(3)]],
    constant uint& word_count [[buffer(4)]],
    constant float& ifc_x [[buffer(5)]],  // IFC container position
    constant float& ifc_y [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= word_count) return;

    Word word = words[gid];
    uint line_idx = word_to_line[gid];
    LineBox line = lines[line_idx];

    // Calculate X by summing widths of previous words on same line
    float x = line.left_offset;
    for (uint i = line.first_item; i < gid; i++) {
        if (word_to_line[i] == line_idx) {
            x += words[i].width;
        }
    }

    InlinePosition pos;
    pos.x = ifc_x + x;
    pos.y = ifc_y + line.y;
    pos.width = word.width;
    pos.height = line.height;
    pos.line_index = line_idx;

    positions[gid] = pos;
}

// =============================================================================
// PASS 5: Generate glyph positions (parallel per character)
// =============================================================================
kernel void generate_glyphs(
    device const uint8_t* text [[buffer(0)]],
    device const Word* words [[buffer(1)]],
    device const InlinePosition* word_positions [[buffer(2)]],
    device const ComputedStyleCompact* styles [[buffer(3)]],
    device GlyphPosition* glyphs [[buffer(4)]],
    device atomic_uint* glyph_count [[buffer(5)]],
    constant uint& word_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]  // Word index
) {
    if (gid >= word_count) return;

    Word word = words[gid];
    if (word.is_whitespace || word.is_newline) return;

    InlinePosition pos = word_positions[gid];
    ComputedStyleCompact style = styles[word.element_index];

    float font_size = style.font_size > 0 ? style.font_size : 16.0;
    float x = pos.x;
    float y = pos.y + pos.height * 0.8;  // Baseline

    for (uint i = 0; i < word.text_length; i++) {
        uint8_t c = text[word.text_start + i];
        if (c == ' ' || c == '\t' || c == '\n') continue;

        uint slot = atomic_fetch_add_explicit(glyph_count, 1, memory_order_relaxed);

        GlyphPosition glyph;
        glyph.x = x;
        glyph.y = y;
        glyph.char_code = c;
        glyph.font_size = font_size;
        glyph.color = style.color;

        glyphs[slot] = glyph;

        // Advance
        x += font_size * 0.6;
    }
}
```

## Pseudocode

```
FUNCTION compute_inline_layout(ifc_element, elements, styles, text_buffer):
    // Get IFC container info
    container_width = boxes[ifc_element].content_width
    container_x = boxes[ifc_element].content_x
    container_y = boxes[ifc_element].content_y
    line_height = styles[ifc_element].line_height * styles[ifc_element].font_size

    // Get text range for this IFC
    text_start, text_end = get_text_range(ifc_element, elements)

    // PASS 1: Break into words
    is_word_start = gpu_dispatch(mark_word_boundaries, text_buffer[text_start:text_end])
    word_indices = gpu_prefix_sum(is_word_start)
    word_count = word_indices[text_end - text_start - 1]
    words = gpu_dispatch(build_words, text_buffer, is_word_start, word_indices)

    // PASS 2: Pack into lines
    lines = allocate(estimated_line_count)
    word_to_line = allocate(word_count)
    line_count = gpu_dispatch(pack_lines, words, lines, word_to_line, container_width, line_height)

    // PASS 3: Apply text-align
    text_align = styles[ifc_element].text_align
    gpu_dispatch(apply_text_align, lines, line_count, container_width, text_align)

    // PASS 4: Position words
    word_positions = gpu_dispatch(position_words, words, lines, word_to_line, container_x, container_y)

    // PASS 5: Generate glyphs
    glyphs = gpu_dispatch(generate_glyphs, text_buffer, words, word_positions, styles)

    RETURN InlineLayoutResult {
        lines,
        line_count,
        word_positions,
        glyphs
    }
```

## Tests

### Test 1: Simple word breaking
```rust
#[test]
fn test_word_breaking() {
    let text = "Hello world foo";
    let words = break_into_words(text);

    assert_eq!(words.len(), 5);  // Hello, " ", world, " ", foo
    assert_eq!(words[0].is_whitespace, 0);
    assert_eq!(words[1].is_whitespace, 1);
    assert_eq!(words[2].is_whitespace, 0);
}
```

### Test 2: Line wrapping
```rust
#[test]
fn test_line_wrapping() {
    let text = "Hello world this is a test";
    let container_width = 100.0;  // Forces wrapping
    let result = compute_inline_layout(text, container_width);

    assert!(result.line_count > 1);

    // First line should not exceed container width
    assert!(result.lines[0].width <= container_width);
}
```

### Test 3: Newline handling
```rust
#[test]
fn test_newline_handling() {
    let text = "Line 1\nLine 2\nLine 3";
    let result = compute_inline_layout(text, 1000.0);  // Wide container

    assert_eq!(result.line_count, 3);
}
```

### Test 4: Text align center
```rust
#[test]
fn test_text_align_center() {
    let html = r#"<p style="text-align: center; width: 200px">Short</p>"#;
    let result = compute_layout(html);

    let line = &result.lines[0];
    let expected_offset = (200.0 - line.width) / 2.0;
    assert!((line.left_offset - expected_offset).abs() < 0.1);
}
```

### Test 5: Text align right
```rust
#[test]
fn test_text_align_right() {
    let html = r#"<p style="text-align: right; width: 200px">Short</p>"#;
    let result = compute_layout(html);

    let line = &result.lines[0];
    let expected_offset = 200.0 - line.width;
    assert!((line.left_offset - expected_offset).abs() < 0.1);
}
```

### Test 6: Word positions
```rust
#[test]
fn test_word_positions() {
    let text = "AB CD";
    let result = compute_inline_layout(text, 1000.0);

    // Words should be sequential in X
    let pos_ab = &result.positions[0];
    let pos_space = &result.positions[1];
    let pos_cd = &result.positions[2];

    assert!(pos_ab.x < pos_space.x);
    assert!(pos_space.x < pos_cd.x);
}
```

### Test 7: Glyph generation
```rust
#[test]
fn test_glyph_generation() {
    let text = "Hi";
    let result = compute_inline_layout(text, 1000.0);

    assert_eq!(result.glyphs.len(), 2);
    assert_eq!(result.glyphs[0].char_code, 'H' as u32);
    assert_eq!(result.glyphs[1].char_code, 'i' as u32);
}
```

### Test 8: Long text wrapping
```rust
#[test]
fn test_long_text_wrapping() {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let result = compute_inline_layout(&text, 300.0);

    // Should have multiple lines
    assert!(result.line_count > 5);

    // No line should exceed container width significantly
    for i in 0..result.line_count {
        assert!(result.lines[i].width <= 310.0,
            "Line {} width {} exceeds container", i, result.lines[i].width);
    }
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/inline_layout.rs` | Create | Inline layout implementation |
| `src/gpu_os/document/inline_layout.metal` | Create | GPU kernels |
| `tests/test_issue_100_inline_layout.rs` | Create | Tests |

## Acceptance Criteria

1. [ ] Word boundary detection
2. [ ] Word width measurement
3. [ ] Line box packing
4. [ ] Line wrapping at container width
5. [ ] Newline handling
6. [ ] text-align: left, center, right
7. [ ] Word positioning
8. [ ] Glyph position generation
9. [ ] Whitespace collapsing (normal mode)
10. [ ] All tests pass
