# Issue #131: O(1) Two-Pass Text Line Layout

## Problem Statement

The current `generate_text_vertices` kernel performs text wrapping and vertex generation in a single pass, requiring O(word_length) lookback when words need to move to a new line.

**Current Code (O(word_length) lookback):**
```metal
// paint.metal lines 494-512
// Reposition word vertices when wrapping to new line
for (uint ci = word_start_char; ci < i; ci++) {
    uint v = offset + ci * 4;
    for (int vv = 0; vv < 4; vv++) {
        // Adjust X and Y position - O(word_length) iterations!
        vertices[v + vv].position.x = /* recalculate */;
        vertices[v + vv].position.y = /* recalculate */;
    }
}
```

**Impact:** Long words (e.g., 50 characters) cause 50 × 4 = 200 vertex adjustments. This happens repeatedly for every line wrap.

## Solution: Two-Pass Line Layout

**Pass 1:** Compute line breaks and line metrics (parallel per text element)
**Pass 2:** Generate vertices using pre-computed line data (O(1) per character)

### Data Structures

```rust
/// Pre-computed line information
#[repr(C)]
pub struct LineInfo {
    pub char_start: u32,    // First character index in this line
    pub char_end: u32,      // Last character index (exclusive)
    pub y_offset: f32,      // Y position of this line
    pub width: f32,         // Actual width of text on this line
}

/// Per-element line data
#[repr(C)]
pub struct TextLineData {
    pub line_count: u32,
    pub _padding: [u32; 3],
    pub lines: [LineInfo; MAX_LINES_PER_ELEMENT],  // e.g., 64 lines max
}
```

### Pass 1: Compute Line Breaks

```metal
// Pass 1: Analyze text and compute line breaks
kernel void compute_line_breaks(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint8_t* text [[buffer(3)]],
    device TextLineData* line_data [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT || elem.text_length == 0) return;

    ComputedStyle style = styles[gid];
    LayoutBox box = boxes[gid];

    float font_size = style.font_size > 0 ? style.font_size : 16.0;
    float line_height = (style.line_height > 0 ? style.line_height : 1.2) * font_size;
    float container_width = box.content_width > 0 ? box.content_width : 10000.0;

    // Compute line breaks
    uint line_count = 0;
    uint line_start = 0;
    float x = 0;
    float y = 0;
    uint word_start = 0;
    float word_start_x = 0;
    bool in_word = false;

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
        bool needs_wrap = (x + advance > container_width && x > 0);

        if (is_newline || needs_wrap) {
            // Record this line
            line_data[gid].lines[line_count].char_start = line_start;
            line_data[gid].lines[line_count].char_end = needs_wrap && in_word ? word_start : i;
            line_data[gid].lines[line_count].y_offset = y;
            line_data[gid].lines[line_count].width = needs_wrap && in_word ? word_start_x : x;
            line_count++;

            // Start new line
            if (needs_wrap && in_word) {
                // Word moves to new line
                line_start = word_start;
                x = x - word_start_x;
            } else {
                line_start = i + 1;
                x = 0;
            }
            y += line_height;

            if (is_newline) {
                x = 0;
                continue;
            }
        }

        x += advance;
    }

    // Record final line
    if (line_start < elem.text_length && line_count < MAX_LINES_PER_ELEMENT) {
        line_data[gid].lines[line_count].char_start = line_start;
        line_data[gid].lines[line_count].char_end = elem.text_length;
        line_data[gid].lines[line_count].y_offset = y;
        line_data[gid].lines[line_count].width = x;
        line_count++;
    }

    line_data[gid].line_count = line_count;
}
```

### Pass 2: Generate Vertices (O(1) per character)

```metal
// Pass 2: Generate vertices using pre-computed line data
kernel void generate_text_vertices_fast(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device const ComputedStyle* styles [[buffer(2)]],
    device const uint8_t* text [[buffer(3)]],
    device const TextLineData* line_data [[buffer(4)]],
    device const uint* vertex_offsets [[buffer(5)]],
    device PaintVertex* vertices [[buffer(6)]],
    constant uint& element_count [[buffer(7)]],
    constant Viewport& viewport [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    if (elem.element_type != ELEM_TEXT || elem.text_length == 0) return;

    ComputedStyle style = styles[gid];
    LayoutBox box = boxes[gid];
    TextLineData lines = line_data[gid];

    float font_size = style.font_size > 0 ? style.font_size : 16.0;

    // Find base offset for vertices
    uint background_count = (style.background_color[3] > 0) ? 4 : 0;
    bool has_border = /* ... */;
    uint border_count = has_border ? 16 : 0;
    uint base_offset = vertex_offsets[gid] + background_count + border_count;

    float2 scale = float2(2.0 / viewport.width, -2.0 / viewport.height);
    float2 bias = float2(-1.0, 1.0);

    float4 color = float4(style.color[0], style.color[1], style.color[2],
                          style.color[3] * style.opacity);

    // Process each character with O(1) position lookup
    uint current_line = 0;
    float x = box.content_x;

    for (uint i = 0; i < elem.text_length; i++) {
        // Find which line this character is on - O(1) using binary search or linear scan
        while (current_line < lines.line_count - 1 &&
               i >= lines.lines[current_line].char_end) {
            current_line++;
            x = box.content_x;  // Reset X for new line
        }

        // Y position is O(1) lookup from pre-computed line data
        float y = box.content_y + lines.lines[current_line].y_offset;

        uint c = text[elem.text_start + i];
        float advance = glyph_advance(c, font_size);

        // Generate vertex quad - NO LOOKBACK NEEDED!
        uint v = base_offset + i * 4;
        bool is_visible = (c != ' ' && c != '\t' && c != '\n' && c != '\r');
        float4 vertex_color = is_visible ? color : float4(0);

        // Top-left
        vertices[v + 0].position = float2(x, y) * scale + bias;
        vertices[v + 0].tex_coord = float2(float(c % 16) / 16.0, float(c / 16) / 16.0);
        vertices[v + 0].color = vertex_color;
        vertices[v + 0].flags = FLAG_TEXT;

        // Top-right, Bottom-right, Bottom-left...
        vertices[v + 1].position = float2(x + advance, y) * scale + bias;
        vertices[v + 2].position = float2(x + advance, y + font_size) * scale + bias;
        vertices[v + 3].position = float2(x, y + font_size) * scale + bias;

        // ... (copy tex_coord, color, flags to other vertices)

        x += advance;
    }
}
```

### Alternative: Parallel Per-Character (Maximum Parallelism)

```metal
// Even more parallel: one thread per character
kernel void generate_char_vertex_parallel(
    device const Element* elements [[buffer(0)]],
    device const TextLineData* line_data [[buffer(1)]],
    device const uint8_t* text [[buffer(2)]],
    device const CharMapping* char_to_element [[buffer(3)]],  // Maps char index to element
    device PaintVertex* vertices [[buffer(4)]],
    constant uint& total_chars [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_chars) return;

    // O(1): Look up which element this character belongs to
    CharMapping mapping = char_to_element[gid];
    uint elem_idx = mapping.element_index;
    uint char_idx = mapping.char_index_in_element;

    // O(1): Look up line data
    TextLineData lines = line_data[elem_idx];

    // O(log n): Binary search to find line (could be O(1) with per-char line index)
    uint line = find_line_for_char(lines, char_idx);

    // O(1): Compute position
    float y = lines.lines[line].y_offset;
    float x = compute_x_in_line(/* ... */);

    // Generate 4 vertices for this character
    // ...
}
```

## Rust-Side Integration

```rust
impl GpuPaint {
    pub fn render_text(&mut self, encoder: &ComputeCommandEncoderRef) {
        // Pass 1: Compute line breaks (one dispatch)
        encoder.set_compute_pipeline_state(&self.compute_line_breaks_pipeline);
        encoder.set_buffer(0, Some(&self.elements_buffer), 0);
        encoder.set_buffer(1, Some(&self.boxes_buffer), 0);
        encoder.set_buffer(2, Some(&self.styles_buffer), 0);
        encoder.set_buffer(3, Some(&self.text_buffer), 0);
        encoder.set_buffer(4, Some(&self.line_data_buffer), 0);
        encoder.dispatch_threadgroups(/* ... */);

        // Memory barrier between passes
        encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);

        // Pass 2: Generate vertices (one dispatch)
        encoder.set_compute_pipeline_state(&self.generate_text_vertices_pipeline);
        // ... set buffers including line_data_buffer
        encoder.dispatch_threadgroups(/* ... */);
    }
}
```

## Benchmarks

### Test Cases

1. **Long paragraph:** 10,000 characters, narrow container (many line breaks)
2. **Code block:** Many short lines, wide container
3. **Mixed content:** Document with various text elements

### Expected Performance

| Scenario | Current (O(word_len) lookback) | New (O(1) per char) | Speedup |
|----------|-------------------------------|---------------------|---------|
| 10K chars, 100 wraps | ~2ms | ~200μs | 10x |
| 1K chars, 50 wraps | ~500μs | ~50μs | 10x |
| No wrapping | ~100μs | ~120μs | 0.8x (slight overhead) |

### Benchmark Code

```rust
#[test]
fn benchmark_text_layout() {
    let device = Device::system_default().unwrap();

    // Create document with long text that wraps many times
    let text = "word ".repeat(2000);  // 10K chars, many wraps
    let doc = create_text_document(&text, 200.0);  // Narrow container

    let old_time = benchmark_single_pass(&device, &doc);
    let new_time = benchmark_two_pass(&device, &doc);

    println!("Single-pass: {:.2}ms, Two-pass: {:.2}ms, Speedup: {:.1}x",
        old_time * 1e3, new_time * 1e3, old_time / new_time);
}
```

## Memory Overhead

- LineInfo: 16 bytes
- TextLineData: 16 + 64 * 16 = 1040 bytes per text element
- 1000 text elements = 1 MB

For memory-constrained scenarios, could use compressed format or streaming.

## Success Criteria

1. **Correctness:** Rendered text identical to current implementation
2. **Performance:** ≥5x speedup for text with many line wraps
3. **Memory:** ≤2KB per text element overhead
4. **Visual quality:** Proper word wrapping and margin handling

## Implementation Steps

1. Define `LineInfo` and `TextLineData` structures
2. Create `compute_line_breaks` Metal kernel
3. Create `generate_text_vertices_fast` Metal kernel
4. Add `line_data_buffer` to `GpuPaint`
5. Implement two-pass dispatch in Rust
6. Add tests comparing visual output
7. Add benchmarks
