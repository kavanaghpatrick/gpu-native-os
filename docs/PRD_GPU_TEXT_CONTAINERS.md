# PRD: GPU-Native Text Containers (Issue #90)

## Problem Statement

Text layout currently uses:
1. **Monospace approximation** (0.6 × font_size) instead of real glyph metrics
2. **Single-threaded line breaking** kernel
3. **Placeholder quads** in paint stage instead of actual glyph rendering

Wikipedia renders 5000+ text elements. We need 100% GPU text layout with zero CPU involvement (except I/O).

## Existing Infrastructure

| Component | File | Status |
|-----------|------|--------|
| Text buffer (512KB) | `parser.rs` | ✅ GPU buffer |
| LineBox structure | `text.rs` | ✅ Exists |
| Break detection | `text.metal` | ✅ Parallel |
| Line breaking | `text.metal` | ⚠️ Single-thread |
| Glyph metrics | `text.metal` | ❌ Hardcoded 0.6× |
| Paint vertices | `paint.metal` | ❌ Placeholder |
| Font atlas | `text_render.rs` | ✅ 8×8 bitmap |

## Design: GPU-Only Architecture

### Data Flow (All GPU)

```
text_buffer (raw bytes from I/O)
        ↓
[Kernel 1: char_to_glyph] ← glyph_metrics buffer
        ↓
glyph_ids + advances buffer
        ↓
[Kernel 2: prefix_sum_widths] (parallel)
        ↓
cumulative_widths buffer
        ↓
[Kernel 3: find_breaks] (parallel)
        ↓
break_opportunities buffer
        ↓
[Kernel 4: assign_lines] (parallel per paragraph)
        ↓
line_assignments buffer
        ↓
[Kernel 5: position_glyphs] (parallel)
        ↓
positioned_glyphs buffer → SDF renderer
```

### GPU Buffers

```rust
// Uploaded once at font load (CPU → GPU)
#[repr(C)]
struct GlyphMetrics {
    advance: f32,           // Horizontal advance width
    bearing_x: f32,         // Left side bearing
    bearing_y: f32,         // Top bearing (baseline to top)
    width: f32,             // Glyph bbox width
    height: f32,            // Glyph bbox height
    atlas_x: u16,           // Atlas position X
    atlas_y: u16,           // Atlas position Y
    atlas_w: u16,           // Atlas size W
    atlas_h: u16,           // Atlas size H
}
// Size: 256 entries × 24 bytes = 6KB (ASCII + extended Latin)

// Per-character output
#[repr(C)]
struct PositionedGlyph {
    x: f32,                 // Screen X
    y: f32,                 // Screen Y
    glyph_id: u32,          // Index into atlas
    color: u32,             // Packed RGBA
    scale: f32,             // Font size / base size
    line_index: u32,        // Which line this glyph is on
    _padding: [f32; 2],
}

// Line metadata
#[repr(C)]
struct LineInfo {
    start_char: u32,        // First char index
    end_char: u32,          // Last char index (exclusive)
    width: f32,             // Total line width
    x_offset: f32,          // After text-align
    y_offset: f32,          // Cumulative line heights
    container_id: u32,      // Which LayoutBox
    _padding: [f32; 2],
}
```

## Metal Kernel Pseudocode

### Kernel 1: Character to Glyph (Parallel)

```metal
kernel void char_to_glyph(
    device const uint8_t* text [[buffer(0)]],
    device const GlyphMetrics* metrics [[buffer(1)]],  // 256 entries
    device float* advances [[buffer(2)]],
    device uint* glyph_ids [[buffer(3)]],
    constant uint& char_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    uint8_t c = text[gid];
    uint glyph_id = (c >= 32 && c < 128) ? (c - 32) : 0;  // ASCII mapping

    glyph_ids[gid] = glyph_id;
    advances[gid] = metrics[glyph_id].advance;
}
```

### Kernel 2: Parallel Prefix Sum for Widths

```metal
// Two-pass Blelloch scan
kernel void prefix_sum_up(
    device float* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* temp [[threadgroup(0)]]
) {
    // Up-sweep phase
    temp[tid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < 1024; stride *= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 1024) {
            temp[idx] += temp[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Down-sweep phase
    if (tid == 0) temp[1023] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride /= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 1024) {
            float t = temp[idx - stride];
            temp[idx - stride] = temp[idx];
            temp[idx] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    data[gid] = temp[tid];
}
```

### Kernel 3: Find Break Opportunities (Parallel)

```metal
kernel void find_breaks(
    device const uint8_t* text [[buffer(0)]],
    device uint* is_break [[buffer(1)]],      // 1 = can break here
    device uint* break_type [[buffer(2)]],    // SPACE, NEWLINE, HYPHEN
    constant uint& char_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    uint8_t c = text[gid];

    // Break AFTER spaces, newlines, hyphens
    if (c == ' ') {
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
```

### Kernel 4: Assign Lines (Parallel per Paragraph)

```metal
// Key insight: Each paragraph can be processed independently
// Within paragraph: greedy algorithm but parallelizable via binary search

kernel void assign_lines(
    device const float* cumulative_widths [[buffer(0)]],
    device const uint* is_break [[buffer(1)]],
    device const uint* paragraph_starts [[buffer(2)]],  // Start indices
    device const uint* paragraph_ends [[buffer(3)]],    // End indices
    device const float* container_widths [[buffer(4)]], // Width per paragraph
    device uint* line_index [[buffer(5)]],              // Output: line # per char
    device atomic_uint* line_count [[buffer(6)]],       // Output: total lines
    constant uint& paragraph_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]  // One thread per paragraph
) {
    if (gid >= paragraph_count) return;

    uint start = paragraph_starts[gid];
    uint end = paragraph_ends[gid];
    float max_width = container_widths[gid];

    uint current_line = atomic_fetch_add_explicit(line_count, 0, memory_order_relaxed);
    float line_start_width = cumulative_widths[start];

    for (uint i = start; i < end; i++) {
        float width_so_far = cumulative_widths[i] - line_start_width;

        if (width_so_far > max_width && is_break[i]) {
            // Start new line
            current_line = atomic_fetch_add_explicit(line_count, 1, memory_order_relaxed);
            line_start_width = cumulative_widths[i];
        }

        line_index[i] = current_line;
    }
}
```

### Kernel 5: Position Glyphs (Fully Parallel)

```metal
kernel void position_glyphs(
    device const float* cumulative_widths [[buffer(0)]],
    device const uint* line_index [[buffer(1)]],
    device const LineInfo* lines [[buffer(2)]],
    device const uint* glyph_ids [[buffer(3)]],
    device const uint* colors [[buffer(4)]],
    device PositionedGlyph* output [[buffer(5)]],
    constant float& line_height [[buffer(6)]],
    constant uint& char_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= char_count) return;

    uint line = line_index[gid];
    LineInfo li = lines[line];

    // X = cumulative width - line start width + container offset + text-align offset
    float x = cumulative_widths[gid] - cumulative_widths[li.start_char] + li.x_offset;

    // Y = line index × line height + container offset
    float y = float(line) * line_height + li.y_offset;

    output[gid].x = x;
    output[gid].y = y;
    output[gid].glyph_id = glyph_ids[gid];
    output[gid].color = colors[gid];
    output[gid].line_index = line;
}
```

## Integration with Existing Code

### Changes to `text.metal`

Replace `measure_text_advances()` with `char_to_glyph()` using real metrics:

```metal
// OLD (monospace approximation)
float advance = (c == ' ') ? font_size * 0.3 : font_size * 0.6;

// NEW (real metrics)
float advance = metrics[glyph_id].advance * (font_size / BASE_FONT_SIZE);
```

### Changes to `layout.metal`

Update intrinsic sizing to use prefix sum:

```metal
// OLD
float text_width(uint text_length, float font_size) {
    return float(text_length) * font_size * 0.6;
}

// NEW - read from pre-computed buffer
float text_width(device const float* cumulative_widths, uint start, uint end) {
    return cumulative_widths[end] - cumulative_widths[start];
}
```

### Changes to `paint.metal`

Connect to positioned glyphs:

```metal
// OLD - placeholder quads
for each char: emit 4 vertices at estimated position

// NEW - read from positioned_glyphs buffer
kernel void generate_text_vertices(
    device const PositionedGlyph* glyphs [[buffer(0)]],
    device const GlyphMetrics* metrics [[buffer(1)]],
    device PaintVertex* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    PositionedGlyph g = glyphs[gid];
    GlyphMetrics m = metrics[g.glyph_id];

    // Generate 4 vertices for glyph quad
    float x0 = g.x + m.bearing_x * g.scale;
    float y0 = g.y - m.bearing_y * g.scale;
    float x1 = x0 + m.width * g.scale;
    float y1 = y0 + m.height * g.scale;

    // UV from atlas
    float u0 = float(m.atlas_x) / ATLAS_WIDTH;
    float v0 = float(m.atlas_y) / ATLAS_HEIGHT;
    float u1 = u0 + float(m.atlas_w) / ATLAS_WIDTH;
    float v1 = v0 + float(m.atlas_h) / ATLAS_HEIGHT;

    uint base = gid * 6;
    output[base + 0] = {float2(x0, y0), float2(u0, v0), unpack_color(g.color), FLAG_TEXT};
    output[base + 1] = {float2(x1, y0), float2(u1, v0), unpack_color(g.color), FLAG_TEXT};
    output[base + 2] = {float2(x1, y1), float2(u1, v1), unpack_color(g.color), FLAG_TEXT};
    output[base + 3] = {float2(x0, y0), float2(u0, v0), unpack_color(g.color), FLAG_TEXT};
    output[base + 4] = {float2(x1, y1), float2(u1, v1), unpack_color(g.color), FLAG_TEXT};
    output[base + 5] = {float2(x0, y1), float2(u0, v1), unpack_color(g.color), FLAG_TEXT};
}
```

## Test Cases

### Test 1: Single Line (No Wrapping)

```rust
#[test]
fn test_single_line_no_wrap() {
    let text = b"Hello World";
    let container_width = 1000.0;  // Wide enough

    let glyphs = gpu_layout_text(text, container_width);

    // All on line 0
    assert!(glyphs.iter().all(|g| g.line_index == 0));

    // X positions are monotonically increasing
    for i in 1..glyphs.len() {
        assert!(glyphs[i].x > glyphs[i-1].x);
    }
}
```

### Test 2: Line Wrapping

```rust
#[test]
fn test_line_wrap_at_space() {
    let text = b"Hello World";
    let container_width = 60.0;  // Force wrap after "Hello"

    let glyphs = gpu_layout_text(text, container_width);

    // "Hello" on line 0, " World" on line 1
    assert_eq!(glyphs[0].line_index, 0);  // H
    assert_eq!(glyphs[5].line_index, 1);  // space starts new line
    assert_eq!(glyphs[6].line_index, 1);  // W
}
```

### Test 3: Prefix Sum Correctness

```rust
#[test]
fn test_cumulative_widths() {
    let text = b"ABC";
    let metrics = load_test_metrics();  // A=10, B=12, C=8

    let cumulative = gpu_prefix_sum(text, &metrics);

    assert_eq!(cumulative[0], 0.0);   // Before A
    assert_eq!(cumulative[1], 10.0);  // After A
    assert_eq!(cumulative[2], 22.0);  // After B
    assert_eq!(cumulative[3], 30.0);  // After C
}
```

### Test 4: 10,000 Characters Performance

```rust
#[test]
fn test_10k_chars_under_16ms() {
    let text: Vec<u8> = (0..10_000).map(|i| b'A' + (i % 26) as u8).collect();

    let start = Instant::now();
    let _ = gpu_layout_text(&text, 800.0);
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 16, "Must complete in one frame");
}
```

### Test 5: Wikipedia Scale (50,000 chars)

```rust
#[test]
#[ignore]  // Run with --ignored
fn test_wikipedia_scale() {
    let text = fetch_wikipedia_text();  // ~50KB

    let start = Instant::now();
    let glyphs = gpu_layout_text(&text, 800.0);
    let elapsed = start.elapsed();

    println!("50K chars: {:?}", elapsed);
    assert!(elapsed.as_millis() < 50);
    assert!(glyphs.len() > 0);
}
```

### Test 6: Multi-Paragraph Parallel

```rust
#[test]
fn test_paragraphs_independent() {
    let text = b"Para one.\n\nPara two.\n\nPara three.";

    let glyphs = gpu_layout_text(text, 100.0);

    // Each paragraph's line numbering is independent
    // Verify Y positions reset per paragraph
}
```

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| 1K chars | < 1ms | Interactive typing |
| 10K chars | < 16ms | One frame (60fps) |
| 50K chars | < 50ms | Wikipedia article |
| 100K chars | < 100ms | Large document |

## Files to Modify

| File | Changes |
|------|---------|
| `src/gpu_os/document/text.rs` | Add new kernel dispatch, buffers |
| `src/gpu_os/document/text.metal` | New kernels 1-5 |
| `src/gpu_os/document/layout.metal` | Use real metrics for intrinsic sizing |
| `src/gpu_os/document/paint.metal` | Generate text vertices from positioned glyphs |
| `src/gpu_os/text_render.rs` | Extract glyph metrics to GPU buffer |
| `tests/test_issue_90_text_containers.rs` | New test file |

## Non-Goals (V1)

- Kerning (add in V2)
- Ligatures (add in V2)
- RTL/BiDi (add in V2)
- Hyphenation (add in V2)
- Justified text (add in V2)

## Success Criteria

1. All text layout computed on GPU (zero CPU text processing)
2. Real glyph metrics from font (not 0.6× approximation)
3. Parallel line breaking (one thread per paragraph)
4. 10K characters in < 16ms
5. Correct line wrapping matching browser behavior
