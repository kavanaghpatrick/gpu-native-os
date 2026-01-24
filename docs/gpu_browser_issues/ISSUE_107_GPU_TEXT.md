# Issue #107: GPU Text Measurement

## Summary
Integrate the existing SDF text engine with the layout pipeline for GPU-accelerated text measurement and rendering.

## Motivation
Text measurement is a critical bottleneck in browser layout:
- Each text run requires font metric queries
- Line breaking requires multiple measurement passes
- Current browsers do this sequentially on CPU

Our existing `text_render.rs` SDF atlas can be extended for GPU-native text measurement.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Text Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Text nodes from DOM + Computed styles                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Phase 1: Glyph Lookup                                    │    │
│  │   - Map characters to glyph indices                      │    │
│  │   - Load glyph metrics from atlas                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Phase 2: Text Measurement                                │    │
│  │   - Calculate advance widths                             │    │
│  │   - Apply kerning (optional)                             │    │
│  │   - Compute line breaks                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Phase 3: Line Layout                                     │    │
│  │   - Position glyphs on lines                             │    │
│  │   - Handle text-align                                    │    │
│  │   - Calculate total height                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  Output:                                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ TextMeasurement per text node                            │    │
│  │ GlyphRun per text node (for rendering)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

```metal
// Glyph metrics from SDF atlas
struct GlyphMetrics {
    float advance;          // Horizontal advance after glyph
    float2 bearing;         // Offset from baseline to top-left
    float2 size;            // Glyph bounding box size
    float4 uv_rect;         // UV coordinates in atlas texture
};

// Per-text-node measurement result
struct TextMeasurement {
    float width;            // Total text width (max line width)
    float height;           // Total text height (all lines)
    uint line_count;
    uint glyph_count;
    uint glyph_run_offset;  // Offset into GlyphRun buffer
};

// Positioned glyph for rendering
struct GlyphPosition {
    float2 position;        // Relative to text node origin
    uint glyph_index;       // Index into glyph metrics
    float scale;            // Font size scaling
};

// Line information for text-align
struct LineInfo {
    uint start_glyph;       // First glyph index
    uint glyph_count;
    float width;
    float baseline_y;
};

// Text run for a single text node
struct TextRun {
    uint text_offset;       // Offset into text buffer
    uint text_length;
    float font_size;
    float line_height;
    float max_width;        // For line breaking
    uint text_align;        // left, center, right, justify
};
```

## Metal Kernel Implementation

```metal
// src/gpu_os/document/text_layout.metal

#include <metal_stdlib>
using namespace metal;

// Character to glyph index mapping
uint get_glyph_index(uint codepoint, device uint* cmap) {
    // Simple ASCII for now, extend to Unicode later
    if (codepoint >= 32 && codepoint < 127) {
        return codepoint - 32;  // ASCII printable range
    }
    return 0;  // Default glyph (space or tofu)
}

//=============================================================================
// Phase 1: Measure text width and find line breaks
//=============================================================================

kernel void measure_text(
    device TextRun* text_runs [[buffer(0)]],
    device uchar* text_buffer [[buffer(1)]],
    device GlyphMetrics* glyph_metrics [[buffer(2)]],
    device TextMeasurement* measurements [[buffer(3)]],
    device uint* line_breaks [[buffer(4)]],  // Output: indices of line breaks
    device atomic_uint* line_break_counts [[buffer(5)]],
    constant uint& text_node_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= text_node_count) return;

    TextRun run = text_runs[gid];
    float font_scale = run.font_size / 16.0;  // Assuming 16px base

    float line_width = 0;
    float max_line_width = 0;
    uint line_count = 1;
    uint glyph_count = 0;

    uint word_start = 0;
    float word_width = 0;

    uint lb_offset = gid * MAX_LINES_PER_TEXT;
    uint lb_count = 0;

    for (uint i = 0; i < run.text_length; i++) {
        uchar c = text_buffer[run.text_offset + i];

        // Handle newlines
        if (c == '\n') {
            max_line_width = max(max_line_width, line_width);
            line_width = 0;
            line_count++;
            if (lb_count < MAX_LINES_PER_TEXT) {
                line_breaks[lb_offset + lb_count++] = i;
            }
            word_start = i + 1;
            word_width = 0;
            continue;
        }

        // Handle spaces (word boundaries)
        if (c == ' ') {
            word_start = i + 1;
            word_width = 0;
        }

        uint glyph_idx = get_glyph_index(c, nullptr);
        GlyphMetrics glyph = glyph_metrics[glyph_idx];
        float advance = glyph.advance * font_scale;

        // Check for line wrap
        if (line_width + advance > run.max_width && line_width > 0) {
            // Wrap at word boundary if possible
            if (word_start > 0 && word_width < line_width) {
                // Break before current word
                max_line_width = max(max_line_width, line_width - word_width);
                line_width = word_width;
                if (lb_count < MAX_LINES_PER_TEXT) {
                    line_breaks[lb_offset + lb_count++] = word_start - 1;
                }
            } else {
                // Break at current character
                max_line_width = max(max_line_width, line_width);
                line_width = 0;
                if (lb_count < MAX_LINES_PER_TEXT) {
                    line_breaks[lb_offset + lb_count++] = i;
                }
            }
            line_count++;
        }

        line_width += advance;
        word_width += advance;
        glyph_count++;
    }

    max_line_width = max(max_line_width, line_width);

    measurements[gid] = TextMeasurement {
        .width = max_line_width,
        .height = line_count * run.line_height * font_scale,
        .line_count = line_count,
        .glyph_count = glyph_count,
        .glyph_run_offset = 0  // Set in allocation pass
    };

    atomic_store_explicit(&line_break_counts[gid], lb_count, memory_order_relaxed);
}

//=============================================================================
// Phase 2: Generate glyph positions
//=============================================================================

kernel void position_glyphs(
    device TextRun* text_runs [[buffer(0)]],
    device uchar* text_buffer [[buffer(1)]],
    device GlyphMetrics* glyph_metrics [[buffer(2)]],
    device TextMeasurement* measurements [[buffer(3)]],
    device uint* line_breaks [[buffer(4)]],
    device uint* line_break_counts [[buffer(5)]],
    device GlyphPosition* glyph_positions [[buffer(6)]],
    device LineInfo* line_infos [[buffer(7)]],
    constant uint& text_node_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= text_node_count) return;

    TextRun run = text_runs[gid];
    TextMeasurement meas = measurements[gid];
    float font_scale = run.font_size / 16.0;
    float line_height = run.line_height * font_scale;

    uint lb_offset = gid * MAX_LINES_PER_TEXT;
    uint lb_count = line_break_counts[gid];

    uint glyph_offset = meas.glyph_run_offset;
    uint glyph_idx = 0;
    uint line_idx = 0;
    uint line_start_glyph = 0;

    float x = 0;
    float y = font_scale * 0.8;  // Baseline offset (approximate)
    float line_width = 0;

    for (uint i = 0; i < run.text_length; i++) {
        uchar c = text_buffer[run.text_offset + i];

        // Check for line break
        bool is_break = (c == '\n');
        for (uint lb = 0; lb < lb_count && !is_break; lb++) {
            if (line_breaks[lb_offset + lb] == i) {
                is_break = true;
            }
        }

        if (is_break) {
            // Store line info
            line_infos[gid * MAX_LINES_PER_TEXT + line_idx] = LineInfo {
                .start_glyph = line_start_glyph,
                .glyph_count = glyph_idx - line_start_glyph,
                .width = line_width,
                .baseline_y = y
            };

            // Apply text-align
            float offset = 0;
            if (run.text_align == TEXT_ALIGN_CENTER) {
                offset = (meas.width - line_width) / 2;
            } else if (run.text_align == TEXT_ALIGN_RIGHT) {
                offset = meas.width - line_width;
            }

            // Adjust positions for this line
            for (uint g = line_start_glyph; g < glyph_idx; g++) {
                glyph_positions[glyph_offset + g].position.x += offset;
            }

            x = 0;
            y += line_height;
            line_width = 0;
            line_idx++;
            line_start_glyph = glyph_idx;

            if (c == '\n') continue;
        }

        if (c == ' ' || c == '\t') {
            // Space character
            GlyphMetrics space = glyph_metrics[0];  // Space glyph
            x += space.advance * font_scale;
            line_width = x;
            continue;
        }

        uint glyph = get_glyph_index(c, nullptr);
        GlyphMetrics metrics = glyph_metrics[glyph];

        glyph_positions[glyph_offset + glyph_idx] = GlyphPosition {
            .position = float2(x + metrics.bearing.x * font_scale,
                              y - metrics.bearing.y * font_scale),
            .glyph_index = glyph,
            .scale = font_scale
        };

        x += metrics.advance * font_scale;
        line_width = x;
        glyph_idx++;
    }

    // Handle last line
    if (glyph_idx > line_start_glyph) {
        line_infos[gid * MAX_LINES_PER_TEXT + line_idx] = LineInfo {
            .start_glyph = line_start_glyph,
            .glyph_count = glyph_idx - line_start_glyph,
            .width = line_width,
            .baseline_y = y
        };

        float offset = 0;
        if (run.text_align == TEXT_ALIGN_CENTER) {
            offset = (meas.width - line_width) / 2;
        } else if (run.text_align == TEXT_ALIGN_RIGHT) {
            offset = meas.width - line_width;
        }

        for (uint g = line_start_glyph; g < glyph_idx; g++) {
            glyph_positions[glyph_offset + g].position.x += offset;
        }
    }
}

//=============================================================================
// Phase 3: Generate text vertices for rendering
//=============================================================================

kernel void generate_text_vertices(
    device GlyphPosition* glyph_positions [[buffer(0)]],
    device GlyphMetrics* glyph_metrics [[buffer(1)]],
    device TextMeasurement* measurements [[buffer(2)]],
    device GPULayoutBox* layout [[buffer(3)]],
    device PaintVertex* vertices [[buffer(4)]],
    device atomic_uint* vertex_count [[buffer(5)]],
    device uint* text_node_indices [[buffer(6)]],
    constant uint& text_node_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= text_node_count) return;

    uint node_idx = text_node_indices[gid];
    TextMeasurement meas = measurements[gid];
    GPULayoutBox box = layout[node_idx];

    float2 origin = box.position + box.content_position;

    // Each glyph = 6 vertices (2 triangles)
    uint base = atomic_fetch_add_explicit(vertex_count, meas.glyph_count * 6, memory_order_relaxed);

    for (uint i = 0; i < meas.glyph_count; i++) {
        GlyphPosition gpos = glyph_positions[meas.glyph_run_offset + i];
        GlyphMetrics glyph = glyph_metrics[gpos.glyph_index];

        float2 pos = origin + gpos.position;
        float2 size = glyph.size * gpos.scale;

        // Quad corners
        float x0 = pos.x;
        float y0 = pos.y;
        float x1 = pos.x + size.x;
        float y1 = pos.y + size.y;

        // UV coordinates from atlas
        float u0 = glyph.uv_rect.x;
        float v0 = glyph.uv_rect.y;
        float u1 = glyph.uv_rect.z;
        float v1 = glyph.uv_rect.w;

        uint vi = base + i * 6;

        // Triangle 1
        vertices[vi + 0] = make_text_vertex(x0, y0, u0, v0);
        vertices[vi + 1] = make_text_vertex(x1, y0, u1, v0);
        vertices[vi + 2] = make_text_vertex(x0, y1, u0, v1);

        // Triangle 2
        vertices[vi + 3] = make_text_vertex(x1, y0, u1, v0);
        vertices[vi + 4] = make_text_vertex(x1, y1, u1, v1);
        vertices[vi + 5] = make_text_vertex(x0, y1, u0, v1);
    }
}
```

## Integration with Existing text_render.rs

```rust
// Extend existing text_render.rs

impl TextRenderer {
    /// Prepare text runs for GPU measurement
    pub fn prepare_text_runs(
        &self,
        gpu_dom: &GPUDom,
        computed_styles: &[GPUComputedStyle],
    ) -> (Buffer, u32) {
        let mut text_runs = Vec::new();

        for (idx, node) in gpu_dom.nodes.iter().enumerate() {
            if node.element_type == ELEM_TEXT {
                let parent_style = &computed_styles[node.parent_idx as usize];

                text_runs.push(TextRun {
                    text_offset: node.text_offset,
                    text_length: node.text_length as u32,
                    font_size: parent_style.font_size,
                    line_height: parent_style.line_height,
                    max_width: f32::INFINITY,  // Updated by layout engine
                    text_align: parent_style.text_align,
                });
            }
        }

        let buffer = self.device.new_buffer_with_data(
            text_runs.as_ptr() as *const _,
            (text_runs.len() * std::mem::size_of::<TextRun>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        (buffer, text_runs.len() as u32)
    }

    /// Run GPU text measurement
    pub fn measure_text_gpu(
        &self,
        command_buffer: &CommandBufferRef,
        text_runs: &Buffer,
        text_buffer: &Buffer,
        text_node_count: u32,
    ) -> Buffer {
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.measure_pipeline);
        encoder.set_buffer(0, Some(text_runs), 0);
        encoder.set_buffer(1, Some(text_buffer), 0);
        encoder.set_buffer(2, Some(&self.glyph_metrics_buffer), 0);
        encoder.set_buffer(3, Some(&self.measurements_buffer), 0);
        encoder.set_buffer(4, Some(&self.line_breaks_buffer), 0);
        encoder.set_buffer(5, Some(&self.line_break_counts_buffer), 0);
        encoder.set_bytes(6, 4, &text_node_count as *const _ as *const _);

        let threads = MTLSize::new(text_node_count as u64, 1, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
        encoder.end_encoding();

        self.measurements_buffer.clone()
    }
}
```

## Benchmarks

```rust
fn bench_text_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_measurement");

    let test_cases = vec![
        ("short", "Hello, World!"),
        ("paragraph", include_str!("test_text/paragraph.txt")),
        ("article", include_str!("test_text/article.txt")),  // ~10K chars
    ];

    for (name, text) in test_cases {
        group.bench_function(format!("cpu_{}", name), |b| {
            b.iter(|| cpu_measure_text(text, 16.0, 400.0))
        });

        group.bench_function(format!("gpu_{}", name), |b| {
            b.iter(|| {
                let cmd = queue.new_command_buffer();
                renderer.measure_text_gpu(&cmd, &runs, &text_buf, 1);
                cmd.commit();
                cmd.wait_until_completed();
            })
        });
    }
}
```

### Expected Results

| Text Size | Characters | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| Short | 13 | 0.01ms | 0.01ms | 1x |
| Paragraph | 500 | 0.5ms | 0.02ms | 25x |
| Article | 10000 | 10ms | 0.1ms | 100x |

## Tests

```rust
#[test]
fn test_single_line() {
    let text = "Hello World";
    let meas = measure_text(text, 16.0, f32::INFINITY);

    assert!(meas.width > 0.0);
    assert_eq!(meas.line_count, 1);
}

#[test]
fn test_line_wrap() {
    let text = "This is a longer text that should wrap";
    let meas = measure_text(text, 16.0, 100.0);  // Narrow container

    assert!(meas.line_count > 1);
    assert!(meas.width <= 100.0);
}

#[test]
fn test_explicit_newlines() {
    let text = "Line 1\nLine 2\nLine 3";
    let meas = measure_text(text, 16.0, f32::INFINITY);

    assert_eq!(meas.line_count, 3);
}

#[test]
fn test_text_align_center() {
    let text = "Centered";
    let positions = position_glyphs(text, 16.0, 200.0, TEXT_ALIGN_CENTER);

    let first_x = positions[0].position.x;
    assert!(first_x > 0.0);  // Should have left margin
}

#[test]
fn test_glyph_count() {
    let text = "ABC";
    let meas = measure_text(text, 16.0, f32::INFINITY);

    assert_eq!(meas.glyph_count, 3);
}
```

## Acceptance Criteria

- [ ] Text measurement matches CPU reference
- [ ] Line breaking works correctly
- [ ] Word wrapping at word boundaries
- [ ] text-align: left/center/right work
- [ ] Multiple font sizes supported
- [ ] Glyph positions correct for rendering
- [ ] Performance: 10x+ speedup on large text
- [ ] Integration with layout engine complete

## Dependencies

- Issue #106: GPU Layout Engine
- Existing text_render.rs SDF infrastructure

## Blocks

- Issue #109: Benchmarking Framework
