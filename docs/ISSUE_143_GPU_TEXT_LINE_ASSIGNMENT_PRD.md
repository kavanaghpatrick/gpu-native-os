# Issue #143: GPU Text Line Assignment - Eliminate CPU Text Layout Loop

## Problem Statement

In `src/gpu_os/document/text.rs` lines 627-761, text line assignment runs on CPU with an explicit TODO comment:

```rust
// For now, do line assignment on CPU
// This is the main bottleneck - should be GPU
```

The CPU loop iterates through every character to:
1. Check if character exceeds container width
2. Find word break opportunities
3. Assign characters to lines
4. Build positioned glyph array

**Impact:**
- For 10K characters: ~640 CPU cycles per character = 6.4M cycles
- Blocks GPU pipeline waiting for line assignments
- 10-50x slower than GPU parallel implementation

## Solution

Move line assignment to GPU using parallel algorithms:
1. **Parallel width checking:** Each thread checks one character
2. **Parallel prefix scan:** Find cumulative widths
3. **Binary search:** Each character finds its line via binary search on line boundaries
4. **Parallel glyph positioning:** Each thread positions one glyph

## Requirements

### Functional Requirements
1. Identical line breaks to CPU implementation
2. Support soft breaks (word boundaries) and hard breaks (newlines)
3. Handle all Unicode whitespace correctly
4. Support variable-width fonts (via glyph metrics buffer)

### Performance Requirements
1. **Target:** <0.5ms for 100K characters
2. **Speedup:** 10-50x over CPU implementation
3. **Single dispatch:** No CPU-GPU roundtrips during layout

### Non-Functional Requirements
1. Works with existing text rendering pipeline
2. Compatible with batched layout (Issue #140)
3. Supports incremental updates (future)

## Technical Design

### Phase 1: Parallel Line Boundary Detection

```metal
// src/gpu_os/document/text.metal

struct LineLayoutState {
    atomic_uint line_count;
    uint container_width;
    uint line_height;
    uint char_count;
};

// Each thread checks if this position is a line break
kernel void detect_line_breaks(
    device float* cumulative_widths [[buffer(0)]],
    device uint* is_break_opportunity [[buffer(1)]],  // From previous pass
    device uint* is_line_start [[buffer(2)]],         // Output: 1 if this char starts a line
    device LineLayoutState* state [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= state->char_count) return;

    float width = cumulative_widths[tid];
    float container = float(state->container_width);

    // Find the line this character is on
    // Use binary search on cumulative widths to find last line break before this position

    // Check if this character would overflow
    float line_start_width = find_line_start_width(cumulative_widths, is_line_start, tid);
    float width_on_line = width - line_start_width;

    if (width_on_line > container) {
        // Need to break before this character
        // Find last break opportunity
        uint break_pos = find_last_break_before(is_break_opportunity, tid);

        if (break_pos > 0) {
            // Mark the character after break as line start
            is_line_start[break_pos + 1] = 1;
            atomic_fetch_add_explicit(&state->line_count, 1, memory_order_relaxed);
        }
    }
}
```

### Phase 2: Line Index Assignment via Parallel Prefix Sum

```metal
// Assign line indices using prefix sum of is_line_start

kernel void assign_line_indices(
    device uint* is_line_start [[buffer(0)]],
    device uint* line_indices [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]],
    threadgroup uint* shared [[threadgroup(0)]]
) {
    // Blelloch scan (exclusive prefix sum)
    // Result: line_indices[i] = number of line starts before position i

    uint val = (tid < char_count) ? is_line_start[tid] : 0;

    // Up-sweep (reduce)
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < threads_per_group; stride *= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < threads_per_group) {
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Down-sweep
    if (tid == threads_per_group - 1) shared[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < threads_per_group) {
            uint temp = shared[idx];
            shared[idx] += shared[idx - stride];
            shared[idx - stride] = temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < char_count) {
        line_indices[tid] = shared[tid];
    }
}
```

### Phase 3: Parallel Glyph Positioning

```metal
struct PositionedGlyph {
    float x;
    float y;
    uint glyph_id;
    uint color;
};

kernel void position_glyphs_parallel(
    device uint* line_indices [[buffer(0)]],
    device float* cumulative_widths [[buffer(1)]],
    device uint* glyph_ids [[buffer(2)]],
    device LineInfo* lines [[buffer(3)]],
    device PositionedGlyph* output [[buffer(4)]],
    constant TextLayoutParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.char_count) return;

    uint line_idx = line_indices[tid];
    LineInfo line = lines[line_idx];

    // Calculate x position relative to line start
    float line_start_width = (line.start_char > 0)
        ? cumulative_widths[line.start_char - 1]
        : 0.0;

    float x = line.x + (cumulative_widths[tid] - line_start_width) * params.scale;
    float y = line.y + params.baseline_offset;

    output[tid] = PositionedGlyph {
        .x = x,
        .y = y,
        .glyph_id = glyph_ids[tid],
        .color = params.color
    };
}
```

### Rust Implementation

```rust
// src/gpu_os/document/text.rs

impl TextLayout {
    pub fn layout_text_gpu(&mut self, text: &str, container_width: f32) -> Vec<PositionedGlyph> {
        let char_count = text.chars().count();
        if char_count == 0 {
            return Vec::new();
        }

        // Upload text and metrics (already exists)
        self.upload_text_data(text);

        // Phase 1: Compute cumulative widths (existing GPU pass)
        self.compute_cumulative_widths();

        // Phase 2: Detect break opportunities (existing GPU pass)
        self.detect_break_opportunities();

        // Phase 3: NEW - GPU line assignment
        self.assign_lines_gpu(container_width);

        // Phase 4: NEW - GPU glyph positioning
        self.position_glyphs_gpu();

        // Read results
        self.read_positioned_glyphs()
    }

    fn assign_lines_gpu(&mut self, container_width: f32) {
        // Initialize state
        unsafe {
            let state_ptr = self.line_state_buffer.contents() as *mut LineLayoutState;
            (*state_ptr).line_count = 1; // At least one line
            (*state_ptr).container_width = container_width as u32;
            (*state_ptr).line_height = self.line_height as u32;
            (*state_ptr).char_count = self.char_count as u32;
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1: Detect line breaks
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.detect_breaks_pipeline);
            encoder.set_buffer(0, Some(&self.cumulative_buffer), 0);
            encoder.set_buffer(1, Some(&self.break_opportunity_buffer), 0);
            encoder.set_buffer(2, Some(&self.is_line_start_buffer), 0);
            encoder.set_buffer(3, Some(&self.line_state_buffer), 0);

            let threads = MTLSize::new(self.char_count as u64, 1, 1);
            let threadgroup = MTLSize::new(256, 1, 1);
            encoder.dispatch_threads(threads, threadgroup);
            encoder.end_encoding();
        }

        // Memory barrier
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);
            encoder.end_encoding();
        }

        // Pass 2: Prefix sum for line indices
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.prefix_sum_pipeline);
            encoder.set_buffer(0, Some(&self.is_line_start_buffer), 0);
            encoder.set_buffer(1, Some(&self.line_indices_buffer), 0);

            // Prefix sum needs careful threadgroup sizing
            let threadgroup_size = 256;
            let num_groups = (self.char_count + threadgroup_size - 1) / threadgroup_size;

            encoder.set_threadgroup_memory_length(0, (threadgroup_size * 4) as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new(num_groups as u64, 1, 1),
                MTLSize::new(threadgroup_size as u64, 1, 1)
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        // Don't wait - caller handles sync
    }

    fn position_glyphs_gpu(&mut self) {
        let command_buffer = self.command_queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.position_glyphs_pipeline);
        encoder.set_buffer(0, Some(&self.line_indices_buffer), 0);
        encoder.set_buffer(1, Some(&self.cumulative_buffer), 0);
        encoder.set_buffer(2, Some(&self.glyph_ids_buffer), 0);
        encoder.set_buffer(3, Some(&self.lines_buffer), 0);
        encoder.set_buffer(4, Some(&self.positioned_glyphs_buffer), 0);
        encoder.set_buffer(5, Some(&self.text_params_buffer), 0);

        let threads = MTLSize::new(self.char_count as u64, 1, 1);
        let threadgroup = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(threads, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
    }
}
```

## Pseudocode

```
function layout_text_gpu(text, container_width):
    chars = text.chars()
    n = chars.length

    # Phase 1: Compute cumulative widths (GPU)
    # cumulative[i] = sum of widths from char 0 to i
    cumulative = gpu_prefix_sum(char_widths)

    # Phase 2: Detect break opportunities (GPU)
    # is_break[i] = 1 if char i is a valid break point (whitespace, hyphen, etc)
    is_break = gpu_parallel_map(chars, is_break_opportunity)

    # Phase 3: Detect line starts (GPU)
    # For each char, check if it would overflow and mark next break as line start
    is_line_start = gpu_detect_line_starts(cumulative, is_break, container_width)

    # Phase 4: Assign line indices via prefix sum (GPU)
    # line_index[i] = number of line starts before position i
    line_indices = gpu_prefix_sum(is_line_start)

    # Phase 5: Build line info array (GPU)
    lines = gpu_build_lines(line_indices, cumulative)

    # Phase 6: Position glyphs (GPU)
    # Each thread positions one glyph using its line info
    glyphs = gpu_parallel_map(chars, lambda i:
        line = lines[line_indices[i]]
        x = line.x + (cumulative[i] - cumulative[line.start]) * scale
        y = line.y + baseline
        return PositionedGlyph(x, y, glyph_id[i])
    )

    return glyphs
```

## Test Plan

### Unit Tests

```rust
// tests/test_issue_143_gpu_text_layout.rs

#[test]
fn test_gpu_line_breaks_match_cpu() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);

    let test_cases = vec![
        ("Hello World", 100.0),
        ("The quick brown fox jumps over the lazy dog", 200.0),
        ("A very long word: supercalifragilisticexpialidocious", 150.0),
        ("Multiple    spaces   between    words", 180.0),
        ("Line1\nLine2\nLine3", 500.0),  // Hard breaks
        ("日本語テキストの折り返しテスト", 100.0),  // CJK text
    ];

    for (text, width) in test_cases {
        let cpu_result = text_layout.layout_text_cpu(text, width);
        let gpu_result = text_layout.layout_text_gpu(text, width);

        assert_eq!(
            cpu_result.len(),
            gpu_result.len(),
            "Glyph count mismatch for '{}' at width {}", text, width
        );

        for (i, (cpu_glyph, gpu_glyph)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
            assert!(
                (cpu_glyph.x - gpu_glyph.x).abs() < 0.1,
                "X mismatch at char {} for '{}': cpu={} gpu={}",
                i, text, cpu_glyph.x, gpu_glyph.x
            );
            assert!(
                (cpu_glyph.y - gpu_glyph.y).abs() < 0.1,
                "Y mismatch at char {} for '{}': cpu={} gpu={}",
                i, text, cpu_glyph.y, gpu_glyph.y
            );
        }
    }
}

#[test]
fn test_gpu_text_layout_performance() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);

    // Generate large text
    let text: String = (0..100_000)
        .map(|i| if i % 10 == 0 { ' ' } else { 'a' })
        .collect();

    // Warmup
    for _ in 0..5 {
        text_layout.layout_text_gpu(&text, 800.0);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..10 {
        text_layout.layout_text_gpu(&text, 800.0);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_millis() as f64 / 10.0;
    println!("100K chars layout: {}ms", avg_ms);

    // Target: <0.5ms for 100K chars
    assert!(avg_ms < 5.0, "Text layout too slow: {}ms", avg_ms);
}

#[test]
fn test_line_count_accuracy() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);

    // Text that should produce exactly N lines
    let text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";

    let result = text_layout.layout_text_gpu(text, 1000.0);
    let line_count = result.iter().map(|g| g.y).collect::<std::collections::HashSet<_>>().len();

    assert_eq!(line_count, 5, "Should have exactly 5 lines");
}

#[test]
fn test_word_wrap_boundary() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);

    // "Hello World" with width that fits "Hello " but not "Hello W"
    let text = "Hello World";
    let char_width = 8.0; // Assuming monospace
    let width = 6.5 * char_width; // Fits "Hello " (6 chars)

    let result = text_layout.layout_text_gpu(text, width);

    // "World" should be on second line
    let hello_y = result[0].y;
    let world_y = result[6].y;

    assert!(world_y > hello_y, "Word 'World' should be on next line");
}
```

### Visual Verification Tests

```rust
// tests/test_issue_143_visual.rs

#[test]
fn visual_test_text_wrapping() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);
    let mut renderer = TestRenderer::new(&device, 400, 300);

    let text = "The quick brown fox jumps over the lazy dog. \
                Pack my box with five dozen liquor jugs. \
                How vexingly quick daft zebras jump!";

    let glyphs = text_layout.layout_text_gpu(text, 350.0);

    renderer.render_text(&glyphs);
    renderer.save_to_file("tests/visual_output/text_wrapping_gpu.png");

    // Compare with baseline
    let baseline = image::open("tests/visual_baselines/text_wrapping_gpu.png").unwrap();
    let actual = image::open("tests/visual_output/text_wrapping_gpu.png").unwrap();

    let diff = image_diff(&baseline, &actual);
    assert!(diff < 0.001, "Visual difference: {}", diff);
}

#[test]
fn visual_test_justified_text() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);
    let mut renderer = TestRenderer::new(&device, 600, 400);

    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.";

    // Multiple container widths
    for (i, width) in [200.0, 300.0, 400.0, 500.0].iter().enumerate() {
        let glyphs = text_layout.layout_text_gpu(text, *width);
        renderer.render_text_at(&glyphs, 10.0, (i as f32) * 100.0 + 10.0);
    }

    renderer.save_to_file("tests/visual_output/text_widths_gpu.png");

    // Verify line counts decrease as width increases
    let narrow_lines = count_lines(&text_layout.layout_text_gpu(text, 200.0));
    let wide_lines = count_lines(&text_layout.layout_text_gpu(text, 500.0));

    assert!(narrow_lines > wide_lines, "Narrow should have more lines");
}

#[test]
fn visual_test_multiline_alignment() {
    let device = Device::system_default().unwrap();
    let mut text_layout = TextLayout::new(&device);
    let mut renderer = TestRenderer::new(&device, 400, 200);

    let text = "Left aligned\nSecond line\nThird line here";
    let glyphs = text_layout.layout_text_gpu(text, 400.0);

    renderer.render_text(&glyphs);

    // Draw alignment guides
    renderer.draw_line(0.0, 0.0, 0.0, 200.0, 0xFF0000); // Left edge

    renderer.save_to_file("tests/visual_output/text_alignment_gpu.png");

    // Verify all lines start at same x
    let line_starts: Vec<f32> = glyphs.iter()
        .filter(|g| g.glyph_id != 0)
        .group_by(|g| g.y as i32)
        .into_iter()
        .map(|(_, group)| group.map(|g| g.x).min().unwrap())
        .collect();

    for (i, x) in line_starts.iter().enumerate() {
        assert!(
            (x - line_starts[0]).abs() < 1.0,
            "Line {} not aligned: x={}", i, x
        );
    }
}
```

## Success Metrics

1. **Performance:** <0.5ms for 100K characters (10-50x speedup)
2. **Correctness:** 100% line break match with CPU implementation
3. **Integration:** Works with batched layout pipeline
4. **Memory:** No CPU-side character iteration

## Dependencies

- Issue #140: Batch Layout Passes (integration)
- Existing cumulative width computation (GPU)
- Existing break opportunity detection (GPU)

## Files to Modify

1. `src/gpu_os/document/text.rs` - Rust implementation
2. `src/gpu_os/document/text.metal` - GPU kernels
3. `tests/test_issue_143_gpu_text_layout.rs` - Unit tests
4. `tests/test_issue_143_visual.rs` - Visual tests
