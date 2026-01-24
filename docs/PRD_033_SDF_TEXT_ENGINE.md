# PRD #33: SDF Text Engine (GPU-Native)

## Overview
Replace the current 8x8 bitmap font with a GPU-native Signed Distance Field (SDF) text rendering engine that provides resolution-independent, scalable text.

**CRITICAL**: ALL computation happens on GPU. No CPU fallbacks.

## Current State
- `text_render.rs`: 8x8 bitmap font with hardcoded ASCII glyphs
- `text.rs`: MSDF shader infrastructure with placeholder atlas (no real font data)

## Implementation Phases

---

## Phase 1: Font Loading with ttf-parser ✅ COMPLETE
**Goal**: Load TrueType fonts and extract glyph outlines

### Deliverables
1. Add `ttf-parser` dependency to Cargo.toml
2. Create `SdfFont` struct that loads .ttf files
3. Extract glyph outlines (bezier curves) for ASCII 32-126
4. Extract font metrics (units per em, ascender, descender)

**Note**: Font parsing is acceptable on CPU - it's a one-time operation at startup. The font file format (TTF/OTF) requires sequential parsing that doesn't benefit from GPU parallelism.

### Tests (all passing)
```
test_ttf_parser_loads_system_font
test_font_metrics_valid
test_glyph_outline_extraction
test_ascii_coverage
test_glyph_advance_values
test_path_segment_types
test_scale_to_pixels
test_cached_glyph_count
```

---

## Phase 2: GPU SDF Generation (Compute Shader)
**Goal**: Generate SDF textures from glyph outlines entirely on GPU

### Why GPU?
- Each pixel's distance calculation is independent (embarrassingly parallel)
- 64x64 SDF = 4096 pixels × 95 glyphs = 389,120 parallel computations
- GPU can process all glyphs simultaneously
- No CPU-GPU data transfer bottleneck

### Deliverables
1. Upload glyph path data to GPU buffer (flattened line segments)
2. Compute shader calculates signed distance for each pixel
3. Inside/outside determination via winding number
4. Output directly to atlas texture (no CPU readback)

### Algorithm (GPU Compute Shader)
```metal
kernel void generate_sdf(
    device const PathSegment* segments [[buffer(0)]],
    device const GlyphInfo* glyph_info [[buffer(1)]],
    texture2d<float, access::write> atlas [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // 1. Map pixel to glyph space
    float2 pixel_pos = map_to_glyph_space(gid, glyph_info);

    // 2. Find minimum distance to all segments (parallel reduction)
    float min_dist = INFINITY;
    for (uint i = glyph_info->segment_start; i < glyph_info->segment_end; i++) {
        float d = distance_to_segment(pixel_pos, segments[i]);
        min_dist = min(min_dist, d);
    }

    // 3. Determine inside/outside via winding number
    int winding = compute_winding(pixel_pos, segments, glyph_info);
    float sign = (winding != 0) ? -1.0 : 1.0;

    // 4. Write signed distance to atlas
    atlas.write(sign * min_dist, gid);
}
```

### Data Structures
```rust
// Uploaded to GPU buffer
#[repr(C)]
struct GpuPathSegment {
    segment_type: u32,  // 0=line, 1=quad, 2=cubic
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],  // unused for lines
    p3: [f32; 2],  // unused for lines/quads
}

#[repr(C)]
struct GpuGlyphInfo {
    atlas_x: u32,
    atlas_y: u32,
    sdf_size: u32,
    segment_start: u32,
    segment_end: u32,
    bounds: [f32; 4],  // glyph bounds for coordinate mapping
}
```

### Tests
```
test_gpu_sdf_generator_creates_texture
test_gpu_sdf_distance_at_edge_is_zero
test_gpu_sdf_inside_is_negative
test_gpu_sdf_outside_is_positive
test_gpu_sdf_all_ascii_generated
```

### Acceptance Criteria
- [ ] All SDF generation runs on GPU compute shader
- [ ] Generate valid SDF for all 95 ASCII glyphs in single dispatch
- [ ] No CPU readback of distance values
- [ ] Distance field is continuous (no sharp discontinuities)

---

## Phase 3: GPU Atlas Management
**Goal**: Pack and manage glyph SDFs in GPU texture atlas

### Why GPU?
- Atlas packing can use GPU sorting for optimal placement
- Texture updates happen directly on GPU
- No CPU-GPU texture upload bottleneck

### Deliverables
1. GPU-side atlas allocator (simple row packing)
2. Track UV coordinates in GPU buffer
3. Glyph lookup table in GPU buffer

### Tests
```
test_atlas_packs_multiple_glyphs
test_atlas_uvs_are_valid
test_atlas_no_overlap
test_atlas_full_ascii_fits
```

---

## Phase 4: GPU SDF Rendering
**Goal**: Render text using SDF atlas with anti-aliased edges

### Shader
```metal
fragment float4 sdf_text_fragment(
    VertexOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]]
) {
    float distance = atlas.sample(samp, in.uv).r;

    // SDF threshold with screen-space anti-aliasing
    float edge = 0.5;
    float width = fwidth(distance) * 0.75;
    float alpha = smoothstep(edge - width, edge + width, distance);

    return float4(in.color.rgb, in.color.a * alpha);
}
```

### Tests
```
test_sdf_render_creates_vertices
test_sdf_render_different_sizes
test_sdf_render_performance_10k_glyphs
```

---

## Phase 5: GPU Text Layout
**Goal**: Text measurement and layout on GPU

### Why GPU?
- Text layout is per-glyph operations (parallel)
- Kerning lookup is table-based (GPU-friendly)
- Can layout multiple text blocks simultaneously

### Deliverables
1. GPU compute shader for text layout
2. Output glyph instances directly to vertex buffer
3. Measure text via parallel reduction

### Tests
```
test_text_measurement_accuracy
test_gpu_layout_line
test_gpu_layout_multiline
```

---

## Phase 6: Integration & Demo Update
**Goal**: Replace old text system, update all demos

### Deliverables
1. Deprecate `text_render.rs` (bitmap font)
2. Update `text.rs` to use new SDF system
3. Update all examples to use new API
4. Visual comparison demo

---

## Test Commands

```bash
# Run all tests
cargo test --test test_issue_33_sdf_text

# Run specific phase
cargo test --test test_issue_33_sdf_text phase1
cargo test --test test_issue_33_sdf_text phase2

# Visual demo
cargo run --release --example sdf_text_demo
```

---

## GPU-Native Philosophy

| Operation | GPU? | Rationale |
|-----------|------|-----------|
| Font file parsing | CPU | One-time, sequential format |
| SDF generation | **GPU** | Embarrassingly parallel |
| Atlas packing | **GPU** | Parallel sorting/placement |
| Atlas texture | **GPU** | No CPU upload |
| Text layout | **GPU** | Per-glyph parallel |
| Text rendering | **GPU** | Fragment shader |

The only CPU work is the initial font file parsing, which happens once at load time.

---

## File Structure

```
src/gpu_os/
├── sdf_text/
│   ├── mod.rs           # Module exports
│   ├── font.rs          # SdfFont - TTF loading (CPU, one-time)
│   ├── gpu_generator.rs # GPU SDF compute shader
│   ├── atlas.rs         # GPU atlas management
│   ├── renderer.rs      # GPU rendering
│   └── shaders.metal    # All SDF shaders

tests/
└── test_issue_33_sdf_text.rs

examples/
└── sdf_text_demo.rs
```
