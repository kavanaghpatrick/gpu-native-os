# Bitmap Font Migration Plan

## Summary
Restore the working 8x8 bitmap font system from commit `cb0c0d4` to replace the broken SDF text rendering.

## Agent Findings Summary

### Files That Need Updates (4 files)
1. **`src/gpu_os/text_render.rs`** - Replace entirely with original bitmap version
2. **`examples/filesystem_browser.rs`** - Update render() call signature
3. **`examples/document_viewer.rs`** - Update render() call signature
4. **`examples/sdf_text_demo.rs`** - Update render() call signature OR remove

### Files That Can Be Removed/Ignored
- `examples/sdf_render_test.rs` - SDF diagnostic tool (created for debugging SDF)
- `examples/generate_sdf_atlas.rs` - SDF atlas generator (not needed for bitmap)

### Files That Stay Unchanged
- `src/gpu_os/text_editor.rs` - Has own embedded font in shader (independent)
- `examples/text_editor.rs` - Uses text_editor.rs (independent)
- `src/gpu_os/sdf_text/*` - Keep module but don't import from text_render.rs

---

## Critical API Differences

### render() Signature Change

**OLD (bitmap - WORKING):**
```rust
pub fn render(
    &mut self,
    encoder: &RenderCommandEncoderRef,  // Direct encoder
    font: &BitmapFont,                   // Font required
    screen_width: f32,
    screen_height: f32,
)
```

**NEW (SDF - BROKEN):**
```rust
pub fn render(
    &mut self,
    command_buffer: &CommandBufferRef,       // Command buffer
    render_pass_desc: &RenderPassDescriptorRef,  // Pass descriptor
    screen_width: f32,
    screen_height: f32,
)
```

### Method Differences

| Method | OLD (bitmap) | NEW (SDF) | Action |
|--------|--------------|-----------|--------|
| `add_text()` | ✓ | ✓ | Keep |
| `add_text_scaled()` | ✓ | ✓ | Keep |
| `add_text_sized()` | ✗ | ✓ | Add to OLD |
| `text_width()` | ✓ | ✓ | Keep |
| `line_height()` | ✓ | ✓ | Keep |
| `clear()` | ✓ | ✓ | Keep |
| `char_count()` | ✓ | ✓ | Keep |
| `segment_count()` | ✗ | ✓ | Add stub |
| `debug_dump_vertices()` | ✗ | ✓ | Add stub |
| `debug_atlas_info()` | ✗ | ✓ | Add stub |

---

## Migration Steps

### Step 1: Restore text_render.rs
```bash
git show cb0c0d4:src/gpu_os/text_render.rs > src/gpu_os/text_render.rs
```

Then add these methods for backwards compatibility:
- `add_text_sized()` - convert font_size to scale
- `segment_count()` - return 0 (stub)
- `debug_dump_vertices()` - no-op (stub)
- `debug_atlas_info()` - no-op (stub)

### Step 2: Update filesystem_browser.rs

**Current code (line ~431):**
```rust
self.text_renderer.render(&command_buffer, &render_desc, WIDTH as f32, HEIGHT as f32);
```

**Change to:**
```rust
let encoder = command_buffer.new_render_command_encoder(&render_desc);
self.text_renderer.render(&encoder, &self.font, WIDTH as f32, HEIGHT as f32);
encoder.end_encoding();
```

### Step 3: Update document_viewer.rs
Same pattern as filesystem_browser.rs

### Step 4: Update or Remove sdf_text_demo.rs
Either update to bitmap API or remove (it was for SDF testing)

### Step 5: Keep sdf_text module
The module can stay for future use, just don't import from text_render.rs

---

## Verification Checklist

- [ ] `cargo build --release` succeeds
- [ ] `cargo run --example filesystem_browser` shows readable text
- [ ] `cargo run --example text_editor` still works (independent system)
- [ ] `cargo test` passes (may need test updates)

---

## Rollback Plan

If migration fails:
```bash
git checkout HEAD -- src/gpu_os/text_render.rs
git checkout HEAD -- examples/filesystem_browser.rs
git checkout HEAD -- examples/document_viewer.rs
```
