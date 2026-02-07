# PRD: Issue #166 - Fix Clear Color Contrast

## Problem
The Metal clear color is the same as the compositor background color, making it impossible to see if the compositor is rendering.

## Root Cause

```rust
// Clear color in visual_megakernel.rs:
clear_color: MTLClearColor::new(0.08, 0.08, 0.12, 1.0)  // Dark gray

// Compositor background in gpu_os.rs:
state.background_color = [0.08, 0.08, 0.12, 1.0];  // Same dark gray!
```

## Solution

### Use Distinct Colors

```rust
// Clear color: BRIGHT MAGENTA (obvious debug color)
clear_color: MTLClearColor::new(1.0, 0.0, 1.0, 1.0)

// Compositor: Dark background (normal)
state.background_color = [0.08, 0.08, 0.12, 1.0];
```

When rendering works correctly:
- Magenta = no geometry rendered (bug)
- Dark gray = compositor rendered successfully

### Alternative: Gradient Background

Make compositor render a gradient instead of solid color to verify vertex generation:

```metal
// In compositor_update:
float gradient = vertex_x / screen_width;
vertex.color = float4(0.08 + gradient * 0.1, 0.08, 0.12, 1.0);
```

## Test Cases

1. Clear color differs from compositor background
2. If clear color visible, compositor didn't render
3. Visual test: run demo, should NOT see magenta

## Files to Modify

- `examples/visual_megakernel.rs` - Change clear color to magenta
