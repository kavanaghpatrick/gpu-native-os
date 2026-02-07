# PRD: Issue #164 - Fix Vertex Buffer Alignment

## Problem
The unified vertex buffer allocates 16 bytes per vertex but RenderVertex is 48 bytes, causing memory corruption and incorrect rendering.

## Root Cause

```rust
// Current (WRONG):
const VERTEX_SIZE: usize = 16;  // Only accounts for position

// RenderVertex actual size:
#[repr(C)]
struct RenderVertex {
    position: [f32; 3],    // 12 bytes
    _pad0: f32,            // 4 bytes  (16 total)
    color: [f32; 4],       // 16 bytes (32 total)
    uv: [f32; 2],          // 8 bytes  (40 total)
    _pad1: [f32; 2],       // 8 bytes  (48 total)
}
```

## Solution

### Fix Buffer Allocation

```rust
// In gpu_app_system.rs:
const RENDER_VERTEX_SIZE: usize = 48;  // Correct size
const MAX_VERTICES_PER_APP: usize = 65536;

// Buffer size calculation:
let unified_buffer_size = MAX_APPS * MAX_VERTICES_PER_APP * RENDER_VERTEX_SIZE;
```

### Fix Offset Calculation

```rust
// Vertex offset for each app:
fn vertex_offset_for_app(app_slot: u32) -> u32 {
    app_slot * MAX_VERTICES_PER_APP as u32
}

// Byte offset for buffer access:
fn byte_offset_for_app(app_slot: u32) -> usize {
    app_slot as usize * MAX_VERTICES_PER_APP * RENDER_VERTEX_SIZE
}
```

### Metal Shader Fix

```metal
// Ensure GPU uses correct stride:
kernel void app_update(...) {
    uint vertex_base = app.vertex_offset;
    uint byte_offset = vertex_base * 48;  // 48 bytes per vertex

    // Write vertex at correct location
    device RenderVertex* vertex = (device RenderVertex*)((device char*)unified_vertices + byte_offset);
}
```

## Test Cases

1. `size_of::<RenderVertex>() == 48`
2. Buffer allocation uses 48-byte stride
3. App vertex offsets are multiples of MAX_VERTICES_PER_APP
4. No memory overlap between apps

## Files to Modify

- `src/gpu_os/gpu_app_system.rs` - Fix VERTEX_SIZE constant and allocation
