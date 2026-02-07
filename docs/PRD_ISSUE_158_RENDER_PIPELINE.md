# PRD: GPU Rendering Pipeline (Issue #158)

## Overview

Implement GPU-driven rendering where apps write vertices with depth and hardware z-buffer handles occlusion - no sorting needed.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| Sort windows by z-order | Each app writes depth, hardware sorts |
| Build draw command list | Each app writes to unified buffer |
| Sequential render loop | One draw call, depth test handles order |
| CPU encodes per-app | GPU writes vertices directly |

## The GPU Insight

The z-buffer IS the sorter:

```metal
// WRONG: Sort then render
sort_windows_by_z();  // O(n log n)
for each window in sorted_order:
    draw(window);

// RIGHT: Render with depth
// Each app writes depth from its z_order - O(1)
vertex.z = my_window.depth;
// Hardware depth test handles occlusion automatically
```

## Design

### Unified Vertex Buffer with Depth

```metal
struct Vertex {
    float3 position;    // x, y, z (z = depth from window)
    float4 color;
    float2 uv;
    float2 _pad;
};

// Each app writes vertices to its region with correct depth
kernel void gpu_app_megakernel(
    device GpuAppDescriptor* apps [[buffer(0)]],
    device Vertex* unified_vertices [[buffer(1)]],
    device GpuWindow* windows [[buffer(2)]],
    uint slot_id [[thread_position_in_grid]]
) {
    device GpuAppDescriptor* app = &apps[slot_id];
    if (!should_i_run(app)) return;

    // Get my depth from window
    float my_depth = windows[app->window_id].depth;

    // Write vertices to my region
    device Vertex* my_vertices = unified_vertices + (app->vertex_offset / sizeof(Vertex));

    // App-specific vertex generation
    switch (app->app_type) {
        case APP_TYPE_GAME_OF_LIFE:
            generate_grid_vertices(app, my_vertices, my_depth);
            break;
        // ... other apps
    }
}

inline void write_quad(
    device Vertex* v,
    float2 pos,
    float2 size,
    float depth,
    float4 color
) {
    // All vertices at same depth
    v[0].position = float3(pos.x, pos.y, depth);
    v[1].position = float3(pos.x + size.x, pos.y, depth);
    v[2].position = float3(pos.x + size.x, pos.y + size.y, depth);
    v[3].position = float3(pos.x, pos.y, depth);
    v[4].position = float3(pos.x + size.x, pos.y + size.y, depth);
    v[5].position = float3(pos.x, pos.y + size.y, depth);
    // Set colors, UVs...
}
```

### Single Draw Call

No per-app draw commands - one draw covers everything:

```metal
// Vertex shader transforms and clips to window
vertex VertexOut render_vertex(
    const device Vertex* vertices [[buffer(0)]],
    constant FrameState& frame [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    Vertex v = vertices[vid];

    VertexOut out;
    // Transform to clip space
    out.position = float4(
        (v.position.x / frame.screen_width) * 2.0 - 1.0,
        1.0 - (v.position.y / frame.screen_height) * 2.0,
        v.position.z,  // Depth preserved
        1.0
    );
    out.color = v.color;
    out.uv = v.uv;

    return out;
}
```

### Vertex Count Tracking

Each app atomically reports its vertex count:

```metal
struct RenderState {
    atomic_uint total_vertex_count;
    uint max_vertices;
};

kernel void finalize_render(
    device const GpuAppDescriptor* apps [[buffer(0)]],
    device RenderState* render [[buffer(1)]],
    constant uint& max_slots [[buffer(2)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    GpuAppDescriptor app = apps[slot_id];
    if (!(app.flags & APP_FLAG_ACTIVE)) return;
    if (!(app.flags & APP_FLAG_VISIBLE)) return;

    // Add my vertex count to total
    atomic_fetch_add_explicit(
        &render->total_vertex_count,
        app.vertex_count,
        memory_order_relaxed
    );
}
```

### CPU Rendering (Minimal)

```rust
fn render_frame(&mut self, encoder: &RenderCommandEncoderRef) {
    // Configure depth test
    encoder.set_depth_stencil_state(&self.depth_state);

    // Bind unified vertex buffer
    encoder.set_vertex_buffer(0, Some(&self.unified_vertex_buffer), 0);
    encoder.set_vertex_buffer(1, Some(&self.frame_state_buffer), 0);

    // One draw call for ALL apps
    let total_vertices = self.read_total_vertex_count();
    encoder.draw_primitives(
        MTLPrimitiveType::Triangle,
        0,
        total_vertices as u64,
    );
}
```

### Window Clipping (Optional Enhancement)

For apps that extend beyond their window, use scissor:

```metal
// Per-app scissor encoded in vertex attributes
struct VertexOut {
    float4 position [[position]];
    float4 color;
    float4 scissor;  // Window bounds for fragment shader clipping
};

fragment float4 render_fragment(
    VertexOut in [[stage_in]]
) {
    // Discard fragments outside window
    if (in.position.x < in.scissor.x ||
        in.position.x > in.scissor.x + in.scissor.z ||
        in.position.y < in.scissor.y ||
        in.position.y > in.scissor.y + in.scissor.w) {
        discard_fragment();
    }

    return in.color;
}
```

## Why No Indirect Command Buffers?

ICBs add complexity without benefit here:

| ICB Approach | Single Draw Approach |
|--------------|---------------------|
| GPU encodes draw commands | GPU writes vertices |
| Multiple draws per app | One draw for all |
| More GPU work | Less GPU work |
| Metal 3 only | Works on Metal 2 |

The unified vertex buffer approach is simpler and faster.

## Implementation

### Rust API

```rust
impl GpuAppSystem {
    /// Run megakernel (apps generate vertices)
    pub fn run_frame(&mut self) {
        // Run megakernel
        // Run finalize_render to sum vertex counts
    }

    /// Get total vertex count for draw call
    pub fn total_vertex_count(&self) -> u32 {
        // Read from render state
    }

    /// Render all apps (single draw call)
    pub fn render(&self, encoder: &RenderCommandEncoderRef) {
        encoder.set_depth_stencil_state(&self.depth_state);
        encoder.set_vertex_buffer(0, Some(&self.unified_vertex_buffer), 0);
        encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            self.total_vertex_count() as u64,
        );
    }
}
```

### Depth State Configuration

```rust
fn create_depth_state(device: &Device) -> DepthStencilState {
    let desc = DepthStencilDescriptor::new();
    desc.set_depth_compare_function(MTLCompareFunction::LessEqual);
    desc.set_depth_write_enabled(true);
    device.new_depth_stencil_state(&desc)
}
```

## Tests

```rust
#[test]
fn test_unified_vertex_buffer() {
    let mut system = GpuAppSystem::new(&device)?;

    let app1 = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();
    let app2 = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();

    system.create_window(app1, Rect::new(0.0, 0.0, 100.0, 100.0));
    system.create_window(app2, Rect::new(100.0, 0.0, 100.0, 100.0));

    system.run_frame();

    // Both apps should have vertices
    assert!(system.get_app(app1).unwrap().vertex_count > 0);
    assert!(system.get_app(app2).unwrap().vertex_count > 0);

    // Total should be sum
    let total = system.total_vertex_count();
    let sum = system.get_app(app1).unwrap().vertex_count +
              system.get_app(app2).unwrap().vertex_count;
    assert_eq!(total, sum);
}

#[test]
fn test_depth_ordering() {
    let mut system = GpuAppSystem::new(&device)?;

    // Create overlapping windows
    let back = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();
    system.create_window(back, Rect::new(0.0, 0.0, 100.0, 100.0));

    let front = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();
    system.create_window(front, Rect::new(50.0, 0.0, 100.0, 100.0));
    system.set_focus(front);  // Brings to front

    system.run_frame();

    // Front window should have higher depth
    let back_depth = system.get_window(back).unwrap().depth;
    let front_depth = system.get_window(front).unwrap().depth;
    assert!(front_depth > back_depth);
}

#[test]
fn test_single_draw_call() {
    let mut system = GpuAppSystem::new(&device)?;

    // Launch many apps
    for _ in 0..20 {
        let app = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();
        system.create_window(app, Rect::new(0.0, 0.0, 50.0, 50.0));
    }

    system.run_frame();

    // Total vertex count should be sum of all apps
    let total = system.total_vertex_count();
    assert!(total > 0);
    assert_eq!(total, 20 * 6);  // 20 apps × 6 vertices per quad
}
```

## Success Metrics

1. **Draw calls**: 1 (single draw for all apps)
2. **No sorting**: Hardware depth test handles z-order
3. **Render time**: O(1) GPU work per app (parallel vertex generation)
4. **Memory efficiency**: Unified buffer, no per-app allocations
