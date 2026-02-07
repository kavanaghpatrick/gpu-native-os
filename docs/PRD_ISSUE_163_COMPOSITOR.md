# PRD: GPU Compositor Integration (Issue #163)

## Overview

The compositor becomes the final rendering stage that reads from the unified vertex buffer and renders to the screen. It's a system app that runs LAST each frame.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| CPU sorts windows by z | Each vertex has depth, hardware z-sorts |
| CPU issues draw per window | Single draw call for all |
| CPU manages render targets | Unified vertex buffer |
| Multiple render passes | One pass with depth test |

## The GPU Insight

The compositor doesn't need to be a separate "compositing" step. With depth values in vertices:

```
All apps write vertices → Unified buffer → Single draw → Hardware z-test → Done
```

The "compositor" app just manages the final render setup and any post-processing effects.

## Design

### Compositor State

```metal
struct CompositorState {
    float screen_width;
    float screen_height;
    uint window_count;
    uint focused_window;

    // Post-processing
    float blur_radius;       // For unfocused windows
    float shadow_opacity;
    float shadow_offset_x;
    float shadow_offset_y;

    // Colors
    float4 background_color;

    // Stats
    uint total_vertices_rendered;
    uint frame_number;
    uint _pad[2];
};
```

### Compositor as Final Stage

The compositor doesn't generate its own vertices - it orchestrates the final render:

```rust
impl GpuOs {
    pub fn render_frame(&mut self, drawable: &MetalDrawableRef) {
        // 1. Run megakernel (all apps update, generate vertices)
        self.system.run_frame();

        // 2. Finalize render (sum vertex counts)
        self.system.finalize_render();

        // 3. Single draw call with depth test
        let encoder = self.command_buffer.new_render_command_encoder(&self.render_pass);

        // Background
        encoder.set_render_pipeline_state(&self.background_pipeline);
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);

        // All app content in one draw
        encoder.set_render_pipeline_state(&self.content_pipeline);
        encoder.set_depth_stencil_state(&self.depth_state);
        encoder.set_vertex_buffer(0, Some(self.system.render_vertices_buffer()), 0);
        encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            self.system.total_vertex_count() as u64,
        );

        encoder.end_encoding();
        self.command_buffer.present_drawable(drawable);
        self.command_buffer.commit();
    }
}
```

### Depth Configuration

```rust
fn create_depth_state(device: &Device) -> DepthStencilState {
    let desc = DepthStencilDescriptor::new();
    desc.set_depth_compare_function(MTLCompareFunction::LessEqual);
    desc.set_depth_write_enabled(true);
    device.new_depth_stencil_state(&desc)
}
```

### Vertex Shader (transforms + clips)

```metal
vertex VertexOut compositor_vertex(
    const device RenderVertex* vertices [[buffer(0)]],
    constant CompositorState& state [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    RenderVertex v = vertices[vid];

    VertexOut out;
    // Transform to clip space [-1, 1]
    out.position = float4(
        (v.position.x / state.screen_width) * 2.0 - 1.0,
        1.0 - (v.position.y / state.screen_height) * 2.0,
        v.position.z,  // Depth preserved
        1.0
    );
    out.color = v.color;
    out.uv = v.uv;

    return out;
}
```

### Fragment Shader

```metal
fragment float4 compositor_fragment(
    VertexOut in [[stage_in]],
    constant CompositorState& state [[buffer(0)]]
) {
    // Simple colored output (textures would be sampled here)
    return in.color;
}
```

### Window Shadows (Optional Enhancement)

Shadows can be rendered as additional vertices with low alpha:

```metal
// In each app's vertex generation, prepend shadow quad
inline void write_window_with_shadow(
    device RenderVertex* verts,
    float2 pos,
    float2 size,
    float depth,
    float4 content_color,
    float shadow_offset,
    float shadow_opacity
) {
    // Shadow (slightly offset, lower depth, transparent black)
    write_quad(verts,
        float2(pos.x + shadow_offset, pos.y + shadow_offset),
        size,
        depth - 0.001,  // Behind content
        float4(0, 0, 0, shadow_opacity)
    );

    // Content
    write_quad(verts + 6, pos, size, depth, content_color);
}
```

## Integration

### Render Pipeline Setup

```rust
impl GpuOs {
    fn setup_render_pipeline(device: &Device) -> Result<RenderPipelineState, String> {
        let desc = RenderPipelineDescriptor::new();

        let library = device.new_library_with_source(COMPOSITOR_SHADER, &CompileOptions::new())?;
        desc.set_vertex_function(Some(&library.get_function("compositor_vertex", None)?));
        desc.set_fragment_function(Some(&library.get_function("compositor_fragment", None)?));

        // Color attachment
        let color = desc.color_attachments().object_at(0).unwrap();
        color.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        color.set_blending_enabled(true);
        color.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        // Depth attachment
        desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        device.new_render_pipeline_state(&desc)
    }
}
```

## Tests

```rust
#[test]
fn test_single_draw_call() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    // Launch multiple apps
    os.launch_app(app_type::TERMINAL);
    os.launch_app(app_type::FILESYSTEM);
    os.launch_app(app_type::TERMINAL);

    os.system.run_frame();
    os.system.finalize_render();

    // Should have vertices from all apps + system apps
    let total = os.system.total_vertex_count();
    assert!(total > 0);
    println!("Total vertices for single draw: {}", total);
}

#[test]
fn test_depth_ordering() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    let back = os.launch_app(app_type::TERMINAL).unwrap();
    let front = os.launch_app(app_type::TERMINAL).unwrap();

    // Front window should have higher depth
    os.system.set_focus(front);

    os.system.run_frame();

    let back_window = os.system.get_window(back).unwrap();
    let front_window = os.system.get_window(front).unwrap();

    assert!(front_window.depth > back_window.depth);
}

#[test]
fn test_render_frame_completes() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.launch_app(app_type::TERMINAL);

    // This would normally render to a drawable
    // For testing, we just verify the frame completes
    os.system.run_frame();
    os.system.finalize_render();

    let total = os.system.total_vertex_count();
    assert!(total > 0);
}
```

## Benchmarks

```rust
#[test]
fn bench_full_frame() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    // Typical desktop: 5 windows
    for _ in 0..5 {
        os.launch_app(app_type::TERMINAL);
    }

    // Warm up
    for _ in 0..10 {
        os.system.mark_all_dirty();
        os.system.run_frame();
        os.system.finalize_render();
    }

    // Benchmark full frame
    let start = Instant::now();
    for _ in 0..1000 {
        os.system.mark_all_dirty();
        os.system.run_frame();
        os.system.finalize_render();
    }
    let duration = start.elapsed();

    let per_frame = duration.as_micros() / 1000;
    println!("Full frame (5 windows): {}us", per_frame);
    assert!(per_frame < 1000, "Full frame should be <1ms");
}

#[test]
fn bench_vertex_count_scaling() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    for window_count in [1, 5, 10, 20] {
        // Reset
        let mut os = GpuOs::boot(&device).unwrap();

        for _ in 0..window_count {
            os.launch_app(app_type::TERMINAL);
        }

        os.system.mark_all_dirty();
        os.system.run_frame();
        os.system.finalize_render();

        println!("{} windows: {} vertices",
            window_count, os.system.total_vertex_count());
    }
}
```

## Success Metrics

1. **Single draw call**: All content in one draw
2. **Frame time**: < 1ms for 10 windows
3. **GPU utilization**: > 80% (efficient batching)
4. **No CPU sorting**: Hardware depth test handles z-order
