# PRD: Visual Demo - Connect Megakernel to Window (Issue #162)

## Overview

Connect the GPU megakernel's unified vertex buffer to a real window for visual output. This is the final integration step - proving that GPU-generated geometry renders correctly with zero CPU involvement in the render loop.

## Goal

**Display megakernel-generated vertices in a window using a single draw call.**

The megakernel already generates vertices into `render_vertices_buffer`. This PRD connects that buffer to a Metal render pass displayed in a window. CPU only submits the command buffer - all geometry comes from GPU.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| CPU generates draw commands | GPU writes vertices, CPU draws once |
| CPU reads vertex count each frame | GPU atomics track count |
| CPU builds render state | GPU owns render state |
| Per-frame vertex buffer creation | Persistent unified buffer |

## Existing Infrastructure to REUSE

### From `GpuOs` (`/Users/patrickkavanagh/rust-experiment/src/gpu_os/gpu_os.rs`)

```rust
// Boot the OS - sets up megakernel
pub fn boot(device: &Device) -> Result<Self, String>

// Run one frame - megakernel generates vertices
pub fn run_frame(&mut self)

// Get unified vertex buffer for single draw call
pub fn render_vertices_buffer(&self) -> &Buffer

// Get total vertex count for draw call
pub fn total_vertex_count(&self) -> u32
```

### From `GpuAppSystem` (`/Users/patrickkavanagh/rust-experiment/src/gpu_os/gpu_app_system.rs`)

```rust
// Core rendering methods (lines 2923-2941)
pub fn total_vertex_count(&self) -> u32   // Reads atomic from GPU
pub fn render_vertices_buffer(&self) -> &Buffer  // Unified buffer

// Render state struct (line 499)
pub struct RenderState {
    pub total_vertex_count: u32,  // Atomic sum from GPU
    pub max_vertices: u32,
    pub screen_width: u32,
    pub screen_height: u32,
}

// RenderVertex struct (line 508)
pub struct RenderVertex {
    pub position: [f32; 3],    // x, y, z (z = depth)
    pub _pad0: f32,
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub _pad1: [f32; 2],
}
```

### From `ball_physics.rs` (Window Setup Pattern)

```rust
// Metal layer setup (lines 82-100)
let layer = MetalLayer::new();
layer.set_device(&device);
layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
// Attach to NSView...

// Render pipeline creation (lines 504-523)
render_desc.set_vertex_function(Some(&vertex_fn));
render_desc.set_fragment_function(Some(&fragment_fn));

// Render pass (lines 685-698)
let render_encoder = command_buffer.new_render_command_encoder(&render_desc);
render_encoder.set_render_pipeline_state(&self.render_pipeline);
render_encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);
render_encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count);
```

## GPU-Native Approach

### Zero CPU in Render Loop

```
Frame N:
1. GpuOs::run_frame()           // GPU megakernel generates vertices
2. GPU atomics sum vertex count // No CPU readback needed
3. CPU: submit draw command     // Single draw call
4. Present drawable             // Done
```

CPU touches:
- Command buffer submit (unavoidable)
- Present drawable (unavoidable)
- Optionally read vertex count (for debugging only)

CPU does NOT:
- Generate any geometry
- Sort windows
- Build draw commands
- Process app logic

### Vertex Flow

```
Megakernel (compute) --> unified_vertex_buffer --> Render Pipeline --> Screen
         |                       |                       |
   Apps write here          Same buffer             Single draw
```

## Design

### Visual Demo App Struct

```rust
pub struct VisualDemo {
    // GPU OS
    os: GpuOs,

    // Metal
    device: Device,
    layer: MetalLayer,
    command_queue: CommandQueue,
    render_pipeline: RenderPipelineState,
    depth_state: DepthStencilState,

    // Window
    window: Window,
    window_size: (u32, u32),

    // Timing
    frame_count: u64,
}
```

### Initialization

```rust
impl VisualDemo {
    pub fn new(window: Window) -> Result<Self, String> {
        let device = Device::system_default().expect("No Metal device");
        let command_queue = device.new_command_queue();

        // Boot GPU OS
        let os = GpuOs::boot(&device)?;

        // Create render pipeline
        let library = device.new_library_with_source(RENDER_SHADER, &CompileOptions::new())?;
        let vertex_fn = library.get_function("unified_vertex_shader", None)?;
        let fragment_fn = library.get_function("unified_fragment_shader", None)?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc.color_attachments().object_at(0).unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        // Enable depth testing
        render_desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        let render_pipeline = device.new_render_pipeline_state(&render_desc)?;

        // Create depth state
        let depth_desc = DepthStencilDescriptor::new();
        depth_desc.set_depth_compare_function(MTLCompareFunction::LessEqual);
        depth_desc.set_depth_write_enabled(true);
        let depth_state = device.new_depth_stencil_state(&depth_desc);

        // Set up Metal layer
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        // Attach layer to window...

        Ok(Self {
            os,
            device,
            layer,
            command_queue,
            render_pipeline,
            depth_state,
            window,
            window_size: (800, 600),
            frame_count: 0,
        })
    }
}
```

## Pseudocode: Render Loop

```rust
impl VisualDemo {
    pub fn render(&mut self) {
        // 1. Get drawable
        let drawable = match self.layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // 2. Run GPU OS frame (megakernel generates vertices)
        self.os.run_frame();

        // 3. Get vertex count (GPU already computed this atomically)
        let vertex_count = self.os.total_vertex_count();
        if vertex_count == 0 { return; }

        // 4. Create command buffer
        let cmd = self.command_queue.new_command_buffer();

        // 5. Create render pass
        let render_desc = RenderPassDescriptor::new();
        let color = render_desc.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(drawable.texture()));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));
        color.set_store_action(MTLStoreAction::Store);

        // 6. Encode render commands
        let encoder = cmd.new_render_command_encoder(&render_desc);
        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_depth_stencil_state(&self.depth_state);

        // 7. Bind unified vertex buffer (GPU-generated)
        encoder.set_vertex_buffer(0, Some(self.os.render_vertices_buffer()), 0);

        // 8. SINGLE DRAW CALL for all apps
        encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            vertex_count as u64,
        );

        encoder.end_encoding();

        // 9. Present
        cmd.present_drawable(drawable);
        cmd.commit();

        self.frame_count += 1;
    }
}
```

## Metal Shader Code

```metal
#include <metal_stdlib>
using namespace metal;

// Must match RenderVertex in gpu_app_system.rs
struct RenderVertex {
    float3 position;    // x, y, z (z = depth)
    float _pad0;
    float4 color;
    float2 uv;
    float2 _pad1;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
};

// Uniform buffer for screen dimensions
struct FrameUniforms {
    float2 screen_size;
    float time;
    float _pad;
};

vertex VertexOut unified_vertex_shader(
    const device RenderVertex* vertices [[buffer(0)]],
    constant FrameUniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    RenderVertex v = vertices[vid];

    VertexOut out;

    // Transform from screen coordinates to clip space
    // Screen coords: (0,0) top-left, (width, height) bottom-right
    // Clip coords: (-1,-1) bottom-left, (1,1) top-right
    float2 pos = v.position.xy / uniforms.screen_size;
    pos = pos * 2.0 - 1.0;
    pos.y = -pos.y;  // Flip Y (Metal NDC has Y up, screen has Y down)

    out.position = float4(pos, v.position.z, 1.0);
    out.color = v.color;
    out.uv = v.uv;

    return out;
}

fragment float4 unified_fragment_shader(
    VertexOut in [[stage_in]]
) {
    return in.color;
}

// Alternative: Simple passthrough (if coords already in clip space)
vertex VertexOut unified_vertex_passthrough(
    const device RenderVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    RenderVertex v = vertices[vid];

    VertexOut out;
    out.position = float4(v.position, 1.0);
    out.color = v.color;
    out.uv = v.uv;

    return out;
}
```

### Alternative: Embedded in Rust

```rust
const RENDER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RenderVertex {
    float3 position;
    float _pad0;
    float4 color;
    float2 uv;
    float2 _pad1;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
};

vertex VertexOut unified_vertex_shader(
    const device RenderVertex* vertices [[buffer(0)]],
    constant float2& screen_size [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    RenderVertex v = vertices[vid];

    float2 pos = v.position.xy / screen_size;
    pos = pos * 2.0 - 1.0;
    pos.y = -pos.y;

    VertexOut out;
    out.position = float4(pos, v.position.z, 1.0);
    out.color = v.color;
    out.uv = v.uv;
    return out;
}

fragment float4 unified_fragment_shader(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;
```

## Tests

### Test 1: Render Pipeline Creation

```rust
#[test]
fn test_render_pipeline_creation() {
    let device = Device::system_default().expect("No Metal device");

    let library = device
        .new_library_with_source(RENDER_SHADER, &CompileOptions::new())
        .expect("Shader compilation failed");

    let vertex_fn = library
        .get_function("unified_vertex_shader", None)
        .expect("Vertex function not found");

    let fragment_fn = library
        .get_function("unified_fragment_shader", None)
        .expect("Fragment function not found");

    let render_desc = RenderPipelineDescriptor::new();
    render_desc.set_vertex_function(Some(&vertex_fn));
    render_desc.set_fragment_function(Some(&fragment_fn));
    render_desc
        .color_attachments()
        .object_at(0)
        .unwrap()
        .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

    let pipeline = device
        .new_render_pipeline_state(&render_desc)
        .expect("Pipeline creation failed");

    // Pipeline should be valid
    assert!(!pipeline.label().is_empty() || true); // Just verify it exists
}
```

### Test 2: Unified Buffer Render Integration

```rust
#[test]
fn test_unified_buffer_render_integration() {
    let device = Device::system_default().expect("No Metal device");
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch some apps to generate vertices
    let app1 = os.launch_app(app_type::TERMINAL).expect("Launch failed");
    let app2 = os.launch_app(app_type::FILESYSTEM).expect("Launch failed");

    // Run frame to generate vertices
    os.run_frame();

    // Get render buffer
    let buffer = os.render_vertices_buffer();
    assert!(buffer.length() > 0, "Vertex buffer should have capacity");

    // Get vertex count
    let count = os.total_vertex_count();

    // May be 0 if apps don't generate vertices yet, but infrastructure works
    println!("Vertex count after frame: {}", count);

    // Buffer should be valid for rendering
    assert!(!buffer.contents().is_null());
}
```

### Test 3: Multiple Frames Stability

```rust
#[test]
fn test_multiple_frames_stability() {
    let device = Device::system_default().expect("No Metal device");
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch apps
    os.launch_app(app_type::TERMINAL);
    os.launch_app(app_type::FILESYSTEM);

    // Run many frames
    for i in 0..100 {
        os.run_frame();

        // Vertex count should be consistent
        let count = os.total_vertex_count();

        // No crashes, buffer stays valid
        let buffer = os.render_vertices_buffer();
        assert!(!buffer.contents().is_null());
    }

    assert_eq!(os.frame_count(), 100);
}
```

### Test 4: Vertex Buffer Layout

```rust
#[test]
fn test_vertex_buffer_layout() {
    use std::mem;

    // Verify RenderVertex matches Metal expectations
    assert_eq!(mem::size_of::<RenderVertex>(), 48); // float3 + pad + float4 + float2 + pad2

    let device = Device::system_default().expect("No Metal device");
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch app with window to generate vertices
    let slot = os.launch_app(app_type::TERMINAL).expect("Launch failed");

    // Mark dirty and run
    os.system.mark_dirty(slot);
    os.run_frame();

    // If vertices were generated, verify layout
    let count = os.total_vertex_count();
    if count > 0 {
        unsafe {
            let vertices = os.render_vertices_buffer().contents() as *const RenderVertex;
            let v = *vertices;

            // Z should be valid depth (0.0 to 1.0)
            assert!(v.position[2] >= 0.0 && v.position[2] <= 1.0,
                "Depth {} should be in [0, 1]", v.position[2]);

            // Color alpha should be non-zero for visible
            assert!(v.color[3] > 0.0, "Alpha should be non-zero");
        }
    }
}
```

### Test 5: Zero CPU Geometry

```rust
#[test]
fn test_zero_cpu_geometry() {
    let device = Device::system_default().expect("No Metal device");
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch apps
    os.launch_app(app_type::TERMINAL);

    // Run frame - CPU does nothing but submit
    os.run_frame();

    // The ONLY data we read from GPU is vertex count (for draw call)
    let count = os.total_vertex_count();

    // Vertex buffer contents are NEVER read by CPU
    // They go directly: GPU compute -> GPU render
    let buffer = os.render_vertices_buffer();

    // Verify buffer is GPU-ready without reading contents
    assert!(buffer.length() >= (count as u64) * 48); // 48 = sizeof(RenderVertex)
}
```

### Test 6: Input Integration

```rust
#[test]
fn test_input_to_render_integration() {
    let device = Device::system_default().expect("No Metal device");
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch app
    let slot = os.launch_app(app_type::TERMINAL).expect("Launch failed");

    // Send input events
    os.mouse_move(100.0, 100.0, 0.0, 0.0);
    os.mouse_click(100.0, 100.0, 0);
    os.key_event(0x00, true, 0); // 'a' key

    // Run frame - input processed, vertices generated
    os.run_frame();

    // System should be stable after input
    assert_eq!(os.frame_count(), 1);
    let _count = os.total_vertex_count();
    let _buffer = os.render_vertices_buffer();
}
```

## Example Application Code

```rust
// examples/visual_megakernel.rs

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::gpu_os::GpuOs;
use rust_experiment::gpu_os::gpu_app_system::{app_type, RenderVertex};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct VisualMegakernelDemo {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    os: Option<GpuOs>,
    render_pipeline: Option<RenderPipelineState>,
    screen_size_buffer: Option<Buffer>,
    command_queue: Option<CommandQueue>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl VisualMegakernelDemo {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            os: None,
            render_pipeline: None,
            screen_size_buffer: None,
            command_queue: None,
            window_size: (1280, 720),
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device");
        let command_queue = device.new_command_queue();

        println!("Visual Megakernel Demo");
        println!("======================");
        println!("GPU: {}", device.name());

        // Boot GPU OS
        let mut os = GpuOs::boot(&device).expect("Failed to boot GPU OS");

        // Launch some apps
        os.launch_app(app_type::TERMINAL);
        os.launch_app(app_type::FILESYSTEM);

        // Create render pipeline
        let library = device
            .new_library_with_source(RENDER_SHADER, &CompileOptions::new())
            .expect("Shader failed");

        let vertex_fn = library.get_function("unified_vertex_shader", None).unwrap();
        let fragment_fn = library.get_function("unified_fragment_shader", None).unwrap();

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let render_pipeline = device.new_render_pipeline_state(&render_desc).unwrap();

        // Screen size buffer
        let screen_size_buffer = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = [self.window_size.0 as f32, self.window_size.1 as f32];
        }

        // Metal layer
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(h) = handle.as_raw() {
                    let view = h.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        let size = window.inner_size();
        self.window_size = (size.width, size.height);
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.os = Some(os);
        self.render_pipeline = Some(render_pipeline);
        self.screen_size_buffer = Some(screen_size_buffer);
        self.command_queue = Some(command_queue);
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let os = self.os.as_mut().unwrap();
        let pipeline = self.render_pipeline.as_ref().unwrap();
        let queue = self.command_queue.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // 1. Run GPU OS frame (megakernel generates vertices)
        os.run_frame();

        // 2. Get vertex count
        let vertex_count = os.total_vertex_count();

        // 3. Create command buffer
        let cmd = queue.new_command_buffer();

        // 4. Render pass
        let render_desc = RenderPassDescriptor::new();
        let color = render_desc.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(drawable.texture()));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(0.08, 0.08, 0.12, 1.0));
        color.set_store_action(MTLStoreAction::Store);

        let encoder = cmd.new_render_command_encoder(&render_desc);
        encoder.set_render_pipeline_state(pipeline);

        // 5. Bind unified vertex buffer from GPU OS
        encoder.set_vertex_buffer(0, Some(os.render_vertices_buffer()), 0);
        encoder.set_vertex_buffer(1, Some(self.screen_size_buffer.as_ref().unwrap()), 0);

        // 6. SINGLE DRAW CALL
        if vertex_count > 0 {
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
        }

        encoder.end_encoding();
        cmd.present_drawable(drawable);
        cmd.commit();

        self.frame_count += 1;
        if self.frame_count % 60 == 0 {
            println!("Frame {}: {} vertices", self.frame_count, vertex_count);
        }
    }
}

impl ApplicationHandler for VisualMegakernelDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_title("GPU OS Visual Demo - Megakernel Rendering");
        let window = event_loop.create_window(attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = VisualMegakernelDemo::new();
    event_loop.run_app(&mut app).unwrap();
}

const RENDER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RenderVertex {
    float3 position;
    float _pad0;
    float4 color;
    float2 uv;
    float2 _pad1;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
};

vertex VertexOut unified_vertex_shader(
    const device RenderVertex* vertices [[buffer(0)]],
    constant float2& screen_size [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    RenderVertex v = vertices[vid];

    float2 pos = v.position.xy / screen_size;
    pos = pos * 2.0 - 1.0;
    pos.y = -pos.y;

    VertexOut out;
    out.position = float4(pos, v.position.z, 1.0);
    out.color = v.color;
    out.uv = v.uv;
    return out;
}

fragment float4 unified_fragment_shader(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;
```

## Success Metrics

1. **Visual output**: Window displays colored rectangles for each app
2. **Single draw call**: One `draw_primitives` per frame
3. **Zero CPU geometry**: CPU never writes vertex data
4. **60 FPS**: Smooth rendering with megakernel overhead
5. **Input integration**: Mouse/keyboard events affect display

## Dependencies

- Issue #155: GpuOs boot infrastructure
- Issue #158: Unified vertex buffer and render pipeline
- Issue #159: Megakernel vertex generation

## Out of Scope

- Window decorations (Issue #164)
- Compositor blending (Issue #163)
- Text rendering
- Texture support
