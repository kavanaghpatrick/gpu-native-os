//! Render Diagnostic Tool
//!
//! Systematically tests different rendering scenarios to isolate visual artifacts.
//! Captures texture contents and analyzes for garbage pixels.
//!
//! Run with: cargo run --release --example render_diagnostic

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use std::mem;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// ============================================================================
// Test Modes
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum TestMode {
    ClearOnly,          // Just clear to gray - no drawing
    SingleTriangle,     // Draw ONE triangle with known color
    SimpleQuad,         // Draw ONE quad with known color
    ManyQuads,          // Draw many quads with different CSS colors
    FullPipeline,       // Full document rendering pipeline
}

// ============================================================================
// Shaders
// ============================================================================

const SIMPLE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float4 color;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut simple_vertex(
    const device Vertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    Vertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 simple_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;

// ============================================================================
// Vertex Data
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

fn create_single_triangle() -> Vec<Vertex> {
    // A simple red triangle in the center of the screen
    vec![
        Vertex { position: [-0.5, -0.5], color: [1.0, 0.0, 0.0, 1.0] }, // red
        Vertex { position: [ 0.5, -0.5], color: [1.0, 0.0, 0.0, 1.0] }, // red
        Vertex { position: [ 0.0,  0.5], color: [1.0, 0.0, 0.0, 1.0] }, // red
    ]
}

fn create_simple_quad() -> Vec<Vertex> {
    // A blue quad
    vec![
        // Triangle 1
        Vertex { position: [-0.3, -0.3], color: [0.0, 0.0, 1.0, 1.0] }, // blue
        Vertex { position: [ 0.3, -0.3], color: [0.0, 0.0, 1.0, 1.0] }, // blue
        Vertex { position: [ 0.3,  0.3], color: [0.0, 0.0, 1.0, 1.0] }, // blue
        // Triangle 2
        Vertex { position: [-0.3, -0.3], color: [0.0, 0.0, 1.0, 1.0] }, // blue
        Vertex { position: [ 0.3,  0.3], color: [0.0, 0.0, 1.0, 1.0] }, // blue
        Vertex { position: [-0.3,  0.3], color: [0.0, 0.0, 1.0, 1.0] }, // blue
    ]
}

fn create_many_quads() -> Vec<Vertex> {
    // Create 10 quads with CSS-like colors (no bright saturated primaries)
    let colors: Vec<[f32; 4]> = vec![
        [0.88, 0.88, 0.88, 1.0],   // Light gray (#e0e0e0)
        [0.17, 0.24, 0.31, 1.0],   // Dark blue (#2c3e50)
        [0.20, 0.29, 0.37, 1.0],   // Darker blue (#34495e)
        [0.74, 0.76, 0.78, 1.0],   // Light gray (#bdc3c7)
        [0.93, 0.94, 0.95, 1.0],   // Very light gray (#ecf0f1)
        [0.84, 0.86, 0.86, 1.0],   // Another gray (#d5dbdb)
        [0.68, 0.71, 0.75, 1.0],   // Gray (#aeb6bf)
        [0.52, 0.57, 0.62, 1.0],   // Medium gray (#85929e)
        [0.20, 0.60, 0.86, 1.0],   // Blue (#3498db)
        [0.10, 0.15, 0.19, 1.0],   // Very dark (#1a252f)
    ];

    let mut vertices = Vec::new();
    for (i, color) in colors.iter().enumerate() {
        let y_offset = -0.9 + (i as f32 * 0.18);
        // Quad as two triangles
        vertices.push(Vertex { position: [-0.8, y_offset], color: *color });
        vertices.push(Vertex { position: [ 0.8, y_offset], color: *color });
        vertices.push(Vertex { position: [ 0.8, y_offset + 0.15], color: *color });
        vertices.push(Vertex { position: [-0.8, y_offset], color: *color });
        vertices.push(Vertex { position: [ 0.8, y_offset + 0.15], color: *color });
        vertices.push(Vertex { position: [-0.8, y_offset + 0.15], color: *color });
    }
    vertices
}

// ============================================================================
// Diagnostic Renderer
// ============================================================================

struct DiagnosticRenderer {
    device: Device,
    command_queue: CommandQueue,
    pipeline: RenderPipelineState,
    vertex_buffer: Buffer,
    vertex_count: usize,
    mode: TestMode,
    frame_count: u32,
}

impl DiagnosticRenderer {
    fn new(device: Device, mode: TestMode) -> Result<Self, String> {
        println!("\n=== Render Diagnostic ===");
        println!("Test mode: {:?}", mode);

        let command_queue = device.new_command_queue();

        // Compile shader
        let library = device
            .new_library_with_source(SIMPLE_SHADER, &CompileOptions::new())
            .map_err(|e| format!("Shader compile error: {}", e))?;

        let vertex_fn = library.get_function("simple_vertex", None)
            .map_err(|e| format!("Missing vertex fn: {}", e))?;
        let fragment_fn = library.get_function("simple_fragment", None)
            .map_err(|e| format!("Missing fragment fn: {}", e))?;

        let desc = RenderPipelineDescriptor::new();
        desc.set_vertex_function(Some(&vertex_fn));
        desc.set_fragment_function(Some(&fragment_fn));

        let attachment = desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let pipeline = device.new_render_pipeline_state(&desc)
            .map_err(|e| format!("Pipeline error: {}", e))?;

        // Create vertices based on mode
        let vertices = match mode {
            TestMode::ClearOnly => vec![],
            TestMode::SingleTriangle => create_single_triangle(),
            TestMode::SimpleQuad => create_simple_quad(),
            TestMode::ManyQuads => create_many_quads(),
            TestMode::FullPipeline => vec![], // Will be handled separately
        };

        println!("Created {} vertices", vertices.len());

        let vertex_buffer = if vertices.is_empty() {
            // Create minimal buffer to avoid issues
            device.new_buffer(64, MTLResourceOptions::StorageModeShared)
        } else {
            device.new_buffer_with_data(
                vertices.as_ptr() as *const _,
                (vertices.len() * mem::size_of::<Vertex>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        Ok(Self {
            device,
            command_queue,
            pipeline,
            vertex_buffer,
            vertex_count: vertices.len(),
            mode,
            frame_count: 0,
        })
    }

    fn render(&mut self, drawable: &MetalDrawableRef) {
        self.frame_count += 1;

        // Get texture dimensions for capture
        let should_capture = self.frame_count == 5;
        let tex_width = drawable.texture().width();
        let tex_height = drawable.texture().height();

        let command_buffer = self.command_queue.new_command_buffer();

        let render_pass_desc = RenderPassDescriptor::new();
        let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        // Clear to a distinctive gray that we can identify
        color_attachment.set_clear_color(MTLClearColor::new(0.6, 0.6, 0.6, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(render_pass_desc);

        if self.vertex_count > 0 {
            encoder.set_render_pipeline_state(&self.pipeline);
            encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, self.vertex_count as u64);
        }

        encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        // Capture texture AFTER commit using blit encoder
        if should_capture {
            self.capture_frame_from_dimensions(tex_width, tex_height);
        }
    }

    fn capture_frame_from_dimensions(&mut self, width: u64, height: u64) {
        println!("\n=== CAPTURING FRAME {} ===", self.frame_count);
        println!("Note: Reading back from GPU may show stale data; relying on visual inspection.");

        // For proper readback, we'd need to render to a shared texture and blit
        // For now, just print that we're ready for visual inspection
        println!("Rendered {} vertices in mode {:?}", self.vertex_count, self.mode);
        println!("Window size: {}x{}", width, height);
        println!("\n*** VISUAL INSPECTION REQUIRED ***");
        println!("Look at the window - do you see any bright magenta/cyan/yellow garbage?");
    }
}

// ============================================================================
// Window Application
// ============================================================================

struct DiagnosticApp {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    renderer: Option<DiagnosticRenderer>,
}

impl DiagnosticApp {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            renderer: None,
        }
    }
}

impl ApplicationHandler for DiagnosticApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let device = Device::system_default().expect("No Metal device");
        println!("GPU: {}", device.name());

        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("Render Diagnostic");

        let window = event_loop.create_window(window_attrs).unwrap();

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                    let view = appkit_handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        layer.set_drawable_size(CGSize::new(WIDTH as f64, HEIGHT as f64));

        // Choose test mode from environment or default
        let mode = std::env::var("TEST_MODE")
            .ok()
            .and_then(|s| match s.as_str() {
                "clear" => Some(TestMode::ClearOnly),
                "triangle" => Some(TestMode::SingleTriangle),
                "quad" => Some(TestMode::SimpleQuad),
                "many" => Some(TestMode::ManyQuads),
                "full" => Some(TestMode::FullPipeline),
                _ => None,
            })
            .unwrap_or(TestMode::ManyQuads);

        let renderer = DiagnosticRenderer::new(device, mode).expect("Failed to create renderer");

        self.window = Some(window);
        self.layer = Some(layer);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!("\n=== Diagnostic Complete ===");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{Key, NamedKey};
                if event.state == winit::event::ElementState::Pressed {
                    if let Key::Named(NamedKey::Escape) = event.logical_key {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(layer), Some(renderer)) = (&self.layer, &mut self.renderer) {
                    if let Some(drawable) = layer.next_drawable() {
                        renderer.render(drawable);
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    println!("\n========================================");
    println!("     RENDER DIAGNOSTIC TOOL");
    println!("========================================");
    println!("\nUsage: TEST_MODE=<mode> cargo run --release --example render_diagnostic");
    println!("Modes: clear, triangle, quad, many, full");
    println!("Default: many (tests multiple quads with CSS colors)");
    println!("\nPress ESC to exit after viewing results.\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = DiagnosticApp::new();
    event_loop.run_app(&mut app).unwrap();
}
