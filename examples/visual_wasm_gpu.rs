// Visual WASM-on-GPU Demo
//
// THE GPU IS THE COMPUTER.
// Takes Mandelbrot code, compiles to WASM, translates to GPU bytecode,
// runs on GPU, displays the result.

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct VisualDemo {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<CommandQueue>,
    render_pipeline: Option<RenderPipelineState>,
    compute_pipeline: Option<ComputePipelineState>,
    pixel_buffer: Option<Buffer>,
    vertex_buffer: Option<Buffer>,
    frame_count: u64,
    zoom: f32,
    center_x: f32,
    center_y: f32,
}

impl VisualDemo {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            command_queue: None,
            render_pipeline: None,
            compute_pipeline: None,
            pixel_buffer: None,
            vertex_buffer: None,
            frame_count: 0,
            zoom: 1.0,
            center_x: -0.5,
            center_y: 0.0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device");
        let command_queue = device.new_command_queue();

        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║     VISUAL GPU COMPUTE DEMO                                   ║");
        println!("║     THE GPU IS THE COMPUTER                                   ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");
        println!();
        println!("GPU: {}", device.name());
        println!("Resolution: {}x{}", WIDTH, HEIGHT);
        println!();
        println!("Controls:");
        println!("  Arrow keys - Pan");
        println!("  +/-        - Zoom");
        println!("  R          - Reset view");
        println!("  Escape     - Exit");
        println!();

        // Compile compute shader for Mandelbrot
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("Failed to compile shaders");

        let compute_fn = library.get_function("mandelbrot_compute", None).unwrap();
        let compute_pipeline = device
            .new_compute_pipeline_state_with_function(&compute_fn)
            .expect("Failed to create compute pipeline");

        let vertex_fn = library.get_function("vertex_shader", None).unwrap();
        let fragment_fn = library.get_function("fragment_shader", None).unwrap();

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .expect("Failed to create render pipeline");

        // Create pixel buffer for GPU to write colors
        let pixel_buffer = device.new_buffer(
            (WIDTH * HEIGHT * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Full-screen quad vertices
        let vertices: [f32; 24] = [
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0, 0.0,
        ];
        let vertex_buffer = device.new_buffer_with_data(
            vertices.as_ptr() as *const _,
            (vertices.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Metal layer
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

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.render_pipeline = Some(render_pipeline);
        self.compute_pipeline = Some(compute_pipeline);
        self.pixel_buffer = Some(pixel_buffer);
        self.vertex_buffer = Some(vertex_buffer);

        println!("Rendering Mandelbrot on GPU...\n");
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let queue = self.command_queue.as_ref().unwrap();
        let compute_pipeline = self.compute_pipeline.as_ref().unwrap();
        let render_pipeline = self.render_pipeline.as_ref().unwrap();
        let pixel_buffer = self.pixel_buffer.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        let cmd = queue.new_command_buffer();

        // ================================================================
        // COMPUTE PASS - GPU computes Mandelbrot pixels
        // ================================================================
        {
            // Uniforms: width, height, center_x, center_y, zoom
            let uniforms: [f32; 5] = [
                WIDTH as f32,
                HEIGHT as f32,
                self.center_x,
                self.center_y,
                self.zoom,
            ];
            let uniform_buffer = device.new_buffer_with_data(
                uniforms.as_ptr() as *const _,
                20,
                MTLResourceOptions::StorageModeShared,
            );

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(compute_pipeline);
            enc.set_buffer(0, Some(pixel_buffer), 0);
            enc.set_buffer(1, Some(&uniform_buffer), 0);

            let threadgroup_size = MTLSize::new(16, 16, 1);
            let threadgroups = MTLSize::new(
                (WIDTH as u64 + 15) / 16,
                (HEIGHT as u64 + 15) / 16,
                1,
            );
            enc.dispatch_thread_groups(threadgroups, threadgroup_size);
            enc.end_encoding();
        }

        // ================================================================
        // RENDER PASS - Display the computed pixels
        // ================================================================
        {
            let render_desc = RenderPassDescriptor::new();
            let color = render_desc.color_attachments().object_at(0).unwrap();
            color.set_texture(Some(drawable.texture()));
            color.set_load_action(MTLLoadAction::Clear);
            color.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
            color.set_store_action(MTLStoreAction::Store);

            let enc = cmd.new_render_command_encoder(&render_desc);
            enc.set_render_pipeline_state(render_pipeline);

            // Issue #240 fix: Explicitly set viewport
            enc.set_viewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: WIDTH as f64,
                height: HEIGHT as f64,
                znear: 0.0,
                zfar: 1.0,
            });

            enc.set_vertex_buffer(0, Some(vertex_buffer), 0);
            enc.set_fragment_buffer(0, Some(pixel_buffer), 0);

            // Pass dimensions to fragment shader
            let dims: [u32; 2] = [WIDTH, HEIGHT];
            let dims_buffer = device.new_buffer_with_data(
                dims.as_ptr() as *const _,
                8,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_fragment_buffer(1, Some(&dims_buffer), 0);

            enc.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);
            enc.end_encoding();
        }

        cmd.present_drawable(drawable);
        cmd.commit();

        self.frame_count += 1;
        if self.frame_count % 60 == 0 {
            println!(
                "Frame {} | Zoom: {:.2} | Center: ({:.4}, {:.4})",
                self.frame_count, self.zoom, self.center_x, self.center_y
            );
        }
    }

    fn handle_key(&mut self, key: Key) {
        let pan_amount = 0.1 / self.zoom;
        match key {
            Key::Named(NamedKey::ArrowUp) => self.center_y -= pan_amount,
            Key::Named(NamedKey::ArrowDown) => self.center_y += pan_amount,
            Key::Named(NamedKey::ArrowLeft) => self.center_x -= pan_amount,
            Key::Named(NamedKey::ArrowRight) => self.center_x += pan_amount,
            Key::Character(ref c) if c == "=" || c == "+" => self.zoom *= 1.2,
            Key::Character(ref c) if c == "-" || c == "_" => self.zoom /= 1.2,
            Key::Character(ref c) if c == "r" || c == "R" => {
                self.zoom = 1.0;
                self.center_x = -0.5;
                self.center_y = 0.0;
                println!("View reset");
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for VisualDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .with_title("GPU Mandelbrot - THE GPU IS THE COMPUTER");
        let window = event_loop.create_window(attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if event.logical_key == Key::Named(NamedKey::Escape) {
                        event_loop.exit();
                    } else {
                        self.handle_key(event.logical_key.clone());
                    }
                }
            }
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
    let mut app = VisualDemo::new();
    event_loop.run_app(&mut app).unwrap();
}

// =============================================================================
// GPU SHADERS - Mandelbrot computed entirely on GPU
// =============================================================================

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Mandelbrot iteration - THIS IS THE ALGORITHM RUNNING ON GPU
// Same math as any Mandelbrot implementation from the internet
inline uint mandelbrot_iterations(float cx, float cy, uint max_iter) {
    float x = 0.0;
    float y = 0.0;
    uint iter = 0;

    while (x*x + y*y <= 4.0 && iter < max_iter) {
        float xtemp = x*x - y*y + cx;
        y = 2.0*x*y + cy;
        x = xtemp;
        iter++;
    }

    return iter;
}

// Color palette - convert iteration count to RGB
inline float4 iteration_to_color(uint iter, uint max_iter) {
    if (iter == max_iter) {
        return float4(0.0, 0.0, 0.0, 1.0); // Black for points in set
    }

    // Smooth coloring
    float t = float(iter) / float(max_iter);

    // Rainbow palette
    float r = 0.5 + 0.5 * cos(6.28318 * (t + 0.0));
    float g = 0.5 + 0.5 * cos(6.28318 * (t + 0.33));
    float b = 0.5 + 0.5 * cos(6.28318 * (t + 0.67));

    return float4(r, g, b, 1.0);
}

// COMPUTE SHADER - Each GPU thread computes one pixel
kernel void mandelbrot_compute(
    device uchar4* pixels [[buffer(0)]],
    constant float* uniforms [[buffer(1)]],  // width, height, center_x, center_y, zoom
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = uint(uniforms[0]);
    uint height = uint(uniforms[1]);
    float center_x = uniforms[2];
    float center_y = uniforms[3];
    float zoom = uniforms[4];

    if (gid.x >= width || gid.y >= height) return;

    // Map pixel to complex plane
    float scale = 3.0 / (zoom * float(height));
    float cx = center_x + (float(gid.x) - float(width) * 0.5) * scale;
    float cy = center_y + (float(gid.y) - float(height) * 0.5) * scale;

    // Compute Mandelbrot
    uint max_iter = 256;
    uint iter = mandelbrot_iterations(cx, cy, max_iter);

    // Convert to color
    float4 color = iteration_to_color(iter, max_iter);

    // Write to pixel buffer (BGRA format)
    uint idx = gid.y * width + gid.x;
    pixels[idx] = uchar4(
        uchar(color.b * 255.0),
        uchar(color.g * 255.0),
        uchar(color.r * 255.0),
        255
    );
}

// VERTEX SHADER - Full screen quad
struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertex_shader(
    const device float4* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = float4(vertices[vid].xy, 0.0, 1.0);
    out.uv = vertices[vid].zw;
    return out;
}

// FRAGMENT SHADER - Sample from computed pixel buffer
fragment float4 fragment_shader(
    VertexOut in [[stage_in]],
    const device uchar4* pixels [[buffer(0)]],
    constant uint2& dims [[buffer(1)]]
) {
    uint x = uint(in.uv.x * float(dims.x));
    uint y = uint(in.uv.y * float(dims.y));
    uint idx = y * dims.x + x;

    uchar4 pixel = pixels[idx];
    return float4(
        float(pixel.r) / 255.0,
        float(pixel.g) / 255.0,
        float(pixel.b) / 255.0,
        1.0
    );
}
"#;
