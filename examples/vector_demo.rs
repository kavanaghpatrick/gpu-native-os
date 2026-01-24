//! GPU Vector Graphics Demo
//!
//! Demonstrates GPU-native vector rendering from Issue #34.
//! Shows bezier paths, shapes, and gradients rendered via GPU compute tessellation.
//!
//! Run with: cargo run --release --example vector_demo

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::vector::{
    AntiAliasMode, Color, GradientStop, LinearGradient, Paint, PathBuilder, RadialGradient, VectorRenderer,
};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;

struct VectorDemoApp {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    device: Option<Device>,
    command_queue: Option<CommandQueue>,
    renderer: Option<VectorRenderer>,
    frame_count: u64,
    start_time: Instant,
    last_vertex_count: u32,
}

impl VectorDemoApp {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            device: None,
            command_queue: None,
            renderer: None,
            frame_count: 0,
            start_time: Instant::now(),
            last_vertex_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");

        println!("========================================");
        println!("  GPU VECTOR GRAPHICS DEMO");
        println!("========================================");
        println!("GPU: {}", device.name());
        println!("\nFeatures:");
        println!("  - GPU compute tessellation");
        println!("  - Bezier path rendering");
        println!("  - Shapes: rect, rounded rect, circle, ellipse");
        println!("  - Solid color fills");
        println!("  - Linear gradients (Issue #35)");
        println!("  - Radial gradients (Issue #35)");
        println!("  - Anti-aliasing (Issue #36)");
        println!("\nControls:");
        println!("  1 - AA Off");
        println!("  2 - AA Fast (default)");
        println!("  3 - AA Quality");

        // Create Metal layer
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

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        let command_queue = device.new_command_queue();

        // Create vector renderer
        println!("\nInitializing GPU vector renderer...");
        let renderer = VectorRenderer::new(&device).expect("Failed to create vector renderer");
        println!("  Renderer ready");

        self.window = Some(window);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.renderer = Some(renderer);
        self.device = Some(device);
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let command_queue = self.command_queue.as_ref().unwrap();
        let renderer = self.renderer.as_mut().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        let time = self.start_time.elapsed().as_secs_f32();
        let width = WINDOW_WIDTH as f32;
        let height = WINDOW_HEIGHT as f32;

        // Clear paths from previous frame
        renderer.clear();

        // === Debug: Start with simple shapes ===

        // 1. Simple triangle (most basic test)
        let mut path = PathBuilder::new();
        path.move_to(100.0, 100.0);
        path.line_to(300.0, 100.0);
        path.line_to(200.0, 300.0);
        path.close();
        renderer.fill_color(&path.build(), Color::RED);

        // 2. Simple rectangle
        let mut path = PathBuilder::new();
        path.rect(400.0, 100.0, 200.0, 150.0);
        renderer.fill_color(&path.build(), Color::GREEN);

        // 3. Circle
        let cx = 200.0 + (time * 0.5).sin() * 50.0;
        let cy = 500.0;
        let mut path = PathBuilder::new();
        path.circle(cx, cy, 60.0);
        renderer.fill_color(&path.build(), Color::BLUE);

        // 4. Star (line segments only)
        let star_cx = 500.0;
        let star_cy = 500.0;
        let outer_r = 80.0;
        let inner_r = 35.0;
        let mut path = PathBuilder::new();
        for i in 0..10 {
            let angle = std::f32::consts::PI * 2.0 * i as f32 / 10.0 - std::f32::consts::PI / 2.0;
            let r = if i % 2 == 0 { outer_r } else { inner_r };
            let x = star_cx + angle.cos() * r;
            let y = star_cy + angle.sin() * r;
            if i == 0 {
                path.move_to(x, y);
            } else {
                path.line_to(x, y);
            }
        }
        path.close();
        renderer.fill_color(&path.build(), Color::YELLOW);

        // 5. Rounded rect with solid color
        let mut path = PathBuilder::new();
        path.rounded_rect(700.0, 400.0, 200.0, 150.0, 20.0);
        renderer.fill_color(&path.build(), Color::MAGENTA);

        // === Issue #35: Gradient Demos ===

        // 6. Linear gradient rectangle (horizontal red to blue)
        let mut path = PathBuilder::new();
        path.rect(700.0, 100.0, 200.0, 100.0);
        let linear_grad = LinearGradient::new(
            [700.0, 100.0],  // start
            [900.0, 100.0],  // end (horizontal)
            vec![
                GradientStop::new(0.0, Color::RED),
                GradientStop::new(0.5, Color::YELLOW),
                GradientStop::new(1.0, Color::BLUE),
            ],
        );
        renderer.fill(&path.build(), Paint::Linear(linear_grad));

        // 7. Linear gradient rectangle (vertical with animation)
        let mut path = PathBuilder::new();
        path.rect(700.0, 220.0, 200.0, 100.0);
        let anim_offset = (time * 0.5).sin() * 50.0;
        let linear_grad = LinearGradient::new(
            [700.0, 220.0 + anim_offset],  // start (animated)
            [700.0, 320.0 + anim_offset],  // end
            vec![
                GradientStop::new(0.0, Color::CYAN),
                GradientStop::new(1.0, Color::MAGENTA),
            ],
        );
        renderer.fill(&path.build(), Paint::Linear(linear_grad));

        // 8. Radial gradient circle
        let rad_cx = 200.0;
        let rad_cy = 650.0;
        let rad_r = 70.0;
        let mut path = PathBuilder::new();
        path.circle(rad_cx, rad_cy, rad_r);
        let radial_grad = RadialGradient::new(
            [rad_cx, rad_cy],  // center
            rad_r,             // radius
            vec![
                GradientStop::new(0.0, Color::WHITE),
                GradientStop::new(0.5, Color::YELLOW),
                GradientStop::new(1.0, Color::RED),
            ],
        );
        renderer.fill(&path.build(), Paint::Radial(radial_grad));

        // 9. Radial gradient rounded rect (sunset effect)
        let mut path = PathBuilder::new();
        path.rounded_rect(400.0, 600.0, 200.0, 100.0, 15.0);
        let sunset_grad = RadialGradient::new(
            [500.0, 650.0],  // center
            150.0,           // radius
            vec![
                GradientStop::new(0.0, Color::from_hex(0xFFFF00)),  // Yellow
                GradientStop::new(0.3, Color::from_hex(0xFF8800)),  // Orange
                GradientStop::new(0.6, Color::from_hex(0xFF0044)),  // Red-pink
                GradientStop::new(1.0, Color::from_hex(0x440088)),  // Purple
            ],
        );
        renderer.fill(&path.build(), Paint::Radial(sunset_grad));

        // === Render ===
        let command_buffer = command_queue.new_command_buffer();

        // Create render pass with clear
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));

        let encoder = command_buffer.new_render_command_encoder(&render_desc);

        // Render vectors (tessellation happens internally)
        renderer.render(&encoder, width, height);

        encoder.end_encoding();

        self.last_vertex_count = renderer.vertex_count();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        self.frame_count += 1;

        // Print stats periodically
        if self.frame_count % 120 == 0 {
            let elapsed = self.start_time.elapsed().as_secs_f32();
            let fps = self.frame_count as f32 / elapsed;
            let aa_mode = renderer.aa_mode();
            let aa_str = match aa_mode {
                AntiAliasMode::None => "Off",
                AntiAliasMode::Fast => "Fast",
                AntiAliasMode::Quality => "Quality",
            };
            println!(
                "Frame {} | {:.1} FPS | {} vertices | AA: {}",
                self.frame_count, fps, self.last_vertex_count, aa_str
            );
        }
    }
}

impl ApplicationHandler for VectorDemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("GPU Vector Graphics Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

            let window = event_loop.create_window(attrs).unwrap();
            self.initialize(window);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        // CRITICAL: Wrap entire event handler in autoreleasepool to drain Metal objects
        autoreleasepool(|| {
            match event {
                WindowEvent::CloseRequested => {
                    println!("\nDemo complete. {} frames rendered.", self.frame_count);
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.render();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::Resized(size) => {
                    if let Some(layer) = &self.layer {
                        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        if let PhysicalKey::Code(key) = event.physical_key {
                            if let Some(renderer) = &mut self.renderer {
                                match key {
                                    KeyCode::Digit1 => {
                                        renderer.set_aa_mode(AntiAliasMode::None);
                                        println!("AA Mode: Off");
                                    }
                                    KeyCode::Digit2 => {
                                        renderer.set_aa_mode(AntiAliasMode::Fast);
                                        println!("AA Mode: Fast");
                                    }
                                    KeyCode::Digit3 => {
                                        renderer.set_aa_mode(AntiAliasMode::Quality);
                                        println!("AA Mode: Quality");
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = VectorDemoApp::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
