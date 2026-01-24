// Mandelbrot Fractal Viewer Demo - Using GpuApp Framework
//
// Beautiful GPU-accelerated Mandelbrot set visualization with smooth coloring.
// The fractal computation happens entirely in the fragment shader for maximum
// efficiency, with each pixel computing its own iteration count in parallel.
//
// Controls:
// - Mouse scroll: Zoom in/out at cursor position
// - Click and drag: Pan the view
// - Arrow keys: Pan in that direction
// - +/=: Zoom in (at center)
// - -/_: Zoom out (at center)
// - [: Decrease max iterations
// - ]: Increase max iterations
// - R: Reset to default view
// - Escape: Exit

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::{GpuApp, GpuRuntime};
use rust_experiment::gpu_os::mandelbrot::MandelbrotViewer;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct MandelbrotDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<MandelbrotViewer>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl MandelbrotDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            app: None,
            window_size: (1024, 768),
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("Mandelbrot Fractal Viewer");
        println!("=========================");
        println!("GPU: {}", device.name());
        println!();
        println!("Controls:");
        println!("  Mouse scroll    - Zoom in/out at cursor");
        println!("  Click + drag    - Pan the view");
        println!("  Arrow keys      - Pan");
        println!("  + / -           - Zoom in/out at center");
        println!("  [ / ]           - Decrease/increase iterations");
        println!("  R               - Reset view");
        println!("  Escape          - Exit");
        println!();
        println!("Preset Locations (number keys):");
        println!("  1 - Seahorse Valley (spirals)");
        println!("  2 - Double Spiral (intricate)");
        println!("  3 - Lightning (branching)");
        println!("  4 - Mini Mandelbrot (tiny copy)");
        println!("  5 - Elephant Valley");
        println!("  6 - Starfish");
        println!("  7 - Tendril (delicate spirals)");
        println!("  8 - Julia Island");
        println!("  9 - Quad Spiral (deep zoom)");
        println!("  0 - Full View (overview)");
        println!();
        println!("Note: Max zoom ~10^6 (f32 precision limit)");

        // Set up Metal layer
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
        self.window_size = (size.width, size.height);
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        // Create runtime and app
        let runtime = GpuRuntime::new(device.clone());
        let mut app = MandelbrotViewer::new(&device).expect("Failed to create Mandelbrot viewer");
        app.set_viewport(size.width as f32, size.height as f32);

        // Print initial status
        println!("Initial view:");
        let (cx, cy) = app.center();
        println!(
            "Zoom: {:.2e}  Center: ({:.10}, {:.10})",
            app.zoom_level(), cx, cy
        );
        println!();

        self.window = Some(window);
        self.layer = Some(layer);
        self.runtime = Some(runtime);
        self.app = Some(app);
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let runtime = self.runtime.as_mut().unwrap();
        let app = self.app.as_mut().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // Get frame timing
        let delta_time = runtime.delta_time();

        // Update app params
        app.update_params(&runtime.memory.frame_state(), delta_time);

        // Create command buffer
        let command_buffer = runtime.command_queue.new_command_buffer();

        // === COMPUTE PASS (generates fullscreen quad) ===
        {
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(app.compute_pipeline());

            // Bind buffers
            compute_encoder.set_buffer(0, Some(&runtime.memory.frame_state_buffer), 0);
            compute_encoder.set_buffer(1, Some(&runtime.memory.input_queue_buffer), 0);
            compute_encoder.set_buffer(2, Some(app.params_buffer()), 0);
            compute_encoder.set_buffer(3, Some(app.vertices_buffer()), 0);

            // Dispatch
            let thread_count = app.thread_count() as u64;
            compute_encoder.dispatch_threads(
                MTLSize::new(thread_count, 1, 1),
                MTLSize::new(thread_count, 1, 1),
            );
            compute_encoder.end_encoding();
        }

        // === RENDER PASS ===
        {
            let render_desc = RenderPassDescriptor::new();
            let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
            color_attachment.set_texture(Some(drawable.texture()));
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_store_action(MTLStoreAction::Store);
            color_attachment.set_clear_color(app.clear_color());

            let render_encoder = command_buffer.new_render_command_encoder(&render_desc);
            render_encoder.set_render_pipeline_state(app.render_pipeline());

            // Bind vertex buffer
            render_encoder.set_vertex_buffer(0, Some(app.vertices_buffer()), 0);

            // Bind params to fragment shader (slot 0)
            render_encoder.set_fragment_buffer(0, Some(app.params_buffer()), 0);

            render_encoder.draw_primitives(
                MTLPrimitiveType::Triangle,
                0,
                app.vertex_count() as u64,
            );
            render_encoder.end_encoding();
        }

        // Present and commit
        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        self.frame_count += 1;

        // Print stats every 300 frames (~5 seconds at 60fps)
        if self.frame_count % 300 == 0 {
            let (cx, cy) = app.center();
            println!(
                "Frame: {}  Zoom: {:.2e}  Center: ({:.6}, {:.6})",
                self.frame_count, app.zoom_level(), cx, cy
            );
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window_size = (width, height);
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }
        if let Some(app) = &mut self.app {
            app.set_viewport(width as f32, height as f32);
        }
    }

    fn handle_cursor_move(&mut self, x: f64, y: f64) {
        let norm_x = (x as f32) / (self.window_size.0 as f32);
        let norm_y = (y as f32) / (self.window_size.1 as f32);

        // Push to runtime's input system
        if let Some(runtime) = &self.runtime {
            runtime.push_mouse_move(norm_x, norm_y);
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState, x: f64, y: f64) {
        if button == MouseButton::Left {
            let pressed = state == ElementState::Pressed;
            let norm_x = (x as f32) / (self.window_size.0 as f32);
            let norm_y = (y as f32) / (self.window_size.1 as f32);

            if let Some(runtime) = &self.runtime {
                runtime.push_mouse_button(0, pressed, norm_x, norm_y);
            }
        }
    }

    fn handle_scroll(&mut self, delta: MouseScrollDelta, x: f64, y: f64) {
        let scroll_y = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 10.0,
        };

        if scroll_y.abs() > 0.01 {
            let norm_x = (x as f32) / (self.window_size.0 as f32);
            let norm_y = (y as f32) / (self.window_size.1 as f32);

            if let Some(app) = &mut self.app {
                app.zoom_at(norm_x, norm_y, scroll_y > 0.0);
            }
        }
    }

    fn handle_key(&mut self, key: Key) {
        let app = match &mut self.app {
            Some(a) => a,
            None => return,
        };

        match key {
            // Pan with arrow keys
            Key::Named(NamedKey::ArrowUp) => app.pan(0.0, -1.0),
            Key::Named(NamedKey::ArrowDown) => app.pan(0.0, 1.0),
            Key::Named(NamedKey::ArrowLeft) => app.pan(-1.0, 0.0),
            Key::Named(NamedKey::ArrowRight) => app.pan(1.0, 0.0),

            // Zoom with +/-
            Key::Character(ref c) if c == "+" || c == "=" => app.zoom_in(),
            Key::Character(ref c) if c == "-" || c == "_" => app.zoom_out(),

            // Iterations with [/]
            Key::Character(ref c) if c == "[" => app.decrease_iterations(),
            Key::Character(ref c) if c == "]" => app.increase_iterations(),

            // Reset
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => app.reset(),

            // Number keys for preset locations
            Key::Character(ref c) if c == "1" => app.goto_preset(0),
            Key::Character(ref c) if c == "2" => app.goto_preset(1),
            Key::Character(ref c) if c == "3" => app.goto_preset(2),
            Key::Character(ref c) if c == "4" => app.goto_preset(3),
            Key::Character(ref c) if c == "5" => app.goto_preset(4),
            Key::Character(ref c) if c == "6" => app.goto_preset(5),
            Key::Character(ref c) if c == "7" => app.goto_preset(6),
            Key::Character(ref c) if c == "8" => app.goto_preset(7),
            Key::Character(ref c) if c == "9" => app.goto_preset(8),
            Key::Character(ref c) if c == "0" => app.goto_preset(9),

            _ => {}
        }
    }
}

impl ApplicationHandler for MandelbrotDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            .with_title("Mandelbrot Fractal Viewer - GPU-Native");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    logical_key: Key::Named(NamedKey::Escape),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                if let Some(app) = &self.app {
                    let (cx, cy) = app.center();
                    println!();
                    println!("Final view:");
                    println!("  Zoom: {:.6e}", app.zoom_level());
                    println!("  Center: ({:.15}, {:.15})", cx, cy);
                    println!("  Frames rendered: {}", self.frame_count);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_move(position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                // Get cursor position from window
                // Note: We don't have position here, so we'll use the last known position
                self.handle_mouse_button(button, state, 0.0, 0.0);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Use center of window as scroll position
                let x = (self.window_size.0 / 2) as f64;
                let y = (self.window_size.1 / 2) as f64;
                self.handle_scroll(delta, x, y);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.handle_key(event.logical_key);
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
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
    println!("Starting Mandelbrot Fractal Viewer...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = MandelbrotDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
