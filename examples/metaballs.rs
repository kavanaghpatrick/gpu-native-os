// Metaballs Demo - Using GpuApp Framework
//
// Beautiful organic blobby shapes computed entirely on the GPU.
// Each pixel computes the metaball field and renders the threshold surface.
//
// Controls:
// - Hold mouse: Attract balls to cursor
// - Space: Toggle permanent attraction
// - R: Reset balls
// - Up/Down: Adjust threshold (blob size)

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::{GpuApp, GpuRuntime, SLOT_FRAME_STATE, SLOT_INPUT_QUEUE, SLOT_APP_PARAMS, SLOT_APP_START};
use rust_experiment::gpu_os::metaballs::Metaballs;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct MetaballsDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<Metaballs>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl MetaballsDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            app: None,
            window_size: (800, 600),
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("Metaballs - GPU-Native OS Demo");
        println!("==============================");
        println!("GPU: {}", device.name());
        println!();
        println!("Controls:");
        println!("  Hold Mouse - Attract balls to cursor");
        println!("  Space      - Toggle permanent attraction");
        println!("  R          - Reset balls");
        println!("  Up/Down    - Adjust blob threshold");
        println!();

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
        let app = Metaballs::new(&device).expect("Failed to create Metaballs app");

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

        // We need a custom render loop because the fragment shader needs buffer bindings
        // that the standard GpuRuntime.run_frame doesn't provide

        // Get command queue from runtime
        let command_buffer = runtime.command_queue.new_command_buffer();

        // === COMPUTE PASS ===
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(app.compute_pipeline());

        // Bind OS buffers (slots 0-1)
        compute_encoder.set_buffer(SLOT_FRAME_STATE, Some(&runtime.memory.frame_state_buffer), 0);
        compute_encoder.set_buffer(SLOT_INPUT_QUEUE, Some(&runtime.memory.input_queue_buffer), 0);

        // Bind app params buffer (slot 2)
        compute_encoder.set_buffer(SLOT_APP_PARAMS, Some(app.params_buffer()), 0);

        // Bind app-specific buffers (slots 3+)
        for (i, buffer) in app.app_buffers().iter().enumerate() {
            compute_encoder.set_buffer(SLOT_APP_START + i as u64, Some(buffer), 0);
        }

        // Dispatch compute
        let thread_count = app.thread_count() as u64;
        compute_encoder.dispatch_threads(
            MTLSize::new(thread_count, 1, 1),
            MTLSize::new(thread_count, 1, 1),
        );
        compute_encoder.end_encoding();

        // === RENDER PASS ===
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);

        let clear = app.clear_color();
        color_attachment.set_clear_color(clear);

        let render_encoder = command_buffer.new_render_command_encoder(&render_desc);

        // Use custom render method that binds fragment buffers
        app.render_with_buffers(&render_encoder);

        render_encoder.end_encoding();

        // Present and commit
        command_buffer.present_drawable(&drawable);
        command_buffer.commit();

        // Update frame state for next frame
        {
            let frame_state = runtime.memory.frame_state_mut();
            frame_state.frame_number = self.frame_count as u32;
            frame_state.time += 1.0 / 120.0; // Approximate timing
        }

        // Let app update params
        let frame_state = runtime.memory.frame_state();
        app.update_params(&frame_state, 1.0 / 120.0);

        self.frame_count += 1;

        // Print stats occasionally
        if self.frame_count % 300 == 0 {
            println!("Frame: {}  Time: {:.1}s", self.frame_count, frame_state.time);
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window_size = (width, height);
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }
    }

    fn handle_cursor_move(&mut self, x: f64, y: f64) {
        let norm_x = (x as f32) / (self.window_size.0 as f32);
        let norm_y = (y as f32) / (self.window_size.1 as f32);

        if let Some(runtime) = &self.runtime {
            runtime.push_mouse_move(norm_x, norm_y);
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            let pressed = state == ElementState::Pressed;

            if let Some(runtime) = &self.runtime {
                runtime.push_mouse_button(0, pressed, 0.0, 0.0);
            }
        }
    }

    fn handle_key(&mut self, key: Key) {
        let app = match &mut self.app {
            Some(a) => a,
            None => return,
        };

        match key {
            Key::Named(NamedKey::Space) => {
                app.toggle_attract();
                println!("Attraction toggled");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                app.reset();
                println!("Balls reset");
            }
            Key::Named(NamedKey::ArrowUp) => {
                app.adjust_threshold(0.1);
                println!("Threshold increased");
            }
            Key::Named(NamedKey::ArrowDown) => {
                app.adjust_threshold(-0.1);
                println!("Threshold decreased");
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for MetaballsDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
            .with_title("Metaballs - GPU-Native OS Demo");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!();
                println!("Final Stats:");
                println!("  Total Frames: {}", self.frame_count);
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_move(position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.handle_mouse_button(button, state);
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
    println!("Starting Metaballs Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = MetaballsDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
