// Boids Flocking Simulation Demo - Using GpuApp Framework
//
// 1024 boids exhibiting emergent flocking behavior on the GPU.
// Each boid follows three simple rules: separation, alignment, cohesion.
//
// Controls:
// - Mouse move: Boids gently avoid cursor
// - Click/Hold: Boids are attracted to cursor
// - S: Scatter boids randomly
// - R: Reset parameters to defaults
// - 1/2: Decrease/Increase separation weight
// - 3/4: Decrease/Increase alignment weight
// - 5/6: Decrease/Increase cohesion weight
// - 7/8: Decrease/Increase visual range

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::boids::BoidsSimulation;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::Key,
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct BoidsDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<BoidsSimulation>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl BoidsDemo {
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
        println!("Boids Flocking Simulation - GPU-Native OS");
        println!("==========================================");
        println!("GPU: {}", device.name());
        println!("Boids: 1024 (one per GPU thread)");
        println!();
        println!("Controls:");
        println!("  Mouse Move - Boids avoid cursor");
        println!("  Click/Hold - Attract boids to cursor");
        println!("  S          - Scatter boids");
        println!("  R          - Reset parameters");
        println!("  1/2        - Separation -/+");
        println!("  3/4        - Alignment -/+");
        println!("  5/6        - Cohesion -/+");
        println!("  7/8        - Visual range -/+");
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
        let app = BoidsSimulation::new(&device).expect("Failed to create Boids simulation");

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

        // Run frame through the OS runtime
        runtime.run_frame(app, &drawable);

        self.frame_count += 1;

        // Print stats periodically
        if self.frame_count % 300 == 0 {
            let params = app.params();
            println!(
                "Frame: {:>6} | Sep: {:.3} | Align: {:.3} | Coh: {:.4} | Range: {:.3}",
                self.frame_count,
                params.separation_weight,
                params.alignment_weight,
                params.cohesion_weight,
                params.visual_range
            );
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
            Key::Character(ref c) if c.eq_ignore_ascii_case("s") => {
                app.scatter();
                println!("SCATTERED");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                app.reset_params();
                println!("RESET to default parameters");
            }
            Key::Character(ref c) if c == "1" => {
                app.adjust_separation(-0.01);
                println!("Separation: {:.3}", app.params().separation_weight);
            }
            Key::Character(ref c) if c == "2" => {
                app.adjust_separation(0.01);
                println!("Separation: {:.3}", app.params().separation_weight);
            }
            Key::Character(ref c) if c == "3" => {
                app.adjust_alignment(-0.01);
                println!("Alignment: {:.3}", app.params().alignment_weight);
            }
            Key::Character(ref c) if c == "4" => {
                app.adjust_alignment(0.01);
                println!("Alignment: {:.3}", app.params().alignment_weight);
            }
            Key::Character(ref c) if c == "5" => {
                app.adjust_cohesion(-0.001);
                println!("Cohesion: {:.4}", app.params().cohesion_weight);
            }
            Key::Character(ref c) if c == "6" => {
                app.adjust_cohesion(0.001);
                println!("Cohesion: {:.4}", app.params().cohesion_weight);
            }
            Key::Character(ref c) if c == "7" => {
                app.adjust_visual_range(-0.01);
                println!("Visual range: {:.3}", app.params().visual_range);
            }
            Key::Character(ref c) if c == "8" => {
                app.adjust_visual_range(0.01);
                println!("Visual range: {:.3}", app.params().visual_range);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for BoidsDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            .with_title("Boids Flocking Simulation - GPU-Native OS");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!();
                println!("Final Stats:");
                println!("  Total Frames: {}", self.frame_count);
                if let Some(app) = &self.app {
                    let params = app.params();
                    println!("  Separation: {:.3}", params.separation_weight);
                    println!("  Alignment: {:.3}", params.alignment_weight);
                    println!("  Cohesion: {:.4}", params.cohesion_weight);
                    println!("  Visual Range: {:.3}", params.visual_range);
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
    println!("Starting Boids Flocking Simulation...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = BoidsDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
