// Particle System Demo - Using GpuApp Framework
//
// 16K particles with GPU physics simulation.
// A visually impressive showpiece demo!
//
// Controls:
// - Click and drag: Attract particles toward cursor
// - Up/Down: Adjust attraction strength
// - Left/Right: Adjust gravity strength
// - R: Reset particles
// - Space: Toggle gravity on/off

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::particles::{ParticleSystem, PARTICLE_COUNT};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct ParticleDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<ParticleSystem>,
    window_size: (u32, u32),
    gravity_enabled: bool,
    saved_gravity: f32,
}

impl ParticleDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            app: None,
            window_size: (1200, 800),
            gravity_enabled: true,
            saved_gravity: 0.3,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("========================================");
        println!("     PARTICLE SYSTEM DEMO");
        println!("========================================");
        println!("GPU: {}", device.name());
        println!("Particles: {}", PARTICLE_COUNT);
        println!("Using GpuApp Framework");
        println!();
        println!("Controls:");
        println!("  Click+Drag  - Attract particles to cursor");
        println!("  Up/Down     - Adjust attraction strength");
        println!("  Left/Right  - Adjust gravity strength");
        println!("  Space       - Toggle gravity on/off");
        println!("  R           - Reset all particles");
        println!("========================================");
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
        let app = ParticleSystem::new(&device).expect("Failed to create Particle System");

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

        // Print stats occasionally
        if runtime.frame_count() % 120 == 0 {
            println!(
                "Frame: {:>6} | {} | Attraction: {:.1} | Gravity: {:.2} | FPS: ~{:.0}",
                runtime.frame_count(),
                app.stats(),
                app.attraction_strength(),
                app.gravity_strength(),
                1.0 / runtime.delta_time()
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

        // Push to runtime's input system
        if let Some(runtime) = &self.runtime {
            runtime.push_mouse_move(norm_x, norm_y);
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            let pressed = state == ElementState::Pressed;

            // Push to runtime's input system
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
            Key::Named(NamedKey::ArrowUp) => {
                app.adjust_attraction(0.5);
                println!("Attraction: {:.1}", app.attraction_strength());
            }
            Key::Named(NamedKey::ArrowDown) => {
                app.adjust_attraction(-0.5);
                println!("Attraction: {:.1}", app.attraction_strength());
            }
            Key::Named(NamedKey::ArrowRight) => {
                app.adjust_gravity(0.1);
                self.saved_gravity = app.gravity_strength();
                self.gravity_enabled = self.saved_gravity > 0.01;
                println!("Gravity: {:.2}", app.gravity_strength());
            }
            Key::Named(NamedKey::ArrowLeft) => {
                app.adjust_gravity(-0.1);
                self.saved_gravity = app.gravity_strength();
                self.gravity_enabled = self.saved_gravity > 0.01;
                println!("Gravity: {:.2}", app.gravity_strength());
            }
            Key::Named(NamedKey::Space) => {
                self.gravity_enabled = !self.gravity_enabled;
                if self.gravity_enabled {
                    // Restore saved gravity
                    let current = app.gravity_strength();
                    app.adjust_gravity(self.saved_gravity - current);
                    println!("Gravity ON: {:.2}", app.gravity_strength());
                } else {
                    // Save current and set to zero
                    self.saved_gravity = app.gravity_strength();
                    app.adjust_gravity(-self.saved_gravity);
                    println!("Gravity OFF");
                }
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                // Reset by creating a new particle system
                if let Some(runtime) = &self.runtime {
                    if let Ok(new_app) = ParticleSystem::new(runtime.device()) {
                        *app = new_app;
                        println!("PARTICLES RESET");
                    }
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for ParticleDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1200, 800))
            .with_title("Particle System - GPU-Native OS (16K Particles)");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!();
                println!("========================================");
                println!("            DEMO COMPLETE");
                println!("========================================");
                if let Some(runtime) = &self.runtime {
                    println!("Total Frames: {}", runtime.frame_count());
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
    println!("\nStarting Particle System Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = ParticleDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
