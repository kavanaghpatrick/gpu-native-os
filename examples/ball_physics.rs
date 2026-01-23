// Ball Physics Demo
//
// 1024 balls simulated entirely on the GPU in a single threadgroup.
// Each thread owns exactly one ball - all collision detection and response
// happens on the GPU.
//
// Controls:
// - Arrow keys: Change gravity direction (tilt simulation)
// - Space: Reset to grid
// - S: Scatter randomly
// - Click: Apply radial impulse from cursor
// - Up/Down (hold): Increase/decrease gravity strength

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::ball_physics::{
    BallPhysics, ACTION_IMPULSE, ACTION_RESET, ACTION_SCATTER,
};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct BallPhysicsApp {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    physics: Option<BallPhysics>,
    last_frame: Instant,
    cursor_x: f32,
    cursor_y: f32,
    mouse_down: bool,
    mouse_clicked: bool,
    window_size: (u32, u32),
    gravity_x: f32,
    gravity_y: f32,
    gravity_strength: f32,
    frame_count: u64,
}

impl BallPhysicsApp {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            physics: None,
            last_frame: Instant::now(),
            cursor_x: 0.5,
            cursor_y: 0.5,
            mouse_down: false,
            mouse_clicked: false,
            window_size: (800, 800),
            gravity_x: 0.0,
            gravity_y: 0.5,
            gravity_strength: 0.5,
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("1024-Ball Physics Playground");
        println!("============================");
        println!("GPU: {}", device.name());
        println!();
        println!("Controls:");
        println!("  Arrow Keys  - Change gravity direction (tilt)");
        println!("  Space       - Reset balls to grid");
        println!("  S           - Scatter randomly");
        println!("  Click       - Apply radial impulse");
        println!("  +/-         - Increase/decrease gravity");
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

        // Create physics simulation
        let physics = BallPhysics::new(&device).expect("Failed to create BallPhysics");

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.physics = Some(physics);
        self.last_frame = Instant::now();
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let physics = self.physics.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // Calculate delta time
        let now = Instant::now();
        let delta_time = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Update gravity from arrow key state
        physics.update_gravity(
            self.gravity_x * self.gravity_strength,
            self.gravity_y * self.gravity_strength,
        );

        // Update cursor position
        physics.update_cursor(self.cursor_x, self.cursor_y);

        // Update mouse state
        physics.set_mouse_state(self.mouse_down, self.mouse_clicked);

        // Set impulse action when clicking
        if self.mouse_clicked {
            physics.set_action(ACTION_IMPULSE);
        }

        // Clear click flag after sending
        self.mouse_clicked = false;

        // Render
        physics.render(&drawable);

        // Print stats occasionally
        self.frame_count += 1;
        if self.frame_count % 120 == 0 {
            let fps = 1.0 / delta_time;
            println!(
                "Frame: {:>6}  FPS: {:>5.1}  Gravity: ({:>5.2}, {:>5.2})",
                self.frame_count,
                fps,
                self.gravity_x * self.gravity_strength,
                self.gravity_y * self.gravity_strength
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
        // Normalize to 0-1 range
        self.cursor_x = (x as f32) / (self.window_size.0 as f32);
        self.cursor_y = (y as f32) / (self.window_size.1 as f32);
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            let was_down = self.mouse_down;
            self.mouse_down = state == ElementState::Pressed;

            // Detect click (transition to pressed)
            if !was_down && self.mouse_down {
                self.mouse_clicked = true;
            }
        }
    }

    fn handle_key(&mut self, key: Key, pressed: bool) {
        let physics = match &self.physics {
            Some(p) => p,
            None => return,
        };

        // Arrow keys control gravity direction (only while pressed)
        match &key {
            Key::Named(NamedKey::ArrowLeft) => {
                if pressed {
                    self.gravity_x = -1.0;
                } else if self.gravity_x < 0.0 {
                    self.gravity_x = 0.0;
                }
            }
            Key::Named(NamedKey::ArrowRight) => {
                if pressed {
                    self.gravity_x = 1.0;
                } else if self.gravity_x > 0.0 {
                    self.gravity_x = 0.0;
                }
            }
            Key::Named(NamedKey::ArrowUp) => {
                if pressed {
                    self.gravity_y = -1.0;
                } else if self.gravity_y < 0.0 {
                    self.gravity_y = 0.5; // Return to default down
                }
            }
            Key::Named(NamedKey::ArrowDown) => {
                if pressed {
                    self.gravity_y = 1.0;
                } else if self.gravity_y > 0.5 {
                    self.gravity_y = 0.5;
                }
            }
            _ => {}
        }

        // Other keys only on press
        if !pressed {
            return;
        }

        match key {
            Key::Named(NamedKey::Space) => {
                physics.set_action(ACTION_RESET);
                println!("RESET - Balls arranged in grid");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("s") => {
                physics.set_action(ACTION_SCATTER);
                println!("SCATTER - Random positions and velocities");
            }
            Key::Character(ref c) if c == "=" || c == "+" => {
                self.gravity_strength = (self.gravity_strength + 0.1).min(2.0);
                println!("Gravity strength: {:.1}", self.gravity_strength);
            }
            Key::Character(ref c) if c == "-" || c == "_" => {
                self.gravity_strength = (self.gravity_strength - 0.1).max(0.0);
                println!("Gravity strength: {:.1}", self.gravity_strength);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for BallPhysicsApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            .with_title("1024-Ball Physics - GPU-Native OS Demo");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!();
                println!("Final Stats:");
                println!("  Total frames: {}", self.frame_count);
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
                self.handle_key(event.logical_key, event.state == ElementState::Pressed);
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
    println!("Starting 1024-Ball Physics Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = BallPhysicsApp::new();
    event_loop.run_app(&mut app).unwrap();
}
