// Wave Simulation Demo - Using GpuApp Framework
//
// 256x256 grid simulating 2D wave propagation on the GPU.
// Beautiful interactive ripple effects!
//
// Controls:
// - Left Click: Create positive ripple (bright)
// - Right Click: Create negative ripple (dark)
// - R: Reset simulation
// - Up/Down: Adjust damping (wave decay)
// - Left/Right: Adjust wave speed
// - +/-: Adjust ripple strength

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::waves::WaveSimulation;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct WaveDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<WaveSimulation>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl WaveDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            app: None,
            window_size: (800, 800),
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("Wave Simulation - GPU-Native OS Demo");
        println!("=====================================");
        println!("GPU: {}", device.name());
        println!("Grid: 256x256 = 65,536 cells");
        println!();
        println!("Controls:");
        println!("  Left Click   - Create positive ripple (bright)");
        println!("  Right Click  - Create negative ripple (dark)");
        println!("  R            - Reset simulation");
        println!("  Up/Down      - Adjust damping (wave decay)");
        println!("  Left/Right   - Adjust wave speed");
        println!("  +/-          - Adjust ripple strength");
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
        let app = WaveSimulation::new(&device).expect("Failed to create Wave Simulation");

        println!("Simulation parameters:");
        println!("  Damping: {:.4}", app.params().damping);
        println!("  Wave Speed: {:.2}", app.params().wave_speed);
        println!("  Ripple Strength: {:.1}", app.params().ripple_strength);
        println!();
        println!("Click anywhere to create ripples!");
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

        // Run frame through the OS runtime
        runtime.run_frame(app, &drawable);

        self.frame_count += 1;

        // Print stats occasionally
        if self.frame_count % 300 == 0 {
            let params = app.params();
            println!(
                "Frame: {:>6}  Damping: {:.4}  Speed: {:.2}  Strength: {:.1}",
                self.frame_count,
                params.damping,
                params.wave_speed,
                params.ripple_strength
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
        let button_code = match button {
            MouseButton::Left => 0,
            MouseButton::Right => 1,
            _ => return,
        };

        let pressed = state == ElementState::Pressed;

        if let Some(runtime) = &self.runtime {
            runtime.push_mouse_button(button_code, pressed, 0.0, 0.0);
        }
    }

    fn handle_key(&mut self, key: Key) {
        let app = match &mut self.app {
            Some(a) => a,
            None => return,
        };

        match key {
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                app.reset();
                println!("RESET");
            }
            Key::Named(NamedKey::ArrowUp) => {
                app.adjust_damping(0.001);
                println!("Damping: {:.4} (slower decay)", app.params().damping);
            }
            Key::Named(NamedKey::ArrowDown) => {
                app.adjust_damping(-0.001);
                println!("Damping: {:.4} (faster decay)", app.params().damping);
            }
            Key::Named(NamedKey::ArrowRight) => {
                app.adjust_wave_speed(0.02);
                println!("Wave Speed: {:.2} (faster)", app.params().wave_speed);
            }
            Key::Named(NamedKey::ArrowLeft) => {
                app.adjust_wave_speed(-0.02);
                println!("Wave Speed: {:.2} (slower)", app.params().wave_speed);
            }
            Key::Character(ref c) if c == "=" || c == "+" => {
                app.adjust_ripple_strength(0.1);
                println!("Ripple Strength: {:.1}", app.params().ripple_strength);
            }
            Key::Character(ref c) if c == "-" || c == "_" => {
                app.adjust_ripple_strength(-0.1);
                println!("Ripple Strength: {:.1}", app.params().ripple_strength);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for WaveDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            .with_title("Wave Simulation - GPU-Native OS");

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
    println!("Starting Wave Simulation Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = WaveDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
