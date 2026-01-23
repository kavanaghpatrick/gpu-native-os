// Game of Life Demo - Using GpuApp Framework
//
// 32x32 Game of Life running on the GPU-Native OS.
// Now uses the GpuApp framework for OS integration.
//
// Controls:
// - Click cells to toggle alive/dead
// - Space: Play/Pause
// - S: Step one generation
// - C: Clear all cells
// - R: Randomize (~30% alive)
// - G: Spawn glider at center
// - Up/Down: Adjust speed

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::game_of_life::{
    GameOfLife, ACTION_CLEAR, ACTION_GLIDER, ACTION_RANDOM, ACTION_STEP,
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct GameOfLifeDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<GameOfLife>,
    window_size: (u32, u32),
}

impl GameOfLifeDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            app: None,
            window_size: (800, 800),
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("Game of Life - GPU-Native OS Demo");
        println!("==================================");
        println!("GPU: {}", device.name());
        println!("Using GpuApp Framework");
        println!();
        println!("Controls:");
        println!("  Click      - Toggle cell");
        println!("  Space      - Play/Pause");
        println!("  S          - Step one generation");
        println!("  C          - Clear all");
        println!("  R          - Randomize");
        println!("  G          - Spawn glider");
        println!("  Up/Down    - Adjust speed");
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
        let mut app = GameOfLife::new(&device).expect("Failed to create Game of Life");

        // Start with a random pattern
        app.set_action(ACTION_RANDOM);

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
        let state = app.simulation_state();
        if state.generation % 60 == 0 && state.generation > 0 {
            println!(
                "Gen: {:>6}  Pop: {:>4}  Speed: {:>2} gen/s  {}  [Frame: {}]",
                state.generation,
                state.population,
                state.speed as u32,
                if state.running != 0 { "RUNNING" } else { "PAUSED" },
                runtime.frame_count()
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
                // Use current cursor position (we don't have it here, so use 0,0)
                // The actual position comes from mouse move events
                runtime.push_mouse_button(0, pressed, 0.0, 0.0);
            }
        }
    }

    fn handle_key(&mut self, key: Key) {
        let app = match &self.app {
            Some(a) => a,
            None => return,
        };

        match key {
            Key::Named(NamedKey::Space) => {
                app.toggle_running();
                let state = app.simulation_state();
                println!("{}", if state.running != 0 { "PLAYING" } else { "PAUSED" });
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("s") => {
                app.set_action(ACTION_STEP);
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("c") => {
                app.set_action(ACTION_CLEAR);
                println!("CLEARED");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                app.set_action(ACTION_RANDOM);
                println!("RANDOMIZED");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("g") => {
                app.set_action(ACTION_GLIDER);
                println!("GLIDER SPAWNED");
            }
            Key::Named(NamedKey::ArrowUp) => {
                app.adjust_speed(5.0);
                println!("Speed: {} gen/s", app.simulation_state().speed as u32);
            }
            Key::Named(NamedKey::ArrowDown) => {
                app.adjust_speed(-5.0);
                println!("Speed: {} gen/s", app.simulation_state().speed as u32);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for GameOfLifeDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            .with_title("Game of Life - GPU-Native OS (GpuApp Framework)");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                if let Some(app) = &self.app {
                    let state = app.simulation_state();
                    println!();
                    println!("Final Stats:");
                    println!("  Generations: {}", state.generation);
                    println!("  Population: {}", state.population);
                }
                if let Some(runtime) = &self.runtime {
                    println!("  Total Frames: {}", runtime.frame_count());
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
    println!("Starting Game of Life Demo (GpuApp Framework)...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = GameOfLifeDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
