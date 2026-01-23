// Game of Life Demo
//
// A 32x32 Game of Life running entirely on the GPU in a single threadgroup.
// All simulation, input handling, and rendering happens in one compute dispatch.
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
use rust_experiment::gpu_os::game_of_life::{
    GameOfLife, FrameParams, ACTION_CLEAR, ACTION_RANDOM, ACTION_GLIDER, ACTION_STEP,
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

struct GameOfLifeApp {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    game: Option<GameOfLife>,
    last_frame: Instant,
    cursor_x: f32,
    cursor_y: f32,
    mouse_down: bool,
    mouse_clicked: bool,
    window_size: (u32, u32),
}

impl GameOfLifeApp {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            game: None,
            last_frame: Instant::now(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            mouse_down: false,
            mouse_clicked: false,
            window_size: (800, 800),
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("Game of Life - GPU-Native Demo");
        println!("==============================");
        println!("GPU: {}", device.name());
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

        // Create game
        let game = GameOfLife::new(&device).expect("Failed to create Game of Life");

        // Start with a random pattern
        game.simulation_state_mut().pending_action = ACTION_RANDOM;

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.game = Some(game);
        self.last_frame = Instant::now();
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let game = self.game.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // Calculate delta time
        let now = Instant::now();
        let delta_time = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Update frame parameters
        let params = FrameParams {
            delta_time,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            mouse_down: if self.mouse_down { 1 } else { 0 },
            mouse_clicked: if self.mouse_clicked { 1 } else { 0 },
        };
        game.update_params(&params);

        // Clear click flag after sending
        self.mouse_clicked = false;

        // Render
        game.render(&drawable);

        // Print stats occasionally
        let state = game.simulation_state();
        if state.generation % 60 == 0 && state.generation > 0 {
            println!(
                "Gen: {:>6}  Pop: {:>4}  Speed: {:>2} gen/s  {}",
                state.generation,
                state.population,
                state.speed as u32,
                if state.running != 0 { "RUNNING" } else { "PAUSED" }
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

            // Detect click (transition from pressed to released)
            if was_down && !self.mouse_down {
                self.mouse_clicked = true;
            }
        }
    }

    fn handle_key(&mut self, key: Key) {
        let game = match &self.game {
            Some(g) => g,
            None => return,
        };

        let sim = game.simulation_state_mut();

        match key {
            Key::Named(NamedKey::Space) => {
                sim.running = 1 - sim.running;
                println!("{}", if sim.running != 0 { "PLAYING" } else { "PAUSED" });
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("s") => {
                sim.pending_action = ACTION_STEP;
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("c") => {
                sim.pending_action = ACTION_CLEAR;
                println!("CLEARED");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                sim.pending_action = ACTION_RANDOM;
                println!("RANDOMIZED");
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("g") => {
                sim.pending_action = ACTION_GLIDER;
                println!("GLIDER SPAWNED");
            }
            Key::Named(NamedKey::ArrowUp) => {
                sim.speed = (sim.speed + 5.0).min(60.0);
                println!("Speed: {} gen/s", sim.speed as u32);
            }
            Key::Named(NamedKey::ArrowDown) => {
                sim.speed = (sim.speed - 5.0).max(1.0);
                println!("Speed: {} gen/s", sim.speed as u32);
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for GameOfLifeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
            .with_title("Game of Life - GPU-Native OS Demo");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                if let Some(game) = &self.game {
                    let state = game.simulation_state();
                    println!();
                    println!("Final Stats:");
                    println!("  Generations: {}", state.generation);
                    println!("  Population: {}", state.population);
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
    println!("Starting Game of Life Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = GameOfLifeApp::new();
    event_loop.run_app(&mut app).unwrap();
}
