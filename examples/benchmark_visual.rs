// GPU vs CPU Benchmark Visual Demo
//
// This is the "publishable demo" that proves the GPU-Native OS thesis visually.
// Shows a split-screen comparison of GPU vs CPU performance for widget operations.
//
// Controls:
// - Up/Down Arrow: Increase/decrease widget count (64 -> 2048)
// - Space: Toggle between sort benchmark and hit-test benchmark
// - R: Randomize widget positions and z-orders
// - Escape: Exit
//
// The demo clearly shows:
// - At low widget counts (~64), CPU and GPU are similar
// - At high widget counts (~2048), GPU is dramatically faster
// - The "crossover point" where GPU becomes the clear winner

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::benchmark_visual::BenchmarkVisual;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct BenchmarkDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    app: Option<BenchmarkVisual>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl BenchmarkDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            app: None,
            window_size: (1280, 800),
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");

        println!();
        println!("=============================================================");
        println!("     GPU vs CPU BENCHMARK - Visual Performance Demo");
        println!("=============================================================");
        println!();
        println!("GPU: {}", device.name());
        println!();
        println!("This demo proves the GPU-Native OS thesis by showing");
        println!("real-time GPU vs CPU performance for widget operations.");
        println!();
        println!("CONTROLS:");
        println!("  Up/Down Arrow  - Change widget count (64 to 2048)");
        println!("  Space          - Toggle Sort / Hit-Test benchmark");
        println!("  R              - Randomize widget positions");
        println!("  Escape         - Exit");
        println!();
        println!("WATCH FOR:");
        println!("  - At 64 widgets: CPU and GPU times are similar");
        println!("  - At 512+ widgets: GPU pulls ahead");
        println!("  - At 2048 widgets: GPU is 10-50x faster!");
        println!();
        println!("The GREEN bar shows GPU time, RED bar shows CPU time.");
        println!("The big number at the bottom shows the speedup ratio.");
        println!();
        println!("=============================================================");
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
        let app = BenchmarkVisual::new(&device).expect("Failed to create benchmark");

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
        if self.frame_count % 120 == 0 {
            let speedup = app.speedup();
            let indicator = if speedup > 2.0 {
                "GPU WINNING!"
            } else if speedup > 1.2 {
                "GPU ahead"
            } else if speedup < 0.8 {
                "CPU ahead"
            } else {
                "~equal"
            };

            println!(
                "Widgets: {:>4} | Mode: {:>8} | Speedup: {:>5.1}x | {}",
                app.widget_count(),
                app.mode_name(),
                speedup,
                indicator
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

    fn handle_key(&mut self, key: Key, event_loop: &ActiveEventLoop) {
        match key {
            Key::Named(NamedKey::Escape) => {
                self.print_final_stats();
                event_loop.exit();
            }
            Key::Named(NamedKey::ArrowUp) => {
                if let Some(app) = &mut self.app {
                    app.increase_widget_count();
                }
            }
            Key::Named(NamedKey::ArrowDown) => {
                if let Some(app) = &mut self.app {
                    app.decrease_widget_count();
                }
            }
            Key::Named(NamedKey::Space) => {
                if let Some(app) = &mut self.app {
                    app.toggle_mode();
                }
            }
            Key::Character(ref c) if c.eq_ignore_ascii_case("r") => {
                if let Some(app) = &mut self.app {
                    app.randomize();
                }
            }
            _ => {}
        }
    }

    fn print_final_stats(&self) {
        println!();
        println!("=============================================================");
        println!("                    BENCHMARK COMPLETE");
        println!("=============================================================");
        println!();
        println!("Total Frames: {}", self.frame_count);

        if let Some(app) = &self.app {
            println!("Final Widget Count: {}", app.widget_count());
            println!("Final Mode: {}", app.mode_name());
            println!("Final Speedup: {:.1}x", app.speedup());
            println!();

            if app.speedup() > 2.0 {
                println!("CONCLUSION: GPU provides significant performance advantage");
                println!("            at higher widget counts, validating the");
                println!("            GPU-Native OS architecture.");
            } else {
                println!("TIP: Increase widget count to see GPU advantage!");
            }
        }

        println!();
        println!("=============================================================");
    }
}

impl ApplicationHandler for BenchmarkDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
            .with_title("GPU vs CPU Benchmark - Visual Performance Demo");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                self.print_final_stats();
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_move(position.x, position.y);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.handle_key(event.logical_key, event_loop);
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = BenchmarkDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
