// GPU-Native OS Demo
// Demonstrates the widget rendering pipeline using Metal compute + fragment shaders

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::{
    memory::{GpuMemory, WidgetCompact},
    render::FrameRenderer,
    text::FontAtlas,
    vsync::{DisplayManager, PerformanceMonitor},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct GpuOsDemo {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    display_manager: Option<DisplayManager>,
    frame_renderer: Option<FrameRenderer>,
    gpu_memory: Option<GpuMemory>,
    font_atlas: Option<FontAtlas>,
    perf_monitor: PerformanceMonitor,
}

impl GpuOsDemo {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            display_manager: None,
            frame_renderer: None,
            gpu_memory: None,
            font_atlas: None,
            perf_monitor: PerformanceMonitor::new(),
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("GPU-Native OS Demo");
        println!("==================");
        println!("GPU: {}", device.name());
        println!("Unified Memory: {}", device.has_unified_memory());

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
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        // Initialize GPU-OS components
        let display_manager = DisplayManager::new(&device);
        let frame_renderer = FrameRenderer::new(&device).expect("Failed to create frame renderer");
        let gpu_memory = GpuMemory::new(&device, 1024);
        let font_atlas = FontAtlas::create_default(&device).expect("Failed to create font atlas");

        // Create demo widgets
        let widgets = Self::create_demo_widgets();
        gpu_memory.write_widgets(&widgets);

        println!("Initialized with {} widgets", gpu_memory.widget_count());

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.display_manager = Some(display_manager);
        self.frame_renderer = Some(frame_renderer);
        self.gpu_memory = Some(gpu_memory);
        self.font_atlas = Some(font_atlas);
    }

    fn create_demo_widgets() -> Vec<WidgetCompact> {
        let mut widgets = Vec::new();

        // Colors as RGB floats - bright colors for testing
        let dark_gray = [0.2, 0.2, 0.25];
        let blue = [0.3, 0.5, 1.0];        // Bright blue header
        let light_gray = [0.35, 0.35, 0.4];
        let darker = [0.15, 0.15, 0.2];
        let green = [0.2, 1.0, 0.3];       // Bright green
        let red = [1.0, 0.2, 0.2];         // Bright red
        let yellow = [1.0, 0.9, 0.2];      // Bright yellow
        let card_color = [0.25, 0.25, 0.3];
        let border_color = [0.5, 0.5, 0.6];

        // Background panel
        let mut bg = WidgetCompact::new(0.05, 0.05, 0.9, 0.9);
        bg.set_colors(dark_gray, dark_gray);
        bg.z_order = 0;
        widgets.push(bg);

        // Header bar
        let mut header = WidgetCompact::new(0.05, 0.05, 0.9, 0.08);
        header.set_colors(blue, blue);
        header.set_corner_radius(8);
        header.z_order = 1;
        widgets.push(header);

        // Sidebar
        let mut sidebar = WidgetCompact::new(0.05, 0.15, 0.2, 0.78);
        sidebar.set_colors(light_gray, light_gray);
        sidebar.set_corner_radius(4);
        sidebar.z_order = 1;
        widgets.push(sidebar);

        // Main content area
        let mut content = WidgetCompact::new(0.27, 0.15, 0.68, 0.78);
        content.set_colors(darker, border_color);
        content.set_corner_radius(8);
        content.set_border_width(1);
        content.z_order = 1;
        widgets.push(content);

        // Sidebar buttons
        let button_colors = [green, red, yellow];
        for (i, &color) in button_colors.iter().enumerate() {
            let mut btn = WidgetCompact::new(0.07, 0.18 + i as f32 * 0.08, 0.16, 0.06);
            btn.set_colors(color, color);
            btn.set_corner_radius(12);
            btn.z_order = 2;
            widgets.push(btn);
        }

        // Content cards (2 rows x 3 columns)
        for i in 0..6 {
            let row = i / 3;
            let col = i % 3;
            let x = 0.30 + col as f32 * 0.21;
            let y = 0.18 + row as f32 * 0.38;

            let mut card = WidgetCompact::new(x, y, 0.19, 0.34);
            card.set_colors(card_color, border_color);
            card.set_corner_radius(8);
            card.set_border_width(1);
            card.z_order = 2;
            widgets.push(card);
        }

        // Status bar at bottom
        let mut status = WidgetCompact::new(0.05, 0.91, 0.9, 0.03);
        status.set_colors(darker, darker);
        status.set_corner_radius(4);
        status.z_order = 1;
        widgets.push(status);

        widgets
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let display_manager = self.display_manager.as_mut().unwrap();
        let frame_renderer = self.frame_renderer.as_ref().unwrap();
        let gpu_memory = self.gpu_memory.as_ref().unwrap();
        let font_atlas = self.font_atlas.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // Begin frame timing
        let frame_ctx = display_manager.begin_frame();

        // Get command buffer
        let command_buffer = display_manager.new_command_buffer();

        // Render the frame using hybrid compute+fragment pipeline
        frame_renderer.render_frame(&command_buffer, &drawable, gpu_memory, font_atlas);

        command_buffer.commit();

        // End frame and record timing
        let timing = display_manager.end_frame(frame_ctx);
        self.perf_monitor.record_frame(&timing);

        // Print stats every 120 frames
        if display_manager.frame_count() % 120 == 0 {
            println!(
                "Frame {}: {:.2} FPS, {:.2}ms avg, {:.1}% dropped",
                display_manager.frame_count(),
                self.perf_monitor.average_fps(),
                timing.total_ms,
                self.perf_monitor.drop_rate()
            );
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }
    }
}

impl ApplicationHandler for GpuOsDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800))
            .with_title("GPU-Native OS Demo - Hybrid Rendering");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!("\nFinal Stats:");
                println!("  Total frames: {}", self.perf_monitor.frame_count);
                println!("  Average FPS: {:.2}", self.perf_monitor.average_fps());
                println!("  Min frame time: {:.2}ms", self.perf_monitor.min_frame_time_ms);
                println!("  Max frame time: {:.2}ms", self.perf_monitor.max_frame_time_ms);
                println!("  Dropped frames: {} ({:.1}%)",
                    self.perf_monitor.dropped_frames,
                    self.perf_monitor.drop_rate()
                );
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
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
    println!("Starting GPU-Native OS Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = GpuOsDemo::new();
    event_loop.run_app(&mut app).unwrap();
}
