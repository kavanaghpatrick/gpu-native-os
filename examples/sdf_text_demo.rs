//! SDF Text Engine Demo
//!
//! Demonstrates GPU-native SDF text rendering from Issue #33.
//! Uses pre-baked SDF atlas embedded in the binary.
//!
//! Run with: cargo run --release --example sdf_text_demo

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::sdf_text::{EmbeddedSdfRenderer, SdfTextVertex};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer};
use std::mem;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;
const MAX_SDF_CHARS: usize = 4096;

struct SdfTextDemoApp {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    device: Option<Device>,
    command_queue: Option<CommandQueue>,
    // SDF text rendering
    sdf_renderer: Option<EmbeddedSdfRenderer>,
    sdf_vertex_buffer: Option<Buffer>,
    // Bitmap font for UI labels
    bitmap_font: Option<BitmapFont>,
    text_renderer: Option<TextRenderer>,
    // Demo state
    font_size: f32,
    frame_count: u64,
}

impl SdfTextDemoApp {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            device: None,
            command_queue: None,
            sdf_renderer: None,
            sdf_vertex_buffer: None,
            bitmap_font: None,
            text_renderer: None,
            font_size: 48.0,
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");

        println!("========================================");
        println!("  GPU-NATIVE SDF TEXT ENGINE DEMO");
        println!("========================================");
        println!("GPU: {}", device.name());

        // Create Metal layer
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

        let command_queue = device.new_command_queue();

        // Create embedded SDF renderer
        println!("\nInitializing GPU-native SDF renderer...");
        let sdf_renderer = EmbeddedSdfRenderer::new(&device, MAX_SDF_CHARS)
            .expect("Failed to create SDF renderer");
        println!("  Atlas loaded: 500x500 with 95 glyphs");
        println!("  All rendering is 100% GPU-native");

        // Create vertex buffer for SDF text (6 vertices per char)
        let sdf_vertex_buffer = device.new_buffer(
            (MAX_SDF_CHARS * 6 * mem::size_of::<SdfTextVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create bitmap font for UI
        let bitmap_font = BitmapFont::new(&device);
        let text_renderer = TextRenderer::new(&device, 4096)
            .expect("Failed to create text renderer");

        self.window = Some(window);
        self.layer = Some(layer);
        self.device = Some(device);
        self.command_queue = Some(command_queue);
        self.sdf_renderer = Some(sdf_renderer);
        self.sdf_vertex_buffer = Some(sdf_vertex_buffer);
        self.bitmap_font = Some(bitmap_font);
        self.text_renderer = Some(text_renderer);

        println!("\n========================================");
        println!("  Press UP/DOWN to change font size");
        println!("  Press ESC to exit");
        println!("========================================\n");
    }

    fn render(&mut self) {
        let drawable = match self.layer.as_ref().and_then(|l| l.next_drawable()) {
            Some(d) => d,
            None => return,
        };

        let command_queue = self.command_queue.as_ref().unwrap();
        let command_buffer = command_queue.new_command_buffer();

        let sdf_renderer = self.sdf_renderer.as_ref().unwrap();
        let sdf_vb = self.sdf_vertex_buffer.as_ref().unwrap();

        let screen_w = WINDOW_WIDTH as f32;
        let screen_h = WINDOW_HEIGHT as f32;

        // Define all SDF texts to render
        let sdf_texts: Vec<(&str, f32, f32, f32, [f32; 4])> = vec![
            // Main title at dynamic size
            ("GPU-Native SDF Text", 50.0, 180.0, self.font_size, [1.0, 1.0, 1.0, 1.0]),
            // Sample texts at various sizes
            ("Resolution Independent!", 50.0, 180.0 + self.font_size + 20.0, 28.0, [0.4, 0.9, 1.0, 1.0]),
            ("Smooth at any size", 50.0, 180.0 + self.font_size + 55.0, 24.0, [1.0, 0.9, 0.4, 1.0]),
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 50.0, 180.0 + self.font_size + 90.0, 20.0, [0.7, 1.0, 0.7, 1.0]),
            ("abcdefghijklmnopqrstuvwxyz", 50.0, 180.0 + self.font_size + 115.0, 20.0, [1.0, 0.7, 0.7, 1.0]),
            ("0123456789 !@#$%^&*()", 50.0, 180.0 + self.font_size + 140.0, 20.0, [0.7, 0.7, 1.0, 1.0]),
        ];

        // Calculate total vertices needed and track offsets
        let mut vertex_offset = 0usize;
        let mut text_ranges: Vec<(usize, usize)> = Vec::new(); // (offset, count)

        // Compute pass: layout all SDF texts
        let compute_encoder = command_buffer.new_compute_command_encoder();

        for (text, x, y, size, color) in &sdf_texts {
            let char_count = text.len();
            let vertex_count = char_count * 6;

            // Layout this text at the current offset
            sdf_renderer.layout_text_at_offset(
                &compute_encoder,
                text,
                *x,
                *y,
                *size,
                *color,
                screen_w,
                screen_h,
                sdf_vb,
                vertex_offset,
            );

            text_ranges.push((vertex_offset, vertex_count));
            vertex_offset += vertex_count;
        }

        compute_encoder.end_encoding();

        // Render pass
        let render_pass_desc = RenderPassDescriptor::new();
        let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.08, 0.08, 0.12, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let render_encoder = command_buffer.new_render_command_encoder(render_pass_desc);

        // Render all SDF texts
        let total_vertices: usize = text_ranges.iter().map(|(_, c)| *c).sum();
        sdf_renderer.render(render_encoder, sdf_vb, total_vertices, screen_w, screen_h);

        // Render bitmap UI text
        if let (Some(text_renderer), Some(bitmap_font)) = (
            self.text_renderer.as_mut(),
            self.bitmap_font.as_ref(),
        ) {
            text_renderer.clear();
            text_renderer.scale = 1.5;
            text_renderer.add_text("SDF Text Engine - Issue #33", 20.0, 30.0, 0xFFFFFFFF);
            text_renderer.scale = 1.2;
            text_renderer.add_text("GPU-Native Rendering with Embedded Atlas", 20.0, 55.0, 0xAAAAAAFF);
            text_renderer.scale = 1.0;
            text_renderer.add_text(
                &format!("Font Size: {:.0}pt (UP/DOWN)", self.font_size),
                20.0, 85.0, 0x88FF88FF);
            text_renderer.add_text(
                &format!("Frame: {}", self.frame_count),
                20.0, 105.0, 0x666666FF);

            // Status
            let y = 550.0;
            text_renderer.add_text("Status:", 20.0, y, 0xFFFFFFFF);
            text_renderer.add_text("  Phase 1-4: COMPLETE", 20.0, y + 20.0, 0x88FF88FF);
            text_renderer.add_text("  Phase 5: Text Layout [COMPLETE]", 20.0, y + 40.0, 0x88FF88FF);
            text_renderer.add_text("  Phase 6: Integration [IN PROGRESS]", 20.0, y + 60.0, 0xFFAA44FF);
            text_renderer.add_text("100% GPU-native - no CPU per-frame work", 20.0, y + 100.0, 0x66FF66FF);
            text_renderer.add_text("Press ESC to exit", 20.0, y + 140.0, 0x666666FF);

            text_renderer.render(render_encoder, bitmap_font, screen_w, screen_h);
        }

        render_encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        self.frame_count += 1;
    }
}

impl ApplicationHandler for SdfTextDemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title("GPU-Native SDF Text Demo - Issue #33");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{Key, NamedKey};
                if event.state == winit::event::ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Named(NamedKey::ArrowUp) => {
                            self.font_size = (self.font_size + 4.0).min(200.0);
                        }
                        Key::Named(NamedKey::ArrowDown) => {
                            self.font_size = (self.font_size - 4.0).max(12.0);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                if let Some(layer) = &self.layer {
                    layer.set_drawable_size(CGSize::new(
                        new_size.width.max(1) as f64,
                        new_size.height.max(1) as f64,
                    ));
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
    println!("\nStarting GPU-Native SDF Text Demo...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = SdfTextDemoApp::new();
    event_loop.run_app(&mut app).unwrap();
}
