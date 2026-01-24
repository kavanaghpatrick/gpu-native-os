//! GPU Image Rendering Demo
//!
//! Demonstrates GPU-native image loading and rendering for documents.
//! Uses Metal's MTKTextureLoader for GPU-accelerated image decoding.
//!
//! Run with: cargo run --release --example document_images

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::document::{
    GpuImageAtlas, GpuImageLoader, ImageInfo, FLAG_IMAGE, ATLAS_WIDTH, ATLAS_HEIGHT,
};
use std::path::Path;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;

/// Vertex for image rendering
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ImageVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: [f32; 4],
}

struct ImageDemoApp {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    device: Option<Device>,
    command_queue: Option<CommandQueue>,
    render_pipeline: Option<RenderPipelineState>,
    vertex_buffer: Option<Buffer>,
    image_atlas: Option<GpuImageAtlas>,
    vertex_count: usize,
    images_loaded: bool,
}

impl ImageDemoApp {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            device: None,
            command_queue: None,
            render_pipeline: None,
            vertex_buffer: None,
            image_atlas: None,
            vertex_count: 0,
            images_loaded: false,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");

        println!("========================================");
        println!("  GPU IMAGE RENDERING DEMO");
        println!("========================================");
        println!("GPU: {}", device.name());
        println!("\nFeatures:");
        println!("  - GPU-native image loading (MTKTextureLoader)");
        println!("  - Texture atlas for efficient batching");
        println!("  - GPU vertex generation");

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

        // Create render pipeline for textured quads
        let render_pipeline = self.create_render_pipeline(&device);

        // Create image atlas
        println!("\nInitializing GPU image atlas...");
        let image_atlas = GpuImageAtlas::new(&device).expect("Failed to create image atlas");
        println!("  Atlas size: {}x{}", ATLAS_WIDTH, ATLAS_HEIGHT);

        // Create vertex buffer (enough for many images)
        let vertex_buffer = device.new_buffer(
            (1024 * std::mem::size_of::<ImageVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        self.window = Some(window);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.render_pipeline = Some(render_pipeline);
        self.vertex_buffer = Some(vertex_buffer);
        self.image_atlas = Some(image_atlas);
        self.device = Some(device);

        // Load test images
        self.load_test_images();
    }

    fn create_render_pipeline(&self, device: &Device) -> RenderPipelineState {
        let shader_source = r#"
            #include <metal_stdlib>
            using namespace metal;

            struct VertexIn {
                float2 position [[attribute(0)]];
                float2 tex_coord [[attribute(1)]];
                float4 color [[attribute(2)]];
            };

            struct VertexOut {
                float4 position [[position]];
                float2 tex_coord;
                float4 color;
            };

            vertex VertexOut vertex_main(
                const device VertexIn* vertices [[buffer(0)]],
                uint vid [[vertex_id]]
            ) {
                VertexOut out;
                out.position = float4(vertices[vid].position, 0.0, 1.0);
                out.tex_coord = vertices[vid].tex_coord;
                out.color = vertices[vid].color;
                return out;
            }

            fragment float4 fragment_main(
                VertexOut in [[stage_in]],
                texture2d<float> atlas [[texture(0)]]
            ) {
                constexpr sampler samp(mag_filter::linear, min_filter::linear);
                float4 tex_color = atlas.sample(samp, in.tex_coord);
                return tex_color * in.color;
            }
        "#;

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shader_source, &options)
            .expect("Failed to compile shader");

        let vertex_fn = library.get_function("vertex_main", None).unwrap();
        let fragment_fn = library.get_function("fragment_main", None).unwrap();

        let desc = RenderPipelineDescriptor::new();
        desc.set_vertex_function(Some(&vertex_fn));
        desc.set_fragment_function(Some(&fragment_fn));

        let attachment = desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        device
            .new_render_pipeline_state(&desc)
            .expect("Failed to create render pipeline")
    }

    fn load_test_images(&mut self) {
        let device = self.device.as_ref().unwrap();
        let command_queue = self.command_queue.as_ref().unwrap();
        let atlas = self.image_atlas.as_mut().unwrap();

        println!("\nLoading test images...");

        // Create test images programmatically (colored rectangles)
        let loader = GpuImageLoader::new(device).expect("Failed to create image loader");

        // Generate test image 1: Red gradient
        let mut rgba1 = vec![0u8; 200 * 150 * 4];
        for y in 0..150 {
            for x in 0..200 {
                let idx = (y * 200 + x) * 4;
                rgba1[idx] = 255;  // R
                rgba1[idx + 1] = (x as f32 / 200.0 * 255.0) as u8;  // G
                rgba1[idx + 2] = (y as f32 / 150.0 * 255.0) as u8;  // B
                rgba1[idx + 3] = 255;  // A
            }
        }
        let tex1 = loader.load_from_rgba(&rgba1, 200, 150).expect("Failed to create texture 1");
        let info1 = atlas.add_image(&tex1, "test_red", command_queue).expect("Failed to add image 1");
        println!("  Image 1: {}x{} at atlas ({}, {})", info1.width, info1.height, info1.atlas_x, info1.atlas_y);

        // Generate test image 2: Blue gradient
        let mut rgba2 = vec![0u8; 180 * 180 * 4];
        for y in 0..180 {
            for x in 0..180 {
                let idx = (y * 180 + x) * 4;
                rgba2[idx] = (x as f32 / 180.0 * 255.0) as u8;  // R
                rgba2[idx + 1] = (y as f32 / 180.0 * 255.0) as u8;  // G
                rgba2[idx + 2] = 255;  // B
                rgba2[idx + 3] = 255;  // A
            }
        }
        let tex2 = loader.load_from_rgba(&rgba2, 180, 180).expect("Failed to create texture 2");
        let info2 = atlas.add_image(&tex2, "test_blue", command_queue).expect("Failed to add image 2");
        println!("  Image 2: {}x{} at atlas ({}, {})", info2.width, info2.height, info2.atlas_x, info2.atlas_y);

        // Generate test image 3: Green checkerboard
        let mut rgba3 = vec![0u8; 160 * 120 * 4];
        for y in 0..120 {
            for x in 0..160 {
                let idx = (y * 160 + x) * 4;
                let checker = ((x / 20) + (y / 20)) % 2 == 0;
                if checker {
                    rgba3[idx] = 50;  // R
                    rgba3[idx + 1] = 200;  // G
                    rgba3[idx + 2] = 50;  // B
                } else {
                    rgba3[idx] = 200;  // R
                    rgba3[idx + 1] = 255;  // G
                    rgba3[idx + 2] = 200;  // B
                }
                rgba3[idx + 3] = 255;  // A
            }
        }
        let tex3 = loader.load_from_rgba(&rgba3, 160, 120).expect("Failed to create texture 3");
        let info3 = atlas.add_image(&tex3, "test_checker", command_queue).expect("Failed to add image 3");
        println!("  Image 3: {}x{} at atlas ({}, {})", info3.width, info3.height, info3.atlas_x, info3.atlas_y);

        // Generate vertices for displaying images
        self.generate_vertices(&[info1, info2, info3]);
        self.images_loaded = true;

        println!("\nImages loaded and ready for GPU rendering!");
    }

    fn generate_vertices(&mut self, images: &[ImageInfo]) {
        let buffer = self.vertex_buffer.as_ref().unwrap();
        let ptr = buffer.contents() as *mut ImageVertex;

        let width = WINDOW_WIDTH as f32;
        let height = WINDOW_HEIGHT as f32;

        let mut vertices = Vec::new();
        let mut x_offset = 50.0;
        let y_offset = 100.0;

        for img in images {
            // Calculate UV coordinates from atlas
            let u0 = img.atlas_x as f32 / ATLAS_WIDTH as f32;
            let v0 = img.atlas_y as f32 / ATLAS_HEIGHT as f32;
            let u1 = (img.atlas_x + img.atlas_width) as f32 / ATLAS_WIDTH as f32;
            let v1 = (img.atlas_y + img.atlas_height) as f32 / ATLAS_HEIGHT as f32;

            // Convert pixel coords to NDC
            let scale_x = 2.0 / width;
            let scale_y = -2.0 / height;
            let bias_x = -1.0;
            let bias_y = 1.0;

            let left = x_offset * scale_x + bias_x;
            let right = (x_offset + img.width as f32) * scale_x + bias_x;
            let top = y_offset * scale_y + bias_y;
            let bottom = (y_offset + img.height as f32) * scale_y + bias_y;

            let color = [1.0f32, 1.0, 1.0, 1.0];

            // Two triangles for quad
            vertices.push(ImageVertex { position: [left, top], tex_coord: [u0, v0], color });
            vertices.push(ImageVertex { position: [right, top], tex_coord: [u1, v0], color });
            vertices.push(ImageVertex { position: [right, bottom], tex_coord: [u1, v1], color });

            vertices.push(ImageVertex { position: [left, top], tex_coord: [u0, v0], color });
            vertices.push(ImageVertex { position: [right, bottom], tex_coord: [u1, v1], color });
            vertices.push(ImageVertex { position: [left, bottom], tex_coord: [u0, v1], color });

            x_offset += img.width as f32 + 50.0;
        }

        // Copy to buffer
        unsafe {
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), ptr, vertices.len());
        }
        self.vertex_count = vertices.len();
    }

    fn render(&mut self) {
        if !self.images_loaded {
            return;
        }

        let layer = self.layer.as_ref().unwrap();
        let command_queue = self.command_queue.as_ref().unwrap();
        let pipeline = self.render_pipeline.as_ref().unwrap();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let atlas = self.image_atlas.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        let command_buffer = command_queue.new_command_buffer();

        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));

        let encoder = command_buffer.new_render_command_encoder(&render_desc);

        encoder.set_render_pipeline_state(pipeline);
        encoder.set_vertex_buffer(0, Some(vertex_buffer), 0);
        encoder.set_fragment_texture(0, Some(atlas.texture()));

        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, self.vertex_count as u64);

        encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}

impl ApplicationHandler for ImageDemoApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("GPU Image Rendering Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT));

            let window = event_loop.create_window(attrs).unwrap();
            self.initialize(window);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        autoreleasepool(|| {
            match event {
                WindowEvent::CloseRequested => {
                    println!("\nDemo complete.");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.render();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::Resized(size) => {
                    if let Some(layer) = &self.layer {
                        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
                    }
                }
                _ => {}
            }
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = ImageDemoApp::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
