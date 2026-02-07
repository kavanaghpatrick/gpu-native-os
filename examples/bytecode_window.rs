//! Bytecode Window Demo
//!
//! Opens a window and renders bytecode-generated quads visually.

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::runtime::YES;
use rust_experiment::gpu_os::gpu_app_system::*;
use std::mem;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct App {
    window: Option<Window>,
    device: Device,
    queue: CommandQueue,
    layer: MetalLayer,
    render_pipeline: RenderPipelineState,
    vertex_buffer: Buffer,
    vertex_count: u32,
}

impl App {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device");
        let queue = device.new_command_queue();
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        // Create simple render pipeline for drawing vertices
        // CRITICAL: Use packed_float3 (12 bytes) to match Rust [f32; 3]
        // Metal float3 is 16 bytes which breaks vertex stride alignment!
        let shader_src = r#"
            #include <metal_stdlib>
            using namespace metal;

            struct RenderVertex {
                packed_float3 position;  // 12 bytes - matches Rust [f32; 3]
                float _pad0;             // 4 bytes to align to 16
                float4 color;
                float2 uv;
                float2 _pad1;
            };

            struct VertexOut {
                float4 position [[position]];
                float4 color;
            };

            vertex VertexOut vertex_main(
                const device RenderVertex* vertices [[buffer(0)]],
                constant float2& screen_size [[buffer(1)]],
                uint vid [[vertex_id]]
            ) {
                RenderVertex v = vertices[vid];

                // Transform to normalized device coordinates
                float2 pos = v.position.xy / screen_size;
                pos = pos * 2.0 - 1.0;
                pos.y = -pos.y;  // Flip Y (Metal NDC has Y up, screen has Y down)

                VertexOut out;
                out.position = float4(pos, v.position.z, 1.0);
                out.color = v.color;
                return out;
            }

            fragment float4 fragment_main(VertexOut in [[stage_in]]) {
                return in.color;
            }
        "#;

        let library = device.new_library_with_source(shader_src, &CompileOptions::new())
            .expect("Failed to compile render shader");

        let vertex_fn = library.get_function("vertex_main", None).unwrap();
        let fragment_fn = library.get_function("fragment_main", None).unwrap();

        let desc = RenderPipelineDescriptor::new();
        desc.set_vertex_function(Some(&vertex_fn));
        desc.set_fragment_function(Some(&fragment_fn));
        desc.color_attachments().object_at(0).unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let render_pipeline = device.new_render_pipeline_state(&desc)
            .expect("Failed to create render pipeline");

        // Generate bytecode and run on GPU
        let (vertex_buffer, vertex_count) = Self::run_bytecode(&device, &queue);

        Self {
            window: None,
            device,
            queue,
            layer,
            render_pipeline,
            vertex_buffer,
            vertex_count,
        }
    }

    fn run_bytecode(device: &Device, queue: &CommandQueue) -> (Buffer, u32) {
        println!("Compiling bytecode shader...");

        let library = device.new_library_with_source(
            GPU_APP_SYSTEM_SHADER,
            &CompileOptions::new(),
        ).expect("Failed to compile shader");

        let megakernel = library.get_function("gpu_app_megakernel_parallel", None)
            .expect("Failed to get megakernel");
        let pipeline = device.new_compute_pipeline_state_with_function(&megakernel)
            .expect("Failed to create pipeline");

        // Create bytecode program - colorful grid
        println!("Creating bytecode program...");
        let mut asm = BytecodeAssembler::new();

        let grid_size = 10;
        let cell_size = 60.0;
        let margin = 40.0;
        let gap = 10.0;

        for y in 0..grid_size {
            for x in 0..grid_size {
                let px = margin + (x as f32) * (cell_size + gap);
                let py = margin + (y as f32) * (cell_size + gap);

                // Position in r4 (xy)
                asm.setx(4, px);
                asm.sety(4, py);

                // Size in r5 (xy)
                asm.setx(5, cell_size);
                asm.sety(5, cell_size);

                // Color in r6 - must be packed u32 (0xRRGGBBAA format)
                let r_f = x as f32 / (grid_size as f32 - 1.0);
                let g_f = y as f32 / (grid_size as f32 - 1.0);
                let b_f = 1.0 - (r_f + g_f) / 2.0;
                let r = (r_f * 255.0) as u32;
                let g = (g_f * 255.0) as u32;
                let b = (b_f * 255.0) as u32;
                let a = 255u32;
                let packed_color = (r << 24) | (g << 16) | (b << 8) | a;
                asm.loadi_uint(6, packed_color);

                // quad(pos_reg, size_reg, color_reg, depth)
                asm.quad(4, 5, 6, 0.5);
            }
        }

        asm.halt();

        let vertex_budget = (grid_size * grid_size * 6) as u32;
        let bytecode = asm.build(vertex_budget);
        println!("Generated {} bytecode instructions", (bytecode.len() - 16) / 8);

        // Create buffers
        let header = AppTableHeader {
            max_slots: 64,
            active_count: 1,
            free_bitmap: [0xFFFFFFFE, 0xFFFFFFFF],
            _pad: [0; 4],
        };

        let header_buffer = device.new_buffer_with_data(
            &header as *const _ as *const _,
            mem::size_of::<AppTableHeader>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let state_size = bytecode.len() + 4096;
        let state_buffer = device.new_buffer(state_size as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = state_buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(bytecode.as_ptr(), ptr, bytecode.len());
        }

        let app = GpuAppDescriptor {
            flags: flags::ACTIVE | flags::VISIBLE | flags::DIRTY,
            app_type: app_type::BYTECODE,
            slot_id: 0,
            window_id: 0,
            state_offset: 0,
            state_size: state_size as u32,
            vertex_offset: 0,
            vertex_size: (vertex_budget as usize * mem::size_of::<RenderVertex>()) as u32,
            vertex_count: 0,
            priority: 1,
            thread_count: 1,
            ..Default::default()
        };

        let apps_buffer = device.new_buffer(
            (64 * mem::size_of::<GpuAppDescriptor>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = apps_buffer.contents() as *mut GpuAppDescriptor;
            *ptr = app;
        }

        let vertex_buffer = device.new_buffer(
            (vertex_budget as usize * mem::size_of::<RenderVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let window_buffer = device.new_buffer(
            (64 * mem::size_of::<GpuWindow>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let frame_number: u32 = 1;
        let frame_buffer = device.new_buffer_with_data(
            &frame_number as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let window_count: u32 = 0;
        let window_count_buffer = device.new_buffer_with_data(
            &window_count as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        // Run bytecode
        println!("Running bytecode on GPU...");
        let cmd_buffer = queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&header_buffer), 0);
        encoder.set_buffer(1, Some(&apps_buffer), 0);
        encoder.set_buffer(2, Some(&state_buffer), 0);
        encoder.set_buffer(3, Some(&frame_buffer), 0);
        encoder.set_buffer(4, Some(&vertex_buffer), 0);
        encoder.set_buffer(5, Some(&window_buffer), 0);
        encoder.set_buffer(6, Some(&window_count_buffer), 0);

        // Issue #236 fix: Buffer 7 = screen_size (width, height as float2)
        // Without this, all vertices end up at (0, 0) due to scaling by 0
        let screen_size: [f32; 2] = [800.0, 800.0];
        encoder.set_bytes(
            7,
            mem::size_of::<[f32; 2]>() as u64,
            &screen_size as *const _ as *const _,
        );

        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();

        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let app_result: &GpuAppDescriptor = unsafe {
            &*(apps_buffer.contents() as *const GpuAppDescriptor)
        };

        println!("Bytecode generated {} vertices", app_result.vertex_count);

        (vertex_buffer, app_result.vertex_count)
    }

    fn render(&self) {
        let Some(drawable) = self.layer.next_drawable() else { return };

        let desc = RenderPassDescriptor::new();
        let attachment = desc.color_attachments().object_at(0).unwrap();
        attachment.set_texture(Some(drawable.texture()));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));
        attachment.set_store_action(MTLStoreAction::Store);

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_render_command_encoder(&desc);

        encoder.set_render_pipeline_state(&self.render_pipeline);

        // Issue #240 fix: Explicitly set viewport
        encoder.set_viewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: 800.0,
            height: 800.0,
            znear: 0.0,
            zfar: 1.0,
        });

        encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);

        // Pass screen size
        let screen_size: [f32; 2] = [800.0, 800.0];
        encoder.set_vertex_bytes(
            1,
            mem::size_of::<[f32; 2]>() as u64,
            &screen_size as *const _ as *const _,
        );

        encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            self.vertex_count as u64,
        );

        encoder.end_encoding();
        cmd_buffer.present_drawable(drawable);
        cmd_buffer.commit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("GPU Bytecode Demo")
            .with_inner_size(winit::dpi::LogicalSize::new(800, 800));

        let window = event_loop.create_window(attrs).unwrap();

        // Set up Metal layer
        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                    let view = appkit_handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(self.layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        let size = window.inner_size();
        self.layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            WindowEvent::Resized(size) => {
                self.layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    println!("=== GPU Bytecode Window Demo ===\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
