// Visual Megakernel Demo - Issue #162
//
// Connects the GPU megakernel's unified vertex buffer to a real window.
// This is THE proof that GPU-generated geometry renders correctly with
// zero CPU involvement in the render loop.
//
// CPU only submits command buffer - all geometry comes from GPU.
//
// Controls:
//   Click dock - Launch app (GPU detects which dock item was clicked)
//   T          - Launch terminal
//   D          - Launch document viewer
//   B          - Launch bytecode demo (GPU-interpreted program)
//   Space      - Launch filesystem browser
//   Escape     - Exit

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::gpu_os::GpuOs;
use rust_experiment::gpu_os::gpu_app_system::{app_type, BytecodeAssembler};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Visual demo app state
struct VisualMegakernelDemo {
    // Window/Metal
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<CommandQueue>,
    render_pipeline: Option<RenderPipelineState>,
    depth_stencil_state: Option<DepthStencilState>,
    depth_texture: Option<Texture>,
    screen_size_buffer: Option<Buffer>,

    // GPU OS
    os: Option<GpuOs>,

    // State
    window_size: (u32, u32),
    cursor_x: f32,
    cursor_y: f32,
    frame_count: u64,
    last_frame: Instant,
}

impl VisualMegakernelDemo {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            command_queue: None,
            render_pipeline: None,
            depth_stencil_state: None,
            depth_texture: None,
            screen_size_buffer: None,
            os: None,
            window_size: (1280, 720),
            cursor_x: 0.0,
            cursor_y: 0.0,
            frame_count: 0,
            last_frame: Instant::now(),
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();

        println!("Visual Megakernel Demo");
        println!("======================");
        println!("GPU: {}", device.name());
        println!();
        println!("This demo proves the GPU megakernel architecture:");
        println!("- GPU generates ALL geometry");
        println!("- CPU only submits command buffer");
        println!("- Single draw call renders everything");
        println!();
        println!("Controls:");
        println!("  Click dock  - Launch app from dock");
        println!("  T           - Launch terminal");
        println!("  D           - Launch document viewer");
        println!("  B           - Launch bytecode demo (GPU-interpreted)");
        println!("  Space       - Launch filesystem browser");
        println!("  Escape      - Exit");
        println!();

        // Boot GPU OS
        let size = window.inner_size();
        self.window_size = (size.width, size.height);

        let mut os = GpuOs::boot_with_size(&device, size.width as f32, size.height as f32)
            .expect("Failed to boot GPU OS");

        println!("GPU OS booted:");
        println!("  Screen: {}x{}", size.width, size.height);
        println!("  System apps: compositor, dock, menubar, window chrome");
        println!("  Shared index: {}", if os.has_shared_index() { "loaded" } else { "not available" });
        println!("  I/O queue: {}", if os.has_io_queue() { "available" } else { "not available" });
        println!();

        // Launch a terminal app to show something
        if let Some(slot) = os.launch_app(app_type::TERMINAL) {
            println!("Launched terminal at slot {}", slot);
        }

        // Create render pipeline
        let library = device
            .new_library_with_source(RENDER_SHADER, &CompileOptions::new())
            .expect("Shader compilation failed");

        let vertex_fn = library
            .get_function("unified_vertex_shader", None)
            .expect("Vertex function not found");
        let fragment_fn = library
            .get_function("unified_fragment_shader", None)
            .expect("Fragment function not found");

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));

        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // Enable alpha blending
        color_attachment.set_blending_enabled(true);
        color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color_attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        color_attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        color_attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        // TEMPORARILY DISABLE depth testing to debug rendering
        // render_desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .expect("Pipeline creation failed");

        // Create depth stencil state (disabled for now)
        // Depth testing: higher depth values are in front (GreaterEqual)
        // Compositor background at depth 0.0, UI elements at 0.9+
        let depth_stencil_desc = DepthStencilDescriptor::new();
        depth_stencil_desc.set_depth_compare_function(MTLCompareFunction::GreaterEqual);
        depth_stencil_desc.set_depth_write_enabled(true);
        let depth_stencil_state = device.new_depth_stencil_state(&depth_stencil_desc);

        // Create depth texture (still create it but won't use)
        let depth_texture = Self::create_depth_texture(&device, size.width, size.height);

        // Screen size buffer (for coordinate transform)
        let screen_size_buffer = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = [self.window_size.0 as f32, self.window_size.1 as f32];
        }

        // Metal layer
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);
        layer.set_opaque(true);  // Ensure layer content is visible

        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                    let view = appkit_handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.render_pipeline = Some(render_pipeline);
        self.depth_stencil_state = Some(depth_stencil_state);
        self.depth_texture = Some(depth_texture);
        self.screen_size_buffer = Some(screen_size_buffer);
        self.os = Some(os);
        self.last_frame = Instant::now();

        println!("Visual demo initialized. Rendering...");
    }

    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        let desc = TextureDescriptor::new();
        desc.set_texture_type(MTLTextureType::D2);
        desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_storage_mode(MTLStorageMode::Private);
        desc.set_usage(MTLTextureUsage::RenderTarget);
        device.new_texture(&desc)
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let os = self.os.as_mut().unwrap();
        let pipeline = self.render_pipeline.as_ref().unwrap();
        let depth_stencil_state = self.depth_stencil_state.as_ref().unwrap();
        let depth_texture = self.depth_texture.as_ref().unwrap();
        let queue = self.command_queue.as_ref().unwrap();
        let screen_size_buffer = self.screen_size_buffer.as_ref().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // ================================================================
        // 1. Run GPU OS frame (megakernel generates vertices)
        // ================================================================
        os.run_frame();

        // ================================================================
        // 2. Get vertex count (GPU already computed this atomically)
        // ================================================================
        let vertex_count = os.total_vertex_count();

        // ================================================================
        // 3. Create command buffer
        // ================================================================
        let cmd = queue.new_command_buffer();

        // ================================================================
        // 4. Create render pass
        // ================================================================
        let render_desc = RenderPassDescriptor::new();
        let color = render_desc.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(drawable.texture()));
        color.set_load_action(MTLLoadAction::Clear);
        // Use MAGENTA clear color so we can see if compositor renders
        // If you see magenta, nothing is rendering. If you see dark gray, compositor worked.
        color.set_clear_color(MTLClearColor::new(1.0, 0.0, 1.0, 1.0));
        color.set_store_action(MTLStoreAction::Store);

        // Depth attachment - higher depth values are in front
        // Compositor at 0.0 (back), UI elements at 0.9+ (front)
        let depth = render_desc.depth_attachment().unwrap();
        depth.set_texture(Some(depth_texture));
        depth.set_load_action(MTLLoadAction::Clear);
        depth.set_store_action(MTLStoreAction::DontCare);
        depth.set_clear_depth(0.0);  // Clear to back (lowest depth)

        let encoder = cmd.new_render_command_encoder(&render_desc);
        encoder.set_depth_stencil_state(depth_stencil_state);
        encoder.set_render_pipeline_state(pipeline);

        // Issue #240 fix: Explicitly set viewport
        encoder.set_viewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: self.window_size.0 as f64,
            height: self.window_size.1 as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        // ================================================================
        // 5. Bind unified vertex buffer (GPU-generated)
        // ================================================================
        encoder.set_vertex_buffer(0, Some(os.render_vertices_buffer()), 0);
        encoder.set_vertex_buffer(1, Some(screen_size_buffer), 0);

        // ================================================================
        // 6. Draw each active app's actual vertices
        // ================================================================
        // Draw only the exact vertices each app wrote, not full slots
        // This avoids rendering garbage/uninitialized vertices
        let draw_calls = os.get_draw_calls();
        for (start_vertex, count) in draw_calls {
            if count > 0 {
                encoder.draw_primitives(MTLPrimitiveType::Triangle, start_vertex, count);
            }
        }

        encoder.end_encoding();

        // ================================================================
        // 7. Present
        // ================================================================
        cmd.present_drawable(drawable);
        cmd.commit();

        // ================================================================
        // Statistics
        // ================================================================
        self.frame_count += 1;

        // Save frame 10 to file for visual QA
        let texture_to_save = if self.frame_count == 10 {
            cmd.wait_until_completed();
            Some(drawable.texture().to_owned())
        } else {
            None
        };

        // Print stats every second
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame).as_secs_f32();
        if self.frame_count % 60 == 0 {
            let fps = 1.0 / elapsed;
            println!(
                "Frame {:>6} | {:>6} vertices | {:>5.1} FPS | {} apps",
                self.frame_count,
                vertex_count,
                fps,
                os.app_count()
            );
        }
        self.last_frame = now;

        // Save framebuffer after all os usage is done
        if let Some(tex) = texture_to_save {
            self.save_framebuffer(&tex);
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window_size = (width, height);

        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }

        // Update screen size buffer
        if let Some(buffer) = &self.screen_size_buffer {
            unsafe {
                let ptr = buffer.contents() as *mut [f32; 2];
                *ptr = [width as f32, height as f32];
            }
        }

        // Recreate depth texture with new size
        if let Some(device) = &self.device {
            self.depth_texture = Some(Self::create_depth_texture(device, width, height));
        }

        // Update OS screen size
        if let Some(os) = &mut self.os {
            os.screen_width = width as f32;
            os.screen_height = height as f32;
        }
    }

    fn handle_cursor_move(&mut self, x: f64, y: f64) {
        self.cursor_x = x as f32;
        self.cursor_y = y as f32;

        // Forward to GPU OS
        if let Some(os) = &mut self.os {
            os.mouse_move(self.cursor_x, self.cursor_y, 0.0, 0.0);
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            if let Some(os) = &mut self.os {
                // Forward click to OS (for dock hover/click detection)
                os.mouse_button(0, state == ElementState::Pressed, self.cursor_x, self.cursor_y);

                // On press, run a frame immediately so GPU can detect the click
                if state == ElementState::Pressed {
                    println!("DEBUG: Click at ({}, {})", self.cursor_x, self.cursor_y);
                    os.debug_dock_state();  // Before frame
                    os.run_frame();  // GPU processes mouse_pressed=1, sets clicked_item
                    os.debug_dock_state();  // After frame

                    // Debug: check dock state
                    if let Some(dock_slot) = os.dock_slot() {
                        let clicked = os.get_clicked_dock_app();
                        println!("DEBUG: dock_slot={}, clicked_app={:?}", dock_slot, clicked);
                    }

                    // Check for dock click immediately after GPU processes
                    if let Some(slot) = os.handle_dock_click() {
                        let app = os.system.get_app(slot);
                        let app_name = app.map(|a| match a.app_type {
                            1 => "Game of Life",
                            4 => "Filesystem",
                            5 => "Terminal",
                            6 => "Document",
                            _ => "App",
                        }).unwrap_or("Unknown");
                        println!("Dock click -> launched {} at slot {}", app_name, slot);
                    }
                }
            }
        }
    }

    fn handle_key(&mut self, key: Key, pressed: bool) {
        // Issue #253: Forward ALL keyboard events to GPU (both press and release)
        if let Some(os) = &mut self.os {
            // Convert Key to HID keycode for GPU
            let keycode = match &key {
                Key::Named(named) => match named {
                    NamedKey::Escape => 0x29,   // HID Escape
                    NamedKey::Enter => 0x28,   // HID Enter
                    NamedKey::Tab => 0x2B,     // HID Tab
                    NamedKey::Backspace => 0x2A, // HID Backspace
                    NamedKey::Space => 0x2C,   // HID Space
                    NamedKey::ArrowUp => 0x52, // HID Up Arrow
                    NamedKey::ArrowDown => 0x51, // HID Down Arrow
                    NamedKey::ArrowLeft => 0x50, // HID Left Arrow
                    NamedKey::ArrowRight => 0x4F, // HID Right Arrow
                    _ => 0,
                },
                Key::Character(c) => {
                    // Map ASCII to approximate HID keycodes
                    if let Some(ch) = c.chars().next() {
                        match ch {
                            'a'..='z' => (ch as u32 - 'a' as u32) + 0x04, // HID a-z
                            'A'..='Z' => (ch.to_ascii_lowercase() as u32 - 'a' as u32) + 0x04,
                            '0' => 0x27, // HID 0
                            '1'..='9' => (ch as u32 - '1' as u32) + 0x1E, // HID 1-9
                            _ => ch as u32, // Fallback to ASCII
                        }
                    } else {
                        0
                    }
                }
                _ => 0,
            };
            if keycode != 0 {
                os.key_event(keycode, pressed, 0);
            }
        }

        // Local handling for demo controls (only on press)
        if !pressed {
            return;
        }

        match &key {
            Key::Named(NamedKey::Space) => {
                // Launch filesystem browser
                if let Some(os) = &mut self.os {
                    if let Some(slot) = os.launch_app(app_type::FILESYSTEM) {
                        println!("Space -> launched filesystem browser at slot {}", slot);
                    }
                }
            }
            Key::Character(c) if c == "t" || c == "T" => {
                // Launch terminal
                if let Some(os) = &mut self.os {
                    if let Some(slot) = os.launch_app(app_type::TERMINAL) {
                        println!("T -> launched terminal at slot {}", slot);
                    }
                }
            }
            Key::Character(c) if c == "d" || c == "D" => {
                // Launch document viewer
                if let Some(os) = &mut self.os {
                    if let Some(slot) = os.launch_app(app_type::DOCUMENT) {
                        println!("D -> launched document viewer at slot {}", slot);
                    }
                }
            }
            Key::Character(c) if c == "b" || c == "B" => {
                // Launch bytecode demo - generates a grid of colored quads
                if let Some(os) = &mut self.os {
                    let bytecode = Self::generate_demo_bytecode();
                    if let Some(slot) = os.launch_bytecode_app(&bytecode) {
                        println!("B -> launched bytecode demo at slot {} ({} bytes of bytecode)", slot, bytecode.len());
                    }
                }
            }
            _ => {}
        }
    }

    fn save_framebuffer(&self, texture: &metal::TextureRef) {
        let width = texture.width() as usize;
        let height = texture.height() as usize;
        let bytes_per_row = width * 4;
        let mut data = vec![0u8; height * bytes_per_row];

        // Read texture data back to CPU
        texture.get_bytes(
            data.as_mut_ptr() as *mut _,
            bytes_per_row as u64,
            metal::MTLRegion {
                origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: metal::MTLSize { width: width as u64, height: height as u64, depth: 1 },
            },
            0,
        );

        // Save as PPM (no dependencies needed)
        use std::io::Write;
        let path = "/tmp/megakernel_frame.ppm";
        if let Ok(mut file) = std::fs::File::create(path) {
            let _ = writeln!(file, "P6\n{} {}\n255", width, height);
            for y in 0..height {
                for x in 0..width {
                    let i = (y * width + x) * 4;
                    // BGRA -> RGB
                    let _ = file.write_all(&[data[i + 2], data[i + 1], data[i]]);
                }
            }
            println!("\n*** SAVED FRAME TO {} ({}x{}) ***\n", path, width, height);
        }
    }

    /// Generate bytecode for a demo app - draws a colorful grid of quads
    fn generate_demo_bytecode() -> Vec<u8> {
        let mut asm = BytecodeAssembler::new();

        let grid_size = 8;
        let cell_size = 50.0;
        let margin = 100.0;  // Start offset from window origin
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

                // Color in r6 - rainbow gradient based on position
                let r = x as f32 / (grid_size as f32 - 1.0);
                let g = y as f32 / (grid_size as f32 - 1.0);
                let b = 1.0 - (r + g) / 2.0;
                asm.setx(6, r);
                asm.sety(6, g);
                asm.setz(6, b);
                asm.setw(6, 1.0);

                // quad(pos_reg, size_reg, color_reg, depth)
                asm.quad(4, 5, 6, 0.5);
            }
        }

        asm.halt();

        let vertex_budget = (grid_size * grid_size * 6) as u32;  // 6 vertices per quad
        asm.build(vertex_budget)
    }
}

impl ApplicationHandler for VisualMegakernelDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .with_title("GPU OS - Visual Megakernel Demo (Issue #162)");
        let window = event_loop.create_window(attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                } else {
                    self.handle_key(event.logical_key.clone(), event.state == ElementState::Pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_move(position.x, position.y);
            }
            WindowEvent::MouseInput { button, state, .. } => {
                println!("MOUSE EVENT: {:?} {:?}", button, state);
                self.handle_mouse_button(button, state);
            }
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = VisualMegakernelDemo::new();
    event_loop.run_app(&mut app).unwrap();
}

// =============================================================================
// Metal Shaders - Unified Vertex Rendering
// =============================================================================
//
// These shaders transform GPU-generated vertices to screen space.
// The vertex buffer is written entirely by the GPU megakernel.

const RENDER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Must match RenderVertex in gpu_app_system.rs
// CRITICAL: Use packed_float3 (12 bytes) to match Rust [f32; 3]
// Metal float3 is 16 bytes which breaks vertex stride alignment!
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
    float2 uv;
};

// Transform screen coordinates to clip space
// Screen: (0,0) top-left, (width, height) bottom-right
// Clip: (-1,-1) bottom-left, (1,1) top-right
vertex VertexOut unified_vertex_shader(
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
    out.uv = v.uv;
    return out;
}

// Simple color output (can be extended for textures)
fragment float4 unified_fragment_shader(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;
