// Visual WASM Apps Demo
//
// THE GPU IS THE COMPUTER.
//
// This demo loads PURE RUST apps (compiled to WASM), translates them to GPU bytecode,
// and runs them visually on the GPU. The Rust apps have NO knowledge that they will
// run on a GPU - they're just normal Rust code.
//
// Controls:
//   1 - Load particles.wasm
//   2 - Load bouncing_balls.wasm
//   3 - Load game_of_life.wasm
//   4 - Load mandelbrot.wasm
//   5 - Load snake.wasm
//   6 - Load clock.wasm
//   7 - Load pong.wasm
//   8 - Load 2048.wasm
//   B - Launch hand-coded bytecode demo (for comparison)
//   Escape - Exit

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{msg_send, rc::autoreleasepool, runtime::YES, sel, sel_impl};
use rust_experiment::gpu_os::gpu_os::GpuOs;
use rust_experiment::gpu_os::gpu_app_system::{app_type, BytecodeAssembler};
use std::fs;
use std::path::Path;
use std::time::Instant;
use wasm_translator::{TranslatorConfig, WasmTranslator};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};
// Note: Previously used ActivationPolicy::Regular but switched to simple EventLoop::new()
// to match visual_megakernel.rs which has working keyboard input

/// Visual demo app state
struct VisualWasmDemo {
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
    apps_loaded: Vec<String>,
    /// Track current loaded app slot for replacement
    current_app_slot: Option<u32>,
}

impl VisualWasmDemo {
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
            apps_loaded: Vec::new(),
            current_app_slot: None,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();

        println!("╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║     VISUAL WASM APPS DEMO - THE GPU IS THE COMPUTER                   ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  Pure Rust apps → WASM → GPU Bytecode → GPU Execution → Visual Output ║");
        println!("╚═══════════════════════════════════════════════════════════════════════╝");
        println!();
        println!("GPU: {}", device.name());
        println!();
        println!("Available WASM Apps (press number to load):");
        println!("  1 - Particles        (physics simulation)");
        println!("  2 - Bouncing Balls   (collision detection)");
        println!("  3 - Game of Life     (cellular automata)");
        println!("  4 - Mandelbrot       (fractal rendering)");
        println!("  5 - Snake            (classic game)");
        println!("  6 - Clock            (animated clock)");
        println!("  7 - Pong             (paddle game)");
        println!("  8 - 2048             (puzzle game)");
        println!("  B - Bytecode demo    (hand-coded comparison)");
        println!();
        println!("  Escape - Exit");
        println!();

        // Boot GPU OS
        let size = window.inner_size();
        self.window_size = (size.width, size.height);

        let mut os = GpuOs::boot_with_size(&device, size.width as f32, size.height as f32)
            .expect("Failed to boot GPU OS");

        // Launch a terminal app to show something
        if let Some(slot) = os.launch_app(app_type::TERMINAL) {
            println!("Launched terminal at slot {}", slot);
        }

        // Create Metal layer
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);
        layer.set_opaque(true);  // CRITICAL: Make layer content visible

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

        // Create render pipeline
        let render_pipeline = Self::create_render_pipeline(&device);
        let depth_stencil_state = Self::create_depth_stencil_state(&device);
        let depth_texture = Self::create_depth_texture(&device, size.width, size.height);

        // Screen size uniform buffer - MUST use StorageModeShared and initialize immediately
        let screen_size_buffer = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
        unsafe {
            let ptr = screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = [size.width as f32, size.height as f32];
        }

        println!("Screen size: {}x{}", size.width, size.height);

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.render_pipeline = Some(render_pipeline);
        self.depth_stencil_state = Some(depth_stencil_state);
        self.depth_texture = Some(depth_texture);
        self.screen_size_buffer = Some(screen_size_buffer);
        self.os = Some(os);
    }

    /// Load a WASM file, translate to bytecode, and run on GPU
    fn load_wasm_app(&mut self, name: &str, path: &str) {
        println!("\n======================================================================");
        println!("Loading WASM App: {}", name);
        println!("======================================================================");

        // Close previous app if one is loaded
        if let Some(slot) = self.current_app_slot.take() {
            if let Some(os) = &mut self.os {
                println!("  Closing previous app at slot {}", slot);
                os.close_app(slot);
            }
        }

        // Check if file exists
        if !Path::new(path).exists() {
            println!("ERROR: WASM file not found: {}", path);
            println!("Hint: Build with: cd test_programs/apps/{} && cargo build --release --target wasm32-unknown-unknown",
                     name.to_lowercase().replace(" ", "_"));
            return;
        }

        // Load WASM file
        let wasm_bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!("ERROR: Failed to read WASM file: {}", e);
                return;
            }
        };

        println!("  WASM size: {} bytes", wasm_bytes.len());

        // Translate to GPU bytecode
        let translator = WasmTranslator::new(TranslatorConfig::default());
        let bytecode = match translator.translate(&wasm_bytes) {
            Ok(bc) => bc,
            Err(e) => {
                println!("ERROR: Translation failed: {:?}", e);
                return;
            }
        };

        println!("  Bytecode size: {} bytes", bytecode.len());

        // Extract instruction count from header
        if bytecode.len() >= 16 {
            let instr_count = u32::from_le_bytes([bytecode[0], bytecode[1], bytecode[2], bytecode[3]]);
            let entry_point = u32::from_le_bytes([bytecode[4], bytecode[5], bytecode[6], bytecode[7]]);
            let vertex_budget = u32::from_le_bytes([bytecode[8], bytecode[9], bytecode[10], bytecode[11]]);
            let flags = u32::from_le_bytes([bytecode[12], bytecode[13], bytecode[14], bytecode[15]]);
            println!("  Instructions: {}", instr_count);
            println!("  BYTECODE HEADER: code_size={}, entry_point={}, vertex_budget={}, flags={}",
                     instr_count, entry_point, vertex_budget, flags);

            // Dump first 20 instructions
            let header_size = 16;
            let inst_size = 8;
            println!("  FIRST 20 INSTRUCTIONS:");
            for i in 0..20.min(instr_count as usize) {
                let offset = header_size + i * inst_size;
                if offset + inst_size <= bytecode.len() {
                    let opcode = bytecode[offset];
                    let dst = bytecode[offset + 1];
                    let src1 = bytecode[offset + 2];
                    let src2 = bytecode[offset + 3];
                    let imm_bytes = [bytecode[offset + 4], bytecode[offset + 5], bytecode[offset + 6], bytecode[offset + 7]];
                    let imm = f32::from_le_bytes(imm_bytes);
                    println!("    PC {:3}: opcode={:#04x} dst={} src1={} src2={} imm={:.4}",
                             i, opcode, dst, src1, src2, imm);
                }
            }
        }

        // Launch on GPU
        if let Some(os) = &mut self.os {
            match os.launch_bytecode_app(&bytecode) {
                Some(slot) => {
                    println!("  Launched at slot: {}", slot);
                    println!("  Status: RUNNING ON GPU!");
                    self.current_app_slot = Some(slot);
                    // Keep only most recent app in the list
                    self.apps_loaded.clear();
                    self.apps_loaded.push(format!("{} (slot {})", name, slot));
                }
                None => {
                    println!("ERROR: Failed to launch app on GPU");
                }
            }
        } else {
            println!("ERROR: GPU OS not initialized - cannot launch app");
        }
    }

    fn render(&mut self) {
        println!("RENDER CALLED");
        let render_start = Instant::now();

        let Some(os) = &mut self.os else { println!("RENDER: No OS"); return };
        let Some(layer) = &self.layer else { println!("RENDER: No layer"); return };
        let Some(command_queue) = &self.command_queue else { println!("RENDER: No command_queue"); return };
        let Some(render_pipeline) = &self.render_pipeline else { println!("RENDER: No render_pipeline"); return };
        let Some(depth_stencil_state) = &self.depth_stencil_state else { println!("RENDER: No depth_stencil_state"); return };
        let Some(depth_texture) = &self.depth_texture else { println!("RENDER: No depth_texture"); return };
        let Some(screen_size_buffer) = &self.screen_size_buffer else { println!("RENDER: No screen_size_buffer"); return };

        // TIMING: Measure run_frame() which includes finalize_render() with wait_until_completed()
        let run_frame_start = Instant::now();
        os.run_frame();
        let run_frame_time = run_frame_start.elapsed();

        let vertex_count = os.total_vertex_count();

        // TIMING: Measure drawable acquisition
        let drawable_start = Instant::now();
        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => { println!("RENDER: No drawable!"); return },
        };
        println!("GOT DRAWABLE");
        let drawable_time = drawable_start.elapsed();

        // Update screen size
        let size_data: [f32; 2] = [self.window_size.0 as f32, self.window_size.1 as f32];
        unsafe {
            let ptr = screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = size_data;
        }

        // Create command buffer
        let command_buffer = command_queue.new_command_buffer();

        // Render pass
        let render_pass = RenderPassDescriptor::new();
        let color_attachment = render_pass.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let depth_attachment = render_pass.depth_attachment().unwrap();
        depth_attachment.set_texture(Some(depth_texture));
        depth_attachment.set_load_action(MTLLoadAction::Clear);
        depth_attachment.set_clear_depth(0.0);  // Clear to back (lowest depth)
        depth_attachment.set_store_action(MTLStoreAction::DontCare);

        let encoder = command_buffer.new_render_command_encoder(&render_pass);
        encoder.set_render_pipeline_state(render_pipeline);
        encoder.set_depth_stencil_state(depth_stencil_state);

        // Issue #240 fix: Explicitly set viewport - Metal default may be (0,0,0,0)
        encoder.set_viewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: self.window_size.0 as f64,
            height: self.window_size.1 as f64,
            znear: 0.0,
            zfar: 1.0,
        });

        encoder.set_vertex_buffer(0, Some(os.render_vertices_buffer()), 0);
        encoder.set_vertex_buffer(1, Some(screen_size_buffer), 0);

        // Draw each active app's actual vertices (not the full slot)
        // This avoids rendering garbage/uninitialized vertices
        let get_draw_calls_start = Instant::now();
        let draw_calls = os.get_draw_calls();
        let get_draw_calls_time = get_draw_calls_start.elapsed();

        // DEBUG: Print draw calls to diagnose rendering issue
        if self.frame_count % 60 == 0 || self.frame_count < 5 {
            println!("DRAW CALLS: {:?} (count={})", draw_calls, draw_calls.len());
        }

        for (start_vertex, count) in &draw_calls {
            if *count > 0 {
                encoder.draw_primitives(MTLPrimitiveType::Triangle, *start_vertex, *count);
            }
        }

        encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        println!("COMMITTED");

        let total_render_time = render_start.elapsed();

        // FPS tracking and TIMING output
        self.frame_count += 1;
        if self.frame_count % 10 == 0 {
            println!("FRAME {}", self.frame_count);
        }
        if self.frame_count % 60 == 0 {
            let elapsed = self.last_frame.elapsed().as_secs_f32();
            let fps = 60.0 / elapsed;
            println!(
                "Frame {:5} | {:5} vertices | {:5.1} FPS | {} apps loaded",
                self.frame_count,
                vertex_count,
                fps,
                self.apps_loaded.len()
            );
            // TIMING: Print detailed timing breakdown
            println!(
                "[TIMING] render()={:.2}ms | run_frame()={:.2}ms | drawable={:.2}ms | get_draw_calls={:.2}ms",
                total_render_time.as_secs_f64() * 1000.0,
                run_frame_time.as_secs_f64() * 1000.0,
                drawable_time.as_secs_f64() * 1000.0,
                get_draw_calls_time.as_secs_f64() * 1000.0
            );
            self.last_frame = Instant::now();
        }
    }

    fn create_render_pipeline(device: &Device) -> RenderPipelineState {
        // EXACT COPY from visual_megakernel.rs - known working shader
        let shader_source = r#"
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

        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .expect("Failed to compile shaders");

        let vertex_fn = library.get_function("unified_vertex_shader", None).unwrap();
        let fragment_fn = library.get_function("unified_fragment_shader", None).unwrap();

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

        desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);

        device.new_render_pipeline_state(&desc).unwrap()
    }

    fn create_depth_stencil_state(device: &Device) -> DepthStencilState {
        let desc = DepthStencilDescriptor::new();
        // Higher depth values are in front (GreaterEqual)
        desc.set_depth_compare_function(MTLCompareFunction::GreaterEqual);
        desc.set_depth_write_enabled(true);
        device.new_depth_stencil_state(&desc)
    }

    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        let desc = TextureDescriptor::new();
        desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        desc.set_storage_mode(MTLStorageMode::Private);
        desc.set_usage(MTLTextureUsage::RenderTarget);
        device.new_texture(&desc)
    }

    /// Generate hand-coded bytecode demo for comparison
    fn launch_bytecode_demo(&mut self) {
        println!("\n======================================================================");
        println!("Launching Hand-Coded Bytecode Demo (for comparison)");
        println!("======================================================================");

        // Close previous app if one is loaded
        if let Some(slot) = self.current_app_slot.take() {
            if let Some(os) = &mut self.os {
                println!("  Closing previous app at slot {}", slot);
                os.close_app(slot);
            }
        }

        let bytecode = Self::generate_demo_bytecode();
        println!("  Bytecode size: {} bytes", bytecode.len());

        if let Some(os) = &mut self.os {
            match os.launch_bytecode_app(&bytecode) {
                Some(slot) => {
                    println!("  Launched at slot: {}", slot);
                    self.current_app_slot = Some(slot);
                    self.apps_loaded.clear();
                    self.apps_loaded.push(format!("Bytecode Demo (slot {})", slot));
                }
                None => {
                    println!("ERROR: Failed to launch bytecode demo");
                }
            }
        }
    }

    fn generate_demo_bytecode() -> Vec<u8> {
        let mut asm = BytecodeAssembler::new();

        // Draw a colorful grid
        let grid_size = 8;
        let cell_size = 50.0;
        let start_x = 100.0;
        let start_y = 100.0;

        for row in 0..grid_size {
            for col in 0..grid_size {
                let x = start_x + col as f32 * cell_size;
                let y = start_y + row as f32 * cell_size;

                // Rainbow colors
                let r = (col as f32 / grid_size as f32);
                let g = (row as f32 / grid_size as f32);
                let b = 1.0 - r;

                // Position
                asm.loadi(4, x);
                asm.sety(4, y);
                // Size
                asm.loadi(5, cell_size - 2.0);
                asm.sety(5, cell_size - 2.0);
                // Color
                asm.loadi(6, r);
                asm.sety(6, g);
                asm.setz(6, b);
                asm.setw(6, 1.0);
                // Emit quad
                asm.quad(4, 5, 6, 0.5);
            }
        }

        asm.halt();
        asm.build(0)
    }
}

impl ApplicationHandler for VisualWasmDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("RESUMED CALLED");
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("Visual WASM Apps - THE GPU IS THE COMPUTER")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
                .with_visible(true)   // Ensure window is visible from creation
                .with_active(true);   // Request focus when window is created
            let window = event_loop.create_window(attrs).unwrap();
            println!("WINDOW CREATED");
            self.initialize(window);

            // macOS: Force app to become active foreground app, stealing focus from terminal
            // This is necessary because cargo run keeps the terminal as the active app
            #[cfg(target_os = "macos")]
            unsafe {
                use objc::runtime::{Class, Object, BOOL};
                let ns_app: *mut Object = objc::msg_send![Class::get("NSApplication").unwrap(), sharedApplication];
                let _: () = objc::msg_send![ns_app, activateIgnoringOtherApps: YES as BOOL];
            }

            // macOS: Explicitly make window key and front to receive keyboard events
            // Windows need makeKeyAndOrderFront called to properly receive keyboard input
            #[cfg(target_os = "macos")]
            unsafe {
                use objc::runtime::Object;
                if let Some(ref window) = self.window {
                    if let Ok(handle) = window.window_handle() {
                        if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                            let ns_window: *mut Object = objc::msg_send![appkit_handle.ns_view.as_ptr() as *mut Object, window];
                            let _: () = objc::msg_send![ns_window, makeKeyAndOrderFront: std::ptr::null::<Object>()];
                        }
                    }
                }
            }

            // Ensure window is visible and focused using winit's cross-platform API
            if let Some(ref window) = self.window {
                window.set_visible(true);
                window.focus_window();
                println!("Window set_visible(true) and focus_window() called");
            }
            println!("ACTIVATION REQUESTED");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.window_size = (size.width, size.height);
                if let Some(layer) = &self.layer {
                    layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
                }
                // Update screen size buffer
                if let Some(buffer) = &self.screen_size_buffer {
                    unsafe {
                        let ptr = buffer.contents() as *mut [f32; 2];
                        *ptr = [size.width as f32, size.height as f32];
                    }
                }
                if let Some(device) = &self.device {
                    self.depth_texture =
                        Some(Self::create_depth_texture(device, size.width, size.height));
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as f32;
                self.cursor_y = position.y as f32;
                if let Some(os) = &mut self.os {
                    os.mouse_move(self.cursor_x, self.cursor_y, 0.0, 0.0);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(os) = &mut self.os {
                    let pressed = state == ElementState::Pressed;
                    let btn = if button == MouseButton::Left { 0u8 } else { 1u8 };
                    os.mouse_button(btn, pressed, self.cursor_x, self.cursor_y);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // LOUD DEBUG - print every keyboard event
                println!("########################################");
                println!("# KEYBOARD EVENT: {:?}", event.logical_key);
                println!("# state={:?} repeat={}", event.state, event.repeat);
                println!("########################################");

                let pressed = event.state == ElementState::Pressed;

                // Issue #253: Forward ALL keyboard events to GPU
                if let Some(os) = &mut self.os {
                    // Convert Key to keycode for GPU
                    let keycode = match &event.logical_key {
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
                if pressed {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        Key::Character(c) => {
                            match c.as_str() {
                                "1" => self.load_wasm_app(
                                    "Particles",
                                    "test_programs/apps/particles/target/wasm32-unknown-unknown/release/particles.wasm",
                                ),
                                "2" => self.load_wasm_app(
                                    "Bouncing Balls",
                                    "test_programs/apps/bouncing_balls/target/wasm32-unknown-unknown/release/bouncing_balls.wasm",
                                ),
                                "3" => self.load_wasm_app(
                                    "Game of Life",
                                    "test_programs/apps/game_of_life/target/wasm32-unknown-unknown/release/game_of_life.wasm",
                                ),
                                "4" => self.load_wasm_app(
                                    "Mandelbrot",
                                    "test_programs/apps/mandelbrot_interactive/target/wasm32-unknown-unknown/release/mandelbrot_interactive.wasm",
                                ),
                                "5" => self.load_wasm_app(
                                    "Snake",
                                    "test_programs/apps/snake/target/wasm32-unknown-unknown/release/snake.wasm",
                                ),
                                "6" => self.load_wasm_app(
                                    "Clock",
                                    "test_programs/apps/clock/target/wasm32-unknown-unknown/release/clock.wasm",
                                ),
                                "7" => self.load_wasm_app(
                                    "Pong",
                                    "test_programs/apps/pong/target/wasm32-unknown-unknown/release/pong.wasm",
                                ),
                                "8" => self.load_wasm_app(
                                    "2048",
                                    "test_programs/apps/game_2048/target/wasm32-unknown-unknown/release/game_2048.wasm",
                                ),
                                "b" | "B" => self.launch_bytecode_demo(),
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                println!(">>> REDRAW REQUESTED <<<");
                if self.frame_count == 0 {
                    println!("First RedrawRequested received");
                }
                // Auto-load apps for testing - cycle through multiple
                if self.frame_count == 60 && self.current_app_slot.is_none() {
                    println!("\n*** AUTO-LOADING BOUNCING BALLS FOR TESTING ***");
                    self.load_wasm_app(
                        "Bouncing Balls",
                        "test_programs/apps/bouncing_balls/target/wasm32-unknown-unknown/release/bouncing_balls.wasm",
                    );
                }
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Focused(focused) => {
                println!("########################################");
                println!("# FOCUS CHANGED: {}", if focused { "GAINED" } else { "LOST" });
                println!("########################################");
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        println!("about_to_wait called");
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    println!("APP STARTING...");

    // Use simple EventLoop::new() like visual_megakernel.rs (which has working keyboard)
    // Previously used ActivationPolicy::Regular which may have caused issues
    let event_loop = EventLoop::new().unwrap();

    println!("EVENT LOOP CREATED");

    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = VisualWasmDemo::new();
    event_loop.run_app(&mut app).unwrap();
}
