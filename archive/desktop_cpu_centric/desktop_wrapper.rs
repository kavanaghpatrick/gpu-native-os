//! DynamicDesktopApp Wrapper (Issue #144)
//!
//! Wraps a DynamicGpuApp to run in a desktop window.
//! Renders the GPU app to a texture, then blits to the window.

use metal::*;

use crate::gpu_os::desktop::app::{AppInputEvent, AppRenderContext, DesktopApp, AppEventType};
use crate::gpu_os::memory::{FrameState, InputEvent, InputEventType};
use crate::gpu_os::app::{GpuApp, SLOT_FRAME_STATE, SLOT_INPUT_QUEUE, SLOT_APP_PARAMS, SLOT_APP_START};
use super::app::DynamicGpuApp;

/// Wraps a DynamicGpuApp to run in a desktop window
pub struct DynamicDesktopApp {
    gpu_app: DynamicGpuApp,
    device: Device,
    command_queue: CommandQueue,

    // Offscreen rendering
    render_texture: Texture,
    depth_texture: Texture,
    blit_pipeline: RenderPipelineState,

    // Frame state buffer (for compute pass)
    frame_state_buffer: Buffer,
    input_queue_buffer: Buffer,

    // Tracking
    frame_number: u64,
    total_time: f32,
    width: f32,
    height: f32,

    // Mouse state
    mouse_x: f32,
    mouse_y: f32,
    mouse_buttons: u32,
}

impl DynamicDesktopApp {
    /// Create a new desktop wrapper for a DynamicGpuApp
    pub fn new(gpu_app: DynamicGpuApp, device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        let (width, height) = gpu_app.preferred_size();

        // Create render texture
        let tex_desc = TextureDescriptor::new();
        tex_desc.set_width(width as u64);
        tex_desc.set_height(height as u64);
        tex_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        tex_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        tex_desc.set_storage_mode(MTLStorageMode::Private);
        let render_texture = device.new_texture(&tex_desc);

        // Create depth texture
        let depth_desc = TextureDescriptor::new();
        depth_desc.set_width(width as u64);
        depth_desc.set_height(height as u64);
        depth_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
        depth_desc.set_usage(MTLTextureUsage::RenderTarget);
        depth_desc.set_storage_mode(MTLStorageMode::Private);
        let depth_texture = device.new_texture(&depth_desc);

        // Create blit pipeline
        let blit_shader = r#"
#include <metal_stdlib>
using namespace metal;

struct BlitVertex {
    float4 position [[position]];
    float2 texcoord;
};

vertex BlitVertex blit_vertex(uint vid [[vertex_id]]) {
    float2 positions[6] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1),
        float2(-1, 1), float2(1, -1), float2(1, 1)
    };
    float2 texcoords[6] = {
        float2(0, 1), float2(1, 1), float2(0, 0),
        float2(0, 0), float2(1, 1), float2(1, 0)
    };
    BlitVertex out;
    out.position = float4(positions[vid], 0, 1);
    out.texcoord = texcoords[vid];
    return out;
}

fragment float4 blit_fragment(
    BlitVertex in [[stage_in]],
    texture2d<float> tex [[texture(0)]]
) {
    constexpr sampler s(filter::linear);
    return tex.sample(s, in.texcoord);
}
"#;

        let compile_opts = CompileOptions::new();
        let library = device
            .new_library_with_source(blit_shader, &compile_opts)
            .map_err(|e| format!("Blit shader compile error: {}", e))?;

        let vertex_fn = library
            .get_function("blit_vertex", None)
            .map_err(|_| "Missing blit_vertex")?;
        let fragment_fn = library
            .get_function("blit_fragment", None)
            .map_err(|_| "Missing blit_fragment")?;

        let pipe_desc = RenderPipelineDescriptor::new();
        pipe_desc.set_vertex_function(Some(&vertex_fn));
        pipe_desc.set_fragment_function(Some(&fragment_fn));
        pipe_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let blit_pipeline = device
            .new_render_pipeline_state(&pipe_desc)
            .map_err(|e| format!("Blit pipeline error: {}", e))?;

        // Create frame state buffer (matches FrameState struct)
        let frame_state_buffer = device.new_buffer(
            std::mem::size_of::<FrameState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create input queue buffer (simplified - just header for now)
        let input_queue_buffer = device.new_buffer(
            4096, // Enough for queue header + events
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            gpu_app,
            device: device.clone(),
            command_queue,
            render_texture,
            depth_texture,
            blit_pipeline,
            frame_state_buffer,
            input_queue_buffer,
            frame_number: 0,
            total_time: 0.0,
            width,
            height,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: 0,
        })
    }

    /// Ensure textures match the required size
    fn ensure_texture_size(&mut self, width: f32, height: f32) {
        if (self.width - width).abs() > 1.0 || (self.height - height).abs() > 1.0 {
            self.width = width;
            self.height = height;

            // Recreate render texture
            let tex_desc = TextureDescriptor::new();
            tex_desc.set_width(width as u64);
            tex_desc.set_height(height as u64);
            tex_desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            tex_desc.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            tex_desc.set_storage_mode(MTLStorageMode::Private);
            self.render_texture = self.device.new_texture(&tex_desc);

            // Recreate depth texture
            let depth_desc = TextureDescriptor::new();
            depth_desc.set_width(width as u64);
            depth_desc.set_height(height as u64);
            depth_desc.set_pixel_format(MTLPixelFormat::Depth32Float);
            depth_desc.set_usage(MTLTextureUsage::RenderTarget);
            depth_desc.set_storage_mode(MTLStorageMode::Private);
            self.depth_texture = self.device.new_texture(&depth_desc);
        }
    }

    /// Update frame state buffer
    fn update_frame_state(&mut self, delta_time: f32) {
        let frame_state = FrameState {
            frame_number: self.frame_number as u32,
            time: self.total_time,
            cursor_x: self.mouse_x / self.width,
            cursor_y: self.mouse_y / self.height,
            focused_widget: 0,
            hovered_widget: 0,
            modifiers: self.mouse_buttons,
            _padding: 0,
        };

        unsafe {
            let ptr = self.frame_state_buffer.contents() as *mut FrameState;
            *ptr = frame_state;
        }

        // Also update GPU app params
        self.gpu_app.update_params(&frame_state, delta_time);
    }

    /// Run the GPU app's compute and render passes to texture
    fn render_to_texture(&mut self) {
        let command_buffer = self.command_queue.new_command_buffer();

        // 1. Compute pass
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(self.gpu_app.compute_pipeline());

        // Bind OS buffers
        compute_encoder.set_buffer(SLOT_FRAME_STATE, Some(&self.frame_state_buffer), 0);
        compute_encoder.set_buffer(SLOT_INPUT_QUEUE, Some(&self.input_queue_buffer), 0);
        compute_encoder.set_buffer(SLOT_APP_PARAMS, Some(self.gpu_app.params_buffer()), 0);

        // Bind app buffers at slots 3+
        for (i, buffer) in self.gpu_app.app_buffers().iter().enumerate() {
            compute_encoder.set_buffer(SLOT_APP_START + i as u64, Some(buffer), 0);
        }

        // Bind vertices buffer (typically at slot 3 for writing)
        compute_encoder.set_buffer(SLOT_APP_START, Some(self.gpu_app.vertices_buffer()), 0);

        // Dispatch compute
        let thread_count = self.gpu_app.thread_count();
        let threads_per_group = 256.min(thread_count);
        let threadgroups = (thread_count + threads_per_group - 1) / threads_per_group;

        compute_encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups as u64, 1, 1),
            MTLSize::new(threads_per_group as u64, 1, 1),
        );
        compute_encoder.end_encoding();

        // 2. Render pass to texture
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(&self.render_texture));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        let clear = self.gpu_app.clear_color();
        color_attachment.set_clear_color(clear);

        let render_encoder = command_buffer.new_render_command_encoder(&render_desc);
        render_encoder.set_render_pipeline_state(self.gpu_app.render_pipeline());
        render_encoder.set_vertex_buffer(0, Some(self.gpu_app.vertices_buffer()), 0);

        let vertex_count = self.gpu_app.vertex_count();
        if vertex_count > 0 {
            render_encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
        }
        render_encoder.end_encoding();

        // Commit and wait (synchronous for now)
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

impl DesktopApp for DynamicDesktopApp {
    fn name(&self) -> &str {
        self.gpu_app.name()
    }

    fn icon_index(&self) -> u32 {
        0 // Default icon for dynamic apps
    }

    fn preferred_size(&self) -> (f32, f32) {
        self.gpu_app.preferred_size()
    }

    fn init(&mut self, _device: &Device) -> Result<(), String> {
        // Already initialized in new()
        Ok(())
    }

    fn update(&mut self, delta_time: f32) {
        self.total_time += delta_time;
        self.frame_number += 1;
    }

    fn render(&mut self, ctx: &mut AppRenderContext) {
        // Ensure textures are the right size
        self.ensure_texture_size(ctx.width, ctx.height);

        // Update mouse position from context
        // Note: ctx doesn't directly give us mouse pos, we track it in handle_input

        // Update frame state
        self.update_frame_state(ctx.delta_time);

        // Render GPU app to texture
        self.render_to_texture();

        // Blit texture to window
        // We need to set up scissor rect to only draw in our window area
        let scissor = MTLScissorRect {
            x: ctx.window_x as u64,
            y: ctx.window_y as u64,
            width: ctx.width as u64,
            height: ctx.height as u64,
        };
        ctx.encoder.set_scissor_rect(scissor);

        // Set viewport to our window area
        let viewport = MTLViewport {
            originX: ctx.window_x as f64,
            originY: ctx.window_y as f64,
            width: ctx.width as f64,
            height: ctx.height as f64,
            znear: 0.0,
            zfar: 1.0,
        };
        ctx.encoder.set_viewport(viewport);

        ctx.encoder.set_render_pipeline_state(&self.blit_pipeline);
        ctx.encoder.set_fragment_texture(0, Some(&self.render_texture));
        ctx.encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);

        // Reset scissor to full screen for subsequent draws
        let full_scissor = MTLScissorRect {
            x: 0,
            y: 0,
            width: ctx.screen_width as u64,
            height: ctx.screen_height as u64,
        };
        ctx.encoder.set_scissor_rect(full_scissor);
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        // Track mouse position
        self.mouse_x = event.mouse_x;
        self.mouse_y = event.mouse_y;

        // Track mouse buttons
        match event.event_type {
            AppEventType::MouseDown => {
                self.mouse_buttons |= 1 << event.mouse_button;
            }
            AppEventType::MouseUp => {
                self.mouse_buttons &= !(1 << event.mouse_button);
            }
            _ => {}
        }

        // Convert to InputEvent for GPU app
        let gpu_event = InputEvent {
            event_type: match event.event_type {
                AppEventType::MouseMove => InputEventType::MouseMove as u16,
                AppEventType::MouseDown => InputEventType::MouseDown as u16,
                AppEventType::MouseUp => InputEventType::MouseUp as u16,
                AppEventType::KeyDown => InputEventType::KeyDown as u16,
                AppEventType::KeyUp => InputEventType::KeyUp as u16,
                _ => return false,
            },
            keycode: event.key_code as u16,
            position: [event.mouse_x, event.mouse_y],
            delta: [0.0, 0.0],
            modifiers: 0,
            timestamp: 0,
        };

        self.gpu_app.handle_input(&gpu_event);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_bundle(dir: &std::path::Path) -> std::path::PathBuf {
        let bundle = dir.join("test.gpuapp");
        fs::create_dir_all(&bundle).unwrap();

        fs::write(
            bundle.join("manifest.toml"),
            r#"
[app]
name = "Desktop Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
thread_count = 64
vertex_count = 6
"#,
        )
        .unwrap();

        fs::write(
            bundle.join("main.metal"),
            r#"
kernel void k(device float4* verts [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid < 6) {
        float2 p[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),float2(-1,1),float2(1,-1),float2(1,1)};
        verts[tid] = float4(p[tid], 0, 1);
    }
}
struct V { float4 p [[position]]; float4 c; };
vertex V v(device float4* verts [[buffer(0)]], uint vid [[vertex_id]]) {
    V o; o.p = verts[vid]; o.c = float4(0.5, 0.8, 1.0, 1.0); return o;
}
fragment float4 f(V in [[stage_in]]) { return in.c; }
"#,
        )
        .unwrap();

        bundle
    }

    #[test]
    fn test_create_desktop_wrapper() {
        let device = Device::system_default().expect("No Metal device");
        let dir = TempDir::new().unwrap();
        let bundle = create_test_bundle(dir.path());

        let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
        let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

        assert_eq!(desktop_app.name(), "Desktop Test");
    }

    #[test]
    fn test_preferred_size() {
        let device = Device::system_default().expect("No Metal device");
        let dir = TempDir::new().unwrap();
        let bundle = create_test_bundle(dir.path());

        let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
        let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

        let (w, h) = desktop_app.preferred_size();
        assert!(w > 0.0);
        assert!(h > 0.0);
    }

    #[test]
    fn test_update_advances_time() {
        let device = Device::system_default().expect("No Metal device");
        let dir = TempDir::new().unwrap();
        let bundle = create_test_bundle(dir.path());

        let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
        let mut desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

        let initial_frame = desktop_app.frame_number;
        desktop_app.update(0.016);
        desktop_app.update(0.016);
        desktop_app.update(0.016);

        assert!(desktop_app.frame_number > initial_frame);
        assert!(desktop_app.total_time > 0.0);
    }

    #[test]
    fn test_texture_resize() {
        let device = Device::system_default().expect("No Metal device");
        let dir = TempDir::new().unwrap();
        let bundle = create_test_bundle(dir.path());

        let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
        let mut desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

        let initial_width = desktop_app.width;
        desktop_app.ensure_texture_size(1024.0, 768.0);

        assert!((desktop_app.width - 1024.0).abs() < 0.1);
        assert!((desktop_app.height - 768.0).abs() < 0.1);
    }
}
