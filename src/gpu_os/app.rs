// GPU Application Framework
//
// Provides a standardized way to build applications on the GPU-Native OS.
// Apps use the OS's input handling, memory management, and rendering infrastructure
// while providing their own compute kernels and app-specific state.

use super::memory::{FrameState, GpuMemory, InputEvent, InputEventType};
use super::input::InputHandler;
use super::vsync::FrameTiming;
use metal::*;
use std::time::Instant;

// ============================================================================
// Buffer Slot Convention
// ============================================================================
//
// Standardized buffer binding slots for all GPU apps:
//
// Slot 0: FrameState (OS-provided) - cursor, time, frame number
// Slot 1: InputQueue (OS-provided) - keyboard/mouse events
// Slot 2: AppParams (app-specific) - app's per-frame parameters
// Slot 3+: App buffers (app-specific) - app's state buffers
//
// This allows apps to share OS infrastructure while having their own state.

pub const SLOT_FRAME_STATE: u64 = 0;
pub const SLOT_INPUT_QUEUE: u64 = 1;
pub const SLOT_APP_PARAMS: u64 = 2;
pub const SLOT_APP_START: u64 = 3;

// ============================================================================
// GpuApp Trait
// ============================================================================

/// Trait for GPU-native applications that run on the OS
pub trait GpuApp {
    /// Application name (for debugging/logging)
    fn name(&self) -> &str;

    /// App's compute pipeline state
    fn compute_pipeline(&self) -> &ComputePipelineState;

    /// App's render pipeline state
    fn render_pipeline(&self) -> &RenderPipelineState;

    /// App's vertex buffer (where compute kernel writes vertices)
    fn vertices_buffer(&self) -> &Buffer;

    /// Number of vertices to draw (updated by compute kernel)
    fn vertex_count(&self) -> usize;

    /// App-specific buffers to bind at slots 3+
    /// Returns (slot_offset, buffer) pairs where slot = SLOT_APP_START + offset
    fn app_buffers(&self) -> Vec<&Buffer>;

    /// App's per-frame parameters buffer (bound at slot 2)
    fn params_buffer(&self) -> &Buffer;

    /// Called before compute dispatch to update app params from OS state
    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32);

    /// Process a single input event (called for each event in queue)
    fn handle_input(&mut self, event: &InputEvent);

    /// Called after frame completes (for stats, logging, etc.)
    fn post_frame(&mut self, _timing: &FrameTiming) {}

    /// Thread count for compute dispatch (default: 1024)
    fn thread_count(&self) -> usize { 1024 }

    /// Clear color for render pass
    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(0.05, 0.05, 0.08, 1.0)
    }
}

// ============================================================================
// GpuRuntime - The OS Runtime that hosts apps
// ============================================================================

/// Runtime that manages the GPU-Native OS and hosts applications
pub struct GpuRuntime {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub memory: GpuMemory,
    pub input: InputHandler,

    // Frame timing
    last_frame: Instant,
    frame_count: u64,
    delta_time: f32,
}

impl GpuRuntime {
    /// Create a new GPU runtime
    pub fn new(device: Device) -> Self {
        let command_queue = device.new_command_queue();
        let memory = GpuMemory::new(&device, 1024); // Support up to 1024 widgets
        let input = InputHandler::new(&device);

        Self {
            device,
            command_queue,
            memory,
            input,
            last_frame: Instant::now(),
            frame_count: 0,
            delta_time: 1.0 / 120.0,
        }
    }

    /// Get the Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get delta time from last frame
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    /// Push a mouse move event
    pub fn push_mouse_move(&self, x: f32, y: f32) {
        self.input.push_mouse_move(x, y, 0.0, 0.0);
    }

    /// Push a mouse button event
    pub fn push_mouse_button(&self, button: u16, pressed: bool, x: f32, y: f32) {
        self.input.push_mouse_button(button, pressed, x, y);
    }

    /// Push a key event
    pub fn push_key(&self, keycode: u16, pressed: bool, modifiers: u32) {
        self.input.push_key(keycode, pressed, modifiers);
    }

    /// Run a single frame for an application
    pub fn run_frame<A: GpuApp>(&mut self, app: &mut A, drawable: &MetalDrawableRef) {
        // Update timing
        let now = Instant::now();
        self.delta_time = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Update OS frame state
        {
            let frame_state = self.memory.frame_state_mut();
            frame_state.frame_number = self.frame_count as u32;
            frame_state.time += self.delta_time;
        }

        // Process input events and forward to app
        let events = self.input.drain_events(64);
        for event in &events {
            // Update OS cursor state from mouse moves
            if event.event_type == InputEventType::MouseMove as u16 {
                let frame_state = self.memory.frame_state_mut();
                frame_state.cursor_x = event.position[0];
                frame_state.cursor_y = event.position[1];
            }

            // Forward event to app
            app.handle_input(event);
        }

        // Let app update its params from OS state
        let frame_state = self.memory.frame_state();
        app.update_params(&frame_state, self.delta_time);

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // === COMPUTE PASS ===
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(app.compute_pipeline());

        // Bind OS buffers (slots 0-1)
        compute_encoder.set_buffer(SLOT_FRAME_STATE, Some(&self.memory.frame_state_buffer), 0);
        compute_encoder.set_buffer(SLOT_INPUT_QUEUE, Some(&self.memory.input_queue_buffer), 0);

        // Bind app params buffer (slot 2)
        compute_encoder.set_buffer(SLOT_APP_PARAMS, Some(app.params_buffer()), 0);

        // Bind app-specific buffers (slots 3+)
        for (i, buffer) in app.app_buffers().iter().enumerate() {
            compute_encoder.set_buffer(SLOT_APP_START + i as u64, Some(buffer), 0);
        }

        // Dispatch single threadgroup
        let thread_count = app.thread_count() as u64;
        compute_encoder.dispatch_threads(
            MTLSize::new(thread_count, 1, 1),
            MTLSize::new(thread_count, 1, 1),
        );
        compute_encoder.end_encoding();

        // === RENDER PASS ===
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);

        let clear = app.clear_color();
        color_attachment.set_clear_color(clear);

        let render_encoder = command_buffer.new_render_command_encoder(&render_desc);
        render_encoder.set_render_pipeline_state(app.render_pipeline());
        render_encoder.set_vertex_buffer(0, Some(app.vertices_buffer()), 0);

        render_encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            app.vertex_count() as u64,
        );
        render_encoder.end_encoding();

        // Present and commit
        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        // Update frame count
        self.frame_count += 1;

        // Notify app of frame completion
        let timing = FrameTiming {
            total_ms: self.delta_time as f64 * 1000.0,
            ..Default::default()
        };
        app.post_frame(&timing);
    }
}

// ============================================================================
// AppBuilder - Helper for creating apps
// ============================================================================

/// Helper for building GPU apps with common patterns
pub struct AppBuilder<'a> {
    device: &'a Device,
    name: String,
}

impl<'a> AppBuilder<'a> {
    pub fn new(device: &'a Device, name: &str) -> Self {
        Self {
            device,
            name: name.to_string(),
        }
    }

    /// Compile a Metal shader library from source
    pub fn compile_library(&self, source: &str) -> Result<Library, String> {
        let options = CompileOptions::new();
        self.device
            .new_library_with_source(source, &options)
            .map_err(|e| format!("{}: Failed to compile shaders: {}", self.name, e))
    }

    /// Create a compute pipeline from a library
    pub fn create_compute_pipeline(
        &self,
        library: &Library,
        function_name: &str,
    ) -> Result<ComputePipelineState, String> {
        let function = library
            .get_function(function_name, None)
            .map_err(|e| format!("{}: Failed to get {}: {}", self.name, function_name, e))?;

        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("{}: Failed to create compute pipeline: {}", self.name, e))
    }

    /// Create a render pipeline with alpha blending
    pub fn create_render_pipeline(
        &self,
        library: &Library,
        vertex_fn: &str,
        fragment_fn: &str,
    ) -> Result<RenderPipelineState, String> {
        let vertex_function = library
            .get_function(vertex_fn, None)
            .map_err(|e| format!("{}: Failed to get {}: {}", self.name, vertex_fn, e))?;

        let fragment_function = library
            .get_function(fragment_fn, None)
            .map_err(|e| format!("{}: Failed to get {}: {}", self.name, fragment_fn, e))?;

        let desc = RenderPipelineDescriptor::new();
        desc.set_vertex_function(Some(&vertex_function));
        desc.set_fragment_function(Some(&fragment_function));

        let attachment = desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        self.device
            .new_render_pipeline_state(&desc)
            .map_err(|e| format!("{}: Failed to create render pipeline: {}", self.name, e))
    }

    /// Create a buffer with shared storage mode
    pub fn create_buffer(&self, size: usize) -> Buffer {
        self.device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Create a buffer initialized with data
    pub fn create_buffer_with_data<T: Copy>(&self, data: &[T]) -> Buffer {
        let size = std::mem::size_of_val(data);
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

// ============================================================================
// Common shader header for apps
// ============================================================================

/// Common Metal shader code that all apps can include
/// Provides OS data structures and helper functions
pub const APP_SHADER_HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// OS-Provided Structures (must match Rust definitions)
// ============================================================================

struct FrameState {
    uint frame_number;
    float time;
    float cursor_x;
    float cursor_y;
    uint focused_widget;
    uint hovered_widget;
    uint modifiers;
    uint _padding;
};

struct InputEvent {
    ushort event_type;
    ushort keycode;
    float2 position;
    float2 delta;
    uint modifiers;
    uint timestamp;
};

struct InputQueue {
    atomic_uint head;
    atomic_uint tail;
    uint _padding[2];
    InputEvent events[256];
};

// Input event types
constant ushort INPUT_NONE = 0;
constant ushort INPUT_MOUSE_MOVE = 1;
constant ushort INPUT_MOUSE_DOWN = 2;
constant ushort INPUT_MOUSE_UP = 3;
constant ushort INPUT_MOUSE_SCROLL = 4;
constant ushort INPUT_KEY_DOWN = 5;
constant ushort INPUT_KEY_UP = 6;
constant ushort INPUT_KEY_REPEAT = 7;

// ============================================================================
// Helper Functions
// ============================================================================

// Simple hash function for pseudo-random numbers
inline uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

// Random float in [0, 1]
inline float random_float(uint seed) {
    return float(hash(seed)) / float(0xFFFFFFFFu);
}

// Random float in [min, max]
inline float random_range(uint seed, float min_val, float max_val) {
    return min_val + random_float(seed) * (max_val - min_val);
}

// HSV to RGB conversion
inline float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float hp = h / 60.0;
    float x = c * (1.0 - abs(fmod(hp, 2.0) - 1.0));
    float3 rgb;
    if (hp < 1) rgb = float3(c, x, 0);
    else if (hp < 2) rgb = float3(x, c, 0);
    else if (hp < 3) rgb = float3(0, c, x);
    else if (hp < 4) rgb = float3(0, x, c);
    else if (hp < 5) rgb = float3(x, 0, c);
    else rgb = float3(c, 0, x);
    float m = v - c;
    return rgb + m;
}

// Pack RGB to ABGR (Metal's BGRA8Unorm format)
inline uint pack_color(float3 rgb) {
    uint r = uint(clamp(rgb.r, 0.0f, 1.0f) * 255.0);
    uint g = uint(clamp(rgb.g, 0.0f, 1.0f) * 255.0);
    uint b = uint(clamp(rgb.b, 0.0f, 1.0f) * 255.0);
    return (255u << 24) | (b << 16) | (g << 8) | r;
}

// Unpack ABGR to RGBA float4
inline float4 unpack_color(uint packed) {
    return float4(
        float((packed >>  0) & 0xFF) / 255.0,
        float((packed >>  8) & 0xFF) / 255.0,
        float((packed >> 16) & 0xFF) / 255.0,
        float((packed >> 24) & 0xFF) / 255.0
    );
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_slots() {
        assert_eq!(SLOT_FRAME_STATE, 0);
        assert_eq!(SLOT_INPUT_QUEUE, 1);
        assert_eq!(SLOT_APP_PARAMS, 2);
        assert_eq!(SLOT_APP_START, 3);
    }
}
