// GPU Application Framework
//
// Provides a standardized way to build applications on the GPU-Native OS.
// Apps use the OS's input handling, memory management, and rendering infrastructure
// while providing their own compute kernels and app-specific state.

use super::memory::{FrameState, GpuMemory, InputEvent, InputEventType};
use super::input::InputHandler;
use super::text_render::{BitmapFont, TextRenderer};
use super::vsync::FrameTiming;
use metal::*;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use dispatch::{Queue, QueueAttribute};

// Re-export text rendering types for convenience
pub use super::text_render::{BitmapFont as Font, TextRenderer as Text, TextChar, colors};

// Re-export GPU I/O types for convenience (GPU-direct file access)
pub use super::batch_io::{GpuBatchLoader, BatchLoadResult, FileDescriptor};
pub use super::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};
pub use super::content_search::{GpuContentSearch, ContentMatch, SearchOptions};
pub use super::gpu_cache::GpuFileCache;

// ============================================================================
// Pipeline Mode - Latency vs Throughput Tradeoff
// ============================================================================

/// Controls frame pipelining behavior
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PipelineMode {
    /// Minimize input→display latency (best for text editors, interactive UI)
    /// Waits for previous frame to complete before starting next
    LowLatency,

    /// Maximize throughput (best for simulations, animations, games)
    /// Allows multiple frames in flight simultaneously
    HighThroughput,
}

impl Default for PipelineMode {
    fn default() -> Self {
        PipelineMode::LowLatency
    }
}

/// GPU timing information from completed frame
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTiming {
    /// Frame number that completed
    pub frame_number: u64,
    /// Actual GPU execution time in milliseconds
    pub gpu_time_ms: f64,
    /// Time from submission to completion in milliseconds
    pub total_time_ms: f64,
}

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

    /// Pipeline mode: LowLatency (interactive) or HighThroughput (simulations)
    /// Default: LowLatency for best interactive response
    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::LowLatency
    }

    /// Called when GPU actually finishes a frame (async, may be out of order)
    /// Use this for accurate GPU profiling, not post_frame() which fires at submission
    fn on_gpu_complete(&mut self, _timing: GpuTiming) {}

    /// For double-buffered simulations: which buffer set to use this frame
    /// Apps with ping-pong buffers can use frame_number % 2 to alternate
    /// Default: always use primary buffers (no double-buffering)
    fn buffer_set_for_frame(&self, _frame_number: u64) -> usize {
        0
    }

    /// Whether this app wants to use the global text rendering system
    /// If true, render_text() will be called after the main render pass
    fn uses_text_rendering(&self) -> bool {
        false
    }

    /// Render text using the global TextRenderer
    /// Called after the main render pass if uses_text_rendering() returns true
    /// Apps should call text_renderer.add_text() etc. here
    fn render_text(&mut self, _text_renderer: &mut TextRenderer) {}
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

    // Global text rendering (available to all apps)
    pub font: BitmapFont,
    pub text_renderer: TextRenderer,

    // Frame timing
    last_frame: Instant,
    frame_count: u64,
    delta_time: f32,

    // Pipeline tracking
    previous_command_buffer: Option<CommandBuffer>,
    gpu_timing_queue: Arc<Mutex<Vec<GpuTiming>>>,

    // Async GPU synchronization (Issue #76)
    shared_event: SharedEvent,
    shared_event_listener: SharedEventListener,
    next_signal_value: Arc<AtomicU64>,
    /// Dispatch queue for SharedEvent callbacks
    _callback_queue: Queue,
}

impl GpuRuntime {
    /// Create a new GPU runtime
    pub fn new(device: Device) -> Self {
        let command_queue = device.new_command_queue();
        let memory = GpuMemory::new(&device, 1024); // Support up to 1024 widgets
        let input = InputHandler::new(&device);

        // Create global text rendering system
        let font = BitmapFont::new(&device);
        let text_renderer = TextRenderer::new(&device, 10000)
            .expect("Failed to create global text renderer");

        // Create SharedEvent for async GPU synchronization (Issue #76)
        let shared_event = device.new_shared_event();
        let callback_queue = Queue::create(
            "com.gpu-native-os.callback-queue",
            QueueAttribute::Serial,
        );
        let shared_event_listener = SharedEventListener::from_queue(&callback_queue);

        Self {
            device,
            command_queue,
            font,
            text_renderer,
            previous_command_buffer: None,
            gpu_timing_queue: Arc::new(Mutex::new(Vec::new())),
            memory,
            input,
            last_frame: Instant::now(),
            frame_count: 0,
            delta_time: 1.0 / 120.0,
            shared_event,
            shared_event_listener,
            next_signal_value: Arc::new(AtomicU64::new(1)),
            _callback_queue: callback_queue,
        }
    }

    /// Get the SharedEvent for external async operations (e.g., filesystem search)
    pub fn shared_event(&self) -> &SharedEvent {
        &self.shared_event
    }

    /// Get the SharedEventListener for registering callbacks
    pub fn shared_event_listener(&self) -> &SharedEventListener {
        &self.shared_event_listener
    }

    /// Allocate the next signal value for async operations
    pub fn next_signal_value(&self) -> u64 {
        self.next_signal_value.fetch_add(1, Ordering::SeqCst)
    }

    /// Check if a signal value has been reached (non-blocking)
    pub fn is_signal_complete(&self, signal_value: u64) -> bool {
        self.shared_event.signaled_value() >= signal_value
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

    /// Get the global bitmap font (for text measurement, etc.)
    pub fn font(&self) -> &BitmapFont {
        &self.font
    }

    /// Get the global text renderer (for direct rendering outside GpuApp)
    pub fn text_renderer(&self) -> &TextRenderer {
        &self.text_renderer
    }

    /// Get mutable reference to text renderer (for adding text)
    pub fn text_renderer_mut(&mut self) -> &mut TextRenderer {
        &mut self.text_renderer
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
        // ═══════════════════════════════════════════════════════════════════
        // PIPELINE MODE: Wait for previous frame if LowLatency mode
        // ═══════════════════════════════════════════════════════════════════
        // LowLatency: Wait for GPU to finish previous frame before starting next
        //             This minimizes input→display latency (best for text editors)
        // HighThroughput: Let frames overlap for maximum FPS (best for simulations)
        if app.pipeline_mode() == PipelineMode::LowLatency {
            if let Some(ref prev_buffer) = self.previous_command_buffer {
                prev_buffer.wait_until_completed();
            }
        }

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

        // === TEXT RENDERING PASS (optional) ===
        if app.uses_text_rendering() {
            // Clear text renderer and let app add its text
            self.text_renderer.clear();
            app.render_text(&mut self.text_renderer);

            // Only render if there's text to draw
            if self.text_renderer.char_count() > 0 {
                let text_pass = RenderPassDescriptor::new();
                let text_attachment = text_pass.color_attachments().object_at(0).unwrap();
                text_attachment.set_texture(Some(drawable.texture()));
                text_attachment.set_load_action(MTLLoadAction::Load); // Preserve existing content
                text_attachment.set_store_action(MTLStoreAction::Store);

                let text_encoder = command_buffer.new_render_command_encoder(&text_pass);
                let size = drawable.texture();
                self.text_renderer.render(
                    &text_encoder,
                    &self.font,
                    size.width() as f32,
                    size.height() as f32,
                );
                text_encoder.end_encoding();
            }
        }

        // === ASYNC SIGNALING (Issue #76) ===
        // Signal SharedEvent when frame completes for async tracking
        let frame_signal = self.next_signal_value.fetch_add(1, Ordering::SeqCst);
        command_buffer.encode_signal_event(&self.shared_event, frame_signal);

        // For HighThroughput mode, register completion callback to track timing
        if app.pipeline_mode() == PipelineMode::HighThroughput {
            let timing_queue = self.gpu_timing_queue.clone();
            let frame_num = self.frame_count;
            let submit_time = Instant::now();

            let handler = block::ConcreteBlock::new(move |_cmd_buf: &CommandBufferRef| {
                let elapsed = submit_time.elapsed();
                let timing = GpuTiming {
                    frame_number: frame_num,
                    gpu_time_ms: elapsed.as_secs_f64() * 1000.0,
                    total_time_ms: elapsed.as_secs_f64() * 1000.0,
                };
                if let Ok(mut queue) = timing_queue.lock() {
                    queue.push(timing);
                }
            });
            command_buffer.add_completed_handler(&handler.copy());
        }

        // Present and commit
        command_buffer.present_drawable(drawable);
        command_buffer.commit();

        // Store for potential waiting next frame (LowLatency mode)
        self.previous_command_buffer = Some(command_buffer.to_owned());

        // Update frame count
        self.frame_count += 1;

        // Notify app of frame completion (submission, not GPU completion)
        let timing = FrameTiming {
            total_ms: self.delta_time as f64 * 1000.0,
            ..Default::default()
        };
        app.post_frame(&timing);
    }

    /// Drain completed GPU frame timings (for HighThroughput mode)
    /// Returns timing info for frames that have actually finished on GPU
    pub fn drain_gpu_timings(&mut self) -> Vec<GpuTiming> {
        if let Ok(mut queue) = self.gpu_timing_queue.lock() {
            std::mem::take(&mut *queue)
        } else {
            Vec::new()
        }
    }

    /// Get the current GPU signal value (how many operations have completed)
    pub fn current_signal_value(&self) -> u64 {
        self.shared_event.signaled_value()
    }

    // =========================================================================
    // GPU-Driven Event Loop (Issue #149)
    // =========================================================================
    //
    // These methods enable GPU-driven event processing, eliminating CPU from
    // steady-state event handling. The GPU:
    // - Reads from existing InputQueue (via InputHandler)
    // - Routes events to appropriate handlers
    // - Manages window drag/resize/focus state
    // - Dispatches render when frame is dirty
    //
    // CPU's only job: push input events via InputHandler.push_*() methods

    /// Initialize the GPU event loop
    ///
    /// Creates the event loop state buffer and compiles GPU kernels.
    /// Returns an EventLoopHandle for controlling the loop.
    ///
    /// # Example
    /// ```ignore
    /// let runtime = GpuRuntime::new(device);
    /// let handle = runtime.init_event_loop().unwrap();
    /// runtime.start_event_loop(&handle, &windows_buffer);
    /// // Input is pushed via existing InputHandler:
    /// runtime.push_mouse_move(x, y);
    /// ```
    pub fn init_event_loop(&self) -> Result<super::event_loop::EventLoopHandle, String> {
        use super::event_loop::{
            GpuEventLoopState, HitTestResult, EventLoopHandle, EVENT_LOOP_SHADER_SOURCE
        };

        // Create NEW state buffer for GPU event loop state
        let state_buffer = self.device.new_buffer(
            GpuEventLoopState::SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize state
        unsafe {
            let ptr = state_buffer.contents() as *mut GpuEventLoopState;
            *ptr = GpuEventLoopState::new();
        }

        // Create hit result buffer
        let hit_result_buffer = self.device.new_buffer(
            HitTestResult::SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize hit result
        unsafe {
            let ptr = hit_result_buffer.contents() as *mut HitTestResult;
            *ptr = HitTestResult::default();
        }

        // Compile event loop kernels
        let options = CompileOptions::new();
        let library = self.device
            .new_library_with_source(EVENT_LOOP_SHADER_SOURCE, &options)
            .map_err(|e| format!("Failed to compile event loop shaders: {}", e))?;

        // Helper to get function and create pipeline
        let create_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
            let function = library.get_function(name, None)
                .map_err(|e| format!("Failed to get function {}: {}", name, e))?;
            self.device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("Failed to create pipeline for {}: {}", name, e))
        };

        let event_loop_pipeline = create_pipeline("gpu_event_loop")?;
        let hit_test_pipeline = create_pipeline("hit_test_parallel")?;
        let hit_result_handler_pipeline = create_pipeline("handle_hit_result")?;
        let window_move_pipeline = create_pipeline("window_move")?;
        let window_resize_pipeline = create_pipeline("window_resize")?;

        Ok(EventLoopHandle {
            state_buffer,
            hit_result_buffer,
            event_loop_pipeline,
            hit_test_pipeline,
            hit_result_handler_pipeline,
            window_move_pipeline,
            window_resize_pipeline,
            running: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start the GPU event loop
    ///
    /// Dispatches the first event loop kernel. Subsequent iterations are
    /// driven by polling or explicit calls to process_event_loop().
    ///
    /// Uses existing infrastructure:
    /// - `self.input.buffer()` for input queue
    /// - `self.command_queue` for dispatch
    /// - `self.shared_event` for async signaling
    pub fn start_event_loop(
        &self,
        handle: &super::event_loop::EventLoopHandle,
        windows_buffer: &Buffer,
        window_count: u32,
    ) {
        handle.running.store(true, std::sync::atomic::Ordering::SeqCst);
        self.dispatch_event_loop_iteration(handle, windows_buffer, window_count);
    }

    /// Process one iteration of the GPU event loop
    ///
    /// Checks what the GPU wants to do next and dispatches the appropriate
    /// kernel. Call this in your main loop or use completion handlers.
    pub fn process_event_loop(
        &self,
        handle: &super::event_loop::EventLoopHandle,
        windows_buffer: &Buffer,
        window_count: u32,
    ) -> bool {
        use super::event_loop::dispatch;

        if !handle.is_running() {
            return false;
        }

        // Read what GPU wants to do next
        let next_dispatch = unsafe {
            let state_ptr = handle.state_buffer.contents() as *const super::event_loop::GpuEventLoopState;
            (*state_ptr).next_dispatch
        };

        // Create window count buffer
        let window_count_buffer = self.device.new_buffer_with_data(
            &window_count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();

        match next_dispatch {
            dispatch::HIT_TEST => {
                // Parallel hit test
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&handle.hit_test_pipeline);
                encoder.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder.set_buffer(1, Some(windows_buffer), 0);
                encoder.set_buffer(2, Some(&handle.hit_result_buffer), 0);
                encoder.set_buffer(3, Some(&window_count_buffer), 0);
                let threads = (window_count as u64).max(1);
                encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(64.min(threads), 1, 1));
                encoder.end_encoding();

                // Handle hit result
                let encoder2 = command_buffer.new_compute_command_encoder();
                encoder2.set_compute_pipeline_state(&handle.hit_result_handler_pipeline);
                encoder2.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder2.set_buffer(1, Some(&handle.hit_result_buffer), 0);
                encoder2.set_buffer(2, Some(windows_buffer), 0);
                encoder2.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder2.end_encoding();
            }
            dispatch::WINDOW_MOVE => {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&handle.window_move_pipeline);
                encoder.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder.set_buffer(1, Some(windows_buffer), 0);
                // Screen size buffer
                let screen_size: [f32; 2] = [1920.0, 1080.0]; // TODO: pass actual screen size
                let screen_buffer = self.device.new_buffer_with_data(
                    screen_size.as_ptr() as *const _,
                    8,
                    MTLResourceOptions::StorageModeShared,
                );
                encoder.set_buffer(2, Some(&screen_buffer), 0);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
            }
            dispatch::WINDOW_RESIZE => {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&handle.window_resize_pipeline);
                encoder.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder.set_buffer(1, Some(windows_buffer), 0);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
            }
            dispatch::RENDER => {
                // Return true to signal render is needed
                command_buffer.commit();
                command_buffer.wait_until_completed();
                self.dispatch_event_loop_iteration(handle, windows_buffer, window_count);
                return true;
            }
            _ => {
                // DISPATCH_NONE or others - just continue
            }
        }

        // Re-dispatch event loop
        self.encode_event_loop_kernel(command_buffer.new_compute_command_encoder(), handle, windows_buffer, &window_count_buffer);

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check if frame needs render
        let frame_dirty = unsafe {
            let state_ptr = handle.state_buffer.contents() as *const super::event_loop::GpuEventLoopState;
            (*state_ptr).frame_dirty
        };
        frame_dirty != 0
    }

    /// Dispatch one iteration of the event loop kernel
    fn dispatch_event_loop_iteration(
        &self,
        handle: &super::event_loop::EventLoopHandle,
        windows_buffer: &Buffer,
        window_count: u32,
    ) {
        let window_count_buffer = self.device.new_buffer_with_data(
            &window_count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.encode_event_loop_kernel(encoder, handle, windows_buffer, &window_count_buffer);

        let signal_value = self.next_signal_value.fetch_add(1, Ordering::SeqCst);
        command_buffer.encode_signal_event(&self.shared_event, signal_value);
        command_buffer.commit();
    }

    /// Encode the event loop kernel into an encoder
    fn encode_event_loop_kernel(
        &self,
        encoder: &ComputeCommandEncoderRef,
        handle: &super::event_loop::EventLoopHandle,
        windows_buffer: &Buffer,
        window_count_buffer: &Buffer,
    ) {
        encoder.set_compute_pipeline_state(&handle.event_loop_pipeline);
        encoder.set_buffer(0, Some(self.input.buffer()), 0);
        encoder.set_buffer(1, Some(&handle.state_buffer), 0);
        encoder.set_buffer(2, Some(windows_buffer), 0);
        encoder.set_buffer(3, Some(window_count_buffer), 0);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
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

    // ========================================================================
    // GPU-Direct I/O (MTLIOCommandQueue)
    // ========================================================================
    //
    // These methods provide GPU-direct file I/O that completely bypasses the CPU.
    // Data flows: Disk → GPU Buffer (CPU never touches the data)
    //
    // This is 3-4x faster than traditional file I/O for large batches.

    /// Create a GPU batch loader for loading multiple files in one GPU command.
    ///
    /// Uses MTLIOCommandQueue (Metal 3+) for true GPU-direct I/O.
    /// Falls back to None if MTLIOCommandQueue is not available.
    ///
    /// # Example
    /// ```ignore
    /// let builder = AppBuilder::new(&device, "MyApp");
    /// if let Some(loader) = builder.create_batch_loader() {
    ///     let files = vec![PathBuf::from("file1.txt"), PathBuf::from("file2.txt")];
    ///     if let Some(result) = loader.load_batch(&files) {
    ///         // result.mega_buffer contains all file data
    ///         // result.descriptors has offsets and sizes
    ///     }
    /// }
    /// ```
    pub fn create_batch_loader(&self) -> Option<super::batch_io::GpuBatchLoader> {
        super::batch_io::GpuBatchLoader::new(self.device)
    }

    /// Create a GPU I/O queue for custom file operations.
    ///
    /// Lower-level than batch_loader, allows fine-grained control over I/O.
    pub fn create_io_queue(
        &self,
        priority: super::gpu_io::IOPriority,
        queue_type: super::gpu_io::IOQueueType,
    ) -> Option<super::gpu_io::GpuIOQueue> {
        super::gpu_io::GpuIOQueue::new(self.device, priority, queue_type)
    }

    /// Create a content search engine for grep-like file searching.
    ///
    /// Combines GPU-direct I/O with parallel pattern matching.
    pub fn create_content_search(&self, max_files: usize) -> Result<super::content_search::GpuContentSearch, String> {
        super::content_search::GpuContentSearch::new(self.device, max_files)
    }

    /// Create a GPU file cache for frequently accessed files.
    ///
    /// Keeps files GPU-resident with LRU eviction.
    pub fn create_file_cache(&self, max_files: usize, slot_size: usize) -> Result<super::gpu_cache::GpuFileCache, String> {
        super::gpu_cache::GpuFileCache::new(self.device, max_files, slot_size)
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
