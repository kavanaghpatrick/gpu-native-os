// Persistent Runtime - All-Threads-Participate Execution Model (Issue #279)
//
// THE GPU IS THE COMPUTER. This runtime proves that GPUs can run indefinitely
// without crashing by ensuring all SIMD threads participate in every loop.
//
// Key principles:
// - All 32 threads in a wavefront enter the main while(true) loop
// - No "if (tid != 0) return;" patterns - these crash after ~5M iterations
// - SIMD lanes = process slots (1:1 mapping for lockstep execution)
// - Idle threads skip execution phase but still hit barriers
// - CPU writes via volatile + fence, never raw pointer writes

use metal::*;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of concurrent processes (matches 64 threads = 2 threadgroups)
pub const MAX_PROCESSES: usize = 64;

/// Heap size per process (64KB)
pub const PROCESS_HEAP_SIZE: usize = 65536;

/// Total bytecode pool size (16MB)
pub const BYTECODE_POOL_SIZE: usize = 16 * 1024 * 1024;

/// Total heap pool size
pub const HEAP_POOL_SIZE: usize = MAX_PROCESSES * PROCESS_HEAP_SIZE;

/// Maximum spawn requests in queue
pub const MAX_SPAWN_QUEUE: usize = 16;

/// Input queue size (256 events)
pub const INPUT_QUEUE_SIZE: usize = 256;

/// Vertex output buffer size (1MB)
pub const VERTEX_OUTPUT_SIZE: usize = 1024 * 1024;

// ============================================================================
// Process Status (must match Metal enum)
// ============================================================================

pub mod process_status {
    pub const EMPTY: u32 = 0;
    pub const READY: u32 = 1;
    pub const RUNNING: u32 = 2;
    pub const BLOCKED_IO: u32 = 3;
    pub const BLOCKED_INPUT: u32 = 4;
    pub const DEAD: u32 = 5;
}

// ============================================================================
// GPU Structs (must match Metal structs exactly)
// ============================================================================

/// Process state - CRITICAL: Must be 432 bytes (16-byte aligned for Metal)
///
/// Layout:
/// - pc (4) + sp (4) + status (4) + bytecode_offset (4) = 16 bytes
/// - bytecode_len (4) + heap_offset (4) + heap_size (4) + blocked_on (4) = 16 bytes
/// - regs[64] = 256 bytes
/// - fregs[32] = 128 bytes
/// - priority (4) + _padding[3] (12) = 16 bytes
/// Total: 16 + 16 - 4 + 256 + 128 + 16 = 432 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Process {
    pub pc: u32,                  // Program counter
    pub sp: u32,                  // Stack pointer
    pub status: u32,              // ProcessStatus (use atomic access)
    pub bytecode_offset: u32,     // Offset into bytecode pool
    pub bytecode_len: u32,        // Length of bytecode
    pub heap_offset: u32,         // Offset into heap pool
    pub heap_size: u32,           // Allocated heap size
    pub regs: [i32; 64],          // Virtual registers (integer)
    pub fregs: [f32; 32],         // Virtual registers (float)
    pub blocked_on: u32,          // What we're waiting for
    pub priority: u32,            // Scheduling priority
    pub _padding: [u32; 3],       // Align to 432 bytes (divisible by 16)
}

// Compile-time size verification - CRITICAL for GPU struct alignment
const _: () = assert!(std::mem::size_of::<Process>() == 432);
const _: () = assert!(std::mem::size_of::<Process>() % 16 == 0);

impl Default for Process {
    fn default() -> Self {
        Self {
            pc: 0,
            sp: PROCESS_HEAP_SIZE as u32,
            status: process_status::EMPTY,
            bytecode_offset: 0,
            bytecode_len: 0,
            heap_offset: 0,
            heap_size: 0,
            regs: [0; 64],
            fregs: [0.0; 32],
            blocked_on: 0,
            priority: 0,
            _padding: [0; 3],
        }
    }
}

/// Spawn request - 16 bytes
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SpawnRequest {
    pub bytecode_offset: u32,
    pub bytecode_len: u32,
    pub priority: u32,
    pub _padding: u32,
}

const _: () = assert!(std::mem::size_of::<SpawnRequest>() == 16);

/// System state - 32 bytes header + spawn queue
///
/// Layout:
/// - process_count (4) + shutdown_flag (4) + frame_counter (4) = 12 bytes
/// - spawn_head (4) + spawn_tail (4) + free_list_head (4) = 12 bytes
/// - _padding[2] (8) = 8 bytes
/// Header total: 32 bytes
/// - spawn_queue[16] = 256 bytes
/// Total: 288 bytes
#[repr(C)]
pub struct SystemState {
    pub process_count: u32,       // Number of active processes
    pub shutdown_flag: u32,       // Set to 1 to stop kernel
    pub frame_counter: u32,       // Increments each frame
    pub spawn_head: u32,          // Spawn queue head (consumer)
    pub spawn_tail: u32,          // Spawn queue tail (producer)
    pub free_list_head: u32,      // O(1) allocation index
    pub _padding: [u32; 2],       // Align to 32 bytes
    pub spawn_queue: [SpawnRequest; MAX_SPAWN_QUEUE],
}

const _: () = assert!(std::mem::size_of::<SystemState>() == 32 + MAX_SPAWN_QUEUE * 16);

impl Default for SystemState {
    fn default() -> Self {
        Self {
            process_count: 0,
            shutdown_flag: 0,
            frame_counter: 0,
            spawn_head: 0,
            spawn_tail: 0,
            free_list_head: 0,
            _padding: [0; 2],
            spawn_queue: [SpawnRequest::default(); MAX_SPAWN_QUEUE],
        }
    }
}

/// Input event for GPU processing
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct InputEvent {
    pub event_type: u32,
    pub key_code: u32,
    pub mouse_x: f32,
    pub mouse_y: f32,
}

const _: () = assert!(std::mem::size_of::<InputEvent>() == 16);

/// Input queue header
#[repr(C)]
pub struct InputQueueHeader {
    pub head: u32,
    pub tail: u32,
    pub _padding: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<InputQueueHeader>() == 16);

/// Vertex for output rendering - CRITICAL: Use packed_float3 semantics
/// Rust [f32; 3] = 12 bytes = Metal packed_float3
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Vertex {
    pub position: [f32; 3],   // 12 bytes (packed_float3 in Metal)
    pub _pad0: f32,           // 4 bytes explicit padding
    pub color: [f32; 4],      // 16 bytes (float4 in Metal)
}

const _: () = assert!(std::mem::size_of::<Vertex>() == 32);

/// Vertex output header
#[repr(C)]
pub struct VertexOutputHeader {
    pub vertex_count: u32,
    pub _padding: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<VertexOutputHeader>() == 16);

// ============================================================================
// PersistentRuntime
// ============================================================================

/// Persistent runtime that runs WASM processes on GPU indefinitely
///
/// Uses the all-threads-participate pattern to avoid the ~5M iteration crash
/// that plagued earlier designs with single-thread patterns.
pub struct PersistentRuntime {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,

    // GPU Buffers
    process_buffer: Buffer,
    bytecode_pool: Buffer,
    heap_pool: Buffer,
    system_state: Buffer,
    input_queue: Buffer,
    vertex_output: Buffer,

    // Tracking
    bytecode_write_offset: usize,
    running: Arc<AtomicBool>,
}

impl PersistentRuntime {
    /// Create a new persistent runtime
    ///
    /// # Arguments
    /// * `device` - Metal device to use
    ///
    /// # Returns
    /// * `Ok(PersistentRuntime)` on success
    /// * `Err(String)` if shader compilation or buffer allocation fails
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Load and compile shader
        let shader_source = include_str!("shaders/persistent_runtime.metal");
        let compile_options = CompileOptions::new();

        let library = device
            .new_library_with_source(shader_source, &compile_options)
            .map_err(|e| format!("Shader compile error: {}", e))?;

        let function = library
            .get_function("persistent_runtime", None)
            .map_err(|e| format!("Function 'persistent_runtime' not found: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation error: {}", e))?;

        // Create buffers with StorageModeShared for CPU/GPU access
        // NOTE: For production, GPU-only buffers could use StorageModePrivate
        // but Shared allows debugging and testing access

        let process_buffer = device.new_buffer(
            (MAX_PROCESSES * std::mem::size_of::<Process>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let bytecode_pool = device.new_buffer(
            BYTECODE_POOL_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let heap_pool = device.new_buffer(
            HEAP_POOL_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let system_state = device.new_buffer(
            std::mem::size_of::<SystemState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Input queue: header (16 bytes) + events (256 * 16 = 4096 bytes)
        let input_queue = device.new_buffer(
            (16 + INPUT_QUEUE_SIZE * std::mem::size_of::<InputEvent>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Vertex output: header (16 bytes) + vertices
        let vertex_output = device.new_buffer(
            (16 + VERTEX_OUTPUT_SIZE) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize system state with volatile writes
        // CRITICAL: Use volatile + fence to ensure GPU sees correct initial values
        unsafe {
            let state = system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).process_count, 0);
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 0);
            std::ptr::write_volatile(&mut (*state).frame_counter, 0);
            std::ptr::write_volatile(&mut (*state).spawn_head, 0);
            std::ptr::write_volatile(&mut (*state).spawn_tail, 0);
            std::ptr::write_volatile(&mut (*state).free_list_head, 0);
            std::ptr::write_volatile(&mut (*state)._padding[0], 0);
            std::ptr::write_volatile(&mut (*state)._padding[1], 0);
            fence(Ordering::Release);
        }

        // Initialize all processes as EMPTY
        unsafe {
            let procs = process_buffer.contents() as *mut Process;
            for i in 0..MAX_PROCESSES {
                std::ptr::write_volatile(&mut (*procs.add(i)).status, process_status::EMPTY);
            }
            fence(Ordering::Release);
        }

        // Initialize input queue
        unsafe {
            let header = input_queue.contents() as *mut InputQueueHeader;
            std::ptr::write_volatile(&mut (*header).head, 0);
            std::ptr::write_volatile(&mut (*header).tail, 0);
            fence(Ordering::Release);
        }

        // Initialize vertex output
        unsafe {
            let header = vertex_output.contents() as *mut VertexOutputHeader;
            std::ptr::write_volatile(&mut (*header).vertex_count, 0);
            fence(Ordering::Release);
        }

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline,
            process_buffer,
            bytecode_pool,
            heap_pool,
            system_state,
            input_queue,
            vertex_output,
            bytecode_write_offset: 0,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Load bytecode into the pool
    ///
    /// # Arguments
    /// * `bytecode` - Raw bytecode bytes to load
    ///
    /// # Returns
    /// * `Ok((offset, len))` - Offset and length in the bytecode pool
    /// * `Err(String)` if pool is full
    pub fn load_bytecode(&mut self, bytecode: &[u8]) -> Result<(u32, u32), String> {
        if self.bytecode_write_offset + bytecode.len() > BYTECODE_POOL_SIZE {
            return Err("Bytecode pool full".to_string());
        }

        let offset = self.bytecode_write_offset as u32;
        let len = bytecode.len() as u32;

        // Copy bytecode to pool with fence for GPU visibility
        unsafe {
            let pool = self.bytecode_pool.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                bytecode.as_ptr(),
                pool.add(self.bytecode_write_offset),
                bytecode.len(),
            );
            fence(Ordering::Release);
        }

        self.bytecode_write_offset += bytecode.len();
        Ok((offset, len))
    }

    /// Spawn a new process from loaded bytecode
    ///
    /// # Arguments
    /// * `bytecode_offset` - Offset in bytecode pool (from load_bytecode)
    /// * `bytecode_len` - Length of bytecode
    /// * `priority` - Process priority (0 = background, 3 = realtime)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(String)` if spawn queue is full
    pub fn spawn(&self, bytecode_offset: u32, bytecode_len: u32, priority: u32) -> Result<(), String> {
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;

            // Read current queue state with volatile
            let tail = std::ptr::read_volatile(&(*state).spawn_tail);
            let head = std::ptr::read_volatile(&(*state).spawn_head);

            // Check if queue is full (max 16 pending spawn requests)
            if tail.wrapping_sub(head) >= MAX_SPAWN_QUEUE as u32 {
                return Err("Spawn queue full".to_string());
            }

            // Write spawn request to queue
            let idx = (tail % MAX_SPAWN_QUEUE as u32) as usize;
            std::ptr::write_volatile(
                &mut (*state).spawn_queue[idx],
                SpawnRequest {
                    bytecode_offset,
                    bytecode_len,
                    priority,
                    _padding: 0,
                },
            );

            // CRITICAL: Release fence ensures spawn request visible before tail update
            fence(Ordering::Release);

            // Advance tail (GPU sees new request after this write)
            std::ptr::write_volatile(&mut (*state).spawn_tail, tail.wrapping_add(1));
        }

        Ok(())
    }

    /// Convenience method to load bytecode and spawn a process
    ///
    /// # Arguments
    /// * `bytecode` - Raw bytecode bytes
    ///
    /// # Returns
    /// * `Ok(process_id)` - The process slot that will be assigned
    /// * `Err(String)` on failure
    pub fn spawn_process(&mut self, bytecode: &[u8]) -> Result<u32, String> {
        let (offset, len) = self.load_bytecode(bytecode)?;
        self.spawn(offset, len, 0)?;

        // Return expected process slot (based on current spawn count)
        // Note: This is approximate - actual slot assigned by GPU
        let state = unsafe {
            let state = self.system_state.contents() as *const SystemState;
            fence(Ordering::Acquire);
            std::ptr::read_volatile(&(*state).free_list_head)
        };
        Ok(state)
    }

    /// Start the persistent kernel in a background thread
    ///
    /// The kernel runs indefinitely until shutdown() is called.
    /// Uses pseudo-persistent pattern with ~1M iteration chunks
    /// to work around Apple Silicon M4 infinite loop restrictions.
    pub fn run(&self) {
        if self.running.load(Ordering::Acquire) {
            return;
        }

        self.running.store(true, Ordering::Release);

        // Clone Arc for the background thread
        let running = Arc::clone(&self.running);
        let command_queue = self.command_queue.clone();
        let pipeline = self.pipeline.clone();
        let process_buffer = self.process_buffer.clone();
        let bytecode_pool = self.bytecode_pool.clone();
        let heap_pool = self.heap_pool.clone();
        let system_state = self.system_state.clone();
        let input_queue = self.input_queue.clone();
        let vertex_output = self.vertex_output.clone();

        // Spawn background thread for kernel dispatch
        // Uses kernel chaining pattern: dispatch, wait, repeat
        std::thread::spawn(move || {
            while running.load(Ordering::Acquire) {
                let command_buffer = command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&process_buffer), 0);
                encoder.set_buffer(1, Some(&bytecode_pool), 0);
                encoder.set_buffer(2, Some(&heap_pool), 0);
                encoder.set_buffer(3, Some(&system_state), 0);
                encoder.set_buffer(4, Some(&input_queue), 0);
                encoder.set_buffer(5, Some(&vertex_output), 0);

                // Dispatch 2 threadgroups of 32 threads = 64 threads = MAX_PROCESSES
                // Each thread handles one process slot (1:1 mapping)
                encoder.dispatch_thread_groups(
                    MTLSize::new(2, 1, 1),   // 2 threadgroups
                    MTLSize::new(32, 1, 1),  // 32 threads per group
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Check for shutdown between dispatches
                // This allows clean exit even if kernel takes time
            }
        });
    }

    /// Start the persistent kernel (blocking version for testing)
    ///
    /// Dispatches the kernel once but does not wait for completion.
    /// Use run() for production - this is mainly for tests that
    /// need precise control over kernel lifetime.
    pub fn start(&mut self) {
        if self.running.load(Ordering::Acquire) {
            return;
        }

        self.running.store(true, Ordering::Release);
        self.run();
    }

    /// Stop the persistent kernel
    ///
    /// Signals shutdown via atomic flag and waits for kernel to exit.
    /// Uses frame counter stability to detect kernel termination.
    pub fn stop(&mut self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        // Signal shutdown
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
            fence(Ordering::Release);
        }

        // Wait for kernel to exit by monitoring frame counter stability
        let mut last_frame = 0u32;
        let mut stable_count = 0;

        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(10));

            fence(Ordering::Acquire);
            let frame = unsafe {
                let state = self.system_state.contents() as *const SystemState;
                std::ptr::read_volatile(&(*state).frame_counter)
            };

            if frame == last_frame {
                stable_count += 1;
                if stable_count >= 3 {
                    // Frame counter stable for 30ms - kernel has stopped
                    break;
                }
            } else {
                stable_count = 0;
                last_frame = frame;
            }
        }

        // Reset shutdown flag for potential restart
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 0);
            fence(Ordering::Release);
        }

        self.running.store(false, Ordering::Release);
    }

    /// Shutdown the runtime (alias for stop)
    pub fn shutdown(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }

        // Signal shutdown
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
            fence(Ordering::Release);
        }

        // Note: This version doesn't wait - caller should use stop() for blocking wait
    }

    /// Get current frame count (volatile read)
    pub fn frame_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            std::ptr::read_volatile(&(*state).frame_counter)
        }
    }

    /// Get current process count (volatile read)
    pub fn process_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            std::ptr::read_volatile(&(*state).process_count)
        }
    }

    /// Read a process's register value (for testing/debugging)
    ///
    /// # Arguments
    /// * `proc_idx` - Process index (0-63)
    /// * `reg` - Register index (0-63)
    ///
    /// # Returns
    /// * `Some(value)` if indices are valid
    /// * `None` if indices are out of bounds
    pub fn read_register(&self, proc_idx: usize, reg: usize) -> Option<i32> {
        if proc_idx >= MAX_PROCESSES || reg >= 64 {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.process_buffer.contents() as *const Process;
            Some(std::ptr::read_volatile(&(*procs.add(proc_idx)).regs[reg]))
        }
    }

    /// Read a process's float register value
    pub fn read_float_register(&self, proc_idx: usize, reg: usize) -> Option<f32> {
        if proc_idx >= MAX_PROCESSES || reg >= 32 {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.process_buffer.contents() as *const Process;
            Some(std::ptr::read_volatile(&(*procs.add(proc_idx)).fregs[reg]))
        }
    }

    /// Check if a process is dead (volatile read)
    pub fn is_dead(&self, proc_idx: usize) -> bool {
        if proc_idx >= MAX_PROCESSES {
            return true;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.process_buffer.contents() as *const Process;
            std::ptr::read_volatile(&(*procs.add(proc_idx)).status) == process_status::DEAD
        }
    }

    /// Get process status
    pub fn get_status(&self, proc_idx: usize) -> Option<u32> {
        if proc_idx >= MAX_PROCESSES {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.process_buffer.contents() as *const Process;
            Some(std::ptr::read_volatile(&(*procs.add(proc_idx)).status))
        }
    }

    /// Check if the runtime is currently running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get vertex count from output buffer
    pub fn vertex_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let header = self.vertex_output.contents() as *const VertexOutputHeader;
            std::ptr::read_volatile(&(*header).vertex_count)
        }
    }

    /// Queue an input event for GPU processing
    pub fn queue_input(&self, event: InputEvent) -> Result<(), String> {
        unsafe {
            let header = self.input_queue.contents() as *mut InputQueueHeader;
            let events = (self.input_queue.contents() as *mut u8).add(16) as *mut InputEvent;

            let tail = std::ptr::read_volatile(&(*header).tail);
            let head = std::ptr::read_volatile(&(*header).head);

            if tail.wrapping_sub(head) >= INPUT_QUEUE_SIZE as u32 {
                return Err("Input queue full".to_string());
            }

            let idx = (tail % INPUT_QUEUE_SIZE as u32) as usize;
            std::ptr::write_volatile(events.add(idx), event);
            fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*header).tail, tail.wrapping_add(1));
        }
        Ok(())
    }

    /// Push an input event to the GPU input queue
    ///
    /// This is an alias for queue_input() for API compatibility.
    /// Uses volatile writes and memory fencing to ensure GPU visibility.
    ///
    /// # Arguments
    /// * `event` - The input event to push (keyboard, mouse, etc.)
    ///
    /// # Returns
    /// * `Ok(())` if the event was queued successfully
    /// * `Err(String)` if the input queue is full
    #[inline]
    pub fn push_input(&self, event: InputEvent) -> Result<(), String> {
        self.queue_input(event)
    }
}

impl Drop for PersistentRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_struct_size() {
        assert_eq!(std::mem::size_of::<Process>(), 432);
        assert_eq!(std::mem::size_of::<Process>() % 16, 0);
    }

    #[test]
    fn test_spawn_request_size() {
        assert_eq!(std::mem::size_of::<SpawnRequest>(), 16);
    }

    #[test]
    fn test_system_state_size() {
        assert_eq!(std::mem::size_of::<SystemState>(), 32 + 16 * 16);
    }

    #[test]
    fn test_vertex_size() {
        assert_eq!(std::mem::size_of::<Vertex>(), 32);
    }
}
