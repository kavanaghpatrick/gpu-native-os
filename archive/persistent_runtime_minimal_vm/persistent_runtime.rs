//! Persistent Runtime - Rust Host Code
//!
//! Issue #279 - GPU-Native Persistent Runtime Host
//!
//! THE GPU IS THE COMPUTER.
//! This module manages the persistent kernel lifecycle.
//! CPU only: buffer creation, spawn requests, shutdown signal.
//! GPU runs continuously, processes work autonomously.

use metal::*;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════════

pub const MAX_PROCESSES: usize = 64;
pub const MAX_SPAWN_QUEUE: usize = 16;
pub const MAX_VERTICES_PER_PROCESS: usize = 1024;
pub const BYTECODE_POOL_SIZE: usize = 16 * 1024 * 1024; // 16MB
pub const HEAP_SIZE: usize = 64 * 1024 * 1024; // 64MB
pub const VERTEX_BUFFER_SIZE: usize = MAX_PROCESSES * MAX_VERTICES_PER_PROCESS * 48; // 48 bytes per vertex

// Process status values (must match Metal)
pub const STATUS_EMPTY: u32 = 0;
pub const STATUS_READY: u32 = 1;
pub const STATUS_RUNNING: u32 = 2;
pub const STATUS_BLOCKED_IO: u32 = 3;
pub const STATUS_BLOCKED_IN: u32 = 4;
pub const STATUS_DEAD: u32 = 5;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES (must match Metal exactly)
// ═══════════════════════════════════════════════════════════════════════════════

/// Process state - CRITICAL: Must match Metal struct exactly (432 bytes, 16-byte aligned)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Process {
    pub pc: u32,
    pub sp: u32,
    pub status: u32,
    pub bytecode_offset: u32,
    pub bytecode_len: u32,
    pub heap_offset: u32,
    pub heap_size: u32,
    pub regs: [i32; 64],
    pub fregs: [f32; 32],
    pub blocked_on: u32,
    pub priority: u32,
    pub _padding: [u32; 3], // CRITICAL: Align to 432 bytes (16-byte aligned)
}

// Compile-time verification of struct layout
const _: () = assert!(std::mem::size_of::<Process>() == 432);
const _: () = assert!(std::mem::size_of::<Process>() % 16 == 0);

impl Default for Process {
    fn default() -> Self {
        Self {
            pc: 0,
            sp: 0,
            status: STATUS_EMPTY,
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

/// Spawn request - 16 bytes, matches Metal
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SpawnRequest {
    pub bytecode_offset: u32,
    pub bytecode_len: u32,
    pub priority: u32,
    pub _reserved: u32,
}

// Compile-time verification
const _: () = assert!(std::mem::size_of::<SpawnRequest>() == 16);

/// System state - shared between CPU and GPU
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SystemState {
    pub process_count: u32,
    pub shutdown_flag: u32,
    pub frame_counter: u32,
    pub spawn_head: u32, // GPU consumes from head
    pub spawn_tail: u32, // CPU writes to tail
    pub free_list_head: u32, // O(1) allocation index
    pub _padding: [u32; 2],
    pub spawn_queue: [SpawnRequest; MAX_SPAWN_QUEUE],
}

// Verify SystemState layout: 32 bytes header + 16 * 16 bytes queue = 288 bytes
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

/// Vertex emitted by GPU processes - must match Metal RenderVertex
/// Uses same layout as gpu_app_system::RenderVertex (48 bytes)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RenderVertex {
    pub position: [f32; 3], // x, y, z (z = depth) - Metal: packed_float3
    pub _pad0: f32,         // Align to float4
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub _pad1: [f32; 2],
}

// Verify vertex is 48 bytes (position:16 + color:16 + uv:16 = 48)
const _: () = assert!(std::mem::size_of::<RenderVertex>() == 48);

/// Vertex counts per process
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexCounts {
    pub counts: [u32; MAX_PROCESSES],
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENT RUNTIME
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU-Native Persistent Runtime
///
/// Manages a persistent kernel that runs continuously on the GPU.
/// CPU responsibilities:
/// - Buffer allocation (at creation)
/// - Spawn requests (via spawn_queue)
/// - Kernel re-dispatch (background thread - due to M4 limitations)
/// - Shutdown signal (at drop)
///
/// IMPORTANT: Apple Silicon M4 blocks truly infinite while(true) loops.
/// This runtime uses bounded kernel chunks (~20M iterations each) with
/// automatic re-dispatch from a background thread to achieve "pseudo-persistence".
pub struct PersistentRuntime {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,

    // GPU buffers
    system_state: Buffer,
    processes: Buffer,
    bytecode_pool: Buffer,
    heap: Buffer,
    vertices: Buffer,
    vertex_counts: Buffer,
    input_events: Buffer,

    // Bytecode pool allocation tracking (CPU-side)
    bytecode_next_offset: u32,

    // Runtime state
    running: bool,
    // Shared shutdown flag for background thread
    shutdown_flag: Arc<AtomicBool>,
    // Background thread handle for kernel re-dispatch
    dispatch_thread: Option<JoinHandle<()>>,
}

impl PersistentRuntime {
    /// Create a new persistent runtime
    pub fn new(device: &Device) -> Result<Self, String> {
        let device = device.clone();
        let queue = device.new_command_queue();

        // Load shader
        let shader_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/gpu_os/shaders/persistent_runtime.metal"
        );
        let shader_source = std::fs::read_to_string(shader_path)
            .map_err(|e| format!("Failed to read shader: {}", e))?;

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&shader_source, &options)
            .map_err(|e| format!("Shader compile failed: {}", e))?;

        let function = library
            .get_function("persistent_runtime", None)
            .map_err(|e| format!("Function 'persistent_runtime' not found: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation failed: {}", e))?;

        // Create buffers with StorageModeShared for CPU-GPU communication
        let system_state = device.new_buffer(
            std::mem::size_of::<SystemState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let processes = device.new_buffer(
            (MAX_PROCESSES * std::mem::size_of::<Process>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let bytecode_pool = device.new_buffer(
            BYTECODE_POOL_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let heap = device.new_buffer(HEAP_SIZE as u64, MTLResourceOptions::StorageModeShared);

        let vertices = device.new_buffer(
            VERTEX_BUFFER_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vertex_counts = device.new_buffer(
            std::mem::size_of::<VertexCounts>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Input events buffer: 16-byte header (InputQueue) + 256 * 16-byte events
        // Header: atomic_uint head, atomic_uint tail, uint _padding[2] = 16 bytes
        let input_events = device.new_buffer(16 + 256 * 16, MTLResourceOptions::StorageModeShared);

        // Initialize buffers with volatile writes + fence
        unsafe {
            // Initialize system state
            let state = system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).process_count, 0);
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 0);
            std::ptr::write_volatile(&mut (*state).frame_counter, 0);
            std::ptr::write_volatile(&mut (*state).spawn_head, 0);
            std::ptr::write_volatile(&mut (*state).spawn_tail, 0);
            std::ptr::write_volatile(&mut (*state).free_list_head, 0);
            fence(Ordering::Release);

            // Initialize processes as EMPTY
            let procs = processes.contents() as *mut Process;
            for i in 0..MAX_PROCESSES {
                std::ptr::write_volatile(&mut (*procs.add(i)).status, STATUS_EMPTY);
            }
            fence(Ordering::Release);

            // Initialize vertex counts to 0
            let counts = vertex_counts.contents() as *mut VertexCounts;
            for i in 0..MAX_PROCESSES {
                std::ptr::write_volatile(&mut (*counts).counts[i], 0);
            }
            fence(Ordering::Release);

            // Initialize input queue header (head = tail = 0, no events pending)
            let input = input_events.contents() as *mut u32;
            std::ptr::write_volatile(input, 0); // head
            std::ptr::write_volatile(input.add(1), 0); // tail
            fence(Ordering::Release);
        }

        Ok(Self {
            device,
            queue,
            pipeline,
            system_state,
            processes,
            bytecode_pool,
            heap,
            vertices,
            vertex_counts,
            input_events,
            bytecode_next_offset: 0,
            running: false,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            dispatch_thread: None,
        })
    }

    /// Upload bytecode to the pool and return the offset
    ///
    /// Returns (offset, byte_len) for use in spawn()
    /// CRITICAL: Returns BYTE count, not word count! Metal divides by 8 to get instruction count.
    pub fn upload_bytecode(&mut self, bytecode: &[u32]) -> Result<(u32, u32), String> {
        let byte_len = (bytecode.len() * 4) as u32;
        let end_offset = self.bytecode_next_offset + byte_len;

        if end_offset as usize > BYTECODE_POOL_SIZE {
            return Err("Bytecode pool full".to_string());
        }

        let offset = self.bytecode_next_offset;

        // Write bytecode with volatile
        unsafe {
            let pool = self.bytecode_pool.contents() as *mut u8;
            let dst = pool.add(offset as usize) as *mut u32;
            for (i, &word) in bytecode.iter().enumerate() {
                std::ptr::write_volatile(dst.add(i), word);
            }
            fence(Ordering::Release);
        }

        self.bytecode_next_offset = end_offset;
        // CRITICAL: Return byte count! Metal does: max_pc = bytecode_len / 8
        Ok((offset, byte_len))
    }

    /// Spawn a new process
    ///
    /// Uses volatile + fence pattern for CPU-GPU communication.
    /// Returns Err if spawn queue is full.
    pub fn spawn(
        &self,
        bytecode_offset: u32,
        bytecode_len: u32,
        priority: u32,
    ) -> Result<(), String> {
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;

            // Volatile reads to get current queue state
            fence(Ordering::Acquire);
            let tail = std::ptr::read_volatile(&(*state).spawn_tail);
            let head = std::ptr::read_volatile(&(*state).spawn_head);

            // Check if queue is full (circular buffer)
            if tail.wrapping_sub(head) >= MAX_SPAWN_QUEUE as u32 {
                return Err("Spawn queue full".to_string());
            }

            // Write spawn request with volatile
            let idx = (tail % MAX_SPAWN_QUEUE as u32) as usize;
            let request = SpawnRequest {
                bytecode_offset,
                bytecode_len,
                priority,
                _reserved: 0,
            };
            std::ptr::write_volatile(&mut (*state).spawn_queue[idx], request);

            // CRITICAL: Release fence ensures request is visible before tail update
            fence(Ordering::Release);

            // Update tail - GPU will see new request
            std::ptr::write_volatile(&mut (*state).spawn_tail, tail.wrapping_add(1));

            // Final fence to ensure tail update is visible
            fence(Ordering::Release);
        }
        Ok(())
    }

    /// Dispatch a single kernel execution
    ///
    /// Helper function used by the background dispatch thread.
    fn dispatch_kernel_once(&self) {
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pipeline);
        // Buffer order must match Metal kernel signature
        enc.set_buffer(0, Some(&self.processes), 0);
        enc.set_buffer(1, Some(&self.bytecode_pool), 0);
        enc.set_buffer(2, Some(&self.heap), 0);
        enc.set_buffer(3, Some(&self.system_state), 0);
        enc.set_buffer(4, Some(&self.input_events), 0);
        enc.set_buffer(5, Some(&self.vertex_counts), 0);
        enc.set_buffer(6, Some(&self.vertices), 0);

        // Single threadgroup of 32 threads (one SIMD group)
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));

        enc.end_encoding();
        cmd.commit();

        // Wait for completion with timeout
        // The kernel runs ~20M iterations which takes ~2-3 seconds
        cmd.wait_until_completed();
    }

    /// Start the persistent kernel
    ///
    /// Spawns a background thread that continuously dispatches the kernel.
    ///
    /// IMPORTANT: Apple Silicon M4 blocks truly infinite while(true) loops.
    /// The kernel uses bounded iterations (~20M) and this thread re-dispatches
    /// automatically to achieve pseudo-persistence.
    pub fn start(&mut self) -> Result<(), String> {
        if self.running {
            return Err("Already running".to_string());
        }

        // Reset shutdown flag
        self.shutdown_flag.store(false, Ordering::Release);

        // Clone references for the thread
        let shutdown_flag = self.shutdown_flag.clone();
        let device = self.device.clone();
        let queue = device.new_command_queue();
        let pipeline = self.pipeline.clone();
        let processes = self.processes.clone();
        let bytecode_pool = self.bytecode_pool.clone();
        let heap = self.heap.clone();
        let system_state = self.system_state.clone();
        let input_events = self.input_events.clone();
        let vertex_counts = self.vertex_counts.clone();
        let vertices = self.vertices.clone();


        // Spawn background dispatch thread
        // This continuously dispatches bounded kernel chunks (~1M iterations each)
        // to achieve pseudo-persistence while allowing CPU to observe state updates
        let thread_handle = thread::spawn(move || {
            while !shutdown_flag.load(Ordering::Acquire) {
                // Dispatch kernel
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();

                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&processes), 0);
                enc.set_buffer(1, Some(&bytecode_pool), 0);
                enc.set_buffer(2, Some(&heap), 0);
                enc.set_buffer(3, Some(&system_state), 0);
                enc.set_buffer(4, Some(&input_events), 0);
                enc.set_buffer(5, Some(&vertex_counts), 0);
                enc.set_buffer(6, Some(&vertices), 0);

                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
                enc.end_encoding();
                cmd.commit();

                // Wait for kernel completion (bounded ~1M iterations, ~0.5s)
                cmd.wait_until_completed();

                // Small sleep to avoid CPU spinning
                thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        self.dispatch_thread = Some(thread_handle);
        self.running = true;

        Ok(())
    }

    /// Stop the persistent kernel
    ///
    /// Sets shutdown flag and waits for background thread to exit.
    pub fn stop(&mut self) {
        if !self.running {
            return;
        }

        // Set GPU shutdown flag with volatile + fence
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
            fence(Ordering::Release);
        }

        // Signal background thread to stop
        self.shutdown_flag.store(true, Ordering::Release);

        // Wait for background thread to exit
        if let Some(thread) = self.dispatch_thread.take() {
            // Give the current kernel dispatch time to complete
            // (bounded to ~20M iterations so ~3 sec max)
            match thread.join() {
                Ok(()) => {}
                Err(e) => eprintln!("Warning: Dispatch thread panicked: {:?}", e),
            }
        }

        self.running = false;
    }

    /// Check if the runtime is currently running
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Get current frame count (volatile read + acquire fence)
    pub fn frame_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            std::ptr::read_volatile(&(*state).frame_counter)
        }
    }

    /// Get active process count (volatile read + acquire fence)
    pub fn process_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            std::ptr::read_volatile(&(*state).process_count)
        }
    }

    /// Read a process register (volatile read + acquire fence)
    pub fn read_register(&self, proc_idx: usize, reg: usize) -> Option<i32> {
        if proc_idx >= MAX_PROCESSES || reg >= 64 {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.processes.contents() as *const Process;
            Some(std::ptr::read_volatile(&(*procs.add(proc_idx)).regs[reg]))
        }
    }

    /// Read a process float register (volatile read + acquire fence)
    pub fn read_float_register(&self, proc_idx: usize, reg: usize) -> Option<f32> {
        if proc_idx >= MAX_PROCESSES || reg >= 32 {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.processes.contents() as *const Process;
            Some(std::ptr::read_volatile(&(*procs.add(proc_idx)).fregs[reg]))
        }
    }

    /// Read process status (volatile read + acquire fence)
    pub fn process_status(&self, proc_idx: usize) -> Option<u32> {
        if proc_idx >= MAX_PROCESSES {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.processes.contents() as *const Process;
            Some(std::ptr::read_volatile(&(*procs.add(proc_idx)).status))
        }
    }

    /// Read vertex count for a process (volatile read + acquire fence)
    pub fn vertex_count(&self, proc_idx: usize) -> Option<u32> {
        if proc_idx >= MAX_PROCESSES {
            return None;
        }
        fence(Ordering::Acquire);
        unsafe {
            let counts = self.vertex_counts.contents() as *const VertexCounts;
            Some(std::ptr::read_volatile(&(*counts).counts[proc_idx]))
        }
    }

    /// Get the vertex buffer for rendering
    pub fn vertex_buffer(&self) -> &Buffer {
        &self.vertices
    }

    /// Get total vertex count across all processes
    pub fn total_vertex_count(&self) -> u32 {
        fence(Ordering::Acquire);
        let mut total = 0u32;
        unsafe {
            let counts = self.vertex_counts.contents() as *const VertexCounts;
            for i in 0..MAX_PROCESSES {
                total += std::ptr::read_volatile(&(*counts).counts[i]);
            }
        }
        total
    }

    /// Check if runtime is healthy (frame counter advancing)
    ///
    /// Returns true if the kernel appears to be running (frame counter changed
    /// in the last check). Note: We can't directly access command buffer status
    /// since Metal returns borrowed references that can't be stored.
    ///
    /// IMPORTANT: Each kernel dispatch takes ~0.5s, so we need to wait at least
    /// that long to see frame counter updates.
    pub fn is_healthy(&self) -> bool {
        if !self.running {
            return false;
        }

        // Check if frame counter is advancing
        // Wait 1200ms to ensure at least two dispatch cycles complete (each ~0.5s)
        // This accounts for timing variability
        let frame1 = self.frame_count();
        std::thread::sleep(std::time::Duration::from_millis(1200));
        let frame2 = self.frame_count();

        frame2 > frame1
    }

    /// Debug: Dump system state
    pub fn debug_dump(&self) {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            println!("=== PersistentRuntime State ===");
            println!(
                "  process_count: {}",
                std::ptr::read_volatile(&(*state).process_count)
            );
            println!(
                "  shutdown_flag: {}",
                std::ptr::read_volatile(&(*state).shutdown_flag)
            );
            println!(
                "  frame_counter: {}",
                std::ptr::read_volatile(&(*state).frame_counter)
            );
            println!(
                "  spawn_head: {}",
                std::ptr::read_volatile(&(*state).spawn_head)
            );
            println!(
                "  spawn_tail: {}",
                std::ptr::read_volatile(&(*state).spawn_tail)
            );
            println!(
                "  free_list_head: {}",
                std::ptr::read_volatile(&(*state).free_list_head)
            );
            println!("  running: {}", self.running);
            println!("  healthy: {}", self.is_healthy());
        }
    }
}

impl Drop for PersistentRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BYTECODE BUILDER (Helper for creating programs)
// ═══════════════════════════════════════════════════════════════════════════════

/// Opcodes matching Metal shader (CRITICAL: values must match exactly)
#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum Opcode {
    Nop = 0x00,
    Const = 0x01,
    Add = 0x02,
    Sub = 0x03,
    Mul = 0x04,
    Div = 0x05,
    Mod = 0x06,
    And = 0x07,
    Or = 0x08,
    Xor = 0x09,
    Jump = 0x0A,
    JumpIf = 0x0B,
    Shl = 0x0C,
    Shr = 0x0D,
    // Note: Gap at 0x0E-0x0F
    Load = 0x10,
    Store = 0x11,
    // Note: Gap at 0x12-0x1F
    Call = 0x20,
    Ret = 0x21,
    // Note: Gap at 0x22-0x2F
    Yield = 0x30,
    // Note: Gap at 0x31-0x3F
    EmitVertex = 0x40,
    // Note: Gap at 0x41-0xFE
    Halt = 0xFF,
}

/// Simple bytecode builder
///
/// CRITICAL: Metal expects 8-byte instructions: [opcode:1][dst:1][src1:1][src2:1][imm:4]
/// Every instruction is 2 u32 words. The second word is the immediate value.
pub struct BytecodeBuilder {
    instructions: Vec<u32>,
}

impl BytecodeBuilder {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Encode instruction header: [opcode:8][dst:8][src1:8][src2:8]
    fn encode_header(opcode: Opcode, dst: u8, src1: u8, src2: u8) -> u32 {
        (opcode as u32) | ((dst as u32) << 8) | ((src1 as u32) << 16) | ((src2 as u32) << 24)
    }

    /// Emit a complete 8-byte instruction (header + immediate)
    fn emit(&mut self, opcode: Opcode, dst: u8, src1: u8, src2: u8, imm: i32) {
        self.instructions
            .push(Self::encode_header(opcode, dst, src1, src2));
        self.instructions.push(imm as u32);
    }

    /// NOP - No operation
    pub fn nop(&mut self) -> &mut Self {
        self.emit(Opcode::Nop, 0, 0, 0, 0);
        self
    }

    /// CONST dst, imm - Load immediate into register
    pub fn const_i32(&mut self, dst: u8, value: i32) -> &mut Self {
        self.emit(Opcode::Const, dst, 0, 0, value);
        self
    }

    /// ADD dst, src1, src2 - dst = src1 + src2
    pub fn add(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Add, dst, src1, src2, 0);
        self
    }

    /// SUB dst, src1, src2 - dst = src1 - src2
    pub fn sub(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Sub, dst, src1, src2, 0);
        self
    }

    /// MUL dst, src1, src2 - dst = src1 * src2
    pub fn mul(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Mul, dst, src1, src2, 0);
        self
    }

    /// DIV dst, src1, src2 - dst = src1 / src2
    pub fn div(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Div, dst, src1, src2, 0);
        self
    }

    /// MOD dst, src1, src2 - dst = src1 % src2
    pub fn mod_(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Mod, dst, src1, src2, 0);
        self
    }

    /// AND dst, src1, src2 - dst = src1 & src2
    pub fn and(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::And, dst, src1, src2, 0);
        self
    }

    /// OR dst, src1, src2 - dst = src1 | src2
    pub fn or(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Or, dst, src1, src2, 0);
        self
    }

    /// XOR dst, src1, src2 - dst = src1 ^ src2
    pub fn xor(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Xor, dst, src1, src2, 0);
        self
    }

    /// JUMP target - Unconditional jump to ABSOLUTE instruction index
    /// Metal does: pc = imm - 1 (then pc++ makes it imm)
    pub fn jump(&mut self, target: i32) -> &mut Self {
        self.emit(Opcode::Jump, 0, 0, 0, target);
        self
    }

    /// JUMP_IF cond, target - Jump to ABSOLUTE instruction index if regs[cond] != 0
    pub fn jump_if(&mut self, cond: u8, target: i32) -> &mut Self {
        self.emit(Opcode::JumpIf, 0, cond, 0, target);
        self
    }

    /// SHL dst, src1, src2 - dst = src1 << src2
    pub fn shl(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Shl, dst, src1, src2, 0);
        self
    }

    /// SHR dst, src1, src2 - dst = src1 >> src2
    pub fn shr(&mut self, dst: u8, src1: u8, src2: u8) -> &mut Self {
        self.emit(Opcode::Shr, dst, src1, src2, 0);
        self
    }

    /// LOAD dst, addr_reg - Load from heap
    pub fn load(&mut self, dst: u8, addr_reg: u8) -> &mut Self {
        self.emit(Opcode::Load, dst, addr_reg, 0, 0);
        self
    }

    /// STORE val_reg, addr_reg - Store to heap
    pub fn store(&mut self, val_reg: u8, addr_reg: u8) -> &mut Self {
        self.emit(Opcode::Store, val_reg, addr_reg, 0, 0);
        self
    }

    /// CALL target - Call subroutine at ABSOLUTE instruction index
    pub fn call(&mut self, target: i32) -> &mut Self {
        self.emit(Opcode::Call, 0, 0, 0, target);
        self
    }

    /// RET - Return from subroutine
    pub fn ret(&mut self) -> &mut Self {
        self.emit(Opcode::Ret, 0, 0, 0, 0);
        self
    }

    /// YIELD - Yield execution for one frame
    pub fn yield_(&mut self) -> &mut Self {
        self.emit(Opcode::Yield, 0, 0, 0, 0);
        self
    }

    /// EMIT_VERTEX x_reg, y_reg, color_reg - Emit a vertex
    pub fn emit_vertex(&mut self, x_reg: u8, y_reg: u8, color_reg: u8) -> &mut Self {
        self.emit(Opcode::EmitVertex, x_reg, y_reg, color_reg, 0);
        self
    }

    /// HALT - Stop process
    pub fn halt(&mut self) -> &mut Self {
        self.emit(Opcode::Halt, 0, 0, 0, 0);
        self
    }

    /// Get the current instruction count (for calculating jump targets)
    /// Note: Each instruction is 2 u32 words, so this returns instructions.len() / 2
    pub fn len(&self) -> usize {
        self.instructions.len() / 2
    }

    /// Build and return the bytecode
    pub fn build(self) -> Vec<u32> {
        self.instructions
    }
}

impl Default for BytecodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

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
        assert_eq!(
            std::mem::size_of::<SystemState>(),
            32 + MAX_SPAWN_QUEUE * 16
        );
    }

    #[test]
    fn test_render_vertex_size() {
        assert_eq!(std::mem::size_of::<RenderVertex>(), 48);
    }

    #[test]
    fn test_bytecode_builder() {
        let mut builder = BytecodeBuilder::new();
        builder.const_i32(0, 42); // r0 = 42
        builder.const_i32(1, 10); // r1 = 10
        builder.add(2, 0, 1); // r2 = r0 + r1
        builder.halt();

        // Verify instruction count before building
        assert_eq!(builder.len(), 4); // 4 instructions

        let bytecode = builder.build();

        // All instructions are 2 words each (8 bytes): 4 instructions = 8 words
        assert_eq!(bytecode.len(), 8);
    }

    #[test]
    fn test_bytecode_builder_loop() {
        // Build: r0 = 0; while (r0 < 10) { r0++ }
        let mut builder = BytecodeBuilder::new();

        builder.const_i32(0, 0); // instr 0: r0 = 0 (counter)
        builder.const_i32(1, 10); // instr 1: r1 = 10 (limit)
        builder.const_i32(2, 1); // instr 2: r2 = 1 (increment)

        let loop_start = builder.len(); // = 3 (instruction index)

        builder.sub(3, 1, 0); // instr 3: r3 = r1 - r0 (remaining iterations)
        builder.add(0, 0, 2); // instr 4: r0 = r0 + 1

        // Jump back to loop_start if r3 != 0 (ABSOLUTE target)
        builder.jump_if(3, loop_start as i32); // instr 5: if r3 != 0, goto instr 3

        builder.halt(); // instr 6

        let bytecode = builder.build();
        // 7 instructions * 2 words = 14 words
        assert_eq!(bytecode.len(), 14);
    }
}
