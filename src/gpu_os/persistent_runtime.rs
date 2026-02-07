// Persistent Runtime (Issue #280)
//
// THE GPU IS THE COMPUTER. This runtime uses the all-threads-participate
// pattern to avoid the ~5M iteration crash that occurred with single-thread
// patterns like `if (tid != 0) return;`.
//
// STUB: This is a placeholder for tests. The actual implementation will be
// merged from another branch.

use metal::*;
use std::sync::atomic::{fence, Ordering};

pub const MAX_PROCESSES: usize = 64;
pub const PROCESS_HEAP_SIZE: usize = 65536;
pub const BYTECODE_POOL_SIZE: usize = 16 * 1024 * 1024;  // 16MB
pub const HEAP_POOL_SIZE: usize = MAX_PROCESSES * PROCESS_HEAP_SIZE;

// CRITICAL: Must match Metal struct exactly (432 bytes, 16-byte aligned)
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
    pub _padding: [u32; 3],  // Align to 432 bytes (16-byte aligned)
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<Process>() == 432);
const _: () = assert!(std::mem::size_of::<Process>() % 16 == 0);

#[repr(C)]
pub struct SystemState {
    pub process_count: u32,
    pub shutdown_flag: u32,
    pub frame_counter: u32,
    pub spawn_head: u32,
    pub spawn_tail: u32,
    pub free_list_head: u32,  // O(1) allocation index
    pub _padding: [u32; 2],   // Align to 32 bytes
    pub spawn_queue: [SpawnRequest; 16],
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<SystemState>() == 32 + 16 * 16);

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SpawnRequest {
    pub bytecode_offset: u32,
    pub bytecode_len: u32,
    pub priority: u32,
    pub _padding: u32,
}

const _: () = assert!(std::mem::size_of::<SpawnRequest>() == 16);

pub struct PersistentRuntime {
    device: Device,
    command_queue: CommandQueue,
    pipeline: Option<ComputePipelineState>,

    // Buffers
    process_buffer: Buffer,
    bytecode_pool: Buffer,
    heap_pool: Buffer,
    system_state: Buffer,
    input_queue: Buffer,
    vertex_output: Buffer,

    // Tracking
    bytecode_write_offset: usize,
    running: bool,
}

impl PersistentRuntime {
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Create buffers
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

        let input_queue = device.new_buffer(
            4096,
            MTLResourceOptions::StorageModeShared,
        );

        let vertex_output = device.new_buffer(
            1024 * 1024,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize system state with volatile writes
        unsafe {
            let state = system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).process_count, 0);
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 0);
            std::ptr::write_volatile(&mut (*state).frame_counter, 0);
            std::ptr::write_volatile(&mut (*state).spawn_head, 0);
            std::ptr::write_volatile(&mut (*state).spawn_tail, 0);
            std::ptr::write_volatile(&mut (*state).free_list_head, 0);
            fence(Ordering::Release);
        }

        // Initialize all processes as EMPTY
        unsafe {
            let procs = process_buffer.contents() as *mut Process;
            for i in 0..MAX_PROCESSES {
                std::ptr::write_volatile(&mut (*procs.add(i)).status, 0);  // EMPTY
            }
            fence(Ordering::Release);
        }

        Ok(Self {
            device: device.clone(),
            command_queue,
            pipeline: None,  // Will be set when shader is available
            process_buffer,
            bytecode_pool,
            heap_pool,
            system_state,
            input_queue,
            vertex_output,
            bytecode_write_offset: 0,
            running: false,
        })
    }

    /// Load bytecode into the pool, return offset and length
    pub fn load_bytecode(&mut self, bytecode: &[u8]) -> Result<(u32, u32), String> {
        if self.bytecode_write_offset + bytecode.len() > BYTECODE_POOL_SIZE {
            return Err("Bytecode pool full".to_string());
        }

        let offset = self.bytecode_write_offset as u32;
        let len = bytecode.len() as u32;

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

    /// Spawn a new process
    pub fn spawn(&self, bytecode_offset: u32, bytecode_len: u32, priority: u32) -> Result<(), String> {
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;

            let tail = std::ptr::read_volatile(&(*state).spawn_tail);
            let head = std::ptr::read_volatile(&(*state).spawn_head);

            // Check queue full (max 16 pending spawn requests)
            if tail.wrapping_sub(head) >= 16 {
                return Err("Spawn queue full".to_string());
            }

            // Write spawn request
            let idx = (tail % 16) as usize;
            std::ptr::write_volatile(
                &mut (*state).spawn_queue[idx],
                SpawnRequest {
                    bytecode_offset,
                    bytecode_len,
                    priority,
                    _padding: 0,
                }
            );

            fence(Ordering::Release);
            std::ptr::write_volatile(&mut (*state).spawn_tail, tail.wrapping_add(1));
        }

        Ok(())
    }

    /// Start the persistent kernel
    pub fn start(&mut self) {
        if self.running {
            return;
        }
        self.running = true;

        // STUB: In the real implementation, this would dispatch the compute kernel
        // For now, we simulate frame counter advancement
    }

    /// Stop the persistent kernel
    pub fn stop(&mut self) {
        if !self.running {
            return;
        }

        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
            fence(Ordering::Release);
        }

        // Wait for kernel to exit
        std::thread::sleep(std::time::Duration::from_millis(50));

        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 0);
        }

        self.running = false;
    }

    /// Get current frame count (volatile read)
    pub fn frame_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            std::ptr::read_volatile(&(*state).frame_counter)
        }
    }

    /// Get process count (volatile read)
    pub fn process_count(&self) -> u32 {
        fence(Ordering::Acquire);
        unsafe {
            let state = self.system_state.contents() as *const SystemState;
            std::ptr::read_volatile(&(*state).process_count)
        }
    }

    /// Read a process's register value (for testing)
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

    /// Check if a process is dead (volatile read)
    pub fn is_dead(&self, proc_idx: usize) -> bool {
        if proc_idx >= MAX_PROCESSES {
            return true;
        }
        fence(Ordering::Acquire);
        unsafe {
            let procs = self.process_buffer.contents() as *const Process;
            std::ptr::read_volatile(&(*procs.add(proc_idx)).status) == 5  // DEAD
        }
    }
}

impl Drop for PersistentRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}
