# PRD: Persistent Runtime - All-Threads-Participate Execution Model

## Problem Statement

The current `GpuAppSystem` crashes the computer after ~5M GPU iterations because it uses single-thread patterns (`if (tid != 0) return;`) in persistent loops. This violates GPU SIMD execution requirements where all 32 threads in a wavefront must participate in any `while(true)` loop.

**Evidence:**
- 2026-01-28: Running `test_gpu_execution` caused complete system freeze
- 9+ instances of `if (tid != 0) return;` in `gpu_app_system.rs`
- Proven: All-threads-participate loops run 87M+ iterations without issue

## Goal

Build a new execution runtime that:
1. Runs indefinitely without crashing (proven pattern)
2. Executes multiple WASM programs concurrently (utilize all SIMD lanes)
3. Maintains zero CPU involvement in steady state
4. Reuses working components (translator, memory, atomics)

## Critical Fixes Applied (2026-01-28 Review)

Based on 5-agent parallel review, the following critical issues were identified and fixed:

| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| **Race condition in process selection** | CRITICAL | Use atomic CAS to exclusively claim process (prevents multiple threads claiming same process) |
| **Nested while loop in spawn queue** | CRITICAL | Parallelize spawn processing - each thread handles one spawn request, no loops on thread 0 alone |
| **O(n) linear scan for free slot** | HIGH | Add `free_list_head` atomic for O(1) allocation |
| **Non-atomic memory access in Rust** | CRITICAL | Use `volatile` + `fence(Ordering::Release/Acquire)` for all GPU-shared memory |
| **Process struct not 16-byte aligned** | HIGH | Add `_padding: [u32; 3]` to reach 432 bytes |
| **Missing tests for edge cases** | HIGH | Added: zero processes, 64 processes, isolation, queue full, 5M regression |

### Why These Fixes Matter

The original PRD had the same class of bug that crashed the computer - a `while` loop running on thread 0 only inside `process_spawn_queue()`. While it was inside a barrier-protected section (better than the original crash), it still violated the "no thread-0-only loops" principle.

The fixed version has ALL threads participate in spawn processing, with each thread handling at most one spawn request per frame.

## Architecture

### Core Insight: SIMD Lanes = Process Slots

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Threadgroup (32 threads)                  │
├────────┬────────┬────────┬────────┬─────────────────────────────┤
│ Lane 0 │ Lane 1 │ Lane 2 │ ...    │ Lane 31                     │
│ Proc 0 │ Proc 1 │ Proc 2 │ ...    │ Proc 31                     │
│ (calc) │ (idle) │ (game) │ ...    │ (indexer)                   │
└────────┴────────┴────────┴────────┴─────────────────────────────┘

All 32 lanes execute the SAME instruction, but operate on DIFFERENT process state.
This is legal SIMD - no divergence, no stalls.
```

### Design Principle: Idle Until Needed, Instant Spin-Up

**No background services by default.** Threads without work simply skip the execute phase.

```
Boot:
  proc_count = 0
  All 32 threads: "Do I have work?" → No → Skip execute phase → Hit barrier → Loop

User launches calculator:
  CPU writes SpawnRequest to queue (1 atomic write)

Next frame (~10μs later):
  Thread 0: Processes spawn queue, initializes Process[0], sets status=READY
  proc_count = 1

Next iteration:
  Thread 0: "Process[0] is READY" → Execute calculator bytecode
  Threads 1-31: "No process for me" → Skip → Hit barrier → Loop
```

**Cost of idle threads:** ~10 instructions per iteration (loop check, barrier). Negligible.

**Spin-up time:** Instant. No kernel launch, no buffer allocation, no CPU roundtrip.
Process table is pre-allocated. Spawn = write offset + set status.

**Why not pre-spawn system services?**
- Simpler mental model (process count = user processes)
- No wasted cycles on unwanted background work
- System services can be added later if needed
- Keeps the runtime minimal and auditable

### Memory Layout

```
Buffer 0: Process Table (read/write)
┌──────────────────────────────────────────────────────────────────┐
│ Process[0]  │ Process[1]  │ ... │ Process[MAX_PROCESSES-1]       │
├─────────────┼─────────────┼─────┼────────────────────────────────┤
│ pc: u32     │ pc: u32     │     │ Program counter                │
│ sp: u32     │ sp: u32     │     │ Stack pointer                  │
│ status: u32 │ status: u32 │     │ READY/RUNNING/BLOCKED/DEAD     │
│ regs[64]    │ regs[64]    │     │ Virtual registers              │
│ heap_base   │ heap_base   │     │ Process heap start             │
│ heap_size   │ heap_size   │     │ Process heap size              │
│ bytecode_ptr│ bytecode_ptr│     │ Pointer to bytecode in pool    │
│ bytecode_len│ bytecode_len│     │ Length of bytecode             │
└─────────────┴─────────────┴─────┴────────────────────────────────┘

Buffer 1: Bytecode Pool (read-only after load)
┌──────────────────────────────────────────────────────────────────┐
│ [Program 0 bytecode...] [Program 1 bytecode...] [Program 2...]   │
└──────────────────────────────────────────────────────────────────┘

Buffer 2: Heap Pool (read/write)
┌──────────────────────────────────────────────────────────────────┐
│ [Process 0 heap] [Process 1 heap] [Process 2 heap] ...           │
└──────────────────────────────────────────────────────────────────┘

Buffer 3: System State (atomics)
┌──────────────────────────────────────────────────────────────────┐
│ process_count: atomic_uint                                        │
│ shutdown_flag: atomic_uint                                        │
│ frame_counter: atomic_uint                                        │
│ spawn_queue[16]: SpawnRequest                                     │
│ spawn_queue_head: atomic_uint                                     │
│ spawn_queue_tail: atomic_uint                                     │
└──────────────────────────────────────────────────────────────────┘

Buffer 4: Input Queue (write by CPU, read by GPU)
┌──────────────────────────────────────────────────────────────────┐
│ events[256]: InputEvent                                           │
│ head: atomic_uint                                                 │
│ tail: atomic_uint                                                 │
└──────────────────────────────────────────────────────────────────┘

Buffer 5: Vertex Output (write by GPU, read by CPU for display)
┌──────────────────────────────────────────────────────────────────┐
│ vertices[MAX_VERTICES]: Vertex                                    │
│ vertex_count: atomic_uint                                         │
└──────────────────────────────────────────────────────────────────┘
```

## Pseudocode

### Main Kernel (Metal)

```metal
// persistent_runtime.metal

constant uint TIMESLICE = 1000;  // Instructions per scheduling quantum
constant uint MAX_PROCESSES = 64;
constant uint PROCESS_HEAP_SIZE = 65536;  // 64KB per process

enum ProcessStatus : uint {
    EMPTY = 0,
    READY = 1,
    RUNNING = 2,
    BLOCKED_IO = 3,
    BLOCKED_INPUT = 4,
    DEAD = 5
};

// CRITICAL: Struct must be 16-byte aligned for Metal
// Total size: 432 bytes (27 * 16)
struct Process {
    uint pc;                    // Program counter (0-3)
    uint sp;                    // Stack pointer (4-7)
    uint status;                // ProcessStatus (8-11) - USE ATOMIC ACCESS
    uint bytecode_offset;       // Offset into bytecode pool (12-15)
    uint bytecode_len;          // Length of bytecode (16-19)
    uint heap_offset;           // Offset into heap pool (20-23)
    uint heap_size;             // Allocated heap size (24-27)
    int regs[64];               // Virtual registers (28-283, 256 bytes)
    float fregs[32];            // Float registers (284-411, 128 bytes)
    uint blocked_on;            // What we're waiting for (412-415)
    uint priority;              // Scheduling priority (416-419)
    uint _padding[3];           // Align to 432 bytes (420-431)
};

struct SystemState {
    atomic_uint process_count;
    atomic_uint shutdown_flag;
    atomic_uint frame_counter;
    atomic_uint spawn_head;
    atomic_uint spawn_tail;
    atomic_uint free_list_head; // O(1) allocation: index into free_slots
    uint _padding[2];           // Align to 32 bytes
    // Spawn queue follows at offset 32
};

struct SpawnRequest {
    uint bytecode_offset;
    uint bytecode_len;
    uint priority;
    uint _padding;
};

// Vertex struct - CRITICAL: Use packed_float3 to match Rust [f32; 3]
struct Vertex {
    packed_float3 position;     // 12 bytes - matches Rust
    float _pad0;                // 4 bytes padding
    float4 color;               // 16 bytes
};

kernel void persistent_runtime(
    device Process* processes [[buffer(0)]],
    device uchar* bytecode_pool [[buffer(1)]],
    device uchar* heap_pool [[buffer(2)]],
    device SystemState* system [[buffer(3)]],
    device InputQueue* input [[buffer(4)]],
    device VertexOutput* vertices [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Calculate global thread ID
    uint global_tid = tgid * 32 + tid;

    // CRITICAL: ALL threads enter this loop
    while (true) {
        // ═══════════════════════════════════════════════════════════
        // PHASE 1: Check shutdown (ALL threads)
        // ═══════════════════════════════════════════════════════════
        if (atomic_load_explicit(&system->shutdown_flag, memory_order_relaxed)) {
            break;  // ALL threads exit together
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 2: Select process (ALL threads, exclusive claim via atomic CAS)
        // ═══════════════════════════════════════════════════════════
        uint proc_count = atomic_load_explicit(&system->process_count, memory_order_relaxed);

        // Simple 1:1 mapping: thread N claims process N (if it exists)
        // This avoids contention when proc_count < 32
        uint my_proc_idx = global_tid;

        // Use atomic CAS to exclusively claim this process
        // CRITICAL: This prevents race condition where multiple threads claim same process
        bool have_work = false;
        if (my_proc_idx < proc_count) {
            uint expected = READY;
            bool claimed = atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)&processes[my_proc_idx].status,
                &expected,
                RUNNING,
                memory_order_relaxed,
                memory_order_relaxed
            );
            if (claimed) {
                have_work = true;
            }
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 3: Execute timeslice (ALL threads execute same instruction)
        // ═══════════════════════════════════════════════════════════
        for (uint i = 0; i < TIMESLICE; i++) {
            device Process* proc = &processes[my_proc_idx];
            device uchar* code = bytecode_pool + proc->bytecode_offset;
            device uchar* heap = heap_pool + proc->heap_offset;

            uint pc = proc->pc;
            uint max_pc = proc->bytecode_len / 8;  // Instructions are 8 bytes

            // BOUNDS CHECK: Verify PC is within bytecode bounds
            // If out of bounds, treat as HALT (prevents reading garbage)
            if (have_work && pc >= max_pc) {
                proc->status = DEAD;
                have_work = false;
            }

            // Fetch instruction (8 bytes) - safe if bounds check passed
            uchar opcode = have_work ? code[pc * 8 + 0] : OP_NOP;
            uchar dst = code[pc * 8 + 1];
            uchar src1 = code[pc * 8 + 2];
            uchar src2 = code[pc * 8 + 3];
            int imm = as_type<int>(*(device uint*)(code + pc * 8 + 4));

            // BOUNDS CHECK: Validate register indices (0-63)
            dst = dst & 0x3F;    // Clamp to 0-63
            src1 = src1 & 0x3F;
            src2 = src2 & 0x3F;

            // Execute (conditional on have_work, but ALL threads execute this block)
            if (have_work) {
                switch (opcode) {
                    case OP_NOP:
                        break;
                    case OP_CONST:
                        proc->regs[dst] = imm;
                        break;
                    case OP_ADD:
                        proc->regs[dst] = proc->regs[src1] + proc->regs[src2];
                        break;
                    case OP_SUB:
                        proc->regs[dst] = proc->regs[src1] - proc->regs[src2];
                        break;
                    case OP_MUL:
                        proc->regs[dst] = proc->regs[src1] * proc->regs[src2];
                        break;
                    case OP_DIV:
                        // Division by zero: result is 0 (safe, no crash)
                        if (proc->regs[src2] != 0) {
                            proc->regs[dst] = proc->regs[src1] / proc->regs[src2];
                        } else {
                            proc->regs[dst] = 0;
                        }
                        break;
                    case OP_MOD:
                        // Modulo by zero: result is 0 (safe, no crash)
                        if (proc->regs[src2] != 0) {
                            proc->regs[dst] = proc->regs[src1] % proc->regs[src2];
                        } else {
                            proc->regs[dst] = 0;
                        }
                        break;
                    case OP_AND:
                        proc->regs[dst] = proc->regs[src1] & proc->regs[src2];
                        break;
                    case OP_OR:
                        proc->regs[dst] = proc->regs[src1] | proc->regs[src2];
                        break;
                    case OP_XOR:
                        proc->regs[dst] = proc->regs[src1] ^ proc->regs[src2];
                        break;
                    case OP_SHL:
                        proc->regs[dst] = proc->regs[src1] << (proc->regs[src2] & 0x1F);
                        break;
                    case OP_SHR:
                        proc->regs[dst] = (uint)proc->regs[src1] >> (proc->regs[src2] & 0x1F);
                        break;
                    case OP_LOAD:
                        // BOUNDS CHECK: Heap access
                        {
                            uint addr = (uint)proc->regs[src1];
                            if (addr + 4 <= proc->heap_size) {
                                proc->regs[dst] = *(device int*)(heap + addr);
                            } else {
                                proc->regs[dst] = 0;  // Out of bounds: return 0
                            }
                        }
                        break;
                    case OP_STORE:
                        // BOUNDS CHECK: Heap access
                        {
                            uint addr = (uint)proc->regs[dst];
                            if (addr + 4 <= proc->heap_size) {
                                *(device int*)(heap + addr) = proc->regs[src1];
                            }
                            // Out of bounds: silently ignore (no crash)
                        }
                        break;
                    case OP_JUMP:
                        proc->pc = imm - 1;  // -1 because we increment below
                        break;
                    case OP_JUMP_IF:
                        if (proc->regs[src1] != 0) {
                            proc->pc = imm - 1;
                        }
                        break;
                    case OP_CALL:
                        // Push return address, jump to function
                        // BOUNDS CHECK: Stack overflow protection
                        if (proc->sp >= 4) {
                            proc->sp -= 4;
                            *(device int*)(heap + proc->sp) = proc->pc + 1;
                            proc->pc = imm - 1;
                        } else {
                            // Stack overflow - kill process
                            proc->status = DEAD;
                            have_work = false;
                        }
                        break;
                    case OP_RET:
                        // Pop return address
                        // BOUNDS CHECK: Stack underflow protection
                        if (proc->sp + 4 <= proc->heap_size) {
                            proc->pc = *(device int*)(heap + proc->sp) - 1;
                            proc->sp += 4;
                        } else {
                            // Stack underflow - kill process
                            proc->status = DEAD;
                            have_work = false;
                        }
                        break;
                    case OP_HALT:
                        proc->status = DEAD;
                        have_work = false;
                        break;
                    case OP_EMIT_QUAD:
                        // GPU intrinsic: emit a quad for rendering
                        emit_quad(vertices, proc->fregs[src1], proc->fregs[src1+1],
                                  proc->fregs[src2], proc->fregs[src2+1],
                                  as_type<uint>(proc->regs[dst]));
                        break;
                    case OP_YIELD:
                        // Voluntarily give up timeslice
                        i = TIMESLICE;  // Exit inner loop
                        break;
                    default:
                        // INVALID OPCODE: Kill process rather than undefined behavior
                        proc->status = DEAD;
                        have_work = false;
                        break;
                }
                proc->pc++;
            }
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 4: Mark process as READY again (ALL threads)
        // ═══════════════════════════════════════════════════════════
        if (have_work && processes[my_proc_idx].status == RUNNING) {
            processes[my_proc_idx].status = READY;
        }

        // ═══════════════════════════════════════════════════════════
        // PHASE 5: System tasks - PARALLELIZED across all threads
        //          NO nested while loops - each thread handles one item
        // ═══════════════════════════════════════════════════════════
        threadgroup_barrier(mem_flags::mem_device);

        // SPAWN QUEUE: All threads participate (up to 16 threads handle 16 spawn requests)
        // This avoids the anti-pattern of thread 0 running a while loop alone
        {
            uint head = atomic_load_explicit(&system->spawn_head, memory_order_relaxed);
            uint tail = atomic_load_explicit(&system->spawn_tail, memory_order_relaxed);
            uint queue_len = tail - head;

            // Each thread handles one spawn request (if available)
            if (tid < queue_len && tid < 16) {
                SpawnRequest req = system->spawn_queue[(head + tid) % 16];

                // O(1) allocation: atomically claim a slot from free list
                uint slot = atomic_fetch_add_explicit(&system->free_list_head, 1, memory_order_relaxed);

                if (slot < MAX_PROCESSES) {
                    // Initialize process
                    processes[slot].pc = 0;
                    processes[slot].sp = PROCESS_HEAP_SIZE;  // Stack grows down
                    processes[slot].bytecode_offset = req.bytecode_offset;
                    processes[slot].bytecode_len = req.bytecode_len;
                    processes[slot].heap_offset = slot * PROCESS_HEAP_SIZE;
                    processes[slot].heap_size = PROCESS_HEAP_SIZE;

                    // NOTE: Memory fence via atomic_store below, no barrier needed here
                    // (barrier would be wrong - not all threads in this block)

                    // Now set status to READY (makes it claimable)
                    // Use memory_order_release to ensure init fields visible before status
                    atomic_store_explicit(
                        (device atomic_uint*)&processes[slot].status,
                        READY,
                        memory_order_release
                    );

                    // Update process count (atomic max)
                    uint old_count = atomic_load_explicit(&system->process_count, memory_order_relaxed);
                    while (slot + 1 > old_count) {
                        atomic_compare_exchange_weak_explicit(
                            &system->process_count,
                            &old_count,
                            slot + 1,
                            memory_order_relaxed,
                            memory_order_relaxed
                        );
                    }
                }
            }

            // Thread 0 updates spawn queue head after all threads processed
            threadgroup_barrier(mem_flags::mem_device);
            if (tid == 0) {
                uint processed = min(queue_len, 16u);
                atomic_store_explicit(&system->spawn_head, head + processed, memory_order_relaxed);
            }
        }

        // INPUT QUEUE: Thread 0 processes (bounded, no while loop)
        // TODO: Parallelize this too if needed
        if (tgid == 0 && tid == 0) {
            // Process up to 32 input events per frame (bounded, no while loop)
            uint head = atomic_load_explicit(&input->head, memory_order_relaxed);
            uint tail = atomic_load_explicit(&input->tail, memory_order_relaxed);
            uint to_process = min(tail - head, 32u);

            for (uint i = 0; i < to_process; i++) {
                // Dispatch input event to focused process
                // (simplified - real impl would route based on focus)
            }

            atomic_store_explicit(&input->head, head + to_process, memory_order_relaxed);

            // Increment frame counter
            atomic_fetch_add_explicit(&system->frame_counter, 1, memory_order_relaxed);
        }

        // ALL threads sync before next iteration
        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

### Rust Host Code

```rust
// persistent_runtime.rs

use metal::*;
use std::sync::atomic::{fence, Ordering, AtomicBool};
use std::sync::Arc;

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
    pipeline: ComputePipelineState,

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

        // Load shader
        let library = device
            .new_library_with_source(include_str!("shaders/persistent_runtime.metal"), &CompileOptions::new())
            .map_err(|e| format!("Shader compile error: {}", e))?;

        let function = library
            .get_function("persistent_runtime", None)
            .map_err(|e| format!("Function not found: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline error: {}", e))?;

        // Create buffers
        // NOTE: Use StorageModeShared for buffers CPU needs to access
        // GPU-only buffers could use StorageModePrivate for better performance
        // but we use Shared here for debugging/testing access

        let process_buffer = device.new_buffer(
            (MAX_PROCESSES * std::mem::size_of::<Process>()) as u64,
            MTLResourceOptions::StorageModeShared,  // CPU needs to read process state
        );

        let bytecode_pool = device.new_buffer(
            BYTECODE_POOL_SIZE as u64,
            MTLResourceOptions::StorageModeShared,  // CPU writes bytecode
        );

        let heap_pool = device.new_buffer(
            HEAP_POOL_SIZE as u64,
            MTLResourceOptions::StorageModeShared,  // Could be Private, but Shared for debug
        );

        let system_state = device.new_buffer(
            std::mem::size_of::<SystemState>() as u64,
            MTLResourceOptions::StorageModeShared,  // CPU writes spawn requests, reads state
        );

        let input_queue = device.new_buffer(
            4096,  // Input queue size
            MTLResourceOptions::StorageModeShared,  // CPU writes input events
        );

        let vertex_output = device.new_buffer(
            1024 * 1024,  // 1MB vertex buffer
            MTLResourceOptions::StorageModeShared,  // CPU reads vertices for display
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
            fence(Ordering::Release);  // Ensure init visible to GPU
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
            pipeline,
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

    /// Load bytecode into the pool, return offset
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
            // CRITICAL: Fence ensures bytecode visible to GPU before we return
            fence(Ordering::Release);
        }

        self.bytecode_write_offset += bytecode.len();
        Ok((offset, len))
    }

    /// Spawn a new process
    pub fn spawn(&self, bytecode_offset: u32, bytecode_len: u32, priority: u32) -> Result<(), String> {
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;

            // Use volatile reads to get current queue state
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

            // CRITICAL: Release fence ensures spawn request visible before tail update
            fence(Ordering::Release);

            // Advance tail (GPU sees new request after this write)
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

        // Dispatch kernel - it runs forever until shutdown
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.process_buffer), 0);
        encoder.set_buffer(1, Some(&self.bytecode_pool), 0);
        encoder.set_buffer(2, Some(&self.heap_pool), 0);
        encoder.set_buffer(3, Some(&self.system_state), 0);
        encoder.set_buffer(4, Some(&self.input_queue), 0);
        encoder.set_buffer(5, Some(&self.vertex_output), 0);

        // Dispatch 2 threadgroups of 32 threads = 64 threads = MAX_PROCESSES
        // Each thread handles one process slot (1:1 mapping)
        encoder.dispatch_thread_groups(
            MTLSize::new(2, 1, 1),  // 2 threadgroups
            MTLSize::new(32, 1, 1), // 32 threads per group
        );

        encoder.end_encoding();
        command_buffer.commit();

        // Note: We do NOT wait - the kernel runs indefinitely
    }

    /// Stop the persistent kernel
    pub fn stop(&mut self) {
        if !self.running {
            return;
        }

        // Signal shutdown using volatile write
        unsafe {
            let state = self.system_state.contents() as *mut SystemState;
            std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
            fence(Ordering::Release);  // Ensure GPU sees shutdown flag
        }

        // Wait for kernel to exit (poll frame counter with volatile reads)
        let mut last_frame = 0u32;
        let mut stable_count = 0;
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(10));

            fence(Ordering::Acquire);  // Ensure we see GPU's writes
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
```

## Tests

### Unit Tests

```rust
// tests/test_persistent_runtime.rs

use rust_experiment::gpu_os::persistent_runtime::PersistentRuntime;
use metal::Device;

/// Test: Kernel starts and stops cleanly
#[test]
fn test_start_stop() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));

    let frames = runtime.frame_count();
    assert!(frames > 0, "Kernel should have run some frames");

    runtime.stop();

    let final_frames = runtime.frame_count();
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert_eq!(runtime.frame_count(), final_frames, "Kernel should have stopped");
}

/// Test: Single process executes and halts
#[test]
fn test_single_process() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Simple bytecode: CONST r0, 42; HALT
    let bytecode = vec![
        0x01, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00,  // CONST r0, 42
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    let (offset, len) = runtime.load_bytecode(&bytecode).expect("Load failed");
    runtime.spawn(offset, len, 0).expect("Spawn failed");

    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0), "Process should have halted");
    assert_eq!(runtime.read_register(0, 0), Some(42), "r0 should be 42");
}

/// Test: Multiple processes run concurrently
#[test]
fn test_multiple_processes() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Process 0: CONST r0, 100; HALT
    let bytecode0 = vec![
        0x01, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,  // CONST r0, 100
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    // Process 1: CONST r0, 200; HALT
    let bytecode1 = vec![
        0x01, 0x00, 0x00, 0x00, 0xC8, 0x00, 0x00, 0x00,  // CONST r0, 200
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    let (off0, len0) = runtime.load_bytecode(&bytecode0).unwrap();
    let (off1, len1) = runtime.load_bytecode(&bytecode1).unwrap();

    runtime.spawn(off0, len0, 0).unwrap();
    runtime.spawn(off1, len1, 0).unwrap();

    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0), "Process 0 should have halted");
    assert!(runtime.is_dead(1), "Process 1 should have halted");
    assert_eq!(runtime.read_register(0, 0), Some(100), "Process 0: r0 should be 100");
    assert_eq!(runtime.read_register(1, 0), Some(200), "Process 1: r0 should be 200");
}

/// Test: Long-running process doesn't crash (critical!)
#[test]
fn test_long_running_no_crash() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Bytecode: Loop 10 million times, then halt
    // CONST r0, 0          ; counter = 0
    // CONST r1, 10000000   ; limit = 10M
    // loop:
    // ADD r0, r0, 1        ; counter++
    // SUB r2, r1, r0       ; r2 = limit - counter
    // JUMP_IF r2, loop     ; if r2 != 0, loop
    // HALT
    let bytecode = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r0, 0
        0x01, 0x01, 0x00, 0x00, 0x80, 0x96, 0x98, 0x00,  // CONST r1, 10000000
        0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,  // ADD r0, r0, 1 (using imm=1)
        0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,  // SUB r2, r1, r0
        0x0B, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00,  // JUMP_IF r2, 2
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();

    // Wait up to 30 seconds for completion
    for _ in 0..300 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if runtime.is_dead(0) {
            break;
        }
    }

    runtime.stop();

    assert!(runtime.is_dead(0), "Process should have completed");
    assert_eq!(runtime.read_register(0, 0), Some(10_000_000), "Counter should be 10M");
}

/// Test: 32 processes run in parallel (full SIMD utilization)
#[test]
fn test_32_processes_parallel() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Each process: CONST r0, <its_index>; HALT
    for i in 0..32u32 {
        let bytecode = vec![
            0x01, 0x00, 0x00, 0x00,
            (i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, ((i >> 16) & 0xFF) as u8, ((i >> 24) & 0xFF) as u8,
            0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
        runtime.spawn(offset, len, 0).unwrap();
    }

    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    runtime.stop();

    for i in 0..32 {
        assert!(runtime.is_dead(i), "Process {} should have halted", i);
        assert_eq!(runtime.read_register(i, 0), Some(i as i32), "Process {} r0 should be {}", i, i);
    }
}
```

### Stress Tests

```rust
/// Test: Run for 60 seconds without crash
#[test]
#[ignore]  // Long test - run with --ignored
fn test_60_second_stability() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Spawn an infinite loop process
    let bytecode = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r0, 0
        0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,  // ADD r0, r0, 1
        0x0A, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,  // JUMP 1
    ];

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();

    let start = std::time::Instant::now();
    let mut last_frame = 0;

    while start.elapsed() < std::time::Duration::from_secs(60) {
        std::thread::sleep(std::time::Duration::from_secs(1));

        let frame = runtime.frame_count();
        assert!(frame > last_frame, "Kernel should still be running at {:?}", start.elapsed());
        last_frame = frame;

        println!("Elapsed: {:?}, Frames: {}", start.elapsed(), frame);
    }

    runtime.stop();
    println!("Completed 60 seconds without crash");
}

/// REGRESSION TEST: Verify we don't hit the ~5M iteration crash threshold
/// This is the specific failure mode that crashed the computer with the old design
#[test]
fn test_5m_iteration_regression() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Count to 6 million (beyond the ~5M crash threshold)
    // Use SUB + JUMP_IF pattern (no JUMP_LT opcode exists)
    let bytecode = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r0, 0 (counter)
        0x01, 0x01, 0x00, 0x00, 0x80, 0x8D, 0x5B, 0x00,  // CONST r1, 6000000 (limit)
        0x01, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,  // CONST r2, 1
        0x02, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,  // ADD r0, r0, r2 (counter++)
        0x03, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,  // SUB r3, r1, r0 (remaining = limit - counter)
        0x0B, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00,  // JUMP_IF r3, 3 (if remaining != 0, goto ADD)
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();

    runtime.start();

    let start = std::time::Instant::now();
    while !runtime.is_dead(0) && start.elapsed() < std::time::Duration::from_secs(120) {
        std::thread::sleep(std::time::Duration::from_millis(500));
        println!("Still running at {:?}, frames: {}", start.elapsed(), runtime.frame_count());
    }

    runtime.stop();

    assert!(runtime.is_dead(0), "Process should complete 6M iterations without crash");
    assert_eq!(runtime.read_register(0, 0), Some(6_000_000), "Counter should reach 6M");
    println!("REGRESSION TEST PASSED: 6M iterations completed in {:?}", start.elapsed());
}
```

### Edge Case Tests

```rust
/// Test: System runs with zero processes (idle but stable)
#[test]
fn test_zero_processes() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Start with no processes
    runtime.start();

    // Let it run for a bit
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Frame counter should still advance
    let frames = runtime.frame_count();
    assert!(frames > 0, "Kernel should run even with no processes");

    std::thread::sleep(std::time::Duration::from_millis(500));
    let more_frames = runtime.frame_count();
    assert!(more_frames > frames, "Kernel should keep running");

    runtime.stop();
}

/// Test: Maximum 64 processes
#[test]
fn test_max_64_processes() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Spawn 64 processes (the maximum)
    for i in 0..64u32 {
        let bytecode = vec![
            0x01, 0x00, 0x00, 0x00,
            (i & 0xFF) as u8, ((i >> 8) & 0xFF) as u8, 0, 0,
            0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
        runtime.spawn(offset, len, 0).unwrap();
    }

    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(1000));
    runtime.stop();

    // All 64 should complete
    for i in 0..64 {
        assert!(runtime.is_dead(i), "Process {} should have halted", i);
        assert_eq!(runtime.read_register(i, 0), Some(i as i32), "Process {} value", i);
    }
}

/// Test: Process isolation - one process cannot affect another
#[test]
fn test_process_isolation() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Process 0: Write 0xDEAD to heap[0], read back to r0
    let bytecode0 = vec![
        0x01, 0x00, 0x00, 0x00, 0xAD, 0xDE, 0x00, 0x00,  // CONST r0, 0xDEAD
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r1, 0 (address)
        0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // STORE heap[r1], r0
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r0, 0 (clear)
        0x10, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,  // LOAD r0, heap[r1]
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    // Process 1: Write 0xBEEF to heap[0], read back to r0
    let bytecode1 = vec![
        0x01, 0x00, 0x00, 0x00, 0xEF, 0xBE, 0x00, 0x00,  // CONST r0, 0xBEEF
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r1, 0 (address)
        0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // STORE heap[r1], r0
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // CONST r0, 0 (clear)
        0x10, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,  // LOAD r0, heap[r1]
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // HALT
    ];

    let (off0, len0) = runtime.load_bytecode(&bytecode0).unwrap();
    let (off1, len1) = runtime.load_bytecode(&bytecode1).unwrap();

    runtime.spawn(off0, len0, 0).unwrap();
    runtime.spawn(off1, len1, 0).unwrap();

    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    runtime.stop();

    // Each process should see only its own value
    assert_eq!(runtime.read_register(0, 0), Some(0xDEAD), "Process 0 should see 0xDEAD");
    assert_eq!(runtime.read_register(1, 0), Some(0xBEEF), "Process 1 should see 0xBEEF");
}

/// Test: Spawn queue full error
#[test]
fn test_spawn_queue_full() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let bytecode = vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    // Don't start the kernel - let spawn queue fill up
    for i in 0..16 {
        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
        let result = runtime.spawn(offset, len, 0);
        assert!(result.is_ok(), "Spawn {} should succeed", i);
    }

    // 17th should fail
    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    let result = runtime.spawn(offset, len, 0);
    assert!(result.is_err(), "Spawn 17 should fail - queue full");
}

/// Test: Bytecode pool full error
#[test]
fn test_bytecode_pool_full() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Try to fill the 16MB pool
    let large_bytecode = vec![0u8; 1024 * 1024];  // 1MB chunks

    for i in 0..16 {
        let result = runtime.load_bytecode(&large_bytecode);
        assert!(result.is_ok(), "Load {} should succeed", i);
    }

    // 17th should fail (pool is 16MB)
    let result = runtime.load_bytecode(&large_bytecode);
    assert!(result.is_err(), "Load 17 should fail - pool full");
}
```

### Complete Opcode Tests

```rust
// Test each opcode individually to ensure correct behavior

/// Opcodes for reference
const OP_NOP: u8 = 0x00;
const OP_CONST: u8 = 0x01;
const OP_ADD: u8 = 0x02;
const OP_SUB: u8 = 0x03;
const OP_MUL: u8 = 0x04;
const OP_DIV: u8 = 0x05;
const OP_MOD: u8 = 0x06;
const OP_AND: u8 = 0x07;
const OP_OR: u8 = 0x08;
const OP_XOR: u8 = 0x09;
const OP_JUMP: u8 = 0x0A;
const OP_JUMP_IF: u8 = 0x0B;
const OP_SHL: u8 = 0x0C;
const OP_SHR: u8 = 0x0D;
const OP_LOAD: u8 = 0x10;
const OP_STORE: u8 = 0x11;
const OP_CALL: u8 = 0x20;
const OP_RET: u8 = 0x21;
const OP_YIELD: u8 = 0x30;
const OP_EMIT_QUAD: u8 = 0x40;
const OP_HALT: u8 = 0xFF;

/// Helper to build instruction
fn instr(opcode: u8, dst: u8, src1: u8, src2: u8, imm: i32) -> [u8; 8] {
    let imm_bytes = imm.to_le_bytes();
    [opcode, dst, src1, src2, imm_bytes[0], imm_bytes[1], imm_bytes[2], imm_bytes[3]]
}

#[test]
fn test_opcode_nop() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // NOP should do nothing, r0 should remain 0
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_NOP, 0, 0, 0, 0));
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0));
    assert_eq!(runtime.read_register(0, 0), Some(0));
}

#[test]
fn test_opcode_const() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // r0 = 42
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, -100));    // r1 = -100
    bytecode.extend_from_slice(&instr(OP_CONST, 63, 0, 0, 999));    // r63 = 999 (max register)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));
    assert_eq!(runtime.read_register(0, 1), Some(-100));
    assert_eq!(runtime.read_register(0, 63), Some(999));
}

#[test]
fn test_opcode_add() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 10));      // r1 = 10
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 32));      // r2 = 32
    bytecode.extend_from_slice(&instr(OP_ADD, 0, 1, 2, 0));         // r0 = r1 + r2 = 42
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));
}

#[test]
fn test_opcode_sub() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 100));     // r1 = 100
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 58));      // r2 = 58
    bytecode.extend_from_slice(&instr(OP_SUB, 0, 1, 2, 0));         // r0 = r1 - r2 = 42
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));
}

#[test]
fn test_opcode_mul() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 6));       // r1 = 6
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 7));       // r2 = 7
    bytecode.extend_from_slice(&instr(OP_MUL, 0, 1, 2, 0));         // r0 = r1 * r2 = 42
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));
}

#[test]
fn test_opcode_div() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 84));      // r1 = 84
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 2));       // r2 = 2
    bytecode.extend_from_slice(&instr(OP_DIV, 0, 1, 2, 0));         // r0 = r1 / r2 = 42
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));
}

#[test]
fn test_opcode_mod() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 47));      // r1 = 47
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 5));       // r2 = 5
    bytecode.extend_from_slice(&instr(OP_MOD, 0, 1, 2, 0));         // r0 = r1 % r2 = 2
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(2));
}

#[test]
fn test_opcode_bitwise() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 0b1100));  // r1 = 12
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 0b1010));  // r2 = 10
    bytecode.extend_from_slice(&instr(OP_AND, 3, 1, 2, 0));         // r3 = r1 & r2 = 8 (0b1000)
    bytecode.extend_from_slice(&instr(OP_OR, 4, 1, 2, 0));          // r4 = r1 | r2 = 14 (0b1110)
    bytecode.extend_from_slice(&instr(OP_XOR, 5, 1, 2, 0));         // r5 = r1 ^ r2 = 6 (0b0110)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 3), Some(8));   // AND
    assert_eq!(runtime.read_register(0, 4), Some(14));  // OR
    assert_eq!(runtime.read_register(0, 5), Some(6));   // XOR
}

#[test]
fn test_opcode_shift() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 1));       // r1 = 1
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 4));       // r2 = 4
    bytecode.extend_from_slice(&instr(OP_SHL, 3, 1, 2, 0));         // r3 = r1 << r2 = 16
    bytecode.extend_from_slice(&instr(OP_CONST, 4, 0, 0, 64));      // r4 = 64
    bytecode.extend_from_slice(&instr(OP_SHR, 5, 4, 2, 0));         // r5 = r4 >> r2 = 4
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 3), Some(16));  // SHL
    assert_eq!(runtime.read_register(0, 5), Some(4));   // SHR
}

#[test]
fn test_opcode_load_store() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 0xDEAD));  // r0 = 0xDEAD (value)
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 100));     // r1 = 100 (address)
    bytecode.extend_from_slice(&instr(OP_STORE, 1, 0, 0, 0));       // heap[r1] = r0
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 0));       // r0 = 0 (clear)
    bytecode.extend_from_slice(&instr(OP_LOAD, 0, 1, 0, 0));        // r0 = heap[r1]
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(0xDEAD));
}

#[test]
fn test_opcode_jump() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 1));       // 0: r0 = 1
    bytecode.extend_from_slice(&instr(OP_JUMP, 0, 0, 0, 3));        // 1: JUMP to 3
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 999));     // 2: r0 = 999 (skipped!)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));        // 3: HALT

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(1));  // Not 999
}

#[test]
fn test_opcode_jump_if() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Test jump taken (r1 != 0)
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 1));       // 0: r1 = 1 (true)
    bytecode.extend_from_slice(&instr(OP_JUMP_IF, 0, 1, 0, 3));     // 1: JUMP_IF r1, 3
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 999));     // 2: r0 = 999 (skipped!)
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // 3: r0 = 42
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));        // 4: HALT

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));  // Jump taken
}

#[test]
fn test_opcode_call_ret() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Simple function call test
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 0));       // 0: r0 = 0
    bytecode.extend_from_slice(&instr(OP_CALL, 0, 0, 0, 4));        // 1: CALL 4
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 100));     // 2: r1 = 100 (after return)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));        // 3: HALT
    // Function at address 4:
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // 4: r0 = 42
    bytecode.extend_from_slice(&instr(OP_RET, 0, 0, 0, 0));         // 5: RET

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert_eq!(runtime.read_register(0, 0), Some(42));   // Set by function
    assert_eq!(runtime.read_register(0, 1), Some(100));  // Set after return
}
```

### Negative Tests (Bounds Checking / Error Handling)

```rust
/// Test: Invalid opcode kills process (doesn't crash system)
#[test]
fn test_invalid_opcode_kills_process() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Bytecode with invalid opcode 0xFE (not defined)
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // r0 = 42
    bytecode.extend_from_slice(&[0xFE, 0, 0, 0, 0, 0, 0, 0]);       // Invalid opcode!
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 999));     // Should not execute
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    // Process should be dead (killed by invalid opcode)
    assert!(runtime.is_dead(0));
    // r0 should still have 42 (last valid value before invalid opcode)
    assert_eq!(runtime.read_register(0, 0), Some(42));
}

/// Test: Division by zero returns 0 (doesn't crash)
#[test]
fn test_division_by_zero() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 100));     // r1 = 100
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 0));       // r2 = 0
    bytecode.extend_from_slice(&instr(OP_DIV, 0, 1, 2, 0));         // r0 = r1 / r2 = 0 (not crash)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0));
    assert_eq!(runtime.read_register(0, 0), Some(0));  // Safe division by zero
}

/// Test: Modulo by zero returns 0 (doesn't crash)
#[test]
fn test_modulo_by_zero() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 100));     // r1 = 100
    bytecode.extend_from_slice(&instr(OP_CONST, 2, 0, 0, 0));       // r2 = 0
    bytecode.extend_from_slice(&instr(OP_MOD, 0, 1, 2, 0));         // r0 = r1 % r2 = 0 (not crash)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0));
    assert_eq!(runtime.read_register(0, 0), Some(0));  // Safe modulo by zero
}

/// Test: Out of bounds heap read returns 0
#[test]
fn test_heap_out_of_bounds_read() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // r0 = 42
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 999999));  // r1 = 999999 (out of bounds)
    bytecode.extend_from_slice(&instr(OP_LOAD, 0, 1, 0, 0));        // r0 = heap[r1] = 0 (safe)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    assert!(runtime.is_dead(0));
    assert_eq!(runtime.read_register(0, 0), Some(0));  // Out of bounds returns 0
}

/// Test: Out of bounds heap write is ignored
#[test]
fn test_heap_out_of_bounds_write() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // r0 = 42
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 999999));  // r1 = 999999 (out of bounds)
    bytecode.extend_from_slice(&instr(OP_STORE, 1, 0, 0, 0));       // heap[r1] = r0 (silently ignored)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    // Process should complete normally (write was ignored, not crash)
    assert!(runtime.is_dead(0));
    assert_eq!(runtime.read_register(0, 0), Some(42));
}

/// Test: Stack overflow kills process gracefully
#[test]
fn test_stack_overflow() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Recursive function that overflows the stack
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 0));       // 0: r0 = 0 (call counter)
    // Infinite recursion:
    bytecode.extend_from_slice(&instr(OP_CONST, 1, 0, 0, 1));       // 1: r1 = 1
    bytecode.extend_from_slice(&instr(OP_ADD, 0, 0, 1, 0));         // 2: r0 = r0 + 1
    bytecode.extend_from_slice(&instr(OP_CALL, 0, 0, 0, 1));        // 3: CALL 1 (infinite recursion)
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));        // 4: HALT (never reached)

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    runtime.stop();

    // Process should be dead (killed by stack overflow)
    assert!(runtime.is_dead(0));
    // r0 should have count of successful calls before overflow
    let call_count = runtime.read_register(0, 0).unwrap();
    assert!(call_count > 0, "Should have made some calls before overflow");
    println!("Stack overflow after {} recursive calls", call_count);
}

/// Test: PC out of bounds kills process
#[test]
fn test_pc_out_of_bounds() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Jump to invalid address
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(&instr(OP_CONST, 0, 0, 0, 42));      // 0: r0 = 42
    bytecode.extend_from_slice(&instr(OP_JUMP, 0, 0, 0, 999));      // 1: JUMP to 999 (out of bounds)

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    // Process should be dead (killed by PC out of bounds)
    assert!(runtime.is_dead(0));
}

/// Test: Empty bytecode immediately halts
#[test]
fn test_empty_bytecode() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Empty bytecode (0 instructions)
    let bytecode: Vec<u8> = vec![];

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    // Process should be dead (PC immediately out of bounds)
    assert!(runtime.is_dead(0));
}

/// Test: Register index clamping (accessing r100 clamps to valid range)
#[test]
fn test_register_index_clamping() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = PersistentRuntime::new(&device).expect("Failed to create runtime");

    // Try to use register 100, which should be clamped to 100 & 0x3F = 36
    let mut bytecode = Vec::new();
    // Using raw bytes to bypass any Rust validation
    bytecode.extend_from_slice(&[OP_CONST, 100, 0, 0, 42, 0, 0, 0]);  // r100 -> r36 = 42
    bytecode.extend_from_slice(&instr(OP_HALT, 0, 0, 0, 0));

    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
    runtime.spawn(offset, len, 0).unwrap();
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    runtime.stop();

    // r36 should have the value (100 & 0x3F = 36)
    assert_eq!(runtime.read_register(0, 36), Some(42));
}
```

## Benchmarks

```rust
// benches/persistent_runtime_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_experiment::gpu_os::persistent_runtime::PersistentRuntime;
use metal::Device;

fn bench_process_spawn(c: &mut Criterion) {
    let device = Device::system_default().expect("No Metal device");

    c.bench_function("spawn_single_process", |b| {
        b.iter_batched(
            || {
                let mut runtime = PersistentRuntime::new(&device).unwrap();
                let bytecode = vec![0x01, 0x00, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00,
                                   0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
                let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
                (runtime, offset, len)
            },
            |(runtime, offset, len)| {
                runtime.spawn(offset, len, 0).unwrap();
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_process_execution(c: &mut Criterion) {
    let device = Device::system_default().expect("No Metal device");

    let mut group = c.benchmark_group("process_execution");

    for iterations in [1000, 10000, 100000, 1000000].iter() {
        group.bench_with_input(
            BenchmarkId::new("loop_iterations", iterations),
            iterations,
            |b, &iters| {
                b.iter_custom(|_| {
                    let mut runtime = PersistentRuntime::new(&device).unwrap();

                    // Build bytecode for N iterations
                    let bytecode = build_loop_bytecode(iters);
                    let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
                    runtime.spawn(offset, len, 0).unwrap();

                    runtime.start();
                    let start = std::time::Instant::now();

                    while !runtime.is_dead(0) {
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }

                    let elapsed = start.elapsed();
                    runtime.stop();
                    elapsed
                });
            },
        );
    }

    group.finish();
}

fn bench_parallel_processes(c: &mut Criterion) {
    let device = Device::system_default().expect("No Metal device");

    let mut group = c.benchmark_group("parallel_processes");

    for num_procs in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_processes", num_procs),
            num_procs,
            |b, &n| {
                b.iter_custom(|_| {
                    let mut runtime = PersistentRuntime::new(&device).unwrap();

                    // Each process does 100K iterations
                    let bytecode = build_loop_bytecode(100_000);

                    for _ in 0..n {
                        let (offset, len) = runtime.load_bytecode(&bytecode).unwrap();
                        runtime.spawn(offset, len, 0).unwrap();
                    }

                    runtime.start();
                    let start = std::time::Instant::now();

                    loop {
                        let mut all_dead = true;
                        for i in 0..n {
                            if !runtime.is_dead(i) {
                                all_dead = false;
                                break;
                            }
                        }
                        if all_dead { break; }
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }

                    let elapsed = start.elapsed();
                    runtime.stop();
                    elapsed
                });
            },
        );
    }

    group.finish();
}

fn build_loop_bytecode(iterations: u32) -> Vec<u8> {
    // CONST r0, 0
    // CONST r1, iterations
    // loop: ADD r0, r0, 1
    // SUB r2, r1, r0
    // JUMP_IF r2, loop
    // HALT
    vec![
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x01, 0x00, 0x00,
        (iterations & 0xFF) as u8,
        ((iterations >> 8) & 0xFF) as u8,
        ((iterations >> 16) & 0xFF) as u8,
        ((iterations >> 24) & 0xFF) as u8,
        0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x03, 0x02, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0B, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00,
        0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ]
}

criterion_group!(benches, bench_process_spawn, bench_process_execution, bench_parallel_processes);
criterion_main!(benches);
```

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Stability | 60+ seconds without crash | `test_60_second_stability` |
| Process spawn | <1ms | Benchmark |
| 32 processes concurrent | All complete | `test_32_processes_parallel` |
| 10M iterations | Complete, correct result | `test_long_running_no_crash` |
| Frame rate | >100 frames/sec | Monitor `frame_counter` |
| CPU usage | <5% steady state | System monitor |

## Migration Path

### Phase 1: Build New Runtime (Week 1)
- [ ] Create `persistent_runtime.metal` with all-threads-participate pattern
- [ ] Create `persistent_runtime.rs` host code
- [ ] Pass basic unit tests

### Phase 2: Test Stability (Week 2)
- [ ] Pass 60-second stability test
- [ ] Pass 32-process parallel test
- [ ] Pass 10M iteration test

### Phase 3: Add WASM Support (Week 3)
- [ ] Integrate existing WASM translator output
- [ ] Map WASM opcodes to runtime opcodes
- [ ] Test with simple WASM programs

### Phase 4: Replace GpuAppSystem (Week 4)
- [ ] Port input handling
- [ ] Port rendering output
- [ ] Delete old `gpu_app_system.rs`

## Components to Reuse

| Component | Location | Reuse? | Notes |
|-----------|----------|--------|-------|
| WASM translator | `wasm_translator/` | YES | Generates bytecode |
| GPU heap allocator | `gpu_heap.rs` | YES | Heap management |
| Memory types | `memory.rs` | YES | Buffer utilities |
| Input types | `input.rs` | YES | Input event structs |
| Vertex types | `render.rs` | YES | Vertex output format |

## Components to Delete

After successful migration:
- `src/gpu_os/gpu_app_system.rs` (7000+ lines of single-thread patterns)
- `src/gpu_os/shaders/gpu_app_loader.metal`
- All tests that use `GpuAppSystem`

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| New runtime has bugs | Keep old code until new passes all tests |
| Performance worse | Benchmark before/after, optimize if needed |
| Missing opcodes | Add incrementally, track in issues |
| Deadlock | All-threads-participate prevents this by design |
| Memory corruption | Use existing proven heap allocator |

## Non-Goals

- Perfect compatibility with old system (clean break)
- Optimizing for single-process performance (focus on multi-process)
- Supporting CPU fallback (GPU only)
