# PRD: GPU-Native Process Model

## Overview

**THE GPU IS THE COMPUTER.**

This document defines a GPU-native process model where processes are NOT CPU syscall abstractions but rather **GPU-resident execution units** - threadgroups with dedicated state buffers, GPU bytecode, and GPU-native scheduling.

## Why This Matters

| Traditional OS (CPU-Centric) | GPU-Native OS |
|------------------------------|---------------|
| Process = isolated address space + executable + OS scheduling | Process = threadgroup + state buffer + GPU bytecode |
| Requires fork/exec syscalls (CPU) | Zero CPU involvement |
| Context switch: 1-10 microseconds | Context switch: 0 (true parallelism) |
| Max ~1000 processes (practical limit) | Max ~10,000+ processes (threadgroup limit) |
| Sequential scheduler picks "next" | All processes evaluate in parallel |
| IPC via kernel (pipes, sockets, shared mem) | IPC via GPU atomics and shared buffers |

## The Fundamental Insight

On a traditional OS, "processes" exist because the CPU can only execute one instruction stream per core. We need isolation, scheduling, and context switching.

On GPU, we have **thousands of execution units running simultaneously**. There is no "next" process - ALL processes run at once. Each "process" is simply a threadgroup working on its own state buffer.

```
Traditional OS:                      GPU-Native OS:

CPU                                  GPU
 |
[P1] → scheduler → [P2] → ...        [P0][P1][P2][P3]...[P63] (all parallel)
     (sequential)                           (simultaneous)
```

## Architecture

### GpuProcess Structure

```rust
/// GPU-resident process descriptor (128 bytes, GPU-aligned)
#[repr(C)]
pub struct GpuProcess {
    // === Identity (16 bytes) ===
    /// Unique process ID (assigned from atomic counter)
    pub pid: u32,
    /// Parent process ID (0 = init/orphan)
    pub parent_pid: u32,
    /// Process group ID (for signal delivery)
    pub pgid: u32,
    /// Session ID
    pub sid: u32,

    // === Execution State (16 bytes) ===
    /// Process status (atomic for GPU access)
    pub status: u32,           // ProcessStatus enum
    /// Exit code (valid when status == ZOMBIE)
    pub exit_code: i32,
    /// Program counter (instruction offset in bytecode)
    pub program_counter: u32,
    /// Stack pointer (offset in state buffer)
    pub stack_pointer: u32,

    // === Memory Regions (32 bytes) ===
    /// Offset into unified state buffer (process heap)
    pub heap_offset: u32,
    /// Size of allocated heap region
    pub heap_size: u32,
    /// Offset into unified state buffer (process stack)
    pub stack_offset: u32,
    /// Size of allocated stack region
    pub stack_size: u32,
    /// Bytecode offset in binary registry buffer
    pub bytecode_offset: u32,
    /// Bytecode size
    pub bytecode_size: u32,
    /// Padding
    pub _pad0: [u32; 2],

    // === Thread Allocation (16 bytes) ===
    /// First thread ID assigned to this process
    pub thread_start: u32,
    /// Number of threads assigned
    pub thread_count: u32,
    /// Threads currently active (can sleep/exit)
    pub active_threads: u32,
    /// Padding
    pub _pad1: u32,

    // === Scheduling (16 bytes) ===
    /// Priority level (0=idle, 1=background, 2=normal, 3=high, 4=realtime)
    pub priority: u32,
    /// Frame when process last ran
    pub last_run_frame: u32,
    /// Accumulated CPU-equivalent cycles
    pub cpu_time: u32,
    /// Nice value (priority adjustment, -20 to +19)
    pub nice: i32,

    // === IPC (16 bytes) ===
    /// Pending signal mask (atomic)
    pub pending_signals: u32,
    /// Signal handler table offset (in state buffer)
    pub signal_handlers_offset: u32,
    /// Message queue head (for IPC)
    pub mq_head: u32,
    /// Message queue tail
    pub mq_tail: u32,

    // === Resources (16 bytes) ===
    /// File descriptor table offset
    pub fd_table_offset: u32,
    /// Current working directory (inode)
    pub cwd_inode: u32,
    /// User ID
    pub uid: u32,
    /// Group ID
    pub gid: u32,
}

/// Process status values
pub mod ProcessStatus {
    pub const EMPTY: u32 = 0;        // Slot available
    pub const RUNNING: u32 = 1;      // Currently executing
    pub const READY: u32 = 2;        // Ready to run
    pub const BLOCKED: u32 = 3;      // Waiting for I/O or signal
    pub const STOPPED: u32 = 4;      // Stopped (SIGSTOP)
    pub const ZOMBIE: u32 = 5;       // Exited, waiting for parent
}
```

### Process Table (GPU-Resident)

The process table lives entirely in GPU memory:

```metal
// GPU-resident process table
struct ProcessTable {
    // Process descriptors (64 slots = 8KB)
    GpuProcess processes[MAX_PROCESSES];

    // Free slot bitmap (8 bytes for 64 processes)
    atomic_uint free_bitmap_lo;  // Slots 0-31
    atomic_uint free_bitmap_hi;  // Slots 32-63

    // Global counters
    atomic_uint next_pid;        // Monotonically increasing PID
    atomic_uint process_count;   // Current number of active processes

    // Init process always at slot 0
    // pid 1 = init (bootstrapped by system)
};

#define MAX_PROCESSES 64
#define INVALID_PID 0
#define INIT_PID 1
```

### Memory Layout

```
GPU Unified Memory (512 MB total)
┌─────────────────────────────────────────────────────────────────────┐
│ Process Table (8 KB)                                                 │
│   [GpuProcess 0] [GpuProcess 1] ... [GpuProcess 63]                 │
│   [free_bitmap] [next_pid] [process_count]                          │
├─────────────────────────────────────────────────────────────────────┤
│ Binary Registry (64 MB)                                              │
│   Pre-compiled GPU bytecode for all available programs              │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │ [Header: name, size, entry_point] [Bytecode...]            │    │
│   │ [Header: "ls"] [Bytecode...]                               │    │
│   │ [Header: "cat"] [Bytecode...]                              │    │
│   │ [Header: "grep"] [Bytecode...]                             │    │
│   │ [Header: "terminal"] [Bytecode...]                         │    │
│   │ ...                                                         │    │
│   └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│ Unified State Buffer (256 MB)                                        │
│   Per-process heap and stack regions                                │
│   ┌──────────┬──────────┬──────────┬───────────────────────────┐   │
│   │ Proc 0   │ Proc 1   │ Proc 5   │ Free Space               │   │
│   │ Heap+Stk │ Heap+Stk │ Heap+Stk │ (bump allocator)         │   │
│   │ 4MB      │ 2MB      │ 8MB      │                           │   │
│   └──────────┴──────────┴──────────┴───────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│ IPC Buffers (32 MB)                                                  │
│   Shared memory regions, message queues, pipes                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │ Message Queue Pool (ring buffers)                          │    │
│   │ Pipe Buffers (circular buffers)                            │    │
│   │ Shared Memory Regions (explicit mapping)                   │    │
│   └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│ Unified Vertex Buffer (64 MB)                                        │
│   Graphics output for all processes                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Process Lifecycle

### Spawning a Process (All on GPU)

```metal
// GPU kernel: spawn a new process
// Called from parent process (e.g., terminal running "ls")
uint gpu_spawn(
    device ProcessTable* table,
    device uint8_t* state_buffer,
    device uint8_t* binary_registry,
    device AllocatorState* allocator,
    uint parent_pid,
    uint binary_name_hash,  // Hash of program name
    uint thread_count_hint
) {
    // 1. Find binary in registry (O(1) hash lookup)
    uint binary_offset = binary_registry_lookup(binary_registry, binary_name_hash);
    if (binary_offset == INVALID_OFFSET) {
        return INVALID_PID;  // Program not found
    }

    device BinaryHeader* header = (device BinaryHeader*)(binary_registry + binary_offset);

    // 2. Allocate process slot (atomic bitmap operation)
    uint slot = allocate_process_slot(table);
    if (slot == INVALID_SLOT) {
        return INVALID_PID;  // No slots available
    }

    // 3. Assign PID (atomic increment)
    uint pid = atomic_fetch_add_explicit(&table->next_pid, 1, memory_order_relaxed);

    // 4. Allocate memory regions (parallel allocator)
    uint heap_size = header->default_heap_size;
    uint stack_size = header->default_stack_size;
    uint heap_offset = parallel_alloc(allocator, heap_size, 16);
    uint stack_offset = parallel_alloc(allocator, stack_size, 16);

    if (heap_offset == INVALID_OFFSET || stack_offset == INVALID_OFFSET) {
        // Rollback slot allocation
        free_process_slot(table, slot);
        return INVALID_PID;  // Out of memory
    }

    // 5. Reserve thread range
    uint thread_count = min(thread_count_hint, header->max_threads);
    uint thread_start = atomic_fetch_add_explicit(
        &global_thread_allocator, thread_count, memory_order_relaxed
    );

    // 6. Initialize process descriptor
    device GpuProcess* proc = &table->processes[slot];
    proc->pid = pid;
    proc->parent_pid = parent_pid;
    proc->pgid = parent_pid;  // Inherit process group
    proc->sid = table->processes[find_slot_by_pid(table, parent_pid)].sid;

    proc->status = ProcessStatus_READY;
    proc->exit_code = 0;
    proc->program_counter = header->entry_point;
    proc->stack_pointer = stack_offset + stack_size;  // Stack grows down

    proc->heap_offset = heap_offset;
    proc->heap_size = heap_size;
    proc->stack_offset = stack_offset;
    proc->stack_size = stack_size;
    proc->bytecode_offset = binary_offset + sizeof(BinaryHeader);
    proc->bytecode_size = header->bytecode_size;

    proc->thread_start = thread_start;
    proc->thread_count = thread_count;
    proc->active_threads = thread_count;

    proc->priority = PRIORITY_NORMAL;
    proc->last_run_frame = 0;
    proc->cpu_time = 0;
    proc->nice = 0;

    proc->pending_signals = 0;
    proc->signal_handlers_offset = 0;  // Default handlers
    proc->mq_head = 0;
    proc->mq_tail = 0;

    proc->fd_table_offset = 0;  // Inherit from parent (COW)
    proc->cwd_inode = table->processes[find_slot_by_pid(table, parent_pid)].cwd_inode;
    proc->uid = table->processes[find_slot_by_pid(table, parent_pid)].uid;
    proc->gid = table->processes[find_slot_by_pid(table, parent_pid)].gid;

    // 7. Memory barrier to ensure all writes visible
    threadgroup_barrier(mem_flags::mem_device);

    // 8. Mark as ready (this makes it runnable)
    atomic_store_explicit(&proc->status, ProcessStatus_RUNNING, memory_order_release);

    // 9. Increment process count
    atomic_fetch_add_explicit(&table->process_count, 1, memory_order_relaxed);

    return pid;
}
```

### Process Execution (Megakernel Pattern)

```metal
// Master kernel: all processes run in parallel
kernel void process_megakernel(
    device ProcessTable* table [[buffer(0)]],
    device uint8_t* state_buffer [[buffer(1)]],
    device uint8_t* binary_registry [[buffer(2)]],
    device RenderVertex* vertex_buffer [[buffer(3)]],
    device InputEvent* input_queue [[buffer(4)]],
    device FrameState* frame [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Each threadgroup handles one process slot
    if (tgid >= MAX_PROCESSES) return;

    device GpuProcess* proc = &table->processes[tgid];

    // Load process status (atomic)
    uint status = atomic_load_explicit(&proc->status, memory_order_acquire);

    // Skip if not running
    if (status != ProcessStatus_RUNNING) return;

    // Check if this thread is assigned to this process
    uint my_thread_in_proc = lid;
    if (my_thread_in_proc >= proc->thread_count) return;

    // Get process memory regions
    device uint8_t* heap = state_buffer + proc->heap_offset;
    device uint8_t* stack = state_buffer + proc->stack_offset;
    device uint8_t* bytecode = binary_registry + proc->bytecode_offset;

    // Execute bytecode interpreter for this thread
    interpret_bytecode(
        proc,
        heap,
        stack,
        bytecode,
        proc->bytecode_size,
        vertex_buffer,
        input_queue,
        frame,
        my_thread_in_proc,
        proc->thread_count
    );

    // Update last run frame (thread 0 only)
    if (my_thread_in_proc == 0) {
        proc->last_run_frame = frame->frame_number;
    }
}

// Bytecode interpreter (per-thread)
inline void interpret_bytecode(
    device GpuProcess* proc,
    device uint8_t* heap,
    device uint8_t* stack,
    device uint8_t* bytecode,
    uint bytecode_size,
    device RenderVertex* vertices,
    device InputEvent* input,
    device FrameState* frame,
    uint tid,
    uint thread_count
) {
    // Register file (32 registers, float4 each)
    float4 regs[32];
    regs[0] = float4(0);                  // ZERO
    regs[1] = float4(tid);                // TID (thread ID within process)
    regs[2] = float4(thread_count);       // THREAD_COUNT
    regs[3] = float4(frame->frame_number); // FRAME
    regs[4] = float4(proc->pid);          // PID

    // Program counter (process-wide, but each thread may branch differently)
    uint pc = proc->program_counter;
    uint max_instructions = 10000;  // Safety limit per frame

    while (pc < bytecode_size && max_instructions-- > 0) {
        // Fetch instruction (8 bytes)
        device Instruction* inst = (device Instruction*)(bytecode + pc * 8);

        uint op = inst->opcode;
        uint dst = inst->dst;
        uint src1 = inst->src1;
        uint src2 = inst->src2;
        float imm = inst->imm_f;

        switch (op) {
            // Arithmetic
            case OP_NOP: break;
            case OP_MOV: regs[dst] = regs[src1]; break;
            case OP_ADD: regs[dst] = regs[src1] + regs[src2]; break;
            case OP_SUB: regs[dst] = regs[src1] - regs[src2]; break;
            case OP_MUL: regs[dst] = regs[src1] * regs[src2]; break;
            case OP_DIV: regs[dst] = regs[src1] / regs[src2]; break;
            case OP_LOADI: regs[dst] = float4(imm); break;

            // Memory
            case OP_LD: {
                uint addr = uint(regs[src1].x) + uint(imm);
                regs[dst] = *(device float4*)(heap + addr);
                break;
            }
            case OP_ST: {
                uint addr = uint(regs[src1].x) + uint(imm);
                *(device float4*)(heap + addr) = regs[src2];
                break;
            }

            // Control flow
            case OP_JMP: pc = uint(imm) - 1; break;
            case OP_JZ: if (regs[src1].x == 0) pc = uint(imm) - 1; break;
            case OP_JNZ: if (regs[src1].x != 0) pc = uint(imm) - 1; break;

            // Process control (syscall equivalents)
            case OP_EXIT: {
                // Only thread 0 can exit the process
                if (tid == 0) {
                    proc->exit_code = int(regs[src1].x);
                    atomic_store_explicit(&proc->status, ProcessStatus_ZOMBIE, memory_order_release);
                }
                return;  // All threads exit
            }

            case OP_SPAWN: {
                // Fork-like: spawn child process
                // Only thread 0 does the spawn
                if (tid == 0) {
                    uint binary_hash = uint(regs[src1].x);
                    uint child_pid = gpu_spawn(/* ... */);
                    regs[dst] = float4(child_pid);
                }
                // Broadcast result to all threads
                regs[dst] = simd_broadcast_first(regs[dst]);
                break;
            }

            case OP_YIELD: {
                // Cooperative yield - save state and return
                if (tid == 0) {
                    proc->program_counter = pc + 1;
                }
                return;
            }

            case OP_WAIT: {
                // Wait for child process (non-blocking check)
                uint child_pid = uint(regs[src1].x);
                int status = check_child_status(child_pid);
                regs[dst] = float4(status);
                break;
            }

            // Graphics output
            case OP_QUAD: {
                // Each thread can emit geometry
                float2 pos = regs[src1].xy;
                float2 size = regs[src1].zw;
                float4 color = regs[src2];
                float depth = imm;
                uint vert_slot = atomic_fetch_add(&global_vertex_count, 6);
                write_quad(vertices + vert_slot, pos, size, depth, color);
                break;
            }

            // Signals
            case OP_KILL: {
                // Send signal to process
                uint target_pid = uint(regs[src1].x);
                uint signal = uint(regs[src2].x);
                send_signal(target_pid, signal);
                break;
            }

            case OP_HALT: return;

            default: break;
        }

        pc++;
    }

    // Save program counter for next frame
    if (tid == 0) {
        proc->program_counter = pc;
    }
}
```

### Process Exit and Cleanup

```metal
// GPU kernel: handle zombie process cleanup
kernel void process_reaper(
    device ProcessTable* table [[buffer(0)]],
    device AllocatorState* allocator [[buffer(1)]],
    device uint8_t* state_buffer [[buffer(2)]],
    uint slot [[thread_position_in_grid]]
) {
    if (slot >= MAX_PROCESSES) return;

    device GpuProcess* proc = &table->processes[slot];
    uint status = atomic_load_explicit(&proc->status, memory_order_acquire);

    // Only process zombies
    if (status != ProcessStatus_ZOMBIE) return;

    // Check if parent has reaped (wait() called)
    // For simplicity, auto-reap after N frames or if parent is dead
    uint parent_slot = find_slot_by_pid(table, proc->parent_pid);
    bool orphan = (parent_slot == INVALID_SLOT) ||
                  (table->processes[parent_slot].status == ProcessStatus_ZOMBIE);

    if (orphan || (frame_number - proc->last_run_frame > ZOMBIE_TIMEOUT_FRAMES)) {
        // Free memory regions
        parallel_free(allocator, proc->heap_offset, proc->heap_size);
        parallel_free(allocator, proc->stack_offset, proc->stack_size);

        // Free thread allocation
        // (thread allocator handles this)

        // Clear descriptor
        proc->status = ProcessStatus_EMPTY;
        proc->pid = 0;

        // Free process slot
        free_process_slot(table, slot);

        // Decrement process count
        atomic_fetch_sub_explicit(&table->process_count, 1, memory_order_relaxed);
    }
}
```

## Scheduling Model

### GPU-Native Scheduling: Self-Evaluation

There is no central scheduler. Each process evaluates ITSELF in parallel:

```metal
// Each process slot evaluates: "Should I run this frame?"
inline bool should_process_run(
    device const GpuProcess* proc,
    device const SchedulerState* sched,
    uint current_frame
) {
    // Not active? Skip in 1 cycle
    uint status = atomic_load_explicit(&proc->status, memory_order_acquire);
    if (status != ProcessStatus_RUNNING && status != ProcessStatus_READY) {
        return false;
    }

    // Blocked on I/O or signal? Skip
    if (status == ProcessStatus_BLOCKED) {
        return false;
    }

    // Stopped? Skip
    if (status == ProcessStatus_STOPPED) {
        return false;
    }

    // Starvation check: always run if starving
    uint frames_since_run = current_frame - proc->last_run_frame;
    if (frames_since_run > STARVATION_THRESHOLD) {
        return true;
    }

    // Priority-based budget check
    uint effective_priority = compute_effective_priority(proc, current_frame);
    uint budget_for_priority = sched->priority_budgets[effective_priority];

    if (atomic_load(&sched->used_budget) >= budget_for_priority) {
        // Over budget - only high priority runs
        if (effective_priority < PRIORITY_HIGH) {
            return false;
        }
    }

    return true;
}

// Compute effective priority with nice value and starvation boost
inline uint compute_effective_priority(
    device const GpuProcess* proc,
    uint current_frame
) {
    int base_priority = int(proc->priority);

    // Apply nice value (-20 to +19 maps to +2 to -2 priority levels)
    int nice_adjustment = -proc->nice / 10;  // -20 -> +2, +19 -> -1
    base_priority += nice_adjustment;

    // Starvation boost
    uint frames_since_run = current_frame - proc->last_run_frame;
    if (frames_since_run > 5) base_priority++;
    if (frames_since_run > 10) base_priority++;

    // Clamp to valid range
    return uint(clamp(base_priority, 0, int(PRIORITY_REALTIME)));
}
```

### Priority Levels

| Level | Name | Use Case | Frame Budget |
|-------|------|----------|--------------|
| 0 | IDLE | Screen savers, background indexing | 5% |
| 1 | BACKGROUND | File sync, updates | 15% |
| 2 | NORMAL | User applications | 60% |
| 3 | HIGH | Focused window, audio | 85% |
| 4 | REALTIME | Input handling, compositor | 100% |

## Inter-Process Communication (IPC)

### Shared Memory Regions

```metal
// Create shared memory region between processes
struct SharedMemoryRegion {
    uint offset;          // Offset in IPC buffer
    uint size;            // Size in bytes
    uint owner_pid;       // Creating process
    uint ref_count;       // Number of processes with access
    uint permissions;     // Read/Write flags per process
};

// GPU-resident shared memory table
struct SharedMemoryTable {
    SharedMemoryRegion regions[MAX_SHARED_REGIONS];
    atomic_uint region_count;
    atomic_uint free_list_head;
};

// Map shared region into process address space
uint gpu_shm_attach(
    device SharedMemoryTable* shm_table,
    device GpuProcess* proc,
    uint region_id
) {
    if (region_id >= MAX_SHARED_REGIONS) return INVALID_OFFSET;

    device SharedMemoryRegion* region = &shm_table->regions[region_id];

    // Check permissions
    if (!has_access(region, proc->pid)) return INVALID_OFFSET;

    // Increment ref count
    atomic_fetch_add_explicit(&region->ref_count, 1, memory_order_relaxed);

    // Return offset (process accesses via IPC buffer base + offset)
    return region->offset;
}
```

### Message Queues

```metal
// Lock-free SPSC ring buffer for process-to-process messages
struct ProcessMessageQueue {
    uint buffer_offset;      // Offset in IPC buffer
    uint buffer_size;        // Power of 2
    atomic_uint head;        // Write position (producer)
    atomic_uint tail;        // Read position (consumer)
    uint sender_pid;
    uint receiver_pid;
};

// Send message (non-blocking)
bool gpu_mq_send(
    device ProcessMessageQueue* mq,
    device uint8_t* ipc_buffer,
    device const uint8_t* data,
    uint size
) {
    uint head = atomic_load_explicit(&mq->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&mq->tail, memory_order_acquire);

    uint available = mq->buffer_size - (head - tail);
    if (size + 4 > available) {
        return false;  // Queue full
    }

    device uint8_t* buf = ipc_buffer + mq->buffer_offset;
    uint mask = mq->buffer_size - 1;

    // Write size header
    uint write_pos = head & mask;
    *(device uint*)(buf + write_pos) = size;
    write_pos = (write_pos + 4) & mask;

    // Write data
    for (uint i = 0; i < size; i++) {
        buf[(write_pos + i) & mask] = data[i];
    }

    // Commit write (release semantics)
    atomic_store_explicit(&mq->head, head + 4 + size, memory_order_release);

    return true;
}

// Receive message (non-blocking)
uint gpu_mq_recv(
    device ProcessMessageQueue* mq,
    device uint8_t* ipc_buffer,
    device uint8_t* data,
    uint max_size
) {
    uint tail = atomic_load_explicit(&mq->tail, memory_order_relaxed);
    uint head = atomic_load_explicit(&mq->head, memory_order_acquire);

    if (head == tail) {
        return 0;  // Queue empty
    }

    device uint8_t* buf = ipc_buffer + mq->buffer_offset;
    uint mask = mq->buffer_size - 1;

    // Read size header
    uint read_pos = tail & mask;
    uint size = *(device uint*)(buf + read_pos);
    read_pos = (read_pos + 4) & mask;

    if (size > max_size) {
        return 0;  // Message too large
    }

    // Read data
    for (uint i = 0; i < size; i++) {
        data[i] = buf[(read_pos + i) & mask];
    }

    // Commit read (release semantics)
    atomic_store_explicit(&mq->tail, tail + 4 + size, memory_order_release);

    return size;
}
```

### Pipes

```metal
// Pipe: unidirectional byte stream between processes
struct GpuPipe {
    uint buffer_offset;      // Offset in IPC buffer
    uint buffer_size;        // Ring buffer size
    atomic_uint head;        // Write position
    atomic_uint tail;        // Read position
    uint read_pid;           // Process that can read
    uint write_pid;          // Process that can write
    atomic_uint flags;       // PIPE_CLOSED_READ, PIPE_CLOSED_WRITE
};

// Write to pipe
uint gpu_pipe_write(
    device GpuPipe* pipe,
    device uint8_t* ipc_buffer,
    device const uint8_t* data,
    uint count,
    uint writer_pid
) {
    if (writer_pid != pipe->write_pid) return 0;
    if (atomic_load(&pipe->flags) & PIPE_CLOSED_READ) return 0;  // SIGPIPE

    device uint8_t* buf = ipc_buffer + pipe->buffer_offset;
    uint mask = pipe->buffer_size - 1;
    uint written = 0;

    while (written < count) {
        uint head = atomic_load_explicit(&pipe->head, memory_order_relaxed);
        uint tail = atomic_load_explicit(&pipe->tail, memory_order_acquire);
        uint available = pipe->buffer_size - (head - tail) - 1;

        if (available == 0) {
            // Buffer full - would block (we return partial write)
            break;
        }

        uint to_write = min(count - written, available);
        for (uint i = 0; i < to_write; i++) {
            buf[(head + i) & mask] = data[written + i];
        }

        atomic_store_explicit(&pipe->head, head + to_write, memory_order_release);
        written += to_write;
    }

    return written;
}
```

### Signals

```metal
// Signal definitions (POSIX-like)
#define SIGKILL  9   // Terminate immediately
#define SIGSTOP 19   // Stop process
#define SIGCONT 18   // Continue if stopped
#define SIGCHLD 17   // Child process terminated
#define SIGUSR1 10   // User-defined signal 1
#define SIGUSR2 12   // User-defined signal 2

// Send signal to process
void gpu_kill(
    device ProcessTable* table,
    uint target_pid,
    uint signal
) {
    uint slot = find_slot_by_pid(table, target_pid);
    if (slot == INVALID_SLOT) return;

    device GpuProcess* proc = &table->processes[slot];

    // Special handling for SIGKILL and SIGSTOP
    if (signal == SIGKILL) {
        proc->exit_code = -SIGKILL;
        atomic_store_explicit(&proc->status, ProcessStatus_ZOMBIE, memory_order_release);
        return;
    }

    if (signal == SIGSTOP) {
        atomic_store_explicit(&proc->status, ProcessStatus_STOPPED, memory_order_release);
        return;
    }

    if (signal == SIGCONT) {
        uint status = atomic_load(&proc->status);
        if (status == ProcessStatus_STOPPED) {
            atomic_store_explicit(&proc->status, ProcessStatus_READY, memory_order_release);
        }
        return;
    }

    // Queue signal for delivery
    atomic_fetch_or_explicit(&proc->pending_signals, 1u << signal, memory_order_relaxed);
}

// Check and deliver pending signals (called at process start of frame)
void deliver_pending_signals(
    device GpuProcess* proc,
    device uint8_t* state_buffer
) {
    uint pending = atomic_exchange_explicit(&proc->pending_signals, 0, memory_order_acquire);
    if (pending == 0) return;

    // Check each signal bit
    for (uint sig = 1; sig < 32; sig++) {
        if (pending & (1u << sig)) {
            // Look up handler
            device SignalHandler* handlers = (device SignalHandler*)(
                state_buffer + proc->signal_handlers_offset
            );

            if (handlers[sig].action == SIG_DFL) {
                // Default action (depends on signal)
                handle_default_signal(proc, sig);
            } else if (handlers[sig].action == SIG_IGN) {
                // Ignore
            } else {
                // Custom handler - queue handler invocation
                queue_signal_handler(proc, sig, handlers[sig].handler_offset);
            }
        }
    }
}
```

## Binary Registry

Pre-compiled GPU bytecode for system utilities and user programs:

```metal
// Binary header in registry
struct BinaryHeader {
    char name[32];           // Null-terminated program name
    uint name_hash;          // For O(1) lookup
    uint bytecode_size;      // Size of bytecode section
    uint entry_point;        // Instruction offset for main()
    uint default_heap_size;  // Recommended heap size
    uint default_stack_size; // Recommended stack size
    uint max_threads;        // Maximum threads this program can use
    uint flags;              // Executable flags
};

// Binary registry structure
struct BinaryRegistry {
    uint entry_count;
    uint hash_table_size;     // Power of 2
    uint hash_table[1024];    // Hash -> entry offset (O(1) lookup)
    // Followed by entries: [BinaryHeader][Bytecode]...
};

// O(1) binary lookup by name hash
uint binary_registry_lookup(
    device BinaryRegistry* registry,
    uint name_hash
) {
    uint mask = registry->hash_table_size - 1;
    uint slot = name_hash & mask;

    // Linear probing for collision resolution
    for (uint i = 0; i < 16; i++) {
        uint offset = registry->hash_table[(slot + i) & mask];
        if (offset == 0) return INVALID_OFFSET;  // Not found

        device BinaryHeader* header = (device BinaryHeader*)(
            (device uint8_t*)registry + offset
        );
        if (header->name_hash == name_hash) {
            return offset;
        }
    }

    return INVALID_OFFSET;  // Not found after probing
}
```

### Standard Binary Set

Pre-compiled GPU-native implementations:

| Binary | Description | Typical Threads |
|--------|-------------|-----------------|
| `init` | First process, spawns shell | 1 |
| `sh` | Shell / command interpreter | 32 |
| `ls` | List directory contents | 1024 (parallel file scan) |
| `cat` | Concatenate and display files | 256 |
| `grep` | Pattern matching | 1024 (parallel search) |
| `find` | Search for files | 2048 (parallel tree walk) |
| `cp` | Copy files | 512 |
| `mv` | Move/rename files | 1 |
| `rm` | Remove files | 256 |
| `mkdir` | Create directories | 1 |
| `terminal` | Terminal emulator | 256 |
| `editor` | Text editor | 512 |
| `browser` | File browser | 1024 |

## std::process Mapping

How Rust's `std::process` concepts map to our GPU-native model:

| std::process | GPU-Native | Notes |
|--------------|------------|-------|
| `Command::new("ls")` | `binary_registry_lookup(hash("ls"))` | O(1) hash lookup |
| `command.arg("-la")` | Write args to process heap at spawn | Args in GPU memory |
| `command.spawn()` | `gpu_spawn(table, parent_pid, binary_hash)` | All on GPU |
| `Child` | `GpuProcess*` (process descriptor pointer) | GPU-resident |
| `child.wait()` | Poll `proc->status` until ZOMBIE | GPU atomic load |
| `child.id()` | `proc->pid` | PID from descriptor |
| `child.kill()` | `gpu_kill(pid, SIGKILL)` | GPU signal delivery |
| `exit(code)` | `OP_EXIT` bytecode instruction | Set status to ZOMBIE |
| `Command::output()` | `gpu_spawn` + capture stdout buffer + `wait` | Pipe for stdout |
| `Command::stdin(Stdio::piped())` | Create `GpuPipe`, attach to child | GPU IPC |

### Example: Running "ls" from Terminal

```rust
// Terminal process wants to run "ls -la /home"

// 1. Build command (in GPU bytecode)
let binary_hash = hash("ls");
let args = ["-la", "/home"];

// 2. Set up pipes for stdout capture
let (read_end, write_end) = gpu_pipe_create();
child_stdin = None;
child_stdout = Some(write_end);

// 3. Spawn child process
let child_pid = gpu_spawn(
    process_table,
    state_buffer,
    binary_registry,
    allocator,
    my_pid,        // Parent
    binary_hash,   // "ls"
    1024,          // Thread count hint
);

// 4. Wait for completion (non-blocking poll in frame loop)
loop {
    if check_process_status(child_pid) == ZOMBIE {
        break;
    }
    yield();  // Let other processes run
}

// 5. Read output from pipe
let output = gpu_pipe_read_all(read_end);

// 6. Reap child (free resources)
wait(child_pid);
```

## Memory Protection

GPU processes don't have true hardware memory protection (no MMU). We use **software bounds checking**:

```metal
// Every memory access goes through bounds check
inline bool check_bounds(
    device GpuProcess* proc,
    uint address,
    uint size,
    uint access_type  // READ or WRITE
) {
    // Check if address is in process heap
    if (address >= proc->heap_offset &&
        address + size <= proc->heap_offset + proc->heap_size) {
        return true;
    }

    // Check if address is in process stack
    if (address >= proc->stack_offset &&
        address + size <= proc->stack_offset + proc->stack_size) {
        return true;
    }

    // Check shared memory regions
    if (check_shared_memory_access(proc, address, size, access_type)) {
        return true;
    }

    // Invalid access - generate SIGSEGV
    queue_signal(proc, SIGSEGV);
    return false;
}

// Bounds-checked memory load
inline float4 safe_load(
    device GpuProcess* proc,
    device uint8_t* state_buffer,
    uint address
) {
    if (!check_bounds(proc, address, 16, ACCESS_READ)) {
        return float4(0);  // Return zero on fault
    }
    return *(device float4*)(state_buffer + address);
}
```

## Limitations and Trade-offs

| Limitation | Reason | Mitigation |
|------------|--------|------------|
| Cannot run x86/ARM binaries | Must be GPU bytecode | WASM-to-GPU compiler pipeline |
| No true memory protection | No GPU MMU | Software bounds checking |
| Limited process count (64) | Threadgroup limits | Sufficient for desktop workloads |
| No preemption within frame | GPU execution model | Cooperative yielding + frame timeout |
| Single-GPU only | Architecture assumption | Future: multi-GPU support |

## Benefits

1. **Zero CPU involvement in process management**
   - No syscalls, no kernel context switches
   - All spawn/exit/signal/wait on GPU

2. **True parallelism**
   - 64 processes run simultaneously
   - No sequential scheduler overhead

3. **Sub-microsecond spawn time**
   - Just atomic bitmap + memory allocation
   - No TLB flush, no page table setup

4. **Native GPU scheduling**
   - Hardware schedules threadgroups
   - No software context switch

5. **Massive thread counts per process**
   - Each process can have 1000+ threads
   - Parallel algorithms out of the box

## Rust API

```rust
/// GPU-native process management API
pub struct GpuProcessSystem {
    table_buffer: Buffer,      // GPU-resident process table
    state_buffer: Buffer,      // Unified state buffer
    binary_registry: Buffer,   // Pre-compiled binaries
    ipc_buffer: Buffer,        // IPC memory region
}

impl GpuProcessSystem {
    /// Create a new process system
    pub fn new(device: &Device) -> Result<Self, ProcessError>;

    /// Register a binary in the registry
    pub fn register_binary(&mut self, name: &str, bytecode: &[u8], config: BinaryConfig);

    /// Get process by PID (for debugging/monitoring)
    pub fn get_process(&self, pid: u32) -> Option<ProcessInfo>;

    /// List all processes
    pub fn list_processes(&self) -> Vec<ProcessInfo>;

    /// Send signal to process
    pub fn kill(&self, pid: u32, signal: Signal) -> Result<(), ProcessError>;

    /// Get process statistics
    pub fn stats(&self) -> ProcessStats;
}

/// Process information (read from GPU)
pub struct ProcessInfo {
    pub pid: u32,
    pub parent_pid: u32,
    pub status: ProcessStatus,
    pub priority: u32,
    pub thread_count: u32,
    pub heap_usage: u32,
    pub cpu_time: u32,
}

/// Process statistics
pub struct ProcessStats {
    pub total_processes: u32,
    pub running: u32,
    pub sleeping: u32,
    pub zombie: u32,
    pub memory_usage: u64,
}
```

## Implementation Plan

### Phase 1: Process Table and Spawn
1. Implement `GpuProcess` structure with proper alignment
2. Create process table with atomic bitmap allocator
3. Implement `gpu_spawn()` with memory allocation
4. Test spawning processes from init

### Phase 2: Process Execution
1. Integrate with bytecode VM
2. Implement process megakernel dispatch
3. Add process-specific register initialization
4. Test single process execution

### Phase 3: Process Exit and Cleanup
1. Implement `OP_EXIT` instruction
2. Add zombie state handling
3. Implement process reaper
4. Test full lifecycle

### Phase 4: IPC
1. Implement message queues
2. Implement pipes
3. Implement shared memory
4. Test inter-process communication

### Phase 5: Signals
1. Implement signal delivery
2. Add signal handlers
3. Implement SIGKILL, SIGSTOP, SIGCONT
4. Test signal handling

### Phase 6: Integration
1. Port shell to use process spawning
2. Implement standard utilities (ls, cat, grep)
3. Add process monitoring/debugging
4. Performance optimization

## Success Metrics

| Metric | Target |
|--------|--------|
| Process spawn time | < 100 microseconds |
| Process count | 64 simultaneous |
| Memory overhead per process | < 16 KB |
| Context switch time | 0 (true parallelism) |
| CPU utilization | < 1% for process management |
| IPC throughput | > 1 GB/s |

## References

- PRD: GPU Bytecode Virtual Machine
- PRD: GPU-Centric App System
- PRD: GPU Threading Model
- PRD: GPU Multi-App Scheduler
- PRD: Phase 6 - GPU Allocator
