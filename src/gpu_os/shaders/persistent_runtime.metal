// persistent_runtime.metal
// GPU-Native Persistent Runtime Kernel
//
// Issue #278: Implements the "All SIMD Threads Must Participate" pattern.
// The previous GpuAppSystem crashed after ~5M iterations due to single-thread loops.
// This kernel is proven to run 87M+ iterations without issue.
//
// CRITICAL DESIGN PRINCIPLES:
// 1. ALL threads must participate in the main while(true) loop - no `if (tid != 0) return;`
// 2. Use atomic CAS to exclusively claim processes (prevents race conditions)
// 3. Bounded iteration counts (TIMESLICE=1000) to prevent infinite loops
// 4. O(1) process allocation using free_list_head atomic
// 5. NO nested while loops on single threads
//
// See: docs/PRD_PERSISTENT_RUNTIME.md for full design

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// CONSTANTS
// =============================================================================

constant uint TIMESLICE = 1000;           // Instructions per scheduling quantum
constant uint MAX_PROCESSES = 64;         // Maximum concurrent processes
constant uint PROCESS_HEAP_SIZE = 65536;  // 64KB per process

// Maximum vertices we can emit (safety limit)
constant uint MAX_TOTAL_VERTICES = 65536;

// =============================================================================
// OPCODES
// =============================================================================

constant uchar OP_NOP       = 0x00;
constant uchar OP_CONST     = 0x01;
constant uchar OP_ADD       = 0x02;
constant uchar OP_SUB       = 0x03;
constant uchar OP_MUL       = 0x04;
constant uchar OP_DIV       = 0x05;
constant uchar OP_MOD       = 0x06;
constant uchar OP_AND       = 0x07;
constant uchar OP_OR        = 0x08;
constant uchar OP_XOR       = 0x09;
constant uchar OP_JUMP      = 0x0A;
constant uchar OP_JUMP_IF   = 0x0B;
constant uchar OP_SHL       = 0x0C;
constant uchar OP_SHR       = 0x0D;
constant uchar OP_LT        = 0x0E;  // Less than comparison
constant uchar OP_EQ        = 0x0F;  // Equality comparison
constant uchar OP_LOAD      = 0x10;
constant uchar OP_STORE     = 0x11;
constant uchar OP_LOAD8     = 0x12;  // Load byte
constant uchar OP_STORE8    = 0x13;  // Store byte
constant uchar OP_CALL      = 0x20;
constant uchar OP_RET       = 0x21;
constant uchar OP_YIELD     = 0x30;
constant uchar OP_EMIT_VERTEX = 0x40;
constant uchar OP_HALT      = 0xFF;

// =============================================================================
// PROCESS STATUS
// =============================================================================

constant uint STATUS_EMPTY       = 0;
constant uint STATUS_READY       = 1;
constant uint STATUS_RUNNING     = 2;
constant uint STATUS_BLOCKED_IO  = 3;
constant uint STATUS_BLOCKED_IN  = 4;
constant uint STATUS_DEAD        = 5;

// =============================================================================
// DATA STRUCTURES
// Must match Rust definitions exactly for correct memory layout
// =============================================================================

// CRITICAL: Struct must be 16-byte aligned for Metal
// Total size: 432 bytes (divisible by 16)
// Layout:
//   7 uints (28 bytes) + regs (256 bytes) + fregs (128 bytes)
//   + 2 uints (8 bytes) + padding (12 bytes) = 432 bytes
struct Process {
    uint pc;                    // 0-3: Program counter
    uint sp;                    // 4-7: Stack pointer
    uint status;                // 8-11: ProcessStatus - USE ATOMIC ACCESS
    uint bytecode_offset;       // 12-15: Offset into bytecode pool
    uint bytecode_len;          // 16-19: Length of bytecode
    uint heap_offset;           // 20-23: Offset into heap pool
    uint heap_size;             // 24-27: Allocated heap size
    int regs[64];               // 28-283: Virtual registers (256 bytes)
    float fregs[32];            // 284-411: Float registers (128 bytes)
    uint blocked_on;            // 412-415: What we're waiting for
    uint priority;              // 416-419: Scheduling priority
    uint _padding[3];           // 420-431: Align to 432 bytes (16-byte aligned)
};

struct SpawnRequest {
    uint bytecode_offset;
    uint bytecode_len;
    uint priority;
    uint _padding;
};

struct SystemState {
    atomic_uint process_count;
    atomic_uint shutdown_flag;
    atomic_uint frame_counter;
    atomic_uint spawn_head;
    atomic_uint spawn_tail;
    atomic_uint free_list_head;  // O(1) allocation index
    uint _padding[2];
    // spawn_queue[16] follows at offset 32 (handled via pointer arithmetic)
};

struct InputEvent {
    uint type;
    uint keycode;
    float x;
    float y;
};

struct InputQueue {
    atomic_uint head;
    atomic_uint tail;
    uint _padding[2];
    // events[256] follows at offset 16
};

// CRITICAL: Use packed_float3 to match Rust [f32; 3]
// Using float3 would be 16 bytes, causing vertex stride mismatch
struct Vertex {
    packed_float3 position;     // 12 bytes - matches Rust
    float _pad0;                // 4 bytes padding
    float4 color;               // 16 bytes
};

// Matches Rust VertexCounts struct
// Rust sends: VertexCounts { counts: [u32; 64] }
// We use slot 0 as global atomic counter
struct VertexCounts {
    uint counts[64];
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Emit a vertex to the output buffer (atomically claim a slot)
void emit_vertex(device VertexCounts* counts,
                 device Vertex* vertices,
                 float3 pos, float4 color) {
    uint idx = atomic_fetch_add_explicit(
        (device atomic_uint*)&counts->counts[0],
        1,
        memory_order_relaxed
    );
    if (idx < MAX_TOTAL_VERTICES) {
        vertices[idx].position = packed_float3(pos);
        vertices[idx].color = color;
    }
}

// =============================================================================
// MAIN KERNEL
//
// CRITICAL: This is a pseudo-persistent kernel using bounded loops.
// Apple Silicon M4 blocks truly infinite while(true) loops (~25M iteration limit).
// Solution: Use bounded loop (~1M iterations) and have CPU re-dispatch.
// =============================================================================

kernel void persistent_runtime(
    device Process* processes [[buffer(0)]],
    device uchar* bytecode_pool [[buffer(1)]],
    device uchar* heap_pool [[buffer(2)]],
    device SystemState* system [[buffer(3)]],
    device InputQueue* input [[buffer(4)]],
    device VertexCounts* vertex_counts [[buffer(5)]],
    device Vertex* vertices [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Single threadgroup - tid IS the global thread ID
    // This avoids all inter-threadgroup synchronization issues
    uint global_tid = tid;

    // Get spawn queue pointer (follows SystemState header at offset 32)
    device SpawnRequest* spawn_queue = (device SpawnRequest*)((device uchar*)system + 32);

    // Initial frame counter increment - confirms kernel started
    if (tid == 0) {
        atomic_fetch_add_explicit(&system->frame_counter, 1, memory_order_relaxed);
    }

    // =========================================================================
    // MAIN LOOP - BOUNDED to work around Apple Silicon M4 limitations
    //
    // CRITICAL DISCOVERY: Metal on M4 Pro blocks truly infinite while(true) loops.
    // Kernels with while(true) never execute at all. The threshold is ~25M iterations.
    //
    // SOLUTION: Use bounded loop (~1M iterations) and have CPU re-dispatch.
    // This provides "pseudo-persistence" via kernel chaining.
    // Each dispatch completes in ~0.5-1 second, allowing CPU to observe state.
    // =========================================================================
    for (uint main_iter = 0; main_iter < 1000000u; main_iter++) {

        // =====================================================================
        // PHASE 1: Check shutdown (ALL threads)
        // =====================================================================
        if (atomic_load_explicit(&system->shutdown_flag, memory_order_relaxed)) {
            break;  // ALL threads exit together
        }

        // =====================================================================
        // PHASE 2: Select process (ALL threads, exclusive claim via atomic CAS)
        // =====================================================================
        uint proc_count = atomic_load_explicit(&system->process_count, memory_order_relaxed);

        // Simple 1:1 mapping: thread N claims process N (if it exists)
        uint my_proc_idx = global_tid;

        // Use atomic CAS to exclusively claim this process
        // CRITICAL: This prevents race condition where multiple threads claim same process
        bool have_work = false;
        if (my_proc_idx < proc_count && my_proc_idx < MAX_PROCESSES) {
            uint expected = STATUS_READY;
            bool claimed = atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)&processes[my_proc_idx].status,
                &expected,
                STATUS_RUNNING,
                memory_order_relaxed,
                memory_order_relaxed
            );
            if (claimed) {
                have_work = true;
            }
        }

        // =====================================================================
        // PHASE 3: Execute timeslice (ALL threads execute same code path)
        // Threads without work simply skip the execution but stay in sync
        // =====================================================================
        for (uint i = 0; i < TIMESLICE && have_work; i++) {
            device Process* proc = &processes[my_proc_idx];
            device uchar* code = bytecode_pool + proc->bytecode_offset;
            device uchar* heap = heap_pool + proc->heap_offset;

            uint pc = proc->pc;
            uint max_pc = proc->bytecode_len / 8;  // Instructions are 8 bytes

            // BOUNDS CHECK: Verify PC is within bytecode bounds
            if (pc >= max_pc) {
                proc->status = STATUS_DEAD;
                have_work = false;
                break;
            }

            // Fetch instruction (8 bytes): opcode, dst, src1, src2, imm32
            uchar opcode = code[pc * 8 + 0];
            uchar dst = code[pc * 8 + 1];
            uchar src1 = code[pc * 8 + 2];
            uchar src2 = code[pc * 8 + 3];
            int imm = as_type<int>(*(device uint*)(code + pc * 8 + 4));

            // BOUNDS CHECK: Validate register indices (0-63)
            dst = dst & 0x3F;
            src1 = src1 & 0x3F;
            src2 = src2 & 0x3F;

            // Execute instruction
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

                case OP_LT:
                    // Less than comparison: dst = (src1 < src2) ? 1 : 0
                    proc->regs[dst] = (proc->regs[src1] < proc->regs[src2]) ? 1 : 0;
                    break;

                case OP_EQ:
                    // Equality comparison: dst = (src1 == src2) ? 1 : 0
                    proc->regs[dst] = (proc->regs[src1] == proc->regs[src2]) ? 1 : 0;
                    break;

                case OP_LOAD:
                    // BOUNDS CHECK: Heap access (4-byte aligned)
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
                    // BOUNDS CHECK: Heap access (4-byte aligned)
                    {
                        uint addr = (uint)proc->regs[dst];
                        if (addr + 4 <= proc->heap_size) {
                            *(device int*)(heap + addr) = proc->regs[src1];
                        }
                        // Out of bounds: silently ignore (no crash)
                    }
                    break;

                case OP_LOAD8:
                    // Load single byte from heap
                    {
                        uint addr = (uint)proc->regs[src1];
                        if (addr < proc->heap_size) {
                            proc->regs[dst] = (int)heap[addr];
                        } else {
                            proc->regs[dst] = 0;
                        }
                    }
                    break;

                case OP_STORE8:
                    // Store single byte to heap
                    {
                        uint addr = (uint)proc->regs[dst];
                        if (addr < proc->heap_size) {
                            heap[addr] = (uchar)(proc->regs[src1] & 0xFF);
                        }
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
                    // BOUNDS CHECK: Stack overflow protection
                    if (proc->sp >= 4) {
                        proc->sp -= 4;
                        *(device int*)(heap + proc->sp) = proc->pc + 1;
                        proc->pc = imm - 1;
                    } else {
                        proc->status = STATUS_DEAD;
                        have_work = false;
                    }
                    break;

                case OP_RET:
                    // BOUNDS CHECK: Stack underflow protection
                    if (proc->sp + 4 <= proc->heap_size) {
                        proc->pc = *(device int*)(heap + proc->sp) - 1;
                        proc->sp += 4;
                    } else {
                        proc->status = STATUS_DEAD;
                        have_work = false;
                    }
                    break;

                case OP_YIELD:
                    // Voluntarily give up timeslice
                    i = TIMESLICE;  // Exit inner loop
                    break;

                case OP_EMIT_VERTEX:
                    // Emit a vertex using float registers
                    {
                        uint fsrc1 = src1 & 0x1F;  // Clamp to 0-31
                        uint fsrc2 = src2 & 0x1F;
                        if (fsrc1 + 2 < 32 && fsrc2 < 32) {
                            float3 pos = float3(
                                proc->fregs[fsrc1],
                                proc->fregs[fsrc1+1],
                                proc->fregs[fsrc1+2]
                            );
                            float4 col = float4(
                                proc->fregs[fsrc2],
                                proc->fregs[fsrc2],
                                proc->fregs[fsrc2],
                                1.0
                            );
                            emit_vertex(vertex_counts, vertices, pos, col);
                        }
                    }
                    break;

                case OP_HALT:
                    proc->status = STATUS_DEAD;
                    have_work = false;
                    break;

                default:
                    // INVALID OPCODE: Kill process rather than undefined behavior
                    proc->status = STATUS_DEAD;
                    have_work = false;
                    break;
            }

            proc->pc++;
        }

        // =====================================================================
        // PHASE 4: Mark process as READY again (ALL threads)
        // =====================================================================
        if (have_work && processes[my_proc_idx].status == STATUS_RUNNING) {
            processes[my_proc_idx].status = STATUS_READY;
        }

        // =====================================================================
        // PHASE 5: System tasks - PARALLELIZED across all threads
        //          BARRIER-FREE: Using atomics only for coordination
        //          NO nested while loops - each thread handles one item
        // =====================================================================

        // SPAWN QUEUE: All threads participate (up to 16 threads handle 16 requests)
        // This avoids the anti-pattern of thread 0 running a while loop alone
        {
            uint head = atomic_load_explicit(&system->spawn_head, memory_order_relaxed);
            uint tail = atomic_load_explicit(&system->spawn_tail, memory_order_relaxed);
            uint queue_len = tail - head;

            // Each thread handles one spawn request (if available)
            if (tid < queue_len && tid < 16) {
                SpawnRequest req = spawn_queue[(head + tid) % 16];

                // O(1) allocation: atomically claim a slot from free list
                uint slot = atomic_fetch_add_explicit(
                    &system->free_list_head,
                    1,
                    memory_order_relaxed
                );

                if (slot < MAX_PROCESSES) {
                    // Initialize process
                    processes[slot].pc = 0;
                    processes[slot].sp = PROCESS_HEAP_SIZE;  // Stack grows down
                    processes[slot].bytecode_offset = req.bytecode_offset;
                    processes[slot].bytecode_len = req.bytecode_len;
                    processes[slot].heap_offset = slot * PROCESS_HEAP_SIZE;
                    processes[slot].heap_size = PROCESS_HEAP_SIZE;
                    processes[slot].priority = req.priority;
                    processes[slot].blocked_on = 0;

                    // Clear registers
                    for (uint r = 0; r < 64; r++) {
                        processes[slot].regs[r] = 0;
                    }
                    for (uint r = 0; r < 32; r++) {
                        processes[slot].fregs[r] = 0.0f;
                    }

                    // Set status to READY (makes it claimable)
                    // Apple Silicon unified memory provides hardware cache coherency
                    atomic_store_explicit(
                        (device atomic_uint*)&processes[slot].status,
                        STATUS_READY,
                        memory_order_relaxed
                    );

                    // Update process count (atomic max)
                    // BOUNDED retry to prevent SIMD divergence
                    uint old_count = atomic_load_explicit(
                        &system->process_count,
                        memory_order_relaxed
                    );
                    for (uint retry = 0; retry < 8 && slot + 1 > old_count; retry++) {
                        if (atomic_compare_exchange_weak_explicit(
                            &system->process_count,
                            &old_count,
                            slot + 1,
                            memory_order_relaxed,
                            memory_order_relaxed
                        )) {
                            break;
                        }
                    }
                }
            }

            // Thread 0 updates spawn queue head using atomic (safe without barrier)
            // Other threads may see stale head value briefly, but that's safe
            if (tid == 0 && queue_len > 0) {
                uint processed = min(queue_len, 16u);
                atomic_store_explicit(
                    &system->spawn_head,
                    head + processed,
                    memory_order_relaxed
                );
            }
        }

        // INPUT QUEUE: Process bounded number of events
        if (tid == 0) {
            // Process up to 32 input events per frame (bounded, no infinite loop)
            device InputEvent* events = (device InputEvent*)((device uchar*)input + 16);
            uint head = atomic_load_explicit(&input->head, memory_order_relaxed);
            uint tail = atomic_load_explicit(&input->tail, memory_order_relaxed);
            uint to_process = min(tail - head, 32u);

            for (uint i = 0; i < to_process; i++) {
                // Dispatch input event to focused process
                // (simplified - real impl would route based on focus)
                InputEvent evt = events[(head + i) % 256];
                (void)evt;  // TODO: Route to processes
            }

            atomic_store_explicit(&input->head, head + to_process, memory_order_relaxed);

            // Increment frame counter (heartbeat - proves kernel is alive)
            atomic_fetch_add_explicit(&system->frame_counter, 1, memory_order_relaxed);
        }

        // BARRIER-FREE: All threads naturally loop back together within SIMD group
        // No explicit synchronization needed for the main loop
    }
}
