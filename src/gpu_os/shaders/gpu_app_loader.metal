//! GPU App Loader - Metal Kernels
//!
//! Issue #168 - GPU App Loader
//!
//! THE GPU IS THE COMPUTER.
//! GPU parses, validates, allocates, and initializes apps.
//! CPU just moves bytes via I/O coprocessor.

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

constant uint GPUAPP_HEADER_SIZE = 64;
constant uint GPUAPP_VERSION = 1;

constant uint MAX_BYTECODE_SIZE = 65536;
constant uint MAX_APP_STATE = 1024 * 1024;
constant uint MAX_APP_VERTICES = 65536;

constant uint INVALID_SLOT = 0xFFFFFFFF;

// App flags
constant uint APP_FLAG_ACTIVE = 1;
constant uint APP_FLAG_VISIBLE = 2;
constant uint APP_FLAG_DIRTY = 4;

// App types
constant uint APP_TYPE_BYTECODE = 101;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// GpuAppFileHeader (64 bytes) - matches Rust struct
struct GpuAppFileHeader {
    uchar magic[6];           // "GPUAPP"
    ushort version;           // 1
    uint flags;
    uint code_offset;         // Offset to bytecode from file start
    uint code_size;           // Number of instructions
    uint state_size;          // Bytes needed for app state
    uint vertex_budget;       // Max vertices
    uint thread_count;        // Recommended thread count
    uint entry_point;         // Starting PC
    uchar name[24];           // Null-terminated
    uchar _reserved[4];
};

// BytecodeInst (8 bytes)
struct BytecodeInst {
    uchar opcode;
    uchar dst;
    uchar src1;
    uchar src2;
    float imm;
};

// BytecodeHeader (16 bytes) - at start of app state
struct BytecodeHeader {
    uint code_size;
    uint entry_point;
    uint vertex_budget;
    uint flags;
};

// App table header
struct AppTableHeader {
    uint magic;
    uint version;
    uint max_slots;
    atomic_uint active_count;  // Issue #259 fix: must be atomic for safe concurrent access
    atomic_uint next_slot_hint;
    uint _pad[3];
    // Followed by free_bitmap and app descriptors
};

// App descriptor (128 bytes) - matches Rust GpuAppDescriptor
struct GpuAppDescriptor {
    // Identity & Lifecycle (16 bytes)
    uint flags;
    uint app_type;
    uint slot_id;
    uint window_id;

    // Memory (16 bytes)
    uint state_offset;
    uint state_size;
    uint vertex_offset;
    uint vertex_size;

    // Render state (16 bytes)
    uint vertex_count;
    uint vertex_budget;
    uint depth;
    uint render_order;

    // Timing (16 bytes)
    uint last_run_frame;
    uint frames_alive;
    uint priority;
    uint cpu_hint;

    // Layout (16 bytes)
    float x, y, width, height;

    // Visual (16 bytes)
    float bg_r, bg_g, bg_b, bg_a;

    // Animation (16 bytes)
    float anim_progress;
    float anim_speed;
    uint anim_state;
    uint _pad1;

    // Debug (16 bytes)
    uint debug_flags;
    uint error_code;
    uint _pad2[2];
};

// Contiguous vertex layout
constant uint VERTS_PER_SLOT = 1024;
constant uint VERTEX_SIZE = 48;  // sizeof(RenderVertex)

// ═══════════════════════════════════════════════════════════════════════════════
// HEADER VALIDATION (O(1))
// ═══════════════════════════════════════════════════════════════════════════════

// Issue #272 fix: Validate .gpuapp header with file bounds checking
// O(1), no loops, SIMD-friendly
inline bool validate_header(device const GpuAppFileHeader* header, uint file_size) {
    // Check magic bytes (unrolled, no loop = no SIMD divergence)
    if (header->magic[0] != 'G') return false;
    if (header->magic[1] != 'P') return false;
    if (header->magic[2] != 'U') return false;
    if (header->magic[3] != 'A') return false;
    if (header->magic[4] != 'P') return false;
    if (header->magic[5] != 'P') return false;

    // Check version
    if (header->version != GPUAPP_VERSION) return false;

    // Check sizes are reasonable
    if (header->code_size > MAX_BYTECODE_SIZE) return false;
    if (header->state_size > MAX_APP_STATE) return false;
    if (header->vertex_budget > MAX_APP_VERTICES) return false;

    // Check code_offset is valid
    if (header->code_offset < GPUAPP_HEADER_SIZE) return false;

    // Issue #272: Bounds check - ensure code section fits within file
    uint bytecode_size = header->code_size * sizeof(BytecodeInst);
    uint code_end = header->code_offset + bytecode_size;

    // Check for overflow
    if (code_end < header->code_offset) return false;  // Overflow

    // Check code doesn't extend past file end
    if (code_end > file_size) return false;

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SLOT ALLOCATION (Atomic)
// ═══════════════════════════════════════════════════════════════════════════════

// Allocate an app slot using atomic bitmap
inline uint allocate_slot(
    device AppTableHeader* header,
    device atomic_uint* free_bitmap,
    uint max_slots
) {
    // Try each bitmap word
    uint words = (max_slots + 31) / 32;

    // Max retries per word to prevent SIMD divergence under high contention
    constant uint MAX_WORD_RETRIES = 64;

    for (uint w = 0; w < words; w++) {
        uint bits = atomic_load_explicit(&free_bitmap[w], memory_order_relaxed);

        // Bounded loop to prevent indefinite spinning
        for (uint retry = 0; retry < MAX_WORD_RETRIES && bits != 0; retry++) {
            // Find first set bit (free slot)
            uint bit = ctz(bits);  // Count trailing zeros
            uint slot = w * 32 + bit;

            if (slot >= max_slots) break;

            // Try to claim it
            uint old = atomic_fetch_and_explicit(&free_bitmap[w], ~(1u << bit), memory_order_relaxed);

            if (old & (1u << bit)) {
                // We got it!
                return slot;
            }

            // Someone else got it, refresh bits and try again
            bits = atomic_load_explicit(&free_bitmap[w], memory_order_relaxed);
        }
    }

    return INVALID_SLOT;  // No free slots
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNELS
// ═══════════════════════════════════════════════════════════════════════════════

// Kernel: Validate a .gpuapp header
// Returns 1 if valid, 0 if invalid
// Issue #272: Now requires file_size for bounds checking
kernel void validate_gpuapp_header(
    device const uchar* header_data [[buffer(0)]],
    constant uint& file_size [[buffer(1)]],
    device uint* result [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Issue #272: Ensure we have at least a header
    if (file_size < GPUAPP_HEADER_SIZE) {
        *result = 0;
        return;
    }

    device const GpuAppFileHeader* header = (device const GpuAppFileHeader*)header_data;
    *result = validate_header(header, file_size) ? 1 : 0;
}

// Kernel: Initialize an app from loaded .gpuapp bytes
// Thread 0 validates and allocates, all threads copy bytecode in parallel
kernel void init_app_from_gpuapp(
    device const uchar* file_data [[buffer(0)]],
    constant uint& file_size [[buffer(1)]],
    device uchar* app_table_data [[buffer(2)]],
    device uchar* unified_state [[buffer(3)]],
    device uint* result [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Parse pointers
    device AppTableHeader* header = (device AppTableHeader*)app_table_data;
    device atomic_uint* free_bitmap = (device atomic_uint*)(app_table_data + 32);  // After header
    device GpuAppDescriptor* apps = (device GpuAppDescriptor*)(app_table_data + 32 + 8);  // After bitmap (2 words = 8 bytes)

    device const GpuAppFileHeader* file_header = (device const GpuAppFileHeader*)file_data;

    // Shared result slot
    threadgroup uint allocated_slot;
    threadgroup bool valid;

    // Initialize shared variables (all threads)
    if (tid == 0) {
        valid = false;
        allocated_slot = INVALID_SLOT;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 validates and allocates
    if (tid == 0) {

        // Validate header (Issue #272: pass file_size for bounds checking)
        if (file_size >= GPUAPP_HEADER_SIZE && validate_header(file_header, file_size)) {
            valid = true;

            // Allocate slot
            allocated_slot = allocate_slot(header, free_bitmap, header->max_slots);

            if (allocated_slot != INVALID_SLOT) {
                // Initialize descriptor basics
                device GpuAppDescriptor* app = &apps[allocated_slot];
                app->slot_id = allocated_slot;
                app->app_type = APP_TYPE_BYTECODE;
                app->vertex_budget = file_header->vertex_budget;
                app->vertex_count = 0;
                app->depth = 10;  // Default depth
                app->priority = 5;  // Default priority

                // Calculate state layout
                uint bytecode_size = file_header->code_size * sizeof(BytecodeInst);
                uint total_state = sizeof(BytecodeHeader) + bytecode_size + file_header->state_size;

                // Use contiguous allocation: slot * state_chunk_size
                uint state_chunk_size = 64 * 1024;  // 64KB per slot
                app->state_offset = allocated_slot * state_chunk_size;
                app->state_size = total_state;

                // Vertex allocation: contiguous
                app->vertex_offset = allocated_slot * VERTS_PER_SLOT * VERTEX_SIZE;
                app->vertex_size = file_header->vertex_budget * VERTEX_SIZE;

                // Write BytecodeHeader at start of state
                device BytecodeHeader* bc_header = (device BytecodeHeader*)(unified_state + app->state_offset);
                bc_header->code_size = file_header->code_size;
                bc_header->entry_point = file_header->entry_point;
                bc_header->vertex_budget = file_header->vertex_budget;
                bc_header->flags = 0;
            }
        }

        *result = allocated_slot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads participate in bytecode copy
    if (!valid || allocated_slot == INVALID_SLOT) return;

    device GpuAppDescriptor* app = &apps[allocated_slot];

    // Source: bytecode in file (after header)
    device const uchar* src = file_data + file_header->code_offset;
    uint copy_size = file_header->code_size * sizeof(BytecodeInst);

    // Destination: after BytecodeHeader in state buffer
    device uchar* dst = unified_state + app->state_offset + sizeof(BytecodeHeader);

    // Parallel copy: each thread copies a chunk
    uint bytes_per_thread = (copy_size + tg_size - 1) / tg_size;
    uint my_start = tid * bytes_per_thread;
    uint my_end = min(my_start + bytes_per_thread, copy_size);

    for (uint i = my_start; i < my_end; i++) {
        dst[i] = src[i];
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0 finalizes (mark app as active)
    if (tid == 0) {
        app->flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE | APP_FLAG_DIRTY;
        atomic_fetch_add_explicit((device atomic_uint*)&header->active_count, 1, memory_order_relaxed);
    }
}

// Kernel: Poll pending loads and initialize apps
// Called every frame by megakernel to check for completed I/O
kernel void poll_pending_loads(
    device uint* pending_handles [[buffer(0)]],
    device atomic_uint* pending_count [[buffer(1)]],
    device uint* io_handle_status [[buffer(2)]],  // From content pipeline
    device uchar* content_pool [[buffer(3)]],
    device uint* io_handle_offsets [[buffer(4)]],
    device uint* io_handle_sizes [[buffer(5)]],
    device uchar* app_table_data [[buffer(6)]],
    device uchar* unified_state [[buffer(7)]],
    device uint* results [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    // Only thread 0 checks handles (could parallelize with more pending)
    if (tid != 0) return;

    uint count = atomic_load_explicit(pending_count, memory_order_relaxed);
    if (count == 0) return;

    // Status constants (from content_pipeline) - use #define since constant isn't allowed in functions
    #define POLL_STATUS_READY 2
    #define POLL_STATUS_ERROR 3

    device AppTableHeader* header = (device AppTableHeader*)app_table_data;
    device atomic_uint* free_bitmap = (device atomic_uint*)(app_table_data + 32);
    device GpuAppDescriptor* apps = (device GpuAppDescriptor*)(app_table_data + 32 + 8);

    uint loaded = 0;
    uint errors = 0;

    // Issue #263 fix: Iterate backwards to safely remove while iterating
    // When we remove an element by swapping with last, backward iteration
    // ensures we don't skip or double-process any elements
    for (int i = int(count) - 1; i >= 0; i--) {
        uint handle = pending_handles[i];
        if (handle == INVALID_SLOT) continue;

        uint status = io_handle_status[handle];
        bool should_remove = false;

        if (status == POLL_STATUS_READY) {
            should_remove = true;

            // Get file data from content pool
            uint offset = io_handle_offsets[handle];
            uint size = io_handle_sizes[handle];
            device uchar* file_data = content_pool + offset;

            // Validate header (Issue #272: pass size for bounds checking)
            device const GpuAppFileHeader* file_header = (device const GpuAppFileHeader*)file_data;

            if (size >= GPUAPP_HEADER_SIZE && validate_header(file_header, size)) {
                // Allocate and initialize app
                uint slot = allocate_slot(header, free_bitmap, header->max_slots);

                if (slot != INVALID_SLOT) {
                    device GpuAppDescriptor* app = &apps[slot];

                    // Initialize descriptor
                    app->slot_id = slot;
                    app->app_type = APP_TYPE_BYTECODE;
                    app->vertex_budget = file_header->vertex_budget;
                    app->vertex_count = 0;
                    app->depth = 10;
                    app->priority = 5;

                    // State layout
                    uint state_chunk_size = 64 * 1024;
                    app->state_offset = slot * state_chunk_size;
                    app->state_size = sizeof(BytecodeHeader) + file_header->code_size * sizeof(BytecodeInst) + file_header->state_size;

                    // Vertex allocation
                    app->vertex_offset = slot * VERTS_PER_SLOT * VERTEX_SIZE;
                    app->vertex_size = file_header->vertex_budget * VERTEX_SIZE;

                    // Write BytecodeHeader
                    device BytecodeHeader* bc_header = (device BytecodeHeader*)(unified_state + app->state_offset);
                    bc_header->code_size = file_header->code_size;
                    bc_header->entry_point = file_header->entry_point;
                    bc_header->vertex_budget = file_header->vertex_budget;
                    bc_header->flags = 0;

                    // Copy bytecode (single-threaded here, could dispatch separate kernel)
                    device const uchar* src = file_data + file_header->code_offset;
                    device uchar* dst = unified_state + app->state_offset + sizeof(BytecodeHeader);
                    uint copy_size = file_header->code_size * sizeof(BytecodeInst);

                    for (uint j = 0; j < copy_size; j++) {
                        dst[j] = src[j];
                    }

                    // Activate app
                    app->flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE | APP_FLAG_DIRTY;
                    atomic_fetch_add_explicit((device atomic_uint*)&header->active_count, 1, memory_order_relaxed);

                    loaded++;
                } else {
                    errors++;  // No slots available
                }
            } else {
                errors++;  // Invalid header
            }
        }
        else if (status == POLL_STATUS_ERROR) {
            // I/O error
            should_remove = true;
            errors++;
        }
        // STATUS_LOADING: keep waiting (don't remove)

        // Remove from pending by swapping with last element
        if (should_remove) {
            pending_handles[i] = pending_handles[count - 1];
            pending_handles[count - 1] = INVALID_SLOT;
            atomic_fetch_sub_explicit(pending_count, 1, memory_order_relaxed);
            count--;
        }
    }

    // Write stats
    results[0] = loaded;
    results[1] = errors;
}
