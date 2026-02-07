//! GPU Content Pipeline - Inline Functions for Megakernel
//!
//! Issue #165 - File Content Pipeline
//!
//! These are inline functions that apps call within the persistent megakernel.
//! No kernel dispatch needed - just function calls.
//!
//! Architecture:
//! - GPU writes requests to I/O queue (atomic)
//! - GPU continues working (never waits)
//! - CPU I/O coprocessor reads queue asynchronously
//! - CPU dispatches MTLIOCommandQueue
//! - CPU updates status when I/O complete
//! - GPU polls status buffer

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

constant uint INVALID_HANDLE = 0xFFFFFFFF;
constant uint INVALID_PATH = 0xFFFFFFFF;

// Request types
constant uint IO_READ = 1;
constant uint IO_WRITE = 2;
constant uint IO_CLOSE = 3;

// Priority levels
constant uint IO_PRIORITY_REALTIME = 0;  // User waiting
constant uint IO_PRIORITY_HIGH = 1;      // App foreground
constant uint IO_PRIORITY_NORMAL = 2;    // Background
constant uint IO_PRIORITY_LOW = 3;       // Prefetch

// Handle status
constant uint STATUS_UNUSED = 0;
constant uint STATUS_LOADING = 1;
constant uint STATUS_READY = 2;
constant uint STATUS_ERROR = 3;
constant uint STATUS_CLOSED = 4;

// Handle flags
constant uint HANDLE_FLAG_READABLE = 1;
constant uint HANDLE_FLAG_WRITABLE = 2;
constant uint HANDLE_FLAG_DIRTY = 4;
constant uint HANDLE_FLAG_STREAMING = 8;

// Error codes
constant uint ERROR_NONE = 0;
constant uint ERROR_PATH_NOT_FOUND = 1;
constant uint ERROR_FILE_NOT_FOUND = 2;
constant uint ERROR_PERMISSION_DENIED = 3;
constant uint ERROR_OUT_OF_MEMORY = 4;
constant uint ERROR_IO_ERROR = 5;
constant uint ERROR_HANDLE_TABLE_FULL = 6;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// IORequest (32 bytes) - GPU → CPU
struct IORequest {
    uint request_type;      // READ | WRITE | CLOSE
    uint path_idx;          // Index into filesystem index
    uint handle_slot;       // Pre-allocated handle slot
    uint app_id;            // Requesting app

    uint offset;            // File offset for partial read
    uint size;              // Bytes to read (0 = entire file)
    uint priority;          // REALTIME | HIGH | NORMAL | LOW
    uint flags;             // STREAMING | MMAP | etc.
};

// FileHandle (64 bytes) - Status shared between GPU and CPU
struct FileHandle {
    // Identity (16 bytes) - written once by GPU
    uint handle_id;         // Unique ID for this handle
    uint path_idx;          // Which file
    uint app_id;            // Owner app
    uint flags;             // READABLE | WRITABLE | DIRTY

    // Buffer location (16 bytes) - written by CPU when allocated
    uint content_offset;    // Offset in content pool
    uint content_size;      // Allocated size (page-aligned)
    uint file_size;         // Actual file size
    uint _pad0;

    // Status (16 bytes) - CPU writes, GPU reads
    atomic_uint status;     // UNUSED | LOADING | READY | ERROR | CLOSED
    uint error_code;        // If status == ERROR
    uint mtime_low;         // File modification time
    uint mtime_high;

    // I/O state (16 bytes)
    atomic_uint bytes_loaded; // For streaming: how much loaded so far
    uint stream_chunk_size;   // If streaming, size of each chunk
    uint _pad1[2];
};

// PipelineState (64 bytes)
struct PipelineState {
    // Request queue (ring buffer)
    atomic_uint request_head;   // GPU writes here (producer)
    atomic_uint request_tail;   // CPU reads here (consumer)
    uint request_capacity;      // 128 typically
    uint _pad0;

    // Handle allocation
    atomic_uint next_handle_id;
    atomic_uint handle_count;
    uint max_handles;           // 1024 typically
    uint _pad1;

    // Content pool (bump allocator)
    atomic_uint pool_head;      // Next free offset
    uint pool_size;
    atomic_uint pool_used;
    uint _pad2;

    // Statistics
    atomic_uint total_reads;
    atomic_uint total_writes;
    atomic_uint total_bytes;
    uint _pad3;
};

// ═══════════════════════════════════════════════════════════════════════════════
// INLINE API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Request file read - returns handle slot, or INVALID_HANDLE if queue full
inline uint request_read(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint path_idx,
    uint app_id,
    uint priority
) {
    // 1. Allocate handle slot (Issue #270 fix: use CAS loop for race-free allocation)
    uint handle_slot;
    uint current = atomic_load_explicit(&state->handle_count, memory_order_relaxed);
    do {
        if (current >= state->max_handles) {
            return INVALID_HANDLE;  // No slots available
        }
        handle_slot = current;
    } while (!atomic_compare_exchange_weak_explicit(
        &state->handle_count, &current, current + 1,
        memory_order_relaxed, memory_order_relaxed));

    // 2. Initialize handle (GPU-side fields only)
    uint handle_id = atomic_fetch_add_explicit(&state->next_handle_id, 1, memory_order_relaxed);
    handles[handle_slot].handle_id = handle_id;
    handles[handle_slot].path_idx = path_idx;
    handles[handle_slot].app_id = app_id;
    handles[handle_slot].flags = HANDLE_FLAG_READABLE;
    atomic_store_explicit(&handles[handle_slot].status, STATUS_LOADING, memory_order_relaxed);

    // 3. Queue the request (lock-free ring buffer)
    uint head = atomic_fetch_add_explicit(&state->request_head, 1, memory_order_relaxed);
    uint idx = head % state->request_capacity;

    request_queue[idx].request_type = IO_READ;
    request_queue[idx].path_idx = path_idx;
    request_queue[idx].handle_slot = handle_slot;
    request_queue[idx].app_id = app_id;
    request_queue[idx].offset = 0;
    request_queue[idx].size = 0;  // Entire file
    request_queue[idx].priority = priority;
    request_queue[idx].flags = 0;

    // Increment stats
    atomic_fetch_add_explicit(&state->total_reads, 1, memory_order_relaxed);

    // 4. GPU continues immediately (doesn't wait for CPU)
    return handle_slot;
}

// Check if file is ready - called in app's update loop
inline uint check_status(device FileHandle* handles, uint handle_slot) {
    return atomic_load_explicit(&handles[handle_slot].status, memory_order_relaxed);
}

// Get file size (only valid when STATUS_READY)
inline uint get_file_size(device FileHandle* handles, uint handle_slot) {
    return handles[handle_slot].file_size;
}

// Get content offset in pool (only valid when STATUS_READY)
inline uint get_content_offset(device FileHandle* handles, uint handle_slot) {
    return handles[handle_slot].content_offset;
}

// Get error code (only valid when STATUS_ERROR)
inline uint get_error_code(device FileHandle* handles, uint handle_slot) {
    return handles[handle_slot].error_code;
}

// Close handle - marks for cleanup
inline void close_handle(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint handle_slot
) {
    // Mark as closed
    atomic_store_explicit(&handles[handle_slot].status, STATUS_CLOSED, memory_order_relaxed);

    // Queue close request (CPU will reclaim pool space)
    uint head = atomic_fetch_add_explicit(&state->request_head, 1, memory_order_relaxed);
    uint idx = head % state->request_capacity;

    request_queue[idx].request_type = IO_CLOSE;
    request_queue[idx].handle_slot = handle_slot;
    request_queue[idx].path_idx = 0;
    request_queue[idx].app_id = 0;
    request_queue[idx].offset = 0;
    request_queue[idx].size = 0;
    request_queue[idx].priority = IO_PRIORITY_LOW;
    request_queue[idx].flags = 0;
}

// Request file write
inline uint request_write(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint handle_slot,
    uint data_offset,  // Where in content_pool
    uint data_size
) {
    // Queue write request
    uint head = atomic_fetch_add_explicit(&state->request_head, 1, memory_order_relaxed);
    uint idx = head % state->request_capacity;

    request_queue[idx].request_type = IO_WRITE;
    request_queue[idx].handle_slot = handle_slot;
    request_queue[idx].path_idx = 0;
    request_queue[idx].app_id = 0;
    request_queue[idx].offset = data_offset;
    request_queue[idx].size = data_size;
    request_queue[idx].priority = IO_PRIORITY_NORMAL;
    request_queue[idx].flags = 0;

    // Increment stats
    atomic_fetch_add_explicit(&state->total_writes, 1, memory_order_relaxed);

    return head;  // Request ID for tracking
}

// Request streaming read (for large files)
inline uint request_stream(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint path_idx,
    uint app_id,
    uint chunk_size  // e.g., 1MB chunks
) {
    // Allocate handle slot (Issue #270 fix: use CAS loop for race-free allocation)
    uint handle_slot;
    uint current = atomic_load_explicit(&state->handle_count, memory_order_relaxed);
    do {
        if (current >= state->max_handles) {
            return INVALID_HANDLE;  // No slots available
        }
        handle_slot = current;
    } while (!atomic_compare_exchange_weak_explicit(
        &state->handle_count, &current, current + 1,
        memory_order_relaxed, memory_order_relaxed));

    // Initialize handle with streaming flag
    uint handle_id = atomic_fetch_add_explicit(&state->next_handle_id, 1, memory_order_relaxed);
    handles[handle_slot].handle_id = handle_id;
    handles[handle_slot].path_idx = path_idx;
    handles[handle_slot].app_id = app_id;
    handles[handle_slot].flags = HANDLE_FLAG_READABLE | HANDLE_FLAG_STREAMING;
    handles[handle_slot].stream_chunk_size = chunk_size;
    atomic_store_explicit(&handles[handle_slot].bytes_loaded, 0, memory_order_relaxed);
    atomic_store_explicit(&handles[handle_slot].status, STATUS_LOADING, memory_order_relaxed);

    // Queue request with streaming flag
    uint head = atomic_fetch_add_explicit(&state->request_head, 1, memory_order_relaxed);
    uint idx = head % state->request_capacity;

    request_queue[idx].request_type = IO_READ;
    request_queue[idx].path_idx = path_idx;
    request_queue[idx].handle_slot = handle_slot;
    request_queue[idx].app_id = app_id;
    request_queue[idx].offset = 0;
    request_queue[idx].size = 0;
    request_queue[idx].priority = IO_PRIORITY_NORMAL;
    request_queue[idx].flags = HANDLE_FLAG_STREAMING;

    return handle_slot;
}

// Check how much is loaded so far (streaming)
inline uint get_bytes_loaded(device FileHandle* handles, uint handle_slot) {
    return atomic_load_explicit(&handles[handle_slot].bytes_loaded, memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION KERNEL
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize pipeline state
kernel void content_pipeline_init(
    device PipelineState* state [[buffer(0)]],
    device FileHandle* handles [[buffer(1)]],
    constant uint& max_handles [[buffer(2)]],
    constant uint& request_capacity [[buffer(3)]],
    constant uint& pool_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Thread 0 initializes state
    if (tid == 0) {
        atomic_store_explicit(&state->request_head, 0, memory_order_relaxed);
        atomic_store_explicit(&state->request_tail, 0, memory_order_relaxed);
        state->request_capacity = request_capacity;

        atomic_store_explicit(&state->next_handle_id, 1, memory_order_relaxed);
        atomic_store_explicit(&state->handle_count, 0, memory_order_relaxed);
        state->max_handles = max_handles;

        atomic_store_explicit(&state->pool_head, 0, memory_order_relaxed);
        state->pool_size = pool_size;
        atomic_store_explicit(&state->pool_used, 0, memory_order_relaxed);

        atomic_store_explicit(&state->total_reads, 0, memory_order_relaxed);
        atomic_store_explicit(&state->total_writes, 0, memory_order_relaxed);
        atomic_store_explicit(&state->total_bytes, 0, memory_order_relaxed);
    }

    // All threads clear handles
    if (tid < max_handles) {
        handles[tid].handle_id = 0;
        handles[tid].path_idx = 0;
        handles[tid].app_id = 0;
        handles[tid].flags = 0;
        handles[tid].content_offset = 0;
        handles[tid].content_size = 0;
        handles[tid].file_size = 0;
        atomic_store_explicit(&handles[tid].status, STATUS_UNUSED, memory_order_relaxed);
        handles[tid].error_code = 0;
        handles[tid].mtime_low = 0;
        handles[tid].mtime_high = 0;
        atomic_store_explicit(&handles[tid].bytes_loaded, 0, memory_order_relaxed);
        handles[tid].stream_chunk_size = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST KERNELS
// ═══════════════════════════════════════════════════════════════════════════════

// Test kernel: Batch request reads (for testing without megakernel)
kernel void test_batch_request_reads(
    device PipelineState* state [[buffer(0)]],
    device IORequest* request_queue [[buffer(1)]],
    device FileHandle* handles [[buffer(2)]],
    device const uint* path_indices [[buffer(3)]],
    device uint* handle_slots [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint handle_slot = request_read(
        state,
        request_queue,
        handles,
        path_indices[tid],
        tid,  // app_id = tid for testing
        IO_PRIORITY_NORMAL
    );

    handle_slots[tid] = handle_slot;
}

// Test kernel: Batch check status
kernel void test_batch_check_status(
    device FileHandle* handles [[buffer(0)]],
    device const uint* handle_slots [[buffer(1)]],
    device uint* statuses [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    statuses[tid] = check_status(handles, handle_slots[tid]);
}

// Test kernel: Batch close handles
kernel void test_batch_close_handles(
    device PipelineState* state [[buffer(0)]],
    device IORequest* request_queue [[buffer(1)]],
    device FileHandle* handles [[buffer(2)]],
    device const uint* handle_slots [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    close_handle(state, request_queue, handles, handle_slots[tid]);
}
