# PRD: File Content Pipeline - GPU App File I/O

**Issue**: #165 - File Content Pipeline
**Priority**: Critical (Apps need to read/write files)
**Status**: Rewritten for GPU-First Architecture

---

## THE GPU IS THE COMPUTER

**The CPU is not a coordinator. The CPU is an I/O coprocessor.**

### CPU-Centric Thinking (WRONG)
```
CPU: check for GPU requests
CPU: found request, open file
CPU: dispatch MTLIOCommandQueue
CPU: wait for completion
CPU: update GPU buffer
CPU: tell GPU "data ready"
GPU: "thanks CPU, processing now"
```

### GPU-Native Thinking (RIGHT)
```
GPU: writes request to I/O queue (atomic)
GPU: continues working on other apps
GPU: periodically checks status buffer
GPU: sees STATUS_READY, processes data
GPU: never waits, never yields to CPU

Meanwhile (asynchronously):
CPU I/O thread: scans request queue
CPU I/O thread: dispatches MTLIOCommandQueue
MTLIOCommandQueue: loads data directly to GPU buffer
MTLIOCommandQueue: completion handler updates status
(GPU sees status change on next check)
```

**The GPU never waits for CPU. The GPU never asks CPU "is it ready yet?"**
**The GPU polls its own status buffer. CPU just happens to update it asynchronously.**

---

## Architecture

### Memory Layout

```
Content Pipeline Buffer (configurable, 64MB default):
┌──────────────────────────────────────────────────────────────┐
│ Pipeline State (64 bytes)                                     │
│   - Queue heads/tails (atomics)                               │
│   - Pool allocator state                                      │
│   - Statistics                                                │
├──────────────────────────────────────────────────────────────┤
│ I/O Request Queue (ring buffer, 4KB)                          │
│   - 128 IORequest entries × 32 bytes each                     │
│   - GPU writes requests, CPU reads them                       │
├──────────────────────────────────────────────────────────────┤
│ Handle Table (64KB)                                           │
│   - 1024 FileHandle entries × 64 bytes each                   │
│   - GPU reads status, CPU updates status                      │
├──────────────────────────────────────────────────────────────┤
│ Content Pool (remaining ~63MB)                                │
│   - Actual file content                                       │
│   - GPU reads/writes content                                  │
│   - MTLIOCommandQueue writes content (async)                  │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU (PERSISTENT KERNEL)                       │
│                                                                      │
│  App needs file:                                                     │
│    1. Allocate handle slot (atomic increment)                        │
│    2. Write IORequest to queue (atomic)                              │
│    3. Continue running (don't wait!)                                 │
│    4. Later: poll handle.status                                      │
│    5. When STATUS_READY: access content_pool[handle.offset]          │
│                                                                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ (GPU buffer, unified memory)
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                     CPU I/O COPROCESSOR THREAD                       │
│                                                                      │
│  Running independently (not orchestrated by GPU):                    │
│    1. Spin on request_queue (check every 100μs)                      │
│    2. When request found:                                            │
│       a. Resolve path_idx → filesystem path                          │
│       b. Open file, get size                                         │
│       c. Allocate content_pool space (bump pointer)                  │
│       d. Dispatch MTLIOCommandQueue (async, non-blocking)            │
│       e. Register completion handler                                 │
│    3. Continue scanning (don't wait for I/O!)                        │
│                                                                      │
│  Completion handler (called by Metal when I/O done):                 │
│    1. Update handle.status = STATUS_READY                            │
│    2. That's it. GPU will see it.                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### IORequest (32 bytes) - GPU → CPU

```metal
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

constant uint IO_READ = 1;
constant uint IO_WRITE = 2;
constant uint IO_CLOSE = 3;

constant uint IO_PRIORITY_REALTIME = 0;  // User waiting
constant uint IO_PRIORITY_HIGH = 1;      // App foreground
constant uint IO_PRIORITY_NORMAL = 2;    // Background
constant uint IO_PRIORITY_LOW = 3;       // Prefetch
```

### FileHandle (64 bytes) - Status shared between GPU and CPU

```metal
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

constant uint STATUS_UNUSED = 0;
constant uint STATUS_LOADING = 1;
constant uint STATUS_READY = 2;
constant uint STATUS_ERROR = 3;
constant uint STATUS_CLOSED = 4;

constant uint HANDLE_FLAG_READABLE = 1;
constant uint HANDLE_FLAG_WRITABLE = 2;
constant uint HANDLE_FLAG_DIRTY = 4;
constant uint HANDLE_FLAG_STREAMING = 8;
```

### PipelineState (64 bytes)

```metal
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
```

---

## GPU-Side API (Metal Functions)

These are **inline functions** called by apps within the persistent megakernel.
**No kernel dispatch needed.**

### request_read - App requests a file

```metal
// Request file read - returns handle slot, or INVALID if queue full
inline uint request_read(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint path_idx,
    uint app_id,
    uint priority
) {
    // 1. Allocate handle slot
    uint handle_slot = atomic_fetch_add_explicit(&state->handle_count, 1, memory_order_relaxed);
    if (handle_slot >= state->max_handles) {
        atomic_fetch_sub_explicit(&state->handle_count, 1, memory_order_relaxed);
        return INVALID_HANDLE;
    }

    // 2. Initialize handle (GPU-side fields only)
    uint handle_id = atomic_fetch_add_explicit(&state->next_handle_id, 1, memory_order_relaxed);
    handles[handle_slot].handle_id = handle_id;
    handles[handle_slot].path_idx = path_idx;
    handles[handle_slot].app_id = app_id;
    handles[handle_slot].flags = HANDLE_FLAG_READABLE;
    atomic_store_explicit(&handles[handle_slot].status, STATUS_LOADING, memory_order_release);

    // 3. Queue the request (lock-free ring buffer)
    uint head = atomic_fetch_add_explicit(&state->request_head, 1, memory_order_relaxed);
    uint idx = head % state->request_capacity;

    request_queue[idx] = IORequest{
        .request_type = IO_READ,
        .path_idx = path_idx,
        .handle_slot = handle_slot,
        .app_id = app_id,
        .offset = 0,
        .size = 0,  // Entire file
        .priority = priority,
        .flags = 0
    };

    // 4. GPU continues immediately (doesn't wait for CPU)
    return handle_slot;
}
```

### check_status - App polls for completion

```metal
// Check if file is ready - called in app's update loop
inline uint check_status(device FileHandle* handles, uint handle_slot) {
    // Atomic load with acquire to see CPU's writes
    return atomic_load_explicit(&handles[handle_slot].status, memory_order_acquire);
}
```

### get_content - App accesses file data

```metal
// Get pointer to file content - only valid when STATUS_READY
inline device uchar* get_content(
    device uchar* content_pool,
    device FileHandle* handles,
    uint handle_slot,
    thread uint* out_size
) {
    FileHandle h = handles[handle_slot];

    if (atomic_load_explicit(&h.status, memory_order_acquire) != STATUS_READY) {
        *out_size = 0;
        return nullptr;
    }

    *out_size = h.file_size;
    return content_pool + h.content_offset;
}
```

### close_handle - App releases file

```metal
// Close handle - marks for cleanup
inline void close_handle(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint handle_slot
) {
    // Mark as closed
    atomic_store_explicit(&handles[handle_slot].status, STATUS_CLOSED, memory_order_release);

    // Queue close request (CPU will reclaim pool space)
    uint head = atomic_fetch_add_explicit(&state->request_head, 1, memory_order_relaxed);
    uint idx = head % state->request_capacity;

    request_queue[idx] = IORequest{
        .request_type = IO_CLOSE,
        .handle_slot = handle_slot,
        // other fields unused
    };
}
```

---

## CPU I/O Coprocessor (Rust)

**This runs on a dedicated thread, completely async from GPU.**
**GPU never waits for this. GPU never calls into this.**

```rust
/// I/O coprocessor - runs independently, services GPU requests
pub struct IOCoprocessor {
    device: Device,
    io_queue: GpuIOQueue,

    // GPU buffer (shared memory)
    pipeline_buffer: Buffer,

    // Path resolution
    fs_index: Arc<GpuFilesystemIndex>,

    // Pending loads (for completion tracking)
    pending: Vec<PendingLoad>,
}

impl IOCoprocessor {
    /// Main loop - runs on dedicated thread
    pub fn run(&mut self) {
        loop {
            // 1. Drain request queue
            self.process_requests();

            // 2. Check completions
            self.check_completions();

            // 3. Brief sleep to avoid spinning
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    fn process_requests(&mut self) {
        let state = self.state();
        let mut tail = state.request_tail.load(Ordering::Acquire);
        let head = state.request_head.load(Ordering::Acquire);

        while tail != head {
            let idx = (tail as usize) % REQUEST_CAPACITY;
            let req = self.request_at(idx);

            match req.request_type {
                IO_READ => self.handle_read(req),
                IO_WRITE => self.handle_write(req),
                IO_CLOSE => self.handle_close(req),
                _ => {}
            }

            tail = tail.wrapping_add(1);
        }

        // Update tail (we've consumed these requests)
        state.request_tail.store(tail, Ordering::Release);
    }

    fn handle_read(&mut self, req: &IORequest) {
        // 1. Resolve path
        let path = match self.fs_index.get_path(req.path_idx) {
            Some(p) => p,
            None => {
                self.set_error(req.handle_slot, ERROR_PATH_NOT_FOUND);
                return;
            }
        };

        // 2. Get file size
        let file_size = match std::fs::metadata(&path) {
            Ok(m) => m.len(),
            Err(_) => {
                self.set_error(req.handle_slot, ERROR_FILE_NOT_FOUND);
                return;
            }
        };

        // 3. Allocate content pool space (bump allocator)
        let aligned_size = (file_size + 4095) & !4095;
        let offset = self.allocate_pool(aligned_size as u32);
        if offset == INVALID_OFFSET {
            self.set_error(req.handle_slot, ERROR_OUT_OF_MEMORY);
            return;
        }

        // 4. Update handle with buffer location
        let handle = self.handle_at(req.handle_slot);
        handle.content_offset = offset;
        handle.content_size = aligned_size as u32;
        handle.file_size = file_size as u32;

        // 5. Dispatch async load via MTLIOCommandQueue
        let pending = GpuIOBuffer::load_file_async_into(
            &self.io_queue,
            &path,
            &self.content_pool_buffer(),
            offset as u64,
            file_size,
        );

        if let Some(p) = pending {
            self.pending.push(PendingLoad {
                handle_slot: req.handle_slot,
                cmd_buffer: p,
            });
        }

        // CPU continues immediately (doesn't wait for I/O!)
    }

    fn check_completions(&mut self) {
        self.pending.retain(|p| {
            if p.cmd_buffer.is_complete() {
                // I/O done - update status (GPU will see this)
                let handle = self.handle_at(p.handle_slot);
                handle.status.store(STATUS_READY, Ordering::Release);
                false  // Remove from pending
            } else {
                true   // Keep checking
            }
        });
    }

    fn set_error(&mut self, handle_slot: u32, error: u32) {
        let handle = self.handle_at(handle_slot);
        handle.error_code = error;
        handle.status.store(STATUS_ERROR, Ordering::Release);
    }
}
```

---

## Integration with Megakernel

Apps access file I/O through inline functions in the persistent kernel:

```metal
kernel void megakernel(
    device PipelineState* io_state [[buffer(IO_STATE)]],
    device IORequest* io_requests [[buffer(IO_REQUESTS)]],
    device FileHandle* io_handles [[buffer(IO_HANDLES)]],
    device uchar* content_pool [[buffer(CONTENT_POOL)]],
    device AppState* apps [[buffer(APPS)]],
    // ...
) {
    uint app_id = /* ... */;
    AppState* app = &apps[app_id];

    // App wants to open a file
    if (app->pending_open_path != INVALID_PATH) {
        app->file_handle = request_read(
            io_state, io_requests, io_handles,
            app->pending_open_path,
            app_id,
            IO_PRIORITY_HIGH
        );
        app->pending_open_path = INVALID_PATH;
        app->loading_file = true;
    }

    // Check if file ready
    if (app->loading_file) {
        uint status = check_status(io_handles, app->file_handle);

        if (status == STATUS_READY) {
            app->loading_file = false;

            uint size;
            device uchar* data = get_content(
                content_pool, io_handles,
                app->file_handle, &size
            );

            // Process file content (GPU-native!)
            process_file(app, data, size);

        } else if (status == STATUS_ERROR) {
            app->loading_file = false;
            app->error = io_handles[app->file_handle].error_code;
        }
        // If still LOADING, continue with other work
    }

    // GPU never blocks, never waits
}
```

---

## Write Support

For writes, the flow is reversed:
1. GPU allocates content pool space
2. GPU writes data to content pool
3. GPU queues WRITE request
4. CPU coprocessor writes to filesystem

```metal
// GPU writes file content
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

    request_queue[idx] = IORequest{
        .request_type = IO_WRITE,
        .handle_slot = handle_slot,
        .offset = data_offset,  // Reuse offset field
        .size = data_size,
    };

    return head;  // Request ID for tracking
}
```

---

## Streaming Support (Large Files)

For files > 16MB, use streaming:

```metal
// Request streaming read
inline uint request_stream(
    device PipelineState* state,
    device IORequest* request_queue,
    device FileHandle* handles,
    uint path_idx,
    uint app_id,
    uint chunk_size  // e.g., 1MB chunks
) {
    uint handle_slot = /* allocate handle */;

    // Set streaming flag
    handles[handle_slot].flags |= HANDLE_FLAG_STREAMING;
    handles[handle_slot].stream_chunk_size = chunk_size;
    atomic_store_explicit(&handles[handle_slot].bytes_loaded, 0, memory_order_release);

    // Queue request with streaming flag
    // ...
}

// Check how much is loaded so far
inline uint get_bytes_loaded(device FileHandle* handles, uint handle_slot) {
    return atomic_load_explicit(&handles[handle_slot].bytes_loaded, memory_order_acquire);
}
```

CPU coprocessor loads chunks incrementally, updating `bytes_loaded` after each chunk.
GPU can start processing as soon as first chunk is ready.

---

## Error Handling

All errors are communicated via the status buffer:

| Error Code | Meaning |
|------------|---------|
| 0 | No error |
| 1 | Path not found in index |
| 2 | File not found on disk |
| 3 | Permission denied |
| 4 | Out of pool memory |
| 5 | I/O error |
| 6 | Handle table full |

GPU checks `handle.error_code` when `status == STATUS_ERROR`.

---

## Test Plan

### Functional Tests

```rust
#[test]
fn test_read_small_file() {
    // Create test file
    // GPU requests read
    // Poll until ready
    // Verify content matches
}

#[test]
fn test_concurrent_reads() {
    // 100 GPU threads request 100 different files
    // All complete without corruption
}

#[test]
fn test_streaming_large_file() {
    // Create 50MB file
    // Request streaming read
    // Process chunks as they arrive
    // Verify complete content
}

#[test]
fn test_write_cycle() {
    // GPU writes data to pool
    // Request write
    // Verify file on disk
}
```

### Performance Tests

```rust
#[test]
fn bench_read_latency() {
    // Target: <5ms for small files
}

#[test]
fn bench_throughput() {
    // Target: >1GB/s for sequential reads
}
```

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| GPU blocking time | 0 | GPU never waits for I/O |
| Read latency (small) | <5ms | Async, but fast |
| Throughput | >1GB/s | MTLIOCommandQueue + SSD |
| Concurrent handles | 1024 | Handle table size |
| CPU overhead | <1% | Runs on dedicated thread |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu_os/content_pipeline.rs` | IOCoprocessor, buffer setup |
| `src/gpu_os/shaders/content_pipeline.metal` | Inline functions for megakernel |
| `tests/test_issue_165_content_pipeline.rs` | Tests |

---

## Anti-Patterns (DO NOT DO)

| Bad Pattern | Why Bad | Good Pattern |
|-------------|---------|--------------|
| GPU waits for read | Blocks entire kernel | GPU polls status, continues working |
| CPU orchestrates | CPU in critical path | CPU is async coprocessor |
| Synchronous I/O | Blocks thread | MTLIOCommandQueue async |
| CPU allocates buffers | Round-trip | GPU allocates via atomic |
| Per-file dispatch | Kernel overhead | Batch requests, persistent kernel |
