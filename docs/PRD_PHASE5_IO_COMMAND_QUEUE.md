# PRD Phase 5: I/O Command Queue Integration

## THE GPU IS THE COMPUTER

**Issue**: Non-Blocking File Access from GPU Apps
**Phase**: 5 of 5
**Duration**: 2 weeks
**Depends On**: Phase 1-4 (Integer Ops, Atomics, DSL, WASM)
**Enables**: GPU apps loading files, assets, data without blocking

---

## Core Architecture Principle

**THE GPU NEVER WAITS. THE GPU NEVER BLOCKS. THE GPU NEVER ASKS.**

```
WRONG (request/response):
  GPU: "I need file X" → WAITS → CPU: "Here's file X" → GPU continues

RIGHT (command queue + poll):
  GPU: writes load command → CONTINUES WORKING → later polls status → uses data
```

The CPU is an **I/O peripheral**, not a coordinator. The CPU:
- Sets up DMA mappings at boot
- Handles hardware interrupts
- Writes completion flags

The CPU does NOT:
- Make decisions
- Block GPU execution
- Coordinate work

---

## Problem Statement

GPU apps need to load files (assets, data, configurations). Current options:
1. Pre-load everything at CPU boot (inflexible, memory-heavy)
2. CPU loads on demand, signals GPU (CPU is coordinator - WRONG)

We need: GPU-initiated, non-blocking file loads where GPU controls the flow.

---

## Technical Design

### Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU-VISIBLE UNIFIED MEMORY                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ I/O Request Queue (GPU writes, CPU reads)                │   │
│  │                                                           │   │
│  │ [Request 0][Request 1][Request 2]...[Request N-1]        │   │
│  │                                                           │   │
│  │ struct IoRequest {                                        │   │
│  │   op: u32,           // READ, WRITE, STAT                │   │
│  │   path_hash: u32,    // Hash into path table             │   │
│  │   buffer_offset: u32,// Where to put data                │   │
│  │   size: u32,         // Bytes requested                  │   │
│  │   flags: u32,        // Options                          │   │
│  │   _pad: [u32; 3],    // Alignment to 32 bytes           │   │
│  │ }                                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Completion Status (CPU writes, GPU reads)                 │   │
│  │                                                           │   │
│  │ [Status 0][Status 1][Status 2]...[Status N-1]            │   │
│  │                                                           │   │
│  │ struct IoStatus {                                         │   │
│  │   state: atomic_uint, // PENDING, IN_PROGRESS, COMPLETE  │   │
│  │   result: i32,        // Bytes read, or error code       │   │
│  │   _pad: [u32; 2],                                        │   │
│  │ }                                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Data Buffers (DMA target)                                 │   │
│  │                                                           │   │
│  │ [Buffer 0: 64KB][Buffer 1: 64KB]...[Buffer N-1: 64KB]    │   │
│  │                                                           │   │
│  │ File data is DMA'd directly here by MTLIOCommandQueue    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Path Table (pre-registered paths)                         │   │
│  │                                                           │   │
│  │ ["/assets/texture.png"]["/data/config.json"]...          │   │
│  │                                                           │   │
│  │ GPU references paths by hash, not string                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Atomic Counters                                           │   │
│  │                                                           │   │
│  │ request_head: atomic_uint  // CPU reads from here        │   │
│  │ request_tail: atomic_uint  // GPU writes here (claims)   │   │
│  │ completion_count: atomic_uint // Total completed         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
1. GPU App wants to load "/assets/level1.dat"
   │
   ▼
2. GPU: slot = atomic_add(&request_tail, 1)  // Claim slot
   │
   ▼
3. GPU: request[slot] = { op: READ, path_hash: hash("level1.dat"), ... }
   │
   ▼
4. GPU: continues other work (NEVER WAITS)
   │
   ▼
5. CPU I/O thread (polling): sees new request
   │
   ▼
6. CPU: MTLIOCommandQueue.load(file → buffer[slot])
   │
   ▼
7. Hardware: DMA transfers file data to buffer
   │
   ▼
8. CPU: atomic_store(&status[slot].state, COMPLETE)
   │
   ▼
9. GPU (later, when convenient): polls status[slot]
   │
   ▼
10. GPU: if COMPLETE, uses data from buffer[slot]
```

### Constants and Structures

```rust
// In src/gpu_os/io_queue.rs

// I/O operation types
pub const IO_OP_READ: u32 = 1;
pub const IO_OP_WRITE: u32 = 2;
pub const IO_OP_STAT: u32 = 3;

// Status values
pub const IO_STATUS_FREE: u32 = 0;
pub const IO_STATUS_PENDING: u32 = 1;
pub const IO_STATUS_IN_PROGRESS: u32 = 2;
pub const IO_STATUS_COMPLETE: u32 = 3;
pub const IO_STATUS_ERROR: u32 = 4;

// Queue configuration
pub const IO_QUEUE_SIZE: usize = 64;        // Max concurrent requests
pub const IO_BUFFER_SIZE: usize = 64 * 1024; // 64KB per buffer
pub const IO_PATH_TABLE_SIZE: usize = 1024;  // Max registered paths

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IoRequest {
    pub op: u32,
    pub path_hash: u32,
    pub buffer_offset: u32,
    pub size: u32,
    pub flags: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IoStatus {
    pub state: u32,  // Accessed atomically
    pub result: i32, // Bytes read or error code
    pub _pad: [u32; 2],
}

// Atomic counters layout (at known offsets)
pub const ATOMIC_REQUEST_HEAD: u32 = 0;
pub const ATOMIC_REQUEST_TAIL: u32 = 1;
pub const ATOMIC_COMPLETION_COUNT: u32 = 2;
```

### Metal Shader: GPU-side I/O Operations

```metal
// In shaders/io_queue.metal

constant uint IO_OP_READ = 1;
constant uint IO_OP_WRITE = 2;
constant uint IO_OP_STAT = 3;

constant uint IO_STATUS_FREE = 0;
constant uint IO_STATUS_PENDING = 1;
constant uint IO_STATUS_IN_PROGRESS = 2;
constant uint IO_STATUS_COMPLETE = 3;
constant uint IO_STATUS_ERROR = 4;

struct IoRequest {
    uint op;
    uint path_hash;
    uint buffer_offset;
    uint size;
    uint flags;
    uint _pad[3];
};

struct IoStatus {
    atomic_uint state;
    int result;
    uint _pad[2];
};

// Submit a read request (returns slot index)
inline uint io_submit_read(
    device IoRequest* requests,
    device atomic_uint* tail,
    uint path_hash,
    uint buffer_offset,
    uint size
) {
    // Claim a slot atomically
    uint slot = atomic_fetch_add_explicit(tail, 1, memory_order_relaxed);
    slot = slot % IO_QUEUE_SIZE;

    // Fill request
    requests[slot].op = IO_OP_READ;
    requests[slot].path_hash = path_hash;
    requests[slot].buffer_offset = buffer_offset;
    requests[slot].size = size;
    requests[slot].flags = 0;

    return slot;
}

// Poll completion status (non-blocking)
inline bool io_poll_complete(
    device IoStatus* statuses,
    uint slot
) {
    uint state = atomic_load_explicit(&statuses[slot].state, memory_order_acquire);
    return state == IO_STATUS_COMPLETE || state == IO_STATUS_ERROR;
}

// Get result after completion
inline int io_get_result(
    device IoStatus* statuses,
    uint slot
) {
    return statuses[slot].result;
}

// Reset slot for reuse
inline void io_reset_slot(
    device IoStatus* statuses,
    uint slot
) {
    atomic_store_explicit(&statuses[slot].state, IO_STATUS_FREE, memory_order_release);
}
```

### Bytecode VM: I/O Opcodes

```rust
// New opcodes for I/O (0xF0-0xFF range)

pub const IO_SUBMIT_READ: u8 = 0xF0;   // dst, path_hash_reg, size_reg
pub const IO_POLL: u8 = 0xF1;          // dst, slot_reg (dst = 1 if complete)
pub const IO_GET_RESULT: u8 = 0xF2;    // dst, slot_reg (dst = bytes read or error)
pub const IO_RESET_SLOT: u8 = 0xF3;    // slot_reg
pub const IO_GET_BUFFER: u8 = 0xF4;    // dst, slot_reg (dst = buffer base address)
```

```metal
// In bytecode_vm.metal, add I/O opcode handlers:

case 0xF0: { // IO_SUBMIT_READ
    uint path_hash = as_type<uint>(regs[src1].x);
    uint size = as_type<uint>(regs[src2].x);

    // Calculate buffer offset (slot * BUFFER_SIZE)
    uint slot = atomic_fetch_add_explicit(&io_counters[1], 1, memory_order_relaxed);
    slot = slot % IO_QUEUE_SIZE;
    uint buffer_offset = slot * IO_BUFFER_SIZE;

    // Submit request
    io_requests[slot].op = IO_OP_READ;
    io_requests[slot].path_hash = path_hash;
    io_requests[slot].buffer_offset = buffer_offset;
    io_requests[slot].size = size;

    // Mark as pending
    atomic_store_explicit(&io_statuses[slot].state, IO_STATUS_PENDING, memory_order_release);

    // Return slot to caller
    regs[dst].x = as_type<float>(slot);
    break;
}

case 0xF1: { // IO_POLL
    uint slot = as_type<uint>(regs[src1].x);
    uint state = atomic_load_explicit(&io_statuses[slot].state, memory_order_acquire);
    bool complete = (state == IO_STATUS_COMPLETE || state == IO_STATUS_ERROR);
    regs[dst].x = as_type<float>(complete ? 1u : 0u);
    break;
}

case 0xF2: { // IO_GET_RESULT
    uint slot = as_type<uint>(regs[src1].x);
    int result = io_statuses[slot].result;
    regs[dst].x = as_type<float>(result);
    break;
}

case 0xF3: { // IO_RESET_SLOT
    uint slot = as_type<uint>(regs[src1].x);
    atomic_store_explicit(&io_statuses[slot].state, IO_STATUS_FREE, memory_order_release);
    break;
}

case 0xF4: { // IO_GET_BUFFER
    uint slot = as_type<uint>(regs[src1].x);
    uint buffer_base = IO_BUFFER_BASE + slot * IO_BUFFER_SIZE;
    regs[dst].x = as_type<float>(buffer_base);
    break;
}
```

### CPU I/O Thread

```rust
// In src/gpu_os/io_thread.rs

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use metal::{Device, MTLResourceOptions};

pub struct IoThread {
    running: AtomicBool,
    handle: Option<thread::JoinHandle<()>>,
}

impl IoThread {
    pub fn start(
        device: Device,
        request_buffer: Buffer,
        status_buffer: Buffer,
        data_buffer: Buffer,
        path_table: Arc<PathTable>,
    ) -> Self {
        let running = AtomicBool::new(true);
        let running_clone = running.clone();

        let handle = thread::spawn(move || {
            Self::run_loop(
                &running_clone,
                &device,
                &request_buffer,
                &status_buffer,
                &data_buffer,
                &path_table,
            );
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    fn run_loop(
        running: &AtomicBool,
        device: &Device,
        request_buffer: &Buffer,
        status_buffer: &Buffer,
        data_buffer: &Buffer,
        path_table: &PathTable,
    ) {
        // Create I/O command queue
        let io_desc = MTLIOCommandQueueDescriptor::new();
        let io_queue = device.new_io_command_queue(&io_desc)
            .expect("Failed to create IOCommandQueue");

        let mut last_processed = 0u32;

        while running.load(Ordering::Relaxed) {
            // Read current tail (where GPU has written)
            let tail = Self::read_atomic_tail(request_buffer);

            // Process new requests
            while last_processed != tail {
                let slot = last_processed % IO_QUEUE_SIZE as u32;
                let request = Self::read_request(request_buffer, slot);

                // Mark as in progress
                Self::write_status_state(status_buffer, slot, IO_STATUS_IN_PROGRESS);

                match request.op {
                    IO_OP_READ => {
                        // Look up path
                        if let Some(path) = path_table.get(request.path_hash) {
                            // Open file handle
                            if let Ok(handle) = device.new_io_file_handle(&path) {
                                // Create I/O command buffer
                                let io_buffer = io_queue.new_command_buffer();

                                // Calculate destination in data buffer
                                let offset = request.buffer_offset as u64;
                                let size = request.size as u64;

                                // Load file to buffer
                                io_buffer.load_buffer(
                                    data_buffer,
                                    offset,
                                    size,
                                    &handle,
                                    0, // source offset
                                );

                                // Commit and wait for completion
                                io_buffer.commit();
                                io_buffer.wait_until_completed();

                                // Mark complete
                                Self::write_status_result(status_buffer, slot, size as i32);
                                Self::write_status_state(status_buffer, slot, IO_STATUS_COMPLETE);
                            } else {
                                // File not found
                                Self::write_status_result(status_buffer, slot, -1);
                                Self::write_status_state(status_buffer, slot, IO_STATUS_ERROR);
                            }
                        } else {
                            // Path not registered
                            Self::write_status_result(status_buffer, slot, -2);
                            Self::write_status_state(status_buffer, slot, IO_STATUS_ERROR);
                        }
                    }
                    IO_OP_STAT => {
                        // Return file size
                        if let Some(path) = path_table.get(request.path_hash) {
                            if let Ok(meta) = std::fs::metadata(&path) {
                                Self::write_status_result(status_buffer, slot, meta.len() as i32);
                                Self::write_status_state(status_buffer, slot, IO_STATUS_COMPLETE);
                            } else {
                                Self::write_status_result(status_buffer, slot, -1);
                                Self::write_status_state(status_buffer, slot, IO_STATUS_ERROR);
                            }
                        }
                    }
                    _ => {
                        Self::write_status_result(status_buffer, slot, -3);
                        Self::write_status_state(status_buffer, slot, IO_STATUS_ERROR);
                    }
                }

                last_processed = last_processed.wrapping_add(1);
            }

            // Small sleep to avoid busy-waiting
            thread::sleep(Duration::from_micros(100));
        }
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

// Path table for registered files
pub struct PathTable {
    paths: HashMap<u32, PathBuf>,
}

impl PathTable {
    pub fn new() -> Self {
        Self { paths: HashMap::new() }
    }

    pub fn register(&mut self, path: &Path) -> u32 {
        let hash = Self::hash_path(path);
        self.paths.insert(hash, path.to_path_buf());
        hash
    }

    pub fn get(&self, hash: u32) -> Option<&Path> {
        self.paths.get(&hash).map(|p| p.as_path())
    }

    fn hash_path(path: &Path) -> u32 {
        // DJB2 hash of filename
        let name = path.file_name().unwrap().to_string_lossy();
        let mut h: u32 = 5381;
        for b in name.bytes() {
            h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as u32);
        }
        h
    }
}
```

### DSL Integration

```rust
// In GPU DSL, add I/O primitives:

gpu_kernel! {
    fn load_asset_demo() {
        // Hash of "level1.dat" (pre-computed or use built-in hash)
        let path_hash = 0x12345678u32;
        let size = 4096u32;

        // Submit read request (non-blocking)
        let slot = io_read(path_hash, size);

        // GPU continues doing other work...
        for i in 0..1000 {
            // ... compute something ...
        }

        // Check if load is complete (non-blocking)
        if io_complete(slot) {
            let bytes_read = io_result(slot);
            if bytes_read > 0 {
                // Get buffer address
                let buffer_base = io_buffer(slot);

                // Use the data
                let first_word = STATE[buffer_base];

                // Reset slot for reuse
                io_reset(slot);
            }
        }
        // If not complete, try again next frame
    }
}
```

---

## Test Cases

### Test File: `tests/test_phase5_io_queue.rs`

```rust
//! Phase 5: I/O Command Queue Tests
//!
//! THE GPU IS THE COMPUTER.
//! GPU initiates file loads, never waits.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;
use metal::Device;

use rust_experiment::gpu_os::io_queue::{IoQueue, IoThread, PathTable};
use rust_experiment::gpu_os::bytecode_vm::BytecodeVM;

fn setup_test_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
    let path = dir.join(name);
    let mut file = File::create(&path).unwrap();
    file.write_all(content).unwrap();
    path
}

#[test]
fn test_io_submit_read() {
    let dir = tempdir().unwrap();
    let test_file = setup_test_file(dir.path(), "test.dat", b"Hello, GPU!");

    let device = Device::system_default().expect("No Metal device");
    let io_queue = IoQueue::new(&device).expect("Failed to create IO queue");

    // Register path
    let path_hash = io_queue.path_table().register(&test_file);

    // Start I/O thread
    let io_thread = IoThread::start(
        device.clone(),
        io_queue.request_buffer().clone(),
        io_queue.status_buffer().clone(),
        io_queue.data_buffer().clone(),
        io_queue.path_table().clone(),
    );

    // Create VM with I/O buffers
    let vm = BytecodeVM::with_io(&device, &io_queue).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Submit read request
    asm.loadi_uint(8, path_hash);
    asm.loadi_uint(9, 64);  // size
    asm.io_submit_read(10, 8, 9);  // r10 = slot

    // Poll loop (in real app, would do other work)
    let poll_start = asm.current_offset();
    asm.io_poll(11, 10);
    asm.jz(11, poll_start);  // Keep polling until complete

    // Get result
    asm.io_get_result(12, 10);  // r12 = bytes read

    // Get buffer and read first byte
    asm.io_get_buffer(13, 10);  // r13 = buffer base
    asm.ld(14, 13, 0);          // r14 = first float4

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let bytes_read = vm.read_register_int(12);
    assert_eq!(bytes_read, 11);  // "Hello, GPU!" = 11 bytes

    // Verify data (first 4 bytes as uint = "Hell")
    let first_word = vm.read_register_uint(14);
    let expected = u32::from_le_bytes([b'H', b'e', b'l', b'l']);
    assert_eq!(first_word, expected);

    io_thread.stop();
}

#[test]
fn test_io_multiple_concurrent_reads() {
    let dir = tempdir().unwrap();
    let file1 = setup_test_file(dir.path(), "a.txt", b"File A content");
    let file2 = setup_test_file(dir.path(), "b.txt", b"File B content");
    let file3 = setup_test_file(dir.path(), "c.txt", b"File C content");

    let device = Device::system_default().expect("No Metal device");
    let io_queue = IoQueue::new(&device).expect("Failed to create IO queue");

    let hash1 = io_queue.path_table().register(&file1);
    let hash2 = io_queue.path_table().register(&file2);
    let hash3 = io_queue.path_table().register(&file3);

    let io_thread = IoThread::start(/* ... */);

    let vm = BytecodeVM::with_io(&device, &io_queue).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Submit 3 reads simultaneously
    asm.loadi_uint(8, hash1);
    asm.loadi_uint(9, 64);
    asm.io_submit_read(10, 8, 9);  // slot for file1

    asm.loadi_uint(8, hash2);
    asm.io_submit_read(11, 8, 9);  // slot for file2

    asm.loadi_uint(8, hash3);
    asm.io_submit_read(12, 8, 9);  // slot for file3

    // Wait for all to complete
    let wait_all = asm.current_offset();
    asm.io_poll(20, 10);
    asm.io_poll(21, 11);
    asm.io_poll(22, 12);
    asm.bit_and(23, 20, 21);
    asm.bit_and(23, 23, 22);
    asm.jz(23, wait_all);

    // Get results
    asm.io_get_result(24, 10);
    asm.io_get_result(25, 11);
    asm.io_get_result(26, 12);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    // All should have read 14 bytes
    assert_eq!(vm.read_register_int(24), 14);
    assert_eq!(vm.read_register_int(25), 14);
    assert_eq!(vm.read_register_int(26), 14);

    io_thread.stop();
}

#[test]
fn test_io_file_not_found() {
    let device = Device::system_default().expect("No Metal device");
    let io_queue = IoQueue::new(&device).expect("Failed to create IO queue");

    // Don't register any paths

    let io_thread = IoThread::start(/* ... */);
    let vm = BytecodeVM::with_io(&device, &io_queue).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Try to read non-existent path
    asm.loadi_uint(8, 0xDEADBEEF);  // Unknown hash
    asm.loadi_uint(9, 64);
    asm.io_submit_read(10, 8, 9);

    // Poll until complete
    let poll = asm.current_offset();
    asm.io_poll(11, 10);
    asm.jz(11, poll);

    // Get result (should be error)
    asm.io_get_result(12, 10);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let result = vm.read_register_int(12);
    assert!(result < 0);  // Negative = error

    io_thread.stop();
}

#[test]
fn test_io_non_blocking_behavior() {
    // Verify GPU doesn't block while I/O is pending

    let dir = tempdir().unwrap();
    // Create a large file to ensure I/O takes measurable time
    let large_content = vec![0u8; 1024 * 1024];  // 1MB
    let large_file = setup_test_file(dir.path(), "large.dat", &large_content);

    let device = Device::system_default().expect("No Metal device");
    let io_queue = IoQueue::new(&device).expect("Failed to create IO queue");
    let hash = io_queue.path_table().register(&large_file);

    let io_thread = IoThread::start(/* ... */);
    let vm = BytecodeVM::with_io(&device, &io_queue).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Submit read
    asm.loadi_uint(8, hash);
    asm.loadi_uint(9, 1024 * 1024);
    asm.io_submit_read(10, 8, 9);

    // Do work while I/O is pending
    asm.loadi_uint(20, 0);  // counter
    let work_loop = asm.current_offset();
    asm.loadi_uint(21, 1);
    asm.int_add(20, 20, 21);

    // Check if I/O is complete
    asm.io_poll(22, 10);
    asm.jz(22, work_loop);  // If not complete, keep working

    // r20 now contains how many iterations we did while waiting

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let iterations = vm.read_register_uint(20);
    println!("GPU did {} iterations while waiting for I/O", iterations);
    assert!(iterations > 0, "GPU should have done work while I/O was pending");

    io_thread.stop();
}

#[test]
fn test_io_stat_operation() {
    let dir = tempdir().unwrap();
    let content = b"12345678901234567890";  // 20 bytes
    let test_file = setup_test_file(dir.path(), "sized.dat", content);

    let device = Device::system_default().expect("No Metal device");
    let io_queue = IoQueue::new(&device).expect("Failed to create IO queue");
    let hash = io_queue.path_table().register(&test_file);

    let io_thread = IoThread::start(/* ... */);
    let vm = BytecodeVM::with_io(&device, &io_queue).expect("Failed to create VM");

    let mut asm = BytecodeAssembler::new();

    // Submit stat request
    asm.loadi_uint(8, hash);
    asm.loadi_uint(9, 0);  // size = 0 for stat
    asm.io_submit_stat(10, 8);

    // Poll
    let poll = asm.current_offset();
    asm.io_poll(11, 10);
    asm.jz(11, poll);

    // Get result (file size)
    asm.io_get_result(12, 10);

    asm.halt();

    vm.load_program(&asm.finish());
    vm.execute(&device);

    let file_size = vm.read_register_int(12);
    assert_eq!(file_size, 20);

    io_thread.stop();
}
```

---

## Benchmarks

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Request submission | < 100 cycles | Atomic add + memory writes |
| Poll check | < 50 cycles | Single atomic load |
| I/O latency (4KB) | < 1ms | MTLIOCommandQueue |
| I/O latency (1MB) | < 10ms | Depends on storage speed |
| Concurrent requests | 64 | Queue size |

### Benchmark: Throughput

```rust
#[bench]
fn bench_io_submit_throughput(b: &mut Bencher) {
    // Measure how fast GPU can submit I/O requests
    // (not waiting for completion)

    let device = Device::system_default().unwrap();
    let io_queue = IoQueue::new(&device).unwrap();
    let vm = BytecodeVM::with_io(&device, &io_queue).unwrap();

    let mut asm = BytecodeAssembler::new();
    for i in 0..64 {
        asm.loadi_uint(8, i);
        asm.loadi_uint(9, 64);
        asm.io_submit_read(10, 8, 9);
    }
    asm.halt();

    vm.load_program(&asm.finish());

    b.iter(|| {
        vm.execute(&device);
    });
}
```

---

## Success Criteria

1. **Non-blocking**: GPU continues work while I/O is pending
2. **Correct data**: File contents arrive intact in GPU buffer
3. **Error handling**: File-not-found returns error code, not crash
4. **Concurrent**: Multiple simultaneous requests work correctly
5. **Performance**: < 1ms overhead for small files

---

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| GPU spin-waits | Wastes GPU cycles | Poll + do other work |
| CPU coordinates | CPU becomes bottleneck | GPU submits, CPU executes |
| Synchronous API | GPU blocks | Async submit + poll |
| String paths at runtime | GPU can't process strings | Pre-registered path hashes |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu_os/io_queue.rs` | I/O queue management |
| `src/gpu_os/io_thread.rs` | CPU I/O service thread |
| `src/gpu_os/shaders/io_queue.metal` | GPU-side I/O operations |
| `tests/test_phase5_io_queue.rs` | Test file |
| `benches/bench_io_queue.rs` | Benchmarks |

---

## Summary: Complete Rust→GPU Pipeline

With all 5 phases complete, we have:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE PIPELINE                             │
│                                                                  │
│  Rust Source Code                                                │
│       │                                                          │
│       ├─── gpu_kernel! macro ───> GPU Bytecode (Phase 3)        │
│       │                                                          │
│       └─── rustc → WASM ───> Translator ───> GPU Bytecode (4)   │
│                                                                  │
│  GPU Bytecode                                                    │
│       │                                                          │
│       ▼                                                          │
│  BytecodeVM (Metal compute shader)                               │
│       │                                                          │
│       ├─── Integer ops (Phase 1)                                │
│       ├─── Atomic ops (Phase 2)                                 │
│       └─── I/O queue ops (Phase 5)                              │
│                                                                  │
│  GPU Execution                                                   │
│       │                                                          │
│       ├─── Compute: runs bytecode                               │
│       ├─── I/O: submits requests, polls completion              │
│       └─── Render: emits vertices                               │
│                                                                  │
│  THE GPU IS THE COMPUTER                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Total estimated time: 14 weeks (3.5 months)**

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Integer Ops | 2 weeks | 2 weeks |
| Phase 2: Atomic Ops | 2 weeks | 4 weeks |
| Phase 3: DSL Macro | 4 weeks | 8 weeks |
| Phase 4: WASM Translator | 4 weeks | 12 weeks |
| Phase 5: I/O Queue | 2 weeks | 14 weeks |
