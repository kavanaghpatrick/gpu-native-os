# GPU Threading Model: You ARE The Threads

## Overview

**THE GPU IS THE COMPUTER.**

On GPU, you don't spawn threads - you ARE thousands of threads running simultaneously. This document explains how to think about parallelism in GPU programs.

## The Fundamental Difference

### CPU Model (Traditional)
```rust
use std::thread;

fn main() {
    let handles: Vec<_> = (0..8).map(|i| {
        thread::spawn(move || {
            compute(i)
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }
}
```
You explicitly create 8 threads. OS schedules them across cores.

### GPU Model (Our System)
```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();  // I am one of 10,000+ threads
    compute(tid)
}
```
Your code IS the thread. The GPU launches thousands of instances automatically.

## How It Works

```
Your Code                          GPU Execution
─────────────                      ─────────────
                                   Thread 0: main() → compute(0)
fn main() {                        Thread 1: main() → compute(1)
    let tid = thread_id();   →     Thread 2: main() → compute(2)
    compute(tid)                   Thread 3: main() → compute(3)
}                                  ...
                                   Thread 9999: main() → compute(9999)
```

Every thread runs the SAME code but with different `thread_id()`.

## Parallel Patterns

### Pattern 1: Parallel Map
```rust
// CPU: spawn threads, join, collect
let results: Vec<_> = data.par_iter().map(|x| x * 2).collect();

// GPU: each thread processes one element
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id() as usize;
    let input = read_state(tid as i32);
    let output = input * 2;
    write_state(tid as i32, output);
    0
}
```

### Pattern 2: Parallel Reduce (Sum)
```rust
// CPU
let sum: i32 = data.par_iter().sum();

// GPU: each thread adds its element atomically
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id() as usize;
    let my_value = read_state(tid as i32);

    // Atomic add to shared result at state[0]
    atomic_add(0, my_value);

    0
}
```

### Pattern 3: Parallel Filter
```rust
// CPU
let filtered: Vec<_> = data.par_iter().filter(|x| **x > 0).collect();

// GPU: each thread checks condition, writes to compacted output
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();
    let value = read_state(tid);

    if value > 0 {
        // Atomically get next output slot
        let slot = atomic_add(OUTPUT_COUNT_IDX, 1);
        write_output(slot, value);
    }

    0
}
```

### Pattern 4: Image Processing (Per-Pixel)
```rust
// Each thread = one pixel
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let gid = thread_id();
    let x = gid % WIDTH;
    let y = gid / WIDTH;

    // Read pixel
    let r = read_input(gid * 4 + 0);
    let g = read_input(gid * 4 + 1);
    let b = read_input(gid * 4 + 2);

    // Process (e.g., grayscale)
    let gray = (r + g + b) / 3;

    // Write output
    set_pixel(x, y, gray, gray, gray, 1.0);

    0
}
```

### Pattern 5: Simulation (Per-Particle)
```rust
// Each thread = one particle
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let pid = thread_id();

    // Read particle state
    let x = read_state(pid * 4 + 0);
    let y = read_state(pid * 4 + 1);
    let vx = read_state(pid * 4 + 2);
    let vy = read_state(pid * 4 + 3);

    // Update position
    let new_x = x + vx * DT;
    let new_y = y + vy * DT;

    // Write back
    write_state(pid * 4 + 0, new_x);
    write_state(pid * 4 + 1, new_y);

    0
}
```

## Thread Organization

### Launch Configuration
```rust
// Host-side (Rust app that launches GPU program)
let thread_count = 1024 * 1024;  // 1M threads
system.launch_app(bytecode, state, thread_count);
```

### Thread Hierarchy
```
Grid (all threads)
├── Threadgroup 0 (e.g., 256 threads)
│   ├── SIMD Group 0 (32 threads - execute in lockstep)
│   ├── SIMD Group 1
│   └── ...
├── Threadgroup 1
└── ...
```

### SIMD Considerations
**Critical**: 32 threads in a SIMD group execute the SAME instruction at the SAME time.

```rust
// BAD: SIMD divergence
if thread_id() % 2 == 0 {
    do_something_expensive();  // Half the threads wait
} else {
    do_something_else();       // Other half waits
}

// GOOD: All threads take same path
let value = if condition { a } else { b };  // Branchless
```

## Synchronization

### Within Threadgroup
```rust
// Barrier - all threads in group reach this point before continuing
threadgroup_barrier();

// Shared memory within threadgroup
write_shared(local_id, my_value);
threadgroup_barrier();
let neighbor = read_shared((local_id + 1) % GROUP_SIZE);
```

### Across All Threads
```rust
// Atomics
let old = atomic_add(address, value);
let success = atomic_compare_exchange(address, expected, new_value);
```

## What You CAN'T Do (Within a Single Kernel Dispatch)

| CPU Pattern | Why It Fails on GPU | GPU Alternative |
|-------------|--------------------|-|
| `thread::spawn()` | Can't create threads at runtime | Launch with max threads upfront; idle threads poll for work |
| `Mutex::lock()` | Would deadlock 1000s of threads | Use atomics (`atomic_compare_exchange`) |
| `channel::send()` | No runtime thread creation | Ring buffer + atomic head/tail pointers |
| `thread::sleep()` | No blocking syscalls | Spin-wait with memory barrier + backoff |
| `thread::join()` | Threads end when kernel ends | Persistent kernels don't end; use atomic barriers |
| `malloc()` | No dynamic allocation | Pre-allocate pool; GPU manages via atomics |
| `open()` / `read()` | No syscalls from GPU | MTLIOCommandQueue (GPU-initiated I/O) |
| Network I/O | No socket access | CPU feeder thread + shared ring buffer |

## Mental Model Shift

### CPU: "I have a task, let me parallelize it"
```rust
let pool = ThreadPool::new(8);
for item in items {
    pool.execute(|| process(item));
}
```

### GPU: "I have 10,000 workers, what should each do?"
```rust
fn main() {
    let my_item = items[thread_id()];
    process(my_item);
}
```

## Example: Full Parallel Sum

```rust
#![no_std]
use gpu_sdk::prelude::*;

const ARRAY_SIZE: i32 = 1_000_000;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();

    // Only process valid indices
    if tid >= ARRAY_SIZE {
        return 0;
    }

    // Read my element
    let my_value = read_state(tid);

    // Phase 1: Local reduction within threadgroup
    // (Uses shared memory, faster than global atomics)
    write_shared(local_id(), my_value);
    threadgroup_barrier();

    // Tree reduction within group
    let mut stride = threadgroup_size() / 2;
    while stride > 0 {
        if local_id() < stride {
            let a = read_shared(local_id());
            let b = read_shared(local_id() + stride);
            write_shared(local_id(), a + b);
        }
        threadgroup_barrier();
        stride /= 2;
    }

    // Phase 2: First thread in each group adds to global sum
    if local_id() == 0 {
        let group_sum = read_shared(0);
        atomic_add(0, group_sum);  // state[0] = final sum
    }

    0
}
```

## Persistent Kernel Execution

**Key Insight: GPU kernels CAN run indefinitely.**

Traditional GPU programming assumes short-lived kernels: launch, compute, terminate. But for a GPU-native OS, we need **persistent kernels** that run like daemons.

### Event-Driven Architecture

```rust
#![no_std]
use gpu_sdk::prelude::*;

const EVENT_NONE: i32 = 0;
const EVENT_INPUT: i32 = 1;
const EVENT_TIMER: i32 = 2;
const EVENT_SHUTDOWN: i32 = -1;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();

    // Persistent main loop
    loop {
        // Poll event queue (atomic read from shared memory)
        let event = atomic_load(EVENT_QUEUE_HEAD);

        if event == EVENT_SHUTDOWN {
            break;  // Graceful termination
        }

        if event == EVENT_NONE {
            // Yield to avoid burning power
            // (Memory barrier + short spin)
            memory_barrier();
            continue;
        }

        // Process event
        match event {
            EVENT_INPUT => handle_input(tid),
            EVENT_TIMER => handle_timer(tid),
            _ => {}
        }

        // Clear event (atomic)
        atomic_compare_exchange(EVENT_QUEUE_HEAD, event, EVENT_NONE);
    }

    0
}
```

### Poll Memory Atomics

The GPU polls memory locations that external sources (CPU, other GPUs) can write to:

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Memory Layout                     │
├─────────────────────────────────────────────────────────┤
│  [0x0000] Event Queue Head (atomic)                     │
│  [0x0004] Event Queue Tail (atomic)                     │
│  [0x0008] Event Ring Buffer [256 events]                │
│  [0x0408] Timer Tick Counter (atomic)                   │
│  [0x040C] Shutdown Flag (atomic)                        │
│  [0x0410] ... Application State ...                     │
└─────────────────────────────────────────────────────────┘
```

**CPU's only job**: Write events to the queue. GPU handles everything else.

### No CPU Dispatch Overhead After Initial Launch

Once a persistent kernel is running:
- **Zero kernel launch latency** (already running)
- **Zero CPU command buffer encoding** (no new dispatches)
- **Zero CPU-GPU synchronization** (GPU polls memory directly)

The CPU becomes a peripheral that occasionally injects events, not a coordinator.

### Power Considerations

GPU threads spinning on memory reads consume power. Mitigation strategies:

1. **Exponential backoff**: Spin for N cycles, then longer pauses
2. **Cooperative yielding**: Use `simd_active_threads_mask()` to reduce active lanes
3. **Tiered polling**: Fast path for active state, slow path for idle
4. **Frame-aligned wake**: Sync with VSync for UI workloads

---

## I/O Threading Model

GPU-native I/O requires different patterns than CPU-centric approaches.

### File I/O: MTLIOCommandQueue (GPU-Initiated)

Metal 3 introduced `MTLIOCommandQueue` for GPU-initiated file loading:

```rust
// Setup (one-time, on CPU)
let io_queue = device.makeIOCommandQueue(descriptor);
let file_handle = MTLIOFileHandle(url: file_url);

// GPU kernel requests file load
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();

    // Request async file read (GPU-initiated)
    let request_slot = atomic_fetch_add(IO_REQUEST_HEAD, 1);
    write_io_request(request_slot, IORequest {
        file_id: my_file_id,
        offset: tid * CHUNK_SIZE,
        length: CHUNK_SIZE,
        dest_buffer_offset: tid * CHUNK_SIZE,
    });

    // Continue other work while I/O happens async
    // ...

    // Poll for completion
    while atomic_load(IO_COMPLETION_FLAGS + tid) == 0 {
        do_other_work();
    }

    // Data is now in buffer
    let data = read_state(DEST_BUFFER + tid * CHUNK_SIZE);
    process(data);

    0
}
```

**Key benefit**: The GPU initiates I/O without CPU involvement per-request. CPU only handles the MTLIOCommandQueue setup once.

### Network I/O: CPU Thread + Ring Buffer + GPU Polling

Network I/O cannot (yet) be fully GPU-initiated. The model is:

```
┌────────────────────────────────────────────────────────────┐
│                    Network I/O Pipeline                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    Ring Buffer    ┌────────────────────┐   │
│   │ CPU Net  │ ───────────────► │   GPU Persistent    │   │
│   │ Thread   │    (lockfree)     │   Kernel Polling    │   │
│   └──────────┘                   └────────────────────┘   │
│        ▲                                   │               │
│        │                                   ▼               │
│   ┌──────────┐                   ┌────────────────────┐   │
│   │ Network  │                   │   GPU Processing    │   │
│   │ Socket   │                   │   (parallel)        │   │
│   └──────────┘                   └────────────────────┘   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

```rust
// CPU thread (minimal - just moves bytes)
fn network_thread(ring_buffer: &mut RingBuffer) {
    loop {
        let packet = socket.recv();  // CPU required: socket syscall
        let slot = ring_buffer.push(packet);  // Lock-free write
        atomic_store(PACKET_READY_FLAG + slot, 1);  // Signal GPU
    }
}

// GPU persistent kernel
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();

    loop {
        // Poll for new packets
        let packet_slot = tid % RING_BUFFER_SIZE;
        if atomic_load(PACKET_READY_FLAG + packet_slot) == 1 {
            // Process packet (parallel across all threads)
            let packet = read_ring_buffer(packet_slot);
            process_packet(packet);

            // Clear flag
            atomic_store(PACKET_READY_FLAG + packet_slot, 0);
        }

        memory_barrier();
    }
}
```

**CPU involvement**: Socket syscalls only (recv/send). All packet parsing, protocol handling, and response generation on GPU.

### Storage: Direct GPU Access via Async I/O

For maximum performance, bypass filesystem entirely:

```rust
// Direct block device access (requires elevated privileges)
let block_device = open_block_device("/dev/disk0s1");
let buffer = device.makeBuffer(length: SECTOR_SIZE * SECTOR_COUNT);

// Map buffer for direct I/O (zero-copy)
let mapped = mmap(buffer, O_DIRECT);

// GPU kernel reads blocks directly
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();
    let sector = tid;

    // Read sector (via MTLIOCommandQueue or mapped buffer)
    let data = read_sector(sector);

    // GPU parses filesystem structures directly
    let inode = parse_inode(data);
    let block_pointers = parse_block_pointers(inode);

    // Continue reading file data
    // ...

    0
}
```

### I/O Threading Summary

| I/O Type | CPU Involvement | GPU Role |
|----------|-----------------|----------|
| **File Read (MTLIOCommandQueue)** | Initial queue setup only | Initiates requests, processes data |
| **File Read (mmap)** | Initial mmap setup only | Direct memory access |
| **Network Recv** | Socket syscall (recv) | Packet parsing, protocol handling |
| **Network Send** | Socket syscall (send) | Packet construction, serialization |
| **Block Storage** | Initial device open only | Direct sector read/write |
| **Keyboard/Mouse** | HID driver interrupt | Event processing, state updates |

---

## Summary

| `std::thread` Concept | GPU Equivalent |
|-----------------------|----------------|
| `thread::spawn(f)` | Just launch more threads |
| `thread::current().id()` | `thread_id()` |
| `thread::join()` | Kernel completion |
| Mutex | Atomics |
| Channel | Shared buffer + atomics |
| Thread pool | The GPU IS a massive thread pool |
| Event loop | Persistent kernel + memory polling |
| Network thread | CPU feeder thread + GPU ring buffer |
| File I/O thread | MTLIOCommandQueue (GPU-initiated) |
| Network I/O | CPU receives, GPU processes via ring buffer |
| Process | Threadgroup with dedicated state buffer |
| Fork | Duplicate state buffer, spawn threads |
| Exec | Load different bytecode, same threads |
| IPC | Shared memory region with atomics |

**You don't spawn threads. You ARE the threads.**

---

## CPU Involvement Breakdown

For clarity, here's exactly what still requires CPU and why:

| Operation | Why CPU Required | GPU Alternative (Future) |
|-----------|------------------|--------------------------|
| Kernel launch | Metal API is CPU-driven | Persistent kernels eliminate re-launch |
| Socket syscalls | OS network stack is CPU-only | Ring buffer pattern (CPU receives, GPU processes) |
| HID interrupts | USB/Bluetooth drivers are CPU | Direct HID buffer access (research) |
| MTLIOCommandQueue setup | Metal API limitation | Keep queue persistent |
| Process creation | **None (see GPU-Native Process Model)** | Atomic allocation + bytecode dispatch |
| Memory allocation | Metal buffer API is CPU | Pre-allocate large pools, GPU manages internally |

**Goal**: Minimize CPU to one-time setup. Steady-state operation should be 100% GPU.

---

## GPU-Native Process Model

GPU threads can be organized as "processes" - providing isolation, resource management, and familiar OS abstractions without any CPU involvement.

### Mapping OS Concepts to GPU

| OS Concept | GPU Equivalent |
|------------|----------------|
| Process | Threadgroup with dedicated state buffer |
| Thread | GPU thread (we have 10,000+) |
| Fork | Duplicate state buffer, spawn threads |
| Exec | Load different bytecode, same threads |
| IPC | Shared memory region with atomics |
| Scheduling | GPU hardware scheduler |

### Process Structure

Each GPU "process" maintains:

```rust
#[repr(C)]
struct GpuProcess {
    // Identity
    pid: u32,                    // Process ID (unique)
    parent_pid: u32,             // Parent process ID

    // Execution
    bytecode_offset: u32,        // Offset into bytecode buffer
    bytecode_length: u32,        // Size of bytecode
    entry_point: u32,            // Entry function offset

    // Memory
    state_buffer_offset: u32,    // Offset into global state pool
    state_buffer_size: u32,      // Allocated state size
    heap_region_start: u32,      // Heap region offset
    heap_region_size: u32,       // Heap region size

    // Threads
    thread_range_start: u32,     // First thread ID assigned
    thread_range_count: u32,     // Number of threads

    // Status
    status: u32,                 // Running=0, Stopped=1, Zombie=2
    exit_code: i32,              // Exit code when terminated

    // Relationships
    first_child: u32,            // First child PID (linked list)
    next_sibling: u32,           // Next sibling PID
}
```

### Process Table (GPU-Resident)

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Process Table                             │
├─────────────────────────────────────────────────────────────────┤
│  [0x0000] Next PID Counter (atomic)                             │
│  [0x0004] Active Process Count (atomic)                         │
│  [0x0008] Free Slot Bitmap [64 bytes = 512 processes]           │
│  [0x0048] Process Array [512 × sizeof(GpuProcess)]              │
│  ...                                                             │
│  [0x8000] State Buffer Pool [16MB]                              │
│  [0x808000] Bytecode Pool [8MB]                                 │
│  [0xC08000] Heap Pool [32MB]                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Spawning a Process (All on GPU)

**Critical**: No CPU involvement in steady-state process creation.

```rust
#[no_mangle]
pub extern "C" fn spawn_process(bytecode_id: i32, initial_state: i32) -> i32 {
    let tid = thread_id();

    // Only one thread per spawn request
    if tid != 0 { return -1; }

    // Step 1: Allocate from process table (atomic)
    let slot = allocate_process_slot();  // Atomic bitmap scan
    if slot < 0 { return -1; }  // No free slots

    // Step 2: Assign PID (atomic increment)
    let pid = atomic_fetch_add(NEXT_PID_ADDR, 1);

    // Step 3: Allocate state buffer from pool
    let state_offset = allocate_state_buffer(DEFAULT_STATE_SIZE);
    if state_offset < 0 { return -1; }

    // Step 4: Copy/reference bytecode
    let bytecode_offset = lookup_bytecode(bytecode_id);

    // Step 5: Reserve thread range
    let thread_start = atomic_fetch_add(NEXT_THREAD_RANGE, THREADS_PER_PROCESS);

    // Step 6: Initialize process structure
    write_process(slot, GpuProcess {
        pid,
        parent_pid: current_pid(),
        bytecode_offset,
        state_buffer_offset: state_offset,
        thread_range_start: thread_start,
        thread_range_count: THREADS_PER_PROCESS,
        status: PROCESS_RUNNING,
        exit_code: 0,
        // ... other fields
    });

    // Step 7: Dispatch - threads in range will pick up new process
    // (Persistent kernel threads poll process table)
    atomic_fetch_add(ACTIVE_PROCESS_COUNT, 1);

    pid
}
```

### Thread-to-Process Binding

Persistent kernel threads poll the process table to find work:

```rust
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();

    loop {
        // Find which process this thread belongs to
        let process = find_process_for_thread(tid);

        if process.is_none() {
            // No work - yield
            memory_barrier();
            continue;
        }

        let proc = process.unwrap();

        // Check process status
        if proc.status != PROCESS_RUNNING {
            continue;
        }

        // Calculate local thread ID within process
        let local_tid = tid - proc.thread_range_start;

        // Execute process bytecode
        execute_bytecode(
            proc.bytecode_offset,
            proc.state_buffer_offset,
            local_tid
        );
    }
}
```

### Fork (GPU-Native)

```rust
pub fn fork() -> i32 {
    let parent = current_process();

    // Allocate new process slot
    let child_slot = allocate_process_slot();
    let child_pid = atomic_fetch_add(NEXT_PID_ADDR, 1);

    // Allocate and COPY state buffer (not share)
    let child_state = allocate_state_buffer(parent.state_buffer_size);
    copy_memory(
        parent.state_buffer_offset,
        child_state,
        parent.state_buffer_size
    );

    // Same bytecode, different state
    write_process(child_slot, GpuProcess {
        pid: child_pid,
        parent_pid: parent.pid,
        bytecode_offset: parent.bytecode_offset,  // Share bytecode
        state_buffer_offset: child_state,          // Separate state
        thread_range_start: allocate_thread_range(),
        status: PROCESS_RUNNING,
        // ...
    });

    // Return 0 to child, child_pid to parent
    // (Determined by which thread range we're in)
    if in_child_thread_range() { 0 } else { child_pid }
}
```

### Exec (GPU-Native)

```rust
pub fn exec(new_bytecode_id: i32) -> i32 {
    let proc = current_process_mut();

    // Load new bytecode reference
    let new_bytecode = lookup_bytecode(new_bytecode_id);
    if new_bytecode < 0 { return -1; }

    // Clear state buffer (fresh start)
    zero_memory(proc.state_buffer_offset, proc.state_buffer_size);

    // Update bytecode pointer - same threads, new program
    proc.bytecode_offset = new_bytecode;
    proc.entry_point = 0;  // Reset to entry

    // Threads will pick up new bytecode on next iteration
    0
}
```

### IPC via Shared Memory

Processes communicate through explicitly shared memory regions:

```rust
// Create shared memory region
pub fn shm_create(size: u32) -> i32 {
    let shm_id = atomic_fetch_add(NEXT_SHM_ID, 1);
    let offset = allocate_shm_region(size);

    register_shm(shm_id, offset, size, current_pid());
    shm_id
}

// Attach to shared memory
pub fn shm_attach(shm_id: i32) -> i32 {
    let shm = lookup_shm(shm_id);
    // Map into process's address space (just return offset)
    shm.offset
}

// IPC via atomics on shared memory
pub fn send_message(shm_offset: i32, msg: i32) {
    // Ring buffer with atomic head/tail
    let slot = atomic_fetch_add(shm_offset + HEAD_OFFSET, 1) % RING_SIZE;
    write_state(shm_offset + DATA_OFFSET + slot, msg);
    atomic_fetch_add(shm_offset + COUNT_OFFSET, 1);
}
```

### Scheduling: Hardware Does It

**Key insight**: GPU hardware scheduler handles thread scheduling automatically.

- No software scheduler needed
- Threadgroups are scheduled by GPU hardware
- Processes are "scheduled" by which threadgroup ranges they own
- Priority can be influenced by thread count allocation

```
Process A: 1024 threads  → More GPU time
Process B: 256 threads   → Less GPU time
Process C: 64 threads    → Minimal GPU time (background)
```

### Process Lifecycle (Zero CPU)

```
┌─────────────────────────────────────────────────────────────────┐
│                  GPU-Native Process Lifecycle                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. SPAWN                                                       │
│      ├─ Allocate slot (atomic bitmap)                           │
│      ├─ Assign PID (atomic counter)                             │
│      ├─ Allocate state buffer (atomic pool)                     │
│      ├─ Reference bytecode                                       │
│      └─ Reserve thread range (atomic counter)                   │
│                                                                  │
│   2. RUN                                                         │
│      ├─ Threads poll process table                              │
│      ├─ Find assigned process                                    │
│      ├─ Execute bytecode                                         │
│      └─ Access state buffer                                      │
│                                                                  │
│   3. FORK                                                        │
│      ├─ Allocate new slot + PID                                 │
│      ├─ COPY state buffer (not share)                           │
│      ├─ Share bytecode reference                                 │
│      └─ Reserve new thread range                                 │
│                                                                  │
│   4. EXEC                                                        │
│      ├─ Load new bytecode reference                             │
│      ├─ Clear state buffer                                       │
│      └─ Same threads, new program                                │
│                                                                  │
│   5. EXIT                                                        │
│      ├─ Set status = ZOMBIE                                      │
│      ├─ Store exit code                                          │
│      ├─ Release thread range                                     │
│      └─ Parent collects (wait)                                   │
│                                                                  │
│   6. REAP                                                        │
│      ├─ Parent reads exit code                                   │
│      ├─ Release state buffer                                     │
│      └─ Free process slot                                        │
│                                                                  │
│   ** ALL STEPS EXECUTE ON GPU - ZERO CPU INVOLVEMENT **          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Is TRUE Parallelism

Traditional OS processes on CPU:
- Time-sliced by OS scheduler
- Context switches are expensive
- "Parallel" often means interleaved

GPU processes:
- **Actually parallel** - threads execute simultaneously
- No context switching - hardware manages execution
- 10,000+ threads = 10,000+ actually concurrent executions

This isn't emulating Unix processes on GPU. This is recognizing that **GPU threads ARE the parallel execution units** that Unix processes were trying to abstract over limited CPU cores.
