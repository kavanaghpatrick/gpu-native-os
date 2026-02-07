# Rust to GPU: Complete Roadmap

## THE GPU IS THE COMPUTER

This document maps the journey from "Rust doesn't work on GPU" to "write **UNMODIFIED** Rust, runs on GPU."

**Core Principle**: The user writes normal Rust code. We do ALL the transformation work.

---

## Hardware Capabilities (Apple Silicon)

Apple Silicon's architecture fundamentally changes what's possible for GPU-native computing. These aren't theoretical features - they're shipping APIs we can use today.

### MTLIOCommandQueue: GPU-Initiated File I/O

Metal 3 introduced `MTLIOCommandQueue` for direct GPU-to-storage data paths:

```swift
// GPU requests file data directly - no CPU involvement
let ioQueue = device.makeIOCommandQueue(descriptor: ioDescriptor)!
let cmdBuffer = ioQueue.makeCommandBuffer()
cmdBuffer.load(buffer, offset: 0, size: fileSize, sourceHandle: fileHandle, sourceOffset: 0)
cmdBuffer.commit()
```

**What this means:**
- GPU compute kernel can trigger file loads asynchronously
- Data streams directly to GPU buffer (unified memory makes this zero-copy)
- CPU is not in the data path - only sets up the request
- Enables GPU-driven filesystem browsing, asset loading, database queries

**Current status:** Available in Metal 3+ (macOS 13+, iOS 16+)

### Unified Memory Architecture (Zero-Copy)

Apple Silicon's unified memory eliminates the CPU-GPU copy overhead that plagues discrete GPUs:

```rust
// Traditional discrete GPU workflow:
cpu_buffer → [PCI-e copy 16GB/s] → gpu_buffer  // ~60ms for 1GB

// Apple Silicon unified memory:
buffer  // Same physical memory, both CPU and GPU access
        // Zero copy, zero latency for "transfers"
```

**Implications:**
- `newBufferWithBytesNoCopy()` for true zero-copy buffer sharing
- mmap'd files are instantly GPU-accessible
- 400GB/s memory bandwidth shared between CPU and GPU
- No "upload to GPU" step - data is already there

### Persistent Kernel Execution

Metal supports persistent compute kernels that run indefinitely:

```metal
// Kernel stays resident, processes events as they arrive
kernel void persistent_event_loop(
    device EventQueue* events [[buffer(0)]],
    device atomic_uint* event_count [[buffer(1)]]
) {
    while (true) {
        // Wait for new events (spin on atomic)
        uint count = atomic_load_explicit(event_count, memory_order_acquire);
        if (count > last_processed) {
            process_event(events[count - 1]);
            last_processed = count;
        }
        // Yield to other threadgroups
        threadgroup_barrier(mem_flags::mem_none);
    }
}
```

**Enables:**
- GPU-resident event loop (input, timers, I/O completion)
- No CPU dispatch overhead per frame
- True GPU-as-main-processor architecture

---

## The Original Problem

```rust
use std::*;             // No std on GPU
let v = Vec::new();     // Needs allocator
let s = String::new();  // Needs allocator
println!("...");        // Needs I/O
std::thread::spawn();   // No thread spawning
async fn fetch();       // No async runtime
TcpStream::connect();   // No network
```

## The Complete Solution

| Problem | Solution | Phase | Status |
|---------|----------|-------|--------|
| Basic Rust ops | WASM-GPU bytecode | 4 | Done |
| Helper functions | Function inlining | 5 | #178 |
| Math (sin, cos) | GPU intrinsics | 5 | #178 |
| `Vec`, `String`, `Box` | GPU allocator | 6 | #179 |
| `println!`, `dbg!` | GPU debug buffer | 7 | #180 |
| `thread::spawn()` | Work queue dispatch | 8 | #182 |
| `async/await` | Parallel dispatch | 8 | #182 |
| `TcpStream`, networking | Request queue | 8 | #182 |
| `Mutex`, `Condvar` | Atomics + barriers | 8 | #182 |
| `Rc<T>` | Atomic refcount | 8 | #182 |
| `thread::sleep` | Frame timing | 8 | #182 |
| `fs::read/write` | MTLIOCommandQueue | 8 | #182 |
| `Command::spawn()` | GPU-native spawning | 8 | #182 |

---

## Phase 4: Basic Translation - COMPLETE

**What works now:**
```rust
#![no_std]

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    let mut sum = 0;
    for i in 1..=n {
        sum += i;
    }
    sum
}
```

- All i32/f32 arithmetic
- Control flow (if, loop, while)
- Local variables
- Memory access

---

## Phase 5: Function Calls (#178)

**What it enables:**
```rust
#![no_std]

fn square(x: i32) -> i32 { x * x }
fn cube(x: i32) -> i32 { x * x * x }

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    square(n) + cube(n)
}
```

Plus GPU intrinsics:
```rust
use gpu_sdk::{thread_id, set_pixel, time};

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let x = thread_id() % 800;
    let y = thread_id() / 800;
    let t = time();

    let r = (x as f32 / 800.0 + t).sin() * 0.5 + 0.5;
    set_pixel(x, y, r, 0.0, 0.0, 1.0);
    0
}
```

---

## Phase 6: GPU Allocator (#179)

**What it enables:**
```rust
#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::boxed::Box;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut numbers = Vec::new();
    numbers.push(1);
    numbers.push(2);
    numbers.push(3);

    let sum: i32 = numbers.iter().sum();

    let greeting = String::from("Hello GPU!");
    let boxed = Box::new(42);

    sum + *boxed  // 48
}
```

---

## Phase 7: Debug I/O (#180)

**What it enables:**
```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    gpu_println!("Computing sum of 1 to n");

    let mut sum = 0;
    for i in 1..=n {
        sum += i;
        gpu_dbg!(sum);  // "main.rs:10: sum = 55"
    }

    gpu_println!("Result: ");
    print_i32(sum);

    sum
}
```

---

## Phase 8: Automatic Code Transformation (#182)

**This is the key phase.** We automatically transform "incompatible" patterns into GPU equivalents. **Zero user code changes.**

### async/await - Parallel Dispatch
```rust
// User writes this (UNCHANGED)
async fn fetch_data(id: u32) -> Data {
    expensive_compute(id).await
}

// We automatically transform to parallel work queue
```

### TcpStream - Request Queue
```rust
// User writes this (UNCHANGED)
let mut stream = TcpStream::connect("api.example.com:80")?;
stream.write_all(b"GET /")?;

// We automatically queue request, CPU processes after dispatch
```

### Condvar - Threadgroup Barrier
```rust
// User writes this (UNCHANGED)
cvar.wait(guard)?;
cvar.notify_all();

// We automatically use threadgroup_barrier()
```

### Rc<T> - Atomic Refcount
```rust
// User writes this (UNCHANGED)
let data = Rc::new(42);
let clone = Rc::clone(&data);

// We automatically use atomic operations
```

### thread::sleep - Frame Timing
```rust
// User writes this (UNCHANGED)
thread::sleep(Duration::from_millis(100));

// We automatically use frame-based timing
```

### Mutex - Atomic Spinlock
```rust
// User writes this (UNCHANGED)
let guard = mutex.lock().unwrap();
*guard += 1;

// We automatically use atomic spinlock (with perf warning if high contention)
```

### fs::read - MTLIOCommandQueue
```rust
// User writes this (UNCHANGED)
let content = fs::read_to_string("data.txt")?;

// We automatically use MTLIOCommandQueue for GPU-direct file I/O
// Data streams directly to GPU buffer, zero CPU involvement
```

### Command::spawn - GPU-Native Process Spawning
```rust
// User writes this (UNCHANGED)
let output = Command::new("my_gpu_program")
    .args(["--input", "data.bin"])
    .output()?;

// We transform this to GPU-native process spawning:
// 1. Allocate state buffer (isolated memory region for new process)
// 2. Load bytecode (WASM→GPU compiled program)
// 3. Reserve thread group (hardware execution units)
// 4. Dispatch (GPU scheduler activates new process)
```

**Key insight:** GPU threads ARE processes. We already have:
- **Thousands of parallel execution units** (GPU threads/wavefronts)
- **Isolated state buffers** (GPU memory regions per process)
- **GPU bytecode** (our WASM→GPU compiler output)
- **GPU scheduler** (hardware + software scheduling)

**Process spawning = allocate state buffer + load bytecode + reserve threads + dispatch**

```metal
// GPU-native process spawning (simplified)
kernel void spawn_process(
    device ProcessTable* processes [[buffer(0)]],
    device BytecodeStore* bytecode [[buffer(1)]],
    device atomic_uint* next_pid [[buffer(2)]]
) {
    uint pid = atomic_fetch_add_explicit(next_pid, 1, memory_order_relaxed);

    // Allocate isolated memory region for new process
    processes[pid].heap_base = allocate_region(PROCESS_HEAP_SIZE);
    processes[pid].stack_base = allocate_region(PROCESS_STACK_SIZE);

    // Load bytecode entry point
    processes[pid].instruction_ptr = bytecode[program_id].entry;
    processes[pid].state = PROCESS_READY;

    // GPU scheduler will dispatch this process to available threads
}
```

**No CPU involvement.** Process spawning is purely GPU-native.

**See**: `docs/PRD_AUTOMATIC_CODE_TRANSFORMATION.md`

---

## The Final State

After all phases, **normal Rust code** runs on GPU with full hardware acceleration:

```rust
// NO #![no_std] NEEDED!
use std::collections::HashMap;
use std::sync::Mutex;
use std::fs;

fn main() {
    // Dynamic allocation - WORKS (GPU allocator)
    let mut map = HashMap::new();
    map.insert("key", 42);

    // Synchronization - WORKS (atomic spinlock)
    let counter = Mutex::new(0);
    *counter.lock().unwrap() += 1;

    // Debug output - WORKS (GPU debug buffer)
    println!("Map: {:?}", map);
    dbg!(counter);

    // File I/O - WORKS (MTLIOCommandQueue, zero-copy)
    let data = fs::read("large_dataset.bin").unwrap();
    // Data streams directly to GPU buffer at SSD speed

    // Networking - WORKS (request queue to CPU helper)
    // let response = reqwest::get("http://api.com").await?;
}
```

**The GPU is the computer. CPU is just a boot loader.**

---

## What Truly Can't Work (Fundamental Hardware Limits)

| Feature | Why | Mitigation |
|---------|-----|------------|
| Direct network packet reception | NIC hardware delivers packets to CPU memory | Request queue pattern; CPU forwards to GPU buffer |

**That's it.** One item. Everything else is GPU-native:

| Feature | How It Works |
|---------|--------------|
| File I/O | MTLIOCommandQueue (GPU-initiated, zero-copy) |
| Process spawning | **GPU-native** (allocate state buffer + load bytecode + reserve threads + dispatch) |
| Environment variables | Snapshot at load time to GPU buffer |
| Dynamic library loading | Static linking (compile-time) |
| Networking | CPU receives → GPU buffer → GPU processes (NIC hardware limitation) |

**Key insight:** GPU threads ARE processes. Process spawning is entirely GPU-native:
- Allocate isolated memory region (process heap/stack)
- Load compiled bytecode (WASM→GPU)
- Reserve thread group (execution units)
- Dispatch via GPU scheduler

The ONLY thing requiring CPU is **network packet reception** (NIC hardware sends packets to CPU memory, not GPU memory). CPU's only job: receive packets, copy to GPU buffer. GPU does everything else.

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 5 - Functions | Medium | High (real programs) | **NOW** |
| 6 - Allocator | Medium | High (Vec/String) | Next |
| 7 - Debug I/O | Low | Medium (debugging) | After 6 |
| 8 - Auto Transform | High | **CRITICAL** (seamless UX) | After 7 |

---

## Success Criteria

**Phase 5 Complete When:**
- [ ] Helper functions inline correctly
- [ ] Recursion detected and rejected
- [ ] `sin()`, `cos()`, `sqrt()` work
- [ ] `set_pixel()` renders output
- [ ] `thread_id()` returns correct value

**Phase 6 Complete When:**
- [ ] `Vec::new()` compiles and runs
- [ ] `Vec::push()` grows correctly
- [ ] Drop frees memory (no leaks)
- [ ] `String` works
- [ ] `Box` works

**Phase 7 Complete When:**
- [ ] `gpu_println!` outputs to buffer
- [ ] `gpu_dbg!` shows file:line
- [ ] Screen overlay renders text
- [ ] Test capture works

**Phase 8 Complete When:**
- [ ] `async/await` auto-transforms to parallel dispatch
- [ ] `TcpStream` auto-transforms to request queue
- [ ] `Condvar` auto-transforms to barrier
- [ ] `Rc<T>` auto-transforms to atomic refcount
- [ ] `thread::sleep` auto-transforms to frame timing
- [ ] `Mutex` auto-transforms to spinlock
- [ ] `fs::read` uses MTLIOCommandQueue for GPU-direct I/O
- [ ] Performance warnings for high-contention patterns
- [ ] Persistent kernel event loop operational (no CPU dispatch per frame)

---

## The User Experience

**Before (other GPU compilers):**
```
ERROR: async/await not supported
ERROR: TcpStream not available
ERROR: Condvar requires OS threads
HINT: Rewrite your entire codebase...
```

**After (our system):**
```
$ cargo build --target wasm32-unknown-unknown
   Compiling my_app v0.1.0
    Finished release [optimized]

$ gpu-run my_app.wasm
Running on GPU with 10240 threads...
[MTLIOCommandQueue] Loaded 1.2GB dataset in 340ms
[Persistent Kernel] Event loop active, 0.1% CPU usage
Result: 42
```

**Zero errors. Zero rewrites. Just works.**

---

## Links

- Phase 5 PRD: `docs/PRD_PHASE5_WASM_FUNCTION_CALLS.md`
- Phase 6 PRD: `docs/PRD_PHASE6_GPU_ALLOCATOR.md`
- Phase 7 PRD: `docs/PRD_PHASE7_GPU_DEBUG_IO.md`
- Phase 8 PRD: `docs/PRD_AUTOMATIC_CODE_TRANSFORMATION.md`
- Threading Model: `docs/PRD_GPU_THREADING_MODEL.md`
- Complete std Coverage: `docs/STD_COMPLETE_COVERAGE.md`
- GitHub Issues: #178, #179, #180, #181, #182
