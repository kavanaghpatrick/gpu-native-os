# PRD: Automatic Code Transformation

## THE GPU IS THE COMPUTER

**Principle**: The user writes normal Rust. Our translator does ALL the work.

---

## The Problem

Current approach shows error messages:
```
ERROR: async/await not supported - use parallel dispatch instead
ERROR: TcpStream not available - queue network requests instead
```

**This is wrong.** The user shouldn't need to change their code.

---

## The Solution: Automatic Transformation

Our WASM→GPU translator automatically transforms incompatible patterns into GPU-native equivalents.

### Transformation Matrix

| User Writes | WASM Pattern | GPU Translation | User Experience |
|-------------|--------------|-----------------|-----------------|
| `async fn` | Future poll loop | Parallel dispatch | **Just works** |
| `await` | Future::poll | Work queue + barrier | **Just works** |
| `TcpStream::connect()` | Call to std::net | CPU request queue* | **Just works** |
| `Condvar::wait()` | Thread block | Threadgroup barrier | **Just works** |
| `Rc<T>` | Non-atomic refcount | Atomic refcount | **Just works** |
| `thread::sleep(d)` | syscall | Frame counter wait | **Just works** |
| `Mutex::lock()` | OS mutex | Atomic spinlock | **Just works** |
| `fs::read()` | syscall | MTLIOCommandQueue | **Just works** |
| `Command::new()` | Process spawn | **GPU_IMPL** | **Just works** |

*Network I/O requires CPU due to NIC hardware (see Research Findings section)

---

## GPU-Native Process Model

**Core Insight: GPU threads ARE processes.** Traditional CPU process spawning uses syscalls to create isolated address spaces and schedule execution. On GPU, we achieve the same semantics using native GPU primitives:

| CPU Concept | GPU-Native Equivalent |
|-------------|----------------------|
| Process address space | Isolated state buffer region |
| Process ID | Thread range allocation |
| fork()/exec() | Bytecode lookup + thread dispatch |
| Process scheduler | GPU thread scheduler (hardware) |
| Inter-process communication | Shared memory regions + atomics |
| Process exit | Completion flag + thread range release |

### Why This Works

1. **GPU threads are hardware-scheduled** - No OS involvement needed
2. **GPU memory allocator provides isolation** - Each process gets its own buffer
3. **Pre-compiled bytecode eliminates exec()** - All programs compiled ahead of time
4. **Persistent sub-kernels enable true parallelism** - Multiple processes run concurrently

### Process Lifecycle (GPU-Native)

```
1. SPAWN REQUEST
   - Look up bytecode module by name ("ls", "grep", etc.)
   - Allocate state buffer from GPU memory pool
   - Reserve thread range from GPU scheduler
   - Initialize process control block

2. EXECUTION
   - Threads in range execute bytecode VM
   - Each thread has isolated registers + shared state buffer
   - Process runs until completion or yield

3. COMPLETION
   - Thread 0 writes completion flag
   - Output copied to parent's buffer
   - Thread range released to scheduler
   - State buffer freed
```

### Comparison with Traditional CPU Process Spawning

```
TRADITIONAL (CPU syscall):
  User code --> fork() syscall --> kernel --> copy page tables -->
  exec() syscall --> kernel --> load binary --> schedule --> run

GPU-NATIVE:
  User code --> lookup bytecode --> allocate state --> dispatch threads --> run

No syscalls. No kernel involvement. No binary loading. Just dispatch.
```

### I/O Classification Summary

| Operation | Implementation | CPU Required? |
|-----------|---------------|---------------|
| **Process spawn** | GPU_IMPL (thread dispatch) | NO |
| **File I/O** | MTLIOCommandQueue | NO (GPU-initiated) |
| **Network I/O** | CPU request queue | YES (NIC hardware) |
| **Memory alloc** | GPU allocator | NO |
| **Thread sync** | Atomics + barriers | NO |

---

## Transformation 1: async/await → Parallel Dispatch

### User Code (Unchanged)
```rust
async fn fetch_data(id: u32) -> Data {
    let result = expensive_compute(id).await;
    process(result).await
}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let data = block_on(fetch_data(42));
    data.value
}
```

### What WASM Shows
```wasm
;; async fn compiles to a state machine
(func $fetch_data (param $id i32) (result i32)
  ;; State machine with poll() calls
  (local $state i32)
  (block $done
    (loop $poll
      ;; Check state, do work, yield
    )
  )
)
```

### GPU Translation
```metal
// Detect: async state machine pattern
// Transform: parallel work items

struct WorkItem {
    uint id;
    uint state;
    int result;
};

kernel void async_dispatch(
    device WorkItem* work_queue,
    device atomic_uint* queue_count,
    uint gid [[thread_position_in_grid]]
) {
    // Each thread picks up a work item
    uint my_item = gid;
    if (my_item >= atomic_load_explicit(queue_count, memory_order_relaxed)) return;

    WorkItem item = work_queue[my_item];

    // Execute the async function's body directly
    // No state machine - just run it
    int result = expensive_compute(item.id);
    result = process(result);

    work_queue[my_item].result = result;
}
```

### Detection Heuristics
1. Function returns `Future` or `Poll`
2. Contains `poll()` calls in loop
3. Has state machine pattern (local `$state` variable, branching table)

### Why This Works
On GPU, you don't need async because:
- You already have 10,000+ threads running in parallel
- "await" points become natural thread boundaries
- The work queue IS the async runtime

---

## Transformation 2: TcpStream → Request Queue

### User Code (Unchanged)
```rust
use std::net::TcpStream;
use std::io::{Read, Write};

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut stream = TcpStream::connect("api.example.com:80").unwrap();
    stream.write_all(b"GET / HTTP/1.1\r\n\r\n").unwrap();

    let mut buffer = [0u8; 1024];
    let n = stream.read(&mut buffer).unwrap();
    n as i32
}
```

### WASM Pattern
```wasm
;; TcpStream::connect becomes import call
(import "std" "tcp_connect" (func $tcp_connect (param i32 i32) (result i32)))
```

### GPU Translation
```metal
// Network request buffer (GPU → CPU)
struct NetworkRequest {
    uint request_id;
    uint request_type;  // 0=connect, 1=write, 2=read
    uint data_offset;
    uint data_len;
    atomic_uint status;  // 0=pending, 1=complete, 2=error
};

// GPU code queues request, CPU processes after dispatch
kernel void network_operation(
    device NetworkRequest* requests,
    device atomic_uint* request_count,
    device uint8_t* data_buffer,
    uint gid [[thread_position_in_grid]]
) {
    // Thread 0 queues the connect request
    if (gid == 0) {
        uint slot = atomic_fetch_add_explicit(request_count, 1, memory_order_relaxed);
        requests[slot].request_type = 0;  // connect
        requests[slot].data_offset = /* hostname offset */;
        atomic_store_explicit(&requests[slot].status, 0, memory_order_relaxed);
    }

    // All threads barrier
    threadgroup_barrier(mem_flags::mem_device);

    // Poll for completion (in same frame or next)
    // Note: This requires multi-dispatch or persistent kernel
}
```

### Detection
1. Import from "std" module with "tcp", "udp", "net" in name
2. Socket syscall numbers in WASI

### CPU Coprocessor
```rust
// After GPU dispatch, CPU processes network queue
fn process_network_queue(requests: &[NetworkRequest]) {
    for req in requests.iter().filter(|r| r.status.load() == 0) {
        match req.request_type {
            0 => { /* TcpStream::connect */ }
            1 => { /* write */ }
            2 => { /* read */ }
        }
        req.status.store(1, Ordering::Release);
    }
}
```

---

## Transformation 3: Condvar → Threadgroup Barrier

### User Code (Unchanged)
```rust
use std::sync::{Condvar, Mutex};

static PAIR: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let (lock, cvar) = &PAIR;

    // Worker threads
    {
        let mut done = lock.lock().unwrap();
        *done = true;
        cvar.notify_all();
    }

    // Waiting thread
    {
        let mut done = lock.lock().unwrap();
        while !*done {
            done = cvar.wait(done).unwrap();
        }
    }

    1
}
```

### GPU Translation
```metal
// Condvar::wait → threadgroup_barrier + atomic check
// Condvar::notify_all → atomic write + barrier

kernel void condvar_translated(
    device atomic_uint* condition,
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // "Worker" threads (equivalent)
    if (gid < WORKER_COUNT) {
        // Do work...
        atomic_store_explicit(condition, 1, memory_order_relaxed);
    }

    // Barrier replaces notify_all + wait
    threadgroup_barrier(mem_flags::mem_device);

    // All threads can now see condition == 1
    uint done = atomic_load_explicit(condition, memory_order_relaxed);
}
```

### Limitations
- `Condvar::wait()` with timeout → frame-based timeout
- Complex wait conditions → may need multiple barriers
- Cross-threadgroup sync → requires atomic polling (slower)

---

## Transformation 4: Rc<T> → Atomic Refcount

### User Code (Unchanged)
```rust
use std::rc::Rc;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let data = Rc::new(42);
    let clone1 = Rc::clone(&data);
    let clone2 = Rc::clone(&data);

    *data + *clone1 + *clone2  // 126
}
```

### WASM Pattern
```wasm
;; Rc uses non-atomic increment
(func $rc_clone (param $ptr i32) (result i32)
  (i32.store (local.get $ptr)
    (i32.add (i32.load (local.get $ptr)) (i32.const 1)))
  (local.get $ptr)
)
```

### GPU Translation
```metal
// Automatically upgrade to atomic operations
// Detection: increment without atomic in refcount position

int rc_clone(device RcBox* rc) {
    // Non-atomic increment → atomic increment
    atomic_fetch_add_explicit(&rc->refcount, 1, memory_order_relaxed);
    return rc;
}

void rc_drop(device RcBox* rc) {
    if (atomic_fetch_sub_explicit(&rc->refcount, 1, memory_order_relaxed) == 1) {
        // Last reference, deallocate
        gpu_dealloc(rc);
    }
}
```

### Detection
1. Struct with first field used as counter
2. Non-atomic increment in "clone" pattern
3. Decrement + conditional dealloc in "drop" pattern

---

## Transformation 5: thread::sleep → Frame Timing

### User Code (Unchanged)
```rust
use std::thread;
use std::time::Duration;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    // Do some work
    let partial = compute_step1();

    // Wait 100ms
    thread::sleep(Duration::from_millis(100));

    // Continue
    compute_step2(partial)
}
```

### GPU Translation
```metal
// sleep → record frame and check later

kernel void with_sleep(
    device uint* frame_counter,
    device uint* sleep_until,
    device int* partial_result,
    uint gid [[thread_position_in_grid]]
) {
    // Dispatch 1: compute_step1
    if (*sleep_until == 0) {
        *partial_result = compute_step1();
        // 100ms ≈ 6 frames at 60fps
        *sleep_until = *frame_counter + 6;
        return;  // Early exit, will be re-dispatched
    }

    // Dispatch 2+: check if sleep complete
    if (*frame_counter < *sleep_until) {
        return;  // Still sleeping
    }

    // Sleep complete, continue
    *sleep_until = 0;
    compute_step2(*partial_result);
}
```

### Host Loop
```rust
loop {
    // Check if any kernels are "sleeping"
    if gpu_has_pending_sleeps() {
        // Re-dispatch with incremented frame counter
        frame_counter += 1;
        gpu.dispatch(kernel, frame_counter);
    }
}
```

---

## Transformation 6: Mutex → Atomic Spinlock

### User Code (Unchanged)
```rust
use std::sync::Mutex;

static COUNTER: Mutex<i32> = Mutex::new(0);

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut guard = COUNTER.lock().unwrap();
    *guard += 1;
    *guard
}
```

### GPU Translation
```metal
// Mutex::lock → atomic spinlock (USE WITH CAUTION)

struct GpuMutex {
    atomic_uint locked;
    int value;
};

void mutex_lock(device GpuMutex* m) {
    // Spinlock - WARNING: can cause GPU hangs if misused
    while (atomic_exchange_explicit(&m->locked, 1, memory_order_acquire) == 1) {
        // Spin
    }
}

void mutex_unlock(device GpuMutex* m) {
    atomic_store_explicit(&m->locked, 0, memory_order_release);
}
```

### Warning Detection
If we detect:
- High contention (many threads locking same mutex)
- Long critical sections
- Nested locks

We emit a **performance warning** (not error):
```
PERF WARNING: High-contention mutex detected. Consider using atomic operations directly.
```

---

## Transformation 7: fs::read → MTLIOCommandQueue (GPU-Initiated)

**Note:** Unlike network I/O, file I/O is GPU-initiated via Metal 3's `MTLIOCommandQueue`. No CPU involvement during data transfer.

### User Code (Unchanged)
```rust
use std::fs;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let content = fs::read_to_string("/data/config.txt").unwrap();
    content.len() as i32
}
```

### GPU Translation
File I/O uses MTLIOCommandQueue - GPU-initiated, not CPU queue:

```metal
// File I/O control block (pre-configured by CPU at init time)
struct FileIORequest {
    uint request_id;
    uint operation;       // 0=read, 1=write
    uint file_handle_id;  // Pre-opened MTLIOFileHandle index
    uint buffer_offset;   // GPU buffer destination
    uint file_offset;     // Offset in file
    uint size;
    atomic_uint status;   // 0=pending, 1=in_progress, 2=complete
};

// GPU triggers I/O via atomic flag (CPU monitors and dispatches MTLIOCommandBuffer)
// This is NOT CPU processing the I/O - CPU just submits to MTLIOCommandQueue
// The actual data transfer is GPU DMA, zero-copy on Apple Silicon

kernel void file_read(
    device FileIORequest* requests,
    device atomic_uint* request_count,
    device atomic_uint* io_trigger,
    device char* file_buffer,
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        uint slot = atomic_fetch_add_explicit(request_count, 1, memory_order_relaxed);
        requests[slot].operation = 0;  // read
        requests[slot].buffer_offset = /* destination */;
        requests[slot].size = /* size */;
        atomic_store_explicit(&requests[slot].status, 0, memory_order_release);

        // Signal CPU to submit MTLIOCommandBuffer (lightweight trigger only)
        atomic_store_explicit(io_trigger, 1, memory_order_release);
    }

    // Poll for I/O completion
    while (atomic_load_explicit(&requests[0].status, memory_order_acquire) != 2) {
        // Spin or do other work
    }

    // File data now in file_buffer - process it
    process_file_data(file_buffer, gid);
}
```

**Why this is different from network:**
- Network: CPU processes packets (NIC hardware requires it)
- File I/O: CPU only submits to MTLIOCommandQueue, GPU DMA does transfer
- On Apple Silicon unified memory: zero-copy, data lands directly in GPU buffer

---

## Implementation in Translator

### Phase 1: Pattern Detection

Add pattern matchers to WASM translator:

```rust
// src/gpu_bytecode/wasm_translator.rs

fn detect_async_pattern(&self, func: &WasmFunction) -> Option<AsyncPattern> {
    // Look for:
    // 1. State machine variable
    // 2. Loop with branch table
    // 3. Poll-like yield points
}

fn detect_network_call(&self, import: &WasmImport) -> Option<NetworkOp> {
    // Check import name for tcp/udp/socket patterns
}

fn detect_condvar_pattern(&self, func: &WasmFunction) -> Option<CondvarUsage> {
    // Look for wait/notify call patterns
}

fn detect_refcount_pattern(&self, func: &WasmFunction) -> Option<RefcountOp> {
    // Non-atomic increment + conditional drop
}
```

### Phase 2: Bytecode Generation

Generate transformed bytecode:

```rust
fn translate_async_fn(&mut self, pattern: AsyncPattern) -> Vec<GpuOp> {
    // Convert state machine to parallel dispatch
    vec![
        GpuOp::WORK_QUEUE_PUSH,
        GpuOp::PARALLEL_DISPATCH,
        // ...
    ]
}

fn translate_network_op(&mut self, op: NetworkOp) -> Vec<GpuOp> {
    vec![
        GpuOp::REQUEST_QUEUE_RESERVE,
        GpuOp::REQUEST_WRITE,
        GpuOp::REQUEST_SUBMIT,
        // ...
    ]
}
```

### Phase 3: Runtime Support

Add GPU runtime buffers:

```rust
pub struct GpuRuntime {
    work_queue: GpuBuffer,          // For async dispatch
    network_requests: GpuBuffer,    // For TcpStream etc
    file_requests: GpuBuffer,       // For fs operations
    sleep_timers: GpuBuffer,        // For thread::sleep
    frame_counter: GpuBuffer,       // Time reference
}
```

---

## New Bytecode Opcodes

| Opcode | Name | Description |
|--------|------|-------------|
| 0x80 | WORK_PUSH | Push work item to queue |
| 0x81 | WORK_POP | Pop work item |
| 0x82 | BARRIER | Threadgroup barrier |
| 0x83 | ATOMIC_INC | Atomic increment |
| 0x84 | ATOMIC_DEC | Atomic decrement |
| 0x85 | ATOMIC_CAS | Compare and swap |
| 0x86 | REQUEST_QUEUE | Queue I/O request |
| 0x87 | REQUEST_POLL | Poll request status |
| 0x88 | FRAME_WAIT | Wait for frame count |
| 0x89 | SPINLOCK | Acquire spinlock |
| 0x8A | SPINUNLOCK | Release spinlock |

---

## Success Criteria

| Feature | User Experience |
|---------|-----------------|
| async/await | **Compiles and runs** - parallel dispatch |
| TcpStream | **Compiles and runs** - request queue |
| Condvar | **Compiles and runs** - barrier |
| Rc<T> | **Compiles and runs** - atomic refcount |
| thread::sleep | **Compiles and runs** - frame timing |
| Mutex | **Compiles and runs** - spinlock (with perf warning) |
| fs::read | **Compiles and runs** - async I/O |

**ZERO user code changes required.**

---

## Additional Transformations (Getting Creative)

### Process Spawning → GPU-Native Process Creation (GPU_IMPL)

**Key Insight: GPU threads ARE processes.** On GPU, "spawning a process" means:
1. Look up pre-compiled GPU bytecode for the target program
2. Allocate isolated state buffer for the new process
3. Reserve thread range from GPU scheduler
4. Dispatch as persistent sub-kernel

This is TRUE parallelism, not simulation. No CPU involvement.

```rust
// User writes this (UNCHANGED)
use std::process::Command;

let output = Command::new("ls")
    .arg("-la")
    .output()
    .expect("failed");
```

**GPU Translation:**
```metal
// GPU-native process creation
// "ls" becomes a pre-compiled GPU bytecode module

struct GpuProcess {
    uint process_id;
    uint bytecode_offset;     // Offset into pre-compiled bytecode table
    uint state_buffer_offset; // Isolated memory region
    uint thread_range_start;  // Reserved threads for this process
    uint thread_range_count;
    atomic_uint status;       // 0=pending, 1=running, 2=complete
    uint output_offset;       // Where to write stdout
};

// GPU scheduler allocates threads for new process
kernel void spawn_process(
    device GpuProcess* processes,
    device atomic_uint* process_count,
    device BytecodeModule* bytecode_table,
    device char* state_pool,
    device atomic_uint* thread_allocator,
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        uint proc_id = atomic_fetch_add_explicit(process_count, 1, memory_order_relaxed);

        // Look up "ls" bytecode (pre-compiled at build time)
        uint bytecode_id = lookup_bytecode("ls");
        processes[proc_id].bytecode_offset = bytecode_table[bytecode_id].offset;

        // Allocate isolated state buffer (GPU memory allocator)
        processes[proc_id].state_buffer_offset = gpu_alloc(&state_pool, STATE_SIZE);

        // Reserve thread range from GPU scheduler
        uint threads_needed = bytecode_table[bytecode_id].thread_count;
        processes[proc_id].thread_range_start =
            atomic_fetch_add_explicit(thread_allocator, threads_needed, memory_order_relaxed);
        processes[proc_id].thread_range_count = threads_needed;

        // Mark as ready to run
        atomic_store_explicit(&processes[proc_id].status, 0, memory_order_release);
    }
}

// Persistent sub-kernel executes the spawned process
kernel void execute_process(
    device GpuProcess* process,
    device BytecodeModule* bytecode_table,
    device char* state_pool,
    uint gid [[thread_position_in_grid]]
) {
    // Only threads in allocated range execute
    if (gid < process->thread_range_start ||
        gid >= process->thread_range_start + process->thread_range_count) {
        return;
    }

    uint local_tid = gid - process->thread_range_start;
    device char* my_state = state_pool + process->state_buffer_offset;
    device BytecodeModule* bytecode = bytecode_table + process->bytecode_offset;

    // Execute bytecode VM for this thread
    execute_bytecode_vm(bytecode, my_state, local_tid);

    // Last thread signals completion
    if (local_tid == 0) {
        atomic_store_explicit(&process->status, 2, memory_order_release);
    }
}
```

### Environment Variables → Load-Time Snapshot

```rust
// User writes this (UNCHANGED)
let path = std::env::var("PATH").unwrap();
let home = std::env::var("HOME").unwrap();
```

**GPU Translation:**
```metal
// At load time, CPU snapshots all env vars into GPU buffer
struct EnvSnapshot {
    uint count;
    EnvEntry entries[MAX_ENV_VARS];
};

struct EnvEntry {
    uint key_offset;
    uint key_len;
    uint value_offset;
    uint value_len;
};

// env::var() becomes O(1) hash lookup in snapshot buffer
uint env_var(device EnvSnapshot* env, const char* key) {
    uint hash = hash_string(key);
    uint slot = hash & (MAX_ENV_VARS - 1);
    // Lookup in pre-populated buffer
    return env->entries[slot].value_offset;
}
```

### Thread-Local Storage → Thread-Indexed Buffer

```rust
// User writes this (UNCHANGED)
thread_local! {
    static COUNTER: Cell<u32> = Cell::new(0);
}

COUNTER.with(|c| c.set(c.get() + 1));
```

**GPU Translation:**
```metal
// TLS becomes per-thread buffer indexed by thread_id
device uint* tls_counter;  // Array of MAX_THREADS elements

kernel void with_tls(device uint* tls_counter, uint gid) {
    // thread_local access = array index
    tls_counter[gid] += 1;
}
```

**Note:** GPU already has millions of "thread locals" - each thread has its own registers!

### Stack Unwinding → Error Flag Propagation

```rust
// User writes this (UNCHANGED)
let result = std::panic::catch_unwind(|| {
    might_panic();
});
```

**GPU Translation:**
```metal
// Transform panic to error flag
// All threads check flag and early-return

device atomic_uint panic_flag;
device uint panic_thread;
device uint panic_location;

void gpu_panic(uint gid, uint location) {
    // First thread to panic wins
    uint expected = 0;
    if (atomic_compare_exchange(&panic_flag, &expected, 1)) {
        panic_thread = gid;
        panic_location = location;
    }
}

// catch_unwind becomes:
kernel void with_catch_unwind(...) {
    // Check if any thread panicked
    if (atomic_load(&panic_flag)) {
        // "Unwind" = early return with error
        return;
    }

    // Normal execution
    might_panic_body();

    // Check again after potential panic point
    if (atomic_load(&panic_flag)) {
        return;
    }
}
```

### Varargs → Array-Based Dispatch

```rust
// User writes this (UNCHANGED)
fn variadic_sum(args: &[i32]) -> i32 {
    args.iter().sum()
}

// Or C-style (rare in Rust but possible via FFI)
```

**GPU Translation:**
```metal
// WASM already transforms varargs to pointer + count
// We just handle it as array

int variadic_sum(device int* args, uint count) {
    int sum = 0;
    for (uint i = 0; i < count; i++) {
        sum += args[i];
    }
    return sum;
}
```

### OS-Specific APIs → Syscall Queue

```rust
// User writes this (UNCHANGED)
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

let perms = fs::metadata("file")?.permissions();
let mode = perms.mode();  // Unix-specific
```

**GPU Translation:**
```metal
// OS-specific = specific syscall numbers
// Queue the syscall, CPU executes

struct SyscallRequest {
    uint syscall_num;  // e.g., stat64 for metadata
    uint arg1, arg2, arg3;
    atomic_uint status;
    uint result;
};

// Unix permissions.mode() = field in stat result
// Already in the response buffer from fs::metadata
```

### Dynamic Library Loading → Pre-Compiled Dispatch

```rust
// User writes this (UNCHANGED)
let lib = libloading::Library::new("plugin.so")?;
let func: Symbol<fn() -> i32> = lib.get(b"plugin_main")?;
let result = func();
```

**GPU Translation:**
Two options:

**Option A: Ahead-of-time compilation**
```rust
// At build time, compile all possible plugins to GPU bytecode
// Runtime "loading" = selecting from pre-compiled table
device FunctionPtr plugin_dispatch_table[MAX_PLUGINS];

// "Load" = lookup in table
uint load_library(const char* name) {
    uint hash = hash_string(name);
    return hash & (MAX_PLUGINS - 1);
}
```

**Option B: JIT via CPU**
```rust
// Queue load request to CPU
// CPU compiles plugin to GPU bytecode
// GPU receives new bytecode in buffer
// Next dispatch includes new code
```

---

## Edge Cases and Limitations

### Performance Warnings (Not Errors)

1. **Process spawning in hot loop** - Thread pool exhaustion
   - Emit: `PERF WARNING: Frequent process spawning. GPU thread pool may exhaust. Consider batching.`
   - (GPU-native processes still consume thread ranges; too many concurrent processes = no threads left)

2. **Sleep in tight loop** - Poor GPU utilization
   - Emit: `PERF WARNING: Sleep in hot path reduces parallelism.`

3. **TLS with large data** - Memory per thread adds up
   - Emit: `PERF WARNING: Large thread-local (1KB * 10000 threads = 10MB)`

### Performance Warnings (Not Errors)

```rust
enum Warning {
    HighContentionMutex,      // Consider atomics
    SleepInHotPath,           // Poor utilization
    FrequentNetworkOps,       // Batch for efficiency
    DeepRecursion,            // May hit stack limit
}
```

---

## Implementation Priority

| Phase | Work | Enables |
|-------|------|---------|
| 5.1 | Rc → atomic | Safe shared ownership |
| 5.2 | Mutex → spinlock | Basic sync |
| 5.3 | Condvar → barrier | Thread coordination |
| 5.4 | sleep → frame timing | Delays work |
| 5.5 | fs::read → MTLIOCommandQueue | File access (GPU-initiated) |
| 5.6 | network → CPU queue | Network access (requires CPU) |
| 5.7 | async/await → dispatch | Full async support |
| 5.8 | **process spawn → GPU_IMPL** | **GPU-native process creation** |

---

## The User Experience

**Before (what we had):**
```
ERROR: async/await not supported on GPU
  Help: Rewrite using parallel dispatch pattern
```

**After (what we're building):**
```
$ cargo build --target wasm32-unknown-unknown
   Compiling my_app v0.1.0
    Finished release [optimized] target(s)

$ gpu-run target/wasm32-unknown-unknown/release/my_app.wasm
Running on GPU with 10240 threads...
Result: 42
```

**Zero errors. Zero rewrites. Just works.**

---

## Research Findings: What Actually Works on Apple Silicon

This section documents proven GPU-native capabilities on Apple Silicon, based on research and implementation experience.

### MTLIOCommandQueue: GPU-Initiated File I/O

Metal 3 introduced `MTLIOCommandQueue`, enabling GPU-initiated asynchronous file I/O without CPU involvement during transfer:

```swift
// Create I/O command queue
let ioQueue = device.makeIOCommandQueue(descriptor: ioDescriptor)

// Create file handle
let fileHandle = try MTLIOFileHandle(url: fileURL, device: device)

// Load directly into GPU buffer
let ioCommandBuffer = ioQueue.makeCommandBuffer()
ioCommandBuffer.load(buffer: gpuBuffer, offset: 0, size: fileSize, sourceHandle: fileHandle, sourceHandleOffset: 0)
ioCommandBuffer.commit()
```

**Key Properties:**
- Zero-copy transfer to GPU buffers on Apple Silicon (unified memory)
- GPU can initiate loads; CPU only sets up the queue initially
- Supports compressed textures with hardware decompression
- Asynchronous - GPU kernel can poll for completion via atomics

**Pattern for GPU-native apps:**
```metal
// GPU polls for I/O completion
kernel void io_aware_kernel(
    device atomic_uint* io_complete_flag,
    device char* file_buffer,
    uint gid [[thread_position_in_grid]]
) {
    // Wait for I/O completion
    while (atomic_load_explicit(io_complete_flag, memory_order_acquire) == 0) {
        // Spin or do other work
    }

    // Process file data directly on GPU
    process_data(file_buffer, gid);
}
```

### Network Reality: The One True CPU Dependency

**Standard network (WiFi, Ethernet) requires CPU.** This is unavoidable:

1. **NIC hardware sends packets to CPU** - Consumer network cards deliver to CPU memory
2. **Network stack is kernel-owned** - Packets go through BSD socket layer
3. **Interrupt handling requires CPU** - NIC interrupts target CPU cores

**The Pattern (cannot be avoided):**
```
[ Network ] --> [ NIC ] --> [ CPU interrupt ] --> [ Kernel ] --> [ CPU buffer ]
                                                                       |
                                                                       v
                                                              [ GPU buffer copy ]
                                                                       |
                                                                       v
                                                              [ GPU processes ]
```

**Best Practice:**
```rust
// CPU receives packets, writes to GPU-visible buffer
fn network_receive_loop(gpu_buffer: &mut MappedBuffer) {
    loop {
        let packet = socket.recv();
        let offset = atomic_increment(&gpu_buffer.write_head);
        gpu_buffer.data[offset..].copy_from_slice(&packet);
        atomic_store(&gpu_buffer.ready_flag, 1);
    }
}

// GPU polls and processes when data arrives
kernel void process_network_data(
    device atomic_uint* ready_flag,
    device char* packet_buffer,
    uint gid [[thread_position_in_grid]]
) {
    if (atomic_load_explicit(ready_flag, memory_order_acquire)) {
        process_packet(packet_buffer, gid);
    }
}
```

### Persistent Kernels: GPU Programs That Never Stop

Metal supports persistent kernels that run indefinitely, proven in this codebase:

**Issue #133 (GPU-Resident Filesystem Index):**
```metal
// Kernel stays running, polls for work
kernel void persistent_filesystem_index(
    device atomic_uint* command_flag,
    device FileSystemCommand* commands,
    device SearchResult* results,
    uint gid [[thread_position_in_grid]]
) {
    while (true) {
        // Poll for new command
        uint cmd = atomic_load_explicit(command_flag, memory_order_acquire);
        if (cmd == 0) continue;

        // Process command
        switch (cmd) {
            case CMD_SEARCH: search_files(commands, results, gid); break;
            case CMD_INDEX: index_directory(commands, gid); break;
        }

        // Signal completion
        if (gid == 0) {
            atomic_store_explicit(command_flag, 0, memory_order_release);
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

**Issue #149 (GPU-Driven Event Dispatch):**
```metal
// Persistent kernel processes input events
kernel void persistent_event_handler(
    device atomic_uint* event_count,
    device InputEvent* events,
    device UIState* state,
    uint gid [[thread_position_in_grid]]
) {
    while (true) {
        uint count = atomic_load_explicit(event_count, memory_order_acquire);
        if (count == 0) continue;

        // Process events in parallel
        if (gid < count) {
            InputEvent event = events[gid];
            handle_event(event, state);
        }

        // Clear event queue
        if (gid == 0) {
            atomic_store_explicit(event_count, 0, memory_order_release);
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

**Key Properties:**
- `while(true)` loops are valid in Metal compute shaders
- Poll memory atomics for I/O completion or new commands
- Threadgroup barriers synchronize within the kernel
- CPU only needs to write to shared buffers, never dispatch

**Pattern: GPU as the Main Loop**
```
Traditional:
  CPU main loop --> dispatch GPU --> wait --> dispatch GPU --> wait

Persistent kernel:
  CPU boots --> dispatch persistent kernel --> done
  GPU runs forever, polls for work
  CPU only writes to shared buffers when it has input
```

---

## What We CAN'T Transform

These are the only true hardware limitations:

### 1. Direct Network Packet Reception

Standard network interfaces (WiFi, Ethernet) require CPU involvement:
- NIC interrupts are delivered to CPU cores
- macOS kernel network stack processes packets
- NIC hardware sends packets to CPU memory, not GPU

### 2. GPU-to-CPU Interrupts

GPU cannot interrupt CPU. Hardware limitation:
- GPU can only signal via shared memory
- CPU must poll or use fence callbacks
- No way for GPU to "wake up" a sleeping CPU thread

**Workaround:** CPU periodically polls completion flags, or uses Metal fence callbacks that fire on CPU thread pool.

---

## I/O Classification Summary

| Category | Implementation | CPU Involvement | Notes |
|----------|---------------|-----------------|-------|
| **Process spawn** | GPU_IMPL | NONE | Thread dispatch + bytecode execution |
| **File I/O** | MTLIOCommandQueue | MINIMAL* | GPU DMA, zero-copy on Apple Silicon |
| **Network I/O** | CPU queue | REQUIRED | NIC hardware limitation |
| **Memory alloc** | GPU allocator | NONE | Bump allocator in GPU buffer |
| **Thread sync** | Atomics + barriers | NONE | Native GPU primitives |
| **Timers** | Frame counter | NONE | GPU tracks frame timing |
| **IPC** | Shared memory | NONE | Atomics for synchronization |

*CPU only submits MTLIOCommandBuffer; actual transfer is GPU DMA

---

**Everything else can run on GPU.** File I/O (via MTLIOCommandQueue), process spawning (via GPU thread dispatch), timers, synchronization, memory management - all can be transformed to GPU-native patterns using the techniques documented above. Network I/O is the ONLY operation that truly requires CPU involvement due to NIC hardware design.
