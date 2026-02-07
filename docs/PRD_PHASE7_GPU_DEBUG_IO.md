# PRD: Phase 7 - GPU Debug I/O (println!, dbg! Support)

## Overview

**THE GPU IS THE COMPUTER.**

Phase 7 provides debug output capabilities for GPU programs, replacing `println!` and `dbg!` with GPU-native equivalents that write to a debug buffer rendered on screen or captured for testing.

## Problem Statement

Currently, GPU programs have no way to output debug information:
```rust
println!("value = {}", x);  // ❌ No stdout on GPU
dbg!(my_struct);            // ❌ No stderr on GPU
```

This makes debugging GPU programs extremely difficult.

## Solution: GPU Debug Buffer

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Debug Ring Buffer                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │ [thread_id][timestamp][type][length][data...]       ││
│  │ [thread_id][timestamp][type][length][data...]       ││
│  │ ...                                                 ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  Write pointer (atomic)  ────────────────────────────►  │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Debug Renderer (GPU)                        │
│  Reads buffer, renders text to screen overlay           │
└─────────────────────────────────────────────────────────┘
```

### Debug Entry Format

```
┌────────────┬────────────┬──────────┬──────────┬─────────────┐
│ thread_id  │ timestamp  │   type   │  length  │    data     │
│  (4 bytes) │ (4 bytes)  │ (1 byte) │ (1 byte) │ (N bytes)   │
└────────────┴────────────┴──────────┴──────────┴─────────────┘
```

### Message Types

| Type | Code | Description |
|------|------|-------------|
| INT | 0x01 | Integer value (i32) |
| FLOAT | 0x02 | Float value (f32) |
| STRING | 0x03 | String (length-prefixed UTF-8) |
| BOOL | 0x04 | Boolean (1 byte) |
| ARRAY | 0x05 | Array start marker |
| STRUCT | 0x06 | Struct start marker |
| NEWLINE | 0x0A | Line separator |

### Intrinsics

```rust
// Low-level debug intrinsics
extern "C" {
    fn __gpu_debug_i32(value: i32);
    fn __gpu_debug_f32(value: f32);
    fn __gpu_debug_str(ptr: *const u8, len: usize);
    fn __gpu_debug_bool(value: bool);
    fn __gpu_debug_newline();
    fn __gpu_debug_flush();  // Force render
}
```

### GPU Implementation (Metal)

```metal
struct DebugBuffer {
    atomic_uint write_pos;
    uint capacity;
    uint8_t data[DEBUG_BUFFER_SIZE];
};

void gpu_debug_i32(int value, uint gid, float time, device DebugBuffer& dbg) {
    // Reserve space atomically
    uint pos = atomic_fetch_add_explicit(&dbg.write_pos, 10, memory_order_relaxed);
    if (pos + 10 > dbg.capacity) return;  // Buffer full

    // Write entry
    *(device uint*)(dbg.data + pos) = gid;
    *(device float*)(dbg.data + pos + 4) = time;
    dbg.data[pos + 8] = 0x01;  // INT type
    *(device int*)(dbg.data + pos + 9) = value;
}

void gpu_debug_str(device const uint8_t* str, uint len, uint gid, float time, device DebugBuffer& dbg) {
    uint entry_size = 10 + len;
    uint pos = atomic_fetch_add_explicit(&dbg.write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg.capacity) return;

    *(device uint*)(dbg.data + pos) = gid;
    *(device float*)(dbg.data + pos + 4) = time;
    dbg.data[pos + 8] = 0x03;  // STRING type
    dbg.data[pos + 9] = len;
    for (uint i = 0; i < len; i++) {
        dbg.data[pos + 10 + i] = str[i];
    }
}
```

### Bytecode Opcodes

| Opcode | Name | Description |
|--------|------|-------------|
| 0x70 | DBG_I32 | Debug print i32 |
| 0x71 | DBG_F32 | Debug print f32 |
| 0x72 | DBG_STR | Debug print string |
| 0x73 | DBG_BOOL | Debug print bool |
| 0x74 | DBG_NL | Debug newline |
| 0x75 | DBG_FLUSH | Force debug render |

### Rust SDK - High-Level API

```rust
// gpu_sdk/src/debug.rs
#![no_std]

/// Print a debug message (like println! but for GPU)
#[macro_export]
macro_rules! gpu_println {
    ($($arg:tt)*) => {{
        // For simple cases, format at compile time
        $crate::debug::_print_args(format_args!($($arg)*));
        $crate::debug::newline();
    }};
}

/// Debug print with file:line (like dbg!)
#[macro_export]
macro_rules! gpu_dbg {
    ($val:expr) => {{
        let val = $val;
        $crate::debug::print_str(concat!(file!(), ":", line!(), ": "));
        $crate::debug::print_str(stringify!($val));
        $crate::debug::print_str(" = ");
        $crate::debug::print_value(&val);
        $crate::debug::newline();
        val
    }};
}

pub fn print_i32(v: i32) {
    extern "C" { fn __gpu_debug_i32(v: i32); }
    unsafe { __gpu_debug_i32(v); }
}

pub fn print_f32(v: f32) {
    extern "C" { fn __gpu_debug_f32(v: f32); }
    unsafe { __gpu_debug_f32(v); }
}

pub fn print_str(s: &str) {
    extern "C" { fn __gpu_debug_str(ptr: *const u8, len: usize); }
    unsafe { __gpu_debug_str(s.as_ptr(), s.len()); }
}

pub fn newline() {
    extern "C" { fn __gpu_debug_newline(); }
    unsafe { __gpu_debug_newline(); }
}

// Trait for printable types
pub trait GpuDebug {
    fn gpu_fmt(&self);
}

impl GpuDebug for i32 {
    fn gpu_fmt(&self) { print_i32(*self); }
}

impl GpuDebug for f32 {
    fn gpu_fmt(&self) { print_f32(*self); }
}

impl GpuDebug for bool {
    fn gpu_fmt(&self) {
        print_str(if *self { "true" } else { "false" });
    }
}

impl GpuDebug for &str {
    fn gpu_fmt(&self) { print_str(self); }
}

pub fn print_value<T: GpuDebug>(v: &T) {
    v.gpu_fmt();
}
```

### User Code

```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    gpu_println!("Starting computation");

    let mut sum = 0;
    for i in 0..n {
        sum += i;
        if i % 10 == 0 {
            gpu_dbg!(sum);  // Prints: "src/main.rs:12: sum = 45"
        }
    }

    gpu_println!("Final sum: ", sum);
    sum
}
```

### Debug Output Rendering

Two modes:

#### 1. Screen Overlay (Visual)
```
┌──────────────────────────────────────────────┐
│                                              │
│                Main Content                  │
│                                              │
├──────────────────────────────────────────────┤
│ DEBUG OUTPUT:                                │
│ [0] Starting computation                     │
│ [0] src/main.rs:12: sum = 45                 │
│ [0] src/main.rs:12: sum = 145                │
│ [0] Final sum: 4950                          │
└──────────────────────────────────────────────┘
```

#### 2. Buffer Capture (Testing)
```rust
// In test code
let debug_output = system.capture_debug_buffer();
assert!(debug_output.contains("sum = 45"));
```

## Handling Multi-Threaded Debug

GPU has thousands of threads. Debug output needs filtering:

### Thread Filtering

```rust
// Only print from thread 0
if gpu_sdk::thread_id() == 0 {
    gpu_println!("Only once!");
}

// Or use the macro
gpu_println_once!("This prints from thread 0 only");
```

### Thread-Tagged Output

```
[thread:0] Starting computation
[thread:0] sum = 45
[thread:1] sum = 46    // Only shown if multi-thread debug enabled
[thread:2] sum = 47
```

### Debug Levels

```rust
// In gpu_sdk
pub enum DebugLevel {
    Off,           // No debug output
    SingleThread,  // Only thread 0
    FirstN(u32),   // First N threads
    All,           // All threads (careful!)
}

pub fn set_debug_level(level: DebugLevel) {
    extern "C" { fn __gpu_set_debug_level(level: u32); }
    unsafe { __gpu_set_debug_level(level as u32); }
}
```

## Test Cases

### Test 1: Basic Integer Print
```rust
#![no_std]
use gpu_sdk::debug::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    print_i32(42);
    newline();
    0
}
```
Expected debug output: `42`

### Test 2: String Print
```rust
#![no_std]
use gpu_sdk::debug::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    print_str("Hello GPU!");
    newline();
    0
}
```
Expected: `Hello GPU!`

### Test 3: gpu_println! Macro
```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let x = 10;
    let y = 20;
    gpu_println!("x + y = ");
    print_i32(x + y);
    0
}
```
Expected: `x + y = 30`

### Test 4: gpu_dbg! Macro
```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let value = 123;
    gpu_dbg!(value);
    value
}
```
Expected: `file.rs:8: value = 123`

### Test 5: Loop Debugging
```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    for i in 0..5 {
        gpu_dbg!(i);
    }
    0
}
```
Expected:
```
file.rs:7: i = 0
file.rs:7: i = 1
file.rs:7: i = 2
file.rs:7: i = 3
file.rs:7: i = 4
```

### Test 6: Thread ID in Output
```rust
#![no_std]
use gpu_sdk::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let tid = thread_id();
    if tid < 3 {
        print_str("[");
        print_i32(tid);
        print_str("] Hello from thread");
        newline();
    }
    0
}
```
Expected (parallel execution):
```
[0] Hello from thread
[1] Hello from thread
[2] Hello from thread
```

### Test 7: Buffer Capture in Tests
```rust
#[test]
fn test_debug_capture() {
    let mut system = GpuAppSystem::new().unwrap();
    // ... load and run bytecode app ...

    let debug = system.capture_debug_buffer();
    assert!(debug.contains("computation complete"));
}
```

## Implementation Notes

### Buffer Size
- Default: 1MB ring buffer
- Configurable at system init
- Wraps around (oldest messages lost if full)

### Performance
- Atomic write pointer - minimal contention
- Batched rendering - don't render every message
- Conditional compilation - debug macros compile to nothing in release

### Persistent Kernel Integration

**Research Note: Continuous Debug Output**
When running in persistent kernel mode (where the GPU kernel never terminates), the debug buffer becomes a continuous output stream:

- **Ring buffer semantics**: The buffer wraps around, providing a sliding window of recent debug output
- **Non-blocking writes**: GPU threads never block waiting for debug buffer space; if full, messages are dropped
- **Async consumption**: A background GPU thread (or CPU poller) can continuously drain the buffer to external storage or network
- **Timestamp ordering**: Each entry includes a GPU timestamp, allowing reconstruction of event ordering even when consumed out-of-order

This enables `printf`-style debugging for GPU programs that run indefinitely, which is essential for GPU-as-computer workloads.

### Release Builds
```rust
// gpu_sdk feature flags
#[cfg(feature = "debug")]
macro_rules! gpu_println { ... }

#[cfg(not(feature = "debug"))]
macro_rules! gpu_println {
    ($($arg:tt)*) => {}  // Compiles to nothing
}
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Debug write latency | < 50 cycles |
| Buffer throughput | > 100M messages/sec |
| Render overhead | < 1ms/frame |
| Zero overhead in release | 0 instructions |

## Dependencies

- Phase 5 (function calls)
- Text rendering system (already exists)
- Ring buffer infrastructure

## Not Included

- Full `format!` macro (requires allocator + complex formatting)
- Structured logging
- File output (no filesystem write from GPU yet)
