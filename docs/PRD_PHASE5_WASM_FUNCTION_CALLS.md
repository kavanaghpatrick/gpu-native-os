# PRD: Phase 5 - WASM Function Call Support

## Overview

**THE GPU IS THE COMPUTER.**

Phase 5 adds function call support to the WASM→GPU bytecode translator. This enables real-world Rust programs that use helper functions, library code, and GPU intrinsics for rendering.

## Problem Statement

Currently, the WASM translator only handles single-function entry points. Real Rust programs:
1. Use helper functions for code organization
2. Call library functions (even `no_std` uses core intrinsics)
3. Need GPU syscalls for rendering (pixel output, buffer access)

Without function call support, we cannot compile practical applications.

## Design Philosophy

### Inlining-First Strategy

**On GPU, function calls are expensive. Inlining is mandatory.**

| Approach | GPU Cost | Why |
|----------|----------|-----|
| Actual CALL/RET | Catastrophic | Stack push/pop causes SIMD divergence |
| Function pointers | Impossible | No indirect jumps in compute shaders |
| Inlining | Zero overhead | All code in single kernel |

**Decision: Inline all user-defined functions. Map imports to GPU intrinsics.**

**Research Note: Persistent Kernel Pattern**
Function inlining is also a prerequisite for persistent kernel execution. In the persistent kernel pattern, the GPU kernel runs continuously (never terminates), processing work items from a queue. This eliminates kernel launch overhead and enables true GPU-driven execution. With all functions inlined, the kernel becomes a self-contained program that can run indefinitely, only yielding back to CPU for I/O operations that cannot yet be handled on GPU.

### Function Index Space

WASM function indices are organized as:
```
Index 0..import_count-1    → Imported functions (GPU intrinsics)
Index import_count..N      → Defined functions (to be inlined)
```

This is critical: `call 0` might be an import (GPU intrinsic), while `call 5` might be a defined function (inline target).

## Architecture

### 1. Import Section Parsing (Currently Missing)

The translator must parse WASM imports to identify GPU intrinsics:

```rust
// In wasm_translator/src/lib.rs
Payload::ImportSection(reader) => {
    for import in reader {
        let import = import.map_err(|e| TranslateError::Parse(e.to_string()))?;
        if let wasmparser::TypeRef::Func(type_idx) = import.ty {
            module.imports.push(ImportedFunc {
                module: import.module.to_string(),
                name: import.name.to_string(),
                type_idx,
            });
        }
    }
}
```

### 2. GPU Intrinsic Mapping

Imports from `env` or `gpu` module map to GPU operations:

| WASM Import | GPU Operation | Description |
|-------------|---------------|-------------|
| `gpu::set_pixel(x, y, r, g, b, a)` | Write to output buffer | Draw pixel at (x,y) |
| `gpu::get_thread_id()` | `gid` | Current thread index |
| `gpu::get_time()` | `time_buffer[0]` | Frame time |
| `gpu::read_state(idx)` | `state[idx]` | Read app state |
| `gpu::write_state(idx, val)` | `state[idx] = val` | Write app state |
| `gpu::atomic_add(idx, val)` | `atomic_fetch_add` | Atomic operation |
| `gpu::read_file(path_ptr, buf_ptr, len)` | MTLIOCommandQueue | Async file read (see below) |
| `gpu::write_file(path_ptr, buf_ptr, len)` | MTLIOCommandQueue | Async file write (see below) |

**Research Note: I/O Intrinsics via MTLIOCommandQueue**
Metal 3 introduced `MTLIOCommandQueue` for GPU-initiated file I/O. This enables the GPU to request file reads/writes without CPU involvement. The intrinsics above map to this infrastructure:
- `read_file`: Enqueues an async read from storage directly into GPU memory
- `write_file`: Enqueues an async write from GPU memory to storage
- Completion signaled via GPU fence/event, allowing persistent kernel to continue other work while I/O completes

This is a critical building block for eliminating CPU from the I/O path.

### 3. Function Inlining

For defined functions, inline at call site:

```rust
// Pseudocode for inlining
fn translate_call(&mut self, func_idx: u32) -> Result<(), TranslateError> {
    let import_count = self.module.imports.len() as u32;

    if func_idx < import_count {
        // GPU intrinsic - emit specialized code
        self.emit_gpu_intrinsic(func_idx)?;
    } else {
        // Defined function - inline it
        let defined_idx = func_idx - import_count;
        self.inline_function(defined_idx)?;
    }
    Ok(())
}

fn inline_function(&mut self, defined_idx: u32) -> Result<(), TranslateError> {
    // Get function body
    let body = &self.code_bodies[defined_idx as usize];
    let type_idx = self.module.functions[defined_idx as usize];
    let func_type = &self.module.types[type_idx as usize];

    // Save current local context
    let saved_locals = self.locals.clone();
    let saved_stack_base = self.stack.depth();

    // Allocate registers for parameters (from stack)
    let param_count = func_type.params.len();
    for i in (0..param_count).rev() {
        let arg = self.stack.pop()?;
        let param_reg = self.locals.allocate_inline_param(i as u32);
        self.emit.mov(param_reg, arg);
    }

    // Allocate registers for locals
    let locals_reader = body.get_locals_reader()?;
    for local in locals_reader {
        let (count, _ty) = local?;
        for _ in 0..count {
            self.locals.allocate_inline_local();
        }
    }

    // Create return label for this inline frame
    let return_label = self.emit.create_label();
    self.inline_return_stack.push(return_label);

    // Translate function body
    let ops_reader = body.get_operators_reader()?;
    for op in ops_reader {
        self.translate_operator(&op?)?;
    }

    // Define return label
    self.emit.define_label(return_label);
    self.inline_return_stack.pop();

    // Restore local context
    self.locals = saved_locals;

    // Result is on stack (if function has return type)
    Ok(())
}
```

### 4. Recursion Handling

**Recursion is NOT supported on GPU.** Detect and error:

```rust
fn inline_function(&mut self, defined_idx: u32) -> Result<(), TranslateError> {
    // Track call stack to detect recursion
    if self.inline_call_stack.contains(&defined_idx) {
        return Err(TranslateError::Unsupported(
            format!("Recursion not supported on GPU: function {}", defined_idx)
        ));
    }

    self.inline_call_stack.push(defined_idx);
    // ... inline body ...
    self.inline_call_stack.pop();
    Ok(())
}
```

### 5. Calling Convention

For inlined functions:

| Register | Usage |
|----------|-------|
| r4-r7 | Parameters 0-3 (like entry point) |
| r8-r15 | Inline frame locals (remapped per frame) |
| r16-r23 | Scratch registers |
| r24-r29 | Reserved for nested inlines |
| r30-r31 | Temporaries (address calc, etc.) |

### 6. GPU Intrinsic Implementation

Example: `set_pixel` intrinsic

```rust
fn emit_gpu_intrinsic(&mut self, import_idx: u32) -> Result<(), TranslateError> {
    let import = &self.module.imports[import_idx as usize];

    match (import.module.as_str(), import.name.as_str()) {
        ("gpu", "set_pixel") => {
            // Pop args: x, y, r, g, b, a
            let a = self.stack.pop()?;
            let b = self.stack.pop()?;
            let g = self.stack.pop()?;
            let r = self.stack.pop()?;
            let y = self.stack.pop()?;
            let x = self.stack.pop()?;

            // Calculate output buffer index: y * width + x
            // Assume width is in state[OUTPUT_WIDTH_IDX]
            self.emit.loadi_uint(30, OUTPUT_WIDTH_IDX);
            self.emit.ld(30, 30, 0.0);  // r30 = width
            self.emit.mul(30, y, 30);    // r30 = y * width
            self.emit.add(30, 30, x);    // r30 = y * width + x

            // Pack color into float4
            self.emit.mov(31, r);  // r31.x = r (will construct float4)
            // ... additional packing ...

            // Store to output buffer (new opcode needed)
            self.emit.st_output(30, 31);
        }

        ("gpu", "get_thread_id") => {
            // Push gid onto stack
            let result = self.stack.push()?;
            self.emit.loadi_uint(result, 0);  // Placeholder - need GID opcode
            // Actually need: self.emit.get_gid(result);
        }

        _ => {
            return Err(TranslateError::Unsupported(
                format!("Unknown import: {}::{}", import.module, import.name)
            ));
        }
    }

    Ok(())
}
```

## New Bytecode Opcodes

### Required for GPU Intrinsics

| Opcode | Name | Description |
|--------|------|-------------|
| 0x50 | GID | `regs[d] = gid` (thread ID) |
| 0x51 | TIME | `regs[d] = time` (frame time) |
| 0x52 | ST_OUT | `output[regs[s1]] = regs[s2]` (write output) |
| 0x53 | LD_IN | `regs[d] = input[regs[s1]]` (read input) |
| 0x54 | ATOMIC_ADD | `regs[d] = atomic_add(&state[s1], s2)` |

### Metal Implementation

```metal
case 0x50: { // GID
    regs[d] = float4(float(gid), 0, 0, 0);
    break;
}

case 0x51: { // TIME
    regs[d] = float4(time, 0, 0, 0);
    break;
}

case 0x52: { // ST_OUT
    uint idx = as_type<uint>(regs[s1].x);
    output[idx] = regs[s2];
    break;
}

case 0x53: { // LD_IN
    uint idx = as_type<uint>(regs[s1].x);
    regs[d] = input[idx];
    break;
}

case 0x54: { // ATOMIC_ADD
    uint idx = as_type<uint>(regs[s1].x);
    uint val = as_type<uint>(regs[s2].x);
    uint old = atomic_fetch_add_explicit(
        (device atomic_uint*)&state[idx].x,
        val, memory_order_relaxed
    );
    regs[d] = float4(as_type<float>(old), 0, 0, 0);
    break;
}
```

## Rust SDK

Provide `#[no_std]` crate for GPU intrinsics:

```rust
// gpu_sdk/src/lib.rs
#![no_std]

mod intrinsics {
    extern "C" {
        pub fn set_pixel(x: i32, y: i32, r: f32, g: f32, b: f32, a: f32);
        pub fn get_thread_id() -> i32;
        pub fn get_time() -> f32;
        pub fn read_state(idx: i32) -> i32;
        pub fn write_state(idx: i32, val: i32);
        pub fn atomic_add(idx: i32, val: i32) -> i32;
    }
}

pub fn set_pixel(x: i32, y: i32, r: f32, g: f32, b: f32, a: f32) {
    unsafe { intrinsics::set_pixel(x, y, r, g, b, a) }
}

pub fn thread_id() -> i32 {
    unsafe { intrinsics::get_thread_id() }
}

pub fn time() -> f32 {
    unsafe { intrinsics::get_time() }
}

// etc.
```

## Test Cases

### Test 1: Simple Helper Function

```rust
#![no_std]

fn square(x: i32) -> i32 {
    x * x
}

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    square(n) + square(n + 1)
}
```

Expected: `main(3)` → `9 + 16 = 25`

### Test 2: Multiple Helper Functions

```rust
#![no_std]

fn add(a: i32, b: i32) -> i32 { a + b }
fn mul(a: i32, b: i32) -> i32 { a * b }

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    mul(add(n, 1), add(n, 2))
}
```

Expected: `main(3)` → `(3+1) * (3+2) = 4 * 5 = 20`

### Test 3: Recursion Detection

```rust
#![no_std]

fn factorial(n: i32) -> i32 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    factorial(n)
}
```

Expected: Translation ERROR - "Recursion not supported on GPU"

### Test 4: GPU Intrinsic - Thread ID

```rust
#![no_std]
use gpu_sdk::thread_id;

#[no_mangle]
pub extern "C" fn main(_: i32) -> i32 {
    thread_id()  // Should return gid
}
```

Expected: Returns thread's global ID

### Test 5: GPU Intrinsic - Pixel Output

```rust
#![no_std]
use gpu_sdk::{thread_id, set_pixel};

#[no_mangle]
pub extern "C" fn main(_: i32) -> i32 {
    let gid = thread_id();
    let x = gid % 800;
    let y = gid / 800;

    // Red gradient
    let r = (x as f32) / 800.0;
    set_pixel(x, y, r, 0.0, 0.0, 1.0);

    0
}
```

Expected: Red gradient rendered to output buffer

### Test 6: Nested Function Calls

```rust
#![no_std]

fn a(x: i32) -> i32 { b(x) + 1 }
fn b(x: i32) -> i32 { c(x) + 1 }
fn c(x: i32) -> i32 { x + 1 }

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    a(n)  // Should be n + 3
}
```

Expected: `main(10)` → `13`

### Test 7: Function with Multiple Parameters

```rust
#![no_std]

fn lerp(a: i32, b: i32, t: i32) -> i32 {
    a + ((b - a) * t) / 100
}

#[no_mangle]
pub extern "C" fn main(_: i32) -> i32 {
    lerp(0, 100, 50)  // Should be 50
}
```

Expected: `50`

### Test 8: Local Variables in Inlined Function

```rust
#![no_std]

fn complex_calc(x: i32) -> i32 {
    let a = x * 2;
    let b = a + 3;
    let c = b * b;
    c - a
}

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    complex_calc(n)
}
```

Expected: `main(5)` → `a=10, b=13, c=169, result=159`

### Test 9: Mutual Recursion Detection

```rust
#![no_std]

fn ping(n: i32) -> i32 { if n <= 0 { 0 } else { pong(n - 1) } }
fn pong(n: i32) -> i32 { if n <= 0 { 0 } else { ping(n - 1) } }

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    ping(n)
}
```

Expected: Translation ERROR - "Recursion not supported on GPU"

### Test 10: Deep Call Chain (Stress Test)

```rust
#![no_std]

fn f1(x: i32) -> i32 { f2(x + 1) }
fn f2(x: i32) -> i32 { f3(x + 1) }
fn f3(x: i32) -> i32 { f4(x + 1) }
fn f4(x: i32) -> i32 { f5(x + 1) }
fn f5(x: i32) -> i32 { f6(x + 1) }
fn f6(x: i32) -> i32 { f7(x + 1) }
fn f7(x: i32) -> i32 { f8(x + 1) }
fn f8(x: i32) -> i32 { x + 1 }

#[no_mangle]
pub extern "C" fn main(n: i32) -> i32 {
    f1(n)  // Should be n + 8
}
```

Expected: `main(0)` → `8`

## Implementation Plan

### Phase 5a: Import Section Parsing
1. Add `imports` field to `WasmModule`
2. Parse `ImportSection` in translator
3. Calculate `import_count` for function index adjustment
4. Tests: Verify import parsing, function index mapping

### Phase 5b: Function Inlining
1. Store all function bodies (not just entry point)
2. Add `inline_function()` to TranslationContext
3. Add recursion detection
4. Handle parameter passing and local allocation
5. Tests: Helper functions, nested calls, deep chains

### Phase 5c: GPU Intrinsics
1. Add new opcodes to bytecode VM (GID, TIME, ST_OUT, etc.)
2. Implement intrinsic emission in translator
3. Create `gpu_sdk` crate with extern declarations
4. Tests: Thread ID, pixel output, atomics

### Phase 5d: Integration
1. End-to-end test with real rendering
2. Performance benchmarks
3. Documentation

## Success Metrics

| Metric | Target |
|--------|--------|
| Helper function calls | Fully inlined, zero overhead |
| Recursion detection | 100% accurate, clear errors |
| GPU intrinsics | set_pixel, thread_id, time working |
| Bytecode size | <10x source WASM size |
| Translation time | <100ms for typical programs |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Register exhaustion with deep inlining | Limit inline depth, spill to state buffer |
| Code size explosion | Track bytecode size, warn on excessive inlining |
| Indirect calls in WASM | Error with clear message - not supported |
| Complex WASM features | Maintain "unsupported" list, error early |

## Appendix: WASM Call Instruction Reference

```
call funcidx     ; Direct call to function at index
call_indirect    ; Indirect call through table (NOT SUPPORTED)
return           ; Return from function
```

WASM call uses function index, which includes imports first:
- funcidx 0..(import_count-1) = imported functions
- funcidx import_count..N = defined functions

Our translator maps:
- Import calls → GPU intrinsic emission
- Defined calls → Inline function body
