# GPU App Failure Analysis

## Summary
- **5 apps pass**: Game of Life, Snake, Pong, Clock, 2048
- **4 apps fail**: Mandelbrot, Particles, Bouncing Balls, Drawing (all return 0)

## System Architecture

We translate: **Rust** → **WASM** → **GPU Bytecode** → **Metal GPU execution**

### Key Components
1. `wasm_translator/` - Converts WASM to GPU bytecode
2. `src/gpu_os/gpu_app_system.rs` - GPU bytecode VM (Metal shader)
3. State buffer layout (after bytecode header + instructions):
   - `state[0]` = return value
   - `state[8-71]` = globals (float4 indexed)
   - `state[72-327]` = spill area (float4 indexed)
   - `state[328+]` = linear memory (byte 5248+)

### Memory Opcodes
- `LD/ST` (0x80/0x81): float4 indexed - `state[idx]` where idx is in 16-byte units
- `LD4/ST4` (0x8e/0x8f): byte addressed - converts byte addr to word index

---

## Failing App #1: Mandelbrot

### Source Code Pattern
```rust
// 64x64 nested loops with floating point Mandelbrot calculation
while py < 64 {
    while px < 64 {
        let iter = mandelbrot(x, y);  // Inline function with f32 math
        emit_quad(...);
        quad_count += 1;
    }
}
return quad_count;  // Expected: 4096
```

### Bytecode Analysis (219 instructions)
```
Key control flow:
[056] JNZ r8, 211     // if py == 64, jump to return (exit outer loop)
[081] JNZ r8, 87      // if px < 64, continue inner loop
[086] JMP 53          // jump back to outer loop start
[211] LOADI_INT r8, 0x1000  // r8 = 4096 (expected return value)
[212] MOV r4, r8      // r4 = 4096
[214] ST state[0], r4 // store result
[215] HALT
```

### Evidence: Uses LD/ST with high float4 indices
```
[072] LOADI_UINT r30, 0x248   // r30 = 584 (float4 index for inline spill)
[073] ST state[r30], r8       // Store to state[584] = byte 9344
...
[077] LOADI_UINT r30, 0x249   // r30 = 585
[078] LD r8, state[r30]       // Load from state[585]
```

### Spill Address Calculation
From `translate.rs` line 2042:
```rust
self.config.globals_base + 512 + self.inline_spill_counter * 64
// = 8 + 512 + 64 = 584 (0x248)
```

### GPU Result: 0 (expected 4096)

---

## Failing App #2: Particles

### Source Code Pattern
```rust
// Array of 50 particles with struct fields
let mut particles: [Particle; 50] = [...];
while i < 50 {
    init_particle(&mut particles[i], seed);
    update_particle(&mut particles[i], dt);  // f32 physics
}
// Render particles
while i < 50 {
    emit_quad(p.x, p.y, ...);
    quad_count += 1;
}
return quad_count;  // Expected: >= 1
```

### Key Difference from Passing Apps
- Uses arrays with struct access
- Multiple inline function calls
- Mutable references to array elements

### GPU Result: 0

---

## Failing App #3: Bouncing Balls

### Source Code Pattern
Similar to Particles - array of ball structs with physics updates.

### GPU Result: 0

---

## Failing App #4: Drawing

### Source Code Pattern
Canvas-style drawing with state.

### GPU Result: 0

---

## Passing Apps Analysis

### What They Have in Common
1. **Loops get fully unrolled by Rust compiler** - no actual LD/ST to spill addresses
2. **Simple control flow** - or control flow that doesn't depend on spilled values
3. **No complex inline function chains** - limited spill depth

### Evidence: Snake (passes, 261 instructions)
```
Uses ST4 (0x8f) with byte addresses like 0x1480 (memory_base region)
NOT LD/ST with float4 indices to spill area
Loops appear to be unrolled - no JMP/JZ for main logic
```

### Evidence: Unoptimized Loop Test (passes, 121 instructions)
```
Uses LD4/ST4 (0x8e/0x8f) with byte addresses:
[024] LOADI_UINT r30, 0x1488  // byte address
[026] LD4 r8, r30             // byte-addressed load (0x8e)
...
[048] ST4 r8, r9              // byte-addressed store (0x8f)
```

---

## Hypothesis: LD/ST vs LD4/ST4 Addressing Mismatch

### The Problem
- Spilled locals use `LD/ST` with float4 indices (e.g., state[584])
- Memory operations now use `LD4/ST4` with byte addresses
- Both should work independently, but...

### Possible Issues
1. **State buffer overlap**: Spill area (float4 72-327) may overlap with something else when accessed as bytes
2. **Inline spill addresses too high**: 584+ are beyond the designated spill area (72-327)
3. **Control flow bug**: Loops may not execute correctly due to comparison/jump issues
4. **LD/ST implementation bug**: High indices may not work correctly

### Memory Layout Question
```
Spill area: state[72-327] = bytes 1152-5247
Inline spill: state[584+] = bytes 9344+
Linear memory: bytes 5248+

CONFLICT: Inline spill (9344+) is IN the linear memory region (5248+)!
```

---

## Questions for Analysis

1. Is the inline function spill area (state[584+]) conflicting with linear memory (byte 5248+)?
2. Why do apps with unrolled loops pass while apps with actual runtime loops fail?
3. Is there a bug in how LD/ST handle high float4 indices?
4. Could the JNZ/JZ conditional jumps be broken for certain comparison results?

---

## Test Results Summary

| App | Instructions | Uses LD/ST Spill | Uses LD4/ST4 | Result |
|-----|--------------|------------------|--------------|--------|
| Game of Life | ~100 | No | Yes | PASS |
| Snake | 261 | No | Yes | PASS |
| Pong | ~100 | No | Yes | PASS |
| Clock | ~100 | No | Yes | PASS |
| 2048 | ~100 | No | Yes | PASS |
| Mandelbrot | 219 | YES (584+) | No | FAIL |
| Particles | 1548 | YES | ? | FAIL |
| Bouncing Balls | 457 | YES | ? | FAIL |
| Drawing | ~500 | YES | ? | FAIL |
