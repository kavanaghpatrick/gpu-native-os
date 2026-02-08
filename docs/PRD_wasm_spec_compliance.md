# PRD: WASM Spec Compliance — Run Any Rust App on GPU

**Issue**: TBD
**Priority**: P0 - Strategic
**Date**: 2026-02-08

## Vision

**Any `no_std` Rust crate that compiles to `wasm32-unknown-unknown` runs unmodified on the GPU.**

No custom APIs. No platform-specific code. Write normal Rust, compile to WASM, it runs on the GPU compute shader. Visual apps get a framebuffer in WASM linear memory — write pixels, GPU displays them.

This isn't a toy VM. It's a WASM runtime that happens to execute on GPU instead of CPU.

## Current State: Surprisingly Close

The VM already has **220 opcodes** and covers most of the WASM MVP spec:

| Feature | Status | Gap |
|---------|--------|-----|
| i32 arithmetic + bitwise | **Implemented** (broken convention) | Bit-cast fix needed |
| i64 full support | **Implemented** (double-register) | Bit-cast fix for I64Const |
| f32 IEEE 754 | **Implemented** | Working |
| f64 double-precision | **Implemented** (double-single emulation, 47-bit) | Working |
| Control flow (block/loop/if/br) | **Implemented** | Working |
| Memory (load/store all widths) | **Implemented** | Bit-cast fix needed |
| Bulk memory (copy/fill/grow) | **Implemented** | Working |
| Function calls (direct) | **Implemented** (inlined) | Working |
| Indirect calls (call_indirect) | **Implemented** (static dispatch) | Working |
| Type conversions (all) | **Implemented** | Some need bit-cast fix |
| Atomic operations (14 ops) | **Implemented** | Working |
| GPU slab allocator | **Implemented** | Working |
| WASI stubs (fd_read/write, etc.) | **Implemented** | Basic |
| Select instruction | **Implemented** | Working |
| Sign extension ops | **Implemented** | Working |
| Saturating truncation | **Implemented** | Working |

### What's Broken (Issue #298)

The integer representation convention (Issue #213) broke the bridge between integers and memory. Values > 2^24 lose precision, memory round-trips corrupt data, branch instructions fail for edge cases. This is a ~45 handler fix. All the infrastructure exists — it just stores integers wrong.

### What's Missing

| Feature | Impact | Effort |
|---------|--------|--------|
| **Recursion** | Blocks any Rust code with recursive data structures, tree traversal, parsing | Hard — needs GPU call stack |
| **br_table** | Blocks Rust `match` with many arms (compiler generates jump tables) | Medium — emit as if-else chain or jump table |
| **Framebuffer output** | Blocks visual apps beyond colored rectangles | Small — memory region convention + display pass |
| **Multi-value returns** | Blocks some Rust patterns, tuple returns | Medium |
| **120KB memory limit** | Blocks apps with large data (images, fonts) | Medium — larger GPU buffer allocation |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER'S RUST CODE                      │
│                                                         │
│  use tiny_skia::*;     // Any no_std Rust crate         │
│  let mut pm = Pixmap::new(800, 600).unwrap();           │
│  let path = PathBuilder::from_circle(400., 300., 100.); │
│  pm.fill_path(&path, &paint, FillRule::Winding, ...);   │
│  present(&pm);         // Write pixels to framebuffer   │
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │ cargo build --target wasm32-unknown-unknown
              ▼
┌─────────────────────────────────────────────────────────┐
│                    WASM BINARY (.wasm)                   │
│  Standard WASM bytecode — nothing custom                │
└─────────────┬───────────────────────────────────────────┘
              │ wasm_translator (Rust, runs once at load)
              ▼
┌─────────────────────────────────────────────────────────┐
│               GPU BYTECODE (220 opcodes)                │
│  Optimized for Metal compute shader execution           │
└─────────────┬───────────────────────────────────────────┘
              │ Dispatched to GPU
              ▼
┌─────────────────────────────────────────────────────────┐
│              METAL COMPUTE SHADER (GPU VM)               │
│                                                         │
│  Executes ALL bytecode on GPU:                          │
│  - Integer math → GPU ALU                               │
│  - Float math → GPU FPU                                 │
│  - Memory access → GPU shared memory                    │
│  - Pixel writes → GPU framebuffer region                │
│  - Display compositing → GPU render pass                │
│                                                         │
│  CPU involvement: ZERO (after initial dispatch)         │
└─────────────────────────────────────────────────────────┘
```

## Phases

### Phase 1: Correct Integer Semantics (Issue #298) — NOW

**Goal**: Make the existing 220 opcodes work correctly.

**What**: Revert integer representation to bit-cast convention. ~45 opcode handlers need `float()` → `as_type<float>()` changes. See [PRD_bitcast_revert.md](PRD_bitcast_revert.md) for full details.

**Unblocks**: All existing WASM apps render correctly (positions + colors). Memory round-trips work. Foundation for everything else.

**Effort**: 1-2 days
**Risk**: Low (mechanical changes, well-understood)
**Test**: Bouncing balls shows 5 balls at correct positions with correct colors.

### Phase 2: Framebuffer Display — HIGH IMPACT, LOW EFFORT

**Goal**: WASM apps write pixels to memory, GPU displays them.

**What**:
1. Reserve a region of WASM linear memory as a pixel buffer (e.g., last 1MB of memory)
2. Add one intrinsic: `present(ptr: *const u8, width: u32, height: u32)`
3. GPU display pass reads the pixel buffer and renders it as a textured full-screen quad
4. Keep emit_quad for simple apps that don't need full rendering

**API** (from Rust WASM app perspective):
```rust
extern "C" {
    fn present(ptr: *const u8, width: u32, height: u32);
}

// Write RGBA pixels starting at some address
let fb: &mut [u32] = unsafe {
    core::slice::from_raw_parts_mut(FB_BASE as *mut u32, WIDTH * HEIGHT)
};
fb[y * WIDTH + x] = 0xFF0000FF;  // Red pixel
unsafe { present(FB_BASE as *const u8, WIDTH as u32, HEIGHT as u32); }
```

**Unblocks**: Any software renderer compiled to WASM can produce visual output.

**Effort**: 2-3 days
**Risk**: Low (GPU already has the memory; just need a texture copy + display quad)
**Test**: WASM app fills framebuffer with gradient, displays on screen.

### Phase 3: br_table Support — ENABLES REAL RUST CODE

**Goal**: Support Rust `match` statements with many arms.

**What**: The Rust compiler generates `br_table` (jump table) for match expressions with >4 arms. Without it, many real Rust patterns fail to compile to our VM.

**Implementation**: Either:
- A) Translate br_table to a chain of if-else comparisons (simple, O(n))
- B) Add a GPU jump table opcode (O(1) but new opcode needed)

**Unblocks**: Most real Rust code patterns. match, enum dispatch, state machines.

**Effort**: 1-2 days
**Risk**: Low
**Test**: Rust match statement with 10+ arms compiles and runs correctly.

### Phase 4: Recursion Support — ENABLES REAL LIBRARIES

**Goal**: Support recursive function calls on GPU.

**What**: Currently blocked because the GPU has no call stack. Options:
- A) **Software call stack in GPU memory**: Allocate a stack region in WASM linear memory, push/pop frames manually. Each function call saves registers + return address.
- B) **Tail-call optimization**: Convert tail-recursive patterns to loops (limited but handles many cases).
- C) **Bounded recursion with depth limit**: Allow recursion up to N levels (e.g., 64), reject deeper.

**Recommendation**: Option A (software call stack). It's what every WASM runtime does — our stack just lives in GPU memory instead of CPU memory.

**Unblocks**: Tree traversal, recursive data structures, parsing, sorting algorithms (quicksort), most Rust libraries.

**Effort**: 1-2 weeks
**Risk**: Medium (register save/restore across calls, stack overflow handling)
**Test**: Recursive fibonacci, quicksort, tree traversal all run correctly.

### Phase 5: Expanded Memory — ENABLES REAL APPLICATIONS

**Goal**: Support apps with significant data (images, fonts, buffers).

**What**: Current limit is ~120KB usable WASM memory. Real apps need:
- Font data: 50-500KB per font
- Framebuffer: 800×600×4 = 1.9MB, 1920×1080×4 = 8.3MB
- Working buffers: varies

**Implementation**:
- Increase GPU buffer allocation per app (Metal allows up to 256MB on Apple Silicon)
- Support memory.grow to expand at runtime
- Consider memory-mapped approach for large data

**Unblocks**: Image processing, font rendering, any app with non-trivial data.

**Effort**: 3-5 days
**Risk**: Low (Metal supports large buffers natively)
**Test**: WASM app allocates 4MB, fills framebuffer at 1024×768.

### Phase 6: First Real Library — PROOF OF CONCEPT

**Goal**: Run a real Rust `no_std` 2D library on the GPU VM.

**Candidates**:

| Library | Size | Dependencies | Feasibility |
|---------|------|-------------|-------------|
| **embedded-graphics** | Small | no_std, pure Rust | High — designed for constrained environments |
| **tiny-skia** | Medium | no_std optional | Medium — some features need alloc |
| **minifb** | Small | Framebuffer only | High — just pixel buffer |
| **raqote** | Large | Many deps | Low — too many dependencies |

**Recommendation**: Start with `embedded-graphics`. It's designed for `no_std` embedded systems — exactly our constraint. It provides:
- Geometric primitives (lines, circles, rectangles, triangles)
- Text rendering (bitmap fonts included)
- Image drawing
- Styled drawing (fill, stroke, colors)
- Zero heap allocation in core

**Integration**:
```rust
#![no_std]
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::*;
use embedded_graphics::pixelcolor::Rgb888;

struct GpuDisplay { /* wraps framebuffer memory */ }
impl DrawTarget for GpuDisplay {
    type Color = Rgb888;
    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where I: IntoIterator<Item = Pixel<Self::Color>> {
        for Pixel(coord, color) in pixels {
            self.framebuffer[coord.y * WIDTH + coord.x] = color.into();
        }
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    let mut display = GpuDisplay::new();
    Circle::new(Point::new(100, 100), 50)
        .into_styled(PrimitiveStyle::with_fill(Rgb888::RED))
        .draw(&mut display).unwrap();
    display.present();
    0
}
```

**Unblocks**: Real visual applications with text, shapes, and images — all running on GPU.

**Effort**: 1 week (including integration work)
**Risk**: Medium (may hit missing WASM features, iterate to fix)
**Test**: embedded-graphics demo app renders circles, text, and rectangles on screen.

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Existing WASM apps render correctly | 8/8 apps with correct positions + colors |
| 2 | Pixel-level rendering works | Gradient fills screen, no artifacts |
| 3 | Rust match expressions work | 10-arm match compiles and runs |
| 4 | Recursive algorithms work | Fibonacci(20) = 6765 on GPU |
| 5 | Large framebuffer works | 1024×768 RGBA buffer fills and displays |
| 6 | Real library runs | embedded-graphics demo renders shapes + text |

## What This Means

After Phase 6, any Rust developer can:

```bash
# Write a visual app in normal Rust
cargo new my_gpu_app --lib
# Add embedded-graphics (or any no_std 2D library)
# Implement DrawTarget for the GPU framebuffer
# Compile to WASM
cargo build --target wasm32-unknown-unknown --release
# Run on GPU
cargo run --example visual_wasm_apps
# Press key to load my_gpu_app.wasm → renders on GPU
```

No Metal knowledge. No GPU programming. No custom APIs beyond `present()`. Just Rust.

## Non-Goals (Explicitly Out of Scope)

- **3D rendering** — This is a 2D GPU compute platform, not a game engine
- **WASM threads** — Single-threaded WASM execution (GPU parallelism is within the VM, not exposed to WASM)
- **Full WASI** — File I/O, networking etc. are future work
- **WASM SIMD** — The GPU has native SIMD via float4 ops, but WASM SIMD128 spec is not targeted
- **Vulkan/cross-platform** — Metal-only for now; portability is a separate project
- **Hot reloading** — Load WASM once, run until exit
- **Multi-app compositing** — One app fills the screen; window management is future work

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Bit-cast fix | 1-2 days | Week 1 |
| Phase 2: Framebuffer | 2-3 days | Week 1 |
| Phase 3: br_table | 1-2 days | Week 2 |
| Phase 4: Recursion | 1-2 weeks | Week 3-4 |
| Phase 5: Memory expansion | 3-5 days | Week 4 |
| Phase 6: embedded-graphics | 1 week | Week 5 |

**5 weeks from broken squares to real visual apps running on GPU.**

## Appendix: WASM Feature Coverage (Current)

```
✅ = Implemented and working
🔧 = Implemented but broken (needs bit-cast fix)
❌ = Not implemented

WASM MVP:
  ✅ i32 operations (all)           🔧 Integer representation
  ✅ i64 operations (all)           🔧 I64Const encoding
  ✅ f32 operations (all)           ✅ Working
  ✅ f64 operations (emulated)      ✅ Working (47-bit precision)
  ✅ Control flow                   ✅ Working
  ✅ Function calls (direct)        ✅ Working (inlined)
  ✅ Indirect calls                 ✅ Working (static dispatch)
  ✅ Local/global variables         ✅ Working
  ✅ Memory load/store              🔧 Needs bit-cast fix
  ✅ Constants                      🔧 I64Const needs fix
  ✅ Select                         ✅ Working
  ✅ Drop                           ✅ Working
  ❌ br_table                       Phase 3
  ❌ Multi-value returns            Future
  ❌ Recursion                      Phase 4

Extensions:
  ✅ Bulk memory (copy/fill/grow)   ✅ Working
  ✅ Sign extension ops             ✅ Working
  ✅ Saturating truncation           ✅ Working
  ✅ Atomics (14 ops)               ✅ Working
  ✅ GPU slab allocator             ✅ Working
  ✅ WASI stubs                     ✅ Basic
  ❌ Reference types                Future
  ❌ SIMD128                        Not planned
  ❌ Exception handling             Not planned
  ❌ Threads                        Not planned
```
