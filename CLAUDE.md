# CLAUDE.md - GPU-Native OS Project

## PROJECT VISION: THE GPU IS THE COMPUTER

**This is NOT a "GPU-accelerated" project. This is a research sandbox proving that GPUs can REPLACE CPUs entirely.**

### Core Thesis
Modern GPUs are general-purpose parallel computers that have been artificially constrained to "graphics acceleration" by legacy software architecture. This project demonstrates that:

1. **GPU waves can replace CPU threads** - A single GPU wavefront (32-64 threads executing in lockstep) can replace traditional CPU threading models
2. **Compute shaders are programs, not effects** - Metal/CUDA compute kernels are Turing-complete; they can run any algorithm
3. **The CPU is a bottleneck, not a coordinator** - Every CPU touchpoint is technical debt we're working to eliminate
4. **Unified memory changes everything** - Apple Silicon's architecture means GPU and CPU share memory; the distinction is artificial

### What We're Proving
- Filesystem search with 3M parallel GPU threads (one per file)
- Text rendering where GPU generates geometry procedurally
- Input handling where GPU processes events directly
- Layout engines running constraint solvers on GPU
- Eventually: GPU-initiated storage access, GPU-parsed fonts, GPU networking

### Anti-Patterns (DO NOT DO)
- "Let CPU handle this, it's easier" - NO. Find the GPU solution.
- "This is a one-time operation, CPU is fine" - NO. One-time ops become patterns.
- "The CPU needs to coordinate" - NO. Use GPU atomics, barriers, and persistent kernels.
- Treating GPU as a "graphics card" - It's a MASSIVELY PARALLEL COMPUTER.

### The Goal
**Zero CPU involvement** in steady-state operation. CPU's only job: boot the system, then hand control to GPU.

---

## Architecture
```
src/lib.rs              # Library entry
src/gpu_os/
  mod.rs                # Module exports
  kernel.rs             # #11 - Unified Worker Model (core compute kernel)
  memory.rs             # #12 - Memory Architecture (GPU buffers)
  input.rs              # #13 - Input Pipeline (HID to GPU)
  layout.rs             # #14 - Layout Engine (constraint solving)
  widget.rs             # #15 - Widget System (compressed state)
  text.rs               # #16 - Text Rendering (7-segment + font atlas)
  render.rs             # #17 - Hybrid Rendering (compute + fragment)
  vsync.rs              # #18 - VSync Execution (frame timing)
  filesystem.rs         # GPU filesystem search (3M+ paths)
  text_render.rs        # Bitmap font text rendering
  shaders/kernel.metal  # GPU kernel source
examples/               # Demo applications
tests/test_issue_*.rs   # Per-issue integration tests
docs/                   # PRDs and design docs
```

## Commands
```bash
cargo check                              # Fast type check
cargo build --release                    # Optimized build
cargo run --release --example filesystem_browser  # Main demo
cargo test                               # Run all tests
```

## Key Patterns

### GPU-First Thinking
Before implementing anything, ask:
1. Can this run on GPU? (Answer is almost always YES)
2. What's the parallel decomposition? (One thread per X)
3. What CPU operations can we eliminate?
4. How do we avoid CPU readback?

### GPU Structs
Always use `#[repr(C)]` with explicit padding for Metal alignment:
```rust
#[repr(C)]
struct GpuStruct {
    data: [f32; 2],
    _padding: [f32; 2],  // Metal float4 = 16-byte alignment
}
```

### Atomic Coordination
Use GPU atomics instead of CPU synchronization:
```metal
device atomic_uint& counter;
uint slot = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
```

### Persistent Data
Keep data GPU-resident. Never round-trip to CPU unless absolutely necessary.

## Current CPU Dependencies (Technical Debt)

These are the CPU operations we're actively working to eliminate:

| Operation | Current State | Target State |
|-----------|--------------|--------------|
| Filesystem scan | CPU `read_dir()` | GPU-initiated storage access |
| Font parsing | CPU TTF parser | GPU bezier extraction |
| Font atlas generation | CPU bitmap loop | GPU compute shader |
| Query tokenization | CPU `split_whitespace()` | GPU string parsing |
| File metadata | CPU `fs::metadata()` | GPU-resident inode cache |
| Index file I/O | CPU file read/write | MTLIOCommandQueue / mmap |

## Research Areas

### GPU Direct Storage
- Metal 3 `MTLIOCommandQueue` for async file loading
- `newBufferWithBytesNoCopy` for zero-copy mmap
- Apple Silicon unified memory eliminates CPU-GPU copies
- BaM/GoFS research: GPU-initiated NVMe commands (Linux only, but informs architecture)

### GPU Font Pipeline
- GPU SDF generation via compute shaders (~1-2ms vs seconds on CPU)
- Direct bezier rendering (Slug library approach) - no atlas needed
- GPU text layout and shaping (future)

### GPU Networking (Future)
- RDMA concepts adapted for GPU
- GPU-driven packet processing

## Debug Tips
- Purple background = geometry renders but wrong color
- Nothing visible = check clip space [-1,1] range
- Corrupted data = check struct alignment/padding
- Search returns duplicates = check `max_results` initialization
- Input lag = add debouncing, check for blocking `wait_until_completed()`

## Success Metrics
1. **CPU utilization during steady state** - Target: <5%
2. **GPU thread utilization** - Target: >80% occupancy
3. **Data round-trips to CPU** - Target: Zero per frame
4. **Lines of CPU code vs GPU code** - Track ratio over time
