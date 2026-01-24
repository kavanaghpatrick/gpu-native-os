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

## CRITICAL: O(1) Lookup Principle

**This is the most important GPU optimization pattern. Violating it causes 10-100x slowdowns.**

### The Problem with O(log n) and O(n) on GPU

On CPU, O(log n) binary search is excellent. On GPU, it's catastrophic:

```
CPU (independent threads):
  Thread 1: finds answer in 2 steps → done
  Thread 2: finds answer in 5 steps → done
  Thread 3: finds answer in 3 steps → done
  Total: 10 steps

GPU (SIMD lockstep - all 32 threads execute SAME instruction):
  All 32 threads MUST wait for slowest thread
  Thread 1: finds in 2 steps → waits 3 more
  Thread 2: finds in 5 steps → done
  Thread 3: finds in 3 steps → waits 2 more
  Total: 32 × 5 = 160 steps (SIMD divergence)
```

### The Rule: Pre-compute → O(1) Lookup

**Trade memory for constant-time access. Pre-compute on setup, O(1) lookup at runtime.**

| Bad Pattern (runtime) | Good Pattern (pre-computed) | Speedup |
|-----------------------|----------------------------|---------|
| Walk parent chain O(depth) | Depth buffer lookup O(1) | 500x |
| Walk sibling chain O(siblings) | Cumulative height buffer O(1) | 25x |
| Linear scan O(n) | Hash table O(1) | 10,000x |
| Count from start O(chars) | Line break buffer O(1) | 10x |
| Binary search O(log n) | Chunked index O(1) | 25x |

### Implementation Checklist

Before writing any GPU kernel loop, ask:
1. **Is there a loop?** → Can I pre-compute results into a buffer?
2. **Is there a search?** → Can I use a hash table instead?
3. **Is there tree traversal?** → Can I flatten into level-order buffer?
4. **Is there conditional branching?** → Will all SIMD threads take same path?

### Examples Applied in This Codebase

```metal
// BAD: O(depth) per element - SIMD divergence disaster
while (parent >= 0) { depth++; parent = elements[parent].parent; }

// GOOD: O(1) lookup from pre-computed buffer
uint depth = depths[gid];
```

```metal
// BAD: O(siblings) per element
int sib = first_child;
while (sib != gid) { y += heights[sib]; sib = next_sibling[sib]; }

// GOOD: O(1) lookup
float y = cumulative_heights[gid];
```

```metal
// BAD: O(n) linear scan for directory lookup
for (int i = 0; i < entry_count; i++) { if (entries[i].parent == dir) ... }

// GOOD: O(1) hash table lookup
uint slot = hash(parent_inode, name_hash) & mask;
```

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
