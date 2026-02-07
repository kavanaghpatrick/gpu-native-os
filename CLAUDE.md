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

---

## CRITICAL: NO BANDAIDS - PRODUCTION CODE ONLY

**This project writes PRODUCTION-READY code. No workarounds. No hacks. No "temporary" fixes.**

### The Rule

When code doesn't work on our platform, we have exactly TWO options:

1. **FIX THE PLATFORM** - Make our translator/VM support the code
2. **CREATE A GITHUB ISSUE** - Document the gap, track it, fix it later

**NEVER:**
- Modify user code to work around platform limitations
- Add "simplified versions" that avoid unsupported features
- Comment out failing parts with "TODO: fix later"
- Create fake implementations that don't do what they claim

### Example: Wrong vs Right

**WRONG (Bandaid):**
```rust
// Original uses i64, but we don't support it, so use i32
const FNV_OFFSET: u32 = 0x811c9dc5;  // Changed from u64!
```

**RIGHT (Production):**
```rust
// Original algorithm - unchanged
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
// If this fails, see GitHub Issue #188 for i64 support
```

### When You Find a Gap

1. **Stop** - Don't try to work around it
2. **Create GitHub Issue** - Document exactly what's unsupported
3. **Keep the original code** - Let the test fail honestly
4. **Move on** - The issue is tracked, it will be fixed properly

### Why This Matters

**THE GPU IS THE COMPUTER** means users write normal Rust and it runs on GPU unchanged. If we allow bandaids:
- Users can't trust their code runs correctly
- We hide platform gaps instead of fixing them
- Technical debt compounds
- The project becomes unmaintainable

**Our job is to make the platform support ALL valid Rust code, not to make Rust code work around our limitations.**

### The Goal
**Zero CPU involvement** in steady-state operation. CPU has exactly TWO jobs:
1. **Boot** - Initialize hardware, load GPU kernel, then exit the loop
2. **Network I/O** - Receive packets from NIC (hardware limitation)

Everything else - process spawning, file I/O, input handling, rendering, scheduling - runs on GPU.

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

### CRITICAL: Metal float3 vs packed_float3
**This causes subtle rendering bugs (lines instead of quads, corrupted vertices).**

| Rust Type | Metal Type | Size | Notes |
|-----------|------------|------|-------|
| `[f32; 3]` | `packed_float3` | 12 bytes | **USE THIS** |
| N/A | `float3` | 16 bytes | NEVER use for vertex data |

Metal's `float3` is 16 bytes (padded to float4), but Rust's `[f32; 3]` is 12 bytes.
Using `float3` in Metal shaders causes vertex stride mismatch and corrupted reads.

```metal
// WRONG - causes vertex misalignment
struct Vertex {
    float3 position;  // 16 bytes - WRONG!
    float4 color;
};

// CORRECT - matches Rust [f32; 3]
struct Vertex {
    packed_float3 position;  // 12 bytes - matches Rust
    float _pad0;             // explicit padding
    float4 color;
};
```

**Symptoms of this bug:** Geometry renders as lines from origin, triangles have wrong vertices.

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

## GPU-Native Data Structure Patterns

**These patterns should be the default for ALL new data structures.**

### Hash Tables: Cuckoo Hashing (NOT Linear Probing)

Linear probing causes SIMD divergence (variable iterations per thread). Cuckoo hashing is **O(1) guaranteed** - exactly 2 lookups per key, always.

```metal
// BAD: Linear probing - variable iterations = SIMD divergence
while (table[slot] != EMPTY) { slot = (slot + 1) % capacity; }

// GOOD: Cuckoo hashing - exactly 2 lookups, always
uint slot1 = hash1(key) & mask;
uint slot2 = hash2(key) & mask;
// Check slot1, then slot2 - done. No loop.
```

### Text Buffers: Edit Log (NOT Gap Buffer)

Gap buffers are CPU-optimized (O(1) at cursor, O(n) elsewhere). Edit logs are GPU-optimized:

| Feature | Gap Buffer | Edit Log |
|---------|------------|----------|
| Insert anywhere | O(n) data move | O(1) atomic append |
| Parallel insert | Sequential | N threads parallel |
| Undo | Complex state restore | Truncate pointer |
| Redo | Separate redo stack | Extend pointer |

### Allocators: Slab Allocation with Size Classes

Arbitrary-sized allocation causes fragmentation. Use fixed size classes with lock-free free lists:

```
Size classes: 64B, 128B, 256B, 512B, 1KB, 4KB, 16KB, 64KB
Each class: atomic CAS on free list head
Result: O(1) alloc/free, zero fragmentation
```

### Batch Everything

**Never design for single-element operations.** Every operation should handle N elements with N threads.

```metal
// BAD: Single insert (1 dispatch per element)
void insert(T value);

// GOOD: Batch insert (1 dispatch for 1000 elements)
void insert_batch(T* values, uint count);
```

### Storage Mode Defaults

| Use Case | Storage Mode | Why |
|----------|--------------|-----|
| Main data (GPU-only) | `StorageModePrivate` | Best GPU performance |
| Debug stats | `StorageModeShared` | CPU needs to read |
| I/O buffers | `StorageModeShared` | MTLIOCommandQueue needs access |
| Everything else | `StorageModePrivate` | Default to GPU-only |

### CPU as I/O Coprocessor

When CPU involvement is unavoidable (file I/O), use the coprocessor pattern:

```
GPU: writes request to queue (atomic)
GPU: continues working (never waits!)
GPU: polls status buffer periodically

CPU (async thread): drains request queue
CPU: dispatches MTLIOCommandQueue
CPU: completion handler updates status
(GPU sees status on next poll)
```

**Key rule: GPU never waits. GPU never blocks. GPU polls and continues.**

### Anti-Patterns to Avoid

| Anti-Pattern | Why Bad | Do This Instead |
|--------------|---------|-----------------|
| `if (tid != 0) return;` | Wastes 63/64 threads | Parallel everything |
| Linear probing | SIMD divergence | Cuckoo hashing |
| Gap buffer | O(n) moves | Edit log |
| Single-element ops | 1000 dispatches | Batch operations |
| `StorageModeShared` default | Slower GPU access | `StorageModePrivate` default |
| CPU manages state | CPU in loop | GPU owns state |
| Barriers in hot path | All threads wait | Lockless atomics |

---

## CRITICAL: Apple Silicon M4 Metal Kernel Limitations

**Discovered 2026-01-29: Metal on Apple Silicon M4 blocks certain loop patterns entirely.**

### The Discovery (Empirically Proven)

Testing revealed that Apple Silicon M4 has strict limits on GPU kernel loops:

| Pattern | Outcome |
|---------|---------|
| `while(true)` (truly infinite) | **BLOCKED** - Kernel never executes |
| `while(true)` with `if (i >= N) break` (N <= 20M) | **WORKS** |
| `for (i = 0; i < N; i++)` where N <= ~25M | **WORKS** |
| Any loop with N > ~25-30M iterations | **BLOCKED** - Kernel never executes |

### Key Findings

1. **Truly infinite loops are blocked** - Kernels with `while(true)` without a numeric bound never execute at all. This appears to be a Metal compiler/runtime safety feature.

2. **There's an iteration threshold of ~25-30M** - Loops with bounds > ~25-30M iterations are also blocked.

3. **GPU writes are not visible to CPU until kernel completes** - This is fundamental Metal behavior. The CPU cannot observe intermediate state updates during kernel execution.

### Solution: Pseudo-Persistent Kernels via Chaining

Instead of truly persistent kernels, use bounded kernel chunks with CPU re-dispatch:

```metal
// WRONG - never executes on M4
kernel void bad_persistent(...) {
    while (true) {  // Truly infinite - BLOCKED
        if (shutdown) break;
        do_work();
    }
}

// CORRECT - pseudo-persistent via kernel chaining
kernel void good_persistent(...) {
    // Bounded loop (~1M iterations, completes in ~0.5s)
    for (uint i = 0; i < 1000000u; i++) {
        if (shutdown) break;
        do_work();
    }
    // CPU re-dispatches after kernel completes
}
```

Rust host code uses a background thread to continuously re-dispatch:
```rust
thread::spawn(move || {
    while !shutdown.load(Ordering::Acquire) {
        cmd.commit();
        cmd.wait_until_completed();  // ~0.5s per dispatch
        // Kernel state now visible to CPU
    }
});
```

### Iteration Count Guidelines

| Iterations | Time | Recommendation |
|------------|------|----------------|
| 1M | ~0.5s | Good for responsive state updates |
| 10M | ~5-6s | Less responsive but still works |
| 20M | ~10s | At threshold - may work |
| 30M+ | N/A | **BLOCKED** - won't execute |

### All SIMD Threads Must Still Participate

The original rule still applies within the bounded loop:

```metal
kernel void persistent(..., uint tid [[thread_index_in_threadgroup]]) {
    for (uint iter = 0; iter < 1000000u; iter++) {
        if (shutdown) break;

        // ALL threads participate
        uint my_idx = tid % work_count;
        process_work(my_idx);

        // Thread 0 handles stats
        if (tid == 0 && (iter % 10000 == 0)) {
            update_stats();
        }
    }
}
```

---

## Current CPU Dependencies (Technical Debt)

These are the CPU operations we're actively working to eliminate:

| Operation | Current State | Target State | Status |
|-----------|--------------|--------------|--------|
| Filesystem scan | CPU `read_dir()` at startup | GPU-initiated storage access | **Cached** - `shared_index.rs` |
| Font parsing | CPU TTF parser | GPU bezier extraction | TODO |
| Font atlas generation | CPU bitmap loop | GPU compute shader | TODO |
| Query tokenization | ~~CPU `split_whitespace()`~~ | GPU string parsing | **DONE** - `gpu_string.rs` |
| File metadata | ~~CPU `fs::metadata()`~~ | GPU-resident inode cache | **DONE** - `shared_index.rs` |
| Index file I/O | ~~CPU file read/write~~ | MTLIOCommandQueue / mmap | **DONE** - `gpu_io.rs`, `mmap_buffer.rs` |
| File content loading | CPU reads file | GPU direct I/O | **SOLVABLE** - MTLIOCommandQueue |
| Text editing | CPU gap buffer | GPU edit log | **PLANNED** - Issue #166 |
| Event loop | CPU polls events | GPU persistent kernel | **DONE** - Issue #133, #149 |

## Proven GPU Capabilities (Research Validated)

Based on comprehensive research (see docs/GPU_CAPABILITIES_RESEARCH.md):

**GPU-Native Process Model**
- Processes are threadgroups with dedicated state buffers, NOT OS processes
- Process spawning is GPU-native: allocate state buffer + launch threadgroup
- No CPU involvement for process creation, scheduling, or termination
- Inter-process communication via GPU atomics and shared buffers

**MTLIOCommandQueue (Native Metal 3)**
- GPU-initiated file I/O
- CPU only opens handles, GPU loads data
- Zero-copy to GPU buffers

**Persistent Kernels** (Empirically Proven 2026-01-28)
- Run indefinitely with while(true)
- Poll memory atomics for events
- Proven in `tests/test_persistent_kernel_proof.rs`: 15+ seconds, 87M iterations, clean shutdown
- **CRITICAL CONSTRAINT**: All SIMD threads must participate in the loop (see below)

**Unified Memory**
- Zero-copy between CPU/GPU
- Hardware cache coherency
- No explicit DMA needed

### Unavoidable CPU Dependencies

Only these truly require CPU:
- Network packet reception (NIC hardware limitation)
- Initial boot (hardware requirement)

Everything else runs on GPU. Process spawning, file I/O, input handling, rendering - all GPU-native.

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

## Debug Tips
- Purple background = geometry renders but wrong color
- Nothing visible = check clip space [-1,1] range
- Corrupted data = check struct alignment/padding
- **Lines from origin instead of quads = using `float3` instead of `packed_float3` in Metal shader (see GPU Structs section)**
- Search returns duplicates = check `max_results` initialization
- Input lag = add debouncing, check for blocking `wait_until_completed()`
- **Kernel never executes (0 heartbeats) = `while(true)` blocked on M4; see "Apple Silicon M4 Metal Kernel Limitations" section**
- Kernel executes but CPU sees 0 during runtime = GPU writes only visible after kernel completes; use shorter iteration count (~1M)

## Success Metrics
1. **CPU utilization during steady state** - Target: <5%
2. **GPU thread utilization** - Target: >80% occupancy
3. **Data round-trips to CPU** - Target: Zero per frame
4. **Lines of CPU code vs GPU code** - Track ratio over time
