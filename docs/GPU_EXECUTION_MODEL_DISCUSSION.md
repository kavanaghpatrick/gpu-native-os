# GPU Execution Model Discussion

**Date**: January 2026
**Topic**: Handling sequential vs parallel Rust code on Apple Silicon GPUs

---

## The Problem

Removing the instruction limit from the bytecode interpreter caused a system hang:

```diff
-    while (running && pc < header->code_size && max_instructions-- > 0) {
+    while (running && pc < header->code_size) {
```

The code comment claimed "Metal's watchdog timeout (~2-5 seconds) catches infinite loops" - **this is FALSE**. There is no reliable GPU watchdog on macOS. The hang occurred because single-threaded GPU execution of ~1M instructions takes seconds, blocking the display compositor.

---

## Core Challenge

How to efficiently execute both:
1. **Sequential Rust code** (Game of Life with nested loops, ~500K+ instructions)
2. **Parallel Rust code** (image processing, physics simulations)

Without massive speed tradeoffs or system hangs.

---

## Apple Silicon GPU Architecture

### Hardware Hierarchy

```
GPU (M1)
├── 8 GPU Cores
│   └── Each core: 24 threadgroups (192 total on chip)
│       └── Each threadgroup: 1-32 SIMD groups
│           └── Each SIMD group: 32 threads (lockstep execution)
```

### Key Numbers

| Resource | Count | Notes |
|----------|-------|-------|
| GPU Cores | 8 | M1, varies by chip |
| Threadgroups per core | 24 | 192 total concurrent |
| Max threads per threadgroup | 1024 | Metal limit |
| SIMD group size | 32 | Always 32 on Apple Silicon |
| Threadgroup memory | 32KB | Shared within threadgroup |
| Memory bandwidth | 100-150 GB/s | Unified memory |

### Critical Insight: SIMD Lockstep Execution

All 32 threads in a SIMD group execute the **same instruction simultaneously**:

```
CPU (independent threads):
  Thread 1: if (x > 0) → takes branch
  Thread 2: if (x > 0) → skips branch
  Thread 3: if (x > 0) → takes branch

GPU SIMD group (lockstep):
  All 32 threads: evaluate condition
  All 32 threads: execute TRUE branch (some masked off)
  All 32 threads: execute FALSE branch (others masked off)
  Result: BOTH branches always execute!
```

This is called **SIMD divergence** - the penalty for conditional code.

---

## Two Agent Models

### Model A: Threadgroup = Agent (Recommended for Independent Work)

```
Threadgroup 0 (Agent A) ──────────────────────────────────
  SIMD 0: threads 0-31   [all work on Agent A's task]
  SIMD 1: threads 32-63  [all work on Agent A's task]
  ...

Threadgroup 1 (Agent B) ──────────────────────────────────
  SIMD 0: threads 0-31   [all work on Agent B's task]
  SIMD 1: threads 32-63  [all work on Agent B's task]
  ...
```

- **192 agents possible** (M1)
- **Fully independent** - no synchronization needed
- **Best for**: Different tasks, different code paths

### Model B: SIMD Group = Agent (For Similar Work)

```
Threadgroup 0 ─────────────────────────────────────────────
  SIMD 0 (Agent A): threads 0-31
  SIMD 1 (Agent B): threads 32-63
  SIMD 2 (Agent C): threads 64-95
  ...must sync at barriers...
```

- **1536 agents possible** (192 threadgroups × 8 SIMD groups average)
- **Must synchronize** at threadgroup barriers
- **Best for**: Same algorithm, different data

### Choosing the Right Model

| Scenario | Recommended Model |
|----------|-------------------|
| 10 independent agents doing different work | 10 threadgroups × 1 SIMD each |
| 10 agents doing same algorithm | 2 threadgroups × 5 SIMDs each |
| Mixed workload | Hybrid - assign based on task type |

---

## Hybrid Execution Model

### Bytecode Classes

Instead of executing all code the same way, classify bytecode by parallelism potential:

| Class | Execution | Efficiency | Use Case |
|-------|-----------|------------|----------|
| **Scalar** | 1 lane active, 31 idle | 3% | Sequential logic, control flow |
| **Vector** | All 32 lanes active | 100% | Array operations, parallel loops |
| **Reduction** | SIMD intrinsics | 100% | Sum, min, max, prefix sum |
| **Broadcast** | 1 compute, 32 receive | 3% compute, 100% distribute | Shared calculations |
| **Shuffle** | Lane exchange | 100% | Neighbor communication |

### SIMD Intrinsics (Metal)

```metal
// Reduction - O(1) instead of O(n) loop
float sum = simd_sum(value);           // Sum all 32 values
float max = simd_max(value);           // Find maximum
float min = simd_min(value);           // Find minimum

// Communication
float broadcast = simd_broadcast(value, 0);  // Lane 0's value to all
float neighbor = simd_shuffle_down(value, 1); // Get value from lane+1
float prefix = simd_prefix_exclusive_sum(value); // Running sum

// Voting
bool any = simd_any(condition);        // True if any lane true
bool all = simd_all(condition);        // True if all lanes true
```

### Automatic Detection in WASM Translator

The translator should detect patterns and emit appropriate bytecode class:

```rust
// Input: Sequential WASM loop
loop {
    sum += array[i];
    i += 1;
    if i >= len { break; }
}

// Output: Parallel reduction bytecode
PARALLEL_LOAD array, lanes 0-31
SIMD_SUM -> sum
```

---

## Unified Memory Architecture

Apple Silicon's unified memory means CPU and GPU share the **same physical RAM**:

```
┌─────────────────────────────────────────────────────────┐
│                    Physical RAM                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Shared Buffer                       │    │
│  │  [App State] [Results] [Debug Info]             │    │
│  └─────────────────────────────────────────────────┘    │
│         ↑                    ↑                          │
│    GPU reads/writes     CPU reads/writes                │
│    (no copy needed)     (no copy needed)                │
└─────────────────────────────────────────────────────────┘
```

**Key implication**: CPU can read GPU results instantly without any data transfer. Just read from the same memory address.

### Storage Modes

| Mode | Use Case |
|------|----------|
| `StorageModePrivate` | GPU-only data, fastest |
| `StorageModeShared` | CPU+GPU access, unified memory |
| `StorageModeManaged` | macOS only, explicit sync (avoid) |

---

## Implementation Recommendations

### 1. Restore Instruction Limit for Sequential Apps

```metal
// Safe limit for sequential execution
uint max_instructions = 100000;

while (running && pc < header->code_size && max_instructions-- > 0) {
    // ... interpreter loop
}

// If limit reached, yield and continue next frame
if (max_instructions == 0) {
    app_state->pc = pc;  // Save progress
    app_state->status = YIELDED;
}
```

### 2. Add Bytecode Classes

```metal
enum BytecodeClass {
    SCALAR,      // Execute on lane 0 only
    VECTOR,      // Execute on all 32 lanes
    REDUCTION,   // Use SIMD intrinsics
    BROADCAST,   // Lane 0 computes, all receive
    SHUFFLE      // Lane-to-lane exchange
};
```

### 3. Parallel App Detection

In the WASM translator, detect parallelizable patterns:
- Independent loop iterations
- Array operations without dependencies
- Map/filter/reduce patterns

### 4. Multi-Threadgroup Dispatch

For truly parallel apps, dispatch multiple threadgroups:

```rust
// Sequential app: 1 threadgroup, 1 SIMD group
encoder.dispatch_threadgroups(
    MTLSize { width: 1, height: 1, depth: 1 },
    MTLSize { width: 32, height: 1, depth: 1 }  // 1 SIMD
);

// Parallel app: multiple threadgroups
encoder.dispatch_threadgroups(
    MTLSize { width: 192, height: 1, depth: 1 },  // 192 threadgroups
    MTLSize { width: 1024, height: 1, depth: 1 }  // Full utilization
);
```

---

## Key Takeaways

1. **No GPU watchdog on macOS** - Cannot rely on Metal to kill runaway shaders
2. **SIMD divergence is real** - Conditional code executes ALL branches
3. **Threadgroups are independent** - Use them as isolated "agents"
4. **SIMD intrinsics are fast** - Replace loops with hardware operations
5. **Unified memory is zero-copy** - No transfer needed between CPU/GPU
6. **Detect parallelism at translate time** - Emit appropriate bytecode class
7. **Chunk sequential execution** - Yield and resume across frames

---

## Files Referenced

- `src/gpu_os/gpu_app_system.rs` - Bytecode interpreter
- `wasm_translator/src/translate.rs` - WASM to bytecode translator
- `test_programs/apps/game_of_life/src/lib.rs` - Sequential test app
- `src/gpu_os/shaders/kernel.metal` - Core GPU kernel
- `docs/PRD_GPU_BYTECODE_VM.md` - Original design doc

---

## Next Steps

1. Restore instruction limit with yielding mechanism
2. Add bytecode class field to instruction format
3. Implement parallelism detection in translator
4. Add SIMD intrinsic bytecode ops (SIMD_SUM, SIMD_BROADCAST, etc.)
5. Test with both sequential (Game of Life) and parallel (image filter) apps
