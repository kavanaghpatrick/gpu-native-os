# Honest Status Assessment: What Actually Works

**Date**: January 2026
**Purpose**: Reality check on GPU-native claims

---

## Summary

| Claim | Code Exists | Actually Works | Status |
|-------|-------------|----------------|--------|
| Persistent kernels run indefinitely | ⚠️ Partial | ❌ **NO** | Batch model, not persistent |
| GPU event loop processes events directly | ✅ Yes | ⚠️ **Limited** | Processes 1 event per dispatch |
| Atomic coordination works | ✅ Yes | ✅ **YES** | Tests pass |
| kernel.metal never idles | ✅ Yes | ⚠️ **Per-dispatch** | Phase-based but terminates |

---

## Detailed Analysis

### 1. Persistent Kernels - **DOES NOT WORK AS CLAIMED**

**What CLAUDE.md claims:**
> "Persistent Kernels - Run indefinitely with while(true)"

**What the code actually says** (`persistent_search.rs:147-154`):
```rust
// NOTE: Metal compute kernels cannot truly run indefinitely due to GPU
// execution time limits. Instead, we use a "work-triggered" model:
// - CPU submits work items to a queue
// - CPU dispatches the kernel to process pending work
// - Kernel processes ALL pending items in one dispatch
// - This amortizes dispatch overhead across multiple searches
```

**Reality**: The "persistent search" is actually **batch processing**:
- CPU calls `process_pending()` to dispatch kernel
- Kernel processes N items, then **terminates**
- CPU must dispatch again for more work

**There is NO `while(true)` loop in the actual Metal shader.**

### 2. GPU Event Loop - **PARTIALLY WORKS**

**What exists** (`event_loop.rs`):
```metal
kernel void gpu_event_loop(...) {
    if (tid != 0) return;  // Single-threaded!

    // Check for ONE input event
    if (head == tail) {
        // No input - maybe render
        return;  // EXITS immediately
    }

    // Process ONE event
    InputEvent event = input_queue->events[tail % QUEUE_SIZE];
    // ... route event
}
```

**Reality**:
- Processes **ONE event per dispatch**, not a continuous loop
- CPU must keep dispatching the kernel
- The "loop" is in the CPU code, not GPU

**This is NOT "GPU processes events directly" - CPU orchestrates every event.**

### 3. Atomic Coordination - **WORKS**

**Evidence**: Tests pass, atomics are used correctly:
```rust
test gpu_os::persistent_search::tests::test_oneshot_search ... ok
test gpu_os::persistent_search::tests::test_oneshot_case_sensitive ... ok
```

**The atomic primitives work correctly** for:
- Work queue head/tail management
- Status flags
- Match counting with `simd_sum`

### 4. kernel.metal Phase-Based Execution - **WORKS (per dispatch)**

**What exists** (`kernel.metal:122-299`):
```metal
kernel void gpu_os_kernel(...) {
    // PHASE 1: INPUT COLLECTION
    // PHASE 2: HIT TESTING
    // PHASE 3: VISIBILITY COUNTING
    // PHASE 4: BITONIC SORT
    // PHASE 5: STATE UPDATE
}
// Kernel ENDS here - no loop
```

**Reality**:
- All 1024 threads participate in all phases ✅
- Efficient parallel execution within dispatch ✅
- But kernel **terminates after one pass** - CPU must dispatch again

---

## The Core Issue

The codebase has **batch-parallel execution** (good), but claims to have **persistent execution** (not implemented).

| Model | What It Means | Status |
|-------|--------------|--------|
| **Batch-parallel** | CPU dispatches, GPU processes N items, returns | ✅ IMPLEMENTED |
| **Persistent** | GPU runs `while(true)`, polls for work forever | ❌ NOT IMPLEMENTED |

The CLAUDE.md vision says:
> "CPU has exactly TWO jobs: Boot, Network I/O"

But the reality is:
> "CPU dispatches every kernel, manages every event loop iteration"

---

## What Would Need to Change

To actually implement persistent kernels:

```metal
// TRUE persistent kernel (NOT currently implemented)
kernel void gpu_os_persistent(...) {
    while (true) {  // <-- This doesn't exist in current code
        // Poll for work
        uint head = atomic_load(&work_queue->head);
        uint tail = atomic_load(&work_queue->tail);

        if (head != tail) {
            // Process work
            process_work_item();
        }

        // Self-throttle to prevent thermal issues
        if (should_throttle()) {
            barrier_yield(COOLDOWN_CYCLES);
        }

        // Check for shutdown
        if (atomic_load(&control->shutdown)) {
            break;
        }
    }
}
```

**This pattern does NOT exist in the codebase.**

---

## Honest Comparison: Goal vs Reality

| CLAUDE.md Goal | Actual State | Gap |
|----------------|--------------|-----|
| CPU utilization <5% | **~50%+** (orchestrates every dispatch) | LARGE |
| GPU runs indefinitely | **Batch dispatch model** | FUNDAMENTAL |
| Zero CPU round-trips | **1 per kernel dispatch** | FUNDAMENTAL |
| GPU polls for work | **CPU polls, GPU processes** | INVERTED |

---

## What DOES Work Well

1. **Parallel processing within dispatches** - kernel.metal efficiently uses all 1024 threads
2. **SIMD intrinsics** - `simd_sum`, `simd_is_first` used correctly
3. **Atomic work queues** - Lock-free queue implementation works
4. **GPU search** - Parallel text search achieves real speedups
5. **Hit testing** - Parallel window hit testing works

---

## Recommendation

The debate's WasmBurst architecture is **actually closer to what the codebase implements** than what CLAUDE.md claims.

Two honest paths forward:

### Path A: Accept Batch Model
- Acknowledge CPU orchestration is fine for now
- Focus on making batch processing efficient
- Update CLAUDE.md to reflect reality

### Path B: Actually Implement Persistent Kernels
- Research: Has anyone done `while(true)` on Metal?
- Test: What happens with a real infinite loop?
- Implement: Add actual persistent kernel with self-throttling
- Validate: Prove it doesn't crash/throttle

**Before claiming "persistent kernels work", someone needs to write and test one.**

---

## Test to Run

Here's how to test if persistent kernels are actually possible:

```rust
#[test]
fn test_truly_persistent_kernel() {
    // Dispatch kernel with while(true)
    // Let it run for 10 seconds
    // Check: Does it crash? Throttle? Get killed?
    // This test DOES NOT EXIST in the codebase
}
```

Until this test exists and passes, "persistent kernels" is aspirational, not proven.
