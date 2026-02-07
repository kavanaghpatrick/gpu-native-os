# GPU Crash Prevention Guide: Lessons Learned from Persistent Kernel Development

> **CRITICAL SAFETY DOCUMENT**
>
> This document captures hard-won lessons from debugging GPU kernels that caused complete system freezes on Apple Silicon (requiring power cycles). These patterns apply to Metal compute shaders, especially persistent kernels with `while(true)` loops.

---

## Table of Contents

1. [The #1 Rule: All SIMD Threads Must Participate](#1-the-1-rule-all-simd-threads-must-participate)
2. [Struct Alignment Checklist](#2-struct-alignment-checklist)
3. [Buffer Size Verification](#3-buffer-size-verification)
4. [Barrier Safety Rules](#4-barrier-safety-rules)
5. [Persistent Kernel Patterns](#5-persistent-kernel-patterns)
6. [Testing Methodology](#6-testing-methodology)
7. [Recovery Strategies](#7-recovery-strategies)
8. [Quick Reference Checklist](#8-quick-reference-checklist)

---

## 1. The #1 Rule: All SIMD Threads Must Participate

**This is the most critical rule. Violating it causes kernel stalls after ~5M iterations, leading to system freeze.**

### The Discovery (Empirically Proven)

We tested true `while(true)` loops on Apple Silicon GPUs:

| Pattern | Outcome | Iterations |
|---------|---------|------------|
| `if (tid != 0) return;` then while(true) | **STALLS/CRASH** | ~5M then frozen |
| All 32 threads in while(true) | **WORKS** | 87M+ (ran 15+ seconds) |

### Why Single-Thread Loops Crash

GPUs execute in SIMD groups of 32 threads. When only thread 0 runs a loop:

```metal
Thread 0: while(true) { work(); }     <- Active
Thread 1-31: return; // masked out   <- Still consuming resources!
```

The masked threads aren't truly idle - they're **diverged**. After ~5M iterations, the hardware stalls (likely resource exhaustion or scheduler timeout). On Apple Silicon, this manifests as a complete system freeze requiring power cycle.

### The Rule

**Every `while(true)` loop must have ALL threads participating:**

```metal
// WRONG - will stall after ~5M iterations
kernel void bad_persistent(...) {
    if (tid != 0) return;  // <-- FATAL: 31 threads masked
    while (true) {
        do_work();
    }
}

// CORRECT - runs indefinitely
kernel void good_persistent(...) {
    while (true) {
        // ALL threads participate in the loop
        uint my_work_idx = tid % work_count;
        process_work(my_work_idx);

        // Thread 0 handles coordination
        if (tid == 0) {
            update_counters();
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ALL threads check shutdown
        if (shutdown_flag) break;
    }
}
```

### Dangerous Patterns to Avoid

| Anti-Pattern | Why Dangerous | Safe Alternative |
|--------------|---------------|------------------|
| `if (tid != 0) return;` before while loop | 31 threads diverged indefinitely | All threads enter loop, use conditional inside |
| `while(queue not empty)` on thread 0 only | Same issue - single-thread loop | All threads process bounded items, barrier sync |
| Nested while loop on one thread | Compounds divergence | Parallelize across threads, bound iterations |

### Refactoring Guide

**Before (CRASHES):**
```metal
kernel void process_spawn_queue(...) {
    // Thread 0 processes all spawn requests
    if (tid == 0) {
        while (spawn_tail > spawn_head) {  // CRASH!
            // process request
            spawn_head++;
        }
    }
}
```

**After (SAFE):**
```metal
kernel void process_spawn_queue(...) {
    // ALL threads participate - each handles one request
    uint queue_len = spawn_tail - spawn_head;

    if (tid < queue_len && tid < MAX_SPAWN_BATCH) {
        // Each thread handles ONE request
        SpawnRequest req = spawn_queue[(spawn_head + tid) % QUEUE_SIZE];
        process_request(req);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0 updates head AFTER all threads done
    if (tid == 0) {
        uint processed = min(queue_len, MAX_SPAWN_BATCH);
        spawn_head += processed;
    }
}
```

---

## 2. Struct Alignment Checklist

**Metal requires 16-byte alignment for all GPU structs. Misalignment causes corruption and crashes.**

### Required Annotations

#### Rust Side
```rust
#[repr(C)]                    // REQUIRED: C-compatible memory layout
#[derive(Clone, Copy)]        // Usually needed for GPU structs
pub struct GpuStruct {
    pub field1: u32,
    pub field2: u32,
    // ... fields ...
    pub _padding: [u32; N],   // Pad to 16-byte multiple
}

// CRITICAL: Compile-time verification
const _: () = assert!(std::mem::size_of::<GpuStruct>() % 16 == 0);
const _: () = assert!(std::mem::size_of::<GpuStruct>() == EXPECTED_SIZE);
```

#### Metal Side
```metal
// Match Rust struct EXACTLY - field order, sizes, padding
struct GpuStruct {
    uint field1;              // 0-3
    uint field2;              // 4-7
    // ... fields matching Rust layout ...
    uint _padding[N];         // Match Rust padding
};
```

### Common Alignment Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing `#[repr(C)]` | Random data corruption | Add `#[repr(C)]` to all GPU structs |
| Metal `float3` vs Rust `[f32; 3]` | Vertex stride mismatch, geometry corruption | Use `packed_float3` in Metal |
| Struct size not 16-byte aligned | Buffer offset errors, crashes | Add explicit padding fields |
| Different padding in Rust vs Metal | Field misalignment | Document byte offsets in comments |

### The float3 Problem (CRITICAL)

Metal's `float3` is 16 bytes (padded like float4), but Rust's `[f32; 3]` is 12 bytes:

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

### Process Struct Example (432 bytes)

```rust
// Rust
#[repr(C)]
pub struct Process {
    pub pc: u32,                    // 0-3
    pub sp: u32,                    // 4-7
    pub status: u32,                // 8-11
    pub bytecode_offset: u32,       // 12-15
    pub bytecode_len: u32,          // 16-19
    pub heap_offset: u32,           // 20-23
    pub heap_size: u32,             // 24-27
    pub regs: [i32; 64],            // 28-283 (256 bytes)
    pub fregs: [f32; 32],           // 284-411 (128 bytes)
    pub blocked_on: u32,            // 412-415
    pub priority: u32,              // 416-419
    pub _padding: [u32; 3],         // 420-431 (12 bytes) -> 432 total
}

const _: () = assert!(std::mem::size_of::<Process>() == 432);
const _: () = assert!(std::mem::size_of::<Process>() % 16 == 0);
```

```metal
// Metal - must match exactly
struct Process {
    uint pc;                    // 0-3
    uint sp;                    // 4-7
    uint status;                // 8-11 - USE ATOMIC ACCESS
    uint bytecode_offset;       // 12-15
    uint bytecode_len;          // 16-19
    uint heap_offset;           // 20-23
    uint heap_size;             // 24-27
    int regs[64];               // 28-283 (256 bytes)
    float fregs[32];            // 284-411 (128 bytes)
    uint blocked_on;            // 412-415
    uint priority;              // 416-419
    uint _padding[3];           // 420-431 -> 432 total
};
```

---

## 3. Buffer Size Verification

**Insufficient buffer sizes cause out-of-bounds reads/writes that can crash the GPU.**

### Buffer Creation Checklist

```rust
// 1. Document expected sizes with constants
const MAX_PROCESSES: usize = 64;
const PROCESS_SIZE: usize = 432;  // Verified by const assert
const PROCESS_BUFFER_SIZE: usize = MAX_PROCESSES * PROCESS_SIZE;

// 2. Create buffers with verified sizes
let processes = device.new_buffer(
    PROCESS_BUFFER_SIZE as u64,
    MTLResourceOptions::StorageModeShared,
);

// 3. Verify buffer length after creation
assert!(processes.length() >= PROCESS_BUFFER_SIZE as u64);
```

### Size Calculation Patterns

| Buffer Type | Size Calculation | Verification |
|-------------|------------------|--------------|
| Array buffer | `count * element_size` | `assert!(buf.length() >= count * size)` |
| Struct buffer | `sizeof::<Struct>()` | Const assert on struct size |
| Variable-length | Header + max_items * item_size | Document header layout |

### Variable-Length Buffer Example

```rust
// Input queue: 16-byte header + 256 * 16-byte events
const INPUT_HEADER_SIZE: usize = 16;  // head, tail, padding
const MAX_INPUT_EVENTS: usize = 256;
const INPUT_EVENT_SIZE: usize = 16;
const INPUT_BUFFER_SIZE: usize = INPUT_HEADER_SIZE + MAX_INPUT_EVENTS * INPUT_EVENT_SIZE;

let input_events = device.new_buffer(
    INPUT_BUFFER_SIZE as u64,
    MTLResourceOptions::StorageModeShared,
);
```

### Bounds Checking in Shaders

**Always bounds-check before accessing arrays:**

```metal
// WRONG - no bounds check
int value = data[index];  // May crash if index >= buffer size

// CORRECT - bounds check first
if (index < buffer_size) {
    int value = data[index];
} else {
    value = 0;  // Safe fallback
}
```

---

## 4. Barrier Safety Rules

**Misused barriers cause deadlocks or undefined behavior.**

### Rule 1: All Threads Must Hit the Same Barrier

```metal
// WRONG - some threads skip barrier
if (tid < 10) {
    threadgroup_barrier(mem_flags::mem_device);  // DEADLOCK!
}

// CORRECT - all threads hit barrier
// (conditional work, unconditional barrier)
if (tid < 10) {
    do_work();
}
threadgroup_barrier(mem_flags::mem_device);  // All 32 threads hit this
```

### Rule 2: No Barriers Inside Conditional Paths

```metal
// WRONG - barrier in conditional
if (condition_that_varies_per_thread) {
    threadgroup_barrier(mem_flags::mem_device);  // UNDEFINED!
}

// CORRECT - move barrier outside conditional
if (condition_that_varies_per_thread) {
    do_work();
}
// All threads reach this point
threadgroup_barrier(mem_flags::mem_device);
```

### Rule 3: No Barriers in While Loops (Unless ALL Threads Loop)

```metal
// WRONG - barrier inside potentially-divergent loop
while (my_work_left > 0) {
    do_work();
    my_work_left--;
    threadgroup_barrier(mem_flags::mem_device);  // DEADLOCK if work differs!
}

// CORRECT - fixed iteration count, all threads loop same times
for (uint i = 0; i < MAX_ITERATIONS; i++) {
    if (i < my_work_count) {
        do_work();
    }
    threadgroup_barrier(mem_flags::mem_device);  // Safe - all threads hit
}
```

### Memory Flags

| Flag | When to Use |
|------|-------------|
| `mem_flags::mem_threadgroup` | Syncing threadgroup-local memory |
| `mem_flags::mem_device` | Syncing device (global) memory |
| `mem_flags::mem_texture` | Syncing texture memory |

---

## 5. Persistent Kernel Patterns

### Pattern: The Proven Safe Loop

This pattern runs 87M+ iterations without issue:

```metal
kernel void persistent_runtime(
    device SystemState* system [[buffer(0)]],
    // ... other buffers ...
    uint tid [[thread_index_in_threadgroup]]
) {
    // Initial heartbeat (confirms kernel started)
    if (tid == 0) {
        atomic_fetch_add_explicit(&system->frame_counter, 1, memory_order_relaxed);
    }

    // MAIN LOOP - ALL threads participate
    while (true) {
        // PHASE 1: Check shutdown (ALL threads)
        if (atomic_load_explicit(&system->shutdown_flag, memory_order_relaxed)) {
            break;  // ALL threads exit together
        }

        // PHASE 2: Claim work (ALL threads attempt)
        bool have_work = false;
        if (tid < process_count) {
            have_work = try_claim_process(tid);
        }

        // PHASE 3: Execute work (conditional, but loop continues)
        for (uint i = 0; i < TIMESLICE && have_work; i++) {
            execute_instruction(tid);
        }

        // PHASE 4: Release work (conditional)
        if (have_work) {
            release_process(tid);
        }

        // PHASE 5: Sync ALL threads before next iteration
        threadgroup_barrier(mem_flags::mem_device);

        // PHASE 6: System tasks (bounded, parallelized)
        process_spawn_queue_parallel(tid);  // All threads, not just thread 0!

        // PHASE 7: Final sync
        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

### Pattern: Parallelized Queue Processing

**Never have thread 0 alone process a queue with a while loop:**

```metal
// SAFE: Parallel queue processing
void process_spawn_queue_parallel(uint tid) {
    uint head = atomic_load_explicit(&spawn_head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&spawn_tail, memory_order_relaxed);
    uint queue_len = tail - head;

    // Each thread handles at most ONE item
    if (tid < queue_len && tid < MAX_BATCH) {
        SpawnRequest req = spawn_queue[(head + tid) % QUEUE_SIZE];
        handle_spawn(req);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Only AFTER all threads done, update head
    if (tid == 0) {
        atomic_store_explicit(&spawn_head, head + min(queue_len, MAX_BATCH), memory_order_relaxed);
    }
}
```

### Pattern: Safe Shutdown

```rust
// Rust host side
pub fn stop(&mut self) {
    // Set shutdown flag with volatile + fence
    unsafe {
        let state = self.system_state.contents() as *mut SystemState;
        std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    // Wait for kernel to exit (frame counter stops advancing)
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(5);
    let mut last_frame = self.frame_count();
    let mut stable_count = 0;

    while start.elapsed() < timeout {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let current_frame = self.frame_count();

        if current_frame == last_frame {
            stable_count += 1;
            if stable_count >= 5 {
                break;  // Kernel exited
            }
        } else {
            stable_count = 0;
            last_frame = current_frame;
        }
    }
}
```

---

## 6. Testing Methodology

### Incremental Complexity Testing

**Never go straight to complex kernels. Build up incrementally:**

```
Level 1: Minimal kernel (proves dispatch works)
    - Single buffer, single write, immediate completion
    - Verify: Data written correctly

Level 2: Persistent loop (proves while(true) works)
    - Simple heartbeat counter
    - Verify: Counter advances for 15+ seconds

Level 3: Multi-buffer (proves buffer binding works)
    - Add buffers one at a time
    - Verify: Each buffer accessible

Level 4: Complex structs (proves alignment works)
    - Add your 432-byte Process struct
    - Verify: All fields read/write correctly

Level 5: Full kernel
    - Only after levels 1-4 pass
```

### The Proof Test Pattern

Always have a minimal proof test:

```rust
#[test]
fn test_persistent_kernel_proof() {
    // Minimal kernel: just heartbeat + shutdown
    let shader = r#"
        kernel void persistent_kernel(
            device atomic_uint* heartbeat [[buffer(0)]],
            device atomic_uint* shutdown [[buffer(1)]],
            uint tid [[thread_index_in_threadgroup]]
        ) {
            while (true) {
                if (atomic_load_explicit(shutdown, memory_order_relaxed)) break;
                if (tid == 0) {
                    atomic_fetch_add_explicit(heartbeat, 1, memory_order_relaxed);
                }
            }
        }
    "#;

    // Run for 15 seconds, verify heartbeat advances
    // This proves persistent loops work on this hardware
}
```

### Regression Tests for Known Crashes

```rust
#[test]
#[ignore = "CAUTION: This pattern caused system freeze"]
fn test_single_thread_loop_regression() {
    // Document the crashing pattern so no one reintroduces it
    // This test should NOT run by default

    // The pattern that crashed:
    // if (tid != 0) return;
    // while (true) { ... }

    // If this pattern appears in code review, reject immediately
}
```

### Buffer Verification Tests

```rust
#[test]
fn test_struct_sizes() {
    // Verify all GPU structs match expected sizes
    assert_eq!(std::mem::size_of::<Process>(), 432);
    assert_eq!(std::mem::size_of::<SpawnRequest>(), 16);
    assert_eq!(std::mem::size_of::<SystemState>(), 288);
    assert_eq!(std::mem::size_of::<RenderVertex>(), 48);

    // Verify 16-byte alignment
    assert_eq!(std::mem::size_of::<Process>() % 16, 0);
}
```

---

## 7. Recovery Strategies

### When System Freezes

**If GPU kernel causes system freeze:**

1. **Force restart** - Hold power button 10+ seconds
2. **Boot to recovery** - Hold power during boot if needed
3. **Check Console.app** - Look for GPU-related crashes in logs after reboot
4. **Reduce kernel complexity** - Go back to last working version

### Prevention: Watchdog Pattern

```rust
// Host-side watchdog
fn run_with_watchdog(&mut self, timeout: Duration) -> Result<(), &'static str> {
    let start = std::time::Instant::now();
    let initial_frame = self.frame_count();

    self.start()?;

    // Poll heartbeat
    while start.elapsed() < timeout {
        std::thread::sleep(Duration::from_millis(100));

        let current_frame = self.frame_count();
        if current_frame == initial_frame {
            // Heartbeat not advancing - kernel may be stuck
            // DON'T wait forever - stop and report
            self.stop();
            return Err("Kernel heartbeat stalled");
        }
    }

    self.stop();
    Ok(())
}
```

### Safe Development Workflow

1. **Use a test Mac** - Don't develop GPU kernels on your primary machine
2. **Save work frequently** - GPU crashes don't give you time to save
3. **Keep a known-good version** - `git stash` before each experiment
4. **Test with short timeouts first** - 5 seconds, not 15 minutes
5. **Add logging at each phase** - Frame counter, phase completion markers
6. **Review diffs carefully** - One wrong `if (tid != 0) return;` can freeze the system

### Emergency Kernel Kill

If you suspect a kernel is about to crash:

```rust
// Force shutdown with minimal delay
unsafe {
    std::ptr::write_volatile(&mut (*state).shutdown_flag, 1);
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}
// Sleep briefly to let GPU see the flag
std::thread::sleep(Duration::from_millis(10));
// If still running after this, you may need to terminate the process
```

---

## 8. Quick Reference Checklist

### Before Writing Any GPU Kernel

- [ ] All structs use `#[repr(C)]`
- [ ] All structs have size verified by `const_assert`
- [ ] All structs are 16-byte aligned (add padding if needed)
- [ ] Metal structs match Rust layout exactly
- [ ] Using `packed_float3` not `float3` for [f32; 3]

### Before Writing Persistent Loops

- [ ] NO `if (tid != 0) return;` before the while loop
- [ ] ALL threads enter the while loop
- [ ] Shutdown check is FIRST thing in loop body
- [ ] All threads check shutdown and break together
- [ ] Barriers only used where ALL threads will hit them
- [ ] Queue processing parallelized, not thread-0-only

### Before Dispatching

- [ ] All buffer bindings match kernel signature order
- [ ] All buffers are large enough for max data
- [ ] StorageModeShared used for CPU-GPU communication
- [ ] Initial values written with volatile + fence

### During Development

- [ ] Testing on non-primary machine if possible
- [ ] Work saved/committed before each test run
- [ ] Using short timeouts (5s) for initial tests
- [ ] Heartbeat/frame counter monitored
- [ ] Proof test passes before adding complexity

### Code Review Checklist

- [ ] No `if (tid != 0) return;` anywhere near while loops
- [ ] No unbounded while loops on single thread
- [ ] All barriers have all threads converging
- [ ] Struct alignment documented and verified
- [ ] Buffer sizes calculated and verified

---

## Appendix: Crash Signatures

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| System freeze, requires power cycle | Single-thread while loop | Search for `if (tid != 0) return` near `while` |
| Kernel returns immediately | Barrier deadlock | Check all threads reach each barrier |
| Garbage data in buffers | Struct misalignment | Verify sizes, check `packed_float3` usage |
| Vertices render as lines from origin | `float3` vs `packed_float3` | Use `packed_float3` for [f32; 3] |
| Heartbeat stops after ~5M iterations | SIMD divergence in loop | Ensure all threads participate |
| Frame counter never advances | Kernel never started | Check pipeline creation, buffer binding order |

---

*Document created: 2026-01-28*
*Based on: Persistent kernel development for GPU-native OS project*
*Hardware tested: Apple Silicon (M-series)*
