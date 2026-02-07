# Persistent Kernel Migration Plan

**Date**: January 2026
**Status**: Action Plan Based on 10-Agent Audit
**Goal**: Migrate from batch processing to true persistent kernels

---

## Executive Summary

The 10-agent audit revealed that while the infrastructure for persistent kernels exists (atomics work, Metal supports indefinite loops), the codebase has **systemic single-thread patterns** that waste 96.9% of GPU resources and cause SIMD divergence stalls.

### Critical Discovery
```
Persistent kernels WORK if: All 32 SIMD threads participate in while(true)
Persistent kernels STALL if: Only thread 0 runs the loop (if tid != 0 return)
```

### Audit Statistics
| Metric | Count | Impact |
|--------|-------|--------|
| Single-thread bottlenecks | 5+ in event_loop.rs | SIMD divergence |
| `wait_until_completed()` calls | 173 | CPU blocking |
| WASM interpreter utilization | 3.1% (1/32 threads) | 96.9% waste |
| Tests enforcing all-thread patterns | 0 | No guardrails |

---

## Phase 1: Critical Single-Thread Fixes (Week 1)

### Priority 1: WASM Interpreter (96.9% GPU Waste)

**File**: `src/gpu_os/gpu_app_system.rs` (line ~4231)

**Current Pattern (WRONG)**:
```metal
kernel void wasm_interpreter(...) {
    if (tid != 0) return;  // <-- 31/32 threads do nothing!

    while (pc < program_length) {
        execute_instruction(pc);
        pc++;
    }
}
```

**Target Pattern (CORRECT)**:
```metal
kernel void wasm_interpreter(...) {
    // ALL threads participate in the loop
    while (true) {
        // Thread 0 fetches next instruction
        threadgroup uint shared_opcode;
        if (tid == 0) {
            shared_opcode = fetch_instruction(pc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint opcode = shared_opcode;

        // ALL threads execute in parallel based on opcode type
        switch (opcode) {
            case VECTOR_ADD:
                // All 32 threads process 32 elements in parallel
                result[tid] = a[tid] + b[tid];
                break;
            case SCALAR_OP:
                // Thread 0 does scalar, others assist or idle efficiently
                if (tid == 0) { scalar_result = compute(); }
                break;
            case MEMORY_COPY:
                // All threads copy in parallel
                dest[base + tid] = src[base + tid];
                break;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 advances PC
        if (tid == 0) {
            pc = next_pc(opcode);
            if (pc >= program_length) break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
```

**Key Changes**:
1. Remove `if (tid != 0) return;`
2. Use threadgroup shared memory for instruction broadcast
3. Parallelize memory operations across all threads
4. Keep barriers tight to minimize sync overhead

### Priority 2: Event Loop Single-Thread Patterns

**File**: `src/gpu_os/event_loop.rs`

**Lines to fix**: 397, 486, 562, 652, 675

**Pattern to eliminate**:
```metal
if (tid != 0) return;  // WRONG
```

**Replace with**:
```metal
// ALL threads check for events
uint my_event_idx = atomic_fetch_add(&event_tail, 1);
if (my_event_idx < event_head) {
    InputEvent event = events[my_event_idx % QUEUE_SIZE];
    process_event(event, tid);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

### Priority 3: gpu_app_loader.metal While Loop

**File**: `src/gpu_os/shaders/gpu_app_loader.metal`

**Current**: `allocate_slot()` has unbounded while loop that only thread 0 runs

**Fix**: Convert to parallel slot search:
```metal
// All threads search different slots in parallel
uint my_slot = tid;
bool found = false;
while (!found && my_slot < MAX_SLOTS) {
    if (atomic_compare_exchange(&slots[my_slot].state, EMPTY, ALLOCATING)) {
        found = true;
        result_slot = my_slot;
    }
    my_slot += 32;  // Stride by SIMD width
}
// Reduce across threads to find winner
uint winner_slot = simd_min(found ? result_slot : UINT_MAX);
```

---

## Phase 2: Eliminate CPU Blocking (Week 2)

### The 173 `wait_until_completed()` Problem

**Root Cause**: Current architecture dispatches kernel, waits for completion, reads results.

**Target**: Dispatch once at boot, poll shared memory for results.

### Classification of Sync Points

| Category | Count | Action |
|----------|-------|--------|
| Render frame sync | ~50 | Eliminate with triple buffering |
| Test assertions | ~80 | Keep (tests need determinism) |
| One-time init | ~25 | Keep (boot only) |
| Event loop dispatch | ~18 | **Eliminate** (persistent kernel) |

### Priority Eliminations

**1. Event Loop Dispatch** (`event_loop.rs`)
```rust
// CURRENT (WRONG)
loop {
    let cmd = queue.new_command_buffer();
    encode_event_loop_dispatch(&cmd);
    cmd.commit();
    cmd.wait_until_completed();  // <-- CPU blocks every frame!
}

// TARGET (CORRECT)
// Boot: dispatch persistent kernel once
let cmd = queue.new_command_buffer();
encode_persistent_kernel(&cmd);
cmd.commit();
// Never wait - kernel runs forever

// Event delivery: write to shared buffer
unsafe {
    let input = input_buffer.contents() as *mut InputQueue;
    (*input).events[head % SIZE] = event;
    atomic_store(&(*input).head, head + 1);
}
// GPU polls this buffer in its while(true) loop
```

**2. Render Frame Sync**
```rust
// CURRENT: CPU waits for each frame
cmd.wait_until_completed();
present_drawable();

// TARGET: Triple buffering, no CPU wait
// Frame N renders while Frame N-1 presents while Frame N-2 completes
let frame_idx = frame_count % 3;
encode_render(&cmd, &buffers[frame_idx]);
cmd.commit();
// Add completion handler, don't wait
cmd.add_completed_handler(|_| {
    // Signal frame ready for present
});
```

---

## Phase 3: Unified Mega-Kernel Architecture (Week 3-4)

### Current: Multiple Kernels, Multiple Dispatches
```
CPU dispatches: event_kernel → layout_kernel → render_kernel → present
Each dispatch: ~50μs overhead
Total: ~150μs/frame just in dispatch overhead
```

### Target: Single Persistent Mega-Kernel
```metal
kernel void gpu_os_main(
    device InputQueue* input,
    device LayoutState* layout,
    device RenderState* render,
    device atomic_uint* frame_ready,
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    while (true) {
        // PHASE 1: INPUT (all threads poll their slice of input queue)
        process_input_slice(input, tid);
        threadgroup_barrier(mem_flags::mem_device);

        // PHASE 2: LAYOUT (all threads compute their elements)
        compute_layout_slice(layout, tid);
        threadgroup_barrier(mem_flags::mem_device);

        // PHASE 3: RENDER (all threads rasterize their primitives)
        render_slice(render, tid);
        threadgroup_barrier(mem_flags::mem_device);

        // PHASE 4: FRAME SYNC (thread 0 signals, all wait)
        if (tid == 0) {
            atomic_fetch_add_explicit(frame_ready, 1, memory_order_release);
        }

        // Self-throttle to ~60fps
        // (GPU-native vsync via barrier spinning)
        for (int i = 0; i < VSYNC_CYCLES; i++) {
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
}
```

### Benefits
- Zero dispatch overhead after boot
- GPU owns all state
- No CPU in steady-state loop
- Natural vsync via barrier spinning

---

## Phase 4: Testing & Validation (Ongoing)

### New Test Categories Needed

**1. All-Threads-Participate Enforcement**
```rust
#[test]
fn test_no_single_thread_patterns() {
    let shader_source = include_str!("../src/gpu_os/shaders/kernel.metal");

    // Detect anti-patterns
    let violations: Vec<_> = shader_source
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            line.contains("if (tid != 0) return") ||
            line.contains("if (tid == 0)") && !line.contains("threadgroup_barrier")
        })
        .collect();

    assert!(violations.is_empty(),
        "Single-thread patterns found: {:?}", violations);
}
```

**2. Persistent Kernel Stress Test**
```rust
#[test]
fn test_persistent_kernel_60_seconds() {
    // Run kernel for 60 seconds
    // Verify: no stalls, consistent heartbeats, clean shutdown
    // This catches SIMD divergence issues that only appear over time
}
```

**3. CPU Utilization Monitoring**
```rust
#[test]
fn test_cpu_under_5_percent() {
    // Run full system for 10 seconds
    // Measure CPU utilization
    // Assert < 5% (per CLAUDE.md goals)
}
```

---

## Migration Checklist

### Week 1: Single-Thread Fixes
- [ ] Fix WASM interpreter (gpu_app_system.rs:4231)
- [ ] Fix event_loop.rs:397 single-thread pattern
- [ ] Fix event_loop.rs:486 single-thread pattern
- [ ] Fix event_loop.rs:562 single-thread pattern
- [ ] Fix event_loop.rs:652 single-thread pattern
- [ ] Fix event_loop.rs:675 single-thread pattern
- [ ] Fix gpu_app_loader.metal allocate_slot()
- [ ] Add test: no_single_thread_patterns

### Week 2: CPU Blocking Elimination
- [ ] Convert event loop to persistent dispatch
- [ ] Implement triple buffering for render
- [ ] Remove 18 event-loop wait_until_completed() calls
- [ ] Add test: cpu_under_5_percent

### Week 3-4: Mega-Kernel Unification
- [ ] Design unified state buffer layout
- [ ] Implement phased mega-kernel
- [ ] Migrate event processing to mega-kernel
- [ ] Migrate layout to mega-kernel
- [ ] Migrate render to mega-kernel
- [ ] Add test: persistent_kernel_60_seconds

### Documentation Updates
- [ ] Update CLAUDE.md: Remove "batch processing" references
- [ ] Update CLAUDE.md: Add "all threads participate" rule
- [ ] Archive HONEST_STATUS_ASSESSMENT.md (no longer accurate)
- [ ] Update PRD docs with new architecture

---

## Risk Mitigation

### Risk: Thermal Throttling
**Mitigation**: GPU-native self-throttling via barrier spinning
```metal
if (should_throttle()) {
    for (int i = 0; i < COOLDOWN_CYCLES; i++) {
        threadgroup_barrier(mem_flags::mem_none);
    }
}
```

### Risk: System Hang During Development
**Mitigation**: Always include shutdown signal check
```metal
while (true) {
    if (atomic_load(&shutdown_flag)) break;  // <-- Always check this first
    // ... rest of kernel
}
```

### Risk: Breaking Existing Tests
**Mitigation**: Parallel branch development
```bash
git worktree add ../persistent-kernel-migration feature/persistent-kernels
# Develop in isolation, merge when stable
```

---

## Success Criteria

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| CPU utilization | ~50% | <5% | Activity Monitor during demo |
| GPU thread utilization | 3.1% | >80% | Metal System Trace |
| Dispatch calls/second | 60+ | 1 (at boot) | Instrument dispatch count |
| `wait_until_completed()` in hot path | 173 | 0 | grep count |
| Single-thread patterns | 5+ | 0 | Static analysis |

---

## Conclusion

The path to true persistent kernels is clear:

1. **Fix single-thread patterns** - This is the root cause of stalls
2. **Eliminate CPU blocking** - Remove wait_until_completed() from hot path
3. **Unify into mega-kernel** - One dispatch at boot, runs forever
4. **Add enforcement tests** - Prevent regression

The infrastructure is proven (test_persistent_kernel_proof.rs shows 15+ seconds, 87M iterations). We just need to apply the "all threads participate" pattern systematically.

**THE GPU IS THE COMPUTER** - This plan makes it reality.
