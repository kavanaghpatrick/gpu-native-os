# Goal vs Debate Results: Critical Comparison

**Date**: January 2026
**Purpose**: Honest assessment of WasmBurst architecture against project vision

---

## Executive Summary

The debate produced a **pragmatic but compromised** architecture that contradicts several core project principles. This document identifies the gaps and proposes paths forward.

| Aspect | CLAUDE.md Goal | WasmBurst Result | Alignment |
|--------|----------------|------------------|-----------|
| CPU Role | Boot + Network only | **Orchestrates every burst** | ❌ CONFLICT |
| Persistent Kernels | Run indefinitely | **500ms max, then cooldown** | ❌ CONFLICT |
| GPU Never Waits | Poll and continue | **Host manages timing** | ❌ CONFLICT |
| CPU Utilization | <5% steady state | **~30-50% (orchestration)** | ❌ CONFLICT |
| GPU Occupancy | >80% | **~50%+ targeted** | ⚠️ PARTIAL |
| Zero CPU Round-trips | Per frame | **Every 1.5s minimum** | ❌ CONFLICT |
| Production Code | No bandaids | **Thermal workarounds** | ⚠️ PARTIAL |

**Verdict**: WasmBurst is a valid **stepping stone** but not the **final architecture**.

---

## Detailed Comparison

### 1. CPU Role

#### CLAUDE.md Goal
> "**Zero CPU involvement** in steady-state operation. CPU has exactly TWO jobs:
> 1. Boot - Initialize hardware, load GPU kernel, then exit the loop
> 2. Network I/O - Receive packets from NIC (hardware limitation)"

#### WasmBurst Result
```
Host (Rust) orchestrates:
  - Prepares buffers before each burst
  - Encodes Metal command buffer
  - Commits and waits for completion
  - Manages 500ms/1s timing
  - Falls back to CPU for sequential code
```

#### Gap Analysis
| Goal | Reality | Gap |
|------|---------|-----|
| CPU exits after boot | CPU runs orchestration loop | **FUNDAMENTAL** |
| GPU self-schedules | Host schedules bursts | **FUNDAMENTAL** |
| CPU only for network | CPU for thermal, timing, fallback | **SIGNIFICANT** |

#### Path to Alignment
The project's existing work on **persistent kernels** (Issue #133, #149) already proves GPU can run indefinitely. WasmBurst's thermal concern is valid but solvable:

```metal
// GPU-NATIVE thermal management (no CPU)
kernel void persistent_app_kernel(...) {
    while (true) {
        // Self-throttle based on work done
        uint work_this_frame = atomic_load(&work_counter);
        if (work_this_frame > THERMAL_THRESHOLD) {
            // GPU-side yield: skip N iterations
            for (int i = 0; i < COOLDOWN_CYCLES; i++) {
                threadgroup_barrier(mem_flags::mem_none);
            }
            atomic_store(&work_counter, 0);
        }

        // Normal execution
        execute_wasm_instruction();
    }
}
```

---

### 2. Persistent Kernels

#### CLAUDE.md Goal
> "**Persistent Kernels**
> - Run indefinitely with while(true)
> - Poll memory atomics for events
> - Proven in Issue #133, #149"

#### WasmBurst Result
> "Dispatch duration: Limit GPU kernels to **500ms max**"
> "Cooldown: After each burst, insert a **1s host-side sleep**"

#### Gap Analysis
| Goal | Reality | Gap |
|------|---------|-----|
| while(true) forever | 500ms then abort | **DIRECT CONTRADICTION** |
| GPU polls for events | Host manages events | **ARCHITECTURAL** |
| Proven working | Abandoned for thermal | **REGRESSION** |

#### Why This Happened
The debate accepted the "thermal throttling" argument without exploring GPU-native solutions:

```
Debate assumption: "M1 throttles to 1.5 TFLOPS after 2 minutes"
Reality check: This is for SUSTAINED PEAK load, not typical workloads

Debate assumption: "Persistent = always at 100% utilization"
Reality: Persistent kernels can self-throttle via barriers
```

#### Path to Alignment
Persistent kernels WITH self-regulation:

```metal
// Self-regulating persistent kernel
kernel void gpu_os_main(...) {
    uint consecutive_idle = 0;

    while (true) {
        bool has_work = check_event_queue();

        if (has_work) {
            process_events();
            consecutive_idle = 0;
        } else {
            consecutive_idle++;
            // Exponential backoff - GPU-native power management
            if (consecutive_idle > 1000) {
                // ~1ms sleep via barrier spinning
                for (int i = 0; i < 10000; i++) {
                    threadgroup_barrier(mem_flags::mem_none);
                }
            }
        }
    }
}
```

---

### 3. GPU Never Waits

#### CLAUDE.md Goal
> "**Key rule: GPU never waits. GPU never blocks. GPU polls and continues.**"
>
> "GPU: writes request to queue (atomic)
> GPU: continues working (never waits!)
> GPU: polls status buffer periodically"

#### WasmBurst Result
```
Host orchestration loop:
  1. Prepare buffers
  2. Encode dispatch
  3. Commit command buffer
  4. WAIT for completion      <-- GPU IS IDLE
  5. 1s cooldown              <-- GPU IS IDLE
  6. Repeat
```

#### Gap Analysis
The WasmBurst model has GPU **idle 67% of the time** (1s cooldown out of 1.5s cycle).

| Metric | Goal | WasmBurst |
|--------|------|-----------|
| GPU duty cycle | 100% | ~33% |
| Idle time per cycle | 0 | 1000ms |
| Who decides when to work | GPU | CPU |

#### Path to Alignment
The existing codebase already has the right pattern in `kernel.metal`:

```metal
// EXISTING - Phase-based execution, GPU never idles
kernel void gpu_os_kernel(...) {
    // PHASE 1: Input collection (all threads)
    // PHASE 2: Hit testing (all threads)
    // PHASE 3: Visibility counting (all threads)
    // PHASE 4: Bitonic sort (all threads)
    // PHASE 5: State update (all threads)
    // No idle phases, no waiting
}
```

---

### 4. Success Metrics

#### CLAUDE.md Targets
> 1. **CPU utilization during steady state** - Target: <5%
> 2. **GPU thread utilization** - Target: >80% occupancy
> 3. **Data round-trips to CPU** - Target: Zero per frame
> 4. **Lines of CPU code vs GPU code** - Track ratio over time

#### WasmBurst Projected Metrics

| Metric | Target | WasmBurst | Gap |
|--------|--------|-----------|-----|
| CPU utilization | <5% | **30-50%** | 6-10x worse |
| GPU occupancy | >80% | ~50% | 40% worse |
| CPU round-trips | 0/frame | **1 per 1.5s** | Infinite worse |
| CPU:GPU code ratio | Minimize | ~40% CPU | Significant |

---

### 5. No Bandaids Policy

#### CLAUDE.md Goal
> "**This project writes PRODUCTION-READY code. No workarounds. No hacks. No "temporary" fixes.**"
>
> "When code doesn't work on our platform, we have exactly TWO options:
> 1. FIX THE PLATFORM
> 2. CREATE A GITHUB ISSUE"

#### WasmBurst Violations

| Issue | WasmBurst Response | Should Be |
|-------|-------------------|-----------|
| Thermal throttling | Host-managed cooldowns | GPU-native throttling |
| Sequential code | Fallback to Wasmtime CPU | GPU sequential execution |
| Register pressure | "Let runtime handle it" | Fix translator to emit optimal code |
| Divergent code | Abort and retry on CPU | GPU divergence handling |

---

## Root Cause Analysis

### Why Did the Debate Produce This Result?

1. **Accepted CPU-centric framing**: The debate optimized for "how to run WASM on GPU" rather than "how to make GPU the computer"

2. **Thermal fear**: Accepted "2 minute throttling" without exploring GPU-native power management

3. **Pragmatism over vision**: Optimized for "1-week MVP" rather than "correct architecture"

4. **wasm-gpu influence**: Adopted wasm-gpu's host-orchestrated model without questioning if it fits our goals

5. **Missing existing work**: Didn't fully consider the persistent kernel work already done (Issue #133, #149)

---

## Reconciliation: Two Paths Forward

### Path A: Accept WasmBurst as Stepping Stone

Use WasmBurst for:
- Rapid prototyping and benchmarking
- Proving WASM-on-GPU is viable
- Learning what patterns work

Then evolve to GPU-native architecture.

**Timeline**: Ship WasmBurst in 2 weeks, then 4-week sprint to GPU-native.

### Path B: Build GPU-Native from Start

Reject WasmBurst's host-orchestration model entirely. Build on existing persistent kernel infrastructure.

**Key changes from WasmBurst**:

| WasmBurst | GPU-Native |
|-----------|------------|
| Host orchestrates bursts | GPU runs persistent kernel |
| 500ms limit | No limit, self-throttling |
| CPU fallback for sequential | GPU executes all code |
| 64KB buffers managed by host | GPU manages own memory |
| Thermal cooldowns | GPU-native power management |

**Timeline**: 4-6 weeks, but correct architecture.

---

## Proposed GPU-Native Architecture

Based on project goals, here's what the architecture SHOULD look like:

### 1. Compilation: Same as WasmBurst
```
WASM → Naga IR → Metal (AOT, <100ms)
```
This part is correct.

### 2. Memory: GPU-Managed Pools
```metal
// GPU allocates from pre-mapped pools
struct GpuHeap {
    device uint8_t* memory;
    device atomic_uint next_free;
    uint pool_size;
};

uint allocate(device GpuHeap* heap, uint size) {
    return atomic_fetch_add(&heap->next_free, size);
}
```
No host involvement in allocation.

### 3. Execution: Persistent Kernel with Self-Throttling
```metal
kernel void wasm_runtime(...) {
    // GPU-native process table
    device WasmProcess* processes;
    device atomic_uint active_count;

    while (true) {
        uint pid = tid % atomic_load(&active_count);
        WasmProcess* proc = &processes[pid];

        // Execute N instructions
        for (int i = 0; i < INSTRUCTIONS_PER_YIELD; i++) {
            execute_instruction(proc);
        }

        // Self-throttle if needed (GPU-native)
        if (should_throttle()) {
            barrier_yield(THROTTLE_CYCLES);
        }

        // Check for new processes (GPU-native spawn)
        check_spawn_queue();
    }
}
```

### 4. Thermal: GPU Self-Regulation
```metal
bool should_throttle() {
    // Track work done via atomic counter
    uint work = atomic_load(&global_work_counter);
    uint target = WORK_PER_THERMAL_WINDOW;
    return work > target;
}

void barrier_yield(uint cycles) {
    // Spin on barriers - reduces power without CPU
    for (uint i = 0; i < cycles; i++) {
        threadgroup_barrier(mem_flags::mem_none);
    }
}
```

### 5. Sequential Code: Single-Lane Execution
```metal
// Sequential code runs on lane 0, others assist or idle efficiently
if (simd_is_first()) {
    // Sequential execution
    execute_sequential_block(proc);
} else {
    // Other lanes do useful work or yield
    simd_ballot(true); // Sync point
}
```

---

## Metrics Comparison

| Metric | CLAUDE.md | WasmBurst | GPU-Native |
|--------|-----------|-----------|------------|
| CPU steady-state | <5% | 30-50% | **<5%** ✓ |
| GPU occupancy | >80% | ~50% | **>80%** ✓ |
| CPU round-trips | 0/frame | 1/1.5s | **0/frame** ✓ |
| Thermal handling | GPU | CPU | **GPU** ✓ |
| Sequential code | GPU | CPU fallback | **GPU** ✓ |

---

## Conclusion

The debate produced a **valid engineering solution** that **violates project principles**.

WasmBurst is useful as:
- A proof-of-concept
- A benchmark baseline
- A learning exercise

But the final architecture must return to the core thesis: **THE GPU IS THE COMPUTER**.

The existing codebase (persistent kernels, GPU event loop, atomic coordination) already demonstrates this is possible. The debate's thermal concerns are solvable with GPU-native self-throttling.

**Recommendation**: Use WasmBurst insights for compilation pipeline, but reject host-orchestration model for execution. Build GPU-native runtime on existing persistent kernel infrastructure.

---

## Action Items

1. **Create Issue**: "GPU-Native WASM Runtime" - document correct architecture
2. **Preserve**: WasmBurst compilation pipeline (WASM → Naga → Metal)
3. **Reject**: Host-orchestrated burst model
4. **Build**: Persistent kernel WASM interpreter with self-throttling
5. **Benchmark**: Compare WasmBurst vs GPU-Native approaches
