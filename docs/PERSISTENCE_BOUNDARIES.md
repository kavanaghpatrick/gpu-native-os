# GPU Persistence Boundaries Analysis

## The Question
Where exactly does "persistence" END in the current GPU software architecture?
If we understand the boundaries, we can simulate persistence by working around them.

---

## Layer-by-Layer Analysis

### Layer 1: Metal API (Rust side)

| Component | Persists? | Lifetime | Notes |
|-----------|-----------|----------|-------|
| `Device` | ✅ YES | App lifetime | Never recreated |
| `CommandQueue` | ✅ YES | App lifetime | Reusable |
| `Buffer` (device memory) | ✅ YES | Until deallocated | **KEY: State lives here** |
| `ComputePipelineState` | ✅ YES | Until deallocated | Shader stays compiled |
| `CommandBuffer` | ❌ NO | Single submit | **BOUNDARY: Must recreate each dispatch** |
| `ComputeCommandEncoder` | ❌ NO | Single encoding | Created per dispatch |
| `SharedEvent` | ✅ YES | Until deallocated | GPU↔CPU signaling |

**Boundary #1**: CommandBuffer must be recreated for each dispatch.
**Cost**: ~20-50µs to create and encode.

### Layer 2: Metal Runtime (Driver)

| Component | Persists? | Notes |
|-----------|-----------|-------|
| GPU command stream | ❌ NO | Flushed after each commit |
| Scheduled work queue | ❌ NO | Drained per command buffer |
| GPU page tables | ✅ YES | Buffers stay mapped |
| Shader binary | ✅ YES | Cached in GPU memory |

**Boundary #2**: Work queue drains after each command buffer completes.
**Implication**: No way to "queue more work" from GPU side in standard Metal.

### Layer 3: GPU Hardware

| Component | Persists? | Notes |
|-----------|-----------|-------|
| Device memory | ✅ YES | Survives kernel end |
| Registers | ❌ NO | Cleared when threadgroup ends |
| Threadgroup memory | ❌ NO | Cleared when threadgroup ends |
| L1/L2 cache | ⚠️ MAYBE | May survive if not evicted |
| Atomics in device mem | ✅ YES | **KEY: Coordination survives** |

**Boundary #3**: Registers and threadgroup memory cleared when kernel ends.
**Implication**: All state must be written to device memory before kernel exit.

### Layer 4: Kernel Execution

| Component | Persists? | Notes |
|-----------|-----------|-------|
| Thread ID | ❌ NO | Assigned per dispatch |
| Threadgroup ID | ❌ NO | Assigned per dispatch |
| Local variables | ❌ NO | In registers, lost |
| Device buffer contents | ✅ YES | **Survives everything** |

**Boundary #4**: Thread identity is lost between dispatches.
**Implication**: Threads must identify themselves from device memory state.

---

## What Actually Persists (Our Building Blocks)

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVICE MEMORY                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ State       │ │ Work Queue  │ │ Atomic Counters     │   │
│  │ Buffers     │ │ (tasks)     │ │ (coordination)      │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Checkpoint Buffer (for restart)                      │   │
│  │ - iteration_count                                    │   │
│  │ - phase (INPUT/LAYOUT/RENDER/etc)                   │   │
│  │ - work_queue_head/tail                              │   │
│  │ - any kernel-specific state                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Survives kernel termination
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 NEXT KERNEL DISPATCH                        │
│  - Reads state from device memory                          │
│  - Continues from checkpoint                               │
│  - No work lost                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Strategies to Simulate Persistence

### Strategy 1: Rapid Re-dispatch (Minimize Gap)

```
Kernel A runs → writes state → exits
                                   ↓ (~50µs gap)
                    CPU immediately dispatches Kernel B
                                   ↓
Kernel B runs → reads state → continues
```

**Implementation**:
- GPU signals completion via SharedEvent
- CPU has next CommandBuffer pre-encoded
- Submit immediately on signal
- Gap: 50-100µs (imperceptible at 60fps = 16,000µs)

### Strategy 2: Mega-Dispatch (Batch Work)

```
Instead of:  dispatch → wait → dispatch → wait → dispatch
Do:          dispatch(1000 iterations) → wait
```

**Implementation**:
- Kernel loops internally for N iterations
- Only exits when work exhausted OR approaching timeout
- Single dispatch covers entire frame

### Strategy 3: Checkpoint Before Timeout

```
Kernel tracks iteration count:
  if (iterations > SAFE_LIMIT) {
      write_checkpoint();
      return;  // Exit gracefully before watchdog
  }
```

**Implementation**:
- Know the watchdog timeout (~2 seconds on macOS)
- Checkpoint at ~1.5 seconds
- CPU sees completion, re-dispatches
- New kernel resumes from checkpoint

### Strategy 4: Supervisor Warp

```
ThreadGroup 0, Warp 0 = Supervisor
  - Monitors iteration count
  - Writes heartbeat to device memory
  - Signals CPU if restart needed

Other warps = Workers
  - Do actual computation
  - Check supervisor signal periodically
```

**Implementation**:
```metal
if (is_supervisor_warp()) {
    // Track time/iterations
    if (should_checkpoint()) {
        atomic_store(&checkpoint_signal, 1);
        write_checkpoint_data();
    }
} else {
    // Do work, check signal occasionally
    if (atomic_load(&checkpoint_signal)) {
        write_my_state();
        return;
    }
}
```

### Strategy 5: Double-Buffered Continuous

```
GPU Dispatch A ──────────────────→ completes
           └── CPU encodes B ──→ GPU Dispatch B ─────→ completes
                           └── CPU encodes C ──→ GPU Dispatch C
```

**Implementation**:
- Overlap CPU encoding with GPU execution
- Triple buffer for safety
- GPU never idle, persistence simulated

---

## The Minimum CPU Involvement

What's the ABSOLUTE MINIMUM CPU work to keep GPU running?

```rust
// Tight loop on CPU - minimal work
loop {
    // Pre-encoded command buffer (stored, not recreated)
    // Just need to update one atomic for "run next iteration"
    atomic_store(&run_signal, frame_number);

    command_buffer.commit();  // ~5µs

    // Don't wait! Check signal instead
    while shared_event.signaled_value() < frame_number {
        // Spin or do other work
    }
}
```

**Cost breakdown**:
- atomic_store: ~100ns
- commit(): ~5µs
- Signal check: ~100ns

**Total CPU overhead per "dispatch": ~5-10µs**

At 120fps (8.3ms frames), this is **0.06%** CPU overhead.
The GPU is effectively running continuously.

---

## Key Insight

**Persistence is about STATE, not EXECUTION.**

The kernel code restarts, but if STATE persists in device memory,
the effect is identical to continuous execution.

```
Traditional view:  Kernel runs → Kernel dies → New kernel starts
Our view:          STATE persists → Kernel is just a "viewer" of state
```

The kernel becomes a pure function: `new_state = f(old_state)`
State lives forever in device memory.
Kernel is ephemeral but that doesn't matter.

---

## Next Steps

1. **Measure actual dispatch overhead** - Is it really 50µs? Could be less.
2. **Implement checkpoint buffer** - Standard format for all kernels
3. **Test supervisor warp pattern** - Does Metal schedule it fairly?
4. **Build rapid re-dispatch loop** - Minimize CPU gap
5. **Profile cache survival** - Does L2 cache survive dispatch gap?
