# GPU as COMPUTING UNIT

## The Fundamental Question

**Why does our computing unit need to ask the host processor for permission to continue working?**

A CPU doesn't:
- Stop after every function and ask "should I continue?"
- Need external "dispatch" to run the next instruction
- Return control to some other processor between operations

The GPU shouldn't either.

---

## CPU Model (What We Want)

```
┌────────────────────────────────────────┐
│            CPU EXECUTION               │
├────────────────────────────────────────┤
│  1. Fetch instruction from memory      │
│  2. Decode instruction                 │
│  3. Execute instruction                │
│  4. Write results to memory            │
│  5. Increment program counter          │
│  6. GOTO 1 (forever)                   │
└────────────────────────────────────────┘

Only stops for:
  - Interrupts (I/O, timers)
  - Explicit halt instruction
  - Power off
```

## Current GPU Model (What We Have)

```
┌────────────────────────────────────────┐
│         "GRAPHICS CARD" MODEL          │
├────────────────────────────────────────┤
│  HOST:  Create command buffer          │
│  HOST:  Encode work                    │
│  HOST:  Submit to GPU                  │
│  GPU:   Execute commands               │
│  GPU:   Signal completion              │
│  HOST:  Read results                   │
│  HOST:  GOTO 1                         │
└────────────────────────────────────────┘

GPU is PASSIVE. Host drives everything.
GPU cannot decide to do more work.
```

## What We Want

```
┌────────────────────────────────────────┐
│         GPU AS COMPUTER MODEL          │
├────────────────────────────────────────┤
│  STARTUP:                              │
│    Host: Load program into GPU memory  │
│    Host: Set up I/O buffers            │
│    Host: Start GPU execution           │
│                                        │
│  STEADY STATE (GPU runs alone):        │
│    GPU: Read work queue from memory    │
│    GPU: Execute work                   │
│    GPU: Write results to memory        │
│    GPU: Check for new work             │
│    GPU: GOTO "Read work queue"         │
│                                        │
│  I/O (minimal host involvement):       │
│    Host: Push input events to buffer   │
│    Host: Read display buffer           │
│    (GPU never stops for this)          │
└────────────────────────────────────────┘

GPU is ACTIVE. GPU drives itself.
Host is just I/O handler.
```

---

## The API Barrier (Not Hardware)

Metal/CUDA/Vulkan APIs assume the "graphics card" model because:

1. **Historical**: GPUs started as fixed-function graphics hardware
2. **Safety**: Runaway GPU code could hang the display
3. **Scheduling**: OS needs to share GPU between apps
4. **Power**: Letting GPU idle saves battery

But the HARDWARE can do more:
- GPU has its own instruction memory
- GPU has program counters
- GPU can loop indefinitely
- GPU can read/write all of device memory
- Apple Silicon: GPU shares memory with CPU (no copy needed)

---

## What Actually Limits Persistence?

### Hardware Level
- **Watchdog timer**: OS kills kernels running >2-5 seconds
- **Scheduler preemption**: OS may swap out our kernel for another app
- **Power management**: GPU may clock down if "idle" (in a spin loop)

### API Level
- **No instruction stream continuation**: Can't say "keep running"
- **Command buffer is one-shot**: Must create new one each time
- **No GPU-to-host interrupts**: GPU can't say "give me more work"

### What's NOT a Limit
- **Memory persistence**: ✅ Device memory survives between executions
- **State preservation**: ✅ Atomics, buffers all persist
- **Continuous loops**: ✅ Kernel can loop internally (within timeout)

---

## Strategies for GPU-as-Computer

### Strategy 1: Minimize Host Interruption

Host does the absolute minimum:
```
loop {
    // Pre-encoded, just poke one atomic
    atomic_store(&run_signal, 1);
    command_buffer.commit();  // 5µs
    // Don't wait - poll signal instead
}
```

Measured overhead: **5-10µs per "cycle"**
At 100,000 cycles/second, that's only 0.5-1% overhead.

### Strategy 2: GPU Work Queue

GPU reads work from a queue in device memory:
```metal
kernel void gpu_computer() {
    while (should_run) {
        Task task = dequeue(&work_queue);
        execute(task);
        // GPU decides what's next
    }
}
```

Host just pushes to the queue. GPU pulls and executes.

### Strategy 3: Indirect Command Buffers (ICB)

GPU generates its own commands:
```
GPU: "I need to run kernel X with these params"
GPU: Write to ICB
GPU: Execute ICB
```

No host involvement in deciding what to run.

### Strategy 4: Checkpoint and Continue

Before hitting timeout:
```metal
if (iteration > SAFE_LIMIT) {
    checkpoint_state();
    return;  // Exit gracefully
}
// Host re-launches, state preserved
```

Host just keeps the program alive. GPU does all real work.

---

## Measurement: Current Overhead

From profiler (actual data):
```
Execution cycle:    133µs total
  - Host encoding:    8µs
  - GPU execution:   ~50µs (minimal work)
  - Synchronization: 75µs

State persistence:  ✅ 100% (confirmed)
Safe iterations:    1,000,000 per cycle
```

**At 133µs per cycle = 7,500 cycles/second**
**At 1M iterations per cycle = 7.5 BILLION iterations/second**

The GPU can do real work. The overhead is noise.

---

## The Path Forward

1. **Stop thinking in frames** - Think in continuous execution
2. **Minimize host code path** - Pre-encode everything possible
3. **GPU decides what's next** - Work queue, not host dispatch
4. **State lives on GPU** - Host is just I/O
5. **Measure everything** - No guessing

The goal: **Host is the I/O co-processor. GPU is the computer.**
