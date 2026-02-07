# Architecture: THE GPU IS THE COMPUTER

## Core Thesis

The GPU is not an accelerator. The GPU is the computer. The CPU is an I/O peripheral.

This document defines the architectural principles that govern all design decisions in this project.

---

## The Fundamental Shift

### Traditional Model (WRONG)

```
CPU (brain) ──commands──> GPU (worker)
     │                         │
     │<──────results───────────│
     │
     └──> Storage, Network, Input, Display
```

The CPU orchestrates. The GPU waits for work. The CPU handles all I/O.

### Our Model (CORRECT)

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU (THE COMPUTER)                            │
│                                                                  │
│  - Runs persistent megakernel                                    │
│  - Owns all application state                                    │
│  - Makes all decisions                                           │
│  - Never waits, never blocks, never asks permission              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              GPU-VISIBLE MEMORY (Unified on Apple Silicon)  ││
│  │                                                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        ││
│  │  │ I/O Command  │ │ Completion   │ │ Event        │        ││
│  │  │ Queues       │ │ Status Flags │ │ Buffers      │        ││
│  │  │ (GPU writes) │ │ (GPU polls)  │ │ (GPU reads)  │        ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
              │                              ▲
              │ Commands flow OUT            │ Data/Events flow IN
              ▼                              │
┌─────────────────────────────────────────────────────────────────┐
│                    HARDWARE LAYER                                │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ NVMe/SSD    │  │ USB/HID     │  │ Network     │              │
│  │ Controller  │  │ Controller  │  │ NIC         │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  DMA transfers data directly to/from GPU-visible memory          │
│  No CPU involvement in data path                                 │
└─────────────────────────────────────────────────────────────────┘
              │
              │ (Setup only, not per-operation)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CPU (I/O PERIPHERAL)                          │
│                                                                  │
│  Responsibilities:                                               │
│  - Boot: Initialize hardware, set up DMA mappings                │
│  - Runtime: Handle hardware interrupts                           │
│  - Runtime: Write events to GPU-visible buffers                  │
│                                                                  │
│  NOT responsible for:                                            │
│  - Making decisions                                              │
│  - Coordinating GPU work                                         │
│  - Being in any request/response loop                            │
│  - Blocking GPU execution                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Principle: GPU Never Waits

The GPU must **never** block waiting for CPU. This means:

### NO Syscalls

Traditional syscalls are request/response:
```
GPU: "I need X"  →  CPU  →  "Here's X"  →  GPU continues
         └─── GPU BLOCKED ───┘
```

This is **wrong**. It makes CPU the coordinator.

### YES Command Queues + Polling

```
GPU: Writes command to queue → Continues working → Later polls status → Uses result
                │                     │                    │
                │                     │                    └─ Non-blocking check
                │                     └─ GPU does useful work
                └─ Fire and forget, no blocking
```

The GPU is always in control. The GPU decides when to check. The GPU never stops.

---

## I/O Model

### Storage (Files)

**Mechanism**: `MTLIOCommandQueue` (Metal 3) or memory-mapped files

```
GPU wants file data:
1. GPU writes load descriptor to I/O command buffer
2. GPU continues executing (no wait)
3. Hardware/DMA loads data to GPU-visible buffer
4. GPU polls completion flag when convenient
5. GPU uses data
```

**Key**: Data moves via DMA. CPU may handle interrupts but is not in data path.

### Input (Keyboard, Mouse, Touch)

**Mechanism**: Event buffers written by CPU interrupt handlers

```
User presses key:
1. Hardware interrupt fires
2. CPU ISR writes event to GPU-visible ring buffer (fast, no allocation)
3. GPU reads event buffer each frame (or more often)
```

**Key**: CPU is an event *producer*, not a request *handler*. GPU pulls, never asks.

### Display (Framebuffer)

**Mechanism**: GPU writes directly to display buffer

```
GPU renders frame:
1. GPU writes pixels to framebuffer (or render target)
2. Display controller reads framebuffer
3. Pixels appear on screen
```

**Key**: This already works correctly. No CPU involvement.

### Network (CPU Required - Hardware Limitation)

**Mechanism**: Ring buffer pattern - CPU receives, GPU processes

```
Packet arrives:
1. NIC delivers packet to CPU (hardware limitation)
2. CPU copies to GPU-visible ring buffer
3. GPU reads packet buffer when convenient

GPU sends packet:
1. GPU writes packet to transmit buffer
2. GPU writes doorbell to NIC
3. NIC transmits
```

**Key**: Same pattern as storage - command queues + event buffers.

---

## Memory Model

### Apple Silicon Unified Memory

CPU and GPU share the same physical memory. This eliminates copies but requires coordination:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED MEMORY                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ GPU-Owned Regions (StorageModePrivate preferred)            ││
│  │                                                              ││
│  │  - Application state                                         ││
│  │  - Vertex buffers                                            ││
│  │  - Compute results                                           ││
│  │  - File caches                                               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Shared Regions (StorageModeShared for I/O boundaries)       ││
│  │                                                              ││
│  │  - I/O command queues                                        ││
│  │  - Completion status flags                                   ││
│  │  - Event buffers (input, network)                            ││
│  │  - Debug/profiling output                                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Ownership Rules

1. **GPU owns application state** - CPU never reads it except for debugging
2. **Shared regions have clear ownership per-field** - Either GPU writes or CPU writes, never both
3. **Atomics for coordination** - Status flags use atomic operations
4. **No locks** - GPU cannot acquire locks (no blocking)

---

## Execution Model

### Persistent Megakernel

The GPU runs a single persistent kernel that never terminates:

```metal
kernel void megakernel(
    device AppState* state [[buffer(0)]],
    device IOBuffers* io [[buffer(1)]],
    device EventBuffers* events [[buffer(2)]],
    // ... more buffers
) {
    while (true) {
        // 1. Poll I/O completions
        process_completed_io(io);

        // 2. Read input events
        process_input_events(events);

        // 3. Run application logic
        run_app_tick(state);

        // 4. Submit new I/O requests (non-blocking)
        submit_pending_io(io);

        // 5. Emit render commands
        emit_vertices(state);

        // Grid-wide barrier for frame sync
        // (or use multiple dispatches if needed)
    }
}
```

### No Kernel Launch Overhead

Traditional GPU programming launches kernels repeatedly:
```
CPU: Launch kernel A → Wait → Launch kernel B → Wait → ...
```

Our model: One kernel, always running. CPU just feeds events and handles hardware.

---

## What This Means for Software Architecture

### Applications

- Apps are bytecode programs running inside the megakernel
- Apps communicate via shared GPU memory, not IPC
- Apps don't make syscalls - they write to command queues

### "Standard Library"

No traditional stdlib. Instead:

| Traditional | GPU-Native |
|-------------|------------|
| `open()`, `read()` | Write to I/O command queue, poll completion |
| `malloc()`, `free()` | GPU-resident arena/slab allocator |
| `pthread_create()` | Not needed - GPU threads are the model |
| `printf()` | Write to debug ring buffer |

### Language Support

For Rust on GPU:

```rust
#![no_std]

// ✅ Works
fn compute(data: &[f32]) -> f32 { ... }

// ❌ Does NOT work (no heap)
fn compute() -> Vec<f32> { ... }

// ✅ GPU-native I/O
fn load_file(path: &str) -> IoHandle {
    io_queue.submit_read(path)  // Returns immediately
}

fn check_file(handle: IoHandle) -> Option<&[u8]> {
    if io_status[handle].is_complete() {
        Some(&io_buffer[handle])
    } else {
        None  // Not ready, caller continues doing other work
    }
}
```

---

## Anti-Patterns (NEVER DO)

| Anti-Pattern | Why It's Wrong | Correct Approach |
|--------------|----------------|------------------|
| GPU waits for CPU response | CPU becomes coordinator | GPU polls, never waits |
| CPU reads GPU state each frame | Adds latency, CPU dependency | GPU writes to debug buffer if needed |
| Synchronous file load | Blocks GPU | Async queue + poll |
| CPU dispatches multiple kernels | Launch overhead, CPU in loop | Persistent megakernel |
| Request/response syscalls | Makes GPU a supplicant | Command queues + event buffers |

---

## Implementation Checklist

### Phase 1: Foundation (Current)
- [x] Persistent megakernel architecture
- [x] GPU-resident app state
- [x] Input event buffer (HID → GPU)
- [x] Bytecode VM for apps

### Phase 2: True GPU I/O
- [ ] MTLIOCommandQueue integration for file loading
- [ ] Completion polling in megakernel
- [ ] File cache in GPU memory

### Phase 3: Full Independence
- [ ] GPU-initiated storage writes
- [ ] Network packet buffers
- [ ] GPU-native memory allocator (arena + slab)

---

## References

- **BaM** (ASPLOS 2023): GPU-initiated NVMe access, 5.3x speedup
- **GPUfs** (TOCS 2014): POSIX-like API for GPU file access
- **GPUnet** (OSDI 2014): GPU-to-GPU networking without CPU
- **MTLIOCommandQueue**: Apple's async file-to-GPU loading
- **Persistent Kernels** (various): Avoiding kernel launch overhead

---

## Summary

**THE GPU IS THE COMPUTER.**

- GPU runs continuously, owns all state, makes all decisions
- CPU is a peripheral that handles hardware and produces events
- No syscalls, no blocking, no request/response
- Command queues for outbound requests, event buffers for inbound data
- GPU polls when convenient, never waits
