# PRD: Single-Threadgroup GPU-Native OS Architecture

**Version**: 2.0
**Date**: 2026-01-23
**Status**: Research Phase

---

## Executive Summary

This document proposes a radical GPU-native operating system architecture where **all OS logic runs within a single GPU threadgroup** (1024 threads), eliminating the need for cross-threadgroup synchronization entirely. State persists in GPU-accessible memory, and the display refresh (VSync at 120Hz) serves as the hardware "heartbeat" that re-triggers the kernel.

**Key Innovation**: By constraining the OS to one threadgroup, we gain access to `threadgroup_barrier()` for perfect synchronization, avoid all cross-threadgroup coordination problems, and reduce CPU involvement to zero application code.

---

## Problem Statement

### Why Cross-Threadgroup Sync Fails on Apple GPUs

Previous research identified these blockers for multi-threadgroup GPU-native OS:

1. **No forward progress guarantee**: Apple GPUs may indefinitely delay threadgroups
2. **No device-scope barriers**: `threadgroup_barrier()` only syncs within one threadgroup
3. **Spin-lock deadlock risk**: Waiting threads may never be scheduled
4. **Watchdog timeout**: Kernels killed after 60-120 seconds
5. **No `grid.sync()` equivalent**: Metal lacks CUDA's cooperative groups

### The Insight

**Don't fight the architecture - embrace it.**

A single threadgroup of 1024 threads on Apple M4:
- Has **guaranteed forward progress** for all threads
- Has **full barrier support** (`threadgroup_barrier`)
- Has **32KB shared memory** with single-cycle access
- Has **32 SIMD groups** (32 threads each) for parallel work
- Can use **all SIMD intrinsics** (shuffle, ballot, reduce, prefix sum)

If 1024 threads can run an OS, we eliminate all synchronization problems.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     HARDWARE LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   IOKit     │  │   Display   │  │     GPU (M4)            │ │
│  │   HID       │  │   120Hz     │  │  ┌─────────────────┐    │ │
│  │   Driver    │  │   VSync     │  │  │ Single          │    │ │
│  └──────┬──────┘  └──────┬──────┘  │  │ Threadgroup     │    │ │
│         │                │         │  │ (1024 threads)  │    │ │
│         ▼                ▼         │  └────────┬────────┘    │ │
│  ┌─────────────────────────────────┴───────────┴─────────────┐ │
│  │                    UNIFIED MEMORY                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │ │
│  │  │  Input   │ │  Widget  │ │  Render  │ │  Framebuffer │  │ │
│  │  │  Queue   │ │  State   │ │  List    │ │  (Display)   │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **IOKit HID** writes input events to `Input Queue` (DMA, no CPU)
2. **Display VSync** triggers MTKView.draw() at 120Hz (system callback)
3. **GPU Kernel** runs single-threadgroup OS:
   - Reads input queue
   - Updates widget state
   - Computes layout
   - Renders to framebuffer
4. **Display controller** reads framebuffer (DMA, no CPU)

### Thread Allocation (1024 threads = 32 SIMD groups)

| SIMD Groups | Threads | Role | Operations |
|-------------|---------|------|------------|
| 0 | 0-31 | Input Processing | Poll input queue, decode events |
| 1-2 | 32-95 | Event Dispatch | Route events to widgets, hit testing |
| 3-6 | 96-223 | Layout Engine | Flexbox-style constraint solving |
| 7-10 | 224-351 | State Management | Widget state updates, animations |
| 11-20 | 352-671 | Geometry Generation | Vertex generation for widgets |
| 21-30 | 672-991 | Tile Rendering | Rasterize widgets to tiles |
| 31 | 992-1023 | Sync & Housekeeping | Frame timing, memory management |

---

## Memory Layout

### Threadgroup Memory (32KB, on-chip)

```c
struct ThreadgroupMemory {
    // Hot data - accessed every frame
    InputEvent pending_events[32];      // 512 bytes
    uint32_t event_count;               // 4 bytes

    // Layout scratch space
    float4 widget_bounds[64];           // 1KB - computed bounds
    float4 layout_constraints[64];      // 1KB - constraint equations

    // Render state
    uint32_t visible_widgets[64];       // 256 bytes - visibility mask
    uint32_t render_order[64];          // 256 bytes - sorted draw order

    // SIMD communication
    float simd_scratch[32][32];         // 4KB - per-SIMD scratch

    // Synchronization
    atomic_uint phase_counter;          // 4 bytes
    atomic_uint work_claimed;           // 4 bytes

    // Total: ~7KB used, 25KB available for expansion
};
```

### Device Memory (Persistent State)

```c
struct DeviceMemory {
    // Input (written by IOKit, read by GPU)
    struct {
        atomic_uint head;
        atomic_uint tail;
        InputEvent events[256];         // Ring buffer
    } input_queue;                      // 4KB

    // Widget Tree (read/write by GPU)
    struct {
        uint32_t count;
        Widget widgets[256];            // Flat array
        uint32_t parent[256];           // Parent indices
        uint32_t first_child[256];      // Child list heads
        uint32_t next_sibling[256];     // Sibling links
    } widget_tree;                      // ~64KB

    // Render Output (write by GPU, read by display)
    struct {
        uint32_t pixels[3840 * 2160];   // 4K framebuffer
    } framebuffer;                      // 33MB

    // Frame State (GPU read/write)
    struct {
        uint64_t frame_number;
        float time_seconds;
        float delta_time;
        uint32_t focused_widget;
        float2 cursor_position;
        uint32_t cursor_buttons;
    } frame_state;                      // 64 bytes
};
```

---

## Kernel Execution Model

### Phase-Based Execution

The kernel executes in **strict phases**, synchronized by `threadgroup_barrier()`:

```metal
kernel void gpu_os_main(
    device DeviceMemory* memory [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ThreadgroupMemory shared;

    // ═══════════════════════════════════════════════════════════
    // PHASE 1: INPUT COLLECTION (SIMD 0)
    // ═══════════════════════════════════════════════════════════
    if (simd_id == 0) {
        // Read new events from device memory ring buffer
        uint head = atomic_load(&memory->input_queue.head);
        uint tail = atomic_load(&memory->input_queue.tail);
        uint count = min(head - tail, 32u);

        if (simd_lane < count) {
            uint idx = (tail + simd_lane) % 256;
            shared.pending_events[simd_lane] = memory->input_queue.events[idx];
        }

        if (simd_lane == 0) {
            shared.event_count = count;
            atomic_store(&memory->input_queue.tail, tail + count);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 2: HIT TESTING (SIMD 1-2)
    // ═══════════════════════════════════════════════════════════
    if (simd_id >= 1 && simd_id <= 2) {
        // Each thread tests one widget against cursor
        uint widget_idx = (simd_id - 1) * 32 + simd_lane;
        if (widget_idx < memory->widget_tree.count) {
            Widget w = memory->widget_tree.widgets[widget_idx];
            float2 cursor = memory->frame_state.cursor_position;

            bool hit = cursor.x >= w.bounds.x &&
                       cursor.x <= w.bounds.x + w.bounds.z &&
                       cursor.y >= w.bounds.y &&
                       cursor.y <= w.bounds.y + w.bounds.w;

            // Use ballot to find all hits, pick frontmost
            simd_vote hits = simd_ballot(hit);
            // ... resolve to single focused widget
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 3: EVENT DISPATCH (SIMD 1-2)
    // ═══════════════════════════════════════════════════════════
    if (simd_id >= 1 && simd_id <= 2) {
        for (uint e = 0; e < shared.event_count; e++) {
            InputEvent event = shared.pending_events[e];
            // Dispatch to appropriate widget handler
            // All threads evaluate, predicated execution
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 4: LAYOUT (SIMD 3-6)
    // ═══════════════════════════════════════════════════════════
    if (simd_id >= 3 && simd_id <= 6) {
        // Parallel constraint solving using prefix sums
        // Level-by-level tree traversal
        // Each SIMD group handles one tree level
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 5: VISIBILITY & SORTING (SIMD 7-10)
    // ═══════════════════════════════════════════════════════════
    if (simd_id >= 7 && simd_id <= 10) {
        // Determine visible widgets
        // Sort by Z-order for correct rendering
        // Use SIMD bitonic sort
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 6: GEOMETRY GENERATION (SIMD 11-20)
    // ═══════════════════════════════════════════════════════════
    if (simd_id >= 11 && simd_id <= 20) {
        // Generate vertices for visible widgets
        // Each SIMD group handles ~6 widgets
        // Output to vertex buffer for render pass
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 7: TILE RENDERING (SIMD 21-30)
    // ═══════════════════════════════════════════════════════════
    if (simd_id >= 21 && simd_id <= 30) {
        // Software rasterization to framebuffer
        // Each SIMD group handles screen tiles
        // SDF rendering for smooth shapes
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════
    // PHASE 8: FRAME FINALIZATION (SIMD 31)
    // ═══════════════════════════════════════════════════════════
    if (simd_id == 31) {
        if (simd_lane == 0) {
            memory->frame_state.frame_number++;
            memory->frame_state.delta_time = 1.0 / 120.0;
        }
    }
}
```

---

## VSync-Driven Execution

### MTKView Integration

```swift
class GPUNativeOSView: MTKView, MTKViewDelegate {
    var osKernelPipeline: MTLComputePipelineState!
    var deviceMemory: MTLBuffer!

    func draw(in view: MTKView) {
        guard let drawable = currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // COMPUTE PASS: Run the OS kernel
        let compute = commandBuffer.makeComputeCommandEncoder()!
        compute.setComputePipelineState(osKernelPipeline)
        compute.setBuffer(deviceMemory, offset: 0, index: 0)

        // CRITICAL: Exactly 1 threadgroup of 1024 threads
        compute.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        compute.endEncoding()

        // BLIT PASS: Copy framebuffer to drawable (or render pass)
        let blit = commandBuffer.makeBlitCommandEncoder()!
        blit.copy(from: framebufferTexture, to: drawable.texture)
        blit.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
```

### What "Runs" This Code?

1. **MTKView.draw()** is called by the system's display link at VSync (120Hz)
2. **No application CPU loop** - the display subsystem triggers it
3. **Completion handlers** are optional - can fire-and-forget

---

## Capacity Analysis

### Can 1024 Threads Handle an OS?

| Task | Threads Needed | Time Budget (8.3ms) | Feasibility |
|------|----------------|---------------------|-------------|
| Input processing (32 events) | 32 | 0.01ms | ✅ Trivial |
| Hit testing (256 widgets) | 64 | 0.02ms | ✅ Trivial |
| Event dispatch | 64 | 0.1ms | ✅ Easy |
| Layout (256 widgets, 4 levels) | 128 | 0.5ms | ✅ Feasible |
| Visibility/sort | 128 | 0.2ms | ✅ Easy |
| Geometry gen | 320 | 1.0ms | ✅ Feasible |
| Tile rendering (4K) | 320 | 5.0ms | ⚠️ Tight |
| **Total** | **1024** | **~6.8ms** | ✅ Within budget |

### Scaling Limits

| Metric | Comfortable | Maximum | Bottleneck |
|--------|-------------|---------|------------|
| Widgets | 64 | 256 | Threadgroup memory |
| Screen resolution | 1080p | 4K | Render time |
| Input events/frame | 32 | 64 | Ring buffer size |
| Tree depth | 4 | 8 | Layout passes |
| Text glyphs | 1000 | 4000 | Atlas lookups |

---

## Advantages

1. **Zero synchronization bugs**: `threadgroup_barrier` is foolproof
2. **No CPU application code**: VSync-triggered, memory-mediated
3. **Guaranteed forward progress**: All 1024 threads always execute
4. **Minimal latency**: Input → Display in single kernel dispatch
5. **Power efficient**: GPU sleeps between VSync, wakes for 6-8ms
6. **No watchdog issues**: Kernel completes in <10ms, well under limit
7. **Deterministic**: Same input → same output, always

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 32KB threadgroup memory insufficient | Medium | High | Hierarchical processing, spill to device memory |
| 4K rendering exceeds 8.3ms | Medium | Medium | Tile-based incremental rendering |
| Complex layouts exceed 256 widgets | Low | Medium | Virtual scrolling, widget recycling |
| IOKit input latency | Low | Low | Separate high-priority input buffer |
| GPU driver bugs | Low | High | Extensive testing, fallback to CPU path |

---

## Research Questions

1. **Can IOKit write directly to GPU buffer?** (DMA path)
2. **What's the actual latency of MTKView.draw() from VSync?**
3. **Can we guarantee 1024-thread occupancy on M4?**
4. **How does tile rendering perform in compute shader vs fragment?**
5. **What's the power consumption of 120Hz GPU wake?**
6. **Can we achieve <1ms input-to-photon latency?**
7. **How do we handle GPU driver reset/recovery?**
8. **Can we do text rendering in 1024 threads?**
9. **How do animations work across frames?** (state interpolation)
10. **What happens if kernel exceeds 8.3ms?** (frame drop behavior)

---

## Success Criteria

1. **120fps sustained** on M4 MacBook Pro
2. **<16ms input-to-display latency** (ideally <8ms)
3. **Zero CPU usage** for steady-state operation
4. **64+ widgets** with smooth scrolling
5. **Text rendering** at 60+ characters per frame
6. **No memory leaks** in 24-hour stress test
7. **Graceful degradation** on GPU throttling

---

## Next Steps

1. **Viability Research**: Parallel analysis by multiple AI systems
2. **Prototype**: Minimal kernel with input→render pipeline
3. **Benchmark**: Measure actual timing on M4 hardware
4. **Iterate**: Optimize bottlenecks identified in profiling

---

*This PRD represents a fundamental rethinking of OS architecture, trading multi-threadgroup parallelism for guaranteed synchronization within a single, well-orchestrated threadgroup.*
