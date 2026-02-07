# PRD: GPU Persistent Kernel Architecture

**Issue**: Path C - True GPU-Native Computing
**Status**: Draft
**Created**: 2026-01-27
**Priority**: Critical - Foundation for "GPU IS THE COMPUTER"

---

## Executive Summary

This PRD defines the architecture for converting our current CPU-dispatched frame loop into a true GPU persistent kernel that runs indefinitely without CPU involvement. Currently, CPU calls `run_frame()` every frame, dispatches the megakernel, waits for completion, and repeats. The target state is: **CPU boots the kernel once, then GPU runs forever**.

### Current State (CPU in Loop)
```
CPU: while(true) {
    dispatch(megakernel)     // CPU initiates every frame
    wait_until_completed()   // CPU blocks
    present_drawable()       // CPU presents
}
```

### Target State (GPU Persistent)
```
CPU: dispatch(persistent_kernel)  // Once at boot
     // CPU exits loop, only handles I/O

GPU: while(true) {
    poll_input_queue()       // GPU reads atomics
    run_all_apps()           // GPU executes bytecode
    render_to_framebuffer()  // GPU renders
    wait_for_vsync()         // GPU polls timing
}
```

---

## Gap Analysis Summary

Based on comprehensive 10-agent investigation:

| Area | Current State | Gap | Effort |
|------|--------------|-----|--------|
| Infrastructure | GpuEventLoopState, InputQueue exist | 85% reusable | Low |
| Input Pipeline | Ring buffer works, mouse works | Need keyboard intrinsics | Medium |
| VSync/Frame Timing | CPU calls nextDrawable() | Need GPU timing buffer | High |
| App Scheduling | CPU launches apps | Need GPU request queue | Medium |
| Rendering | CPU presents drawable | Need GPU framebuffer + CPU blit | High |
| Error Recovery | None | Need heartbeat/watchdog | Medium |
| Debugging | Phase 7 debug I/O exists | Wire to persistent kernel | Low |
| Metal Constraints | N/A | Watchdog timer (~2-5s) | Design |

---

## Technical Requirements

### 1. Persistent Kernel Loop Structure

The megakernel must transition from "run once and return" to "run forever with barriers":

```metal
// CURRENT (returns after one frame)
kernel void megakernel(...) {
    if (tid != 0) return;
    execute_one_frame();
    // Returns to CPU
}

// TARGET (persistent)
kernel void persistent_megakernel(...) {
    uint tid = thread_position_in_grid.x;

    while (true) {
        // Heartbeat - prevents GPU watchdog kill
        if (tid == 0) {
            atomic_fetch_add_explicit(&heartbeat, 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Check for shutdown request
        if (atomic_load_explicit(&shutdown_flag, memory_order_relaxed)) {
            break;
        }

        // Poll input queue (all threads can help process)
        process_input_batch(tid, ...);
        threadgroup_barrier(mem_flags::mem_device);

        // Run apps (distributed across threads)
        run_app_frame(tid, ...);
        threadgroup_barrier(mem_flags::mem_device);

        // Render (parallel vertex generation)
        render_frame(tid, ...);
        threadgroup_barrier(mem_flags::mem_device);

        // VSync wait (thread 0 polls timing buffer)
        if (tid == 0) {
            wait_for_vsync();
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

### 2. GPU Watchdog Timer Mitigation

macOS kills GPU kernels that run too long without memory barriers. Research shows:
- **Timeout**: ~2-5 seconds without barrier
- **Solution**: Regular `threadgroup_barrier(mem_flags::mem_device)` calls
- **Frequency**: Every frame iteration (60+ times/second) is sufficient

```metal
// Every frame iteration must include:
threadgroup_barrier(mem_flags::mem_device);  // Keeps watchdog happy
```

### 3. VSync/Frame Timing Buffer

CPU cannot call `nextDrawable()` from GPU. Solution: timing buffer with GPU polling.

```rust
#[repr(C)]
pub struct FrameTimingBuffer {
    /// CPU writes current display timestamp (mach_absolute_time)
    pub display_timestamp: AtomicU64,
    /// CPU writes next vsync deadline
    pub next_vsync_deadline: AtomicU64,
    /// GPU reads to calculate wait time
    pub gpu_timestamp_offset: AtomicU64,
    /// Frame counter for sync
    pub frame_number: AtomicU64,
}
```

**CPU Thread (runs independently)**:
```rust
fn vsync_updater_thread(timing: &FrameTimingBuffer, display_link: CVDisplayLink) {
    loop {
        // CVDisplayLink callback updates timing
        let now = mach_absolute_time();
        let next_vsync = display_link.next_vsync_time();
        timing.display_timestamp.store(now, Ordering::Release);
        timing.next_vsync_deadline.store(next_vsync, Ordering::Release);
    }
}
```

**GPU Polling**:
```metal
void wait_for_vsync(device FrameTimingBuffer* timing) {
    uint64_t target = atomic_load_explicit(&timing->next_vsync_deadline,
                                            memory_order_relaxed);
    // Spin-wait until deadline (GPU cycles are cheap)
    while (get_gpu_timestamp() < target) {
        // Could yield to other wavefronts via barrier
        threadgroup_barrier(mem_flags::mem_none);
    }
}
```

### 4. App Lifecycle Queue

GPU needs to handle app launch/terminate without CPU dispatch:

```rust
#[repr(C)]
pub struct AppLifecycleQueue {
    /// Requests: LAUNCH=1, TERMINATE=2, FOCUS=3
    pub requests: [AppRequest; 16],
    pub head: AtomicU32,  // CPU writes
    pub tail: AtomicU32,  // GPU reads
}

#[repr(C)]
pub struct AppRequest {
    pub request_type: u32,
    pub app_id: u32,
    pub wasm_ptr: u64,      // For LAUNCH: pointer to WASM bytecode
    pub wasm_len: u32,
}
```

**CPU**: Writes launch requests to queue
**GPU**: Polls queue, processes requests, launches apps

### 5. Framebuffer + CPU Blit

GPU cannot call `drawable.present()`. Solution: GPU renders to buffer, CPU blits.

```rust
pub struct GpuFramebuffer {
    /// GPU writes RGBA pixels here
    pub pixels: Buffer,  // MTLStorageModeShared for CPU access
    pub width: u32,
    pub height: u32,
    /// GPU sets to 1 when frame complete
    pub frame_ready: AtomicU32,
    /// CPU sets to 0 after blit
    pub frame_consumed: AtomicU32,
}
```

**GPU** (end of frame):
```metal
// Write final pixels to framebuffer
render_to_buffer(framebuffer->pixels, ...);
atomic_store_explicit(&framebuffer->frame_ready, 1, memory_order_release);
// Wait for CPU to consume before next frame
while (atomic_load_explicit(&framebuffer->frame_consumed,
                             memory_order_acquire) == 0) {
    threadgroup_barrier(mem_flags::mem_none);
}
atomic_store_explicit(&framebuffer->frame_consumed, 0, memory_order_relaxed);
```

**CPU** (async blit thread):
```rust
fn blit_thread(framebuffer: &GpuFramebuffer, layer: CAMetalLayer) {
    loop {
        // Wait for GPU to produce frame
        while framebuffer.frame_ready.load(Ordering::Acquire) == 0 {
            std::hint::spin_loop();
        }

        // Get drawable and blit
        let drawable = layer.next_drawable();
        blit_buffer_to_drawable(&framebuffer.pixels, drawable);
        drawable.present();

        // Signal GPU we consumed
        framebuffer.frame_ready.store(0, Ordering::Relaxed);
        framebuffer.frame_consumed.store(1, Ordering::Release);
    }
}
```

### 6. Heartbeat and Watchdog

For graceful error recovery:

```rust
#[repr(C)]
pub struct KernelHealthBuffer {
    /// GPU increments every frame
    pub heartbeat: AtomicU64,
    /// Last heartbeat CPU observed
    pub last_observed_heartbeat: AtomicU64,
    /// CPU sets to request shutdown
    pub shutdown_requested: AtomicU32,
    /// GPU sets when shutting down
    pub shutdown_acknowledged: AtomicU32,
    /// Error code if kernel crashed
    pub error_code: AtomicU32,
}
```

**CPU Monitor Thread**:
```rust
fn monitor_thread(health: &KernelHealthBuffer) {
    loop {
        thread::sleep(Duration::from_millis(100));
        let current = health.heartbeat.load(Ordering::Acquire);
        let last = health.last_observed_heartbeat.load(Ordering::Relaxed);

        if current == last {
            // Kernel might be stuck
            warn!("GPU heartbeat stalled at {}", current);
            // After N failures, request restart
        }
        health.last_observed_heartbeat.store(current, Ordering::Relaxed);
    }
}
```

### 7. Debug Output Ring Buffer

Wire existing Phase 7 debug I/O to persistent kernel:

```rust
#[repr(C)]
pub struct DebugRingBuffer {
    pub messages: [DebugMessage; 1024],
    pub head: AtomicU32,  // GPU writes
    pub tail: AtomicU32,  // CPU reads
}

#[repr(C)]
pub struct DebugMessage {
    pub level: u32,       // 0=trace, 1=debug, 2=info, 3=warn, 4=error
    pub app_id: u32,
    pub message: [u8; 120],
}
```

---

## Implementation Phases

### Phase 1: Persistent Loop Foundation (2-3 days)
**Goal**: Megakernel runs forever without crashing

- [ ] Add `while(true)` loop to megakernel
- [ ] Add heartbeat counter increment
- [ ] Add `threadgroup_barrier` at strategic points
- [ ] Add shutdown flag check
- [ ] Verify kernel survives >10 seconds

**Success Criteria**: Kernel runs 60 seconds without watchdog kill

### Phase 2: Input Pipeline Integration (1-2 days)
**Goal**: GPU processes input without CPU dispatch

- [ ] Connect existing InputQueue to persistent loop
- [ ] Add keyboard intrinsics (already documented in Issue #149)
- [ ] GPU polls input every iteration
- [ ] Test mouse + keyboard input

**Success Criteria**: Bouncing balls responds to keyboard without CPU dispatch

### Phase 3: Frame Timing and VSync (2-3 days)
**Goal**: GPU maintains 60fps without CPU timing

- [ ] Create FrameTimingBuffer struct
- [ ] CPU thread updates timing from CVDisplayLink
- [ ] GPU polls timing buffer for vsync
- [ ] Implement frame pacing (wait for vsync)

**Success Criteria**: Consistent 60fps with <1ms frame time variance

### Phase 4: Framebuffer Rendering (2-3 days)
**Goal**: GPU renders, CPU blits to display

- [ ] Create GpuFramebuffer with shared storage
- [ ] Modify render pass to target buffer (not drawable)
- [ ] CPU blit thread copies to drawable
- [ ] Double-buffer to avoid tearing

**Success Criteria**: Visual output matches current quality at 60fps

### Phase 5: App Lifecycle Queue (1-2 days)
**Goal**: Launch/terminate apps without CPU dispatch

- [ ] Create AppLifecycleQueue struct
- [ ] GPU polls queue for requests
- [ ] Implement LAUNCH, TERMINATE, FOCUS requests
- [ ] CPU only writes to queue, never dispatches

**Success Criteria**: Can launch 3 apps via queue without CPU dispatch

### Phase 6: Error Recovery and Debugging (1-2 days)
**Goal**: Production-ready stability

- [ ] Implement KernelHealthBuffer
- [ ] CPU monitor thread with restart capability
- [ ] Wire DebugRingBuffer to persistent loop
- [ ] Graceful shutdown on error

**Success Criteria**: Kernel recovers from app crash without full restart

---

## Risk Assessment

### High Risk
1. **GPU Watchdog Timer**
   - **Risk**: macOS kills long-running kernels
   - **Mitigation**: Regular barriers, heartbeat monitoring
   - **Validation**: Test 10-minute continuous operation

2. **VSync Synchronization**
   - **Risk**: Tearing or stuttering without proper timing
   - **Mitigation**: CVDisplayLink + GPU polling
   - **Validation**: Frame time histogram analysis

### Medium Risk
3. **Memory Coherence**
   - **Risk**: CPU/GPU see stale data
   - **Mitigation**: Proper memory ordering (acquire/release)
   - **Validation**: Stress test with high-frequency updates

4. **Blit Latency**
   - **Risk**: Added latency from CPU blit
   - **Mitigation**: Double buffering, async blit thread
   - **Validation**: Measure end-to-end latency

### Low Risk
5. **Debug Buffer Overflow**
   - **Risk**: Fast GPU overruns slow CPU reader
   - **Mitigation**: Ring buffer with overwrite policy
   - **Validation**: Flood test with high debug output

---

## Success Criteria

### Functional
- [ ] Kernel runs >1 hour without crash
- [ ] Apps respond to input without CPU dispatch
- [ ] Visual output at 60fps, no tearing
- [ ] Can launch/terminate apps via queue
- [ ] Graceful shutdown on request

### Performance
- [ ] CPU utilization <5% during steady state
- [ ] GPU occupancy >50% (better than current single-thread)
- [ ] Frame latency <16.6ms (60fps target)
- [ ] Input latency <1 frame

### Stability
- [ ] Survives app crash without kernel restart
- [ ] Heartbeat monitoring detects stalls
- [ ] Debug output readable during operation
- [ ] Clean shutdown completes in <1 second

---

## Migration Path

### Step 1: Shadow Mode
Run persistent kernel alongside current CPU loop. Compare outputs.

### Step 2: Switchable Mode
Flag to enable persistent kernel. Can revert if issues.

### Step 3: Full Migration
Remove CPU loop. Persistent kernel is default.

### Step 4: Optimization
- Parallelize bytecode execution across threads
- Optimize GPU framebuffer format
- Reduce barrier frequency where safe

---

## Appendix: Reference Implementations

### CUDA Persistent Threads
- megakernel pattern with `while(true)`
- work-stealing from global queues
- Used in: OptiX, production ray tracers

### Unreal Nanite
- GPU-driven rendering without CPU submission
- Indirect dispatch for variable workloads
- Hierarchical culling entirely on GPU

### Vulkan Timeline Semaphores
- GPU-to-GPU synchronization without CPU
- Chain command buffers with dependencies
- Models for our frame pacing

---

## Appendix: Existing Infrastructure to Reuse

### From event_loop.rs
```rust
pub struct GpuEventLoopState {
    pub frame_number: u64,
    pub should_quit: u32,
    pub needs_redraw: u32,
    // ... already 85% of what we need
}
```

### From memory.rs
```rust
pub struct InputQueue {
    pub events: [InputEvent; INPUT_QUEUE_SIZE],
    pub head: AtomicU32,
    pub tail: AtomicU32,
}
// Already working, just needs keyboard
```

### From gpu_app_system.rs
```rust
// Megakernel infrastructure exists
// Just needs while(true) wrapper and barriers
```

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Persistent Loop | 2-3 days | None |
| Phase 2: Input | 1-2 days | Phase 1 |
| Phase 3: VSync | 2-3 days | Phase 1 |
| Phase 4: Framebuffer | 2-3 days | Phase 1, 3 |
| Phase 5: App Queue | 1-2 days | Phase 1 |
| Phase 6: Error Recovery | 1-2 days | Phase 1-5 |

**Critical Path**: Phase 1 → Phase 3 → Phase 4 = 6-9 days
**Total with Parallel Work**: 8-12 days
**With Buffer**: 10-17 days

---

## Open Questions

1. **Thread Count**: Should persistent kernel use 256 threads (current) or more?
   - More threads = better parallelism for rendering
   - Risk: More complex synchronization

2. **Double vs Triple Buffering**: For framebuffer
   - Double: Lower latency, simpler
   - Triple: Smoother, handles CPU blit variance

3. **App Isolation**: What happens if one app infinite loops?
   - Current: Single-threaded, would block everything
   - Option: Per-app timeout with forced termination

4. **Debug Mode**: Should debug build have different kernel?
   - More barriers for debugging
   - Slower but safer

---

## Conclusion

The persistent kernel architecture is achievable with existing infrastructure. 85% of the required components exist. The main gaps are:

1. **The while(true) loop itself** - straightforward
2. **VSync/frame timing** - requires CPU thread + GPU polling
3. **Framebuffer + CPU blit** - required for display
4. **Watchdog mitigation** - barriers are sufficient

Estimated effort: **10-17 days** for full implementation with proper testing.

This is the foundation for "THE GPU IS THE COMPUTER" - once complete, CPU involvement drops to <5% and exists only for:
- Initial boot (hardware requirement)
- Network I/O (NIC limitation)
- Display present (Metal limitation - CPU must call present())

Everything else runs on GPU indefinitely.
