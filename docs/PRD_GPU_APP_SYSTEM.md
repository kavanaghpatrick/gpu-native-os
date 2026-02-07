# PRD: GPU-Centric App System (Issue #154)

## Vision: THE GPU IS THE COMPUTER

This is NOT a "GPU-accelerated" app system. This is a **GPU-NATIVE** app system where:

- **GPU decides** which apps run each frame
- **GPU manages** app memory allocation and deallocation
- **GPU handles** app lifecycle (launch, suspend, resume, close)
- **CPU only** submits command buffers and handles I/O

### Why This Matters

| Traditional (CPU-Centric) | GPU-Native |
|---------------------------|------------|
| CPU loops through apps | GPU parallel processes all apps |
| CPU encodes per-app commands | GPU megakernel contains all app logic |
| 1 context switch per app | 0 context switches (single kernel) |
| O(n) per frame for n apps | O(1) amortized per frame |
| CPU bottleneck at 10+ apps | GPU scales to 1000+ apps |

## Existing Infrastructure to Leverage

### 1. Work Queue (`work_queue.rs`)
- GPU pulls work from atomic queue
- Quantum-based execution with checkpointing
- State persists across frames
- **Reuse**: Queue pattern for app dispatch requests

### 2. Persistent Kernel (`persistent_search.rs`)
- Batch processing of work items
- STATUS_EMPTY → READY → PROCESSING → DONE state machine
- SIMD-accelerated reduction
- **Reuse**: State machine pattern for app lifecycle

### 3. Parallel Allocator (`parallel_alloc.rs`)
- Hillis-Steele prefix sum: O(log n) allocation
- 1 atomic per batch (not per allocation)
- SIMD shuffle for warp-cooperative allocation
- **Reuse**: App memory allocation

### 4. GpuApp Trait (`app.rs`)
- Standardized buffer slots (0=FrameState, 1=Input, 2+AppParams)
- Compute + render pipeline per app
- **Transform**: Compile all apps into single megakernel

## Architecture

### Core Concept: GPU App Table

```
┌─────────────────────────────────────────────────────────────────────┐
│ GPU-Resident App Table (persistent in device memory)                │
├─────────────────────────────────────────────────────────────────────┤
│ Slot 0:  [FLAGS|TYPE|STATE_OFF|VERTEX_OFF|FRAME|INPUT_HEAD|...]    │
│ Slot 1:  [FLAGS|TYPE|STATE_OFF|VERTEX_OFF|FRAME|INPUT_HEAD|...]    │
│ Slot 2:  [EMPTY]                                                    │
│ ...                                                                 │
│ Slot 63: [FLAGS|TYPE|STATE_OFF|VERTEX_OFF|FRAME|INPUT_HEAD|...]    │
├─────────────────────────────────────────────────────────────────────┤
│ Free Slot Bitmap: 0b1111111111111100... (atomic for GPU allocation) │
└─────────────────────────────────────────────────────────────────────┘
```

### App Descriptor (128 bytes, GPU-resident)

```metal
struct GpuAppDescriptor {
    // Identity & Lifecycle (16 bytes)
    uint flags;              // ACTIVE|VISIBLE|DIRTY|SUSPENDED|FOCUS
    uint app_type;           // Index into app type table
    uint slot_id;            // This slot's index
    uint window_id;          // Associated window (0 if none)

    // Memory Pointers (32 bytes)
    uint state_offset;       // Offset into unified state buffer
    uint state_size;
    uint vertex_offset;      // Offset into unified vertex buffer
    uint vertex_size;
    uint param_offset;       // Offset into unified param buffer
    uint param_size;
    uint _pad0[2];

    // Execution State (16 bytes)
    uint frame_number;       // Last frame this app ran
    uint input_head;         // Head of app's input ring buffer
    uint input_tail;         // Tail of app's input ring buffer
    uint thread_count;       // Compute dispatch size

    // Rendering (16 bytes)
    uint vertex_count;       // Dynamic vertex count
    uint clear_color;        // Packed RGBA
    float2 preferred_size;   // Preferred window size

    // GPU Scheduling (16 bytes)
    uint priority;           // 0=background, 1=normal, 2=high, 3=realtime
    uint last_run_frame;     // For scheduling fairness
    uint accumulated_time;   // CPU cycles equivalent (for throttling)
    uint _pad1;

    // Input Queue (32 bytes)
    uint input_events[8];    // Inline input ring buffer (8 packed events)
};
```

### Megakernel Architecture

All app logic compiles into a SINGLE compute kernel:

```metal
// Master kernel that contains ALL app logic
kernel void gpu_app_megakernel(
    device GpuAppDescriptor* apps [[buffer(0)]],       // App table
    device FrameState* frame [[buffer(1)]],             // OS frame state
    device uchar* unified_state [[buffer(2)]],          // All app state
    device Vertex* unified_vertices [[buffer(3)]],      // All app vertices
    device InputEvent* input_queue [[buffer(4)]],       // Global input queue
    device uint* free_slot_bitmap [[buffer(5)]],        // Slot allocation
    device AllocatorState* allocator [[buffer(6)]],     // Memory allocator
    device SchedulerState* scheduler [[buffer(7)]],     // GPU scheduler
    constant uint& frame_number [[buffer(8)]],
    uint slot_id [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Load app descriptor for this slot
    GpuAppDescriptor app = apps[slot_id];

    // Skip inactive slots
    if (!(app.flags & APP_FLAG_ACTIVE)) return;

    // Skip if not scheduled this frame
    if (!should_run_this_frame(app, scheduler, frame_number)) return;

    // Dispatch to app-specific logic based on type
    switch (app.app_type) {
        case APP_TYPE_GAME_OF_LIFE:
            game_of_life_update(app, unified_state, unified_vertices);
            break;
        case APP_TYPE_TEXT_EDITOR:
            text_editor_update(app, unified_state, unified_vertices, input_queue);
            break;
        case APP_TYPE_PARTICLES:
            particles_update(app, unified_state, unified_vertices);
            break;
        case APP_TYPE_FILESYSTEM:
            filesystem_update(app, unified_state, unified_vertices);
            break;
        // ... all app types
    }

    // Mark as rendered this frame
    apps[slot_id].last_run_frame = frame_number;
    apps[slot_id].flags &= ~APP_FLAG_DIRTY;
}
```

### GPU App Lifecycle

#### Launch (GPU-initiated)

```metal
// Called from terminal or other apps
void gpu_launch_app(
    device GpuAppDescriptor* apps,
    device uint* free_slot_bitmap,
    device AllocatorState* allocator,
    uint app_type,
    float2 window_pos
) {
    // 1. Allocate slot (atomic bitmap operation)
    uint slot = allocate_slot(free_slot_bitmap);
    if (slot == INVALID_SLOT) return; // No slots available

    // 2. Get app's memory requirements from type table
    AppTypeInfo info = get_app_type_info(app_type);

    // 3. Allocate memory using parallel allocator
    uint state_offset = parallel_alloc(allocator, info.state_size, 16);
    uint vertex_offset = parallel_alloc(allocator, info.max_vertex_size, 16);
    uint param_offset = parallel_alloc(allocator, info.param_size, 16);

    // 4. Initialize descriptor
    apps[slot] = (GpuAppDescriptor){
        .flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE | APP_FLAG_DIRTY,
        .app_type = app_type,
        .slot_id = slot,
        .state_offset = state_offset,
        .vertex_offset = vertex_offset,
        .param_offset = param_offset,
        // ... other fields
    };

    // 5. Call app's init function
    call_app_init(app_type, apps + slot, unified_state + state_offset);
}
```

#### Close (GPU-initiated)

```metal
// Called when user closes window
void gpu_close_app(
    device GpuAppDescriptor* apps,
    device uint* free_slot_bitmap,
    device AllocatorState* allocator,
    uint slot_id
) {
    GpuAppDescriptor* app = &apps[slot_id];

    // 1. Call app's cleanup function
    call_app_cleanup(app->app_type, app);

    // 2. Free memory
    parallel_free(allocator, app->state_offset, app->state_size);
    parallel_free(allocator, app->vertex_offset, app->vertex_size);
    parallel_free(allocator, app->param_offset, app->param_size);

    // 3. Free slot
    free_slot(free_slot_bitmap, slot_id);

    // 4. Clear descriptor
    apps[slot_id].flags = 0;
}
```

### GPU Scheduler

```metal
struct SchedulerState {
    atomic_uint active_app_count;
    atomic_uint total_compute_budget;  // Cycles available this frame
    atomic_uint used_compute_budget;   // Cycles consumed
    uint priority_thresholds[4];       // Budget per priority level
    uint frame_quantum;                // Max cycles per app per frame
};

// Cooperative scheduling - apps yield to others
bool should_run_this_frame(
    GpuAppDescriptor app,
    device SchedulerState* sched,
    uint frame_number
) {
    // Always run if dirty and focused
    if (app.flags & APP_FLAG_FOCUS && app.flags & APP_FLAG_DIRTY) {
        return true;
    }

    // Check compute budget for this priority level
    uint budget = sched->priority_thresholds[app.priority];
    uint used = atomic_load(&sched->used_compute_budget);

    if (used >= budget) {
        // Over budget - skip low priority apps
        if (app.priority < PRIORITY_HIGH) return false;
    }

    // Fairness: skip if ran recently and others are waiting
    if (app.last_run_frame == frame_number - 1) {
        // Check if starving apps exist
        // (simplified - real impl uses work stealing)
        return true;
    }

    return app.flags & APP_FLAG_DIRTY;
}
```

### Input Distribution

```metal
// GPU distributes input to focused app
kernel void distribute_input(
    device GpuAppDescriptor* apps,
    device InputEvent* global_input,
    device InputQueueHeader* input_header,
    uint slot_id [[thread_position_in_grid]]
) {
    GpuAppDescriptor* app = &apps[slot_id];

    // Only focused app receives keyboard input
    if (!(app->flags & APP_FLAG_FOCUS)) {
        // Still receive mouse events if cursor is over window
        if (!cursor_over_window(app)) return;
    }

    // Copy events from global queue to app's local queue
    uint head = atomic_load(&input_header->head);
    uint tail = atomic_load(&input_header->tail);

    while (head != tail) {
        InputEvent event = global_input[head % MAX_INPUT_EVENTS];

        // Filter: keyboard only to focused, mouse to all under cursor
        if (should_deliver_event(app, event)) {
            uint app_tail = app->input_tail;
            app->input_events[app_tail % 8] = pack_event(event);
            app->input_tail = app_tail + 1;
        }

        head++;
    }
}
```

## Unified Memory Model

All apps share unified buffers:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Unified State Buffer (256 MB)                                       │
│ ┌──────────┬──────────┬──────────┬────────────────────────────────┐ │
│ │ App 0    │ App 1    │ App 5    │ Free Space                     │ │
│ │ 2MB      │ 512KB    │ 8MB      │ (bump pointer @ offset X)      │ │
│ └──────────┴──────────┴──────────┴────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Unified Vertex Buffer (64 MB)                                        │
│ ┌──────────┬──────────┬──────────┬────────────────────────────────┐ │
│ │ App 0    │ App 1    │ App 5    │ Free Space                     │ │
│ │ 256KB    │ 1MB      │ 128KB    │                                │ │
│ └──────────┴──────────┴──────────┴────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

Apps access their memory via offsets stored in their descriptor:
```metal
// In app update function
device GameOfLifeState* state =
    (device GameOfLifeState*)(unified_state + app.state_offset);
device Vertex* vertices = unified_vertices + app.vertex_offset;
```

## Rendering Strategy

### Option A: Indirect Command Buffers (Metal 3)

GPU encodes its own draw commands:

```metal
// GPU builds ICB for all visible apps
kernel void build_render_commands(
    device GpuAppDescriptor* apps,
    device ICBContainer* icb,
    device Vertex* unified_vertices,
    uint slot_id [[thread_position_in_grid]]
) {
    GpuAppDescriptor app = apps[slot_id];
    if (!(app.flags & APP_FLAG_VISIBLE)) return;

    // Each app encodes its own draw command
    IndirectRenderCommand cmd;
    cmd.vertex_start = app.vertex_offset;
    cmd.vertex_count = app.vertex_count;
    // ... other render state

    uint cmd_index = atomic_fetch_add(&icb->command_count, 1);
    icb->commands[cmd_index] = cmd;
}

// CPU just submits:
// encoder.executeCommandsInBuffer(icb, range: 0..<MAX_APPS)
```

### Option B: Two-Pass Render (Simpler, Metal 2 compatible)

1. **Pass 1 (Compute)**: All apps write to unified vertex buffer
2. **Pass 2 (Render)**: Single draw call with dynamic vertex count

```rust
// CPU side (minimal)
encoder.draw_primitives(
    MTLPrimitiveType::Triangle,
    0,                          // Vertex start
    total_vertex_count_buffer,  // GPU writes actual count
);
```

## Implementation Plan

### Phase 1: GPU App Table & Megakernel Core

1. Create `GpuAppDescriptor` struct with proper Metal alignment
2. Implement GPU slot allocator (bitmap + atomic CAS)
3. Create megakernel skeleton with app type dispatch
4. Port one simple app (Game of Life) to megakernel format
5. Verify single-app execution works

### Phase 2: Memory Management Integration

1. Integrate `parallel_alloc.rs` for app memory
2. Implement GPU-initiated allocation for app launch
3. Implement GPU-initiated deallocation for app close
4. Add memory defragmentation (background compaction)

### Phase 3: Multi-App Scheduling

1. Implement GPU scheduler with priority levels
2. Add cooperative yielding (frame quantum limits)
3. Implement work stealing for load balancing
4. Add app suspension/resume

### Phase 4: Input & Window Integration

1. Implement input distribution kernel
2. Connect to existing window system
3. Add focus management
4. Implement hit testing for mouse events

### Phase 5: Rendering Pipeline

1. Implement unified vertex buffer
2. Add indirect command buffer support (Metal 3)
3. Fallback to two-pass for Metal 2
4. Add depth sorting for overlapping windows

### Phase 6: App Migration

Port existing apps to megakernel format:
- [ ] Game of Life
- [ ] Particles
- [ ] Text Editor
- [ ] Filesystem Browser
- [ ] Terminal
- [ ] Document Viewer

## CPU's Remaining Role

The CPU is NOT eliminated - it handles what GPUs cannot:

1. **Metal Pipeline Compilation** - One-time at app install
2. **I/O Operations** - File read/write, network
3. **Command Buffer Submission** - ~1 call per frame
4. **Window System Integration** - macOS Cocoa events

```rust
// CPU per-frame work (minimal)
fn frame_loop(&mut self) {
    // 1. Poll macOS events → write to GPU input buffer
    self.input.poll_events(&self.input_buffer);

    // 2. Submit megakernel
    let cmd = self.command_queue.new_command_buffer();
    self.encode_megakernel(cmd);
    cmd.present_drawable(drawable);
    cmd.commit();

    // 3. Optionally process I/O requests from GPU
    self.process_io_queue();
}
```

## Success Metrics

1. **CPU Utilization** < 5% during steady state
2. **Context Switches** = 0 (single megakernel)
3. **Memory Fragmentation** < 10% after 1000 app launches
4. **App Launch Latency** < 1ms (GPU-side)
5. **Scale**: 64+ concurrent apps without degradation

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu_os/gpu_app_system.rs` | Core GPU app table and lifecycle |
| `src/gpu_os/gpu_scheduler.rs` | GPU-native priority scheduler |
| `src/gpu_os/megakernel.rs` | Megakernel compilation and dispatch |
| `src/gpu_os/shaders/megakernel.metal` | Metal megakernel with all apps |
| `tests/test_gpu_app_system.rs` | Integration tests |

## Pseudocode Summary

```
EACH FRAME:
  GPU:
    1. Check scheduler for runnable apps (parallel over all slots)
    2. Each active slot runs its app's compute (megakernel dispatch)
    3. Apps write vertices to unified buffer
    4. Apps update their frame_number and clear DIRTY flag
    5. (Optional) GPU encodes draw commands to ICB

  CPU:
    1. Poll input → write to GPU buffer
    2. Submit single command buffer with megakernel
    3. Process any I/O requests from GPU

GPU LAUNCH APP:
  1. Atomic slot allocation from bitmap
  2. Parallel prefix allocation for memory
  3. Initialize descriptor
  4. Call app's init function

GPU CLOSE APP:
  1. Call app's cleanup function
  2. Free memory (mark for compaction)
  3. Free slot (atomic bitmap update)
  4. Clear descriptor
```
