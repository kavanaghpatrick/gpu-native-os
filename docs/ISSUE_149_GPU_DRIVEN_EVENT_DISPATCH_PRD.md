# Issue #149: GPU-Driven Event Dispatch - Eliminate CPU from Event Loop

## Integration with Existing Infrastructure

**IMPORTANT:** This issue extends existing code rather than duplicating it.

### Existing Code to REUSE (DO NOT RECREATE)

| Component | Location | Description |
|-----------|----------|-------------|
| `GpuRuntime` | app.rs:156-181 | Extend this, don't create new `GpuEventLoop` |
| `GpuRuntime.command_queue` | app.rs:158 | Use for all dispatch |
| `GpuRuntime.shared_event` | app.rs:176 | Use for async signaling |
| `InputHandler.queue_buffer` | input.rs:40 | This IS the GPU input queue |
| `InputHandler.push_*()` | input.rs:89-125 | Use for pushing events |
| `InputEvent` | memory.rs:48-64 | 28-byte event structure |
| `InputQueue` | memory.rs:83-93 | Ring buffer with head/tail |

### NEW Structures to Add

| Component | Purpose |
|-----------|---------|
| `GpuEventLoopState` | GPU-resident drag/resize/focus state (NEW) |
| `HitTestResult` | GPU hit test output (NEW) |
| GPU kernels | Event routing, hit testing, window ops (NEW) |

### Integration Pattern

```rust
// CORRECT: Extend GpuRuntime
impl GpuRuntime {
    // New fields added to GpuRuntime:
    // - event_loop_state_buffer: Buffer (GpuEventLoopState)
    // - event_loop_pipeline: ComputePipelineState
    // - hit_test_pipeline: ComputePipelineState

    pub fn start_gpu_event_loop(&self) {
        // Uses existing self.command_queue
        // Uses existing self.input.buffer() for input queue
        // Uses existing self.shared_event for async
    }
}

// WRONG: Don't create separate struct
// pub struct GpuEventLoop { device, command_queue, ... } // DUPLICATES GpuRuntime
```

---

## Problem Statement

**This is the fundamental architectural issue.** Despite having GPU kernels for hit testing, window operations, and input handling, the CPU still orchestrates all dispatch:

```rust
// Current: CPU orchestrates everything
fn on_mouse_down(&mut self, x: f32, y: f32) {
    // CPU dispatches hit test
    let hit = self.window_manager.hit_test_gpu(x, y);  // GPU runs, CPU WAITS

    // CPU reads result and decides
    match hit.region {
        HitRegion::Title => {
            // CPU dispatches window drag
            self.window_manager.start_drag_gpu(hit.window_index);  // GPU runs, CPU WAITS
        }
        HitRegion::Content => {
            // CPU dispatches to app
            self.dispatch_to_app(hit.window_index);  // More CPU work
        }
        // ...
    }
}
```

**The CPU is doing what the GPU should do:** reading results, making decisions, dispatching next operations.

## The Vision

**GPU handles the entire event loop:**

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU EVENT LOOP                            │
│                                                              │
│   Input Queue ─→ Event Router ─→ Hit Test ─→ Dispatcher     │
│        │              │              │            │          │
│        │              │              │            ▼          │
│        │              │              │     ┌──────────────┐  │
│        │              │              │     │ Window Ops   │  │
│        │              │              │     │ App Input    │  │
│        │              │              │     │ Menu/Dock    │  │
│        │              │              │     │ Render       │  │
│        │              │              │     └──────────────┘  │
│        │              │              │            │          │
│        └──────────────┴──────────────┴────────────┘          │
│                           ▲                                  │
│                           │                                  │
│                    (loops forever)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (only for external I/O)
                            ▼
                    ┌──────────────┐
                    │     CPU      │
                    │  - File I/O  │
                    │  - Network   │
                    │  - System    │
                    └──────────────┘
```

## Solution: GPU Work Graphs

Use Metal's indirect command buffers and persistent kernels to create a GPU-resident event loop:

### Key Components

1. **GPU Event Router Kernel** - Runs continuously, processes input queue
2. **GPU Dispatch Table** - Function pointers (kernel indices) for each event type
3. **GPU State Machine** - Tracks drag/resize/focus state on GPU
4. **Indirect Command Buffer** - GPU encodes its own dispatch commands

## Technical Design

### GPU Event Router (Persistent Kernel)

**NOTE:** Uses EXISTING `InputQueue` and `InputEvent` from app.rs/memory.rs.

```metal
// src/gpu_os/event_loop.metal

// REUSE: These structures already exist in APP_SHADER_HEADER (app.rs:617-646)
// struct InputEvent { ushort event_type; ushort keycode; float2 position; float2 delta; uint modifiers; uint timestamp; };
// struct InputQueue { atomic_uint head; atomic_uint tail; uint _padding[2]; InputEvent events[256]; };

// Dispatch targets
#define DISPATCH_NONE           0
#define DISPATCH_HIT_TEST       1
#define DISPATCH_WINDOW_MOVE    2
#define DISPATCH_WINDOW_RESIZE  3
#define DISPATCH_WINDOW_FOCUS   4
#define DISPATCH_APP_INPUT      5
#define DISPATCH_MENU_CLICK     6
#define DISPATCH_DOCK_CLICK     7
#define DISPATCH_RENDER         8

#define INVALID_WINDOW 0xFFFFFFFF
#define QUEUE_SIZE 256

// NEW: GPU Event Loop State (separate from InputQueue which already exists)
// This contains interaction state NOT in InputQueue
struct GpuEventLoopState {
    // Interaction state
    uint drag_window;
    float drag_start_x;
    float drag_start_y;
    float window_start_x;
    float window_start_y;

    uint resize_window;
    uint resize_edge;
    float resize_start_x;
    float resize_start_y;
    float window_start_w;
    float window_start_h;

    uint focused_window;
    uint hovered_window;

    // Dispatch control
    atomic_uint next_dispatch;
    uint dispatch_param;

    // Frame state
    atomic_uint frame_dirty;
    float mouse_x;
    float mouse_y;
    uint mouse_buttons;
};

// Main event loop - runs FOREVER on GPU
// Buffer 0: EXISTING InputQueue from InputHandler.buffer()
// Buffer 1: NEW GpuEventLoopState
kernel void gpu_event_loop(
    device InputQueue* input_queue [[buffer(0)]],   // REUSE from InputHandler
    device GpuEventLoopState* state [[buffer(1)]],  // NEW state buffer
    device Window* windows [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Only thread 0 processes events (others assist with parallel ops)
    if (tid != 0) return;

    while (true) {  // Persistent kernel - runs forever
        // Read from EXISTING InputQueue (head/tail are in the queue, not state)
        uint tail = atomic_load_explicit(&input_queue->tail, memory_order_acquire);
        uint head = atomic_load_explicit(&input_queue->head, memory_order_relaxed);

        if (head == tail) {
            // No input - check if we need to render
            if (atomic_load_explicit(&state->frame_dirty, memory_order_relaxed)) {
                atomic_store_explicit(&state->next_dispatch, DISPATCH_RENDER, memory_order_release);
                atomic_store_explicit(&state->frame_dirty, 0, memory_order_relaxed);
                return;  // Exit to let render dispatch happen
            }
            continue;  // Spin wait for input
        }

        // Process next event from EXISTING InputQueue
        InputEvent event = input_queue->events[head % QUEUE_SIZE];
        atomic_fetch_add_explicit(&input_queue->head, 1, memory_order_release);

        // Update mouse position
        if (event.event_type == INPUT_MOUSE_MOVE ||
            event.event_type == INPUT_MOUSE_DOWN ||
            event.event_type == INPUT_MOUSE_UP) {
            state->mouse_x = event.position.x;
            state->mouse_y = event.position.y;
        }

        // Route event using EXISTING InputEventType values from memory.rs
        switch (event.event_type) {
            case INPUT_MOUSE_DOWN:  // 2
                state->mouse_buttons |= (1 << event.keycode);
                atomic_store_explicit(&state->next_dispatch, DISPATCH_HIT_TEST, memory_order_release);
                return;

            case INPUT_MOUSE_MOVE:  // 1
                if (state->drag_window != INVALID_WINDOW) {
                    atomic_store_explicit(&state->next_dispatch, DISPATCH_WINDOW_MOVE, memory_order_release);
                    state->dispatch_param = state->drag_window;
                    return;
                } else if (state->resize_window != INVALID_WINDOW) {
                    atomic_store_explicit(&state->next_dispatch, DISPATCH_WINDOW_RESIZE, memory_order_release);
                    return;
                }
                atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
                break;

            case INPUT_MOUSE_UP:  // 3
                state->mouse_buttons &= ~(1 << event.keycode);
                state->drag_window = INVALID_WINDOW;
                state->resize_window = INVALID_WINDOW;
                atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
                break;

            case INPUT_KEY_DOWN:   // 5
            case INPUT_KEY_UP:     // 6
            case INPUT_KEY_REPEAT: // 7
                if (state->focused_window != INVALID_WINDOW) {
                    atomic_store_explicit(&state->next_dispatch, DISPATCH_APP_INPUT, memory_order_release);
                    state->dispatch_param = state->focused_window;
                    return;
                }
                break;
        }
    }
}

// Hit test result handler - called after hit test completes
kernel void handle_hit_test_result(
    device EventLoopState* state [[buffer(0)]],
    device HitTestResult* hit [[buffer(1)]],
    device Window* windows [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    HitTestResult result = *hit;

    switch (result.region) {
        case REGION_TITLE:
            // Start window drag
            state->drag_window = result.window_index;
            state->focused_window = result.window_index;
            // Bring to front
            atomic_store_explicit(&state->next_dispatch, DISPATCH_WINDOW_FOCUS, memory_order_release);
            break;

        case REGION_RESIZE:
            // Start window resize
            state->resize_window = result.window_index;
            state->resize_edge = result.resize_edge;
            state->focused_window = result.window_index;
            break;

        case REGION_CLOSE_BUTTON:
            // Close window (GPU can do this!)
            windows[result.window_index].flags &= ~WINDOW_VISIBLE;
            atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
            break;

        case REGION_CONTENT:
            // Dispatch to app
            state->focused_window = result.window_index;
            atomic_store_explicit(&state->next_dispatch, DISPATCH_APP_INPUT, memory_order_release);
            atomic_store_explicit(&state->dispatch_param, result.window_index, memory_order_relaxed);
            break;

        case REGION_NONE:
            // Click on desktop - deselect
            state->focused_window = INVALID_WINDOW;
            break;
    }

    atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
}
```

### GPU Work Graph with Indirect Dispatch

```metal
// Indirect dispatch arguments - GPU fills these in
struct IndirectDispatchArgs {
    uint threadgroups_x;
    uint threadgroups_y;
    uint threadgroups_z;
    uint kernel_index;  // Which kernel to run next
};

// Master dispatcher - reads next_dispatch and sets up indirect args
kernel void setup_next_dispatch(
    device EventLoopState* state [[buffer(0)]],
    device IndirectDispatchArgs* args [[buffer(1)]],
    constant DispatchTable& table [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint dispatch = atomic_load_explicit(&state->next_dispatch, memory_order_acquire);

    // Reset for next iteration
    atomic_store_explicit(&state->next_dispatch, DISPATCH_NONE, memory_order_release);

    // Set up indirect dispatch
    switch (dispatch) {
        case DISPATCH_HIT_TEST:
            args->threadgroups_x = (table.window_count + 63) / 64;
            args->threadgroups_y = 1;
            args->threadgroups_z = 1;
            args->kernel_index = table.hit_test_kernel;
            break;

        case DISPATCH_WINDOW_MOVE:
            args->threadgroups_x = 1;
            args->kernel_index = table.window_move_kernel;
            break;

        case DISPATCH_RENDER:
            args->threadgroups_x = table.vertex_groups;
            args->kernel_index = table.render_kernel;
            break;

        // ... other dispatches
    }
}
```

### Rust Implementation: Extending GpuRuntime

**IMPORTANT:** We extend `GpuRuntime` (app.rs) rather than creating a new struct.

```rust
// src/gpu_os/event_loop.rs

use super::app::GpuRuntime;
use super::memory::{InputEvent, InputQueue};  // REUSE existing types
use super::input::InputHandler;                // REUSE existing handler

/// GPU-resident event loop state (NEW - doesn't exist yet)
/// This is the ONLY new state structure needed.
#[repr(C)]
pub struct GpuEventLoopState {
    // Interaction state (GPU-resident)
    pub drag_window: u32,
    pub drag_start_x: f32,
    pub drag_start_y: f32,
    pub window_start_x: f32,
    pub window_start_y: f32,

    pub resize_window: u32,
    pub resize_edge: u32,
    pub resize_start_x: f32,
    pub resize_start_y: f32,
    pub window_start_w: f32,
    pub window_start_h: f32,

    pub focused_window: u32,
    pub hovered_window: u32,

    // Dispatch control
    pub next_dispatch: u32,
    pub dispatch_param: u32,

    // Frame state
    pub frame_dirty: u32,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_buttons: u32,
}

pub const INVALID_WINDOW: u32 = 0xFFFFFFFF;

/// Extend GpuRuntime with event loop capability
impl GpuRuntime {
    /// Initialize GPU event loop (call once at startup)
    /// Returns the event loop state buffer for later access
    pub fn init_event_loop(&self) -> EventLoopHandle {
        // Create NEW state buffer for GPU event loop state
        let state_buffer = self.device.new_buffer(
            std::mem::size_of::<GpuEventLoopState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize state
        unsafe {
            let ptr = state_buffer.contents() as *mut GpuEventLoopState;
            (*ptr) = GpuEventLoopState {
                drag_window: INVALID_WINDOW,
                resize_window: INVALID_WINDOW,
                focused_window: INVALID_WINDOW,
                hovered_window: INVALID_WINDOW,
                next_dispatch: 0,
                dispatch_param: 0,
                frame_dirty: 0,
                mouse_x: 0.0,
                mouse_y: 0.0,
                mouse_buttons: 0,
                drag_start_x: 0.0,
                drag_start_y: 0.0,
                window_start_x: 0.0,
                window_start_y: 0.0,
                resize_edge: 0,
                resize_start_x: 0.0,
                resize_start_y: 0.0,
                window_start_w: 0.0,
                window_start_h: 0.0,
            };
        }

        // Compile event loop kernels
        let library = self.device.new_library_with_source(
            EVENT_LOOP_SHADER_SOURCE,
            &CompileOptions::new()
        ).expect("Failed to compile event loop shaders");

        let event_loop_pipeline = self.device.new_compute_pipeline_state_with_function(
            &library.get_function("gpu_event_loop", None).unwrap()
        ).unwrap();

        EventLoopHandle {
            state_buffer,
            event_loop_pipeline,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the GPU event loop - CPU only called once!
    pub fn start_event_loop(&self, handle: &EventLoopHandle, windows_buffer: &Buffer) {
        handle.running.store(true, Ordering::SeqCst);

        // Use EXISTING command_queue from GpuRuntime
        let command_buffer = self.command_queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&handle.event_loop_pipeline);

        // REUSE existing InputHandler.buffer() as input queue (slot 0)
        encoder.set_buffer(0, Some(self.input.buffer()), 0);

        // NEW event loop state buffer (slot 1)
        encoder.set_buffer(1, Some(&handle.state_buffer), 0);

        // Windows buffer (slot 2)
        encoder.set_buffer(2, Some(windows_buffer), 0);

        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        // Use EXISTING shared_event for async signaling
        let signal_value = self.next_signal_value();
        command_buffer.encode_signal_event(self.shared_event(), signal_value);

        // Completion handler continues the loop
        let handle_clone = handle.clone();
        let runtime_ref = self as *const GpuRuntime;
        let handler = block::ConcreteBlock::new(move |_: &CommandBufferRef| {
            if handle_clone.running.load(Ordering::SeqCst) {
                unsafe { (*runtime_ref).continue_event_loop(&handle_clone); }
            }
        });

        command_buffer.add_completed_handler(&handler.copy());
        command_buffer.commit();
    }

    fn continue_event_loop(&self, handle: &EventLoopHandle) {
        // Read what GPU wants to do next
        let dispatch = unsafe {
            let state_ptr = handle.state_buffer.contents() as *const GpuEventLoopState;
            (*state_ptr).next_dispatch
        };

        // Dispatch requested kernel using EXISTING command_queue
        let command_buffer = self.command_queue.new_command_buffer();

        match dispatch {
            DISPATCH_NONE => {
                // Re-dispatch event loop
            }
            DISPATCH_HIT_TEST => {
                // Encode hit test kernel
            }
            DISPATCH_WINDOW_MOVE => {
                // Encode window move kernel
            }
            DISPATCH_RENDER => {
                // Encode render
            }
            _ => {}
        }

        // Continue loop with completion handler...
    }

    // Input is pushed using EXISTING InputHandler methods:
    // self.input.push_mouse_move(x, y, dx, dy);
    // self.input.push_mouse_button(button, pressed, x, y);
    // self.input.push_key(keycode, pressed, modifiers);
    // NO NEW push_input() method needed!
}

/// Handle returned by init_event_loop()
pub struct EventLoopHandle {
    state_buffer: Buffer,
    event_loop_pipeline: ComputePipelineState,
    running: Arc<AtomicBool>,
}

impl EventLoopHandle {
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn read_state(&self) -> GpuEventLoopState {
        unsafe { *(self.state_buffer.contents() as *const GpuEventLoopState) }
    }
}
```

### Even Better: Full GPU Autonomy with MTLResidencySet

```rust
// Make buffers GPU-resident so GPU can access without CPU
impl GpuEventLoop {
    pub fn make_resident(&self) {
        let residency_set = self.device.new_residency_set();
        residency_set.add_allocation(&self.state_buffer);
        residency_set.add_allocation(&self.input_queue_buffer);
        residency_set.add_allocation(&self.windows_buffer);
        residency_set.commit();
    }
}
```

## Architecture Comparison

### Before (CPU Orchestrated)
```
Frame 1: CPU dispatch hit_test → GPU runs → CPU waits → CPU reads → CPU dispatch move → GPU runs → CPU waits
Frame 2: CPU dispatch hit_test → GPU runs → CPU waits → CPU reads → CPU dispatch move → GPU runs → CPU waits
...

CPU involvement: Every single frame, multiple times
```

### After (GPU Driven)
```
Startup: CPU dispatch event_loop → GPU runs forever
Frame 1: GPU: event → hit_test → handle → move → render (no CPU)
Frame 2: GPU: event → hit_test → handle → move → render (no CPU)
...

CPU involvement: Only at startup and for I/O
```

## Pseudocode

### Data Structures

```
# EXISTING: InputEvent (memory.rs:48-64) - DO NOT RECREATE
# Already defined in memory.rs and APP_SHADER_HEADER
struct InputEvent:                    # EXISTING - 28 bytes
    event_type: u16                   # InputEventType enum
    keycode: u16                      # HID keycode or mouse button
    position: [f32; 2]                # Cursor position
    delta: [f32; 2]                   # Movement delta
    modifiers: u32                    # Modifier keys
    timestamp: u32                    # Frame-relative timestamp

# EXISTING: InputQueue (memory.rs:83-93) - DO NOT RECREATE
# Already defined, used by InputHandler
struct InputQueue:                    # EXISTING - in InputHandler.buffer()
    head: atomic_uint                 # Read position (GPU updates)
    tail: atomic_uint                 # Write position (CPU updates via InputHandler.push_*)
    _padding: [u32; 2]
    events: [InputEvent; 256]         # Ring buffer

# NEW: GpuEventLoopState - ADD THIS
# Contains interaction state NOT in InputQueue
struct GpuEventLoopState:             # NEW - add to event_loop.rs
    # Interaction state machine
    drag_window: uint                 # INVALID if not dragging
    drag_start_x: float
    drag_start_y: float
    window_start_x: float
    window_start_y: float

    resize_window: uint               # INVALID if not resizing
    resize_edge: uint
    resize_start_x: float
    resize_start_y: float
    window_start_w: float
    window_start_h: float

    focused_window: uint              # Currently focused window
    hovered_window: uint              # Window under cursor

    # Dispatch control
    next_dispatch: atomic_uint        # What to do next
    dispatch_param: uint              # Parameter for dispatch

    # Frame state
    frame_dirty: atomic_uint          # 1 if need to render
    mouse_x: float
    mouse_y: float
    mouse_buttons: uint               # Bitmask of pressed buttons

# NEW: HitTestResult - ADD THIS
struct HitTestResult:
    window_index: uint
    region: uint                      # TITLE, CONTENT, CLOSE, MINIMIZE, MAXIMIZE, RESIZE
    resize_edge: uint                 # LEFT, RIGHT, TOP, BOTTOM (can be combined)
    local_x: float
    local_y: float

INVALID_WINDOW = 0xFFFFFFFF
QUEUE_SIZE = 256                      # Matches InputQueue::CAPACITY
```

### Main Event Loop Kernel

```
# GPU Event Loop - processes events until it needs to dispatch a sub-kernel
# Buffer 0: EXISTING InputQueue from InputHandler.buffer()
# Buffer 1: NEW GpuEventLoopState
kernel gpu_event_loop(input_queue: InputQueue, state: GpuEventLoopState, windows, params):
    tid = thread_position_in_grid
    if tid != 0: return  # Only one thread runs the event loop

    loop forever:
        # 1. Check for new input (head/tail are in InputQueue, NOT state)
        tail = atomic_load(input_queue.tail)  # CPU writes via InputHandler.push_*()
        head = atomic_load(input_queue.head)  # GPU reads

        if head == tail:
            # No input available
            if atomic_load(state.frame_dirty):
                # Need to render
                atomic_store(state.next_dispatch, DISPATCH_RENDER)
                atomic_store(state.frame_dirty, 0)
                return  # Exit kernel to let render happen

            # Spin wait (GPU thread yields to others)
            continue

        # 2. Dequeue event
        event = input_queue[head % QUEUE_SIZE]
        atomic_add(state.queue_head, 1)

        # 3. Update mouse position for all mouse events
        if event.type in [MOUSE_DOWN, MOUSE_UP, MOUSE_MOVE]:
            state.mouse_x = event.x
            state.mouse_y = event.y

        # 4. Route event by type
        switch event.type:

            case MOUSE_DOWN:
                # Store mouse state
                state.mouse_buttons |= (1 << event.button)

                # Request hit test
                atomic_store(state.next_dispatch, DISPATCH_HIT_TEST)
                return  # Exit to run hit test kernel

            case MOUSE_UP:
                state.mouse_buttons &= ~(1 << event.button)

                # End any drag/resize
                if state.drag_window != INVALID_WINDOW:
                    state.drag_window = INVALID_WINDOW
                    atomic_store(state.frame_dirty, 1)

                if state.resize_window != INVALID_WINDOW:
                    state.resize_window = INVALID_WINDOW
                    atomic_store(state.frame_dirty, 1)

            case MOUSE_MOVE:
                if state.drag_window != INVALID_WINDOW:
                    # Continue dragging - dispatch window move
                    atomic_store(state.next_dispatch, DISPATCH_WINDOW_MOVE)
                    state.dispatch_param = state.drag_window
                    return

                elif state.resize_window != INVALID_WINDOW:
                    # Continue resizing - dispatch window resize
                    atomic_store(state.next_dispatch, DISPATCH_WINDOW_RESIZE)
                    state.dispatch_param = state.resize_window
                    return

                else:
                    # Just hovering - update hover state
                    atomic_store(state.next_dispatch, DISPATCH_HOVER_TEST)
                    return

            case KEY_DOWN:
                if state.focused_window != INVALID_WINDOW:
                    # Dispatch to focused app
                    atomic_store(state.next_dispatch, DISPATCH_APP_INPUT)
                    state.dispatch_param = state.focused_window
                    return
                else:
                    # Global hotkey handling
                    handle_global_hotkey(event)

            case KEY_UP:
                if state.focused_window != INVALID_WINDOW:
                    atomic_store(state.next_dispatch, DISPATCH_APP_INPUT)
                    state.dispatch_param = state.focused_window
                    return

            case CHAR:
                if state.focused_window != INVALID_WINDOW:
                    atomic_store(state.next_dispatch, DISPATCH_APP_INPUT)
                    state.dispatch_param = state.focused_window
                    return

        # Mark frame dirty for any event that might change display
        atomic_store(state.frame_dirty, 1)
```

### Hit Test Kernel (Parallel)

```
# Parallel hit test - one thread per window
kernel hit_test_parallel(windows, state, hit_result, window_count):
    tid = thread_position_in_grid
    if tid >= window_count: return

    window = windows[tid]

    # Skip invisible windows
    if not (window.flags & VISIBLE): return

    x = state.mouse_x
    y = state.mouse_y

    # Bounds check
    if x < window.x or x > window.x + window.width: return
    if y < window.y or y > window.y + window.height: return

    # Determine hit region
    local_x = x - window.x
    local_y = y - window.y

    region = REGION_CONTENT
    resize_edge = EDGE_NONE

    # Title bar check
    if local_y < TITLE_BAR_HEIGHT:
        region = REGION_TITLE

        # Check buttons (right side of title bar)
        button_x = window.width - BUTTON_SIZE - BUTTON_MARGIN
        if local_x >= button_x:
            region = REGION_CLOSE
        elif local_x >= button_x - BUTTON_SIZE - BUTTON_SPACING:
            region = REGION_MAXIMIZE
        elif local_x >= button_x - 2*(BUTTON_SIZE + BUTTON_SPACING):
            region = REGION_MINIMIZE

    # Resize edge check (takes precedence if near edge)
    EDGE_SIZE = 8.0
    if local_x < EDGE_SIZE:
        resize_edge |= EDGE_LEFT
    if local_x > window.width - EDGE_SIZE:
        resize_edge |= EDGE_RIGHT
    if local_y < EDGE_SIZE:
        resize_edge |= EDGE_TOP
    if local_y > window.height - EDGE_SIZE:
        resize_edge |= EDGE_BOTTOM

    if resize_edge != EDGE_NONE:
        region = REGION_RESIZE

    # Atomic max to find topmost window (highest z_order wins)
    # Encode: z_order in high 32 bits, packed data in low 32 bits
    encoded = (uint64(window.z_order) << 32) | pack(tid, region, resize_edge)
    atomic_max(hit_result.encoded, encoded)
```

### Hit Test Result Handler

```
# Process hit test result - runs after hit_test_parallel
kernel handle_hit_result(state, hit_result, windows):
    tid = thread_position_in_grid
    if tid != 0: return

    # Decode result
    encoded = hit_result.encoded
    if encoded == 0:
        # No window hit - click on desktop
        state.focused_window = INVALID_WINDOW
        atomic_store(state.frame_dirty, 1)
        return

    window_index = (encoded >> 16) & 0xFFFF
    region = (encoded >> 8) & 0xFF
    resize_edge = encoded & 0xFF

    window = windows[window_index]

    switch region:
        case REGION_TITLE:
            # Start window drag
            state.drag_window = window_index
            state.drag_start_x = state.mouse_x
            state.drag_start_y = state.mouse_y
            state.window_start_x = window.x
            state.window_start_y = window.y

            # Focus and bring to front
            state.focused_window = window_index
            atomic_store(state.next_dispatch, DISPATCH_BRING_TO_FRONT)
            state.dispatch_param = window_index

        case REGION_CLOSE:
            # Close window
            windows[window_index].flags &= ~VISIBLE
            atomic_store(state.frame_dirty, 1)

        case REGION_MINIMIZE:
            windows[window_index].flags |= MINIMIZED
            atomic_store(state.frame_dirty, 1)

        case REGION_MAXIMIZE:
            if windows[window_index].flags & MAXIMIZED:
                # Restore
                windows[window_index].flags &= ~MAXIMIZED
                # Restore saved bounds...
            else:
                # Maximize
                windows[window_index].flags |= MAXIMIZED
                # Save and set to screen bounds...
            atomic_store(state.frame_dirty, 1)

        case REGION_RESIZE:
            # Start resize
            state.resize_window = window_index
            state.resize_edge = resize_edge
            state.resize_start_x = state.mouse_x
            state.resize_start_y = state.mouse_y
            state.window_start_w = window.width
            state.window_start_h = window.height
            state.window_start_x = window.x
            state.window_start_y = window.y
            state.focused_window = window_index

        case REGION_CONTENT:
            # Focus window and dispatch to app
            state.focused_window = window_index
            atomic_store(state.next_dispatch, DISPATCH_APP_INPUT)
            state.dispatch_param = window_index

    atomic_store(state.frame_dirty, 1)

    # Reset hit result for next time
    hit_result.encoded = 0
```

### Window Move Kernel

```
kernel window_move(state, windows):
    tid = thread_position_in_grid
    if tid != 0: return

    window_idx = state.drag_window
    if window_idx == INVALID_WINDOW: return

    # Calculate delta from drag start
    dx = state.mouse_x - state.drag_start_x
    dy = state.mouse_y - state.drag_start_y

    # Apply to window
    windows[window_idx].x = state.window_start_x + dx
    windows[window_idx].y = state.window_start_y + dy

    # Clamp to screen bounds
    windows[window_idx].x = clamp(windows[window_idx].x, 0, screen_width - 50)
    windows[window_idx].y = clamp(windows[window_idx].y, 0, screen_height - 50)

    atomic_store(state.frame_dirty, 1)
```

### Window Resize Kernel

```
kernel window_resize(state, windows):
    tid = thread_position_in_grid
    if tid != 0: return

    window_idx = state.resize_window
    if window_idx == INVALID_WINDOW: return

    edge = state.resize_edge
    dx = state.mouse_x - state.resize_start_x
    dy = state.mouse_y - state.resize_start_y

    window = &windows[window_idx]

    MIN_WIDTH = 100.0
    MIN_HEIGHT = 50.0

    if edge & EDGE_LEFT:
        new_x = state.window_start_x + dx
        new_w = state.window_start_w - dx
        if new_w >= MIN_WIDTH:
            window.x = new_x
            window.width = new_w

    if edge & EDGE_RIGHT:
        new_w = state.window_start_w + dx
        if new_w >= MIN_WIDTH:
            window.width = new_w

    if edge & EDGE_TOP:
        new_y = state.window_start_y + dy
        new_h = state.window_start_h - dy
        if new_h >= MIN_HEIGHT:
            window.y = new_y
            window.height = new_h

    if edge & EDGE_BOTTOM:
        new_h = state.window_start_h + dy
        if new_h >= MIN_HEIGHT:
            window.height = new_h

    atomic_store(state.frame_dirty, 1)
```

### CPU Side (Minimal) - Uses Existing GpuRuntime

```
# CPU: Initialize event loop (extends existing GpuRuntime)
function GpuRuntime.init_event_loop():
    # REUSE: self.input is existing InputHandler with queue_buffer
    # REUSE: self.command_queue is existing command queue
    # REUSE: self.shared_event is existing async signaling

    # NEW: Only allocate event loop state (InputQueue already exists!)
    state_buffer = gpu_alloc(sizeof(GpuEventLoopState))

    # Initialize NEW state (queue head/tail are in existing InputQueue)
    state = state_buffer.contents()
    state.drag_window = INVALID_WINDOW
    state.resize_window = INVALID_WINDOW
    state.focused_window = INVALID_WINDOW
    state.frame_dirty = 0

    return EventLoopHandle { state_buffer, ... }

# CPU: Start event loop (called ONCE)
function GpuRuntime.start_event_loop(handle, windows_buffer):
    # Use EXISTING command_queue from GpuRuntime
    cmd = self.command_queue.new_command_buffer()

    encoder = cmd.new_compute_encoder()
    encoder.set_pipeline(event_loop_pipeline)

    # REUSE: Existing InputHandler.buffer() as input queue!
    encoder.set_buffer(0, self.input.buffer())  # EXISTING

    # NEW: Event loop state
    encoder.set_buffer(1, handle.state_buffer)  # NEW

    encoder.set_buffer(2, windows_buffer)
    encoder.dispatch_threads(1, 1, 1)
    encoder.end()

    # Use EXISTING shared_event for async
    cmd.encode_signal_event(self.shared_event, self.next_signal_value())

    cmd.add_completed_handler(on_event_loop_complete)
    cmd.commit()

# CPU: Handle completion
function on_event_loop_complete(handle):
    if not handle.running: return

    dispatch = handle.state_buffer.contents().next_dispatch

    # Use EXISTING command_queue
    cmd = self.command_queue.new_command_buffer()

    switch dispatch:
        case DISPATCH_NONE:
            pass
        case DISPATCH_HIT_TEST:
            encode_hit_test(cmd)
            encode_hit_result_handler(cmd)
        case DISPATCH_WINDOW_MOVE:
            encode_window_move(cmd)
        case DISPATCH_RENDER:
            encode_render(cmd)

    encode_event_loop(cmd)
    cmd.add_completed_handler(on_event_loop_complete)
    cmd.commit()

# CPU: Push input using EXISTING InputHandler methods
# NO NEW push_input() NEEDED - use existing methods:
#
#   runtime.input.push_mouse_move(x, y, dx, dy);
#   runtime.input.push_mouse_button(button, pressed, x, y);
#   runtime.input.push_key(keycode, pressed, modifiers);
#   runtime.input.push_scroll(x, y, dx, dy);
#
# These already write to InputHandler.queue_buffer which the GPU reads!
```

## Test Plan

### Unit Tests (Using Existing Infrastructure)

```rust
// tests/test_issue_149_gpu_event_loop.rs

use gpu_native_os::gpu_os::app::GpuRuntime;
use gpu_native_os::gpu_os::event_loop::{GpuEventLoopState, EventLoopHandle, INVALID_WINDOW};
use gpu_native_os::gpu_os::memory::InputEvent;  // REUSE existing type

#[test]
fn test_gpu_event_loop_processes_click() {
    let device = Device::system_default().unwrap();

    // Use EXISTING GpuRuntime
    let mut runtime = GpuRuntime::new(device);

    // Initialize event loop (extends GpuRuntime)
    let handle = runtime.init_event_loop();

    // Create windows buffer
    let windows_buffer = create_test_windows(&runtime.device);

    // Start the loop
    runtime.start_event_loop(&handle, &windows_buffer);

    // Push a click using EXISTING InputHandler methods
    runtime.input.push_mouse_button(0, true, 150.0, 125.0);

    // Wait for GPU to process
    std::thread::sleep(Duration::from_millis(100));

    // Check that window was focused
    let state = handle.read_state();
    assert_eq!(state.focused_window, 0);

    handle.stop();
}

#[test]
fn test_uses_existing_input_handler() {
    let device = Device::system_default().unwrap();
    let runtime = GpuRuntime::new(device);
    let handle = runtime.init_event_loop();

    // Verify we're using the SAME input buffer
    // Push event via existing InputHandler
    runtime.input.push_mouse_move(100.0, 100.0, 0.0, 0.0);

    // The GPU event loop reads from runtime.input.buffer()
    // NOT from a separate input_queue_buffer
    assert_eq!(runtime.input.pending_count(), 1);
}

#[test]
fn test_gpu_event_loop_no_cpu_dispatch_during_drag() {
    let device = Device::system_default().unwrap();
    let runtime = GpuRuntime::new(device);
    let handle = runtime.init_event_loop();
    let windows_buffer = create_test_windows(&runtime.device);

    runtime.start_event_loop(&handle, &windows_buffer);

    // Simulate drag using EXISTING InputHandler
    runtime.input.push_mouse_button(0, true, 150.0, 110.0); // Title bar

    for i in 0..100 {
        runtime.input.push_mouse_move(150.0 + i as f32, 110.0, 1.0, 0.0);
    }

    runtime.input.push_mouse_button(0, false, 250.0, 110.0);

    std::thread::sleep(Duration::from_millis(500));
    handle.stop();
}

#[test]
fn test_integration_with_existing_shared_event() {
    let device = Device::system_default().unwrap();
    let runtime = GpuRuntime::new(device);

    // Verify we use EXISTING shared_event from GpuRuntime
    let signal_before = runtime.current_signal_value();

    let handle = runtime.init_event_loop();
    let windows_buffer = create_test_windows(&runtime.device);
    runtime.start_event_loop(&handle, &windows_buffer);

    // Wait for first dispatch
    std::thread::sleep(Duration::from_millis(50));

    // Signal value should have increased (proves we're using existing async infra)
    let signal_after = runtime.current_signal_value();
    assert!(signal_after > signal_before);

    handle.stop();
}
```

### Benchmark

```rust
#[test]
fn benchmark_gpu_vs_cpu_event_loop() {
    let device = Device::system_default().unwrap();

    // GPU event loop
    let gpu_loop = GpuEventLoop::new(&device);
    gpu_loop.start();

    // Measure latency from event push to window update
    let start = Instant::now();
    for _ in 0..10000 {
        gpu_loop.push_input(InputEvent::mouse_move(100.0, 100.0));
    }
    gpu_loop.wait_idle();
    let gpu_time = start.elapsed();

    // CPU event loop (old way)
    let cpu_loop = CpuEventLoop::new(&device);
    let start = Instant::now();
    for _ in 0..10000 {
        cpu_loop.process_event(InputEvent::mouse_move(100.0, 100.0));
    }
    let cpu_time = start.elapsed();

    println!("GPU: {:?}, CPU: {:?}", gpu_time, cpu_time);
    assert!(gpu_time < cpu_time / 2, "GPU should be at least 2x faster");
}
```

## Success Metrics

1. **CPU calls during steady state:** ZERO (only input push)
2. **Event-to-render latency:** <1ms (no CPU round-trip)
3. **Events per second:** >100K (limited only by GPU)
4. **CPU utilization:** <1% during active interaction

## Dependencies

- Issue #133: Persistent Kernels (foundation)
- Issue #142: Async Frame Callbacks (for initial dispatch)

## Files to Modify/Create

### Modify Existing

1. **`src/gpu_os/app.rs`** - Add event loop methods to `GpuRuntime`:
   - `init_event_loop()` - Returns `EventLoopHandle`
   - `start_event_loop()` - Starts GPU event loop
   - `continue_event_loop()` - Completion handler logic

2. **`src/gpu_os/mod.rs`** - Export new event_loop module

### Create New

3. **`src/gpu_os/event_loop.rs`** - New module containing:
   - `GpuEventLoopState` struct (interaction state)
   - `EventLoopHandle` struct (returned by init)
   - `HitTestResult` struct
   - Shader source string

4. **`src/gpu_os/event_loop.metal`** - GPU kernels:
   - `gpu_event_loop` - Persistent event router
   - `hit_test_parallel` - Parallel hit testing
   - `handle_hit_result` - Result handler
   - `window_move`, `window_resize` - Window ops

5. **`tests/test_issue_149_gpu_event_loop.rs`** - Integration tests

### NO Changes Needed

- `src/gpu_os/memory.rs` - Already has `InputEvent`, `InputQueue`
- `src/gpu_os/input.rs` - Already has `InputHandler` with push methods

## This Changes Everything

This is not just an optimization - it's the **architectural shift** the project is about. Once implemented:

- CPU becomes a peripheral (handles I/O only)
- GPU is the computer (handles all logic)
- Latency drops dramatically (no round-trips)
- Throughput scales with GPU cores

**The GPU is the computer. The CPU is for I/O.**
