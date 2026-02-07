// Issue #149: GPU-Driven Event Dispatch
//
// This module implements a GPU-resident event loop that eliminates CPU
// from steady-state event processing. The GPU handles:
// - Event routing from InputQueue
// - Hit testing (parallel)
// - Window drag/resize operations
// - Focus management
// - Render dispatch
//
// CPU's only job: push input events to the existing InputQueue via InputHandler.

use metal::*;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::Arc;

/// Invalid window index sentinel
pub const INVALID_WINDOW: u32 = 0xFFFFFFFF;

/// Dispatch targets for GPU event loop
pub mod dispatch {
    pub const NONE: u32 = 0;
    pub const HIT_TEST: u32 = 1;
    pub const WINDOW_MOVE: u32 = 2;
    pub const WINDOW_RESIZE: u32 = 3;
    pub const WINDOW_FOCUS: u32 = 4;
    pub const APP_INPUT: u32 = 5;
    pub const MENU_CLICK: u32 = 6;
    pub const DOCK_CLICK: u32 = 7;
    pub const RENDER: u32 = 8;
    pub const BRING_TO_FRONT: u32 = 9;
    pub const HOVER_TEST: u32 = 10;
    pub const WINDOW_CLOSE: u32 = 11;
}

/// Hit regions for windows
pub mod region {
    pub const NONE: u32 = 0;
    pub const TITLE: u32 = 1;
    pub const CONTENT: u32 = 2;
    pub const CLOSE: u32 = 3;
    pub const MINIMIZE: u32 = 4;
    pub const MAXIMIZE: u32 = 5;
    pub const RESIZE: u32 = 6;
}

/// Resize edge flags (can be combined)
pub mod edge {
    pub const NONE: u32 = 0;
    pub const LEFT: u32 = 1;
    pub const RIGHT: u32 = 2;
    pub const TOP: u32 = 4;
    pub const BOTTOM: u32 = 8;
}

/// GPU-resident event loop state
///
/// This is NEW state that doesn't exist in InputQueue.
/// Contains interaction state machine for drag/resize/focus.
///
/// Note: Queue head/tail are in the existing InputQueue (memory.rs),
/// NOT duplicated here.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuEventLoopState {
    // Drag interaction state
    pub drag_window: u32,
    pub drag_start_x: f32,
    pub drag_start_y: f32,
    pub window_start_x: f32,
    pub window_start_y: f32,

    // Resize interaction state
    pub resize_window: u32,
    pub resize_edge: u32,
    pub resize_start_x: f32,
    pub resize_start_y: f32,
    pub window_start_w: f32,
    pub window_start_h: f32,

    // Focus state
    pub focused_window: u32,
    pub hovered_window: u32,

    // Dispatch control (GPU sets, CPU reads)
    pub next_dispatch: u32,
    pub dispatch_param: u32,

    // Frame state
    pub frame_dirty: u32,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_buttons: u32,

    // Padding to 16-byte alignment (96 bytes total)
    pub _padding: [u32; 5],
}

impl GpuEventLoopState {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new() -> Self {
        Self {
            drag_window: INVALID_WINDOW,
            resize_window: INVALID_WINDOW,
            focused_window: INVALID_WINDOW,
            hovered_window: INVALID_WINDOW,
            next_dispatch: dispatch::NONE,
            dispatch_param: 0,
            frame_dirty: 0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: 0,
            drag_start_x: 0.0,
            drag_start_y: 0.0,
            window_start_x: 0.0,
            window_start_y: 0.0,
            resize_edge: edge::NONE,
            resize_start_x: 0.0,
            resize_start_y: 0.0,
            window_start_w: 0.0,
            window_start_h: 0.0,
            _padding: [0; 5],
        }
    }
}

/// Result of parallel hit testing
/// Uses two 32-bit atomics since Metal doesn't support 64-bit atomics
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HitTestResult {
    /// z_order of the hit window (higher wins)
    pub z_order: u32,
    /// Packed data: window_index(16) | region(8) | resize_edge(8)
    pub packed_data: u32,
}

impl HitTestResult {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Decode window index from result
    pub fn window_index(&self) -> u32 {
        (self.packed_data >> 16) & 0xFFFF
    }

    /// Decode hit region from result
    pub fn region(&self) -> u32 {
        (self.packed_data >> 8) & 0xFF
    }

    /// Decode resize edge from result
    pub fn resize_edge(&self) -> u32 {
        self.packed_data & 0xFF
    }

    /// Check if anything was hit
    pub fn is_hit(&self) -> bool {
        self.z_order != 0 || self.packed_data != 0
    }
}

/// Window structure for GPU event loop
/// Matches the layout expected by GPU kernels
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuWindow {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub z_order: u32,
    pub flags: u32,
    pub _padding: [u32; 2],
}

impl GpuWindow {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Window flags
pub mod window_flags {
    pub const VISIBLE: u32 = 1;
    pub const MINIMIZED: u32 = 2;
    pub const MAXIMIZED: u32 = 4;
    pub const FOCUSED: u32 = 8;
}

/// Handle returned by init_event_loop()
///
/// Holds the event loop state buffer and running flag.
/// Does NOT duplicate GpuRuntime's command_queue or input buffer.
pub struct EventLoopHandle {
    /// NEW: GPU-resident event loop state
    pub state_buffer: Buffer,
    /// NEW: Hit test result buffer
    pub hit_result_buffer: Buffer,
    /// Compute pipeline for event loop kernel
    pub event_loop_pipeline: ComputePipelineState,
    /// Compute pipeline for hit test kernel
    pub hit_test_pipeline: ComputePipelineState,
    /// Compute pipeline for hit result handler
    pub hit_result_handler_pipeline: ComputePipelineState,
    /// Compute pipeline for window move
    pub window_move_pipeline: ComputePipelineState,
    /// Compute pipeline for window resize
    pub window_resize_pipeline: ComputePipelineState,
    /// Running flag
    pub running: Arc<AtomicBool>,
}

impl EventLoopHandle {
    /// Stop the event loop
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Read the current event loop state
    /// Issue #256: Add memory fence to ensure we see GPU writes
    pub fn read_state(&self) -> GpuEventLoopState {
        // Acquire fence ensures we see all GPU writes before this read
        fence(Ordering::Acquire);
        unsafe { std::ptr::read_volatile(self.state_buffer.contents() as *const GpuEventLoopState) }
    }

    /// Read hit test result
    /// Issue #256: Add memory fence to ensure we see GPU writes
    pub fn read_hit_result(&self) -> HitTestResult {
        fence(Ordering::Acquire);
        unsafe { std::ptr::read_volatile(self.hit_result_buffer.contents() as *const HitTestResult) }
    }
}

impl Clone for EventLoopHandle {
    fn clone(&self) -> Self {
        Self {
            state_buffer: self.state_buffer.clone(),
            hit_result_buffer: self.hit_result_buffer.clone(),
            event_loop_pipeline: self.event_loop_pipeline.clone(),
            hit_test_pipeline: self.hit_test_pipeline.clone(),
            hit_result_handler_pipeline: self.hit_result_handler_pipeline.clone(),
            window_move_pipeline: self.window_move_pipeline.clone(),
            window_resize_pipeline: self.window_resize_pipeline.clone(),
            running: self.running.clone(),
        }
    }
}

/// GPU shader source for event loop kernels
pub const EVENT_LOOP_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

#define INVALID_WINDOW 0xFFFFFFFF
#define QUEUE_SIZE 256
#define TITLE_BAR_HEIGHT 30.0
#define EDGE_SIZE 8.0
#define BUTTON_SIZE 14.0
#define BUTTON_MARGIN 8.0
#define BUTTON_SPACING 8.0

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
#define DISPATCH_BRING_TO_FRONT 9
#define DISPATCH_HOVER_TEST     10
#define DISPATCH_WINDOW_CLOSE   11

// Hit regions
#define REGION_NONE     0
#define REGION_TITLE    1
#define REGION_CONTENT  2
#define REGION_CLOSE    3
#define REGION_MINIMIZE 4
#define REGION_MAXIMIZE 5
#define REGION_RESIZE   6

// Edge flags
#define EDGE_NONE   0
#define EDGE_LEFT   1
#define EDGE_RIGHT  2
#define EDGE_TOP    4
#define EDGE_BOTTOM 8

// Window flags
#define WINDOW_VISIBLE   1
#define WINDOW_MINIMIZED 2
#define WINDOW_MAXIMIZED 4
#define WINDOW_FOCUSED   8

// Input event types (from memory.rs InputEventType)
#define INPUT_NONE        0
#define INPUT_MOUSE_MOVE  1
#define INPUT_MOUSE_DOWN  2
#define INPUT_MOUSE_UP    3
#define INPUT_MOUSE_SCROLL 4
#define INPUT_KEY_DOWN    5
#define INPUT_KEY_UP      6
#define INPUT_KEY_REPEAT  7

// ============================================================================
// Structures (must match Rust definitions)
// ============================================================================

// EXISTING: InputEvent from memory.rs (28 bytes)
// Use packed_float2 to match Rust's [f32; 2] alignment (4-byte, not 8-byte)
struct InputEvent {
    ushort event_type;
    ushort keycode;
    packed_float2 position;  // packed_float2 has 4-byte alignment like Rust
    packed_float2 delta;
    uint modifiers;
    uint timestamp;
};

// EXISTING: InputQueue from memory.rs
struct InputQueue {
    atomic_uint head;
    atomic_uint tail;
    uint _padding[2];
    InputEvent events[256];
};

// NEW: GPU Event Loop State
struct GpuEventLoopState {
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

    atomic_uint next_dispatch;
    uint dispatch_param;

    atomic_uint frame_dirty;
    float mouse_x;
    float mouse_y;
    uint mouse_buttons;

    uint _padding[5];  // Padding to 96 bytes (16-byte aligned)
};

// NEW: Hit test result (two 32-bit atomics since Metal doesn't support 64-bit)
struct HitTestResult {
    atomic_uint z_order;      // Higher z_order wins
    atomic_uint packed_data;  // window_index(16) | region(8) | resize_edge(8)
};

// Window structure
struct Window {
    float x;
    float y;
    float width;
    float height;
    uint z_order;
    uint flags;
    uint _padding[2];
};

// ============================================================================
// GPU Event Loop Kernel (Persistent)
// ============================================================================

kernel void gpu_event_loop(
    device InputQueue* input_queue [[buffer(0)]],
    device GpuEventLoopState* state [[buffer(1)]],
    device Window* windows [[buffer(2)]],
    constant uint& window_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Check for new input
    // In a ring buffer: head = write position, tail = read position
    // CPU writes at head, GPU reads from tail
    uint head = atomic_load_explicit(&input_queue->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&input_queue->tail, memory_order_relaxed);

    if (head == tail) {
        // No input - check if we need to render
        if (atomic_load_explicit(&state->frame_dirty, memory_order_relaxed)) {
            atomic_store_explicit(&state->next_dispatch, DISPATCH_RENDER, memory_order_relaxed);
            atomic_store_explicit(&state->frame_dirty, 0, memory_order_relaxed);
        }
        return;
    }

    // Process next event from tail (consumer reads from tail)
    InputEvent event = input_queue->events[tail % QUEUE_SIZE];
    // Advance tail (consumer increments tail after reading)
    atomic_fetch_add_explicit(&input_queue->tail, 1, memory_order_relaxed);

    // Update mouse position for mouse events
    if (event.event_type == INPUT_MOUSE_MOVE ||
        event.event_type == INPUT_MOUSE_DOWN ||
        event.event_type == INPUT_MOUSE_UP) {
        state->mouse_x = event.position.x;
        state->mouse_y = event.position.y;
    }

    // Route event by type
    switch (event.event_type) {
        case INPUT_MOUSE_DOWN:
            state->mouse_buttons |= (1 << event.keycode);
            atomic_store_explicit(&state->next_dispatch, DISPATCH_HIT_TEST, memory_order_relaxed);
            break;

        case INPUT_MOUSE_MOVE:
            if (state->drag_window != INVALID_WINDOW) {
                atomic_store_explicit(&state->next_dispatch, DISPATCH_WINDOW_MOVE, memory_order_relaxed);
                state->dispatch_param = state->drag_window;
            } else if (state->resize_window != INVALID_WINDOW) {
                atomic_store_explicit(&state->next_dispatch, DISPATCH_WINDOW_RESIZE, memory_order_relaxed);
                state->dispatch_param = state->resize_window;
            } else {
                atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
            }
            break;

        case INPUT_MOUSE_UP:
            state->mouse_buttons &= ~(1 << event.keycode);
            if (state->drag_window != INVALID_WINDOW) {
                state->drag_window = INVALID_WINDOW;
                atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
            }
            if (state->resize_window != INVALID_WINDOW) {
                state->resize_window = INVALID_WINDOW;
                atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
            }
            break;

        case INPUT_KEY_DOWN:
        case INPUT_KEY_UP:
        case INPUT_KEY_REPEAT:
            if (state->focused_window != INVALID_WINDOW) {
                atomic_store_explicit(&state->next_dispatch, DISPATCH_APP_INPUT, memory_order_relaxed);
                state->dispatch_param = state->focused_window;
            }
            break;

        default:
            break;
    }
}

// ============================================================================
// Hit Test Kernel (Parallel - one thread per window)
// ============================================================================

// Sequential hit test - simpler and race-free
// For typical desktop window counts (<20), this is fast enough
kernel void hit_test_parallel(
    device GpuEventLoopState* state [[buffer(0)]],
    device Window* windows [[buffer(1)]],
    device HitTestResult* result [[buffer(2)]],
    constant uint& window_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Only run on thread 0 (sequential for correctness)
    if (tid != 0) return;

    float x = state->mouse_x;
    float y = state->mouse_y;

    uint best_z = 0;
    uint best_packed = 0;

    // Check all windows, find topmost hit
    for (uint i = 0; i < window_count; i++) {
        Window window = windows[i];

        // Skip invisible windows
        if (!(window.flags & WINDOW_VISIBLE)) continue;

        // Bounds check
        if (x < window.x || x > window.x + window.width) continue;
        if (y < window.y || y > window.y + window.height) continue;

        // This window is hit - check if it's topmost
        // Use > not >= to ensure first window with same z_order wins (prevents all-same-z_order bug)
        if (window.z_order > best_z || (window.z_order == best_z && best_packed == 0)) {
            best_z = window.z_order;

            // Determine hit region
            float local_x = x - window.x;
            float local_y = y - window.y;

            uint region = REGION_CONTENT;
            uint resize_edge = EDGE_NONE;

            // Title bar check
            if (local_y < TITLE_BAR_HEIGHT) {
                region = REGION_TITLE;

                // Check buttons (right side of title bar)
                float button_x = window.width - BUTTON_SIZE - BUTTON_MARGIN;
                if (local_x >= button_x) {
                    region = REGION_CLOSE;
                } else if (local_x >= button_x - BUTTON_SIZE - BUTTON_SPACING) {
                    region = REGION_MAXIMIZE;
                } else if (local_x >= button_x - 2 * (BUTTON_SIZE + BUTTON_SPACING)) {
                    region = REGION_MINIMIZE;
                }
            }

            // Resize edge check
            if (local_x < EDGE_SIZE) resize_edge |= EDGE_LEFT;
            if (local_x > window.width - EDGE_SIZE) resize_edge |= EDGE_RIGHT;
            if (local_y < EDGE_SIZE) resize_edge |= EDGE_TOP;
            if (local_y > window.height - EDGE_SIZE) resize_edge |= EDGE_BOTTOM;

            if (resize_edge != EDGE_NONE) {
                region = REGION_RESIZE;
            }

            // Pack: window_index(16) | region(8) | resize_edge(8)
            best_packed = (i << 16) | (region << 8) | resize_edge;
        }
    }

    // Store result
    atomic_store_explicit(&result->z_order, best_z, memory_order_relaxed);
    atomic_store_explicit(&result->packed_data, best_packed, memory_order_relaxed);
}

// ============================================================================
// Hit Test Result Handler
// ============================================================================

kernel void handle_hit_result(
    device GpuEventLoopState* state [[buffer(0)]],
    device HitTestResult* result [[buffer(1)]],
    device Window* windows [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint z_order = atomic_load_explicit(&result->z_order, memory_order_relaxed);
    uint packed = atomic_load_explicit(&result->packed_data, memory_order_relaxed);

    // Reset for next time
    atomic_store_explicit(&result->z_order, 0, memory_order_relaxed);
    atomic_store_explicit(&result->packed_data, 0, memory_order_relaxed);

    if (z_order == 0 && packed == 0) {
        // No window hit - click on desktop
        state->focused_window = INVALID_WINDOW;
        atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
        return;
    }

    uint window_index = (packed >> 16) & 0xFFFF;
    uint region = (packed >> 8) & 0xFF;
    uint resize_edge = packed & 0xFF;

    Window window = windows[window_index];

    switch (region) {
        case REGION_TITLE:
            // Start window drag
            state->drag_window = window_index;
            state->drag_start_x = state->mouse_x;
            state->drag_start_y = state->mouse_y;
            state->window_start_x = window.x;
            state->window_start_y = window.y;
            state->focused_window = window_index;
            atomic_store_explicit(&state->next_dispatch, DISPATCH_BRING_TO_FRONT, memory_order_relaxed);
            state->dispatch_param = window_index;
            break;

        case REGION_CLOSE:
            // Dispatch to CPU to properly close window and clean up app
            atomic_store_explicit(&state->next_dispatch, DISPATCH_WINDOW_CLOSE, memory_order_relaxed);
            state->dispatch_param = window_index;
            break;

        case REGION_MINIMIZE:
            windows[window_index].flags |= WINDOW_MINIMIZED;
            atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
            break;

        case REGION_MAXIMIZE:
            if (windows[window_index].flags & WINDOW_MAXIMIZED) {
                windows[window_index].flags &= ~WINDOW_MAXIMIZED;
            } else {
                windows[window_index].flags |= WINDOW_MAXIMIZED;
            }
            atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
            break;

        case REGION_RESIZE:
            state->resize_window = window_index;
            state->resize_edge = resize_edge;
            state->resize_start_x = state->mouse_x;
            state->resize_start_y = state->mouse_y;
            state->window_start_w = window.width;
            state->window_start_h = window.height;
            state->window_start_x = window.x;
            state->window_start_y = window.y;
            state->focused_window = window_index;
            break;

        case REGION_CONTENT:
            state->focused_window = window_index;
            atomic_store_explicit(&state->next_dispatch, DISPATCH_APP_INPUT, memory_order_relaxed);
            state->dispatch_param = window_index;
            break;

        default:
            break;
    }

    atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
}

// ============================================================================
// Window Move Kernel
// ============================================================================

kernel void window_move(
    device GpuEventLoopState* state [[buffer(0)]],
    device Window* windows [[buffer(1)]],
    constant float2& screen_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint window_idx = state->drag_window;
    if (window_idx == INVALID_WINDOW) return;

    float dx = state->mouse_x - state->drag_start_x;
    float dy = state->mouse_y - state->drag_start_y;

    windows[window_idx].x = clamp(state->window_start_x + dx, 0.0f, screen_size.x - 50.0f);
    windows[window_idx].y = clamp(state->window_start_y + dy, 0.0f, screen_size.y - 50.0f);

    atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
}

// ============================================================================
// Window Resize Kernel
// ============================================================================

kernel void window_resize(
    device GpuEventLoopState* state [[buffer(0)]],
    device Window* windows [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint window_idx = state->resize_window;
    if (window_idx == INVALID_WINDOW) return;

    uint edge = state->resize_edge;
    float dx = state->mouse_x - state->resize_start_x;
    float dy = state->mouse_y - state->resize_start_y;

    float MIN_WIDTH = 100.0;
    float MIN_HEIGHT = 50.0;

    if (edge & EDGE_LEFT) {
        float new_x = state->window_start_x + dx;
        float new_w = state->window_start_w - dx;
        if (new_w >= MIN_WIDTH) {
            windows[window_idx].x = new_x;
            windows[window_idx].width = new_w;
        }
    }

    if (edge & EDGE_RIGHT) {
        float new_w = state->window_start_w + dx;
        if (new_w >= MIN_WIDTH) {
            windows[window_idx].width = new_w;
        }
    }

    if (edge & EDGE_TOP) {
        float new_y = state->window_start_y + dy;
        float new_h = state->window_start_h - dy;
        if (new_h >= MIN_HEIGHT) {
            windows[window_idx].y = new_y;
            windows[window_idx].height = new_h;
        }
    }

    if (edge & EDGE_BOTTOM) {
        float new_h = state->window_start_h + dy;
        if (new_h >= MIN_HEIGHT) {
            windows[window_idx].height = new_h;
        }
    }

    atomic_store_explicit(&state->frame_dirty, 1, memory_order_relaxed);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_loop_state_size() {
        // Verify size for Metal alignment (must be multiple of 16)
        assert_eq!(GpuEventLoopState::SIZE, 96,
            "GpuEventLoopState size: {} (expected 96)", GpuEventLoopState::SIZE);
        assert!(GpuEventLoopState::SIZE % 16 == 0,
            "GpuEventLoopState must be 16-byte aligned");
    }

    #[test]
    fn test_hit_result_decode() {
        let result = HitTestResult {
            // z_order=10, window_index=5, region=TITLE(1), resize_edge=NONE(0)
            z_order: 10,
            packed_data: (5 << 16) | (1 << 8) | 0,
        };

        assert_eq!(result.window_index(), 5);
        assert_eq!(result.region(), region::TITLE);
        assert_eq!(result.resize_edge(), edge::NONE);
        assert!(result.is_hit());
    }

    #[test]
    fn test_event_loop_state_new() {
        let state = GpuEventLoopState::new();
        assert_eq!(state.drag_window, INVALID_WINDOW);
        assert_eq!(state.resize_window, INVALID_WINDOW);
        assert_eq!(state.focused_window, INVALID_WINDOW);
        assert_eq!(state.next_dispatch, dispatch::NONE);
    }
}
