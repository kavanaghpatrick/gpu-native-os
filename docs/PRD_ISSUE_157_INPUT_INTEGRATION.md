# PRD: GPU Input & Window Integration (Issue #157)

## Overview

Integrate input handling where each app tests ITSELF for input relevance - no central dispatcher.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| "Distribute input to apps" | Each app checks: "Is this input for me?" |
| Central hit-test loop | Each app tests its own bounds |
| Focus manager dispatches keys | Focused app reads from shared queue |
| Event routing | Parallel predicate on each app |

## The GPU Insight

There is no "distributor". Each app evaluates:

```metal
// WRONG: Central dispatcher
for each event:
    target = find_target_app(event)
    target.receive(event)

// RIGHT: Parallel self-evaluation
uint slot_id = thread_position;
for each event in global_queue:
    if (is_this_event_for_me(event, apps[slot_id])) {
        handle_event(event);
    }
```

## Design

### Input Event Structure

```metal
struct InputEvent {
    uint event_type;      // KEY_DOWN, KEY_UP, MOUSE_MOVE, MOUSE_CLICK
    uint key_or_button;   // Key code or mouse button
    float2 position;      // Cursor position
    uint modifiers;       // Shift, Ctrl, Alt, Cmd
    uint frame;           // Frame when event occurred
};

struct InputQueue {
    atomic_uint head;     // Consumer position
    atomic_uint tail;     // Producer position (CPU writes here)
    uint capacity;
    uint _pad;
    // Events follow
};
```

### Parallel Input Processing

Each app processes input in parallel:

```metal
kernel void gpu_process_input(
    device InputQueue* queue [[buffer(0)]],
    device InputEvent* events [[buffer(1)]],
    device GpuAppDescriptor* apps [[buffer(2)]],
    device GpuWindow* windows [[buffer(3)]],
    constant uint& max_slots [[buffer(4)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];
    if (!(app->flags & APP_FLAG_ACTIVE)) return;

    // Get my window
    GpuWindow window = windows[app->window_id];
    bool i_am_focused = (app->flags & APP_FLAG_FOCUS) != 0;

    // Read event queue (all apps read same queue)
    uint head = atomic_load_explicit(&queue->head, memory_order_acquire);
    uint tail = atomic_load_explicit(&queue->tail, memory_order_acquire);

    for (uint i = head; i < tail; i++) {
        InputEvent event = events[i % queue->capacity];

        bool this_is_for_me = false;

        if (event.event_type == EVENT_KEY_DOWN || event.event_type == EVENT_KEY_UP) {
            // Keyboard: only focused app
            this_is_for_me = i_am_focused;
        } else {
            // Mouse: app under cursor
            this_is_for_me = point_in_rect(event.position, window);
        }

        if (this_is_for_me) {
            // Add to my local queue
            uint my_tail = app->input_tail;
            app->input_events[my_tail % 8] = pack_event(event);
            app->input_tail = my_tail + 1;
            app->flags |= APP_FLAG_DIRTY;  // Wake me up
        }
    }
}

// Helper: O(1) bounds check
inline bool point_in_rect(float2 p, GpuWindow w) {
    return p.x >= w.x && p.x < w.x + w.width &&
           p.y >= w.y && p.y < w.y + w.height;
}
```

### Focus Management (Parallel)

Focus is just a flag - no "manager":

```metal
// When mouse clicks, each app checks if it should gain focus
kernel void gpu_handle_click_focus(
    device GpuAppDescriptor* apps [[buffer(0)]],
    device GpuWindow* windows [[buffer(1)]],
    constant float2& click_pos [[buffer(2)]],
    constant uint& max_slots [[buffer(3)]],
    device atomic_uint* topmost_slot [[buffer(4)]],
    device atomic_uint* topmost_z [[buffer(5)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];
    if (!(app->flags & APP_FLAG_ACTIVE)) return;
    if (!(app->flags & APP_FLAG_VISIBLE)) return;

    GpuWindow window = windows[app->window_id];

    // Am I under the click?
    if (!point_in_rect(click_pos, window)) return;

    // Parallel max to find topmost
    uint my_z = window.z_order;
    uint old_z = atomic_load_explicit(topmost_z, memory_order_relaxed);

    while (my_z > old_z) {
        if (atomic_compare_exchange_weak_explicit(
            topmost_z, &old_z, my_z,
            memory_order_relaxed, memory_order_relaxed
        )) {
            atomic_store_explicit(topmost_slot, slot_id, memory_order_relaxed);
            break;
        }
    }
}

// After finding topmost, update focus flags
kernel void gpu_apply_focus(
    device GpuAppDescriptor* apps [[buffer(0)]],
    constant uint& new_focus_slot [[buffer(1)]],
    constant uint& max_slots [[buffer(2)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    if (slot_id == new_focus_slot) {
        apps[slot_id].flags |= APP_FLAG_FOCUS;
    } else {
        apps[slot_id].flags &= ~APP_FLAG_FOCUS;
    }
}
```

### Window Z-Order with Depth Buffer

Instead of maintaining z-order list, use depth values:

```metal
// Each window has a depth value
struct GpuWindow {
    float x, y, width, height;
    float depth;        // 0.0 = back, 1.0 = front
    uint app_slot;
    uint flags;
    uint _pad;
};

// When focusing, just update depth - O(1)
inline void bring_to_front(device GpuWindow* window) {
    window->depth = 1.0;  // Front
}

// Periodically normalize depths (parallel)
kernel void normalize_depths(
    device GpuWindow* windows [[buffer(0)]],
    constant uint& window_count [[buffer(1)]],
    uint wid [[thread_position_in_grid]]
) {
    if (wid >= window_count) return;

    // Simple normalization: depth = depth * 0.99
    // Recently focused windows stay near 1.0
    // Old windows drift toward 0.0
    windows[wid].depth *= 0.99;
}
```

## Implementation

### Rust API

```rust
impl GpuAppSystem {
    /// Queue input event (called by CPU)
    pub fn queue_input(&mut self, event: InputEvent) {
        // Write to tail of GPU input queue
    }

    /// Process all queued input (GPU runs parallel dispatch)
    pub fn process_input(&mut self) {
        // Run gpu_process_input kernel
    }

    /// Handle click focus (find topmost, update flags)
    pub fn handle_click(&mut self, position: (f32, f32)) {
        // Run gpu_handle_click_focus then gpu_apply_focus
    }

    /// Get currently focused app
    pub fn focused_app(&self) -> Option<u32> {
        // Scan for APP_FLAG_FOCUS
    }
}
```

## Tests

```rust
#[test]
fn test_keyboard_to_focused() {
    let mut system = GpuAppSystem::new(&device)?;

    let app1 = system.launch_app(app_type::TEXT_EDITOR, 4096, 1024).unwrap();
    let app2 = system.launch_app(app_type::TEXT_EDITOR, 4096, 1024).unwrap();

    system.set_focus(app1);

    system.queue_input(InputEvent::key_down(KEY_A));
    system.process_input();

    // Only focused app received it
    let app1_desc = system.get_app(app1).unwrap();
    let app2_desc = system.get_app(app2).unwrap();

    assert!(app1_desc.input_tail > 0);
    assert_eq!(app2_desc.input_tail, 0);
}

#[test]
fn test_mouse_to_window_under_cursor() {
    let mut system = GpuAppSystem::new(&device)?;

    let app1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.create_window(app1, Rect::new(0.0, 0.0, 100.0, 100.0));

    let app2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.create_window(app2, Rect::new(200.0, 0.0, 100.0, 100.0));

    // Click in app1's window
    system.queue_input(InputEvent::mouse_click(50.0, 50.0));
    system.process_input();

    assert!(system.get_app(app1).unwrap().input_tail > 0);
    assert_eq!(system.get_app(app2).unwrap().input_tail, 0);
}

#[test]
fn test_focus_changes_on_click() {
    let mut system = GpuAppSystem::new(&device)?;

    let app1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.create_window(app1, Rect::new(0.0, 0.0, 100.0, 100.0));

    let app2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.create_window(app2, Rect::new(100.0, 0.0, 100.0, 100.0));

    // Initially no focus
    system.handle_click((50.0, 50.0));  // Click app1

    assert!(system.get_app(app1).unwrap().is_focused());
    assert!(!system.get_app(app2).unwrap().is_focused());

    system.handle_click((150.0, 50.0));  // Click app2

    assert!(!system.get_app(app1).unwrap().is_focused());
    assert!(system.get_app(app2).unwrap().is_focused());
}

#[test]
fn test_overlapping_windows_topmost_gets_click() {
    let mut system = GpuAppSystem::new(&device)?;

    let back = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.create_window(back, Rect::new(0.0, 0.0, 100.0, 100.0));

    let front = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    system.create_window(front, Rect::new(50.0, 50.0, 100.0, 100.0));

    // Click in overlap area - front should get it
    system.handle_click((75.0, 75.0));

    assert!(system.get_app(front).unwrap().is_focused());
    assert!(!system.get_app(back).unwrap().is_focused());
}
```

## Success Metrics

1. **Input latency**: < 0.1ms from queue to app
2. **Parallel evaluation**: All 64 apps check input simultaneously
3. **No central dispatcher**: Zero sequential event routing
4. **Focus resolution**: O(1) parallel max for overlapping windows
