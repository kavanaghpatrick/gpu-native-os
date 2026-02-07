# PRD: GPU-Native Dock as Megakernel App (Issue #156)

## 1. Goal

Implement a GPU-native dock at the bottom of the screen. The dock is not just "rendered by GPU" - it is **computed entirely on GPU** with zero CPU involvement per frame. Each dock icon is processed by a dedicated GPU thread, enabling true parallel hover detection, icon rendering, and animation.

### Key Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| CPU loop over icons | One thread per icon |
| CPU hover detection | Parallel distance calculation |
| CPU animation update | Per-thread lerp to target |
| Sequential vertex generation | Parallel quad emission |

### Success Definition

- **Zero CPU per frame**: Dock updates entirely in megakernel
- **60 FPS**: With 32 icons and magnification animation
- **Hover latency**: < 16ms from mouse move to visual feedback
- **Memory**: Single allocation in unified state buffer

---

## 2. Existing Infrastructure to REUSE

| Need | Existing Infrastructure | File |
|------|------------------------|------|
| Vertex generation | `RenderVertex` struct, `write_quad()` pattern | `gpu_app_system.rs` |
| Input events | `InputEvent`, `queue_input()`, `process_input()` | `gpu_app_system.rs` |
| App descriptor | `GpuAppDescriptor` with state/vertex offsets | `gpu_app_system.rs` |
| Megakernel dispatch | `dispatch_app_update()` switch statement | `gpu_app_system.rs` |
| Animation time | Frame state timestamp | `gpu_app_system.rs` |
| Hit testing | `point_in_rect()` pattern | `gpu_app_system.rs` |

### RenderVertex Structure (existing)

```metal
struct RenderVertex {
    float3 position;    // x, y, z (z = depth)
    float4 color;
    float2 uv;
    float2 _pad;
};
```

### InputEvent Structure (existing)

```metal
struct InputEvent {
    uint event_type;      // EVENT_MOUSE_MOVE, EVENT_MOUSE_DOWN, etc.
    uint key_or_button;   // Mouse button
    float2 position;      // Cursor position
    uint modifiers;       // Shift, Ctrl, Alt, Cmd
    uint frame;           // Frame when event occurred
    uint _pad[2];
};
```

---

## 3. GPU-Native Approach

### 3.1 Thread Model

- **32 threads** for 32 dock icon slots (one thread per icon)
- **Thread 0**: Also handles input processing and layout computation
- **All threads**: Parallel hover detection, animation update, vertex generation

### 3.2 Parallel Hover Detection

Each thread computes distance from cursor to its icon center. No serial loop required.

```
Thread 0: distance(cursor, icon_0_center) < threshold? -> hovered[0]
Thread 1: distance(cursor, icon_1_center) < threshold? -> hovered[1]
...
Thread 31: distance(cursor, icon_31_center) < threshold? -> hovered[31]
```

### 3.3 Parallel Icon Rendering

Each thread generates 6 vertices (2 triangles) for its icon quad. All 32 icons rendered simultaneously.

### 3.4 State Machine

Each icon has independent animation state:
- `IDLE`: Normal size
- `HOVERED`: Magnifying toward target
- `CLICKED`: Flash effect
- `BOUNCING`: Launch bounce animation

---

## 4. Metal Shader Pseudocode

### 4.1 Data Structures

```metal
// ============================================================================
// DOCK DATA STRUCTURES
// ============================================================================

constant uint MAX_DOCK_ITEMS = 32;
constant uint DOCK_ITEM_VISIBLE   = 0x01;
constant uint DOCK_ITEM_RUNNING   = 0x02;
constant uint DOCK_ITEM_HOVERED   = 0x04;
constant uint DOCK_ITEM_BOUNCING  = 0x08;
constant uint DOCK_ITEM_CLICKED   = 0x10;

struct DockItem {
    uint app_type;           // What app to launch on click
    uint flags;              // VISIBLE, RUNNING, HOVERED, BOUNCING, CLICKED
    uint running_count;      // Number of running instances
    float current_size;      // Animated size (interpolates to target)
    float target_size;       // Target size (base or magnified)
    float bounce_phase;      // Bounce animation phase [0, 2*PI]
    float center_x;          // Computed center X position
    float center_y;          // Computed center Y position
    float4 icon_color;       // Icon color (or texture coords later)
};

struct DockState {
    // Counts and indices
    uint item_count;         // Number of active items (0-32)
    uint hovered_item;       // Index of hovered item (UINT_MAX if none)
    uint clicked_item;       // Index of clicked item (UINT_MAX if none)
    uint _count_pad;

    // Screen geometry
    float screen_width;
    float screen_height;
    float dock_y;            // Y position of dock top edge
    float dock_height;

    // Icon sizing
    float base_icon_size;    // Default icon size (e.g., 48px)
    float magnified_size;    // Size when hovered (e.g., 72px)
    float icon_spacing;      // Gap between icons
    float magnification_radius; // How far magnification spreads

    // Animation
    float animation_speed;   // Lerp factor per frame (0.0-1.0)
    float bounce_height;     // Max bounce height in pixels
    float bounce_speed;      // Bounce animation speed
    float time;              // Current time for animations

    // Cursor
    float2 cursor_pos;       // Last known cursor position
    uint cursor_in_dock;     // 1 if cursor is in dock area
    uint _cursor_pad;

    // Padding for 16-byte alignment
    uint _pad[2];

    // Items array follows (32 * sizeof(DockItem))
};
```

### 4.2 Icon Quad Generation (32 Slots)

```metal
// ============================================================================
// ICON QUAD GENERATION - Each thread generates one icon's vertices
// ============================================================================

inline void generate_icon_quad(
    device RenderVertex* vertices,
    uint icon_index,
    float center_x,
    float center_y,
    float size,
    float bounce_offset,
    float4 color,
    float depth
) {
    // Compute quad bounds
    float half = size * 0.5;
    float left = center_x - half;
    float right = center_x + half;
    float top = center_y - half - bounce_offset;  // Bounce goes up (negative Y)
    float bottom = center_y + half - bounce_offset;

    // Base vertex index (6 vertices per icon)
    uint base = icon_index * 6;
    device RenderVertex* v = vertices + base;

    // Triangle 1: top-left, top-right, bottom-right
    v[0].position = float3(left, top, depth);
    v[0].color = color;
    v[0].uv = float2(0.0, 0.0);

    v[1].position = float3(right, top, depth);
    v[1].color = color;
    v[1].uv = float2(1.0, 0.0);

    v[2].position = float3(right, bottom, depth);
    v[2].color = color;
    v[2].uv = float2(1.0, 1.0);

    // Triangle 2: top-left, bottom-right, bottom-left
    v[3].position = float3(left, top, depth);
    v[3].color = color;
    v[3].uv = float2(0.0, 0.0);

    v[4].position = float3(right, bottom, depth);
    v[4].color = color;
    v[4].uv = float2(1.0, 1.0);

    v[5].position = float3(left, bottom, depth);
    v[5].color = color;
    v[5].uv = float2(0.0, 1.0);
}
```

### 4.3 Hover Detection (Distance from Cursor)

```metal
// ============================================================================
// PARALLEL HOVER DETECTION - Each thread checks its own icon
// ============================================================================

inline bool is_cursor_near_icon(
    float2 cursor_pos,
    float icon_center_x,
    float icon_center_y,
    float icon_size,
    float magnification_radius
) {
    // Distance from cursor to icon center
    float dx = cursor_pos.x - icon_center_x;
    float dy = cursor_pos.y - icon_center_y;
    float distance_sq = dx * dx + dy * dy;

    // Icon is "hovered" if cursor within icon bounds + some margin
    float hit_radius = icon_size * 0.5 + 4.0;  // 4px margin
    return distance_sq < (hit_radius * hit_radius);
}

// Calculate magnification factor based on distance (for neighbor scaling)
inline float magnification_factor(
    float cursor_x,
    float icon_center_x,
    float magnification_radius,
    float base_magnification  // e.g., 1.5 for 50% larger
) {
    float distance = abs(cursor_x - icon_center_x);

    if (distance > magnification_radius) {
        return 1.0;  // No magnification
    }

    // Smooth falloff: 1.0 at edge, base_magnification at center
    float t = 1.0 - (distance / magnification_radius);
    float smooth_t = t * t * (3.0 - 2.0 * t);  // Smoothstep
    return 1.0 + (base_magnification - 1.0) * smooth_t;
}
```

### 4.4 Magnification Effect (Icons Grow When Hovered)

```metal
// ============================================================================
// MAGNIFICATION EFFECT - Smooth scaling with neighbor influence
// ============================================================================

inline float compute_target_size(
    device DockState* state,
    device DockItem* item,
    uint item_index,
    float cursor_x
) {
    // If cursor not in dock, all icons return to base size
    if (!state->cursor_in_dock) {
        return state->base_icon_size;
    }

    // Calculate magnification based on distance to cursor
    float mag = magnification_factor(
        cursor_x,
        item->center_x,
        state->magnification_radius,
        state->magnified_size / state->base_icon_size
    );

    return state->base_icon_size * mag;
}

// Animate current size toward target (called per thread)
inline void animate_size(
    device DockItem* item,
    float target_size,
    float animation_speed
) {
    // Lerp toward target
    float diff = target_size - item->current_size;
    item->current_size += diff * animation_speed;

    // Snap if close enough
    if (abs(diff) < 0.1) {
        item->current_size = target_size;
    }
}
```

### 4.5 Click Detection

```metal
// ============================================================================
// CLICK DETECTION - Thread 0 processes input events
// ============================================================================

inline void process_dock_input(
    device DockState* state,
    device DockItem* items,
    device InputEvent* events,
    uint event_count
) {
    state->clicked_item = UINT_MAX;
    state->hovered_item = UINT_MAX;

    for (uint i = 0; i < event_count; i++) {
        InputEvent event = events[i];

        // Update cursor position
        if (event.event_type == EVENT_MOUSE_MOVE ||
            event.event_type == EVENT_MOUSE_DOWN) {
            state->cursor_pos = event.position;

            // Check if cursor is in dock area
            state->cursor_in_dock = (event.position.y >= state->dock_y) ? 1 : 0;

            if (state->cursor_in_dock) {
                // Find which icon is under cursor
                for (uint j = 0; j < state->item_count; j++) {
                    if (!(items[j].flags & DOCK_ITEM_VISIBLE)) continue;

                    if (is_cursor_near_icon(
                        event.position,
                        items[j].center_x,
                        items[j].center_y,
                        items[j].current_size,
                        state->magnification_radius
                    )) {
                        state->hovered_item = j;

                        if (event.event_type == EVENT_MOUSE_DOWN) {
                            state->clicked_item = j;
                            items[j].flags |= DOCK_ITEM_CLICKED;
                        }
                        break;
                    }
                }
            }
        }
    }
}
```

### 4.6 Complete Dock Update Function

```metal
// ============================================================================
// DOCK UPDATE - Full megakernel app implementation
// ============================================================================

inline void dock_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    uint tid,
    uint tg_size
) {
    device DockState* state = (device DockState*)(unified_state + app->state_offset);
    device DockItem* items = (device DockItem*)(state + 1);
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));

    // ========================================================================
    // PHASE 1: Input Processing (Thread 0 only)
    // ========================================================================
    if (tid == 0) {
        // Input events would be passed via app's input queue
        // For now, cursor_pos is updated by CPU before frame

        // Reset click state
        state->clicked_item = UINT_MAX;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 2: Parallel Hover Detection (All threads)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        // Each thread checks if cursor is over its icon
        bool hovered = state->cursor_in_dock && is_cursor_near_icon(
            state->cursor_pos,
            items[tid].center_x,
            items[tid].center_y,
            items[tid].current_size,
            state->magnification_radius
        );

        if (hovered) {
            items[tid].flags |= DOCK_ITEM_HOVERED;
            // Atomic write to find hovered item (lowest index wins)
            atomic_min((device atomic_uint*)&state->hovered_item, tid);
        } else {
            items[tid].flags &= ~DOCK_ITEM_HOVERED;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 3: Compute Target Sizes (All threads in parallel)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        float target = compute_target_size(state, &items[tid], tid, state->cursor_pos.x);
        items[tid].target_size = target;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 4: Animate Sizes (All threads in parallel)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        animate_size(&items[tid], items[tid].target_size, state->animation_speed);

        // Update bounce animation
        if (items[tid].flags & DOCK_ITEM_BOUNCING) {
            items[tid].bounce_phase += state->bounce_speed;
            if (items[tid].bounce_phase > 6.28318530718) {  // 2*PI
                items[tid].flags &= ~DOCK_ITEM_BOUNCING;
                items[tid].bounce_phase = 0.0;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 5: Compute Layout (Thread 0 only)
    // ========================================================================
    if (tid == 0) {
        // Calculate total width with current sizes
        float total_width = 0.0;
        for (uint i = 0; i < state->item_count; i++) {
            if (items[i].flags & DOCK_ITEM_VISIBLE) {
                total_width += items[i].current_size + state->icon_spacing;
            }
        }
        total_width -= state->icon_spacing;  // No trailing space

        // Center horizontally, position at bottom
        float start_x = (state->screen_width - total_width) * 0.5;
        float center_y = state->screen_height - (state->dock_height * 0.5);

        float current_x = start_x;
        for (uint i = 0; i < state->item_count; i++) {
            if (items[i].flags & DOCK_ITEM_VISIBLE) {
                items[i].center_x = current_x + items[i].current_size * 0.5;
                items[i].center_y = center_y;
                current_x += items[i].current_size + state->icon_spacing;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 6: Generate Vertices (All threads in parallel)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        // Compute bounce offset
        float bounce_offset = 0.0;
        if (items[tid].flags & DOCK_ITEM_BOUNCING) {
            bounce_offset = sin(items[tid].bounce_phase) * state->bounce_height;
        }

        // Generate quad for this icon
        generate_icon_quad(
            verts,
            tid,
            items[tid].center_x,
            items[tid].center_y,
            items[tid].current_size,
            bounce_offset,
            items[tid].icon_color,
            0.99  // Dock always near top of z-order
        );
    }

    // ========================================================================
    // PHASE 7: Update Vertex Count (Thread 0 only)
    // ========================================================================
    if (tid == 0) {
        uint visible_count = 0;
        for (uint i = 0; i < state->item_count; i++) {
            if (items[i].flags & DOCK_ITEM_VISIBLE) {
                visible_count++;
            }
        }
        app->vertex_count = visible_count * 6;
    }
}
```

---

## 5. Animation State Machine (Bounce on Launch)

### States

```
                   +---------+
                   |  IDLE   |<-------------------+
                   +---------+                    |
                        |                         |
                   [app launch]                   |
                        v                         |
                   +---------+                    |
                   |BOUNCING |----[complete]------+
                   +---------+
                        |
                   [hover while bouncing]
                        v
                   +---------+
                   |HOVERED  |
                   |BOUNCING |
                   +---------+
```

### Bounce Animation

```metal
// Bounce follows a sin curve for 1-3 bounces
float bounce_animation(float phase, float height) {
    // Three bounces with decreasing amplitude
    float bounce1 = sin(phase) * height;
    float bounce2 = sin(phase * 2.0) * height * 0.5;
    float bounce3 = sin(phase * 3.0) * height * 0.25;

    // Phase determines which bounce we're in
    if (phase < 3.14159) {
        return bounce1;
    } else if (phase < 4.71238) {
        return bounce2;
    } else {
        return bounce3;
    }
}
```

### Trigger Bounce on App Launch

```rust
// In GpuAppSystem or GpuOs
pub fn notify_app_launched(&mut self, app_type: u32) {
    let state = self.get_dock_state_mut();
    let items = self.get_dock_items_mut();

    for item in items.iter_mut().take(state.item_count as usize) {
        if item.app_type == app_type {
            item.running_count += 1;
            item.flags |= DOCK_ITEM_RUNNING | DOCK_ITEM_BOUNCING;
            item.bounce_phase = 0.0;
            break;
        }
    }
}
```

---

## 6. Tests (Rust Test Code)

```rust
// tests/test_issue_156_dock.rs

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::*;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device found")
}

// ============================================================================
// BASIC DOCK TESTS
// ============================================================================

#[test]
fn test_dock_launches_as_megakernel_app() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK);
    assert!(dock_slot.is_some(), "Should be able to launch dock");

    let app = system.get_app(dock_slot.unwrap()).unwrap();
    assert_eq!(app.app_type, app_type::DOCK);
    assert!(app.flags & flags::ACTIVE != 0, "Dock should be active");
    assert!(app.flags & flags::VISIBLE != 0, "Dock should be visible");
}

#[test]
fn test_dock_state_initialization() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);

    let state = system.get_dock_state(dock_slot);
    assert_eq!(state.screen_width, 1920.0);
    assert_eq!(state.screen_height, 1080.0);
    assert!(state.dock_height > 0.0, "Dock height should be set");
    assert!(state.base_icon_size > 0.0, "Icon size should be set");
}

#[test]
fn test_dock_add_items() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);

    // Add items to dock
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);
    system.add_dock_item(dock_slot, app_type::FILESYSTEM, [0.0, 1.0, 0.0, 1.0]);
    system.add_dock_item(dock_slot, app_type::SETTINGS, [0.0, 0.0, 1.0, 1.0]);

    let state = system.get_dock_state(dock_slot);
    assert_eq!(state.item_count, 3, "Should have 3 dock items");
}

#[test]
fn test_dock_runs_in_megakernel() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    system.mark_dirty(dock_slot);
    system.run_frame();

    let app = system.get_app(dock_slot).unwrap();
    assert_eq!(app.last_run_frame, 1, "Dock should have run in frame 1");
    assert!(app.vertex_count > 0, "Dock should have generated vertices");
}

// ============================================================================
// HOVER DETECTION TESTS
// ============================================================================

#[test]
fn test_dock_hover_detection() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    // Run a frame to compute initial layout
    system.mark_dirty(dock_slot);
    system.run_frame();

    // Move cursor to dock area (center of screen, near bottom)
    let dock_y = 1080.0 - 40.0;  // Assuming 80px dock height
    system.queue_input(InputEvent::mouse_move(960.0, dock_y));
    system.process_input();
    system.mark_dirty(dock_slot);
    system.run_frame();

    // Check hover was detected
    let state = system.get_dock_state(dock_slot);
    assert!(state.hovered_item != u32::MAX, "Should detect hover over dock item");
    assert_eq!(state.hovered_item, 0, "First item should be hovered");
}

#[test]
fn test_dock_hover_leaves_when_cursor_exits() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    // Hover over dock
    system.queue_input(InputEvent::mouse_move(960.0, 1040.0));
    system.process_input();
    system.mark_dirty(dock_slot);
    system.run_frame();

    // Move cursor away from dock
    system.queue_input(InputEvent::mouse_move(960.0, 500.0));  // Middle of screen
    system.process_input();
    system.mark_dirty(dock_slot);
    system.run_frame();

    let state = system.get_dock_state(dock_slot);
    assert_eq!(state.hovered_item, u32::MAX, "No item should be hovered");
    assert_eq!(state.cursor_in_dock, 0, "Cursor should not be in dock");
}

// ============================================================================
// MAGNIFICATION TESTS
// ============================================================================

#[test]
fn test_dock_magnification_on_hover() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    // Get initial size
    system.mark_dirty(dock_slot);
    system.run_frame();
    let items = system.get_dock_items(dock_slot);
    let initial_size = items[0].current_size;

    // Hover and run several frames for animation
    system.queue_input(InputEvent::mouse_move(960.0, 1040.0));
    system.process_input();

    for _ in 0..20 {
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    // Size should have increased
    let items = system.get_dock_items(dock_slot);
    assert!(items[0].current_size > initial_size,
            "Icon should magnify: {} > {}", items[0].current_size, initial_size);
}

#[test]
fn test_dock_neighbor_magnification() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);

    // Add three items
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);
    system.add_dock_item(dock_slot, app_type::FILESYSTEM, [0.0, 1.0, 0.0, 1.0]);
    system.add_dock_item(dock_slot, app_type::SETTINGS, [0.0, 0.0, 1.0, 1.0]);

    // Run initial layout
    system.mark_dirty(dock_slot);
    system.run_frame();

    // Get center item position and hover over it
    let items = system.get_dock_items(dock_slot);
    let center_x = items[1].center_x;
    let dock_y = 1040.0;

    system.queue_input(InputEvent::mouse_move(center_x, dock_y));
    system.process_input();

    for _ in 0..20 {
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    let items = system.get_dock_items(dock_slot);
    let state = system.get_dock_state(dock_slot);

    // Center item should be largest
    assert!(items[1].current_size > items[0].current_size,
            "Center item should be larger than neighbor");

    // Neighbors should still be larger than base (partial magnification)
    assert!(items[0].current_size > state.base_icon_size * 0.99,
            "Neighbors should have some magnification");
}

// ============================================================================
// CLICK TESTS
// ============================================================================

#[test]
fn test_dock_click_detection() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    // Run initial layout
    system.mark_dirty(dock_slot);
    system.run_frame();

    // Click on dock item
    system.queue_input(InputEvent::mouse_click(960.0, 1040.0, 0));
    system.process_input();
    system.mark_dirty(dock_slot);
    system.run_frame();

    let state = system.get_dock_state(dock_slot);
    assert!(state.clicked_item != u32::MAX, "Click should be detected");
    assert_eq!(state.clicked_item, 0, "First item should be clicked");
}

#[test]
fn test_dock_click_returns_app_type() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);
    system.add_dock_item(dock_slot, app_type::FILESYSTEM, [0.0, 1.0, 0.0, 1.0]);

    system.mark_dirty(dock_slot);
    system.run_frame();

    // Get second item position
    let items = system.get_dock_items(dock_slot);
    let item_x = items[1].center_x;

    system.queue_input(InputEvent::mouse_click(item_x, 1040.0, 0));
    system.process_input();
    system.mark_dirty(dock_slot);
    system.run_frame();

    // Should be able to get clicked app type
    let clicked_type = system.get_clicked_dock_app(dock_slot);
    assert_eq!(clicked_type, Some(app_type::FILESYSTEM));
}

// ============================================================================
// ANIMATION TESTS
// ============================================================================

#[test]
fn test_dock_bounce_animation() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    system.mark_dirty(dock_slot);
    system.run_frame();

    // Trigger bounce
    system.trigger_dock_bounce(dock_slot, 0);

    let items = system.get_dock_items(dock_slot);
    assert!(items[0].flags & DOCK_ITEM_BOUNCING != 0, "Item should be bouncing");

    // Record Y positions over several frames
    let mut y_positions = Vec::new();
    for _ in 0..30 {
        system.mark_dirty(dock_slot);
        system.run_frame();

        // Read vertex Y position (would need vertex buffer access)
        let items = system.get_dock_items(dock_slot);
        y_positions.push(items[0].bounce_phase);
    }

    // Bounce phase should have progressed
    assert!(y_positions.last().unwrap() > y_positions.first().unwrap(),
            "Bounce phase should progress");
}

#[test]
fn test_dock_bounce_completes() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    system.mark_dirty(dock_slot);
    system.run_frame();

    system.trigger_dock_bounce(dock_slot, 0);

    // Run enough frames for bounce to complete
    for _ in 0..100 {
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    let items = system.get_dock_items(dock_slot);
    assert!(items[0].flags & DOCK_ITEM_BOUNCING == 0, "Bounce should complete");
    assert_eq!(items[0].bounce_phase, 0.0, "Bounce phase should reset");
}

// ============================================================================
// VERTEX GENERATION TESTS
// ============================================================================

#[test]
fn test_dock_generates_correct_vertex_count() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);

    // Add 5 items
    for i in 0..5 {
        let color = [(i as f32) / 5.0, 0.5, 0.5, 1.0];
        system.add_dock_item(dock_slot, app_type::CUSTOM, color);
    }

    system.mark_dirty(dock_slot);
    system.run_frame();

    let app = system.get_app(dock_slot).unwrap();
    assert_eq!(app.vertex_count, 5 * 6, "5 icons * 6 vertices each = 30");
}

#[test]
fn test_dock_vertices_are_valid() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    system.mark_dirty(dock_slot);
    system.run_frame();

    // Read vertices from buffer
    let vertices = system.read_dock_vertices(dock_slot);

    // Should have 6 vertices
    assert_eq!(vertices.len(), 6);

    // All vertices should be in screen bounds
    for v in &vertices {
        assert!(v.position[0] >= 0.0 && v.position[0] <= 1920.0,
                "X should be in bounds: {}", v.position[0]);
        assert!(v.position[1] >= 0.0 && v.position[1] <= 1080.0,
                "Y should be in bounds: {}", v.position[1]);
    }

    // Color should match what we set
    assert_eq!(vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
}

// ============================================================================
// PERFORMANCE BENCHMARK
// ============================================================================

#[test]
fn bench_dock_32_items() {
    use std::time::Instant;

    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);

    // Add 32 dock items (maximum)
    for i in 0..32 {
        let hue = (i as f32) / 32.0;
        let color = [hue, 1.0 - hue, 0.5, 1.0];
        system.add_dock_item(dock_slot, app_type::CUSTOM, color);
    }

    // Warm up
    for _ in 0..10 {
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    // Benchmark
    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    let duration = start.elapsed();
    let per_frame_us = duration.as_micros() / iterations as u128;

    println!("Dock (32 items): {}us/frame", per_frame_us);
    assert!(per_frame_us < 500, "Dock should update in <500us, got {}us", per_frame_us);
}

#[test]
fn bench_dock_with_animation() {
    use std::time::Instant;

    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(1920.0, 1080.0);

    for i in 0..16 {
        let color = [(i as f32) / 16.0, 0.5, 0.5, 1.0];
        system.add_dock_item(dock_slot, app_type::CUSTOM, color);
    }

    // Simulate continuous hover animation
    system.queue_input(InputEvent::mouse_move(960.0, 1040.0));
    system.process_input();

    // Warm up
    for _ in 0..10 {
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    // Benchmark with hover (magnification active)
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        // Move cursor slightly to simulate real usage
        let x = 960.0 + (i % 100) as f32 * 0.5;
        system.queue_input(InputEvent::mouse_move(x, 1040.0));
        system.process_input();
        system.mark_dirty(dock_slot);
        system.run_frame();
    }

    let duration = start.elapsed();
    let per_frame_us = duration.as_micros() / iterations as u128;

    println!("Dock (16 items, animated): {}us/frame", per_frame_us);
    assert!(per_frame_us < 1000, "Dock with animation should update in <1ms, got {}us", per_frame_us);
}
```

---

## 7. Implementation Checklist

### Phase 1: Data Structures
- [ ] Define `DockState` struct in Metal shader
- [ ] Define `DockItem` struct in Metal shader
- [ ] Add Rust mirror structs with `#[repr(C)]`
- [ ] Update `AppTypeInfo` for DOCK with correct sizes

### Phase 2: Basic Rendering
- [ ] Implement `generate_icon_quad()` helper
- [ ] Implement basic `dock_update()` with single-threaded vertex gen
- [ ] Test: Dock displays colored squares

### Phase 3: Parallel Hover
- [ ] Implement `is_cursor_near_icon()` helper
- [ ] Add parallel hover detection (all threads)
- [ ] Implement `atomic_min` for hovered_item
- [ ] Test: Hover is detected

### Phase 4: Magnification
- [ ] Implement `magnification_factor()` with smoothstep
- [ ] Add `compute_target_size()` with neighbor influence
- [ ] Implement `animate_size()` lerp
- [ ] Test: Icons grow on hover

### Phase 5: Click & Launch
- [ ] Add click detection in input processing
- [ ] Expose `get_clicked_dock_app()` API
- [ ] Wire up to app launch system
- [ ] Test: Click launches app

### Phase 6: Bounce Animation
- [ ] Implement bounce animation math
- [ ] Add `trigger_dock_bounce()` API
- [ ] Connect to app launch notification
- [ ] Test: Launch triggers bounce

### Phase 7: Polish
- [ ] Add running indicator (dot below icon)
- [ ] Add tooltip support (hover delay)
- [ ] Performance optimization
- [ ] Documentation

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| CPU usage per frame | 0% | No CPU code in dock update |
| Frame time (32 icons) | < 500us | Benchmark test |
| Hover latency | < 16ms | Input to visual change |
| Animation smoothness | 60 FPS | No dropped frames |
| Memory | Single allocation | No per-frame alloc |
