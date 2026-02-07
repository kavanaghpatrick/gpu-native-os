# PRD: Dock as Megakernel App (Issue #161)

## Overview

Port the dock to run as an app inside GpuAppSystem's megakernel. The dock becomes just another app that the GPU updates each frame.

## Infrastructure Used

| Need | Infrastructure | File |
|------|---------------|------|
| Mouse events | GPU ring buffer | `input.rs` |
| Hit testing | GPU event dispatch | `event_loop.rs` |
| Icon rendering | Hybrid render | `render.rs` |
| Animation | GPU time-based | `memory.rs` (FrameState.time) |

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| `dock.update()` called by CPU | Megakernel calls `dock_update()` |
| CPU manages dock state | Dock state in unified buffer |
| Separate render pipeline | Writes to unified vertex buffer |
| CPU hover detection | `event_loop.rs` GPU hit-test |

## The GPU Insight

The dock is an app like any other. It just happens to:
- Always be visible
- Have REALTIME priority
- Draw at a fixed screen position
- Respond to mouse hover

```metal
// In megakernel switch
case APP_TYPE_DOCK:
    dock_update(app, unified_state, unified_vertices, tid, tg_size);
    break;
```

## Design

### Dock State Structure

```metal
struct DockState {
    uint item_count;
    uint hovered_item;      // UINT_MAX if none
    uint clicked_item;      // UINT_MAX if none
    float screen_width;
    float screen_height;
    float dock_height;
    float icon_size;
    float icon_spacing;
    float magnification;    // 1.0 = no magnification, 1.5 = 50% larger on hover
    float animation_speed;
    uint _pad[2];
    // DockItem items[MAX_DOCK_ITEMS] follows
};

struct DockItem {
    uint app_type;          // What app to launch
    uint flags;             // VISIBLE, RUNNING, HOVERED, BOUNCING
    uint running_count;     // Number of instances
    float current_size;     // Animated size
    float target_size;      // Target size (base or magnified)
    float bounce_phase;     // Bounce animation phase
    float x, y;             // Computed position
    // Icon data (texture coords or color)
    float4 icon_color;
};
```

### Dock Update Function

```metal
inline void dock_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device InputEvent* input_events,
    uint input_count,
    uint tid,
    uint tg_size
) {
    device DockState* state = (device DockState*)(unified_state + app->state_offset);
    device DockItem* items = (device DockItem*)(state + 1);

    // Thread 0: Process input, update hover state
    if (tid == 0) {
        state->hovered_item = UINT_MAX;
        state->clicked_item = UINT_MAX;

        // Check mouse position against dock bounds
        for (uint i = 0; i < input_count; i++) {
            InputEvent event = input_events[i];
            if (event.event_type == EVENT_MOUSE_MOVE || event.event_type == EVENT_MOUSE_DOWN) {
                // Is mouse in dock area?
                float dock_y = state->screen_height - state->dock_height;
                if (event.position.y >= dock_y) {
                    // Find hovered item
                    for (uint j = 0; j < state->item_count; j++) {
                        if (point_in_icon(event.position, items[j])) {
                            state->hovered_item = j;
                            if (event.event_type == EVENT_MOUSE_DOWN) {
                                state->clicked_item = j;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // All threads: Update item animations (parallel)
    uint items_per_thread = (state->item_count + tg_size - 1) / tg_size;
    uint start = tid * items_per_thread;
    uint end = min(start + items_per_thread, state->item_count);

    for (uint i = start; i < end; i++) {
        // Update target size based on hover
        items[i].target_size = (i == state->hovered_item)
            ? state->icon_size * state->magnification
            : state->icon_size;

        // Animate current size toward target
        float diff = items[i].target_size - items[i].current_size;
        items[i].current_size += diff * state->animation_speed;

        // Update bounce animation
        if (items[i].flags & DOCK_ITEM_BOUNCING) {
            items[i].bounce_phase += 0.1;
            if (items[i].bounce_phase > 6.28) {
                items[i].flags &= ~DOCK_ITEM_BOUNCING;
                items[i].bounce_phase = 0;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0: Compute layout
    if (tid == 0) {
        float total_width = 0;
        for (uint i = 0; i < state->item_count; i++) {
            total_width += items[i].current_size + state->icon_spacing;
        }
        total_width -= state->icon_spacing;

        float x = (state->screen_width - total_width) / 2.0;
        float y = state->screen_height - state->dock_height / 2.0;

        for (uint i = 0; i < state->item_count; i++) {
            items[i].x = x + items[i].current_size / 2.0;
            items[i].y = y - sin(items[i].bounce_phase) * 20.0;
            x += items[i].current_size + state->icon_spacing;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // All threads: Generate vertices (parallel)
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));

    for (uint i = start; i < end; i++) {
        // 6 vertices per icon (quad)
        uint base = i * 6;
        float half = items[i].current_size / 2.0;
        float depth = 0.99;  // Dock always on top

        write_quad(
            verts + base,
            float2(items[i].x - half, items[i].y - half),
            float2(items[i].current_size, items[i].current_size),
            depth,
            items[i].icon_color
        );
    }

    if (tid == 0) {
        app->vertex_count = state->item_count * 6;
    }
}
```

### App Type Registration

```rust
AppTypeInfo {
    type_id: app_type::DOCK,
    name: "Dock",
    state_size: mem::size_of::<DockState>() + MAX_DOCK_ITEMS * mem::size_of::<DockItem>(),
    vertex_size: MAX_DOCK_ITEMS * 6 * mem::size_of::<RenderVertex>(),
    thread_count: 64,
},
```

### Rust API

```rust
impl GpuOs {
    /// Add app to dock
    pub fn add_to_dock(&mut self, app_type: u32, icon_color: [f32; 4]) {
        unsafe {
            let state = self.get_dock_state_mut();
            let items = self.get_dock_items_mut();

            let idx = state.item_count as usize;
            items[idx] = DockItem {
                app_type,
                flags: DOCK_ITEM_VISIBLE,
                current_size: state.icon_size,
                target_size: state.icon_size,
                icon_color,
                ..Default::default()
            };
            state.item_count += 1;
        }
    }

    /// Get clicked dock item (if any)
    pub fn dock_clicked_app(&self) -> Option<u32> {
        let state = self.get_dock_state();
        if state.clicked_item != u32::MAX {
            let items = self.get_dock_items();
            Some(items[state.clicked_item as usize].app_type)
        } else {
            None
        }
    }

    /// Notify dock an app launched
    pub fn notify_dock_app_launched(&mut self, app_type: u32) {
        let items = self.get_dock_items_mut();
        for item in items.iter_mut() {
            if item.app_type == app_type {
                item.running_count += 1;
                item.flags |= DOCK_ITEM_RUNNING | DOCK_ITEM_BOUNCING;
                break;
            }
        }
    }
}
```

## Tests

```rust
#[test]
fn test_dock_launches_as_app() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).unwrap();

    let dock = system.launch_by_type(app_type::DOCK);
    assert!(dock.is_some());

    let app = system.get_app(dock.unwrap()).unwrap();
    assert_eq!(app.app_type, app_type::DOCK);
}

#[test]
fn test_dock_hover_detection() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    // Add item to dock
    os.add_to_dock(app_type::TERMINAL, [0.5, 0.5, 0.5, 1.0]);

    // Mouse move over dock
    let dock_y = os.screen_height - 50.0;
    os.system.queue_input(InputEvent::mouse_move(os.screen_width / 2.0, dock_y));
    os.system.process_input();
    os.system.run_frame();

    // Check hover state
    let state = os.get_dock_state();
    assert!(state.hovered_item != u32::MAX, "Should detect hover");
}

#[test]
fn test_dock_click_launches_app() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.add_to_dock(app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    // Click on dock item
    let dock_y = os.screen_height - 50.0;
    os.system.queue_input(InputEvent::mouse_click(os.screen_width / 2.0, dock_y, 0));
    os.system.process_input();
    os.system.run_frame();

    // Check click was registered
    assert_eq!(os.dock_clicked_app(), Some(app_type::TERMINAL));
}

#[test]
fn test_dock_magnification_animation() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.add_to_dock(app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    // Get initial size
    let items = os.get_dock_items();
    let initial_size = items[0].current_size;

    // Hover
    let dock_y = os.screen_height - 50.0;
    os.system.queue_input(InputEvent::mouse_move(os.screen_width / 2.0, dock_y));
    os.system.process_input();

    // Run several frames for animation
    for _ in 0..10 {
        os.system.mark_all_dirty();
        os.system.run_frame();
    }

    // Size should have increased
    let items = os.get_dock_items();
    assert!(items[0].current_size > initial_size, "Icon should magnify on hover");
}
```

## Benchmarks

```rust
#[test]
fn bench_dock_32_items() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    // Add 32 dock items
    for i in 0..32 {
        os.add_to_dock(app_type::CUSTOM, [i as f32 / 32.0, 0.5, 0.5, 1.0]);
    }

    // Warm up
    for _ in 0..10 {
        os.system.mark_all_dirty();
        os.system.run_frame();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..1000 {
        os.system.mark_all_dirty();
        os.system.run_frame();
    }
    let duration = start.elapsed();

    let per_frame = duration.as_micros() / 1000;
    println!("Dock (32 items): {}us/frame", per_frame);
    assert!(per_frame < 500, "Dock should render in <500us");
}
```

## Success Metrics

1. **No CPU per frame**: Dock updates entirely in megakernel
2. **60 FPS**: With 32 items and animations
3. **Hover latency**: < 16ms from mouse move to visual feedback
4. **Memory**: Single allocation in unified state buffer
