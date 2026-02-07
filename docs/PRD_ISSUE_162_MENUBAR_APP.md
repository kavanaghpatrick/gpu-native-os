# PRD: MenuBar as Megakernel App (Issue #162)

## Overview

Port the menu bar to run as an app inside GpuAppSystem's megakernel. The menu bar becomes a system app with REALTIME priority.

## Infrastructure Used

| Need | Infrastructure | File |
|------|---------------|------|
| Mouse events | GPU ring buffer | `input.rs` |
| Hit testing | GPU event dispatch | `event_loop.rs` |
| Text rendering | Bitmap font | `text_render.rs` |
| Dropdown layout | Level-parallel | `document/layout.metal` |

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| CPU processes menu clicks | `event_loop.rs` GPU hit-test |
| CPU manages dropdown state | Menu state in unified buffer |
| Separate menu renderer | Writes to unified vertex buffer |
| Event callback system | GPU reads from input queue |

## Design

### MenuBar State Structure

```metal
struct MenuBarState {
    float screen_width;
    float bar_height;
    uint menu_count;
    uint open_menu;          // Index of open menu, UINT_MAX if none
    uint hovered_menu;       // Index of hovered menu
    uint hovered_item;       // Index of hovered item in open menu
    uint selected_item;      // Clicked item (triggers action)
    float time;              // For animations
    uint _pad[2];
    // MenuItem menus[MAX_MENUS] follows
    // DropdownItem items[MAX_ITEMS] follows after menus
};

struct MenuItem {
    float x;                 // Computed position
    float width;             // Computed width
    uint flags;              // VISIBLE, ENABLED, SEPARATOR
    uint first_item;         // First dropdown item index
    uint item_count;         // Number of dropdown items
    uint _pad;
    char name[32];
};

struct DropdownItem {
    uint action_id;          // Action to trigger when clicked
    uint flags;              // ENABLED, CHECKED, SEPARATOR
    uint shortcut_key;       // Keyboard shortcut
    uint shortcut_mods;      // Modifier keys
    char name[32];
};
```

### MenuBar Update Function

```metal
inline void menubar_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    uint tid,
    uint tg_size
) {
    device MenuBarState* state = (device MenuBarState*)(unified_state + app->state_offset);
    device MenuItem* menus = (device MenuItem*)(state + 1);

    // Thread 0: Update state from input
    if (tid == 0) {
        state->selected_item = UINT_MAX;
        // Input processing happens via app's local input queue
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Generate vertices for menu bar background
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));

    if (tid == 0) {
        // Menu bar background
        write_quad(verts, float2(0, 0), float2(state->screen_width, state->bar_height),
            0.98, float4(0.95, 0.95, 0.95, 0.9));  // Translucent gray

        uint vertex_count = 6;

        // Menu titles
        for (uint i = 0; i < state->menu_count; i++) {
            float4 color = (i == state->hovered_menu || i == state->open_menu)
                ? float4(0.3, 0.5, 0.9, 1.0)   // Highlighted
                : float4(0.0, 0.0, 0.0, 1.0);  // Normal

            // Menu title text would go here (simplified as colored quad)
            write_quad(verts + vertex_count,
                float2(menus[i].x, 2.0),
                float2(menus[i].width, state->bar_height - 4.0),
                0.98, color);
            vertex_count += 6;
        }

        // Open dropdown (if any)
        if (state->open_menu != UINT_MAX) {
            MenuItem menu = menus[state->open_menu];
            device DropdownItem* items = (device DropdownItem*)((device uchar*)menus +
                MAX_MENUS * sizeof(MenuItem));

            float dropdown_y = state->bar_height;
            float dropdown_height = menu.item_count * 24.0;

            // Dropdown background
            write_quad(verts + vertex_count,
                float2(menu.x, dropdown_y),
                float2(200.0, dropdown_height),
                0.97, float4(1.0, 1.0, 1.0, 0.95));
            vertex_count += 6;

            // Dropdown items
            for (uint i = 0; i < menu.item_count; i++) {
                uint item_idx = menu.first_item + i;
                float4 item_color = (item_idx == state->hovered_item)
                    ? float4(0.3, 0.5, 0.9, 1.0)
                    : float4(0.0, 0.0, 0.0, 1.0);

                write_quad(verts + vertex_count,
                    float2(menu.x + 4.0, dropdown_y + i * 24.0 + 2.0),
                    float2(192.0, 20.0),
                    0.97, item_color);
                vertex_count += 6;
            }
        }

        app->vertex_count = vertex_count;
    }
}
```

### Integration with GpuOs

```rust
impl GpuOs {
    /// Add a menu to the menu bar
    pub fn add_menu(&mut self, name: &str, items: &[MenuItemDef]) {
        unsafe {
            let state = self.get_menubar_state_mut();
            let menus = self.get_menus_mut();
            let dropdown_items = self.get_dropdown_items_mut();

            let menu_idx = state.menu_count as usize;
            let first_item = self.next_dropdown_item_idx;

            // Add menu
            menus[menu_idx] = MenuItem {
                flags: MENU_VISIBLE | MENU_ENABLED,
                first_item: first_item as u32,
                item_count: items.len() as u32,
                ..Default::default()
            };
            menus[menu_idx].set_name(name);

            // Add dropdown items
            for (i, item) in items.iter().enumerate() {
                dropdown_items[first_item + i] = DropdownItem {
                    action_id: item.action_id,
                    flags: item.flags,
                    shortcut_key: item.shortcut_key,
                    shortcut_mods: item.shortcut_mods,
                    ..Default::default()
                };
                dropdown_items[first_item + i].set_name(&item.name);
            }

            state.menu_count += 1;
            self.next_dropdown_item_idx += items.len();
        }
    }

    /// Check if a menu action was triggered
    pub fn menu_action(&self) -> Option<u32> {
        let state = self.get_menubar_state();
        if state.selected_item != u32::MAX {
            let items = self.get_dropdown_items();
            Some(items[state.selected_item as usize].action_id)
        } else {
            None
        }
    }
}
```

## Tests

```rust
#[test]
fn test_menubar_launches_as_app() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).unwrap();

    let menubar = system.launch_by_type(app_type::MENUBAR);
    assert!(menubar.is_some());
}

#[test]
fn test_add_menu() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.add_menu("File", &[
        MenuItemDef::new("New", ACTION_NEW, KEY_N, MOD_CMD),
        MenuItemDef::new("Open", ACTION_OPEN, KEY_O, MOD_CMD),
        MenuItemDef::separator(),
        MenuItemDef::new("Quit", ACTION_QUIT, KEY_Q, MOD_CMD),
    ]);

    let state = os.get_menubar_state();
    assert_eq!(state.menu_count, 1);

    let menus = os.get_menus();
    assert_eq!(menus[0].item_count, 4);
}

#[test]
fn test_menu_click() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.add_menu("File", &[
        MenuItemDef::new("Quit", ACTION_QUIT, 0, 0),
    ]);

    // Click on menu
    os.system.queue_input(InputEvent::mouse_click(10.0, 10.0, 0));
    os.system.process_input();
    os.system.run_frame();

    // Click on item
    os.system.queue_input(InputEvent::mouse_click(10.0, 30.0, 0));
    os.system.process_input();
    os.system.run_frame();

    assert_eq!(os.menu_action(), Some(ACTION_QUIT));
}
```

## Benchmarks

```rust
#[test]
fn bench_menubar_render() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    // Add typical menus
    os.add_menu("File", &file_menu_items());
    os.add_menu("Edit", &edit_menu_items());
    os.add_menu("View", &view_menu_items());
    os.add_menu("Window", &window_menu_items());
    os.add_menu("Help", &help_menu_items());

    let start = Instant::now();
    for _ in 0..1000 {
        os.system.mark_dirty(os.menubar_slot);
        os.system.run_frame();
    }
    let duration = start.elapsed();

    let per_frame = duration.as_micros() / 1000;
    println!("MenuBar: {}us/frame", per_frame);
}
```

## Success Metrics

1. **No CPU per frame**: Menu updates in megakernel
2. **Keyboard shortcuts**: Handled in GPU input processing
3. **Smooth dropdowns**: < 16ms open/close animation
4. **Memory**: Fixed allocation in unified buffer
