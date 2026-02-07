# PRD: MenuBar as Megakernel App (Issue #157)

## Overview

Implement a fully GPU-native menu bar that runs as a megakernel app inside GpuAppSystem. The menu bar sits at the top of the screen with a translucent background, displays menu titles (File, Edit, View, etc.), handles hover states, opens dropdown menus, and processes menu item clicks - all on the GPU with zero CPU involvement per frame.

## Goal

Create a GPU-native menu bar that:
1. Renders at the top of the screen with translucent background
2. Displays menu titles with parallel text generation
3. Handles hover detection with parallel hit testing (one thread per menu item)
4. Opens dropdown menus on click
5. Triggers actions when menu items are selected
6. Supports keyboard shortcuts processed on GPU

**Success criteria**: Menu bar updates entirely in megakernel dispatch with no CPU per-frame involvement.

## Existing Infrastructure to REUSE

| Need | Infrastructure | File | Notes |
|------|---------------|------|-------|
| Text rendering | Bitmap font system | `text_render.rs` | 8x8 bitmap font, `TextChar` struct, batch rendering |
| Vertex generation | `write_quad()` helper | `benchmark_visual.rs` | Standard quad generation pattern |
| Input events | GPU ring buffer | `input.rs` | `InputEvent`, `InputQueue`, mouse/key events |
| Hit testing | GPU event dispatch | `event_loop.rs` | Point-in-rect testing pattern |
| App descriptor | Megakernel system | `gpu_app_system.rs` | `GpuAppDescriptor`, unified state/vertex buffers |
| App type registry | `APP_TYPES` constant | `gpu_app_system.rs` | `app_type::MENUBAR = 202`, 8KB state, 1000 vertices |

## GPU-Native Approach

### Parallelization Strategy

| Task | Parallel Decomposition | Threads Used |
|------|----------------------|--------------|
| Hover detection | One thread per menu item | `tid < menu_count` |
| Text generation | One thread per character | `tid < total_chars` |
| Dropdown items | One thread per dropdown item | `tid < item_count` |
| Background quads | Thread 0 only (sequential) | 1 thread |

### CPU vs GPU Responsibilities

| CPU Pattern (WRONG) | GPU Pattern (RIGHT) |
|---------------------|---------------------|
| CPU processes menu clicks | GPU reads from input queue directly |
| CPU manages dropdown state | `MenuBarState` in unified buffer |
| CPU renders text to texture | GPU generates text quads in-place |
| CPU hit-tests mouse position | Parallel hit-test: `thread[i]` tests `menu[i]` |
| Event callback system | GPU writes `selected_item`, CPU polls once |
| Separate menu renderer | Writes to unified vertex buffer |

## Data Structures

### MenuBarState (GPU-resident)

```metal
// Constants
constant uint MAX_MENUS = 16;
constant uint MAX_ITEMS_PER_MENU = 32;
constant uint MAX_TOTAL_ITEMS = 256;
constant uint MENU_NAME_LEN = 32;
constant uint ITEM_NAME_LEN = 32;
constant uint UINT_MAX_VAL = 0xFFFFFFFF;

// Flags
constant uint MENU_FLAG_VISIBLE = 1 << 0;
constant uint MENU_FLAG_ENABLED = 1 << 1;
constant uint ITEM_FLAG_ENABLED = 1 << 0;
constant uint ITEM_FLAG_CHECKED = 1 << 1;
constant uint ITEM_FLAG_SEPARATOR = 1 << 2;

struct MenuBarState {
    // Dimensions (16 bytes)
    float screen_width;
    float bar_height;          // Typically 22-24 pixels
    float padding_x;           // Horizontal padding between menus
    float text_scale;          // Text scale factor (default 1.5)

    // Counts (8 bytes)
    uint menu_count;
    uint total_item_count;

    // Interaction state (16 bytes)
    uint open_menu;            // Index of currently open menu, UINT_MAX if none
    uint hovered_menu;         // Index of hovered menu title, UINT_MAX if none
    uint hovered_item;         // Index of hovered dropdown item, UINT_MAX if none
    uint selected_item;        // Item clicked this frame, UINT_MAX if none

    // Input state (16 bytes)
    float2 mouse_pos;
    uint mouse_down;
    uint mouse_clicked;

    // Animation (8 bytes)
    float time;
    float dropdown_anim;       // 0.0 = closed, 1.0 = fully open

    // Padding (for 16-byte alignment)
    uint _pad[2];

    // Followed by:
    // MenuItem menus[MAX_MENUS]
    // DropdownItem items[MAX_TOTAL_ITEMS]
};

struct MenuItem {
    // Position (computed each frame)
    float x;
    float width;

    // Configuration
    uint flags;                // VISIBLE, ENABLED
    uint first_item;           // Index of first dropdown item
    uint item_count;           // Number of dropdown items
    uint _pad;

    // Name (null-terminated, padded to 32 bytes)
    char name[MENU_NAME_LEN];

    // Total: 56 bytes
};

struct DropdownItem {
    // Action
    uint action_id;            // Action to trigger when clicked
    uint flags;                // ENABLED, CHECKED, SEPARATOR

    // Keyboard shortcut
    uint shortcut_key;         // Key code (0 if none)
    uint shortcut_mods;        // Modifier flags (CMD, SHIFT, etc.)

    // Name (null-terminated, padded to 32 bytes)
    char name[ITEM_NAME_LEN];

    // Total: 48 bytes
};
```

### Rust Mirror Structs

```rust
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MenuBarState {
    pub screen_width: f32,
    pub bar_height: f32,
    pub padding_x: f32,
    pub text_scale: f32,

    pub menu_count: u32,
    pub total_item_count: u32,

    pub open_menu: u32,
    pub hovered_menu: u32,
    pub hovered_item: u32,
    pub selected_item: u32,

    pub mouse_pos: [f32; 2],
    pub mouse_down: u32,
    pub mouse_clicked: u32,

    pub time: f32,
    pub dropdown_anim: f32,

    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MenuItem {
    pub x: f32,
    pub width: f32,
    pub flags: u32,
    pub first_item: u32,
    pub item_count: u32,
    pub _pad: u32,
    pub name: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DropdownItem {
    pub action_id: u32,
    pub flags: u32,
    pub shortcut_key: u32,
    pub shortcut_mods: u32,
    pub name: [u8; 32],
}
```

## Metal Shader Pseudocode

### Helper: Write Quad

```metal
inline void write_quad(
    device RenderVertex* verts,
    float2 pos,
    float2 size,
    float depth,
    float4 color
) {
    // Triangle 1: TL, TR, BR
    verts[0] = { float4(pos.x, pos.y, depth, 1.0), color };
    verts[1] = { float4(pos.x + size.x, pos.y, depth, 1.0), color };
    verts[2] = { float4(pos.x + size.x, pos.y + size.y, depth, 1.0), color };
    // Triangle 2: TL, BR, BL
    verts[3] = { float4(pos.x, pos.y, depth, 1.0), color };
    verts[4] = { float4(pos.x + size.x, pos.y + size.y, depth, 1.0), color };
    verts[5] = { float4(pos.x, pos.y + size.y, depth, 1.0), color };
}
```

### Helper: Write Text Character

```metal
// TextChar compatible with text_render.rs shader
struct TextCharGpu {
    float x;
    float y;
    uint char_code;
    uint color;  // RGBA packed as 0xRRGGBBAA
};

inline void write_text_char(
    device TextCharGpu* chars,
    uint idx,
    float x,
    float y,
    char c,
    uint color
) {
    chars[idx].x = x;
    chars[idx].y = y;
    chars[idx].char_code = (uint)c;
    chars[idx].color = color;
}

inline uint write_text_string(
    device TextCharGpu* chars,
    uint start_idx,
    float x,
    float y,
    constant char* str,
    uint max_len,
    float scale,
    uint color
) {
    float spacing = 8.0 * scale;
    float cx = x;
    uint count = 0;

    for (uint i = 0; i < max_len && str[i] != 0; i++) {
        char c = str[i];
        if (c >= 32 && c <= 126) {
            write_text_char(chars, start_idx + count, cx, y, c, color);
            cx += spacing;
            count++;
        }
    }

    return count;
}
```

### Helper: Point in Rect

```metal
inline bool point_in_rect(float2 point, float2 rect_pos, float2 rect_size) {
    return point.x >= rect_pos.x && point.x < rect_pos.x + rect_size.x &&
           point.y >= rect_pos.y && point.y < rect_pos.y + rect_size.y;
}
```

### Main MenuBar Update Kernel

```metal
inline void menubar_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device InputEvent* input_events,
    uint input_count,
    uint tid,
    uint tg_size
) {
    // Get state pointers
    device MenuBarState* state = (device MenuBarState*)(unified_state + app->state_offset);
    device MenuItem* menus = (device MenuItem*)((device uchar*)state + sizeof(MenuBarState));
    device DropdownItem* items = (device DropdownItem*)((device uchar*)menus + MAX_MENUS * sizeof(MenuItem));
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));

    // ─────────────────────────────────────────────────────────────────
    // PHASE 1: Input Processing (Thread 0 only)
    // ─────────────────────────────────────────────────────────────────
    if (tid == 0) {
        // Reset per-frame state
        state->selected_item = UINT_MAX_VAL;
        state->mouse_clicked = 0;

        // Process input events from queue
        for (uint i = 0; i < input_count; i++) {
            InputEvent ev = input_events[i];

            if (ev.event_type == EVENT_MOUSE_MOVE) {
                state->mouse_pos = float2(ev.position[0], ev.position[1]);
            }
            else if (ev.event_type == EVENT_MOUSE_DOWN) {
                state->mouse_pos = float2(ev.position[0], ev.position[1]);
                state->mouse_down = 1;
                state->mouse_clicked = 1;
            }
            else if (ev.event_type == EVENT_MOUSE_UP) {
                state->mouse_down = 0;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ─────────────────────────────────────────────────────────────────
    // PHASE 2: Parallel Hover Detection
    // One thread per menu title
    // ─────────────────────────────────────────────────────────────────
    threadgroup atomic_uint hovered_menu_atomic;
    threadgroup atomic_uint hovered_item_atomic;

    if (tid == 0) {
        atomic_store_explicit(&hovered_menu_atomic, UINT_MAX_VAL, memory_order_relaxed);
        atomic_store_explicit(&hovered_item_atomic, UINT_MAX_VAL, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread tests one menu title
    if (tid < state->menu_count) {
        MenuItem menu = menus[tid];
        float2 menu_rect_pos = float2(menu.x, 0);
        float2 menu_rect_size = float2(menu.width, state->bar_height);

        if (point_in_rect(state->mouse_pos, menu_rect_pos, menu_rect_size)) {
            atomic_store_explicit(&hovered_menu_atomic, tid, memory_order_relaxed);
        }
    }

    // If dropdown is open, test dropdown items in parallel
    if (state->open_menu != UINT_MAX_VAL) {
        MenuItem open_menu = menus[state->open_menu];
        uint item_base = open_menu.first_item;

        // Thread tests item if within range
        if (tid < open_menu.item_count) {
            uint item_idx = item_base + tid;
            float dropdown_x = open_menu.x;
            float dropdown_y = state->bar_height + tid * 24.0;
            float2 item_rect_pos = float2(dropdown_x, dropdown_y);
            float2 item_rect_size = float2(200.0, 24.0);

            if (point_in_rect(state->mouse_pos, item_rect_pos, item_rect_size)) {
                atomic_store_explicit(&hovered_item_atomic, item_idx, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ─────────────────────────────────────────────────────────────────
    // PHASE 3: State Update (Thread 0)
    // ─────────────────────────────────────────────────────────────────
    if (tid == 0) {
        state->hovered_menu = atomic_load_explicit(&hovered_menu_atomic, memory_order_relaxed);
        state->hovered_item = atomic_load_explicit(&hovered_item_atomic, memory_order_relaxed);

        // Handle clicks
        if (state->mouse_clicked) {
            // Click on menu title - toggle dropdown
            if (state->hovered_menu != UINT_MAX_VAL) {
                if (state->open_menu == state->hovered_menu) {
                    state->open_menu = UINT_MAX_VAL;  // Close
                } else {
                    state->open_menu = state->hovered_menu;  // Open
                }
            }
            // Click on dropdown item - trigger action
            else if (state->hovered_item != UINT_MAX_VAL) {
                state->selected_item = state->hovered_item;
                state->open_menu = UINT_MAX_VAL;  // Close dropdown
            }
            // Click outside - close dropdown
            else {
                state->open_menu = UINT_MAX_VAL;
            }
        }

        // Update animation
        if (state->open_menu != UINT_MAX_VAL) {
            state->dropdown_anim = min(state->dropdown_anim + 0.2, 1.0);
        } else {
            state->dropdown_anim = max(state->dropdown_anim - 0.2, 0.0);
        }

        state->time += 0.016;  // ~60fps
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ─────────────────────────────────────────────────────────────────
    // PHASE 4: Compute Menu Positions (Thread 0)
    // ─────────────────────────────────────────────────────────────────
    if (tid == 0) {
        float x = state->padding_x;
        float char_width = 8.0 * state->text_scale;

        for (uint i = 0; i < state->menu_count; i++) {
            menus[i].x = x;

            // Calculate width based on name length
            uint name_len = 0;
            for (uint j = 0; j < MENU_NAME_LEN && menus[i].name[j] != 0; j++) {
                name_len++;
            }
            menus[i].width = name_len * char_width + state->padding_x * 2;

            x += menus[i].width;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ─────────────────────────────────────────────────────────────────
    // PHASE 5: Parallel Vertex Generation
    // ─────────────────────────────────────────────────────────────────

    // Thread 0: Background and dropdown background
    if (tid == 0) {
        uint vidx = 0;

        // Menu bar background (translucent)
        write_quad(verts + vidx, float2(0, 0), float2(state->screen_width, state->bar_height),
                   0.99, float4(0.95, 0.95, 0.97, 0.92));
        vidx += 6;

        // Highlight for hovered/open menu
        if (state->hovered_menu != UINT_MAX_VAL || state->open_menu != UINT_MAX_VAL) {
            uint highlight_idx = (state->open_menu != UINT_MAX_VAL)
                ? state->open_menu : state->hovered_menu;
            MenuItem highlight_menu = menus[highlight_idx];

            write_quad(verts + vidx,
                       float2(highlight_menu.x, 2),
                       float2(highlight_menu.width, state->bar_height - 4),
                       0.985, float4(0.3, 0.5, 0.9, 0.3));
            vidx += 6;
        }

        // Open dropdown background
        if (state->open_menu != UINT_MAX_VAL && state->dropdown_anim > 0.01) {
            MenuItem open_menu = menus[state->open_menu];
            float dropdown_height = open_menu.item_count * 24.0 * state->dropdown_anim;

            // Dropdown shadow
            write_quad(verts + vidx,
                       float2(open_menu.x + 4, state->bar_height + 4),
                       float2(200.0, dropdown_height),
                       0.96, float4(0.0, 0.0, 0.0, 0.2));
            vidx += 6;

            // Dropdown background
            write_quad(verts + vidx,
                       float2(open_menu.x, state->bar_height),
                       float2(200.0, dropdown_height),
                       0.97, float4(1.0, 1.0, 1.0, 0.98));
            vidx += 6;

            // Hovered item highlight
            if (state->hovered_item != UINT_MAX_VAL) {
                uint rel_idx = state->hovered_item - open_menu.first_item;
                if (rel_idx < open_menu.item_count) {
                    write_quad(verts + vidx,
                               float2(open_menu.x + 2, state->bar_height + rel_idx * 24.0 + 2),
                               float2(196.0, 20.0),
                               0.975, float4(0.3, 0.5, 0.9, 0.5));
                    vidx += 6;
                }
            }
        }

        app->vertex_count = vidx;
    }

    // Note: Text is rendered separately via TextRenderer
    // Menu titles and dropdown item text use the bitmap font system from text_render.rs
    // The megakernel updates state; rendering passes the text chars to TextRenderer
}
```

### Text Generation (Separate Pass)

```metal
// Called after menubar_update to generate text characters
// Uses the same TextChar format as text_render.rs

kernel void menubar_text_generate(
    device MenuBarState* state [[buffer(0)]],
    device MenuItem* menus [[buffer(1)]],
    device DropdownItem* items [[buffer(2)]],
    device TextCharGpu* text_chars [[buffer(3)]],
    device atomic_uint* char_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Each thread handles one menu title
    if (tid < state->menu_count) {
        MenuItem menu = menus[tid];

        // Calculate starting char position atomically
        uint name_len = 0;
        for (uint i = 0; i < MENU_NAME_LEN && menu.name[i] != 0; i++) {
            name_len++;
        }

        uint char_base = atomic_fetch_add_explicit(char_count, name_len, memory_order_relaxed);

        // Generate text characters
        float x = menu.x + state->padding_x;
        float y = (state->bar_height - 8.0 * state->text_scale) / 2.0;
        uint color = (tid == state->hovered_menu || tid == state->open_menu)
            ? 0x000000FF : 0x333333FF;

        for (uint i = 0; i < name_len; i++) {
            write_text_char(text_chars, char_base + i, x, y, menu.name[i], color);
            x += 8.0 * state->text_scale;
        }
    }

    // Generate dropdown item text (if open)
    if (state->open_menu != UINT_MAX_VAL && state->dropdown_anim > 0.5) {
        MenuItem open_menu = menus[state->open_menu];

        // Thread handles one dropdown item
        if (tid < open_menu.item_count) {
            DropdownItem item = items[open_menu.first_item + tid];

            uint name_len = 0;
            for (uint i = 0; i < ITEM_NAME_LEN && item.name[i] != 0; i++) {
                name_len++;
            }

            uint char_base = atomic_fetch_add_explicit(char_count, name_len, memory_order_relaxed);

            float x = open_menu.x + 8.0;
            float y = state->bar_height + tid * 24.0 + 4.0;
            uint color = ((open_menu.first_item + tid) == state->hovered_item)
                ? 0xFFFFFFFF : 0x000000FF;

            for (uint i = 0; i < name_len; i++) {
                write_text_char(text_chars, char_base + i, x, y, item.name[i], color);
                x += 8.0 * state->text_scale;
            }
        }
    }
}
```

## Rust API

```rust
impl GpuOs {
    /// Initialize menu bar with default configuration
    pub fn init_menubar(&mut self, screen_width: f32) {
        let state = self.get_menubar_state_mut();
        state.screen_width = screen_width;
        state.bar_height = 24.0;
        state.padding_x = 12.0;
        state.text_scale = 1.5;
        state.menu_count = 0;
        state.total_item_count = 0;
        state.open_menu = u32::MAX;
        state.hovered_menu = u32::MAX;
        state.hovered_item = u32::MAX;
        state.selected_item = u32::MAX;
    }

    /// Add a menu to the menu bar
    pub fn add_menu(&mut self, name: &str, items: &[MenuItemDef]) {
        unsafe {
            let state = self.get_menubar_state_mut();
            let menus = self.get_menus_mut();
            let dropdown_items = self.get_dropdown_items_mut();

            let menu_idx = state.menu_count as usize;
            let first_item = state.total_item_count as usize;

            // Initialize menu entry
            menus[menu_idx] = MenuItem {
                flags: MENU_FLAG_VISIBLE | MENU_FLAG_ENABLED,
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
            state.total_item_count += items.len() as u32;
        }
    }

    /// Check if a menu action was triggered this frame
    pub fn poll_menu_action(&self) -> Option<u32> {
        let state = self.get_menubar_state();
        if state.selected_item != u32::MAX {
            let items = self.get_dropdown_items();
            Some(items[state.selected_item as usize].action_id)
        } else {
            None
        }
    }

    /// Get the menu bar height (for laying out other UI)
    pub fn menubar_height(&self) -> f32 {
        self.get_menubar_state().bar_height
    }
}

/// Menu item definition for API
pub struct MenuItemDef {
    pub name: String,
    pub action_id: u32,
    pub flags: u32,
    pub shortcut_key: u32,
    pub shortcut_mods: u32,
}

impl MenuItemDef {
    pub fn new(name: &str, action_id: u32) -> Self {
        Self {
            name: name.to_string(),
            action_id,
            flags: ITEM_FLAG_ENABLED,
            shortcut_key: 0,
            shortcut_mods: 0,
        }
    }

    pub fn with_shortcut(mut self, key: u32, mods: u32) -> Self {
        self.shortcut_key = key;
        self.shortcut_mods = mods;
        self
    }

    pub fn separator() -> Self {
        Self {
            name: String::new(),
            action_id: 0,
            flags: ITEM_FLAG_SEPARATOR,
            shortcut_key: 0,
            shortcut_mods: 0,
        }
    }

    pub fn checked(mut self) -> Self {
        self.flags |= ITEM_FLAG_CHECKED;
        self
    }

    pub fn disabled(mut self) -> Self {
        self.flags &= !ITEM_FLAG_ENABLED;
        self
    }
}
```

## Tests

```rust
use crate::gpu_os::*;
use metal::Device;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

#[test]
fn test_menubar_launches_as_app() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).unwrap();

    let menubar_slot = system.launch_by_type(app_type::MENUBAR);
    assert!(menubar_slot.is_some(), "MenuBar should launch successfully");

    let app = system.get_app(menubar_slot.unwrap()).unwrap();
    assert_eq!(app.app_type, app_type::MENUBAR);
    assert!(app.flags & flags::ACTIVE != 0, "MenuBar should be active");
}

#[test]
fn test_menubar_state_initialization() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);

    let state = os.get_menubar_state();
    assert_eq!(state.screen_width, 1920.0);
    assert_eq!(state.bar_height, 24.0);
    assert_eq!(state.menu_count, 0);
    assert_eq!(state.open_menu, u32::MAX);
    assert_eq!(state.hovered_menu, u32::MAX);
}

#[test]
fn test_add_single_menu() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    os.add_menu("File", &[
        MenuItemDef::new("New", ACTION_NEW).with_shortcut(KEY_N, MOD_CMD),
        MenuItemDef::new("Open", ACTION_OPEN).with_shortcut(KEY_O, MOD_CMD),
        MenuItemDef::separator(),
        MenuItemDef::new("Quit", ACTION_QUIT).with_shortcut(KEY_Q, MOD_CMD),
    ]);

    let state = os.get_menubar_state();
    assert_eq!(state.menu_count, 1);
    assert_eq!(state.total_item_count, 4);

    let menus = os.get_menus();
    assert_eq!(menus[0].item_count, 4);
    assert_eq!(menus[0].first_item, 0);

    let items = os.get_dropdown_items();
    assert_eq!(items[0].action_id, ACTION_NEW);
    assert_eq!(items[3].action_id, ACTION_QUIT);
}

#[test]
fn test_add_multiple_menus() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);

    os.add_menu("File", &[
        MenuItemDef::new("New", ACTION_NEW),
        MenuItemDef::new("Open", ACTION_OPEN),
    ]);

    os.add_menu("Edit", &[
        MenuItemDef::new("Cut", ACTION_CUT),
        MenuItemDef::new("Copy", ACTION_COPY),
        MenuItemDef::new("Paste", ACTION_PASTE),
    ]);

    let state = os.get_menubar_state();
    assert_eq!(state.menu_count, 2);
    assert_eq!(state.total_item_count, 5);

    let menus = os.get_menus();
    assert_eq!(menus[0].first_item, 0);
    assert_eq!(menus[0].item_count, 2);
    assert_eq!(menus[1].first_item, 2);
    assert_eq!(menus[1].item_count, 3);
}

#[test]
fn test_hover_detection() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    os.add_menu("File", &[MenuItemDef::new("New", ACTION_NEW)]);

    // Run frame to compute menu positions
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    let menus = os.get_menus();
    let menu_center_x = menus[0].x + menus[0].width / 2.0;

    // Move mouse over menu
    os.system.queue_input(InputEvent::mouse_move(menu_center_x, 12.0, 0.0, 0.0));
    os.system.process_input();
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    let state = os.get_menubar_state();
    assert_eq!(state.hovered_menu, 0, "Should detect hover over first menu");
}

#[test]
fn test_menu_click_opens_dropdown() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    os.add_menu("File", &[
        MenuItemDef::new("New", ACTION_NEW),
        MenuItemDef::new("Open", ACTION_OPEN),
    ]);

    // Compute menu positions
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    let menus = os.get_menus();
    let menu_center_x = menus[0].x + menus[0].width / 2.0;

    // Click on menu
    os.system.queue_input(InputEvent::mouse_click(menu_center_x, 12.0, 0));
    os.system.process_input();
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    let state = os.get_menubar_state();
    assert_eq!(state.open_menu, 0, "Clicking menu should open dropdown");
}

#[test]
fn test_dropdown_item_click_triggers_action() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    os.add_menu("File", &[
        MenuItemDef::new("Quit", ACTION_QUIT),
    ]);

    // Open the menu
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();
    let menus = os.get_menus();
    let menu_x = menus[0].x + menus[0].width / 2.0;

    os.system.queue_input(InputEvent::mouse_click(menu_x, 12.0, 0));
    os.system.process_input();
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    // Click on dropdown item (24px below menu bar, at menu x position)
    let item_y = 24.0 + 12.0;  // bar_height + half item height
    os.system.queue_input(InputEvent::mouse_click(menu_x, item_y, 0));
    os.system.process_input();
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    // Check action was triggered
    let action = os.poll_menu_action();
    assert_eq!(action, Some(ACTION_QUIT), "Clicking item should trigger action");
}

#[test]
fn test_click_outside_closes_dropdown() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    os.add_menu("File", &[MenuItemDef::new("New", ACTION_NEW)]);

    // Open menu
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();
    let menus = os.get_menus();
    let menu_x = menus[0].x + menus[0].width / 2.0;

    os.system.queue_input(InputEvent::mouse_click(menu_x, 12.0, 0));
    os.system.process_input();
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    assert_eq!(os.get_menubar_state().open_menu, 0);

    // Click outside (far right of screen)
    os.system.queue_input(InputEvent::mouse_click(1800.0, 200.0, 0));
    os.system.process_input();
    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    assert_eq!(os.get_menubar_state().open_menu, u32::MAX, "Should close on outside click");
}

#[test]
fn test_vertex_generation() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    os.add_menu("File", &[
        MenuItemDef::new("New", ACTION_NEW),
        MenuItemDef::new("Open", ACTION_OPEN),
    ]);
    os.add_menu("Edit", &[
        MenuItemDef::new("Cut", ACTION_CUT),
    ]);

    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    let app = os.system.get_app(os.menubar_slot).unwrap();

    // Should have at least background quad (6 vertices)
    assert!(app.vertex_count >= 6, "Should generate background vertices");
}

#[test]
fn test_parallel_hover_detection_correctness() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);

    // Add many menus to test parallelism
    for i in 0..8 {
        os.add_menu(&format!("Menu{}", i), &[
            MenuItemDef::new("Item", 100 + i),
        ]);
    }

    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    // Test hovering each menu
    let menus = os.get_menus();
    for i in 0..8 {
        let menu_x = menus[i].x + menus[i].width / 2.0;

        os.system.queue_input(InputEvent::mouse_move(menu_x, 12.0, 0.0, 0.0));
        os.system.process_input();
        os.system.mark_dirty(os.menubar_slot);
        os.system.run_frame();

        let state = os.get_menubar_state();
        assert_eq!(state.hovered_menu, i as u32, "Should detect hover on menu {}", i);
    }
}
```

## Benchmarks

```rust
#[test]
fn bench_menubar_render() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);

    // Add typical macOS menus
    os.add_menu("File", &file_menu_items());     // ~12 items
    os.add_menu("Edit", &edit_menu_items());     // ~10 items
    os.add_menu("View", &view_menu_items());     // ~8 items
    os.add_menu("Window", &window_menu_items()); // ~6 items
    os.add_menu("Help", &help_menu_items());     // ~4 items

    // Warm up
    for _ in 0..10 {
        os.system.mark_dirty(os.menubar_slot);
        os.system.run_frame();
    }

    // Benchmark closed state
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        os.system.mark_dirty(os.menubar_slot);
        os.system.run_frame();
    }
    let closed_time = start.elapsed().as_micros() / 1000;

    // Open a menu with many items
    let menus = os.get_menus();
    os.system.queue_input(InputEvent::mouse_click(menus[0].x + 10.0, 12.0, 0));
    os.system.process_input();
    os.system.run_frame();

    // Benchmark open state
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        os.system.mark_dirty(os.menubar_slot);
        os.system.run_frame();
    }
    let open_time = start.elapsed().as_micros() / 1000;

    println!("MenuBar (closed): {}us/frame", closed_time);
    println!("MenuBar (open):   {}us/frame", open_time);

    assert!(closed_time < 100, "Closed menubar should render in <100us");
    assert!(open_time < 200, "Open menubar should render in <200us");
}

#[test]
fn bench_hover_detection_latency() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).unwrap();

    os.init_menubar(1920.0);
    for i in 0..16 {
        os.add_menu(&format!("M{}", i), &[MenuItemDef::new("Item", i)]);
    }

    os.system.mark_dirty(os.menubar_slot);
    os.system.run_frame();

    let start = std::time::Instant::now();
    for i in 0..1000 {
        let x = (i % 16) as f32 * 60.0 + 30.0;
        os.system.queue_input(InputEvent::mouse_move(x, 12.0, 0.0, 0.0));
        os.system.process_input();
        os.system.mark_dirty(os.menubar_slot);
        os.system.run_frame();
    }
    let total = start.elapsed();

    let per_frame = total.as_micros() / 1000;
    println!("Hover detection: {}us/frame", per_frame);
    assert!(per_frame < 500, "Hover detection should complete in <500us");
}
```

## Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| CPU per frame | 0 | All logic in GPU megakernel |
| Hover latency | < 16ms | One frame at 60fps |
| Dropdown open | < 16ms | Smooth animation start |
| Memory | Fixed 8KB | No dynamic allocation |
| Max menus | 16 | Sufficient for typical apps |
| Max items | 256 | ~16 items per menu average |

## Implementation Notes

1. **Text rendering integration**: The MenuBar generates `TextChar` structs that are passed to the existing `TextRenderer` from `text_render.rs`. This avoids duplicating the bitmap font rendering code.

2. **Animation**: Dropdown animation uses `dropdown_anim` to interpolate between closed (0.0) and open (1.0). The animation runs at ~60fps increments (0.2 per frame = ~5 frames to fully open).

3. **Depth ordering**: Menu bar uses depth 0.99, dropdown shadow 0.96, dropdown background 0.97, highlights 0.975-0.985. This ensures proper z-ordering without explicit depth sorting.

4. **Hit testing parallelism**: Each thread tests exactly one menu item, then one dropdown item. This maps well to GPU SIMD - all threads do the same work on different data.

5. **State update synchronization**: Uses `threadgroup_barrier` between phases to ensure all threads see consistent state. Atomic operations for hover detection allow parallel writes without races.
