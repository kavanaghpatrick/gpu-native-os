# PRD: Window Chrome as Megakernel App (Issue #159)

## Goal

Generate all window decorations (title bars, traffic light buttons, borders, resize handles) entirely on GPU in parallel. Each GPU thread generates chrome for one window simultaneously, eliminating per-window CPU loops.

**CPU involvement**: Zero during steady state. CPU only handles window close/minimize/maximize *actions* after GPU detects button clicks.

## Overview

Window chrome is geometry. Instead of CPU iterating through windows and drawing decorations, a single GPU dispatch launches N threads (one per window), and each thread writes vertices for its window's chrome to a unified vertex buffer.

```
CPU Pattern (WRONG):               GPU Pattern (RIGHT):
for window in windows:              thread 0 -> window 0 chrome
    draw_title_bar(window)          thread 1 -> window 1 chrome
    draw_buttons(window)            thread 2 -> window 2 chrome
    draw_border(window)             ...all simultaneously...
```

## Existing Infrastructure to REUSE

| Need | Infrastructure | File | Notes |
|------|---------------|------|-------|
| Window positions | `GpuWindow` struct | `gpu_app_system.rs` L462-473 | x, y, width, height, depth, flags |
| Hit testing | `HitTestResult` + regions | `event_loop.rs` L128-161 | CLOSE, MINIMIZE, MAXIMIZE regions |
| Event dispatch | `GpuEventLoopState` | `event_loop.rs` L63-97 | drag_window, resize_window, focused_window |
| Vertex buffer | `RenderVertex` | `gpu_app_system.rs` L506-514 | position, color, uv |
| App descriptor | `GpuAppDescriptor` | `gpu_app_system.rs` L181-220 | vertex_offset, vertex_count, state_offset |
| Window flags | `window_flags` module | `event_loop.rs` L182-187 | VISIBLE, MINIMIZED, FOCUSED |
| Parallel update | `dispatch_app_update()` | `gpu_app_system.rs` L1277-1309 | Routes to app-specific update |

## GPU-Native Design Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| CPU draws each window's chrome | GPU generates all chrome in parallel |
| Per-window for-loop | One thread per window |
| CPU hit-tests buttons | GPU parallel hit test via `HitTestResult` |
| Separate chrome renderer | Writes to unified vertex buffer |
| CPU tracks hover state | GPU atomics for hovered_button |

## WindowChromeState Structure

The chrome app maintains state for all windows (not per-window state):

```metal
// Constants
constant uint CHROME_TITLE_BAR = 6;       // 6 vertices (quad)
constant uint CHROME_BUTTON_VERTS = 18;    // 18 vertices (6-triangle circle approx)
constant uint CHROME_BORDER_VERTS = 24;    // 4 borders x 6 vertices each
constant uint CHROME_RESIZE_HANDLE = 6;    // 6 vertices (quad)
constant uint CHROME_VERTS_PER_WINDOW =
    CHROME_TITLE_BAR + CHROME_BUTTON_VERTS * 3 + CHROME_BORDER_VERTS + CHROME_RESIZE_HANDLE;
// = 6 + 54 + 24 + 6 = 90 vertices per window

// Button indices encoded as (window_idx << 8) | button_type
constant uint BUTTON_CLOSE = 0;
constant uint BUTTON_MINIMIZE = 1;
constant uint BUTTON_MAXIMIZE = 2;
constant uint BUTTON_NONE = 0xFF;
constant uint WINDOW_NONE = 0xFFFFFFFF;

struct WindowChromeState {
    // Window tracking
    uint window_count;
    uint focused_window;

    // Interaction state
    uint dragging_window;      // Window being dragged (WINDOW_NONE if none)
    uint resizing_window;      // Window being resized (WINDOW_NONE if none)
    uint hovered_button;       // Encoded: (window_idx << 8) | button_type
    uint clicked_button;       // Set by GPU, cleared by CPU after action
    float2 drag_offset;        // Mouse offset from window origin
    float2 resize_origin;      // Original size at resize start

    // Chrome dimensions (consistent across all windows)
    float title_bar_height;    // Default: 28.0
    float border_width;        // Default: 1.0
    float button_radius;       // Default: 6.0
    float button_spacing;      // Default: 8.0 (gap between buttons)
    float button_left_margin;  // Default: 12.0 (from left edge to first button)
    float corner_radius;       // Default: 10.0 (for rounded corners, future)

    // Colors (RGBA)
    float4 title_focused;      // Default: (0.9, 0.9, 0.9, 1.0) - light gray
    float4 title_unfocused;    // Default: (0.7, 0.7, 0.7, 1.0) - darker gray
    float4 close_color;        // Default: (1.0, 0.38, 0.36, 1.0) - red
    float4 minimize_color;     // Default: (1.0, 0.76, 0.03, 1.0) - yellow
    float4 maximize_color;     // Default: (0.15, 0.78, 0.38, 1.0) - green
    float4 border_color;       // Default: (0.6, 0.6, 0.6, 1.0) - medium gray
    float4 button_hover_tint;  // Default: (1.0, 1.0, 1.0, 0.3) - brighten on hover

    uint _pad[2];  // Align to 16 bytes
};
```

### Rust-side Structure

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WindowChromeState {
    pub window_count: u32,
    pub focused_window: u32,
    pub dragging_window: u32,
    pub resizing_window: u32,
    pub hovered_button: u32,
    pub clicked_button: u32,
    pub drag_offset: [f32; 2],
    pub resize_origin: [f32; 2],

    pub title_bar_height: f32,
    pub border_width: f32,
    pub button_radius: f32,
    pub button_spacing: f32,
    pub button_left_margin: f32,
    pub corner_radius: f32,
    pub _dim_pad: [f32; 2],

    pub title_focused: [f32; 4],
    pub title_unfocused: [f32; 4],
    pub close_color: [f32; 4],
    pub minimize_color: [f32; 4],
    pub maximize_color: [f32; 4],
    pub border_color: [f32; 4],
    pub button_hover_tint: [f32; 4],

    pub _pad: [u32; 2],
}

impl Default for WindowChromeState {
    fn default() -> Self {
        Self {
            window_count: 0,
            focused_window: u32::MAX,
            dragging_window: u32::MAX,
            resizing_window: u32::MAX,
            hovered_button: u32::MAX,
            clicked_button: u32::MAX,
            drag_offset: [0.0, 0.0],
            resize_origin: [0.0, 0.0],

            title_bar_height: 28.0,
            border_width: 1.0,
            button_radius: 6.0,
            button_spacing: 8.0,
            button_left_margin: 12.0,
            corner_radius: 10.0,
            _dim_pad: [0.0, 0.0],

            title_focused: [0.9, 0.9, 0.9, 1.0],
            title_unfocused: [0.7, 0.7, 0.7, 1.0],
            close_color: [1.0, 0.38, 0.36, 1.0],
            minimize_color: [1.0, 0.76, 0.03, 1.0],
            maximize_color: [0.15, 0.78, 0.38, 1.0],
            border_color: [0.6, 0.6, 0.6, 1.0],
            button_hover_tint: [1.0, 1.0, 1.0, 0.3],

            _pad: [0, 0],
        }
    }
}
```

## Metal Shader Implementation

### Vertex Generation Utilities

```metal
// Write a quad (2 triangles = 6 vertices)
inline void write_quad(
    device RenderVertex* v,
    float2 origin,      // top-left corner
    float2 size,        // width, height
    float depth,
    float4 color
) {
    // Triangle 1: top-left, top-right, bottom-right
    v[0].position = float3(origin.x, origin.y, depth);
    v[0].color = color;
    v[0].uv = float2(0, 0);

    v[1].position = float3(origin.x + size.x, origin.y, depth);
    v[1].color = color;
    v[1].uv = float2(1, 0);

    v[2].position = float3(origin.x + size.x, origin.y + size.y, depth);
    v[2].color = color;
    v[2].uv = float2(1, 1);

    // Triangle 2: top-left, bottom-right, bottom-left
    v[3].position = float3(origin.x, origin.y, depth);
    v[3].color = color;
    v[3].uv = float2(0, 0);

    v[4].position = float3(origin.x + size.x, origin.y + size.y, depth);
    v[4].color = color;
    v[4].uv = float2(1, 1);

    v[5].position = float3(origin.x, origin.y + size.y, depth);
    v[5].color = color;
    v[5].uv = float2(0, 1);
}

// Write a circle approximation (6 triangles = 18 vertices)
inline void write_circle(
    device RenderVertex* v,
    float2 center,
    float radius,
    float depth,
    float4 color
) {
    // 6 triangles approximating a circle
    // Each triangle: center, edge[i], edge[i+1]
    for (uint i = 0; i < 6; i++) {
        float angle1 = float(i) * M_PI_F / 3.0;
        float angle2 = float(i + 1) * M_PI_F / 3.0;

        float2 p1 = center + radius * float2(cos(angle1), sin(angle1));
        float2 p2 = center + radius * float2(cos(angle2), sin(angle2));

        uint base = i * 3;
        v[base + 0].position = float3(center.x, center.y, depth);
        v[base + 0].color = color;
        v[base + 0].uv = float2(0.5, 0.5);

        v[base + 1].position = float3(p1.x, p1.y, depth);
        v[base + 1].color = color;
        v[base + 1].uv = float2(0.5 + 0.5 * cos(angle1), 0.5 + 0.5 * sin(angle1));

        v[base + 2].position = float3(p2.x, p2.y, depth);
        v[base + 2].color = color;
        v[base + 2].uv = float2(0.5 + 0.5 * cos(angle2), 0.5 + 0.5 * sin(angle2));
    }
}

// Write resize handle (diagonal lines pattern) - simplified as quad with distinct color
inline void write_resize_handle(
    device RenderVertex* v,
    float2 origin,
    float size,
    float depth,
    float4 color
) {
    write_quad(v, origin, float2(size, size), depth, color);
}
```

### Main Chrome Update Function

```metal
// Window Chrome update - parallel vertex generation
// Each thread generates chrome for one window
inline void window_chrome_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    uint window_count,
    uint tid,
    uint tg_size
) {
    device WindowChromeState* state = (device WindowChromeState*)(unified_state + app->state_offset);

    // Thread 0 updates window count
    if (tid == 0) {
        state->window_count = window_count;
    }

    // Early exit for threads beyond window count
    if (tid >= window_count) {
        if (tid == 0) {
            app->vertex_count = window_count * CHROME_VERTS_PER_WINDOW;
        }
        return;
    }

    GpuWindow window = windows[tid];

    // Skip invisible windows
    if (!(window.flags & WINDOW_VISIBLE)) {
        return;
    }

    // Get vertex buffer offset for this window
    device RenderVertex* verts = unified_vertices + (app->vertex_offset / sizeof(RenderVertex));
    uint base = tid * CHROME_VERTS_PER_WINDOW;

    // Determine if this window is focused
    bool is_focused = (tid == state->focused_window);
    float4 title_color = is_focused ? state->title_focused : state->title_unfocused;
    float depth = window.depth + 0.001;  // Chrome slightly in front of content

    uint vert_idx = 0;

    // =========================================================================
    // 1. TITLE BAR (6 vertices)
    // =========================================================================
    // Title bar sits above the content area
    float title_y = window.y - state->title_bar_height;
    write_quad(
        verts + base + vert_idx,
        float2(window.x, title_y),
        float2(window.width, state->title_bar_height),
        depth,
        title_color
    );
    vert_idx += 6;

    // =========================================================================
    // 2. TRAFFIC LIGHT BUTTONS (3 x 18 = 54 vertices)
    // =========================================================================
    float btn_center_y = title_y + state->title_bar_height / 2.0;
    float btn_x = window.x + state->button_left_margin;

    // Hover state: check if this window's button is hovered
    uint hovered_window_idx = (state->hovered_button >> 8);
    uint hovered_button_type = (state->hovered_button & 0xFF);

    // Close button (red)
    float4 close_col = state->close_color;
    if (hovered_window_idx == tid && hovered_button_type == BUTTON_CLOSE) {
        close_col = close_col + state->button_hover_tint;
    }
    write_circle(
        verts + base + vert_idx,
        float2(btn_x, btn_center_y),
        state->button_radius,
        depth,
        close_col
    );
    vert_idx += 18;

    // Minimize button (yellow)
    btn_x += state->button_radius * 2.0 + state->button_spacing;
    float4 minimize_col = state->minimize_color;
    if (hovered_window_idx == tid && hovered_button_type == BUTTON_MINIMIZE) {
        minimize_col = minimize_col + state->button_hover_tint;
    }
    write_circle(
        verts + base + vert_idx,
        float2(btn_x, btn_center_y),
        state->button_radius,
        depth,
        minimize_col
    );
    vert_idx += 18;

    // Maximize button (green)
    btn_x += state->button_radius * 2.0 + state->button_spacing;
    float4 maximize_col = state->maximize_color;
    if (hovered_window_idx == tid && hovered_button_type == BUTTON_MAXIMIZE) {
        maximize_col = maximize_col + state->button_hover_tint;
    }
    write_circle(
        verts + base + vert_idx,
        float2(btn_x, btn_center_y),
        state->button_radius,
        depth,
        maximize_col
    );
    vert_idx += 18;

    // =========================================================================
    // 3. BORDERS (4 x 6 = 24 vertices)
    // =========================================================================
    float border_w = state->border_width;
    float full_height = window.height + state->title_bar_height;

    // Left border
    write_quad(
        verts + base + vert_idx,
        float2(window.x - border_w, title_y),
        float2(border_w, full_height),
        depth,
        state->border_color
    );
    vert_idx += 6;

    // Right border
    write_quad(
        verts + base + vert_idx,
        float2(window.x + window.width, title_y),
        float2(border_w, full_height),
        depth,
        state->border_color
    );
    vert_idx += 6;

    // Top border (above title bar)
    write_quad(
        verts + base + vert_idx,
        float2(window.x - border_w, title_y - border_w),
        float2(window.width + border_w * 2.0, border_w),
        depth,
        state->border_color
    );
    vert_idx += 6;

    // Bottom border
    write_quad(
        verts + base + vert_idx,
        float2(window.x - border_w, window.y + window.height),
        float2(window.width + border_w * 2.0, border_w),
        depth,
        state->border_color
    );
    vert_idx += 6;

    // =========================================================================
    // 4. RESIZE HANDLE (6 vertices)
    // =========================================================================
    float handle_size = 16.0;
    write_resize_handle(
        verts + base + vert_idx,
        float2(window.x + window.width - handle_size, window.y + window.height - handle_size),
        handle_size,
        depth,
        float4(0.5, 0.5, 0.5, 0.5)  // Semi-transparent gray
    );
    vert_idx += 6;

    // Thread 0 commits the total vertex count
    if (tid == 0) {
        app->vertex_count = window_count * CHROME_VERTS_PER_WINDOW;
    }
}
```

### Input Processing (Hit Testing and Interaction)

```metal
// Hit test for chrome elements - runs per-window in parallel
// Updates hovered_button and clicked_button atomically
inline void chrome_hit_test(
    device WindowChromeState* state,
    device GpuWindow* windows,
    float2 mouse_pos,
    uint window_count,
    uint tid,
    bool mouse_down
) {
    if (tid >= window_count) return;

    GpuWindow window = windows[tid];
    if (!(window.flags & WINDOW_VISIBLE)) return;

    float title_y = window.y - state->title_bar_height;
    float btn_center_y = title_y + state->title_bar_height / 2.0;

    // Check buttons (only if in title bar y-range)
    if (mouse_pos.y >= title_y && mouse_pos.y < window.y) {
        float btn_x = window.x + state->button_left_margin;

        // Close button
        if (distance(mouse_pos, float2(btn_x, btn_center_y)) < state->button_radius) {
            uint encoded = (tid << 8) | BUTTON_CLOSE;
            // Atomic max so highest z-order window wins
            // Using window.z_order as priority
            // For simplicity, we can use non-atomic since one thread = one window
            state->hovered_button = encoded;
            if (mouse_down) {
                state->clicked_button = encoded;
            }
            return;
        }

        // Minimize button
        btn_x += state->button_radius * 2.0 + state->button_spacing;
        if (distance(mouse_pos, float2(btn_x, btn_center_y)) < state->button_radius) {
            uint encoded = (tid << 8) | BUTTON_MINIMIZE;
            state->hovered_button = encoded;
            if (mouse_down) {
                state->clicked_button = encoded;
            }
            return;
        }

        // Maximize button
        btn_x += state->button_radius * 2.0 + state->button_spacing;
        if (distance(mouse_pos, float2(btn_x, btn_center_y)) < state->button_radius) {
            uint encoded = (tid << 8) | BUTTON_MAXIMIZE;
            state->hovered_button = encoded;
            if (mouse_down) {
                state->clicked_button = encoded;
            }
            return;
        }

        // In title bar but not on a button -> initiate drag
        if (mouse_pos.x >= window.x && mouse_pos.x < window.x + window.width) {
            if (mouse_down && state->dragging_window == WINDOW_NONE) {
                state->dragging_window = tid;
                state->drag_offset = mouse_pos - float2(window.x, window.y);
            }
        }
    }

    // Check resize handle (bottom-right corner)
    float handle_size = 16.0;
    if (mouse_pos.x >= window.x + window.width - handle_size &&
        mouse_pos.x < window.x + window.width &&
        mouse_pos.y >= window.y + window.height - handle_size &&
        mouse_pos.y < window.y + window.height) {
        if (mouse_down && state->resizing_window == WINDOW_NONE) {
            state->resizing_window = tid;
            state->resize_origin = float2(window.width, window.height);
        }
    }
}

// Process drag/resize during mouse move
inline void chrome_process_drag(
    device WindowChromeState* state,
    device GpuWindow* windows,
    float2 mouse_pos
) {
    // Handle window dragging
    if (state->dragging_window != WINDOW_NONE) {
        uint idx = state->dragging_window;
        windows[idx].x = mouse_pos.x - state->drag_offset.x;
        windows[idx].y = mouse_pos.y - state->drag_offset.y;
    }

    // Handle window resizing
    if (state->resizing_window != WINDOW_NONE) {
        uint idx = state->resizing_window;
        // Calculate new size based on mouse delta from resize start
        float new_width = max(200.0, state->resize_origin.x + (mouse_pos.x - (windows[idx].x + state->resize_origin.x)));
        float new_height = max(100.0, state->resize_origin.y + (mouse_pos.y - (windows[idx].y + state->resize_origin.y)));
        windows[idx].width = new_width;
        windows[idx].height = new_height;
    }
}

// Release drag/resize on mouse up
inline void chrome_process_mouse_up(device WindowChromeState* state) {
    state->dragging_window = WINDOW_NONE;
    state->resizing_window = WINDOW_NONE;
}
```

## Rust Integration

### Process Chrome Actions (CPU-side)

The CPU reads `clicked_button` after each frame and performs the action:

```rust
impl GpuAppSystem {
    /// Process chrome button clicks - called after frame
    pub fn process_chrome_actions(&mut self) {
        let chrome_slot = self.find_app_by_type(app_type::WINDOW_CHROME);
        if chrome_slot.is_none() {
            return;
        }
        let chrome_slot = chrome_slot.unwrap();

        let state = self.read_chrome_state(chrome_slot);

        if state.clicked_button != u32::MAX {
            let window_idx = (state.clicked_button >> 8) as u32;
            let button_type = (state.clicked_button & 0xFF) as u32;

            match button_type {
                BUTTON_CLOSE => self.close_window(window_idx),
                BUTTON_MINIMIZE => self.minimize_window(window_idx),
                BUTTON_MAXIMIZE => self.maximize_window(window_idx),
                _ => {}
            }

            // Clear clicked state
            self.clear_chrome_clicked_button(chrome_slot);
        }
    }

    fn read_chrome_state(&self, slot: u32) -> WindowChromeState {
        unsafe {
            let app = self.get_app_descriptor(slot).unwrap();
            let state_ptr = (self.unified_state_buffer.contents() as *const u8)
                .add(app.state_offset as usize) as *const WindowChromeState;
            *state_ptr
        }
    }

    fn clear_chrome_clicked_button(&mut self, slot: u32) {
        unsafe {
            let app = self.get_app_descriptor(slot).unwrap();
            let state_ptr = (self.unified_state_buffer.contents() as *mut u8)
                .add(app.state_offset as usize) as *mut WindowChromeState;
            (*state_ptr).clicked_button = u32::MAX;
        }
    }

    fn close_window(&mut self, window_idx: u32) {
        // Find app owning this window and terminate it
        if let Some(app_slot) = self.find_app_with_window(window_idx) {
            self.terminate_app(app_slot);
        }
    }

    fn minimize_window(&mut self, window_idx: u32) {
        unsafe {
            let windows = self.windows_buffer.contents() as *mut GpuWindow;
            (*windows.add(window_idx as usize)).flags |= window_flags::MINIMIZED;
        }
    }

    fn maximize_window(&mut self, window_idx: u32) {
        unsafe {
            let windows = self.windows_buffer.contents() as *mut GpuWindow;
            let window = &mut *windows.add(window_idx as usize);

            if window.flags & window_flags::MAXIMIZED != 0 {
                // Restore from maximized
                window.flags &= !window_flags::MAXIMIZED;
                // Restore saved bounds (would need to track these)
            } else {
                // Maximize to screen
                window.flags |= window_flags::MAXIMIZED;
                window.x = 0.0;
                window.y = 28.0; // Below menu bar
                window.width = self.screen_width;
                window.height = self.screen_height - 28.0;
            }
        }
    }
}
```

### App Registration

Register window chrome as a system app:

```rust
pub const SYSTEM_APPS: &[AppTemplate] = &[
    AppTemplate {
        type_id: app_type::WINDOW_CHROME,
        name: "Window Chrome",
        state_size: std::mem::size_of::<WindowChromeState>() as u32,
        vertex_size: 64 * CHROME_VERTS_PER_WINDOW * std::mem::size_of::<RenderVertex>() as u32,
        thread_count: 64,  // Max 64 windows
    },
    // ... other system apps
];

pub const CHROME_VERTS_PER_WINDOW: u32 = 90;  // 6 + 54 + 24 + 6
```

## Tests

### Basic Chrome Generation

```rust
#[test]
fn test_chrome_generates_vertices_for_each_window() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    // Launch 3 windows
    let slots: Vec<u32> = (0..3).map(|i| {
        let slot = system.launch_app(app_type::TERMINAL).unwrap();
        system.set_window(slot, 100.0 + i as f32 * 200.0, 100.0, 400.0, 300.0);
        slot
    }).collect();

    // Run frame to generate chrome
    system.run_frame();

    // Verify chrome app generated correct vertex count
    let chrome_slot = system.find_app_by_type(app_type::WINDOW_CHROME).unwrap();
    let chrome_app = system.get_app_descriptor(chrome_slot).unwrap();

    // 3 windows x 90 vertices each = 270
    assert_eq!(chrome_app.vertex_count, 3 * 90);
}
```

### Window Dragging

```rust
#[test]
fn test_window_drag() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    // Create a window
    let slot = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot, 100.0, 100.0, 400.0, 300.0);

    let initial_window = system.get_window(slot).unwrap();
    let title_bar_y = initial_window.y - 14.0; // Middle of 28px title bar

    // Mouse down on title bar
    system.queue_input(InputEvent::mouse_down(150.0, title_bar_y, 0));
    system.process_input();
    system.run_frame();

    // Drag to new position
    system.queue_input(InputEvent::mouse_move(250.0, title_bar_y));
    system.process_input();
    system.run_frame();

    // Verify window moved
    let moved_window = system.get_window(slot).unwrap();
    assert!(moved_window.x > initial_window.x, "Window should have moved right");

    // Mouse up
    system.queue_input(InputEvent::mouse_up(250.0, title_bar_y, 0));
    system.process_input();
    system.run_frame();

    // Verify drag state cleared
    let chrome_state = system.read_chrome_state(
        system.find_app_by_type(app_type::WINDOW_CHROME).unwrap()
    );
    assert_eq!(chrome_state.dragging_window, u32::MAX);
}
```

### Button Click Detection

```rust
#[test]
fn test_close_button_click() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    // Create a window
    let slot = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot, 100.0, 100.0, 400.0, 300.0);

    let window = system.get_window(slot).unwrap();

    // Close button is at x + 12, y - 14 (middle of title bar)
    let close_x = window.x + 12.0;
    let close_y = window.y - 14.0;

    // Click close button
    system.queue_input(InputEvent::mouse_down(close_x, close_y, 0));
    system.process_input();
    system.run_frame();

    // Process chrome actions
    system.process_chrome_actions();

    // Window/app should be closed
    assert!(system.get_app_descriptor(slot).is_none() ||
            !system.get_app_descriptor(slot).unwrap().is_active());
}

#[test]
fn test_minimize_button_click() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    let slot = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot, 100.0, 100.0, 400.0, 300.0);

    let window = system.get_window(slot).unwrap();

    // Minimize button is at x + 12 + (6*2) + 8 = x + 32
    let minimize_x = window.x + 32.0;
    let minimize_y = window.y - 14.0;

    system.queue_input(InputEvent::mouse_down(minimize_x, minimize_y, 0));
    system.process_input();
    system.run_frame();
    system.process_chrome_actions();

    let updated_window = system.get_window(slot).unwrap();
    assert!(updated_window.flags & window_flags::MINIMIZED != 0);
}

#[test]
fn test_maximize_button_toggles() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    let slot = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot, 100.0, 100.0, 400.0, 300.0);

    let window = system.get_window(slot).unwrap();
    let original_width = window.width;

    // Maximize button at x + 52 (after close and minimize)
    let maximize_x = window.x + 52.0;
    let maximize_y = window.y - 14.0;

    // Click to maximize
    system.queue_input(InputEvent::mouse_down(maximize_x, maximize_y, 0));
    system.process_input();
    system.run_frame();
    system.process_chrome_actions();

    let maximized_window = system.get_window(slot).unwrap();
    assert!(maximized_window.flags & window_flags::MAXIMIZED != 0);
    assert!(maximized_window.width > original_width);

    // Click again to restore
    system.queue_input(InputEvent::mouse_down(maximize_x, maximize_y, 0));
    system.process_input();
    system.run_frame();
    system.process_chrome_actions();

    let restored_window = system.get_window(slot).unwrap();
    assert!(restored_window.flags & window_flags::MAXIMIZED == 0);
}
```

### Window Resize

```rust
#[test]
fn test_window_resize() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    let slot = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot, 100.0, 100.0, 400.0, 300.0);

    let window = system.get_window(slot).unwrap();
    let original_width = window.width;
    let original_height = window.height;

    // Resize handle is bottom-right 16x16 area
    let handle_x = window.x + window.width - 8.0;
    let handle_y = window.y + window.height - 8.0;

    // Mouse down on resize handle
    system.queue_input(InputEvent::mouse_down(handle_x, handle_y, 0));
    system.process_input();
    system.run_frame();

    // Drag to expand
    system.queue_input(InputEvent::mouse_move(handle_x + 50.0, handle_y + 30.0));
    system.process_input();
    system.run_frame();

    let resized_window = system.get_window(slot).unwrap();
    assert!(resized_window.width > original_width);
    assert!(resized_window.height > original_height);

    // Mouse up
    system.queue_input(InputEvent::mouse_up(handle_x + 50.0, handle_y + 30.0, 0));
    system.process_input();
    system.run_frame();
}
```

### Hover State

```rust
#[test]
fn test_button_hover_state() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    let slot = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot, 100.0, 100.0, 400.0, 300.0);

    let window = system.get_window(slot).unwrap();
    let close_x = window.x + 12.0;
    let close_y = window.y - 14.0;

    // Move mouse over close button (no click)
    system.queue_input(InputEvent::mouse_move(close_x, close_y));
    system.process_input();
    system.run_frame();

    let chrome_state = system.read_chrome_state(
        system.find_app_by_type(app_type::WINDOW_CHROME).unwrap()
    );

    let hovered_window = (chrome_state.hovered_button >> 8) as u32;
    let hovered_button = (chrome_state.hovered_button & 0xFF) as u32;

    assert_eq!(hovered_window, slot);
    assert_eq!(hovered_button, 0); // BUTTON_CLOSE
}
```

### Performance Benchmark

```rust
#[test]
fn bench_chrome_many_windows() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    // Create 20 windows
    for i in 0..20 {
        let slot = system.launch_app(app_type::TERMINAL).unwrap();
        system.set_window(
            slot,
            50.0 + (i % 5) as f32 * 150.0,
            50.0 + (i / 5) as f32 * 150.0,
            300.0,
            200.0
        );
    }

    // Warm up
    for _ in 0..10 {
        system.mark_all_dirty();
        system.run_frame();
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        system.mark_all_dirty();
        system.run_frame();
    }
    let duration = start.elapsed();

    let us_per_frame = duration.as_micros() / 1000;
    println!("Chrome (20 windows): {}us/frame", us_per_frame);

    // Should be well under 1ms (16ms budget for 60fps)
    assert!(us_per_frame < 1000, "Chrome generation too slow: {}us", us_per_frame);
}
```

### Focus Tracking

```rust
#[test]
fn test_chrome_tracks_focused_window() {
    let device = metal::Device::system_default().expect("No Metal device");
    let mut system = GpuAppSystem::new(&device).unwrap();

    // Create two windows
    let slot1 = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot1, 100.0, 100.0, 400.0, 300.0);

    let slot2 = system.launch_app(app_type::TERMINAL).unwrap();
    system.set_window(slot2, 200.0, 200.0, 400.0, 300.0);

    // Focus window 2
    system.set_focus(slot2);
    system.run_frame();

    let chrome_state = system.read_chrome_state(
        system.find_app_by_type(app_type::WINDOW_CHROME).unwrap()
    );

    assert_eq!(chrome_state.focused_window, slot2);

    // Click on window 1 to focus it
    let window1 = system.get_window(slot1).unwrap();
    system.queue_input(InputEvent::mouse_down(window1.x + 50.0, window1.y + 50.0, 0));
    system.process_input();
    system.run_frame();

    let chrome_state = system.read_chrome_state(
        system.find_app_by_type(app_type::WINDOW_CHROME).unwrap()
    );

    assert_eq!(chrome_state.focused_window, slot1);
}
```

## Success Metrics

1. **Parallel generation**: All window chrome generated in single GPU dispatch
2. **Vertex count**: Exactly `window_count * 90` vertices generated
3. **Smooth dragging**: Window position updates every frame during drag
4. **Button response**: `clicked_button` set within same frame as click event
5. **No CPU per-window loop**: CPU only reads/clears `clicked_button`, never iterates windows
6. **Performance**: < 100us for 20 windows on Apple Silicon

## Integration with Existing Systems

### Event Loop Integration

The window chrome system integrates with the existing `event_loop.rs` hit testing:

```rust
// In GpuAppSystem::process_input_frame()
let hit = self.event_loop_handle.read_hit_result();
if hit.is_hit() {
    match hit.region() {
        region::CLOSE | region::MINIMIZE | region::MAXIMIZE => {
            // Let chrome app handle button clicks
            // It will set clicked_button
        }
        region::TITLE => {
            // Chrome app will start drag
        }
        region::RESIZE => {
            // Chrome app will start resize
        }
        region::CONTENT => {
            // Route to app
        }
        _ => {}
    }
}
```

### Megakernel Dispatch

The chrome update is dispatched via the megakernel alongside other apps:

```metal
// In megakernel dispatch
case APP_TYPE_WINDOW_CHROME:
    window_chrome_update(
        app,
        unified_state,
        unified_vertices,  // Pass vertex buffer
        windows,           // Pass window array
        window_count,
        tid,
        tg_size
    );
    break;
```

## Future Enhancements

1. **Rounded corners**: Use signed distance field for smooth corners
2. **Window shadows**: Generate shadow geometry with gradient
3. **Title text**: Integrate with bitmap font system for window titles
4. **Button icons**: Add X, -, +/full-screen icons inside buttons
5. **Animation**: Smooth minimize/maximize/close animations
6. **Themes**: Support dark mode, custom color schemes
