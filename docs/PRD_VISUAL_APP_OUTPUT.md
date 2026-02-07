# PRD: Visual Output for User Apps (Terminal, Filesystem, Document)

## Problem Statement

When apps are launched from the dock in the visual megakernel demo, they are created correctly with the right app_type (TERMINAL=5, FILESYSTEM=4, DOCUMENT=6), but they don't render visibly. The root cause is that these app types fall through to `counter_app_update()` in `dispatch_app_update()`, which only increments a counter without generating any vertex geometry.

## Current State

### Working Components
1. **Dock click detection** - GPU correctly detects clicked dock item
2. **App launching** - Apps are created with correct type, state, and vertex allocations
3. **Window chrome rendering** - `window_chrome_update()` generates title bars, buttons, borders
4. **System apps rendering** - Compositor, Dock, MenuBar all render correctly

### The Gap
```metal
inline void dispatch_app_update(...) {
    switch (app->app_type) {
        case APP_TYPE_COMPOSITOR: compositor_update(...); break;
        case APP_TYPE_DOCK: dock_update(...); break;
        case APP_TYPE_MENUBAR: menubar_update(...); break;
        case APP_TYPE_WINDOW_CHROME: window_chrome_update(...); break;
        default:
            // TERMINAL, FILESYSTEM, DOCUMENT fall through here!
            counter_app_update(app, unified_state, tid, tg_size);
            break;
    }
}
```

## Architecture Analysis

### Vertex Generation Pattern (from dock_update, window_chrome_update)

Every visual app must:
1. Calculate base vertex pointer: `device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex)`
2. Use `write_quad()` to generate 6 vertices per rectangle
3. Set `app->vertex_count` to total vertices generated
4. Use depth values for layering (0.0=back, 1.0=front)

### Coordinate System
- Screen space: (0,0) top-left, (width, height) bottom-right
- Clip space: (-1,-1) bottom-left, (1,1) top-right
- Vertex shader transforms screen to clip space

### Existing Helpers
```metal
// Write a quad (6 vertices = 2 triangles)
inline void write_quad(
    device RenderVertex* v,
    float2 origin,      // top-left corner
    float2 size,        // width, height
    float depth,        // z-order (0.0-1.0)
    float4 color        // RGBA
);
```

### App State Structure
Each app has:
- `app->state_offset` - byte offset into unified_state buffer
- `app->vertex_offset` - byte offset into unified_vertices buffer
- `app->vertex_count` - number of vertices generated (set by app)
- `app->window_id` - associated window (for position/size)

## Design

### App-Specific Update Functions

#### 1. Terminal App (`terminal_update`)
```
State Layout:
  - uint: cursor_pos
  - uint: scroll_offset
  - uint: line_count
  - uint: reserved
  - char[]: text_buffer (remaining state space)

Visual Output:
  - Background fill (dark gray/black)
  - Cursor blink indicator
  - Text content (future: use text rendering system)

Initial Implementation:
  - Just render colored background with app type indicator
  - Shows "TERMINAL" placeholder text area
```

#### 2. Filesystem App (`filesystem_update`)
```
State Layout:
  - uint: current_dir_hash
  - uint: selected_index
  - uint: scroll_offset
  - uint: file_count
  - FileEntry[]: directory listing

Visual Output:
  - Background fill (lighter gray)
  - File list area (scrollable)
  - Selection highlight

Initial Implementation:
  - Render colored background with icon/indicator
  - Show file browser placeholder
```

#### 3. Document App (`document_update`)
```
State Layout:
  - uint: cursor_pos
  - uint: selection_start
  - uint: selection_end
  - uint: scroll_offset
  - char[]: document_content

Visual Output:
  - White/cream background (document area)
  - Text content
  - Cursor and selection highlighting

Initial Implementation:
  - Render document-colored background
  - Show text area placeholder
```

### Implementation Approach

#### Phase 1: Colored Placeholders (MVP)
Each app renders a solid colored rectangle that fills its window area, making apps visible immediately after launch.

```metal
// Each app gets a distinct color
constant float4 TERMINAL_COLOR = float4(0.1, 0.1, 0.15, 1.0);    // Dark blue-gray
constant float4 FILESYSTEM_COLOR = float4(0.2, 0.2, 0.25, 1.0);  // Medium gray
constant float4 DOCUMENT_COLOR = float4(0.95, 0.95, 0.9, 1.0);   // Cream/paper

inline void terminal_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    uint window_count,
    uint tid,
    uint tg_size
) {
    if (tid != 0) return;  // Single-threaded for now

    // Find our window
    device GpuWindow* window = NULL;
    for (uint i = 0; i < window_count; i++) {
        if (windows[i].app_slot == app->slot_id) {
            window = &windows[i];
            break;
        }
    }
    if (!window) return;

    // Get vertex buffer
    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);

    // Draw background fill
    float depth = 0.5;  // Middle layer (behind chrome, in front of compositor)
    write_quad(verts, float2(window->x, window->y),
               float2(window->width, window->height),
               depth, TERMINAL_COLOR);

    app->vertex_count = 6;
}
```

#### Phase 2: Window Integration
- Look up window position from `windows` buffer using `app->window_id`
- Render content within window bounds
- Respect window chrome offset (title bar height)

#### Phase 3: Interactive Content
- Add text rendering using existing text system
- Handle input events from GPU input queue
- Update state based on user interaction

## Pseudocode

### dispatch_app_update (Updated)
```metal
inline void dispatch_app_update(...) {
    switch (app->app_type) {
        // ... existing cases ...

        case APP_TYPE_TERMINAL:
            terminal_update(app, unified_state, unified_vertices, windows, window_count, tid, tg_size);
            break;

        case APP_TYPE_FILESYSTEM:
            filesystem_update(app, unified_state, unified_vertices, windows, window_count, tid, tg_size);
            break;

        case APP_TYPE_DOCUMENT:
            document_update(app, unified_state, unified_vertices, windows, window_count, tid, tg_size);
            break;

        default:
            counter_app_update(app, unified_state, tid, tg_size);
            break;
    }
}
```

### terminal_update
```metal
inline void terminal_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    uint window_count,
    uint tid,
    uint tg_size
) {
    // Only thread 0 generates geometry (single-threaded MVP)
    if (tid != 0) return;

    // Find window for this app
    device GpuWindow* win = find_window_for_slot(windows, window_count, app->slot_id);
    if (!win) {
        app->vertex_count = 0;
        return;
    }

    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
    uint vert_idx = 0;

    // Title bar offset (window chrome takes top 30px)
    float title_height = 30.0;
    float content_y = win->y;  // Content starts below title bar (handled by chrome)
    float content_height = win->height;

    // 1. Background fill
    write_quad(verts + vert_idx,
               float2(win->x, content_y),
               float2(win->width, content_height),
               0.5,  // depth
               float4(0.1, 0.1, 0.15, 1.0));
    vert_idx += 6;

    // 2. Cursor (blinking rectangle)
    // TODO: Add cursor based on state

    // 3. Text content
    // TODO: Integrate with text rendering system

    app->vertex_count = vert_idx;
}
```

### filesystem_update
```metal
inline void filesystem_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    uint window_count,
    uint tid,
    uint tg_size
) {
    if (tid != 0) return;

    device GpuWindow* win = find_window_for_slot(windows, window_count, app->slot_id);
    if (!win) {
        app->vertex_count = 0;
        return;
    }

    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
    uint vert_idx = 0;

    // 1. Background
    write_quad(verts + vert_idx,
               float2(win->x, win->y),
               float2(win->width, win->height),
               0.5,
               float4(0.2, 0.2, 0.25, 1.0));  // Medium gray
    vert_idx += 6;

    // 2. Sidebar (left panel)
    float sidebar_width = 150.0;
    write_quad(verts + vert_idx,
               float2(win->x, win->y),
               float2(sidebar_width, win->height),
               0.51,
               float4(0.15, 0.15, 0.2, 1.0));  // Darker sidebar
    vert_idx += 6;

    // 3. File list area
    // TODO: Render file entries

    app->vertex_count = vert_idx;
}
```

### document_update
```metal
inline void document_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    uint window_count,
    uint tid,
    uint tg_size
) {
    if (tid != 0) return;

    device GpuWindow* win = find_window_for_slot(windows, window_count, app->slot_id);
    if (!win) {
        app->vertex_count = 0;
        return;
    }

    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
    uint vert_idx = 0;

    // 1. Paper background
    write_quad(verts + vert_idx,
               float2(win->x, win->y),
               float2(win->width, win->height),
               0.5,
               float4(0.98, 0.98, 0.95, 1.0));  // Off-white/cream
    vert_idx += 6;

    // 2. Margin indicators (subtle lines)
    float margin = 40.0;
    write_quad(verts + vert_idx,
               float2(win->x + margin, win->y),
               float2(1.0, win->height),
               0.51,
               float4(0.9, 0.9, 0.85, 1.0));  // Light line
    vert_idx += 6;

    // 3. Text content
    // TODO: Integrate text rendering

    app->vertex_count = vert_idx;
}
```

### Helper: find_window_for_slot
```metal
inline device GpuWindow* find_window_for_slot(
    device GpuWindow* windows,
    uint window_count,
    uint slot_id
) {
    for (uint i = 0; i < window_count; i++) {
        if (windows[i].app_slot == slot_id) {
            return &windows[i];
        }
    }
    return NULL;
}
```

## Tests

### Test 1: Terminal Renders Visible
```rust
#[test]
fn test_terminal_renders_visible() {
    let system = create_test_system();

    // Launch terminal
    let slot = system.launch_app(app_type::TERMINAL, 100.0, 100.0, 400.0, 300.0);

    // Run one frame
    system.run_frame();

    // Check vertex count > 0
    let desc = system.get_app_descriptor(slot);
    assert!(desc.vertex_count >= 6, "Terminal should generate at least one quad");

    // Check vertices are in window bounds
    let vertices = system.get_vertices_for_slot(slot);
    for v in vertices {
        assert!(v.position.x >= 100.0 && v.position.x <= 500.0);
        assert!(v.position.y >= 100.0 && v.position.y <= 400.0);
    }
}
```

### Test 2: Filesystem Renders Visible
```rust
#[test]
fn test_filesystem_renders_visible() {
    let system = create_test_system();

    let slot = system.launch_app(app_type::FILESYSTEM, 150.0, 150.0, 500.0, 400.0);
    system.run_frame();

    let desc = system.get_app_descriptor(slot);
    assert!(desc.vertex_count >= 12, "Filesystem should generate background + sidebar");
}
```

### Test 3: Document Renders Visible
```rust
#[test]
fn test_document_renders_visible() {
    let system = create_test_system();

    let slot = system.launch_app(app_type::DOCUMENT, 200.0, 100.0, 600.0, 500.0);
    system.run_frame();

    let desc = system.get_app_descriptor(slot);
    assert!(desc.vertex_count >= 6, "Document should generate at least background");

    // Check paper color (off-white)
    let vertices = system.get_vertices_for_slot(slot);
    let bg_color = vertices[0].color;
    assert!(bg_color.r > 0.9 && bg_color.g > 0.9 && bg_color.b > 0.9);
}
```

### Test 4: Apps Have Distinct Visual Appearance
```rust
#[test]
fn test_apps_have_distinct_colors() {
    let system = create_test_system();

    let terminal = system.launch_app(app_type::TERMINAL, 0.0, 0.0, 200.0, 200.0);
    let filesystem = system.launch_app(app_type::FILESYSTEM, 250.0, 0.0, 200.0, 200.0);
    let document = system.launch_app(app_type::DOCUMENT, 500.0, 0.0, 200.0, 200.0);

    system.run_frame();

    let t_color = system.get_background_color(terminal);
    let f_color = system.get_background_color(filesystem);
    let d_color = system.get_background_color(document);

    // All should be different
    assert_ne!(t_color, f_color);
    assert_ne!(f_color, d_color);
    assert_ne!(t_color, d_color);

    // Terminal should be darkest
    assert!(t_color.luminance() < f_color.luminance());

    // Document should be brightest
    assert!(d_color.luminance() > f_color.luminance());
}
```

### Test 5: Window Association
```rust
#[test]
fn test_app_renders_in_correct_window() {
    let system = create_test_system();

    let slot = system.launch_app(app_type::TERMINAL, 100.0, 200.0, 300.0, 400.0);
    system.run_frame();

    // All vertices should be within window bounds
    let vertices = system.get_vertices_for_slot(slot);
    for v in vertices {
        assert!(v.position.x >= 100.0, "x too small");
        assert!(v.position.x <= 400.0, "x too large"); // 100 + 300
        assert!(v.position.y >= 200.0, "y too small");
        assert!(v.position.y <= 600.0, "y too large"); // 200 + 400
    }
}
```

## Implementation Plan

### Phase 1: Colored Placeholders (Immediate)
1. Add `terminal_update()`, `filesystem_update()`, `document_update()` functions to shader
2. Add cases to `dispatch_app_update()` switch statement
3. Each function renders a solid colored background quad
4. Test: Apps are visible after launch

### Phase 2: Window Integration
1. Add `find_window_for_slot()` helper
2. Update functions to use window position/size from buffer
3. Add proper depth ordering (behind chrome, in front of compositor)
4. Test: Apps render in correct window bounds

### Phase 3: Basic UI Elements
1. Terminal: cursor blink, text area
2. Filesystem: sidebar, file list area
3. Document: margins, text area
4. Test: Distinct visual elements visible

### Phase 4: Text Integration
1. Integrate with existing text rendering system
2. Add text content to app state
3. Render text within app bounds
4. Test: Text visible in apps

## Success Criteria

1. Launching Terminal from dock shows dark-colored window
2. Launching Filesystem from dock shows gray window with sidebar
3. Launching Document from dock shows paper-white window
4. All apps render within their window bounds
5. Window chrome (title bar, buttons) appears above app content
6. Multiple apps can be launched and all remain visible
7. Apps are distinguishable by color at minimum

## Files to Modify

| File | Change |
|------|--------|
| `src/gpu_os/gpu_app_system.rs` | Add terminal_update, filesystem_update, document_update functions; update dispatch_app_update switch |
| `tests/test_visual_app_output.rs` | Create new test file with all tests |
| `examples/visual_megakernel.rs` | May need debug output updates |
