# Issue #134: GPU Desktop Environment

## Vision

A GPU-native desktop environment that can launch, manage, and composite multiple applications - proving the GPU can be the OS, not just run apps.

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU Desktop                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────┐   ┌─────────────────────┐            │
│   │ Filesystem Browser ×│   │ Document Viewer   ×│            │
│   ├─────────────────────┤   ├─────────────────────┤            │
│   │  📁 src/            │   │  <html>             │            │
│   │  📁 examples/       │   │    <h1>Hello</h1>   │            │
│   │  📄 Cargo.toml      │   │  </html>            │            │
│   └─────────────────────┘   └─────────────────────┘            │
│                                                                 │
│   ┌─────────────────────┐                                      │
│   │ GPU Ripgrep       ×│                                      │
│   ├─────────────────────┤                                      │
│   │ > TODO              │                                      │
│   │ 42 matches          │                                      │
│   └─────────────────────┘                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ [📁Files] [📄Docs] [🔍Search] [✨Particles] [🌀Mandelbrot]     │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Matters

1. **GPU as OS, not GPU-accelerated OS** - The GPU manages windows, not just renders them
2. **Unified demo environment** - All 52 examples accessible from one interface
3. **Real window management** - Focus, z-order, drag, resize
4. **GPU-native compositing** - No CPU in the render loop

## Architecture

### GPU-Side Data Structures

```rust
/// Window state (GPU-resident)
#[repr(C)]
pub struct Window {
    pub id: u32,
    pub app_id: u32,           // Which app owns this window
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub z_order: u32,          // Higher = on top
    pub flags: u32,            // FOCUSED, MINIMIZED, MAXIMIZED, etc.
    pub title_start: u32,      // Offset into title text buffer
    pub title_length: u32,
}

/// Desktop state (GPU-resident)
#[repr(C)]
pub struct DesktopState {
    pub window_count: u32,
    pub focused_window: u32,   // Window ID with focus
    pub dock_height: f32,
    pub drag_state: DragState,
}

/// Drag operation (GPU-side tracking)
#[repr(C)]
pub struct DragState {
    pub active: u32,           // 0 = not dragging
    pub window_id: u32,
    pub offset_x: f32,
    pub offset_y: f32,
}

/// App registration
#[repr(C)]
pub struct AppInfo {
    pub id: u32,
    pub name_start: u32,
    pub name_length: u32,
    pub icon_texture_id: u32,
    pub factory_fn: fn() -> Box<dyn GpuApp>,
}
```

### Window Manager Kernel

```metal
kernel void window_manager(
    device Window* windows [[buffer(0)]],
    device DesktopState* state [[buffer(1)]],
    device InputEvent* events [[buffer(2)]],
    constant uint& event_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= event_count) return;

    InputEvent event = events[gid];

    // Handle window interactions
    if (event.event_type == MOUSE_DOWN) {
        // Hit test windows in z-order (highest first)
        for (int z = MAX_Z; z >= 0; z--) {
            for (int w = 0; w < state->window_count; w++) {
                Window win = windows[w];
                if (win.z_order != z) continue;

                if (point_in_rect(event.x, event.y, win)) {
                    // Check if click is in title bar
                    if (event.y < win.y + TITLE_BAR_HEIGHT) {
                        // Check close button
                        if (event.x > win.x + win.width - 30) {
                            // Close window (mark for removal)
                            windows[w].flags |= FLAG_CLOSING;
                        } else {
                            // Start drag
                            state->drag_state.active = 1;
                            state->drag_state.window_id = win.id;
                            state->drag_state.offset_x = event.x - win.x;
                            state->drag_state.offset_y = event.y - win.y;
                        }
                    }

                    // Focus this window
                    state->focused_window = win.id;
                    bring_to_front(windows, state, win.id);
                    return;
                }
            }
        }

        // Click on dock?
        if (event.y > state->dock_y) {
            uint app_idx = (event.x - DOCK_START_X) / DOCK_ICON_WIDTH;
            if (app_idx < app_count) {
                // Launch app (signal to CPU)
                state->launch_request = app_idx;
            }
        }
    }

    if (event.event_type == MOUSE_MOVE && state->drag_state.active) {
        uint wid = state->drag_state.window_id;
        windows[wid].x = event.x - state->drag_state.offset_x;
        windows[wid].y = event.y - state->drag_state.offset_y;
    }

    if (event.event_type == MOUSE_UP) {
        state->drag_state.active = 0;
    }
}
```

### Window Compositing Kernel

```metal
kernel void composite_windows(
    device Window* windows [[buffer(0)]],
    device DesktopState* state [[buffer(1)]],
    device PaintVertex* desktop_vertices [[buffer(2)]],
    device uint* app_vertex_offsets [[buffer(3)]],  // Where each app's vertices start
    device PaintVertex* output [[buffer(4)]],
    constant uint& window_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Render windows in z-order
    // 1. Desktop background
    // 2. Windows (lowest z to highest)
    // 3. Dock
    // 4. Focused window chrome highlight

    // Each window's content comes from its app's vertex buffer
    // Window manager adds chrome (title bar, borders, shadow)
}
```

## Rust-Side Implementation

```rust
pub struct GpuDesktop {
    // GPU state
    windows_buffer: Buffer,
    state_buffer: Buffer,

    // App management
    apps: Vec<AppInfo>,
    running_apps: HashMap<u32, Box<dyn GpuApp>>,

    // Pipelines
    window_manager_pipeline: ComputePipelineState,
    compositor_pipeline: ComputePipelineState,

    // Rendering
    paint_engine: GpuPaintEngine,
}

impl GpuDesktop {
    pub fn register_app(&mut self, name: &str, factory: fn() -> Box<dyn GpuApp>) {
        self.apps.push(AppInfo {
            id: self.apps.len() as u32,
            name: name.to_string(),
            factory,
        });
    }

    pub fn launch_app(&mut self, app_id: u32) -> u32 {
        let app = (self.apps[app_id as usize].factory)();
        let window_id = self.create_window(app_id);
        self.running_apps.insert(window_id, app);
        window_id
    }

    pub fn close_window(&mut self, window_id: u32) {
        self.running_apps.remove(&window_id);
        // Update GPU window buffer
    }

    pub fn frame(&mut self, encoder: &ComputeCommandEncoderRef) {
        // 1. Run window manager kernel (handle input)
        self.dispatch_window_manager(encoder);

        // 2. Run each app's update
        for (window_id, app) in &mut self.running_apps {
            app.update(encoder, /* app-specific params */);
        }

        // 3. Composite all windows
        self.dispatch_compositor(encoder);
    }
}
```

## App Registry

Pre-register all existing demos:

```rust
fn register_apps(desktop: &mut GpuDesktop) {
    // Productivity
    desktop.register_app("Filesystem Browser", || Box::new(FilesystemBrowser::new()));
    desktop.register_app("Document Viewer", || Box::new(DocumentViewer::new()));
    desktop.register_app("GPU Ripgrep", || Box::new(GpuRipgrep::new()));

    // Simulations
    desktop.register_app("Particles", || Box::new(ParticleDemo::new()));
    desktop.register_app("Boids", || Box::new(BoidsDemo::new()));
    desktop.register_app("Game of Life", || Box::new(GameOfLife::new()));
    desktop.register_app("Mandelbrot", || Box::new(Mandelbrot::new()));
    desktop.register_app("Metaballs", || Box::new(Metaballs::new()));
    desktop.register_app("Waves", || Box::new(Waves::new()));
    desktop.register_app("Ball Physics", || Box::new(BallPhysics::new()));
}
```

## Window Chrome

```css
/* GPU-native CSS for window chrome */
.window {
    background: #f0f0f0;
    border: 1px solid #999;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.window-titlebar {
    height: 28px;
    background: linear-gradient(#e8e8e8, #d0d0d0);
    border-radius: 8px 8px 0 0;
    display: flex;
    align-items: center;
    padding: 0 8px;
}

.window-title {
    flex: 1;
    font-size: 13px;
    font-weight: 500;
    color: #333;
}

.window-button {
    width: 12px;
    height: 12px;
    border-radius: 6px;
    margin-left: 8px;
}

.window-button.close { background: #ff5f57; }
.window-button.minimize { background: #ffbd2e; }
.window-button.maximize { background: #28c940; }

.dock {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255,255,255,0.8);
    border-radius: 16px 16px 0 0;
    padding: 8px 16px;
    display: flex;
    gap: 12px;
}

.dock-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    cursor: pointer;
}

.dock-icon:hover {
    transform: scale(1.1);
}
```

## Input Routing

```rust
impl GpuDesktop {
    fn route_input(&self, event: InputEvent) -> InputTarget {
        // 1. Check if click is on dock
        if event.y > self.dock_y {
            return InputTarget::Dock;
        }

        // 2. Check windows in z-order (top to bottom)
        for window in self.windows_by_z_desc() {
            if window.contains(event.x, event.y) {
                if event.y < window.y + TITLE_BAR_HEIGHT {
                    return InputTarget::WindowChrome(window.id);
                } else {
                    return InputTarget::App(window.id);
                }
            }
        }

        // 3. Desktop background
        InputTarget::Desktop
    }
}
```

## Success Criteria

1. **Launch apps** - Click dock icon → window appears
2. **Close apps** - Click X → window disappears, app state cleaned up
3. **Focus windows** - Click window → brings to front, receives input
4. **Drag windows** - Drag title bar → window moves
5. **Multiple apps** - 3+ windows open simultaneously
6. **App isolation** - Each app has independent state
7. **60 FPS** - Smooth compositing at 60 Hz

## Implementation Steps

1. Define GPU data structures (Window, DesktopState, DragState)
2. Create window_manager Metal kernel
3. Create compositor Metal kernel
4. Implement GpuDesktop Rust struct
5. Add window chrome rendering (title bar, buttons)
6. Implement dock rendering
7. Add app registry and factory pattern
8. Wire up input routing
9. Test with 3 demo apps
10. Polish visuals and interactions

## Future Extensions

- Window resize (drag edges)
- Window minimize/maximize
- Window snapping (drag to edge)
- App menu bar
- System tray
- Notifications
- Multiple desktops/spaces
