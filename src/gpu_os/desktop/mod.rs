//! GPU Desktop Environment
//!
//! A complete desktop environment running on GPU compute shaders.
//! Provides window management, compositing, dock, and app framework.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────┐
//! │              DesktopState                  │
//! │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
//! │  │ Window[] │ │ DragState│ │  Dock    │   │
//! │  └──────────┘ └──────────┘ └──────────┘   │
//! └────────────────────────────────────────────┘
//!                     │
//!         ┌───────────┼───────────┐
//!         ▼           ▼           ▼
//!   ┌──────────┐ ┌──────────┐ ┌──────────┐
//!   │  Window  │ │Compositor│ │   Dock   │
//!   │ Manager  │ │  Kernel  │ │  Kernel  │
//!   │  Kernel  │ │          │ │          │
//!   └──────────┘ └──────────┘ └──────────┘
//!         │           │           │
//!         └───────────┼───────────┘
//!                     ▼
//!              ┌──────────┐
//!              │  Output  │
//!              │ Texture  │
//!              └──────────┘
//! ```
//!
//! # Modules
//!
//! - `types` - Core data structures (Window, DesktopState, DragState)
//! - `window_manager` - Window management kernel (hit test, move, resize)
//! - `compositor` - Window compositing and chrome rendering
//! - `dock` - Dock system (app icons, running indicators)
//! - `app` - Application framework (DesktopApp trait)

pub mod types;
pub mod window_manager;
pub mod compositor;
pub mod dock;
pub mod app;
pub mod window_textures;
pub mod apps;
pub mod menu_bar;

pub use types::*;
pub use window_manager::*;
pub use compositor::*;
pub use dock::*;
pub use app::*;
pub use window_textures::*;
pub use apps::*;
pub use menu_bar::*;

// Re-export commonly used items
pub use types::{
    Window, DesktopState, DragState,
    WINDOW_FLAG_VISIBLE, WINDOW_FLAG_FOCUSED, WINDOW_FLAG_MINIMIZED,
    WINDOW_FLAG_MAXIMIZED, WINDOW_FLAG_DRAGGING, WINDOW_FLAG_RESIZING,
    WINDOW_FLAG_DIRTY, WINDOW_FLAG_BORDERLESS, WINDOW_FLAG_FIXED_SIZE,
    WINDOW_FLAG_MODAL, WINDOW_FLAG_NEEDS_REDRAW,
    RESIZE_NONE, RESIZE_LEFT, RESIZE_RIGHT, RESIZE_TOP, RESIZE_BOTTOM,
    MAX_WINDOWS, TITLE_BAR_HEIGHT, BUTTON_SIZE, CORNER_RADIUS,
};

use metal::*;
use std::mem;

/// Integrated GPU Desktop Environment
///
/// Combines window manager, compositor, dock, and app framework
/// into a single cohesive system.
pub struct GpuDesktop {
    pub device: Device,
    pub command_queue: CommandQueue,

    // State
    pub state: DesktopState,
    windows_buffer: Buffer,

    // Components
    pub window_manager: GpuWindowManager,
    pub compositor: GpuCompositor,
    pub dock: GpuDock,
    pub menu_bar: GpuMenuBar,
    pub apps: AppRegistry,
    pub window_textures: WindowTextures,

    // Settings
    pub background_color: [f32; 4],

    // Timing for app render context
    total_time: f32,
    frame_count: u64,
}

impl GpuDesktop {
    /// Create a new GPU desktop
    pub fn new(
        device: &Device,
        screen_width: f32,
        screen_height: f32,
        pixel_format: MTLPixelFormat,
    ) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Create components
        let window_manager = GpuWindowManager::new(device)?;
        let compositor = GpuCompositor::new(device, pixel_format)?;
        let mut dock = GpuDock::new(device, pixel_format)?;
        let menu_bar = GpuMenuBar::new(device, screen_width)?;

        // Initialize state
        let state = DesktopState::new(screen_width, screen_height);
        dock.state = DockState::new(screen_width, screen_height);

        // Create windows buffer
        let windows_buffer = device.new_buffer(
            (MAX_WINDOWS * mem::size_of::<Window>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device: device.clone(),
            command_queue,
            state,
            windows_buffer,
            window_manager,
            compositor,
            dock,
            menu_bar,
            apps: AppRegistry::new(),
            window_textures: WindowTextures::new(device, pixel_format),
            background_color: [0.1, 0.1, 0.15, 1.0],  // Dark blue-gray
            total_time: 0.0,
            frame_count: 0,
        })
    }

    /// Sync state to GPU buffers
    fn sync_to_gpu(&self) {
        // Copy windows to GPU buffer
        let ptr = self.windows_buffer.contents() as *mut Window;
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.state.windows.as_ptr(),
                ptr,
                self.state.window_count as usize,
            );
        }
    }

    /// Sync state from GPU buffers
    fn sync_from_gpu(&mut self) {
        let ptr = self.windows_buffer.contents() as *const Window;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr,
                self.state.windows.as_mut_ptr(),
                self.state.window_count as usize,
            );
        }
    }

    /// Create a new window for an app
    pub fn create_window(
        &mut self,
        title: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Option<u32> {
        self.state.create_window(title, x, y, width, height)
    }

    /// Close a window
    pub fn close_window(&mut self, window_id: u32) {
        // Close associated app first
        self.apps.close_by_window(window_id);
        self.state.close_window(window_id);
    }

    /// Launch an app in a new window
    pub fn launch_app(
        &mut self,
        app_id: u32,
        app: Box<dyn DesktopApp>,
    ) -> Result<(u32, u32), String> {
        let (pref_w, pref_h) = app.preferred_size();
        let name = app.name().to_string();

        // Find non-overlapping position for new window
        let (x, y) = self.state.find_non_overlapping_position(pref_w, pref_h);

        let window_id = self.state.create_window(&name, x, y, pref_w, pref_h)
            .ok_or_else(|| "Failed to create window".to_string())?;

        // Launch app
        let instance_id = self.apps.launch(app_id, window_id, app, &self.device)?;

        // Update dock running indicator
        for i in 0..self.dock.state.config.item_count as usize {
            if self.dock.state.items[i].app_id == app_id {
                self.dock.state.items[i].instance_count += 1;
            }
        }

        Ok((window_id, instance_id))
    }

    /// Handle mouse down event
    pub fn on_mouse_down(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.state.mouse_x = x;
        self.state.mouse_y = y;
        self.state.mouse_buttons |= 1 << button;

        // Check dock first (it's on top)
        if let Some(item_idx) = self.dock.state.item_at(x, y) {
            let app_id = self.dock.state.items[item_idx].app_id;
            // TODO: Launch or focus app
            return true;
        }

        // Sync to GPU and do hit test
        self.sync_to_gpu();
        let hit = self.window_manager.hit_test(
            &self.windows_buffer,
            self.state.window_count,
            x,
            y,
        );

        if hit.window_id != 0 {
            // Handle window interaction
            match hit.region {
                window_manager::REGION_CLOSE_BUTTON => {
                    self.close_window(hit.window_id);
                    return true;
                }
                window_manager::REGION_MINIMIZE_BUTTON => {
                    if let Some(win) = self.state.get_window_mut(hit.window_id) {
                        win.flags |= WINDOW_FLAG_MINIMIZED;
                    }
                    return true;
                }
                window_manager::REGION_MAXIMIZE_BUTTON => {
                    let (_, _, w, h) = self.state.usable_area();
                    if let Some(win) = self.state.get_window_mut(hit.window_id) {
                        if win.flags & WINDOW_FLAG_MAXIMIZED != 0 {
                            // TODO: Restore previous size
                            win.flags &= !WINDOW_FLAG_MAXIMIZED;
                        } else {
                            win.x = 0.0;
                            win.y = 0.0;
                            win.width = w;
                            win.height = h;
                            win.update_content_area();
                            win.flags |= WINDOW_FLAG_MAXIMIZED;
                        }
                    }
                    return true;
                }
                window_manager::REGION_TITLEBAR => {
                    // Start window drag
                    self.state.focus_window(hit.window_id);
                    if let Some(win) = self.state.get_window(hit.window_id) {
                        self.state.drag.start_move(hit.window_id, x, y, win.x, win.y);
                    }
                    return true;
                }
                window_manager::REGION_RESIZE => {
                    // Start resize
                    self.state.focus_window(hit.window_id);
                    if let Some(win) = self.state.get_window(hit.window_id) {
                        self.state.drag.start_resize(
                            hit.window_id, hit.resize_edge,
                            x, y, win.x, win.y, win.width, win.height,
                        );
                    }
                    return true;
                }
                window_manager::REGION_CONTENT => {
                    // Focus window and dispatch to app
                    self.state.focus_window(hit.window_id);
                    if let Some(win) = self.state.get_window(hit.window_id) {
                        let event = AppInputEvent {
                            event_type: AppEventType::MouseDown,
                            key_code: 0,
                            mouse_x: x - win.x - win.content_x,
                            mouse_y: y - win.y - win.content_y,
                            mouse_button: button,
                            modifiers: KeyModifiers::default(),
                        };
                        self.apps.dispatch_input(hit.window_id, &event);
                    }
                    return true;
                }
                _ => {}
            }
        }

        false
    }

    /// Handle mouse up event
    pub fn on_mouse_up(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.state.mouse_x = x;
        self.state.mouse_y = y;
        self.state.mouse_buttons &= !(1 << button);

        if self.state.drag.is_dragging() {
            self.state.drag.end_drag();
            return true;
        }

        false
    }

    /// Handle mouse move event
    pub fn on_mouse_move(&mut self, x: f32, y: f32) {
        let dx = x - self.state.mouse_x;
        let dy = y - self.state.mouse_y;
        self.state.mouse_x = x;
        self.state.mouse_y = y;

        // Update dock hover
        self.dock.state.update_hover(x, y);

        // Handle window drag/resize
        if self.state.drag.is_moving() {
            self.sync_to_gpu();
            self.window_manager.move_window(
                &self.windows_buffer,
                self.state.window_count,
                self.state.drag.window_id,
                dx, dy,
                self.state.screen_width,
                self.state.screen_height,
                self.state.dock_height,
            );
            self.sync_from_gpu();
        } else if self.state.drag.is_resizing() {
            self.sync_to_gpu();
            self.window_manager.resize_window(
                &self.windows_buffer,
                self.state.window_count,
                self.state.drag.window_id,
                dx, dy,
                self.state.drag.resize_edge,
                self.state.screen_width,
                self.state.screen_height,
                self.state.dock_height,
            );
            self.sync_from_gpu();
        }
    }

    /// Handle key down event
    pub fn on_key_down(&mut self, key_code: u32, modifiers: KeyModifiers) -> bool {
        if self.state.focused_window != 0 {
            let event = AppInputEvent {
                event_type: AppEventType::KeyDown,
                key_code,
                mouse_x: 0.0,
                mouse_y: 0.0,
                mouse_button: 0,
                modifiers,
            };
            return self.apps.dispatch_input(self.state.focused_window, &event);
        }
        false
    }

    /// Update the desktop state
    pub fn update(&mut self, delta_time: f32) {
        self.total_time += delta_time;

        // Update all running apps
        self.apps.update_all(delta_time);

        // Update dock layout
        self.dock.update_layout();
    }

    /// Render the desktop
    pub fn render(&mut self, encoder: &RenderCommandEncoderRef) {
        // Sync windows to GPU
        self.sync_to_gpu();

        // Generate compositor vertices
        self.compositor.generate_vertices(
            &self.windows_buffer,
            self.state.window_count,
            self.state.screen_width,
            self.state.screen_height,
            self.state.focused_window,
        );

        // Render compositor (windows)
        self.compositor.render(encoder, &self.windows_buffer);

        // Render app content for each visible window
        self.frame_count += 1;
        for i in 0..self.state.window_count as usize {
            let window = &self.state.windows[i];
            if window.flags & WINDOW_FLAG_VISIBLE != 0 && window.flags & WINDOW_FLAG_MINIMIZED == 0 {
                let window_id = window.id;
                let window_x = window.x;
                let window_y = window.y;
                let content_x = window.content_x;
                let content_y = window.content_y;
                let content_w = window.content_width;
                let content_h = window.content_height;

                // Create render context for this window
                let mut ctx = AppRenderContext {
                    encoder,
                    width: content_w,
                    height: content_h,
                    delta_time: 0.0,  // Already passed in update
                    total_time: self.total_time,
                    frame: self.frame_count,
                    window_x: window_x + content_x,
                    window_y: window_y + content_y,
                    screen_width: self.state.screen_width,
                    screen_height: self.state.screen_height,
                };

                // Render app content
                self.apps.render_for_window(window_id, &mut ctx);
            }
        }

        // Render dock
        self.dock.render(encoder);

        // Render menu bar
        self.menu_bar.render(encoder, self.state.screen_width, self.state.screen_height);
    }
}
