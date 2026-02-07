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

use crate::gpu_os::input::InputHandler;
use crate::gpu_os::dynamic_app::{AppDiscovery, DynamicGpuApp, DynamicDesktopApp};
use crate::gpu_os::event_loop::{
    EventLoopHandle, GpuEventLoopState, GpuWindow as EventLoopWindow,
    dispatch, region, window_flags as el_window_flags, EVENT_LOOP_SHADER_SOURCE,
    INVALID_WINDOW,
};

/// Integrated GPU Desktop Environment
///
/// Combines window manager, compositor, dock, and app framework
/// into a single cohesive system.
///
/// # GPU Event Loop (Issue #149)
///
/// The desktop now supports GPU-driven event dispatch where:
/// - Input events are pushed to a GPU ring buffer via `push_*` methods
/// - GPU kernels handle hit testing, window drag/resize, and focus
/// - CPU only handles app-specific input dispatch
///
/// Use `init_gpu_event_loop()` then `process_gpu_events()` for GPU-driven mode.
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

    // GPU Event Loop (Issue #149)
    input_handler: InputHandler,
    event_loop_handle: Option<EventLoopHandle>,
    event_loop_windows_buffer: Option<Buffer>,  // GpuWindow format for event loop

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

        // Create input handler for GPU event loop
        let input_handler = InputHandler::new(device);

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
            input_handler,
            event_loop_handle: None,
            event_loop_windows_buffer: None,
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

        // Process launch requests from apps
        self.process_launch_requests();

        // Update dock layout
        self.dock.update_layout();
    }

    /// Process launch requests from running apps
    fn process_launch_requests(&mut self) {
        // Collect launch requests from all apps
        let requests = self.apps.collect_launch_requests();

        // Launch each requested app
        for app_name in requests {
            if let Err(e) = self.launch_app_by_name(&app_name) {
                eprintln!("Failed to launch {}: {}", app_name, e);
            }
        }
    }

    /// Launch an app by name
    ///
    /// First tries built-in apps (terminal, files, documents, editor),
    /// then attempts to load from .gpuapp bundles in search paths.
    pub fn launch_app_by_name(&mut self, name: &str) -> Result<(u32, u32), String> {
        let name_lower = name.to_lowercase();

        // Try built-in apps first
        let builtin_result: Option<(u32, Box<dyn DesktopApp>)> = match name_lower.as_str() {
            "terminal" => {
                let id = self.apps.find_or_register("Terminal", 3);
                Some((id, Box::new(TerminalApp::new())))
            }
            "files" => {
                let id = self.apps.find_or_register("Files", 0);
                Some((id, Box::new(FileBrowserApp::new())))
            }
            "documents" => {
                let id = self.apps.find_or_register("Documents", 2);
                let mut viewer = DocumentViewerApp::new();
                viewer.set_title("New Document");
                viewer.load_html("<h1>New Document</h1><p>Empty document</p>");
                Some((id, Box::new(viewer)))
            }
            "editor" => {
                let id = self.apps.find_or_register("Editor", 4);
                Some((id, Box::new(TextEditorApp::new())))
            }
            _ => None,
        };

        if let Some((app_id, app)) = builtin_result {
            return self.launch_app(app_id, app);
        }

        // Try loading as dynamic app from .gpuapp bundle
        let discovery = AppDiscovery::new();
        if let Some(bundle_path) = discovery.find_by_name(&name_lower) {
            // Load GPU app from bundle
            let gpu_app = DynamicGpuApp::load(&bundle_path, &self.device)?;

            // Wrap in desktop app
            let desktop_app = DynamicDesktopApp::new(gpu_app, &self.device)?;

            // Register and launch
            let app_id = self.apps.find_or_register(&name_lower, 0);
            return self.launch_app(app_id, Box::new(desktop_app));
        }

        Err(format!("Unknown app: {}. Try 'apps' to see available apps.", name))
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

    // ========================================================================
    // GPU Event Loop (Issue #149)
    // ========================================================================

    /// Initialize the GPU event loop
    ///
    /// This compiles the event loop shaders and creates the necessary buffers.
    /// After calling this, use `push_*` methods to queue input and
    /// `process_gpu_events()` to run the GPU event loop.
    pub fn init_gpu_event_loop(&mut self) -> Result<(), String> {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        // Compile event loop kernels
        let options = CompileOptions::new();
        let library = self.device
            .new_library_with_source(EVENT_LOOP_SHADER_SOURCE, &options)
            .map_err(|e| format!("Failed to compile event loop shaders: {}", e))?;

        // Helper to get function and create pipeline
        let create_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
            let function = library.get_function(name, None)
                .map_err(|e| format!("Failed to get function {}: {}", name, e))?;
            self.device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("Failed to create pipeline for {}: {}", name, e))
        };

        let event_loop_pipeline = create_pipeline("gpu_event_loop")?;
        let hit_test_pipeline = create_pipeline("hit_test_parallel")?;
        let hit_result_handler_pipeline = create_pipeline("handle_hit_result")?;
        let window_move_pipeline = create_pipeline("window_move")?;
        let window_resize_pipeline = create_pipeline("window_resize")?;

        // Create state buffer
        let state_buffer = self.device.new_buffer(
            GpuEventLoopState::SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize state
        unsafe {
            let ptr = state_buffer.contents() as *mut GpuEventLoopState;
            *ptr = GpuEventLoopState::new();
        }

        // Create hit result buffer and initialize to zeros
        let hit_result_buffer = self.device.new_buffer(
            std::mem::size_of::<crate::gpu_os::event_loop::HitTestResult>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // Initialize hit result to zeros to avoid garbage values
        unsafe {
            let ptr = hit_result_buffer.contents() as *mut u64;
            *ptr = 0;
        }

        // Create event loop windows buffer (GpuWindow format)
        let event_loop_windows_buffer = self.device.new_buffer(
            (MAX_WINDOWS * std::mem::size_of::<EventLoopWindow>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        self.event_loop_handle = Some(EventLoopHandle {
            state_buffer,
            hit_result_buffer,
            event_loop_pipeline,
            hit_test_pipeline,
            hit_result_handler_pipeline,
            window_move_pipeline,
            window_resize_pipeline,
            running: Arc::new(AtomicBool::new(true)),
        });

        self.event_loop_windows_buffer = Some(event_loop_windows_buffer);

        Ok(())
    }

    /// Check if GPU event loop is initialized
    pub fn has_gpu_event_loop(&self) -> bool {
        self.event_loop_handle.is_some()
    }

    /// Push a mouse move event to the GPU input queue
    pub fn push_mouse_move(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
        self.input_handler.push_mouse_move(x, y, dx, dy);
    }

    /// Push a mouse button event to the GPU input queue
    pub fn push_mouse_button(&mut self, button: u16, pressed: bool, x: f32, y: f32) {
        self.input_handler.push_mouse_button(button, pressed, x, y);
    }

    /// Push a key event to the GPU input queue
    pub fn push_key(&mut self, keycode: u16, pressed: bool, modifiers: u32) {
        self.input_handler.push_key(keycode, pressed, modifiers);
    }

    /// Sync desktop windows to the event loop's GpuWindow format
    fn sync_windows_to_event_loop(&self) {
        if let Some(ref buffer) = self.event_loop_windows_buffer {
            let ptr = buffer.contents() as *mut EventLoopWindow;
            for i in 0..self.state.window_count as usize {
                let win = &self.state.windows[i];
                // Use z_order if set, otherwise use index+1 to ensure unique non-zero values
                let effective_z_order = if win.z_order > 0 { win.z_order } else { (i as u32) + 1 };
                let gpu_win = EventLoopWindow {
                    x: win.x,
                    y: win.y,
                    width: win.width,
                    height: win.height,
                    z_order: effective_z_order,
                    flags: if win.flags & WINDOW_FLAG_VISIBLE != 0 {
                        el_window_flags::VISIBLE
                    } else { 0 } |
                    if win.flags & WINDOW_FLAG_MINIMIZED != 0 {
                        el_window_flags::MINIMIZED
                    } else { 0 } |
                    if win.flags & WINDOW_FLAG_MAXIMIZED != 0 {
                        el_window_flags::MAXIMIZED
                    } else { 0 } |
                    if win.flags & WINDOW_FLAG_FOCUSED != 0 {
                        el_window_flags::FOCUSED
                    } else { 0 },
                    _padding: [0; 2],
                };
                unsafe {
                    *ptr.add(i) = gpu_win;
                }
            }
        }
    }

    /// Sync window positions from event loop back to desktop state
    /// Only syncs the window being actively dragged or resized to avoid
    /// overwriting other windows with potentially stale GPU data
    fn sync_windows_from_event_loop(&mut self, drag_window_idx: u32, resize_window_idx: u32) {
        if let Some(ref buffer) = self.event_loop_windows_buffer {
            let ptr = buffer.contents() as *const EventLoopWindow;

            // Only sync the actively modified window
            if drag_window_idx != INVALID_WINDOW {
                let i = drag_window_idx as usize;
                if i < self.state.window_count as usize {
                    let gpu_win = unsafe { *ptr.add(i) };
                    let win = &mut self.state.windows[i];
                    win.x = gpu_win.x;
                    win.y = gpu_win.y;
                    win.update_content_area();
                }
            }

            if resize_window_idx != INVALID_WINDOW {
                let i = resize_window_idx as usize;
                if i < self.state.window_count as usize {
                    let gpu_win = unsafe { *ptr.add(i) };
                    let win = &mut self.state.windows[i];
                    win.x = gpu_win.x;
                    win.y = gpu_win.y;
                    win.width = gpu_win.width;
                    win.height = gpu_win.height;
                    win.update_content_area();
                }
            }
        }
    }

    /// Process GPU events and handle dispatches
    ///
    /// This is the main GPU event loop processing method. Call this each frame
    /// to process input events on the GPU.
    ///
    /// Returns true if a render is needed.
    pub fn process_gpu_events(&mut self) -> bool {
        // Clone the handle to avoid borrow conflicts
        let handle = match self.event_loop_handle.clone() {
            Some(h) => h,
            None => return false,
        };

        if !handle.is_running() {
            return false;
        }

        // Sync windows to GPU
        self.sync_windows_to_event_loop();

        let windows_buffer = self.event_loop_windows_buffer.as_ref().unwrap();
        let window_count = self.state.window_count;

        // Read what GPU wants to do next
        let el_state = handle.read_state();
        let next_dispatch = el_state.next_dispatch;

        // Create window count buffer
        let window_count_buffer = self.device.new_buffer_with_data(
            &window_count as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();

        match next_dispatch {
            d if d == dispatch::HIT_TEST => {
                // Reset dispatch first to prevent re-triggering
                unsafe {
                    let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                    (*state_ptr).next_dispatch = dispatch::NONE;
                }

                // Parallel hit test
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&handle.hit_test_pipeline);
                encoder.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder.set_buffer(1, Some(windows_buffer), 0);
                encoder.set_buffer(2, Some(&handle.hit_result_buffer), 0);
                encoder.set_buffer(3, Some(&window_count_buffer), 0);
                let threads = (window_count as u64).max(1);
                encoder.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(64.min(threads), 1, 1));
                encoder.end_encoding();

                // Handle hit result
                let encoder2 = command_buffer.new_compute_command_encoder();
                encoder2.set_compute_pipeline_state(&handle.hit_result_handler_pipeline);
                encoder2.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder2.set_buffer(1, Some(&handle.hit_result_buffer), 0);
                encoder2.set_buffer(2, Some(windows_buffer), 0);
                encoder2.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder2.end_encoding();
            }
            d if d == dispatch::WINDOW_MOVE => {
                // Reset dispatch first
                unsafe {
                    let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                    (*state_ptr).next_dispatch = dispatch::NONE;
                }

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&handle.window_move_pipeline);
                encoder.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder.set_buffer(1, Some(windows_buffer), 0);
                // Screen size buffer
                let screen_size: [f32; 2] = [self.state.screen_width, self.state.screen_height];
                let screen_buffer = self.device.new_buffer_with_data(
                    screen_size.as_ptr() as *const _,
                    8,
                    MTLResourceOptions::StorageModeShared,
                );
                encoder.set_buffer(2, Some(&screen_buffer), 0);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
            }
            d if d == dispatch::WINDOW_RESIZE => {
                // Reset dispatch first
                unsafe {
                    let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                    (*state_ptr).next_dispatch = dispatch::NONE;
                }

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&handle.window_resize_pipeline);
                encoder.set_buffer(0, Some(&handle.state_buffer), 0);
                encoder.set_buffer(1, Some(windows_buffer), 0);
                encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
                encoder.end_encoding();
            }
            d if d == dispatch::APP_INPUT => {
                // CPU handles app input dispatch
                let window_idx = el_state.dispatch_param as usize;
                if window_idx < self.state.window_count as usize {
                    let win = &self.state.windows[window_idx];
                    let window_id = win.id;

                    // Create app input event
                    let event = AppInputEvent {
                        event_type: if el_state.mouse_buttons != 0 {
                            AppEventType::MouseDown
                        } else {
                            AppEventType::MouseMove
                        },
                        key_code: 0,
                        mouse_x: el_state.mouse_x - win.x - win.content_x,
                        mouse_y: el_state.mouse_y - win.y - win.content_y,
                        mouse_button: 0,
                        modifiers: KeyModifiers::default(),
                    };
                    self.apps.dispatch_input(window_id, &event);
                }
                // Reset dispatch
                unsafe {
                    let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                    (*state_ptr).next_dispatch = dispatch::NONE;
                }
            }
            d if d == dispatch::BRING_TO_FRONT => {
                // GPU requests to bring window to front
                // IMPORTANT: Do NOT call focus_window() here as it reorders the windows array
                // which would break GPU's drag_window index mapping.
                // Instead, just update flags without reordering.
                let window_idx = el_state.dispatch_param as usize;
                if window_idx < self.state.window_count as usize {
                    // Clear focused flag on previous window
                    if self.state.focused_window != 0 {
                        for i in 0..self.state.window_count as usize {
                            if self.state.windows[i].id == self.state.focused_window {
                                self.state.windows[i].flags &= !WINDOW_FLAG_FOCUSED;
                                break;
                            }
                        }
                    }
                    // Set focused flag on new window (without reordering)
                    self.state.windows[window_idx].flags |= WINDOW_FLAG_FOCUSED;
                    self.state.focused_window = self.state.windows[window_idx].id;

                    // Update z_order to be highest WITHOUT reordering the array
                    let max_z = self.state.windows[..self.state.window_count as usize]
                        .iter()
                        .map(|w| w.z_order)
                        .max()
                        .unwrap_or(0);
                    self.state.windows[window_idx].z_order = max_z + 1;
                }
                // Reset dispatch
                unsafe {
                    let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                    (*state_ptr).next_dispatch = dispatch::NONE;
                }
            }
            d if d == dispatch::WINDOW_CLOSE => {
                // GPU requests to close window and clean up app
                let window_idx = el_state.dispatch_param as usize;
                if window_idx < self.state.window_count as usize {
                    let window_id = self.state.windows[window_idx].id;

                    // Close the app (frees Rust memory, calls on_close)
                    self.apps.close_by_window(window_id);

                    // Hide the window
                    self.state.windows[window_idx].flags &= !WINDOW_FLAG_VISIBLE;

                    // Clear focused state if this was the focused window
                    if self.state.focused_window == window_id {
                        self.state.focused_window = 0;
                    }
                }
                // Reset dispatch
                unsafe {
                    let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                    (*state_ptr).next_dispatch = dispatch::NONE;
                }
            }
            _ => {
                // DISPATCH_NONE or unhandled - reset to prevent stuck state
                if next_dispatch != dispatch::NONE {
                    unsafe {
                        let state_ptr = handle.state_buffer.contents() as *mut GpuEventLoopState;
                        (*state_ptr).next_dispatch = dispatch::NONE;
                    }
                }
            }
        }

        // Re-dispatch event loop kernel
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&handle.event_loop_pipeline);
            encoder.set_buffer(0, Some(self.input_handler.buffer()), 0);
            encoder.set_buffer(1, Some(&handle.state_buffer), 0);
            encoder.set_buffer(2, Some(windows_buffer), 0);
            encoder.set_buffer(3, Some(&window_count_buffer), 0);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read state to get drag/resize window indices
        let el_state = handle.read_state();

        // Sync window positions back from GPU (only for actively modified windows)
        self.sync_windows_from_event_loop(el_state.drag_window, el_state.resize_window);

        // Update focus state WITHOUT calling focus_window() which reorders the array
        // Only update if focused_window changed and no drag/resize is active
        if el_state.focused_window != INVALID_WINDOW
            && el_state.drag_window == INVALID_WINDOW
            && el_state.resize_window == INVALID_WINDOW
        {
            let idx = el_state.focused_window as usize;
            if idx < self.state.window_count as usize {
                let window_id = self.state.windows[idx].id;
                // Only update if focus actually changed
                if self.state.focused_window != window_id {
                    // Clear old focus flag
                    for i in 0..self.state.window_count as usize {
                        if self.state.windows[i].id == self.state.focused_window {
                            self.state.windows[i].flags &= !WINDOW_FLAG_FOCUSED;
                            break;
                        }
                    }
                    // Set new focus flag (without reordering)
                    self.state.windows[idx].flags |= WINDOW_FLAG_FOCUSED;
                    self.state.focused_window = window_id;
                }
            }
        }

        // Return whether frame needs render
        el_state.frame_dirty != 0
    }

    /// Stop the GPU event loop
    pub fn stop_gpu_event_loop(&mut self) {
        if let Some(ref handle) = self.event_loop_handle {
            handle.stop();
        }
    }
}
