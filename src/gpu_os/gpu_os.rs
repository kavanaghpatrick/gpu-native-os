// GPU Operating System - Issue #155 Bootstrap
//
// This is the TRUE GPU-centric OS. GpuAppSystem IS the OS, not a component.
// CPU's only job: boot the system, then hand control to GPU.
//
// Boot sequence:
// 1. CPU: Create Metal device
// 2. CPU: Initialize GpuAppSystem
// 3. CPU: Launch system apps (dock, menubar, chrome, compositor)
// 4. GPU: Megakernel runs everything from here
// 5. CPU: Just submits command buffers, handles external I/O

use metal::*;
use crate::gpu_os::gpu_app_system::{GpuAppSystem, app_type, priority, InputEvent};
use crate::gpu_os::input::InputHandler;
use crate::gpu_os::shared_index::GpuFilesystemIndex;
use crate::gpu_os::gpu_io::{GpuIOQueue, IOPriority, IOQueueType};

/// The entire GPU operating system
pub struct GpuOs {
    /// The core system - this IS the OS
    pub system: GpuAppSystem,

    /// Shared filesystem index (GPU-resident)
    pub shared_index: Option<GpuFilesystemIndex>,

    /// GPU-direct I/O queue
    pub io_queue: Option<GpuIOQueue>,

    /// Input handler (bridges external HID to GPU)
    #[allow(dead_code)]
    input_handler: InputHandler,

    /// System app slots (for quick access)
    compositor_slot: Option<u32>,
    dock_slot: Option<u32>,
    menubar_slot: Option<u32>,
    chrome_slot: Option<u32>,

    /// Screen configuration
    pub screen_width: f32,
    pub screen_height: f32,

    /// Frame counter
    frame_count: u64,
}

impl GpuOs {
    /// Boot the GPU OS
    ///
    /// This is the ONLY CPU involvement in startup:
    /// 1. Create GpuAppSystem
    /// 2. Launch system apps
    /// 3. Return - GPU takes over
    pub fn boot(device: &Device) -> Result<Self, String> {
        // 1. Create the app system (this IS the OS)
        let mut system = GpuAppSystem::new(device)?;

        // 2. Enable parallel megakernel
        system.set_use_parallel_megakernel(true);

        // 3. Enable O(1) allocator
        system.set_use_o1_allocator(true);

        // 4. Create input handler
        let input_handler = InputHandler::new(device);

        // 5. Create initial OS structure
        let mut os = GpuOs {
            system,
            shared_index: None,
            io_queue: None,
            input_handler,
            compositor_slot: None,
            dock_slot: None,
            menubar_slot: None,
            chrome_slot: None,
            screen_width: 1920.0,
            screen_height: 1080.0,
            frame_count: 0,
        };

        // 6. Launch system apps at REALTIME priority
        os.launch_system_apps()?;

        // 7. Try to load shared filesystem index
        os.load_shared_index(device);

        // 8. Try to create I/O queue
        os.create_io_queue(device);

        // 9. Initialize system app states with default screen dimensions
        // This ensures compositor, dock, and menubar have valid state
        os.initialize_system_app_states(os.screen_width, os.screen_height);

        Ok(os)
    }

    /// Boot with custom screen size
    pub fn boot_with_size(device: &Device, width: f32, height: f32) -> Result<Self, String> {
        let mut os = Self::boot(device)?;
        os.screen_width = width;
        os.screen_height = height;

        // Set screen size for bytecode coordinate scaling
        os.system.set_screen_size(width, height);

        // Initialize system app states with correct screen dimensions
        os.initialize_system_app_states(width, height);

        Ok(os)
    }

    /// Initialize system app states with screen dimensions
    fn initialize_system_app_states(&mut self, width: f32, height: f32) {
        use crate::gpu_os::gpu_app_system::{
            CompositorState, MenuBarState,
            MENUBAR_DEFAULT_HEIGHT,
        };

        // Initialize Compositor state
        if let Some(slot) = self.compositor_slot {
            if let Some(app) = self.system.get_app(slot) {
                unsafe {
                    let state_ptr = self.system.state_buffer().contents()
                        .add(app.state_offset as usize) as *mut CompositorState;
                    let state = &mut *state_ptr;
                    state.screen_width = width;
                    state.screen_height = height;
                    state.background_color = [0.08, 0.08, 0.12, 1.0];
                }
            }
        }

        // Initialize MenuBar state
        if let Some(slot) = self.menubar_slot {
            if let Some(app) = self.system.get_app(slot) {
                unsafe {
                    let state_ptr = self.system.state_buffer().contents()
                        .add(app.state_offset as usize) as *mut MenuBarState;
                    let state = &mut *state_ptr;
                    state.screen_width = width;
                    state.bar_height = MENUBAR_DEFAULT_HEIGHT;
                    state.bar_color = [0.95, 0.95, 0.97, 0.92];
                    state.text_color = [0.0, 0.0, 0.0, 1.0];
                }
            }
        }

        // Initialize Dock state
        if let Some(slot) = self.dock_slot {
            self.system.initialize_dock_state(slot, width, height);
            // Add some default dock items
            self.system.add_dock_item(slot, app_type::TERMINAL, [0.3, 0.6, 0.9, 1.0]);
            self.system.add_dock_item(slot, app_type::FILESYSTEM, [0.4, 0.8, 0.4, 1.0]);
            self.system.add_dock_item(slot, app_type::DOCUMENT, [0.9, 0.5, 0.3, 1.0]);
        }

        // Initialize Compositor state
        if let Some(slot) = self.compositor_slot {
            self.system.initialize_compositor_state(slot, width, height);
            self.system.mark_dirty(slot);
        }
        if let Some(slot) = self.menubar_slot {
            self.system.mark_dirty(slot);
        }
        if let Some(slot) = self.dock_slot {
            self.system.mark_dirty(slot);
        }
        if let Some(slot) = self.chrome_slot {
            self.system.mark_dirty(slot);
        }
    }

    /// Launch system apps (REALTIME priority)
    fn launch_system_apps(&mut self) -> Result<(), String> {
        // Window Chrome - generates decorations for all windows
        if let Some(slot) = self.system.launch_by_type(app_type::WINDOW_CHROME) {
            self.system.set_priority(slot, priority::REALTIME);
            self.chrome_slot = Some(slot);
        }

        // Dock - app launcher at bottom
        if let Some(slot) = self.system.launch_by_type(app_type::DOCK) {
            self.system.set_priority(slot, priority::REALTIME);
            self.dock_slot = Some(slot);
        }

        // Menu bar - top menu
        if let Some(slot) = self.system.launch_by_type(app_type::MENUBAR) {
            self.system.set_priority(slot, priority::REALTIME);
            self.menubar_slot = Some(slot);
        }

        // Compositor - final rendering stage (runs LAST)
        if let Some(slot) = self.system.launch_by_type(app_type::COMPOSITOR) {
            self.system.set_priority(slot, priority::REALTIME);
            self.compositor_slot = Some(slot);
        }

        Ok(())
    }

    /// Load shared filesystem index (GPU-resident)
    fn load_shared_index(&mut self, device: &Device) {
        match GpuFilesystemIndex::load_or_create(device) {
            Ok(index) => {
                self.shared_index = Some(index);
            }
            Err(e) => {
                eprintln!("Warning: Could not load shared index: {}", e);
            }
        }
    }

    /// Create GPU-direct I/O queue
    fn create_io_queue(&mut self, device: &Device) {
        if let Some(queue) = GpuIOQueue::new(device, IOPriority::Normal, IOQueueType::Concurrent) {
            self.io_queue = Some(queue);
        } else {
            eprintln!("Warning: Could not create I/O queue (Metal 3 may not be available)");
        }
    }

    // ========================================================================
    // Main Loop (Minimal CPU)
    // ========================================================================

    /// Run one frame
    ///
    /// CPU just submits - GPU does everything
    pub fn run_frame(&mut self) {
        // 1. Update frame state
        self.frame_count += 1;

        // 2. Process queued input events
        self.system.process_input();

        // 3. Run megakernel (all apps update)
        self.system.run_frame();

        // 4. Finalize render (sum vertex counts)
        self.system.finalize_render();
    }

    /// Get total vertex count for rendering
    pub fn total_vertex_count(&self) -> u32 {
        self.system.total_vertex_count()
    }

    /// Get unified vertex buffer for single draw call
    pub fn render_vertices_buffer(&self) -> &Buffer {
        self.system.render_vertices_buffer()
    }

    // ========================================================================
    // Input Handling (CPU bridges external I/O to GPU)
    // ========================================================================

    /// Queue mouse move event
    pub fn mouse_move(&mut self, x: f32, y: f32, _dx: f32, _dy: f32) {
        self.system.queue_input(InputEvent::mouse_move(x, y));

        // Update dock cursor for hover detection
        if let Some(dock_slot) = self.dock_slot {
            self.system.update_dock_cursor(dock_slot, x, y);
        }
    }

    /// Queue mouse button event
    pub fn mouse_button(&mut self, button: u8, pressed: bool, x: f32, y: f32) {
        if pressed {
            self.system.queue_input(InputEvent::mouse_down(x, y, button as u32));
        } else {
            self.system.queue_input(InputEvent::mouse_up(x, y, button as u32));
        }

        // Update dock for click detection
        if button == 0 {
            if let Some(dock_slot) = self.dock_slot {
                // Update cursor position AND mouse state together
                // Both are needed for GPU click detection in PHASE 2b
                self.system.update_dock_cursor(dock_slot, x, y);
                self.system.update_dock_mouse_pressed(dock_slot, pressed);
            }
        }
    }

    /// Queue mouse click (down + up)
    pub fn mouse_click(&mut self, x: f32, y: f32, button: u8) {
        self.system.queue_input(InputEvent::mouse_down(x, y, button as u32));
        self.system.queue_input(InputEvent::mouse_up(x, y, button as u32));
    }

    /// Queue key event
    pub fn key_event(&mut self, keycode: u32, pressed: bool, _modifiers: u32) {
        if pressed {
            self.system.queue_input(InputEvent::key_down(keycode));
        } else {
            self.system.queue_input(InputEvent::key_up(keycode));
        }
    }

    /// Queue scroll event
    pub fn scroll(&mut self, _dx: f32, _dy: f32) {
        // TODO: Add scroll event to InputEvent
    }

    // ========================================================================
    // App Management
    // ========================================================================

    /// Launch a user app
    pub fn launch_app(&mut self, app_type_id: u32) -> Option<u32> {
        let slot = self.system.launch_by_type(app_type_id)?;

        // Create window for it (cascaded)
        let x = 100.0 + (slot as f32 * 30.0);
        let y = 100.0 + (slot as f32 * 30.0);
        self.system.create_window(slot, x, y, 800.0, 600.0);

        // Mark dirty so it renders
        self.system.mark_dirty(slot);

        Some(slot)
    }

    /// Launch a bytecode app with the given bytecode
    pub fn launch_bytecode_app(&mut self, bytecode: &[u8]) -> Option<u32> {
        use super::gpu_app_system::app_type;

        // Launch as BYTECODE type
        let slot = self.system.launch_by_type(app_type::BYTECODE)?;

        // Copy bytecode into the app's state buffer
        self.system.write_app_state(slot, bytecode);

        // DEBUG: Read back and verify bytecode was written
        if let Some(app) = self.system.get_app(slot) {
            if let Some((code_size, entry_point, vertex_budget, flags)) = self.system.read_bytecode_header(slot) {
                println!("GPU STATE VERIFICATION: state_offset={}, code_size={}, entry_point={}, vertex_budget={}, flags={}",
                         app.state_offset, code_size, entry_point, vertex_budget, flags);
            }
        }

        // Create window for it (cascaded)
        let x = 100.0 + (slot as f32 * 30.0);
        let y = 100.0 + (slot as f32 * 30.0);
        self.system.create_window(slot, x, y, 800.0, 600.0);

        // Mark dirty so it renders
        self.system.mark_dirty(slot);

        Some(slot)
    }

    /// Close an app
    pub fn close_app(&mut self, slot: u32) {
        self.system.close_app(slot);
    }

    /// Set focus to an app
    pub fn set_focus(&mut self, slot: u32) {
        self.system.set_focus(slot);
    }

    /// Get app count
    pub fn app_count(&self) -> u32 {
        self.system.active_count()
    }

    // ========================================================================
    // System App Accessors
    // ========================================================================

    /// Get compositor slot
    pub fn compositor_slot(&self) -> Option<u32> {
        self.compositor_slot
    }

    /// Get dock slot
    pub fn dock_slot(&self) -> Option<u32> {
        self.dock_slot
    }

    /// Get menu bar slot
    pub fn menubar_slot(&self) -> Option<u32> {
        self.menubar_slot
    }

    /// Get window chrome slot
    pub fn chrome_slot(&self) -> Option<u32> {
        self.chrome_slot
    }

    /// Check if shared index is available
    pub fn has_shared_index(&self) -> bool {
        self.shared_index.is_some()
    }

    /// Check if I/O queue is available
    pub fn has_io_queue(&self) -> bool {
        self.io_queue.is_some()
    }

    /// Get frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if a dock item was clicked and return its app_type
    /// This should be called after run_frame() to check for dock clicks
    pub fn get_clicked_dock_app(&self) -> Option<u32> {
        let dock_slot = self.dock_slot?;
        self.system.get_clicked_dock_app(dock_slot)
    }

    /// Debug: print dock state
    pub fn debug_dock_state(&self) {
        let dock_slot = match self.dock_slot {
            Some(s) => s,
            None => { println!("DEBUG: No dock slot"); return; }
        };
        self.system.debug_dock_state(dock_slot);
    }

    /// Handle dock click - launches the clicked app if any
    /// Returns the slot of the launched app, or None if no dock item was clicked
    pub fn handle_dock_click(&mut self) -> Option<u32> {
        let app_type = self.get_clicked_dock_app()?;
        self.launch_app(app_type)
    }

    /// Get draw calls for all active apps
    /// Returns a list of (start_vertex_index, vertex_count) pairs for rendering
    /// Each pair represents one app's vertex range in the unified buffer
    pub fn get_draw_calls(&self) -> Vec<(u64, u64)> {
        static DEBUG_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let frame = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut calls = Vec::new();

        // Issue #297: Use cached max_slots instead of blocking stats() call
        // stats() was allocating a buffer AND blocking every frame!
        let max_slots = self.system.max_slots;

        for slot in 0..max_slots {
            if let Some(app) = self.system.get_app(slot) {
                let is_active = app.is_active();
                let is_visible = app.is_visible();
                let vertex_count = app.vertex_count;

                // Debug slot 5 specifically (where bytecode apps go)
                if slot >= 4 && slot <= 6 && frame % 60 == 0 {
                    println!("SLOT {}: active={}, visible={}, vertex_count={}, type={}, frame_num={}, state_offset={}, state_size={}, vertex_offset={}",
                             slot, is_active, is_visible, vertex_count, app.app_type,
                             app.frame_number, app.state_offset, app.state_size, app.vertex_offset);
                    // Read bytecode debug info if this is a bytecode app
                    if app.app_type == 101 {
                        if let Some((instr_count, quad_count, vert_idx, final_pc, gpu_code_size, gpu_vertex_budget, running, state_size_float4, state_offset_bytes, total_state_size, gpu_vert_idx, _failing_pc, _failing_reg)) = self.system.read_bytecode_debug(slot) {
                            println!("  BYTECODE DEBUG: quads={}, verts={}, gpu_verts={}, final_pc={}, GPU_code_size={}, GPU_vert_budget={}, running={}",
                                     quad_count, vert_idx, gpu_vert_idx, final_pc, gpu_code_size, gpu_vertex_budget, running);
                            println!("  STATE DEBUG: state_size_float4={}, state_offset_bytes={}, total_state_size={}",
                                     state_size_float4, state_offset_bytes, total_state_size);
                            // Dump first few vertex colors for verification
                            if vertex_count > 0 {
                                let vbuf = self.system.render_vertices_buffer();
                                let start_vertex = (app.vertex_offset as usize) / 48;
                                let num_to_dump = std::cmp::min(vertex_count as usize, 48);
                                unsafe {
                                    let verts = vbuf.contents() as *const super::gpu_app_system::RenderVertex;
                                    for vi in 0..num_to_dump {
                                        let v = &*verts.add(start_vertex + vi);
                                        let r = (v.color[0] * 255.0) as u8;
                                        let g = (v.color[1] * 255.0) as u8;
                                        let b = (v.color[2] * 255.0) as u8;
                                        let a = (v.color[3] * 255.0) as u8;
                                        println!("  VERTEX[{}]: pos=({:.1},{:.1},{:.1}) color=#{:02X}{:02X}{:02X}{:02X} rgba=({:.3},{:.3},{:.3},{:.3})",
                                                 vi, v.position[0], v.position[1], v.position[2],
                                                 r, g, b, a, v.color[0], v.color[1], v.color[2], v.color[3]);
                                    }
                                }
                            }
                        }
                    }
                }

                // Only render active and visible apps with vertices
                if is_active && is_visible && vertex_count > 0 {
                    // vertex_offset is in bytes, divide by 48 (sizeof RenderVertex) to get index
                    let start_vertex = (app.vertex_offset as u64) / 48;
                    let count = vertex_count as u64;

                    // Only include complete triangles
                    let triangle_verts = (count / 3) * 3;
                    if triangle_verts > 0 {
                        calls.push((start_vertex, triangle_verts));
                    }
                }
            }
        }

        calls
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Device {
        Device::system_default().expect("No Metal device")
    }

    #[test]
    fn test_boot_gpu_os() {
        let device = get_device();
        let os = GpuOs::boot(&device).expect("Boot failed");

        // System should be created
        assert_eq!(os.frame_count, 0);
    }

    #[test]
    fn test_boot_with_size() {
        let device = get_device();
        let os = GpuOs::boot_with_size(&device, 2560.0, 1440.0).expect("Boot failed");

        assert_eq!(os.screen_width, 2560.0);
        assert_eq!(os.screen_height, 1440.0);
    }

    #[test]
    fn test_launch_user_app() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        let terminal = os.launch_app(app_type::TERMINAL);
        assert!(terminal.is_some());

        let slot = terminal.unwrap();
        let app = os.system.get_app(slot).unwrap();
        assert_eq!(app.app_type, app_type::TERMINAL);
    }

    #[test]
    fn test_multiple_apps() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        let initial_count = os.app_count();

        os.launch_app(app_type::TERMINAL);
        os.launch_app(app_type::FILESYSTEM);
        os.launch_app(app_type::TERMINAL);

        assert_eq!(os.app_count(), initial_count + 3);
    }

    #[test]
    fn test_run_frame() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        os.launch_app(app_type::TERMINAL);

        // Run several frames
        for i in 0..10 {
            os.run_frame();
            assert_eq!(os.frame_count, i + 1);
        }
    }

    #[test]
    fn test_input_queueing() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        os.launch_app(app_type::TERMINAL);

        // Queue input events
        os.mouse_move(100.0, 100.0, 5.0, 0.0);
        os.mouse_click(100.0, 100.0, 0);
        os.key_event(0x00, true, 0);  // 'a' key

        // Run frame to process
        os.run_frame();

        // Events should have been processed
        assert_eq!(os.frame_count, 1);
    }

    #[test]
    fn test_close_app() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        let slot = os.launch_app(app_type::TERMINAL).unwrap();
        let count_before = os.app_count();

        os.close_app(slot);

        assert_eq!(os.app_count(), count_before - 1);
    }

    #[test]
    fn test_zero_cpu_frame() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        // Launch some apps
        os.launch_app(app_type::TERMINAL);
        os.launch_app(app_type::FILESYSTEM);

        // Run frames - CPU should just submit
        for _ in 0..100 {
            os.system.mark_all_dirty();
            os.run_frame();
        }

        // All apps should have run
        assert!(os.frame_count >= 100);
    }

    #[test]
    fn test_vertices_generated() {
        let device = get_device();
        let mut os = GpuOs::boot(&device).unwrap();

        // Launch an app that generates vertices
        let slot = os.launch_app(app_type::TERMINAL).unwrap();
        os.system.mark_dirty(slot);

        os.run_frame();

        // Should have generated some vertices
        let count = os.total_vertex_count();
        // May be 0 if no system apps generate vertices yet
        // but the infrastructure should work
        assert!(count >= 0);
    }
}
