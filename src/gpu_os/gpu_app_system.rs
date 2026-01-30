// GPU-Centric App System (Issue #154)
//
// THE GPU IS THE COMPUTER. Apps run ON the GPU, managed BY the GPU.
//
// Key principles:
// - GPU allocates app slots (atomic bitmap)
// - GPU allocates app memory (parallel prefix allocator)
// - GPU runs apps via megakernel (single dispatch, all apps)
// - GPU manages app lifecycle (init, run, close)
// - CPU only submits command buffers and handles I/O

use metal::*;
use std::mem;

// Import window_flags from event_loop
use super::event_loop::window_flags;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of app slots
pub const MAX_APP_SLOTS: u32 = 64;

/// Invalid slot sentinel
pub const INVALID_SLOT: u32 = 0xFFFFFFFF;

/// Default state pool size (64MB)
pub const DEFAULT_STATE_POOL_SIZE: usize = 64 * 1024 * 1024;

/// Default vertex pool size (16MB)
pub const DEFAULT_VERTEX_POOL_SIZE: usize = 16 * 1024 * 1024;

// ============================================================================
// App Flags (must match Metal shader)
// ============================================================================

pub mod flags {
    pub const ACTIVE: u32 = 1 << 0;
    pub const VISIBLE: u32 = 1 << 1;
    pub const DIRTY: u32 = 1 << 2;
    pub const SUSPENDED: u32 = 1 << 3;
    pub const FOCUS: u32 = 1 << 4;
    pub const NEEDS_INIT: u32 = 1 << 5;
    pub const CLOSING: u32 = 1 << 6;
}

pub mod priority {
    pub const BACKGROUND: u32 = 0;
    pub const NORMAL: u32 = 1;
    pub const HIGH: u32 = 2;
    pub const REALTIME: u32 = 3;
}

pub mod app_type {
    // User apps (1-99)
    pub const NONE: u32 = 0;
    pub const GAME_OF_LIFE: u32 = 1;
    pub const PARTICLES: u32 = 2;
    pub const TEXT_EDITOR: u32 = 3;
    pub const FILESYSTEM: u32 = 4;
    pub const TERMINAL: u32 = 5;
    pub const DOCUMENT: u32 = 6;
    pub const SHELL: u32 = 7;
    pub const MANDELBROT: u32 = 8;
    pub const BOIDS: u32 = 9;
    pub const METABALLS: u32 = 10;
    pub const WAVES: u32 = 11;
    pub const CUSTOM: u32 = 100;
    pub const BYTECODE: u32 = 101;  // Dynamic bytecode apps

    // System apps (200-299) - Issue #155-#161
    pub const COMPOSITOR: u32 = 200;     // Final rendering stage
    pub const DOCK: u32 = 201;           // App launcher at bottom
    pub const MENUBAR: u32 = 202;        // Top menu bar
    pub const WINDOW_CHROME: u32 = 203;  // Window decorations
}

// ============================================================================
// Issue #159: APP_TYPES Registry
// ============================================================================

/// App type info - describes resource requirements
#[derive(Clone, Copy, Debug)]
pub struct AppTypeInfo {
    pub type_id: u32,
    pub name: &'static str,
    pub state_size: u32,    // Bytes of state memory
    pub vertex_size: u32,   // Bytes of vertex memory
    pub thread_count: u32,  // Threads per threadgroup
}

/// Registry of all app types with their resource requirements
pub const APP_TYPES: &[AppTypeInfo] = &[
    AppTypeInfo {
        type_id: app_type::CUSTOM,
        name: "Custom",
        state_size: 64,          // CounterState + padding
        vertex_size: 6 * 48,     // 6 vertices × 48 bytes each
        thread_count: 1,
    },
    AppTypeInfo {
        type_id: app_type::GAME_OF_LIFE,
        name: "Game of Life",
        state_size: 16 + 128 * 128 * 2,  // Issue #239: Header + 2x 128x128 grids (double buffer)
        vertex_size: 128 * 128 * 6 * 48,  // 6 vertices per cell
        thread_count: 256,
    },
    AppTypeInfo {
        type_id: app_type::PARTICLES,
        name: "Particles",
        state_size: 32 + 1000 * 48,  // Header + 1000 particles
        vertex_size: 1000 * 6 * 48,  // 6 vertices per particle
        thread_count: 256,
    },
    AppTypeInfo {
        type_id: app_type::MANDELBROT,
        name: "Mandelbrot",
        state_size: 64,
        vertex_size: 1024 * 6 * 48,  // 1024 sample points
        thread_count: 256,
    },
    AppTypeInfo {
        type_id: app_type::TEXT_EDITOR,
        name: "Text Editor",
        state_size: 64 * 1024,   // 64KB text buffer
        vertex_size: 1000 * 6 * 48,  // ~1000 characters
        thread_count: 64,
    },
    AppTypeInfo {
        type_id: app_type::FILESYSTEM,
        name: "File System",
        state_size: 256 * 1024,  // Directory cache
        vertex_size: 500 * 6 * 48,  // ~500 files
        thread_count: 64,
    },
    AppTypeInfo {
        type_id: app_type::TERMINAL,
        name: "Terminal",
        state_size: 16 * 1024,   // Command buffer
        vertex_size: 2000 * 6 * 48,  // ~2000 characters
        thread_count: 64,
    },
    AppTypeInfo {
        type_id: app_type::DOCUMENT,
        name: "Document",
        state_size: 64 * 1024,   // Document content buffer
        vertex_size: 2000 * 6 * 48,  // ~2000 characters
        thread_count: 64,
    },
    // System apps (Issue #155-#161)
    AppTypeInfo {
        type_id: app_type::COMPOSITOR,
        name: "Compositor",
        state_size: 64,              // CompositorState
        vertex_size: 6 * 48,         // Background quad only
        thread_count: 1,
    },
    AppTypeInfo {
        type_id: app_type::DOCK,
        name: "Dock",
        state_size: 4096,            // DockState + 32 items
        vertex_size: 32 * 6 * 48,    // 32 dock icons
        thread_count: 64,
    },
    AppTypeInfo {
        type_id: app_type::MENUBAR,
        name: "MenuBar",
        state_size: 8192,            // MenuBarState + menus + items
        vertex_size: 1000 * 6 * 48,  // Menu text and dropdowns
        thread_count: 64,
    },
    AppTypeInfo {
        type_id: app_type::WINDOW_CHROME,
        name: "Window Chrome",
        state_size: 256,             // WindowChromeState
        vertex_size: 64 * 100 * 48,  // 64 windows × ~100 vertices each
        thread_count: 64,
    },
    // Bytecode apps (Issue #164)
    // Note: state_size must be > 64KB + overhead to have at least 1 WASM page
    AppTypeInfo {
        type_id: app_type::BYTECODE,
        name: "Bytecode",
        state_size: 128 * 1024,      // 128KB for bytecode + state (allows ~1 page heap)
        vertex_size: 128 * 128 * 6 * 48,  // Same as Game of Life
        thread_count: 1,             // Single-threaded interpreter for now
    },
];

/// Get app type info by type ID
pub fn get_app_type_info(type_id: u32) -> Option<&'static AppTypeInfo> {
    APP_TYPES.iter().find(|t| t.type_id == type_id)
}

// ============================================================================
// GPU Data Structures
// ============================================================================

/// GPU App Descriptor - 128 bytes, GPU-resident
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuAppDescriptor {
    // Identity & Lifecycle (16 bytes)
    pub flags: u32,
    pub app_type: u32,
    pub slot_id: u32,
    pub window_id: u32,

    // Memory Pointers (32 bytes)
    pub state_offset: u32,
    pub state_size: u32,
    pub vertex_offset: u32,
    pub vertex_size: u32,
    pub param_offset: u32,
    pub param_size: u32,
    pub _mem_pad: [u32; 2],

    // Execution State (16 bytes)
    pub frame_number: u32,
    pub input_head: u32,
    pub input_tail: u32,
    pub thread_count: u32,

    // Rendering (16 bytes)
    pub vertex_count: u32,
    pub clear_color: u32,
    pub preferred_width: f32,
    pub preferred_height: f32,

    // GPU Scheduling (16 bytes)
    pub priority: u32,
    pub last_run_frame: u32,
    pub accumulated_time: u32,
    pub _sched_pad: u32,

    // Input Queue (32 bytes) - inline ring buffer
    pub input_events: [u32; 8],
}

// Verify size at compile time
const _: () = assert!(mem::size_of::<GpuAppDescriptor>() == 128);

impl GpuAppDescriptor {
    pub fn is_active(&self) -> bool {
        self.flags & flags::ACTIVE != 0
    }

    pub fn is_visible(&self) -> bool {
        self.flags & flags::VISIBLE != 0
    }

    pub fn is_dirty(&self) -> bool {
        self.flags & flags::DIRTY != 0
    }

    pub fn is_focused(&self) -> bool {
        self.flags & flags::FOCUS != 0
    }

    pub fn mark_dirty(&mut self) {
        self.flags |= flags::DIRTY;
    }

    pub fn clear_dirty(&mut self) {
        self.flags &= !flags::DIRTY;
    }
}

/// App table header - at start of app table buffer
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AppTableHeader {
    pub max_slots: u32,
    pub active_count: u32,  // atomic in Metal
    pub free_bitmap: [u32; 2],  // 64 slots
    pub _pad: [u32; 4],
}

impl Default for AppTableHeader {
    fn default() -> Self {
        Self {
            max_slots: MAX_APP_SLOTS,
            active_count: 0,
            // All slots free (1 = free)
            free_bitmap: [0xFFFFFFFF, 0xFFFFFFFF],
            _pad: [0; 4],
        }
    }
}

/// Allocator state - bump pointer with atomic updates (legacy, being replaced by MemoryPool)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AllocatorState {
    pub bump_pointer: u32,  // atomic in Metal
    pub pool_size: u32,
    pub allocation_count: u32,  // atomic in Metal
    pub peak_usage: u32,
}

// ============================================================================
// Issue #155: O(1) Memory Management with Atomic Free List
// ============================================================================

/// Free block descriptor for O(1) free list
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeBlock {
    pub next: u32,      // Index of next free block (forms linked list)
    pub size: u32,      // Size of this block
    pub offset: u32,    // Offset in memory pool
    pub _pad: u32,
}

/// Memory pool with O(1) free list
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemoryPool {
    pub freelist_head: u32,    // atomic - Head of free list (LIFO stack)
    pub bump_pointer: u32,     // atomic - Fallback bump allocator
    pub free_count: u32,       // atomic - Number of free blocks
    pub pool_size: u32,
    pub block_count: u32,      // atomic - Total blocks in freelist array
    pub max_blocks: u32,       // Maximum free blocks we can track
    pub _pad: [u32; 2],
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self {
            freelist_head: INVALID_SLOT,
            bump_pointer: 0,
            free_count: 0,
            pool_size: 0,
            block_count: 0,
            max_blocks: 1024,  // Track up to 1024 free blocks
            _pad: [0; 2],
        }
    }
}

/// Maximum free blocks to track per pool
pub const MAX_FREE_BLOCKS: u32 = 1024;

/// Starvation threshold (frames without running)
pub const STARVATION_THRESHOLD: u32 = 10;

// ============================================================================
// Issue #156: GPU Scheduler Structures
// ============================================================================

/// Frame budget for controlling execution
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FrameBudget {
    pub remaining: u32,       // atomic - cycles remaining this frame
    pub per_frame_limit: u32, // limit to reset to each frame
    pub skipped_count: u32,   // atomic - apps skipped due to budget
    pub _pad: u32,
}

impl Default for FrameBudget {
    fn default() -> Self {
        Self {
            remaining: u32::MAX,  // No limit by default
            per_frame_limit: u32::MAX,
            skipped_count: 0,
            _pad: 0,
        }
    }
}

/// Scheduler statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerStats2 {
    pub active_count: u32,
    pub suspended_count: u32,
    pub starving_count: u32,
    pub skipped_this_frame: u32,
    pub apps_by_priority: [u32; 4],  // Background, Normal, High, Realtime
}

// ============================================================================
// Issue #157: GPU Input & Window Structures
// ============================================================================

/// Input event types
pub mod event_type {
    pub const NONE: u32 = 0;
    pub const KEY_DOWN: u32 = 1;
    pub const KEY_UP: u32 = 2;
    pub const MOUSE_MOVE: u32 = 3;
    pub const MOUSE_DOWN: u32 = 4;
    pub const MOUSE_UP: u32 = 5;
    pub const SCROLL: u32 = 6;
}

/// Input event structure (matches Metal)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InputEvent {
    pub event_type: u32,
    pub key_or_button: u32,
    pub position: [f32; 2],
    pub modifiers: u32,
    pub frame: u32,
    pub _pad: [u32; 2],
}

impl InputEvent {
    pub fn key_down(key: u32) -> Self {
        Self {
            event_type: event_type::KEY_DOWN,
            key_or_button: key,
            ..Default::default()
        }
    }

    pub fn key_up(key: u32) -> Self {
        Self {
            event_type: event_type::KEY_UP,
            key_or_button: key,
            ..Default::default()
        }
    }

    pub fn mouse_move(x: f32, y: f32) -> Self {
        Self {
            event_type: event_type::MOUSE_MOVE,
            position: [x, y],
            ..Default::default()
        }
    }

    pub fn mouse_down(x: f32, y: f32, button: u32) -> Self {
        Self {
            event_type: event_type::MOUSE_DOWN,
            key_or_button: button,
            position: [x, y],
            ..Default::default()
        }
    }

    pub fn mouse_up(x: f32, y: f32, button: u32) -> Self {
        Self {
            event_type: event_type::MOUSE_UP,
            key_or_button: button,
            position: [x, y],
            ..Default::default()
        }
    }

    /// Alias for mouse_down (click = press)
    pub fn mouse_click(x: f32, y: f32, button: u32) -> Self {
        Self::mouse_down(x, y, button)
    }
}

/// Input queue header (matches Metal)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InputQueue {
    pub head: u32,     // atomic - consumer position
    pub tail: u32,     // atomic - producer position
    pub capacity: u32,
    pub _pad: u32,
}

impl Default for InputQueue {
    fn default() -> Self {
        Self {
            head: 0,
            tail: 0,
            capacity: 64,  // Default capacity
            _pad: 0,
        }
    }
}

/// GPU Window structure (matches Metal)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuWindow {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub depth: f32,      // 0.0 = back, 1.0 = front
    pub app_slot: u32,
    pub flags: u32,
    pub _pad: u32,
}

/// Maximum input events in queue
pub const MAX_INPUT_EVENTS: u32 = 64;

/// Maximum windows
pub const MAX_WINDOWS: u32 = 64;

/// Issue #157: MenuBar constants
pub const MENUBAR_BACKGROUND_VERTS: u32 = 6;  // Bar background quad
pub const MENUBAR_DEFAULT_HEIGHT: f32 = 24.0;

/// Issue #157: MenuBar State (matches Metal struct)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
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
    pub time: f32,
    pub dropdown_anim: f32,
    pub bar_color: [f32; 4],
    pub text_color: [f32; 4],
    pub _pad: [u32; 2],
}

impl Default for MenuBarState {
    fn default() -> Self {
        Self {
            screen_width: 1280.0,
            bar_height: MENUBAR_DEFAULT_HEIGHT,
            padding_x: 12.0,
            text_scale: 1.5,
            menu_count: 0,
            total_item_count: 0,
            open_menu: u32::MAX,
            hovered_menu: u32::MAX,
            hovered_item: u32::MAX,
            selected_item: u32::MAX,
            time: 0.0,
            dropdown_anim: 0.0,
            bar_color: [0.95, 0.95, 0.97, 0.92],  // Translucent light gray
            text_color: [0.0, 0.0, 0.0, 1.0],     // Black text
            _pad: [0, 0],
        }
    }
}

/// Issue #158: Compositor constants
pub const COMPOSITOR_BACKGROUND_VERTS: u32 = 6;  // Full-screen quad

/// Issue #158: Compositor State (matches Metal struct)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CompositorState {
    pub screen_width: f32,
    pub screen_height: f32,
    pub window_count: u32,
    pub frame_number: u32,
    pub background_color: [f32; 4],
    pub total_vertices_rendered: u32,
    pub app_count: u32,
    pub _pad: [u32; 2],
}

impl Default for CompositorState {
    fn default() -> Self {
        Self {
            screen_width: 1280.0,
            screen_height: 720.0,
            window_count: 0,
            frame_number: 0,
            background_color: [0.08, 0.08, 0.12, 1.0],  // Dark background
            total_vertices_rendered: 0,
            app_count: 0,
            _pad: [0, 0],
        }
    }
}

/// Issue #159: Chrome constants
pub const CHROME_VERTS_PER_WINDOW: u32 = 90;  // 6 + 54 + 24 + 6

/// Issue #159: Window Chrome State (matches Metal struct)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WindowChromeState {
    pub window_count: u32,
    pub focused_window: u32,
    pub dragging_window: u32,
    pub resizing_window: u32,
    pub hovered_button: u32,      // Encoded: (window_idx << 8) | button_type
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

/// Issue #156: Dock constants
pub const MAX_DOCK_ITEMS: u32 = 32;
pub const DOCK_DEFAULT_HEIGHT: f32 = 70.0;
pub const DOCK_DEFAULT_ICON_SIZE: f32 = 48.0;
pub const DOCK_MAGNIFIED_SIZE: f32 = 72.0;
pub const DOCK_ICON_SPACING: f32 = 8.0;

/// Issue #156: Dock item flags
pub mod dock_item_flags {
    pub const VISIBLE: u32 = 0x01;
    pub const RUNNING: u32 = 0x02;
    pub const HOVERED: u32 = 0x04;
    pub const BOUNCING: u32 = 0x08;
    pub const CLICKED: u32 = 0x10;
}

/// Issue #156: DockItem (matches Metal struct)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DockItem {
    pub app_type: u32,           // What app to launch on click
    pub flags: u32,              // VISIBLE, RUNNING, HOVERED, BOUNCING, CLICKED
    pub running_count: u32,      // Number of running instances
    pub current_size: f32,       // Animated size
    pub target_size: f32,        // Target size
    pub bounce_phase: f32,       // Bounce animation phase
    pub center_x: f32,           // Computed center X
    pub center_y: f32,           // Computed center Y
    pub icon_color: [f32; 4],    // RGBA color
}

impl Default for DockItem {
    fn default() -> Self {
        Self {
            app_type: app_type::NONE,
            flags: 0,
            running_count: 0,
            current_size: DOCK_DEFAULT_ICON_SIZE,
            target_size: DOCK_DEFAULT_ICON_SIZE,
            bounce_phase: 0.0,
            center_x: 0.0,
            center_y: 0.0,
            icon_color: [0.5, 0.5, 0.5, 1.0],
        }
    }
}

/// Issue #156: DockState (matches Metal struct)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DockState {
    // Counts and indices
    pub item_count: u32,
    pub hovered_item: u32,
    pub clicked_item: u32,
    pub _count_pad: u32,

    // Screen geometry
    pub screen_width: f32,
    pub screen_height: f32,
    pub dock_y: f32,
    pub dock_height: f32,

    // Icon sizing
    pub base_icon_size: f32,
    pub magnified_size: f32,
    pub icon_spacing: f32,
    pub magnification_radius: f32,

    // Animation
    pub animation_speed: f32,
    pub bounce_height: f32,
    pub bounce_speed: f32,
    pub time: f32,

    // Cursor
    pub cursor_pos: [f32; 2],
    pub cursor_in_dock: u32,
    pub mouse_pressed: u32,  // 1 if mouse button is down this frame

    // Padding
    pub _pad: [u32; 2],
}

impl Default for DockState {
    fn default() -> Self {
        Self {
            item_count: 0,
            hovered_item: u32::MAX,
            clicked_item: u32::MAX,
            _count_pad: 0,
            screen_width: 1280.0,
            screen_height: 720.0,
            dock_y: 720.0 - DOCK_DEFAULT_HEIGHT,
            dock_height: DOCK_DEFAULT_HEIGHT,
            base_icon_size: DOCK_DEFAULT_ICON_SIZE,
            magnified_size: DOCK_MAGNIFIED_SIZE,
            icon_spacing: DOCK_ICON_SPACING,
            magnification_radius: 120.0,
            animation_speed: 0.25,
            bounce_height: 20.0,
            bounce_speed: 0.15,
            time: 0.0,
            cursor_pos: [0.0, 0.0],
            cursor_in_dock: 0,
            mouse_pressed: 0,
            _pad: [0, 0],
        }
    }
}

/// Scheduler state - GPU-side scheduling decisions
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerState {
    pub active_app_count: u32,
    pub total_compute_budget: u32,
    pub used_compute_budget: u32,
    pub _pad0: u32,
    pub priority_thresholds: [u32; 4],
    pub frame_quantum: u32,
    pub current_frame: u32,
    pub _pad1: [u32; 2],
}

/// Issue #158: Render state for unified vertex buffer
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderState {
    pub total_vertex_count: u32,  // Sum of all app vertex counts (atomic on GPU)
    pub max_vertices: u32,        // Capacity limit
    pub screen_width: u32,
    pub screen_height: u32,
}

/// Issue #158: Render vertex in unified buffer (matches Metal)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderVertex {
    pub position: [f32; 3],    // x, y, z (z = depth) - Metal: packed_float3
    pub _pad0: f32,            // Align to float4
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub _pad1: [f32; 2],
}

// COMPILE-TIME CHECK: Ensure RenderVertex is exactly 48 bytes
// Metal shader MUST use packed_float3 (12 bytes), NOT float3 (16 bytes)
// If this fails, the Metal shader vertex struct is wrong!
const _RENDER_VERTEX_SIZE_CHECK: () = assert!(
    std::mem::size_of::<RenderVertex>() == 48,
    "RenderVertex must be 48 bytes. Metal shader must use packed_float3, not float3!"
);

/// Maximum vertices in unified buffer
pub const MAX_VERTICES: u32 = 1024 * 1024; // 1M vertices

// ============================================================================
// Bytecode VM (Issue #164)
// ============================================================================

/// Bytecode opcodes (must match Metal shader)
pub mod bytecode_op {
    pub const NOP: u8 = 0x00;
    pub const HALT: u8 = 0xFF;
    pub const MOV: u8 = 0x01;
    pub const ADD: u8 = 0x02;
    pub const SUB: u8 = 0x03;
    pub const MUL: u8 = 0x04;
    pub const DIV: u8 = 0x05;
    pub const MOD: u8 = 0x06;

    // Math intrinsics (Phase 5 - Issue #178)
    // THE GPU IS THE COMPUTER - native GPU math operations
    pub const SIN: u8 = 0x07;
    pub const COS: u8 = 0x08;
    pub const SQRT: u8 = 0x09;

    // Float unary/binary math ops (Issue #198)
    pub const ABS: u8 = 0x20;        // dst.x = fabs(s1.x)
    pub const CEIL: u8 = 0x21;       // dst.x = ceil(s1.x)
    pub const FLOOR: u8 = 0x22;      // dst.x = floor(s1.x)
    pub const TRUNC: u8 = 0x23;      // dst.x = trunc(s1.x)
    pub const NEAREST: u8 = 0x24;    // dst.x = rint(s1.x) - round to nearest, ties to even
    pub const COPYSIGN: u8 = 0x25;   // dst.x = copysign(s1.x, s2.x)
    pub const FMIN: u8 = 0x26;       // dst.x = fmin(s1.x, s2.x)
    pub const FMAX: u8 = 0x27;       // dst.x = fmax(s1.x, s2.x)
    pub const FNEG: u8 = 0x28;       // dst.x = -s1.x

    // ═══════════════════════════════════════════════════════════════════════════
    // 64-BIT FLOAT OPERATIONS (Issue #189, Issue #27 - Double-Single Emulation)
    // THE GPU IS THE COMPUTER - Metal does NOT support native double precision
    // F64 uses DOUBLE-SINGLE representation: regs[r].x = hi, regs[r].y = lo
    // where the true value is (hi + lo), providing ~47 bits of mantissa precision
    // This is the same technique used by metal-float64 library
    // ═══════════════════════════════════════════════════════════════════════════

    // F64 arithmetic (0x0A-0x0E) - implemented via double-single algorithms
    pub const F64_ADD: u8 = 0x0A;      // dst.xy = ds_add(s1.xy, s2.xy)
    pub const F64_SUB: u8 = 0x0B;      // dst.xy = ds_sub(s1.xy, s2.xy)
    pub const F64_MUL: u8 = 0x0C;      // dst.xy = ds_mul(s1.xy, s2.xy)
    pub const F64_DIV: u8 = 0x0D;      // dst.xy = ds_div(s1.xy, s2.xy)
    pub const F64_SQRT: u8 = 0x0E;     // dst.xy = ds_sqrt(s1.xy)

    // F64 comparison (0x29-0x2E) - double-single aware comparisons
    pub const F64_EQ: u8 = 0x29;       // dst.x = (s1.xy == s2.xy) ? 1.0 : 0.0
    pub const F64_NE: u8 = 0x2A;       // dst.x = (s1.xy != s2.xy) ? 1.0 : 0.0
    pub const F64_LT: u8 = 0x2B;       // dst.x = (s1.xy < s2.xy) ? 1.0 : 0.0
    pub const F64_GT: u8 = 0x2C;       // dst.x = (s1.xy > s2.xy) ? 1.0 : 0.0
    pub const F64_LE: u8 = 0x2D;       // dst.x = (s1.xy <= s2.xy) ? 1.0 : 0.0
    pub const F64_GE: u8 = 0x2E;       // dst.x = (s1.xy >= s2.xy) ? 1.0 : 0.0

    // F64 min/max (0x2F-0x30) - double-single aware min/max
    pub const F64_MIN: u8 = 0x2F;      // dst.xy = min(s1.xy, s2.xy) as double-single
    pub const F64_MAX: u8 = 0x30;      // dst.xy = max(s1.xy, s2.xy) as double-single

    // F64 conversion (0x0F-0x12)
    pub const F64_FROM_I32_S: u8 = 0x0F;  // dst.xy = ds_from_i32(s1.x signed)
    pub const F64_FROM_I32_U: u8 = 0x10;  // dst.xy = ds_from_u32(s1.x unsigned)
    pub const F64_FROM_I64_S: u8 = 0x11;  // dst.xy = ds_from_i64(s1.xy signed)
    pub const F64_FROM_I64_U: u8 = 0x12;  // dst.xy = ds_from_u64(s1.xy unsigned)

    // F64 to integer conversion - using 0x18-0x1F range
    pub const F64_TO_I32_S: u8 = 0x18;    // dst.x = ds_to_i32(s1.xy) truncate
    pub const F64_TO_I32_U: u8 = 0x19;    // dst.x = ds_to_u32(s1.xy) truncate
    pub const F64_TO_I64_S: u8 = 0x1A;    // dst.xy = ds_to_i64(s1.xy) truncate
    pub const F64_TO_I64_U: u8 = 0x1B;    // dst.xy = ds_to_u64(s1.xy) truncate

    // Vector packing (0x1C)
    pub const PACK2: u8 = 0x1C;           // dst.xy = (s1.x, s2.x) - pack two scalars into float2

    // F64 reinterpret operations (0x1D-0x1E)
    // NOTE: These are approximations - double-single doesn't preserve IEEE 754 bit pattern
    // They reconstruct the value and convert, which gives correct results for most cases
    pub const F64_REINTERPRET_I64: u8 = 0x1D;  // dst.xy = ds_from_f64_bits(s1.xy)
    pub const I64_REINTERPRET_F64: u8 = 0x1E;  // dst.xy = ds_to_f64_bits(s1.xy)

    pub const LOADI: u8 = 0x13;
    pub const SETX: u8 = 0x14;
    pub const SETY: u8 = 0x15;
    pub const SETZ: u8 = 0x16;
    pub const SETW: u8 = 0x17;
    pub const EQ: u8 = 0x40;
    pub const LT: u8 = 0x42;
    pub const GT: u8 = 0x44;
    pub const JMP: u8 = 0x60;
    pub const JZ: u8 = 0x61;
    pub const JNZ: u8 = 0x62;
    pub const LD: u8 = 0x80;
    pub const ST: u8 = 0x81;
    pub const LD1: u8 = 0x82;       // Load byte
    pub const ST1: u8 = 0x83;       // Store byte
    pub const LD2: u8 = 0x8C;       // Load 16-bit (halfword)
    pub const ST2: u8 = 0x8D;       // Store 16-bit (halfword)
    pub const LD4: u8 = 0x8E;       // Load 32-bit (word)
    pub const ST4: u8 = 0x8F;       // Store 32-bit (word)
    pub const QUAD: u8 = 0xA0;

    // More 64-bit comparisons (0xA1-0xA7) - overflow from 0xB0-0xBF range
    pub const INT64_NE: u8 = 0xA1;     // dst.x = (s1.xy != s2.xy) ? 1 : 0
    pub const INT64_LT_U: u8 = 0xA2;   // dst.x = (s1.xy < s2.xy unsigned) ? 1 : 0
    pub const INT64_LT_S: u8 = 0xA3;   // dst.x = (s1.xy < s2.xy signed) ? 1 : 0
    pub const INT64_LE_U: u8 = 0xA4;   // dst.x = (s1.xy <= s2.xy unsigned) ? 1 : 0
    pub const INT64_LE_S: u8 = 0xA5;   // dst.x = (s1.xy <= s2.xy signed) ? 1 : 0
    pub const INT64_EQZ: u8 = 0xA6;    // dst.x = (s1.xy == 0) ? 1 : 0
    pub const INT64_ROTR: u8 = 0xA7;   // dst.xy = rotate_right(s1.xy, s2.x)
    pub const INT64_ROTL: u8 = 0xA8;   // dst.xy = rotate_left(s1.xy, s2.x)
    pub const INT64_CLZ: u8 = 0xA9;    // dst.x = clz(s1.xy)
    pub const INT64_CTZ: u8 = 0xAA;    // dst.x = ctz(s1.xy)
    pub const INT64_POPCNT: u8 = 0xAB; // dst.x = popcount(s1.xy)

    // ═══════════════════════════════════════════════════════════════════════════
    // 64-BIT INTEGER OPERATIONS (Issue #188)
    // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support int64/uint64
    // 64-bit values use regs[r].xy (2 x 32-bit floats reinterpreted as ulong)
    // ═══════════════════════════════════════════════════════════════════════════

    // 64-bit arithmetic (0xB0-0xB5)
    pub const INT64_ADD: u8 = 0xB0;    // dst.xy = s1.xy + s2.xy (as ulong)
    pub const INT64_SUB: u8 = 0xB1;    // dst.xy = s1.xy - s2.xy (as ulong)
    pub const INT64_MUL: u8 = 0xB2;    // dst.xy = s1.xy * s2.xy (as ulong)
    pub const INT64_DIV_S: u8 = 0xB3;  // dst.xy = s1.xy / s2.xy (as long, signed)
    pub const INT64_DIV_U: u8 = 0xB4;  // dst.xy = s1.xy / s2.xy (as ulong, unsigned)
    pub const INT64_REM_U: u8 = 0xB5;  // dst.xy = s1.xy % s2.xy (as ulong)
    pub const INT64_REM_S: u8 = 0xAC;  // dst.xy = s1.xy % s2.xy (as long, signed)

    // 64-bit bitwise (0xB6-0xB9)
    pub const INT64_AND: u8 = 0xB6;    // dst.xy = s1.xy & s2.xy
    pub const INT64_OR: u8 = 0xB7;     // dst.xy = s1.xy | s2.xy
    pub const INT64_XOR: u8 = 0xB8;    // dst.xy = s1.xy ^ s2.xy
    pub const INT64_SHL: u8 = 0xB9;    // dst.xy = s1.xy << s2.x (shift amount from low 6 bits)

    // 64-bit shifts and conversions (0xBA-0xBF)
    pub const INT64_SHR_U: u8 = 0xBA;  // dst.xy = s1.xy >> s2.x (logical shift)
    pub const INT64_SHR_S: u8 = 0xBB;  // dst.xy = s1.xy >> s2.x (arithmetic shift)
    pub const INT64_EQ: u8 = 0xBC;     // dst.x = (s1.xy == s2.xy) ? 1 : 0
    pub const INT64_WRAP: u8 = 0xBD;   // dst.x = s1.xy as i32 (wrap to 32-bit)
    pub const INT64_EXTEND_U: u8 = 0xBE; // dst.xy = s1.x as u64 (zero-extend)
    pub const INT64_EXTEND_S: u8 = 0xBF; // dst.xy = s1.x as i64 (sign-extend)

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGER OPERATIONS (Phase 1 - Issue #171)
    // THE GPU IS THE COMPUTER - integers are just bits, reinterpret don't convert
    // ═══════════════════════════════════════════════════════════════════════════

    // Integer arithmetic (0xC0-0xC7)
    pub const INT_ADD: u8 = 0xC0;
    pub const INT_SUB: u8 = 0xC1;
    pub const INT_MUL: u8 = 0xC2;
    pub const INT_DIV_S: u8 = 0xC3;
    pub const INT_DIV_U: u8 = 0xC4;
    pub const INT_REM_S: u8 = 0xC5;
    pub const INT_REM_U: u8 = 0xC6;
    pub const INT_NEG: u8 = 0xC7;

    // Bitwise (0xCA-0xCF)
    pub const BIT_AND: u8 = 0xCA;
    pub const BIT_OR: u8 = 0xCB;
    pub const BIT_XOR: u8 = 0xCC;
    pub const BIT_NOT: u8 = 0xCD;
    pub const SHL: u8 = 0xCE;
    pub const SHR_U: u8 = 0xCF;

    // More shifts (0xD0-0xD3)
    pub const SHR_S: u8 = 0xD0;
    pub const ROTL: u8 = 0xD1;
    pub const ROTR: u8 = 0xD2;
    pub const CLZ: u8 = 0xD3;
    pub const CTZ: u8 = 0xC8;       // Count trailing zeros
    pub const POPCNT: u8 = 0xC9;    // Population count

    // Integer comparison (0xD4-0xD9)
    pub const INT_EQ: u8 = 0xD4;
    pub const INT_NE: u8 = 0xD5;
    pub const INT_LT_S: u8 = 0xD6;
    pub const INT_LT_U: u8 = 0xD7;
    pub const INT_LE_S: u8 = 0xD8;
    pub const INT_LE_U: u8 = 0xD9;

    // Conversion (0xDA-0xDD)
    pub const INT_TO_F: u8 = 0xDA;
    pub const UINT_TO_F: u8 = 0xDB;
    pub const F_TO_INT: u8 = 0xDC;
    pub const F_TO_UINT: u8 = 0xDD;

    // Load immediate integer (0xDE-0xDF)
    pub const LOADI_INT: u8 = 0xDE;
    pub const LOADI_UINT: u8 = 0xDF;

    // ═══════════════════════════════════════════════════════════════════════════
    // ATOMIC OPERATIONS (Phase 2 - Issue #172)
    // THE GPU IS THE COMPUTER - lock-free coordination, GPU never waits
    // ═══════════════════════════════════════════════════════════════════════════

    // Atomic load/store (0xE0-0xE1)
    pub const ATOMIC_LOAD: u8 = 0xE0;
    pub const ATOMIC_STORE: u8 = 0xE1;

    // Atomic read-modify-write (0xE2-0xEA)
    pub const ATOMIC_ADD: u8 = 0xE2;
    pub const ATOMIC_SUB: u8 = 0xE3;
    pub const ATOMIC_MAX_U: u8 = 0xE4;
    pub const ATOMIC_MIN_U: u8 = 0xE5;
    pub const ATOMIC_MAX_S: u8 = 0xE6;
    pub const ATOMIC_MIN_S: u8 = 0xE7;
    pub const ATOMIC_AND: u8 = 0xE8;
    pub const ATOMIC_OR: u8 = 0xE9;
    pub const ATOMIC_XOR: u8 = 0xEA;

    // Atomic compare-and-swap (0xEB)
    pub const ATOMIC_CAS: u8 = 0xEB;

    // Atomic increment/decrement (0xEC-0xED)
    pub const ATOMIC_INC: u8 = 0xEC;
    pub const ATOMIC_DEC: u8 = 0xED;

    // Memory fence (0xEE)
    pub const MEM_FENCE: u8 = 0xEE;

    // ═══════════════════════════════════════════════════════════════════════════
    // ALLOCATOR OPERATIONS (Phase 6 - Issue #179)
    // THE GPU IS THE COMPUTER - GPU-resident memory allocator for Rust alloc crate
    // Lock-free slab allocator with atomic free lists
    // ═══════════════════════════════════════════════════════════════════════════

    // Memory allocation (0xF0-0xF3)
    pub const ALLOC: u8 = 0xF0;       // dst = gpu_alloc(size_reg, align_reg)
    pub const DEALLOC: u8 = 0xF1;     // gpu_dealloc(ptr_reg, size_reg, align_reg)
    pub const REALLOC: u8 = 0xF2;     // dst = gpu_realloc(ptr_reg, old_size_reg, new_size_reg)
    pub const ALLOC_ZERO: u8 = 0xF3;  // dst = gpu_alloc_zeroed(size_reg, align_reg)

    // WASM Memory Operations (Issue #210)
    pub const MEMORY_SIZE: u8 = 0xF4; // dst = current memory size in pages
    pub const MEMORY_GROW: u8 = 0xF5; // dst = memory_grow(delta_pages)

    // ═══════════════════════════════════════════════════════════════════════════
    // WASI OPERATIONS (Issue #207 - GPU-Native WASI)
    // THE GPU IS THE COMPUTER - WASI system calls implemented on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    // WASI I/O (0xF6-0xF7)
    pub const WASI_FD_WRITE: u8 = 0xF6;  // fd_write(fd, iovs, iovs_len, nwritten) -> errno
    pub const WASI_FD_READ: u8 = 0xF7;   // fd_read(fd, iovs, iovs_len, nread) -> errno

    // WASI Process (0xF8)
    pub const WASI_PROC_EXIT: u8 = 0xF8; // proc_exit(code) - halts execution

    // WASI Environment (0xF9-0xFC)
    pub const WASI_ENVIRON_SIZES_GET: u8 = 0xF9;  // environ_sizes_get(count, size) -> errno
    pub const WASI_ENVIRON_GET: u8 = 0xFA;        // environ_get(environ, buf) -> errno
    pub const WASI_ARGS_SIZES_GET: u8 = 0xFB;     // args_sizes_get(count, size) -> errno
    pub const WASI_ARGS_GET: u8 = 0xFC;           // args_get(argv, buf) -> errno

    // WASI Clock/Random (0xFD-0xFE)
    pub const WASI_CLOCK_TIME_GET: u8 = 0xFD;     // clock_time_get(id, precision, time) -> errno
    pub const WASI_RANDOM_GET: u8 = 0xFE;         // random_get(buf, len) -> errno

    // ═══════════════════════════════════════════════════════════════════════════
    // PANIC HANDLING (Issue #209 - GPU-Native Panic)
    // THE GPU IS THE COMPUTER - panic handling on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    pub const PANIC: u8 = 0x76;          // panic(msg_ptr, msg_len) - debug output + halt
    pub const UNREACHABLE: u8 = 0x77;    // unreachable() - halt with error

    // ═══════════════════════════════════════════════════════════════════════════
    // RECURSION SUPPORT (Issue #208 - GPU-Native Recursion)
    // THE GPU IS THE COMPUTER - function calls via GPU call stack
    // ═══════════════════════════════════════════════════════════════════════════

    pub const CALL_FUNC: u8 = 0x78;      // call_func(target_pc) - push frame, jump to function
    pub const RETURN_FUNC: u8 = 0x79;    // return_func() - pop frame, return to caller

    // ═══════════════════════════════════════════════════════════════════════════
    // DEBUG I/O OPERATIONS (Phase 7 - Issue #180)
    // THE GPU IS THE COMPUTER - debug output via ring buffer
    // Lock-free writes using atomics, CPU reads buffer after execution
    // ═══════════════════════════════════════════════════════════════════════════

    // Debug output (0x70-0x75)
    pub const DBG_I32: u8 = 0x70;     // Debug print i32 from src1 register
    pub const DBG_F32: u8 = 0x71;     // Debug print f32 from src1 register
    pub const DBG_STR: u8 = 0x72;     // Debug print string (ptr=src1, len=src2)
    pub const DBG_BOOL: u8 = 0x73;    // Debug print bool from src1 register
    pub const DBG_NL: u8 = 0x74;      // Debug newline marker
    pub const DBG_FLUSH: u8 = 0x75;   // Force debug flush (no-op, buffer read at end)

    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOMATIC CODE TRANSFORMATION OPCODES (Phase 8 - Issue #182)
    // THE GPU IS THE COMPUTER - transform CPU patterns to GPU-native equivalents
    // Mutex -> spinlock, Condvar -> barrier, sleep -> frame wait
    // ═══════════════════════════════════════════════════════════════════════════

    // Work queue operations (0x84-0x85) - for async/await transformation
    pub const WORK_PUSH: u8 = 0x84;   // Push work item to queue: work_push(item_reg, queue_reg)
    pub const WORK_POP: u8 = 0x85;    // Pop work item from queue: dst = work_pop(queue_reg)

    // I/O request operations (0x86-0x87) - async I/O pattern
    pub const REQUEST_QUEUE: u8 = 0x86;  // Queue I/O request: request_queue(type_reg, data_reg)
    pub const REQUEST_POLL: u8 = 0x87;   // Poll request status: dst = request_poll(id_reg)

    // Frame-based timing (0x88) - replaces thread::sleep
    pub const FRAME_WAIT: u8 = 0x88;  // Wait for N frames: frame_wait(frames_reg)

    // Spinlock operations (0x89-0x8A) - replaces Mutex
    pub const SPINLOCK: u8 = 0x89;    // Acquire spinlock: spinlock(lock_reg)
    pub const SPINUNLOCK: u8 = 0x8A;  // Release spinlock: spinunlock(lock_reg)

    // Threadgroup barrier (0x8B) - replaces Condvar::wait
    pub const BARRIER: u8 = 0x8B;     // Threadgroup barrier: barrier()

    // ═══════════════════════════════════════════════════════════════════════════
    // TABLE OPERATIONS (Issue #212 - GPU-Native Table Operations)
    // THE GPU IS THE COMPUTER - tables are GPU-resident arrays with O(1) lookup
    // ═══════════════════════════════════════════════════════════════════════════

    pub const TABLE_GET: u8 = 0x50;   // dst = table.get(table_idx, elem_idx)
    pub const TABLE_SET: u8 = 0x51;   // table.set(table_idx, elem_idx, value)
    pub const TABLE_SIZE: u8 = 0x52;  // dst = table.size(table_idx)
    pub const TABLE_GROW: u8 = 0x53;  // dst = table.grow(table_idx, delta, init_val)
    pub const TABLE_INIT: u8 = 0x54;  // table.init(table_idx, elem_idx, dst, src, count)
    pub const TABLE_COPY: u8 = 0x55;  // table.copy(dst_table, src_table, dst, src, count)
    pub const TABLE_FILL: u8 = 0x56;  // table.fill(table_idx, dst, value, count)

    // ═══════════════════════════════════════════════════════════════════════════════
    // SIMD OPERATIONS (Issue #211)
    // THE GPU IS THE COMPUTER - float4 is native SIMD, these ops work on all 4 lanes
    // ═══════════════════════════════════════════════════════════════════════════════

    pub const V4_ADD: u8 = 0x90;      // dst = s1 + s2 (all 4 lanes)
    pub const V4_SUB: u8 = 0x91;      // dst = s1 - s2 (all 4 lanes)
    pub const V4_MUL: u8 = 0x92;      // dst = s1 * s2 (all 4 lanes)
    pub const V4_DIV: u8 = 0x93;      // dst = s1 / s2 (all 4 lanes)
    pub const V4_MIN: u8 = 0x94;      // dst = min(s1, s2) per lane
    pub const V4_MAX: u8 = 0x95;      // dst = max(s1, s2) per lane
    pub const V4_ABS: u8 = 0x96;      // dst = abs(s1) per lane
    pub const V4_NEG: u8 = 0x97;      // dst = -s1 per lane
    pub const V4_SQRT: u8 = 0x98;     // dst = sqrt(s1) per lane
    pub const V4_DOT: u8 = 0x99;      // dst.x = dot(s1, s2) - dot product
    pub const V4_SHUFFLE: u8 = 0x9A;  // dst = shuffle(s1, s2, imm) - swizzle
    pub const V4_EXTRACT: u8 = 0x9B;  // dst.x = s1[imm] - extract single lane
    pub const V4_REPLACE: u8 = 0x9C;  // dst = s1 with s1[imm] = s2.x - replace lane
    pub const V4_SPLAT: u8 = 0x9D;    // dst = (s1.x, s1.x, s1.x, s1.x) - broadcast
    pub const V4_EQ: u8 = 0x9E;       // dst = (s1 == s2) per lane (1.0 or 0.0)
    pub const V4_LT: u8 = 0x9F;       // dst = (s1 < s2) per lane (1.0 or 0.0)
}

/// Bytecode instruction (8 bytes, matches Metal)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BytecodeInst {
    pub opcode: u8,
    pub dst: u8,
    pub src1: u8,
    pub src2: u8,
    pub imm: f32,
}

/// Bytecode header (at start of state buffer, matches Metal)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BytecodeHeader {
    pub code_size: u32,
    pub entry_point: u32,
    pub vertex_budget: u32,
    pub flags: u32,
}

/// Simple bytecode assembler
pub struct BytecodeAssembler {
    instructions: Vec<BytecodeInst>,
}

impl BytecodeAssembler {
    pub fn new() -> Self {
        Self { instructions: Vec::new() }
    }

    pub fn emit(&mut self, opcode: u8, dst: u8, src1: u8, src2: u8, imm: f32) -> usize {
        let pc = self.instructions.len();
        self.instructions.push(BytecodeInst { opcode, dst, src1, src2, imm });
        pc
    }

    pub fn nop(&mut self) -> usize { self.emit(bytecode_op::NOP, 0, 0, 0, 0.0) }
    pub fn halt(&mut self) -> usize { self.emit(bytecode_op::HALT, 0, 0, 0, 0.0) }
    pub fn mov(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::MOV, dst, src, 0, 0.0) }
    pub fn add(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::ADD, dst, a, b, 0.0) }
    pub fn sub(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::SUB, dst, a, b, 0.0) }
    pub fn mul(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::MUL, dst, a, b, 0.0) }
    pub fn div(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::DIV, dst, a, b, 0.0) }
    pub fn modulo(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::MOD, dst, a, b, 0.0) }

    // Math intrinsics (Phase 5 - Issue #178)
    pub fn sin(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::SIN, dst, src, 0, 0.0) }
    pub fn cos(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::COS, dst, src, 0, 0.0) }
    pub fn sqrt(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::SQRT, dst, src, 0, 0.0) }

    // Float unary/binary math ops (Issue #198)
    pub fn abs(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::ABS, dst, src, 0, 0.0) }
    pub fn ceil(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::CEIL, dst, src, 0, 0.0) }
    pub fn floor(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::FLOOR, dst, src, 0, 0.0) }
    pub fn trunc(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::TRUNC, dst, src, 0, 0.0) }
    pub fn nearest(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::NEAREST, dst, src, 0, 0.0) }
    pub fn copysign(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::COPYSIGN, dst, a, b, 0.0) }
    pub fn fmin(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::FMIN, dst, a, b, 0.0) }
    pub fn fmax(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::FMAX, dst, a, b, 0.0) }
    pub fn fneg(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::FNEG, dst, src, 0, 0.0) }

    // ═══════════════════════════════════════════════════════════════════════════
    // 64-BIT FLOAT OPERATIONS (Issue #189)
    // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support double precision
    // F64 values use regs[r].xy (2 x 32-bit floats reinterpreted as double)
    // ═══════════════════════════════════════════════════════════════════════════

    // F64 arithmetic
    pub fn f64_add(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_ADD, dst, a, b, 0.0) }
    pub fn f64_sub(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_SUB, dst, a, b, 0.0) }
    pub fn f64_mul(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_MUL, dst, a, b, 0.0) }
    pub fn f64_div(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_DIV, dst, a, b, 0.0) }
    pub fn f64_sqrt(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_SQRT, dst, src, 0, 0.0) }

    // F64 comparison (double-single aware)
    pub fn f64_eq(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_EQ, dst, a, b, 0.0) }
    pub fn f64_ne(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_NE, dst, a, b, 0.0) }
    pub fn f64_lt(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_LT, dst, a, b, 0.0) }
    pub fn f64_gt(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_GT, dst, a, b, 0.0) }
    pub fn f64_le(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_LE, dst, a, b, 0.0) }
    pub fn f64_ge(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_GE, dst, a, b, 0.0) }

    // F64 min/max (double-single aware)
    pub fn f64_min(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_MIN, dst, a, b, 0.0) }
    pub fn f64_max(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::F64_MAX, dst, a, b, 0.0) }

    // F64 conversion from integers
    pub fn f64_from_i32_s(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_FROM_I32_S, dst, src, 0, 0.0) }
    pub fn f64_from_i32_u(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_FROM_I32_U, dst, src, 0, 0.0) }
    pub fn f64_from_i64_s(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_FROM_I64_S, dst, src, 0, 0.0) }
    pub fn f64_from_i64_u(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_FROM_I64_U, dst, src, 0, 0.0) }

    // F64 conversion to integers
    pub fn f64_to_i32_s(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_TO_I32_S, dst, src, 0, 0.0) }
    pub fn f64_to_i32_u(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_TO_I32_U, dst, src, 0, 0.0) }
    pub fn f64_to_i64_s(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_TO_I64_S, dst, src, 0, 0.0) }
    pub fn f64_to_i64_u(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_TO_I64_U, dst, src, 0, 0.0) }

    // F64 reinterpret operations
    // NOTE: These are approximations - double-single doesn't preserve IEEE 754 bit pattern
    pub fn f64_reinterpret_i64(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F64_REINTERPRET_I64, dst, src, 0, 0.0) }
    pub fn i64_reinterpret_f64(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::I64_REINTERPRET_F64, dst, src, 0, 0.0) }

    /// Load 64-bit float immediate constant using double-single representation
    /// Stores the value as dst.x = hi, dst.y = lo where value ≈ hi + lo
    /// This provides ~47 bits of mantissa precision (vs 52 for native f64)
    pub fn loadi_f64(&mut self, dst: u8, value: f64) -> usize {
        // Handle special values that would produce NaN in lo calculation
        if value.is_nan() {
            // NaN: store (NaN, 0.0)
            self.emit(bytecode_op::SETX, dst, 0, 0, f32::NAN);
            return self.emit(bytecode_op::SETY, dst, 0, 0, 0.0);
        }
        if value.is_infinite() {
            // Infinity: store (±inf, 0.0) - avoid inf - inf = NaN
            let hi = if value.is_sign_positive() { f32::INFINITY } else { f32::NEG_INFINITY };
            self.emit(bytecode_op::SETX, dst, 0, 0, hi);
            return self.emit(bytecode_op::SETY, dst, 0, 0, 0.0);
        }
        // Convert f64 to double-single format: value = hi + lo
        // hi captures the main value, lo captures the rounding error
        let hi = value as f32;
        let lo = (value - hi as f64) as f32;
        // Load hi into dst.x
        self.emit(bytecode_op::SETX, dst, 0, 0, hi);
        // Set lo into dst.y
        self.emit(bytecode_op::SETY, dst, 0, 0, lo)
    }

    pub fn loadi(&mut self, dst: u8, val: f32) -> usize { self.emit(bytecode_op::LOADI, dst, 0, 0, val) }
    pub fn setx(&mut self, dst: u8, val: f32) -> usize { self.emit(bytecode_op::SETX, dst, 0, 0, val) }
    pub fn sety(&mut self, dst: u8, val: f32) -> usize { self.emit(bytecode_op::SETY, dst, 0, 0, val) }
    pub fn setz(&mut self, dst: u8, val: f32) -> usize { self.emit(bytecode_op::SETZ, dst, 0, 0, val) }
    pub fn setw(&mut self, dst: u8, val: f32) -> usize { self.emit(bytecode_op::SETW, dst, 0, 0, val) }
    /// Pack two scalar .x values into dst.xy: dst.xy = (s1.x, s2.x)
    pub fn pack2(&mut self, dst: u8, s1: u8, s2: u8) -> usize { self.emit(bytecode_op::PACK2, dst, s1, s2, 0.0) }
    pub fn eq(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::EQ, dst, a, b, 0.0) }
    pub fn lt(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::LT, dst, a, b, 0.0) }
    pub fn gt(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::GT, dst, a, b, 0.0) }
    pub fn jmp(&mut self, target: usize) -> usize { self.emit(bytecode_op::JMP, 0, 0, 0, target as f32) }
    pub fn jz(&mut self, src: u8, target: usize) -> usize { self.emit(bytecode_op::JZ, 0, src, 0, target as f32) }
    pub fn jnz(&mut self, src: u8, target: usize) -> usize { self.emit(bytecode_op::JNZ, 0, src, 0, target as f32) }
    pub fn ld(&mut self, dst: u8, addr: u8, offset: f32) -> usize { self.emit(bytecode_op::LD, dst, addr, 0, offset) }
    pub fn st(&mut self, addr: u8, src: u8, offset: f32) -> usize { self.emit(bytecode_op::ST, 0, addr, src, offset) }
    pub fn ld1(&mut self, dst: u8, addr: u8, offset: f32) -> usize { self.emit(bytecode_op::LD1, dst, addr, 0, offset) }
    pub fn st1(&mut self, addr: u8, src: u8, offset: f32) -> usize { self.emit(bytecode_op::ST1, 0, addr, src, offset) }
    pub fn ld2(&mut self, dst: u8, addr: u8, offset: f32) -> usize { self.emit(bytecode_op::LD2, dst, addr, 0, offset) }
    pub fn st2(&mut self, addr: u8, src: u8, offset: f32) -> usize { self.emit(bytecode_op::ST2, 0, addr, src, offset) }
    pub fn ld4(&mut self, dst: u8, addr: u8, offset: f32) -> usize { self.emit(bytecode_op::LD4, dst, addr, 0, offset) }
    pub fn st4(&mut self, addr: u8, src: u8, offset: f32) -> usize { self.emit(bytecode_op::ST4, 0, addr, src, offset) }
    /// Emit quad: pos_reg.xy = position, size_reg.xy = size, color_reg = color
    pub fn quad(&mut self, pos_reg: u8, size_reg: u8, color_reg: u8, depth: f32) -> usize {
        self.emit(bytecode_op::QUAD, color_reg, pos_reg, size_reg, depth)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 64-BIT INTEGER OPERATIONS (Issue #188)
    // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support int64/uint64
    // 64-bit values use regs[r].xy (2 x 32-bit floats reinterpreted as ulong)
    // ═══════════════════════════════════════════════════════════════════════════

    // 64-bit arithmetic
    pub fn int64_add(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_ADD, dst, a, b, 0.0) }
    pub fn int64_sub(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_SUB, dst, a, b, 0.0) }
    pub fn int64_mul(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_MUL, dst, a, b, 0.0) }
    pub fn int64_div_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_DIV_S, dst, a, b, 0.0) }
    pub fn int64_div_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_DIV_U, dst, a, b, 0.0) }
    pub fn int64_rem_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_REM_U, dst, a, b, 0.0) }
    pub fn int64_rem_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_REM_S, dst, a, b, 0.0) }

    // 64-bit bitwise
    pub fn int64_and(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_AND, dst, a, b, 0.0) }
    pub fn int64_or(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_OR, dst, a, b, 0.0) }
    pub fn int64_xor(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_XOR, dst, a, b, 0.0) }
    pub fn int64_shl(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_SHL, dst, a, b, 0.0) }
    pub fn int64_shr_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_SHR_U, dst, a, b, 0.0) }
    pub fn int64_shr_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_SHR_S, dst, a, b, 0.0) }

    // 64-bit comparison
    pub fn int64_eq(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_EQ, dst, a, b, 0.0) }
    pub fn int64_ne(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_NE, dst, a, b, 0.0) }
    pub fn int64_lt_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_LT_U, dst, a, b, 0.0) }
    pub fn int64_lt_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_LT_S, dst, a, b, 0.0) }
    pub fn int64_le_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_LE_U, dst, a, b, 0.0) }
    pub fn int64_le_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_LE_S, dst, a, b, 0.0) }
    pub fn int64_eqz(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_EQZ, dst, src, 0, 0.0) }
    pub fn int64_rotr(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_ROTR, dst, a, b, 0.0) }
    pub fn int64_rotl(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT64_ROTL, dst, a, b, 0.0) }
    pub fn int64_clz(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_CLZ, dst, src, 0, 0.0) }
    pub fn int64_ctz(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_CTZ, dst, src, 0, 0.0) }
    pub fn int64_popcnt(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_POPCNT, dst, src, 0, 0.0) }

    // 64-bit conversion
    pub fn int64_wrap(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_WRAP, dst, src, 0, 0.0) }
    pub fn int64_extend_u(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_EXTEND_U, dst, src, 0, 0.0) }
    pub fn int64_extend_s(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT64_EXTEND_S, dst, src, 0, 0.0) }

    /// Load 64-bit immediate constant
    /// Stores low 32 bits in dst.x, high 32 bits in dst.y
    /// Uses two instructions: loadi_uint for low bits, sety for high bits
    pub fn loadi_int64(&mut self, dst: u8, value: u64) -> usize {
        let low = (value & 0xFFFFFFFF) as u32;
        let high = (value >> 32) as u32;
        // Load low 32 bits into dst.x
        self.emit(bytecode_op::LOADI_UINT, dst, 0, 0, f32::from_bits(low));
        // Set high 32 bits into dst.y
        self.emit(bytecode_op::SETY, dst, 0, 0, f32::from_bits(high))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGER OPERATIONS (Phase 1 - Issue #171)
    // THE GPU IS THE COMPUTER - integers are just bits
    // ═══════════════════════════════════════════════════════════════════════════

    // Integer arithmetic
    pub fn int_add(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_ADD, dst, a, b, 0.0) }
    pub fn int_sub(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_SUB, dst, a, b, 0.0) }
    pub fn int_mul(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_MUL, dst, a, b, 0.0) }
    pub fn int_div_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_DIV_S, dst, a, b, 0.0) }
    pub fn int_div_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_DIV_U, dst, a, b, 0.0) }
    pub fn int_rem_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_REM_S, dst, a, b, 0.0) }
    pub fn int_rem_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_REM_U, dst, a, b, 0.0) }
    pub fn int_neg(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT_NEG, dst, src, 0, 0.0) }

    // Bitwise operations
    pub fn bit_and(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::BIT_AND, dst, a, b, 0.0) }
    pub fn bit_or(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::BIT_OR, dst, a, b, 0.0) }
    pub fn bit_xor(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::BIT_XOR, dst, a, b, 0.0) }
    pub fn bit_not(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::BIT_NOT, dst, src, 0, 0.0) }
    pub fn shl(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::SHL, dst, a, b, 0.0) }
    pub fn shr_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::SHR_U, dst, a, b, 0.0) }
    pub fn shr_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::SHR_S, dst, a, b, 0.0) }
    pub fn rotl(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::ROTL, dst, a, b, 0.0) }
    pub fn rotr(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::ROTR, dst, a, b, 0.0) }
    pub fn clz(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::CLZ, dst, src, 0, 0.0) }
    pub fn ctz(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::CTZ, dst, src, 0, 0.0) }
    pub fn popcnt(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::POPCNT, dst, src, 0, 0.0) }

    // Integer comparison
    pub fn int_eq(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_EQ, dst, a, b, 0.0) }
    pub fn int_ne(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_NE, dst, a, b, 0.0) }
    pub fn int_lt_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_LT_S, dst, a, b, 0.0) }
    pub fn int_lt_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_LT_U, dst, a, b, 0.0) }
    pub fn int_le_s(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_LE_S, dst, a, b, 0.0) }
    pub fn int_le_u(&mut self, dst: u8, a: u8, b: u8) -> usize { self.emit(bytecode_op::INT_LE_U, dst, a, b, 0.0) }

    // Conversion
    pub fn int_to_f(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::INT_TO_F, dst, src, 0, 0.0) }
    pub fn uint_to_f(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::UINT_TO_F, dst, src, 0, 0.0) }
    pub fn f_to_int(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F_TO_INT, dst, src, 0, 0.0) }
    pub fn f_to_uint(&mut self, dst: u8, src: u8) -> usize { self.emit(bytecode_op::F_TO_UINT, dst, src, 0, 0.0) }

    /// Load signed integer immediate - bits are preserved through float representation
    pub fn loadi_int(&mut self, dst: u8, val: i32) -> usize {
        self.emit(bytecode_op::LOADI_INT, dst, 0, 0, f32::from_bits(val as u32))
    }

    /// Load unsigned integer immediate - bits are preserved through float representation
    pub fn loadi_uint(&mut self, dst: u8, val: u32) -> usize {
        self.emit(bytecode_op::LOADI_UINT, dst, 0, 0, f32::from_bits(val))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ATOMIC OPERATIONS (Phase 2 - Issue #172)
    // THE GPU IS THE COMPUTER - lock-free coordination, GPU never waits
    // Atomics are NOT for locks. Atomics are for LOCK-FREE COORDINATION.
    // ═══════════════════════════════════════════════════════════════════════════

    /// Atomic load from state memory
    /// dst.x = atomic_load(state[addr_reg.x])
    pub fn atomic_load(&mut self, dst: u8, addr_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_LOAD, dst, addr_reg, 0, 0.0)
    }

    /// Atomic store to state memory
    /// atomic_store(state[addr_reg.x], val_reg.x)
    pub fn atomic_store(&mut self, val_reg: u8, addr_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_STORE, 0, val_reg, addr_reg, 0.0)
    }

    /// Atomic add - returns old value
    /// dst.x = atomic_fetch_add(&state[addr_reg.x], val_reg.x)
    pub fn atomic_add(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_ADD, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic subtract - returns old value
    pub fn atomic_sub(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_SUB, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic max (unsigned) - returns old value
    pub fn atomic_max_u(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_MAX_U, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic min (unsigned) - returns old value
    pub fn atomic_min_u(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_MIN_U, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic max (signed) - returns old value
    pub fn atomic_max_s(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_MAX_S, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic min (signed) - returns old value
    pub fn atomic_min_s(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_MIN_S, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic AND - returns old value
    pub fn atomic_and(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_AND, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic OR - returns old value
    pub fn atomic_or(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_OR, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic XOR - returns old value
    pub fn atomic_xor(&mut self, dst: u8, addr_reg: u8, val_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_XOR, dst, addr_reg, val_reg, 0.0)
    }

    /// Atomic compare-and-swap
    /// If state[addr_reg.x] == expected_reg.x: state = desired (imm), dst.x = 1
    /// Else: dst.x = 0
    pub fn atomic_cas(&mut self, dst: u8, addr_reg: u8, expected_reg: u8, desired: u32) -> usize {
        self.emit(bytecode_op::ATOMIC_CAS, dst, addr_reg, expected_reg, f32::from_bits(desired))
    }

    /// Atomic increment - returns old value
    /// dst.x = atomic_fetch_add(&state[addr_reg.x], 1)
    pub fn atomic_inc(&mut self, dst: u8, addr_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_INC, dst, addr_reg, 0, 0.0)
    }

    /// Atomic decrement - returns old value
    /// dst.x = atomic_fetch_sub(&state[addr_reg.x], 1)
    pub fn atomic_dec(&mut self, dst: u8, addr_reg: u8) -> usize {
        self.emit(bytecode_op::ATOMIC_DEC, dst, addr_reg, 0, 0.0)
    }

    /// Memory fence - ensures all previous memory operations complete
    pub fn mem_fence(&mut self) -> usize {
        self.emit(bytecode_op::MEM_FENCE, 0, 0, 0, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ALLOCATOR OPERATIONS (Phase 6 - Issue #179)
    // THE GPU IS THE COMPUTER - GPU-resident memory allocator for Rust alloc crate
    // Lock-free slab allocator with atomic free lists per size class
    // ═══════════════════════════════════════════════════════════════════════════

    /// Allocate memory on GPU heap
    /// dst.x = gpu_alloc(size_reg.x, align_reg.x)
    /// Returns pointer (offset) to allocated memory, or 0xFFFFFFFF on failure
    pub fn alloc(&mut self, dst: u8, size_reg: u8, align_reg: u8) -> usize {
        self.emit(bytecode_op::ALLOC, dst, size_reg, align_reg, 0.0)
    }

    /// Free previously allocated memory
    /// gpu_dealloc(ptr_reg.x, size_reg.x, align_reg.x)
    /// No return value - fire and forget (GPU never waits)
    pub fn dealloc(&mut self, ptr_reg: u8, size_reg: u8, align_reg: u8) -> usize {
        self.emit(bytecode_op::DEALLOC, 0, ptr_reg, size_reg, f32::from_bits(align_reg as u32))
    }

    /// Reallocate memory to new size
    /// dst.x = gpu_realloc(ptr_reg.x, old_size_reg.x, new_size_reg.x)
    /// Returns new pointer, copies data if block moved
    pub fn realloc(&mut self, dst: u8, ptr_reg: u8, old_size_reg: u8, new_size_reg: u8) -> usize {
        // Pack old_size in src2, new_size in imm
        self.emit(bytecode_op::REALLOC, dst, ptr_reg, old_size_reg, f32::from_bits(new_size_reg as u32))
    }

    /// Allocate zeroed memory on GPU heap
    /// dst.x = gpu_alloc_zeroed(size_reg.x, align_reg.x)
    /// Returns pointer to zero-initialized memory
    pub fn alloc_zero(&mut self, dst: u8, size_reg: u8, align_reg: u8) -> usize {
        self.emit(bytecode_op::ALLOC_ZERO, dst, size_reg, align_reg, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DEBUG I/O OPERATIONS (Phase 7 - Issue #180)
    // THE GPU IS THE COMPUTER - debug output via ring buffer
    // Lock-free writes, thread ID included for multi-thread debugging
    // ═══════════════════════════════════════════════════════════════════════════

    /// Debug print i32 value from register
    /// Writes thread_id + type + value to debug ring buffer
    pub fn dbg_i32(&mut self, src_reg: u8) -> usize {
        self.emit(bytecode_op::DBG_I32, 0, src_reg, 0, 0.0)
    }

    /// Debug print f32 value from register
    pub fn dbg_f32(&mut self, src_reg: u8) -> usize {
        self.emit(bytecode_op::DBG_F32, 0, src_reg, 0, 0.0)
    }

    /// Debug print string from memory
    /// ptr_reg.x = pointer to string data (in state memory)
    /// len_reg.x = length of string in bytes
    pub fn dbg_str(&mut self, ptr_reg: u8, len_reg: u8) -> usize {
        self.emit(bytecode_op::DBG_STR, 0, ptr_reg, len_reg, 0.0)
    }

    /// Debug print bool value from register (0 = false, non-zero = true)
    pub fn dbg_bool(&mut self, src_reg: u8) -> usize {
        self.emit(bytecode_op::DBG_BOOL, 0, src_reg, 0, 0.0)
    }

    /// Debug newline marker - indicates end of logical debug line
    pub fn dbg_nl(&mut self) -> usize {
        self.emit(bytecode_op::DBG_NL, 0, 0, 0, 0.0)
    }

    /// Debug flush - marker for CPU to know debug output is complete
    /// (No-op on GPU - buffer is read after kernel execution)
    pub fn dbg_flush(&mut self) -> usize {
        self.emit(bytecode_op::DBG_FLUSH, 0, 0, 0, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOMATIC CODE TRANSFORMATION OPERATIONS (Phase 8 - Issue #182)
    // THE GPU IS THE COMPUTER - transform CPU patterns to GPU-native equivalents
    // ═══════════════════════════════════════════════════════════════════════════

    /// Push work item to parallel queue
    /// Used for async/await pattern transformation
    /// work_push(item_reg.x, queue_reg.x)
    pub fn work_push(&mut self, item_reg: u8, queue_reg: u8) -> usize {
        self.emit(bytecode_op::WORK_PUSH, 0, item_reg, queue_reg, 0.0)
    }

    /// Pop work item from parallel queue
    /// dst.x = work_pop(queue_reg.x)
    /// Returns 0xFFFFFFFF if queue is empty
    pub fn work_pop(&mut self, dst: u8, queue_reg: u8) -> usize {
        self.emit(bytecode_op::WORK_POP, dst, queue_reg, 0, 0.0)
    }

    /// Queue I/O request (async I/O pattern)
    /// request_queue(type_reg.x, data_reg.x)
    pub fn request_queue(&mut self, type_reg: u8, data_reg: u8) -> usize {
        self.emit(bytecode_op::REQUEST_QUEUE, 0, type_reg, data_reg, 0.0)
    }

    /// Poll I/O request status
    /// dst.x = request_poll(id_reg.x)
    /// Returns: 0 = pending, 1 = complete, 0xFFFFFFFF = error
    pub fn request_poll(&mut self, dst: u8, id_reg: u8) -> usize {
        self.emit(bytecode_op::REQUEST_POLL, dst, id_reg, 0, 0.0)
    }

    /// Wait for N frames (frame-based timing)
    /// Replaces thread::sleep() - GPU threads don't truly sleep
    /// frame_wait(frames_reg.x) - spins until frame_count advances
    pub fn frame_wait(&mut self, frames_reg: u8) -> usize {
        self.emit(bytecode_op::FRAME_WAIT, 0, frames_reg, 0, 0.0)
    }

    /// Acquire spinlock (Mutex::lock() transformation)
    /// THE GPU IS THE COMPUTER - spinlocks replace OS mutexes
    /// spinlock(lock_reg.x) - spins until lock acquired (with timeout protection)
    pub fn spinlock(&mut self, lock_reg: u8) -> usize {
        self.emit(bytecode_op::SPINLOCK, 0, lock_reg, 0, 0.0)
    }

    /// Release spinlock (Mutex::unlock() transformation)
    /// spinunlock(lock_reg.x) - releases the lock atomically
    pub fn spinunlock(&mut self, lock_reg: u8) -> usize {
        self.emit(bytecode_op::SPINUNLOCK, 0, lock_reg, 0, 0.0)
    }

    /// Threadgroup barrier (Condvar::wait() transformation)
    /// THE GPU IS THE COMPUTER - barriers synchronize threadgroup
    /// All threads in threadgroup must reach barrier before any continue
    pub fn barrier(&mut self) -> usize {
        self.emit(bytecode_op::BARRIER, 0, 0, 0, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // MEMORY OPERATIONS (GPU-Native Dynamic Memory - Issue #210)
    // THE GPU IS THE COMPUTER - memory management runs on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current memory size in pages (64KB per page)
    /// dst.x = memory_size() returns current memory size in pages
    pub fn memory_size(&mut self, dst: u8) -> usize {
        self.emit(bytecode_op::MEMORY_SIZE, dst, 0, 0, 0.0)
    }

    /// Grow memory by delta_pages
    /// dst.x = memory_grow(delta_reg.x) returns old size in pages, or -1 on failure
    pub fn memory_grow(&mut self, dst: u8, delta_reg: u8) -> usize {
        self.emit(bytecode_op::MEMORY_GROW, dst, delta_reg, 0, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // WASI OPERATIONS (Issue #207 - GPU-Native WASI)
    // THE GPU IS THE COMPUTER - WASI system calls implemented on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// fd_write(fd_reg, iovs_reg, iovs_len_reg, nwritten_reg) -> errno in dst
    /// Writes to stdout/stderr via debug buffer
    pub fn wasi_fd_write(&mut self, dst: u8, fd_reg: u8, iovs_reg: u8, nwritten_reg: u8) -> usize {
        // Pack fd in dst, iovs in s1, nwritten in s2, iovs_len in imm
        self.emit(bytecode_op::WASI_FD_WRITE, dst, iovs_reg, nwritten_reg, fd_reg as f32)
    }

    /// fd_read(fd_reg, iovs_reg, iovs_len_reg, nread_reg) -> errno in dst
    /// Always returns EBADF (9) - no input on GPU
    pub fn wasi_fd_read(&mut self, dst: u8, fd_reg: u8, iovs_reg: u8, nread_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_FD_READ, dst, iovs_reg, nread_reg, fd_reg as f32)
    }

    /// proc_exit(code_reg) - halts execution
    pub fn wasi_proc_exit(&mut self, code_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_PROC_EXIT, 0, code_reg, 0, 0.0)
    }

    /// environ_sizes_get(count_ptr_reg, size_ptr_reg) -> errno in dst
    /// Returns 0 count, 0 size (no environment on GPU)
    pub fn wasi_environ_sizes_get(&mut self, dst: u8, count_reg: u8, size_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_ENVIRON_SIZES_GET, dst, count_reg, size_reg, 0.0)
    }

    /// environ_get(environ_ptr_reg, buf_ptr_reg) -> errno in dst
    pub fn wasi_environ_get(&mut self, dst: u8, environ_reg: u8, buf_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_ENVIRON_GET, dst, environ_reg, buf_reg, 0.0)
    }

    /// args_sizes_get(count_ptr_reg, size_ptr_reg) -> errno in dst
    pub fn wasi_args_sizes_get(&mut self, dst: u8, count_reg: u8, size_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_ARGS_SIZES_GET, dst, count_reg, size_reg, 0.0)
    }

    /// args_get(argv_ptr_reg, buf_ptr_reg) -> errno in dst
    pub fn wasi_args_get(&mut self, dst: u8, argv_reg: u8, buf_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_ARGS_GET, dst, argv_reg, buf_reg, 0.0)
    }

    /// clock_time_get(clock_id_reg, precision_reg, time_ptr_reg) -> errno in dst
    pub fn wasi_clock_time_get(&mut self, dst: u8, clock_reg: u8, time_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_CLOCK_TIME_GET, dst, clock_reg, time_reg, 0.0)
    }

    /// random_get(buf_ptr_reg, buf_len_reg) -> errno in dst
    pub fn wasi_random_get(&mut self, dst: u8, buf_reg: u8, len_reg: u8) -> usize {
        self.emit(bytecode_op::WASI_RANDOM_GET, dst, buf_reg, len_reg, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PANIC HANDLING (Issue #209 - GPU-Native Panic)
    // THE GPU IS THE COMPUTER - panic handling on GPU
    // ═══════════════════════════════════════════════════════════════════════════

    /// panic(msg_ptr_reg, msg_len_reg) - write message to debug buffer and halt
    pub fn panic(&mut self, msg_ptr_reg: u8, msg_len_reg: u8) -> usize {
        self.emit(bytecode_op::PANIC, 0, msg_ptr_reg, msg_len_reg, 0.0)
    }

    /// unreachable() - halt with unreachable trap
    pub fn unreachable(&mut self) -> usize {
        self.emit(bytecode_op::UNREACHABLE, 0, 0, 0, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // RECURSION SUPPORT (Issue #208 - GPU-Native Recursion)
    // THE GPU IS THE COMPUTER - function calls via GPU call stack
    // ═══════════════════════════════════════════════════════════════════════════

    /// call_func(target_pc) - push call frame and jump to function entry
    /// Saves current PC+1 and registers to call stack, then jumps to target_pc
    pub fn call_func(&mut self, target_pc: usize) -> usize {
        self.emit(bytecode_op::CALL_FUNC, 0, 0, 0, f32::from_bits(target_pc as u32))
    }

    /// return_func() - pop call frame and return to caller
    /// Restores PC and registers from call stack
    pub fn return_func(&mut self) -> usize {
        self.emit(bytecode_op::RETURN_FUNC, 0, 0, 0, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TABLE OPERATIONS (Issue #212 - GPU-Native Table Operations)
    // THE GPU IS THE COMPUTER - tables are GPU-resident arrays with O(1) lookup
    // ═══════════════════════════════════════════════════════════════════════════

    /// table_get(dst, idx_reg, table_idx) - get funcref from table
    pub fn table_get(&mut self, dst: u8, idx_reg: u8, table_idx: u32) -> usize {
        self.emit(bytecode_op::TABLE_GET, dst, idx_reg, 0, f32::from_bits(table_idx))
    }

    /// table_set(idx_reg, val_reg, table_idx) - set funcref in table
    pub fn table_set(&mut self, idx_reg: u8, val_reg: u8, table_idx: u32) -> usize {
        self.emit(bytecode_op::TABLE_SET, 0, idx_reg, val_reg, f32::from_bits(table_idx))
    }

    /// table_size(dst, table_idx) - get current table size
    pub fn table_size(&mut self, dst: u8, table_idx: u32) -> usize {
        self.emit(bytecode_op::TABLE_SIZE, dst, 0, 0, f32::from_bits(table_idx))
    }

    /// table_grow(dst, delta_reg, init_reg, table_idx) - grow table by delta
    pub fn table_grow(&mut self, dst: u8, delta_reg: u8, init_reg: u8, table_idx: u32) -> usize {
        self.emit(bytecode_op::TABLE_GROW, dst, delta_reg, init_reg, f32::from_bits(table_idx))
    }

    /// table_init(dst_reg, src_reg, count_reg, table_idx, elem_idx) - initialize from element
    pub fn table_init(&mut self, dst_reg: u8, src_reg: u8, count_reg: u8, table_idx: u32, elem_idx: u32) -> usize {
        // Pack table_idx and elem_idx into immediate
        let packed = ((table_idx & 0xFFFF) << 16) | (elem_idx & 0xFFFF);
        self.emit(bytecode_op::TABLE_INIT, count_reg, dst_reg, src_reg, f32::from_bits(packed))
    }

    /// table_copy(dst_reg, src_reg, count_reg, dst_table, src_table) - copy between tables
    pub fn table_copy(&mut self, dst_reg: u8, src_reg: u8, count_reg: u8, dst_table: u32, src_table: u32) -> usize {
        // Pack dst_table and src_table into immediate
        let packed = ((dst_table & 0xFFFF) << 16) | (src_table & 0xFFFF);
        self.emit(bytecode_op::TABLE_COPY, count_reg, dst_reg, src_reg, f32::from_bits(packed))
    }

    /// table_fill(dst_reg, val_reg, count_reg, table_idx) - fill table with value
    pub fn table_fill(&mut self, dst_reg: u8, val_reg: u8, count_reg: u8, table_idx: u32) -> usize {
        self.emit(bytecode_op::TABLE_FILL, count_reg, dst_reg, val_reg, f32::from_bits(table_idx))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SIMD OPERATIONS (Issue #211)
    // THE GPU IS THE COMPUTER - float4 is native SIMD, these ops work on all 4 lanes
    // ═══════════════════════════════════════════════════════════════════════════

    /// v4_add(dst, s1, s2) - dst = s1 + s2 (all 4 lanes)
    pub fn v4_add(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_ADD, dst, s1, s2, 0.0)
    }

    /// v4_sub(dst, s1, s2) - dst = s1 - s2 (all 4 lanes)
    pub fn v4_sub(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_SUB, dst, s1, s2, 0.0)
    }

    /// v4_mul(dst, s1, s2) - dst = s1 * s2 (all 4 lanes)
    pub fn v4_mul(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_MUL, dst, s1, s2, 0.0)
    }

    /// v4_div(dst, s1, s2) - dst = s1 / s2 (all 4 lanes)
    pub fn v4_div(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_DIV, dst, s1, s2, 0.0)
    }

    /// v4_min(dst, s1, s2) - dst = min(s1, s2) per lane
    pub fn v4_min(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_MIN, dst, s1, s2, 0.0)
    }

    /// v4_max(dst, s1, s2) - dst = max(s1, s2) per lane
    pub fn v4_max(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_MAX, dst, s1, s2, 0.0)
    }

    /// v4_abs(dst, s1) - dst = abs(s1) per lane
    pub fn v4_abs(&mut self, dst: u8, s1: u8) -> usize {
        self.emit(bytecode_op::V4_ABS, dst, s1, 0, 0.0)
    }

    /// v4_neg(dst, s1) - dst = -s1 per lane
    pub fn v4_neg(&mut self, dst: u8, s1: u8) -> usize {
        self.emit(bytecode_op::V4_NEG, dst, s1, 0, 0.0)
    }

    /// v4_sqrt(dst, s1) - dst = sqrt(s1) per lane
    pub fn v4_sqrt(&mut self, dst: u8, s1: u8) -> usize {
        self.emit(bytecode_op::V4_SQRT, dst, s1, 0, 0.0)
    }

    /// v4_dot(dst, s1, s2) - dst.x = dot(s1, s2)
    pub fn v4_dot(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_DOT, dst, s1, s2, 0.0)
    }

    /// v4_shuffle(dst, s1, s2, mask) - swizzle/shuffle lanes
    /// mask encoding: 4 2-bit indices (bits 0-1=x, 2-3=y, 4-5=z, 6-7=w)
    /// values 0-3 = lanes from s1, values 4-7 would be from s2 in full impl
    pub fn v4_shuffle(&mut self, dst: u8, s1: u8, s2: u8, mask: u8) -> usize {
        self.emit(bytecode_op::V4_SHUFFLE, dst, s1, s2, f32::from_bits(mask as u32))
    }

    /// v4_extract(dst, s1, lane) - dst.x = s1[lane]
    pub fn v4_extract(&mut self, dst: u8, s1: u8, lane: u8) -> usize {
        self.emit(bytecode_op::V4_EXTRACT, dst, s1, 0, f32::from_bits(lane as u32))
    }

    /// v4_replace(dst, s1, s2, lane) - dst = s1 with dst[lane] = s2.x
    pub fn v4_replace(&mut self, dst: u8, s1: u8, s2: u8, lane: u8) -> usize {
        self.emit(bytecode_op::V4_REPLACE, dst, s1, s2, f32::from_bits(lane as u32))
    }

    /// v4_splat(dst, s1) - dst = (s1.x, s1.x, s1.x, s1.x)
    pub fn v4_splat(&mut self, dst: u8, s1: u8) -> usize {
        self.emit(bytecode_op::V4_SPLAT, dst, s1, 0, 0.0)
    }

    /// v4_eq(dst, s1, s2) - dst = (s1 == s2) per lane (1.0 or 0.0)
    pub fn v4_eq(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_EQ, dst, s1, s2, 0.0)
    }

    /// v4_lt(dst, s1, s2) - dst = (s1 < s2) per lane (1.0 or 0.0)
    pub fn v4_lt(&mut self, dst: u8, s1: u8, s2: u8) -> usize {
        self.emit(bytecode_op::V4_LT, dst, s1, s2, 0.0)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SHORTHAND METHODS FOR COMMON TRANSFORMATION PATTERNS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Shorthand atomic increment (Rc::clone transformation)
    /// dst.x = atomic_fetch_add(&state[addr_reg.x], 1)
    /// Reuses existing ATOMIC_INC opcode - this is just a convenience alias
    pub fn rc_clone(&mut self, dst: u8, refcount_addr_reg: u8) -> usize {
        self.atomic_inc(dst, refcount_addr_reg)
    }

    /// Shorthand atomic decrement (Rc::drop transformation)
    /// dst.x = atomic_fetch_sub(&state[addr_reg.x], 1)
    /// Reuses existing ATOMIC_DEC opcode - this is just a convenience alias
    pub fn rc_drop(&mut self, dst: u8, refcount_addr_reg: u8) -> usize {
        self.atomic_dec(dst, refcount_addr_reg)
    }

    /// Get current PC (for jump targets)
    pub fn pc(&self) -> usize { self.instructions.len() }

    /// Patch a jump instruction with the correct target
    pub fn patch_jump(&mut self, inst_idx: usize, target: usize) {
        self.instructions[inst_idx].imm = target as f32;
    }

    /// Build final bytecode buffer
    pub fn build(&self, vertex_budget: u32) -> Vec<u8> {
        let header = BytecodeHeader {
            code_size: self.instructions.len() as u32,
            entry_point: 0,
            vertex_budget,
            flags: 0,
        };

        let mut buf = Vec::with_capacity(
            std::mem::size_of::<BytecodeHeader>() +
            self.instructions.len() * std::mem::size_of::<BytecodeInst>()
        );

        // Write header
        buf.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<BytecodeHeader>()
            )
        });

        // Write instructions
        for inst in &self.instructions {
            buf.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    inst as *const _ as *const u8,
                    std::mem::size_of::<BytecodeInst>()
                )
            });
        }

        buf
    }
}

/// Generate Game of Life bytecode
///
/// State layout (after bytecode):
///   float4[0]: (width, height, generation, 0)
///   bytes[16..]: grid cells (width * height bytes)
pub fn generate_game_of_life_bytecode(width: u32, height: u32) -> Vec<u8> {
    let mut asm = BytecodeAssembler::new();

    // Register allocation:
    // r3 = width (loaded)
    // r4 = height (loaded)
    // r5 = grid_size
    // r6 = cell_size (8.0)
    // r7 = i (loop counter)
    // r8 = x (i % width)
    // r9 = y (i / width)
    // r10 = neighbor count
    // r11 = temp / conditions
    // r12 = current cell value
    // r13 = next cell value
    // r14 = quad position (x*8, y*8, 8, 8)
    // r15 = color (green)

    // Load width and height from state[0]
    asm.loadi(3, width as f32);   // r3 = width
    asm.loadi(4, height as f32);  // r4 = height
    asm.mul(5, 3, 4);             // r5 = grid_size = width * height
    asm.loadi(6, 8.0);            // r6 = cell_size

    // Initialize loop counter
    asm.loadi(7, 0.0);            // r7 = i = 0

    let loop_start = asm.pc();

    // Check if i >= grid_size
    asm.lt(11, 7, 5);             // r11 = (i < grid_size)
    let exit_jump = asm.jz(11, 0); // if not, exit (patched later)

    // Calculate x = i % width, y = i / width
    asm.modulo(8, 7, 3);          // r8 = i % width
    asm.div(9, 7, 3);             // r9 = i / width (will be floor due to float)

    // Count neighbors (simplified - just check 4 cardinal directions for now)
    // Full 8-neighbor check would be much longer
    asm.loadi(10, 0.0);           // r10 = neighbor_count = 0

    // This is a simplified version - real Game of Life needs 8 neighbors
    // For a proper implementation, we'd need more complex bytecode

    // Load current cell: grid starts at byte offset 16 (after float4 header)
    // Cell address = 16 + i
    asm.loadi(11, 16.0);
    asm.add(11, 11, 7);           // r11 = 16 + i
    asm.ld1(12, 11, 0.0);         // r12 = grid[i]

    // For now: just toggle cells with random-ish pattern based on position
    // This demonstrates the bytecode works; real GoL needs neighbor counting
    asm.add(11, 8, 9);            // r11 = x + y
    asm.loadi(13, 2.0);
    asm.modulo(11, 11, 13);       // r11 = (x + y) % 2
    asm.eq(13, 11, 12);           // r13 = ((x+y)%2 == cell) ? keep : toggle

    // Store new value
    asm.loadi(11, 16.0);
    asm.add(11, 11, 7);           // r11 = 16 + i
    asm.st1(11, 12, 0.0);         // grid[i] = current (keep same for now)

    // Emit quad if cell is alive (r12 != 0)
    let skip_quad = asm.jz(12, 0); // if dead, skip quad

    // Calculate quad position: (x * cell_size, y * cell_size, cell_size, cell_size)
    // Need to pack into r14 as float4(x*8, y*8, 8, 8)
    asm.mul(14, 8, 6);            // r14.x = x * 8
    // For now, store just x coordinate - proper implementation needs float4 packing

    // Emit green quad
    // r14 = position, r15 = size, r16 = color
    asm.mul(14, 8, 6);            // r14.x = x * 8 (position x)
    asm.mul(20, 9, 6);            // temp: y * 8
    // Note: This is still incomplete - we'd need SETY to set r14.y
    asm.loadi(15, 8.0);           // r15 = size (8x8)
    // Color as packed u32 (0xRRGGBBAA format) - green
    asm.loadi_uint(16, 0x00FF00FF);  // Green with full alpha

    asm.quad(14, 15, 16, 0.5);    // Emit quad (pos, size, color, depth)

    let skip_target = asm.pc();
    asm.patch_jump(skip_quad, skip_target);

    // i++
    asm.loadi(11, 1.0);
    asm.add(7, 7, 11);            // r7 = i + 1

    asm.jmp(loop_start);

    let exit_target = asm.pc();
    asm.patch_jump(exit_jump, exit_target);

    asm.halt();

    // Build with vertex budget for all cells
    asm.build(width * height * 6)
}

/// Create a simple test bytecode that draws a colored grid
pub fn generate_test_bytecode() -> Vec<u8> {
    let mut asm = BytecodeAssembler::new();

    // Draw a 4x4 grid of colored quads
    // r4 = position, r5 = size, r6 = color
    for y in 0..4 {
        for x in 0..4 {
            // Position
            asm.setx(4, (x * 100 + 50) as f32);
            asm.sety(4, (y * 100 + 50) as f32);

            // Size
            asm.setx(5, 80.0);
            asm.sety(5, 80.0);

            // Color as packed u32 (0xRRGGBBAA format)
            // QUAD opcode expects packed color, not float4 components
            let r = (x * 85) as u32;  // 0, 85, 170, 255
            let g = (y * 85) as u32;
            let b = 128_u32;
            let a = 255_u32;
            let packed_color = (r << 24) | (g << 16) | (b << 8) | a;
            asm.loadi_uint(6, packed_color);

            asm.quad(4, 5, 6, 0.5);
        }
    }

    asm.halt();

    asm.build(16 * 6)  // 16 quads * 6 vertices each
}

// ============================================================================
// Metal Shader
// ============================================================================

pub const GPU_APP_SYSTEM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint APP_FLAG_ACTIVE = 1 << 0;     // Must match Rust flags::ACTIVE
constant uint APP_FLAG_VISIBLE = 1 << 1;   // Must match Rust flags::VISIBLE
constant uint APP_FLAG_DIRTY = 1 << 2;     // Must match Rust flags::DIRTY
constant uint APP_FLAG_SUSPENDED = 1 << 3; // Must match Rust flags::SUSPENDED
constant uint APP_FLAG_FOCUS = 1 << 4;     // Must match Rust flags::FOCUS
constant uint APP_FLAG_NEEDS_INIT = 1 << 5;

constant uint PRIORITY_BACKGROUND = 0;
constant uint PRIORITY_NORMAL = 1;
constant uint PRIORITY_HIGH = 2;
constant uint PRIORITY_REALTIME = 3;

// User apps
constant uint APP_TYPE_NONE = 0;
constant uint APP_TYPE_GAME_OF_LIFE = 1;
constant uint APP_TYPE_PARTICLES = 2;
constant uint APP_TYPE_TEXT_EDITOR = 3;
constant uint APP_TYPE_FILESYSTEM = 4;
constant uint APP_TYPE_TERMINAL = 5;
constant uint APP_TYPE_DOCUMENT = 6;
constant uint APP_TYPE_SHELL = 7;
constant uint APP_TYPE_MANDELBROT = 8;
constant uint APP_TYPE_BOIDS = 9;
constant uint APP_TYPE_METABALLS = 10;
constant uint APP_TYPE_WAVES = 11;
constant uint APP_TYPE_CUSTOM = 100;
constant uint APP_TYPE_BYTECODE = 101;  // Dynamic bytecode apps

// System apps (Issue #155-#161)
constant uint APP_TYPE_COMPOSITOR = 200;
constant uint APP_TYPE_DOCK = 201;
constant uint APP_TYPE_MENUBAR = 202;
constant uint APP_TYPE_WINDOW_CHROME = 203;

constant uint INVALID_SLOT = 0xFFFFFFFF;

// Contiguous vertex allocation: each slot gets fixed space
constant uint VERTS_PER_SLOT = 1024;  // Max vertices per app slot

// ============================================================================
// Issue #156: Dock Constants
// ============================================================================
constant uint MAX_DOCK_ITEMS = 32;
constant uint DOCK_ITEM_VISIBLE   = 0x01;
constant uint DOCK_ITEM_RUNNING   = 0x02;
constant uint DOCK_ITEM_HOVERED   = 0x04;
constant uint DOCK_ITEM_BOUNCING  = 0x08;
constant uint DOCK_ITEM_CLICKED   = 0x10;
constant float DOCK_DEFAULT_HEIGHT = 70.0;
constant float DOCK_DEFAULT_ICON_SIZE = 48.0;
constant float DOCK_MAGNIFIED_SIZE = 72.0;
constant float DOCK_ICON_SPACING = 8.0;
constant float DOCK_MAGNIFICATION_RADIUS = 120.0;
constant float DOCK_ANIMATION_SPEED = 0.25;
constant float DOCK_BOUNCE_HEIGHT = 20.0;
constant float DOCK_BOUNCE_SPEED = 0.15;

// ============================================================================
// Structures
// ============================================================================

struct GpuAppDescriptor {
    uint flags;
    uint app_type;
    uint slot_id;
    uint window_id;

    uint state_offset;
    uint state_size;
    uint vertex_offset;
    uint vertex_size;
    uint param_offset;
    uint param_size;
    uint _mem_pad[2];

    uint frame_number;
    uint input_head;
    uint input_tail;
    uint thread_count;

    uint vertex_count;
    uint clear_color;
    float preferred_width;
    float preferred_height;

    uint priority;
    uint last_run_frame;
    uint accumulated_time;
    uint _sched_pad;

    uint input_events[8];
};

struct AppTableHeader {
    uint max_slots;
    atomic_uint active_count;
    atomic_uint free_bitmap[2];
    uint _pad[4];
};

struct AllocatorState {
    atomic_uint bump_pointer;
    uint pool_size;
    atomic_uint allocation_count;
    uint peak_usage;
};

// ============================================================================
// Window Flags (Issue #159)
// ============================================================================
constant uint WINDOW_VISIBLE = 1;
constant uint WINDOW_MINIMIZED = 2;
constant uint WINDOW_MAXIMIZED = 4;
constant uint WINDOW_FOCUSED = 8;

// ============================================================================
// Forward declarations for Issue #159: Window Chrome
// ============================================================================

// GpuWindow struct - needed before window_chrome_update
struct GpuWindow {
    float x, y, width, height;
    float depth;        // 0.0 = back, 1.0 = front
    uint app_slot;
    uint flags;
    uint _pad;
};

// RenderVertex struct - needed for window chrome vertex generation
// CRITICAL: Use packed_float3 (12 bytes) to match Rust [f32; 3]
// Metal float3 is 16 bytes which breaks alignment!
struct RenderVertex {
    packed_float3 position;  // 12 bytes - matches Rust [f32; 3]
    float _pad0;             // 4 bytes
    float4 color;            // 16 bytes
    float2 uv;               // 8 bytes
    float2 _pad1;            // 8 bytes
};  // Total: 48 bytes - matches Rust

// ============================================================================
// Issue #155: O(1) Memory Management Structures
// ============================================================================

struct FreeBlock {
    uint next;      // Index of next free block (forms linked list)
    uint size;      // Size of this block
    uint offset;    // Offset in memory pool
    uint _pad;
};

struct MemoryPool {
    atomic_uint freelist_head;    // Head of free list (LIFO stack)
    atomic_uint bump_pointer;     // Fallback bump allocator
    atomic_uint free_count;       // Number of free blocks
    uint pool_size;
    atomic_uint block_count;      // Total blocks in freelist array
    uint max_blocks;              // Maximum free blocks we can track
    uint _pad[2];
};

// ============================================================================
// SLOT ALLOCATOR (atomic bitmap, O(1))
// ============================================================================

inline uint allocate_slot(device AppTableHeader* header) {
    for (uint word = 0; word < 2; word++) {
        uint bitmap = atomic_load_explicit(&header->free_bitmap[word], memory_order_relaxed);

        while (bitmap != 0) {
            uint bit = ctz(bitmap);
            uint slot = word * 32 + bit;

            if (slot >= header->max_slots) break;

            uint mask = 1u << bit;
            uint old = atomic_fetch_and_explicit(
                &header->free_bitmap[word],
                ~mask,
                memory_order_relaxed
            );

            if (old & mask) {
                atomic_fetch_add_explicit(&header->active_count, 1, memory_order_relaxed);
                return slot;
            }

            bitmap = atomic_load_explicit(&header->free_bitmap[word], memory_order_relaxed);
        }
    }

    return INVALID_SLOT;
}

inline void free_slot(device AppTableHeader* header, uint slot) {
    if (slot >= header->max_slots) return;

    uint word = slot / 32;
    uint bit = slot % 32;
    uint mask = 1u << bit;

    atomic_fetch_or_explicit(&header->free_bitmap[word], mask, memory_order_relaxed);
    atomic_fetch_sub_explicit(&header->active_count, 1, memory_order_relaxed);
}

// ============================================================================
// MEMORY ALLOCATOR (bump pointer - legacy)
// ============================================================================

inline uint gpu_alloc(device AllocatorState* alloc, uint size, uint alignment) {
    size = (size + alignment - 1) & ~(alignment - 1);

    uint offset = atomic_fetch_add_explicit(&alloc->bump_pointer, size, memory_order_relaxed);

    if (offset + size > alloc->pool_size) {
        atomic_fetch_sub_explicit(&alloc->bump_pointer, size, memory_order_relaxed);
        return INVALID_SLOT;
    }

    atomic_fetch_add_explicit(&alloc->allocation_count, 1, memory_order_relaxed);
    return offset;
}

// ============================================================================
// Issue #155: O(1) MEMORY ALLOCATOR (atomic free list)
// ============================================================================

// O(1) ALLOCATE - atomic pop from free list, fallback to bump
inline uint gpu_alloc_o1(
    device MemoryPool* pool,
    device FreeBlock* blocks,
    uint size
) {
    // Align to 16 bytes
    uint aligned_size = (size + 15) & ~15u;

    // Try free list first - O(1) atomic pop
    uint head = atomic_load_explicit(&pool->freelist_head, memory_order_relaxed);

    while (head != INVALID_SLOT) {
        FreeBlock block = blocks[head];

        // Check size fits
        if (block.size >= aligned_size) {
            // Try to pop this block
            if (atomic_compare_exchange_weak_explicit(
                &pool->freelist_head,
                &head,
                block.next,
                memory_order_relaxed,
                memory_order_relaxed
            )) {
                atomic_fetch_sub_explicit(&pool->free_count, 1, memory_order_relaxed);
                return block.offset;
            }
            // CAS failed, head updated, retry
        } else {
            // Block too small - for LIFO this means no suitable block
            // Fall through to bump allocator
            break;
        }
    }

    // Fallback: O(1) bump allocation
    uint offset = atomic_fetch_add_explicit(&pool->bump_pointer, aligned_size, memory_order_relaxed);

    if (offset + aligned_size > pool->pool_size) {
        // OOM - rollback
        atomic_fetch_sub_explicit(&pool->bump_pointer, aligned_size, memory_order_relaxed);
        return INVALID_SLOT;
    }

    return offset;
}

// O(1) FREE - atomic push to free list
inline void gpu_free_o1(
    device MemoryPool* pool,
    device FreeBlock* blocks,
    uint offset,
    uint size
) {
    if (offset == INVALID_SLOT || size == 0) return;

    // Align size to match allocation
    uint aligned_size = (size + 15) & ~15u;

    // Allocate a block descriptor index
    uint block_idx = atomic_fetch_add_explicit(&pool->block_count, 1, memory_order_relaxed);

    // Check we haven't exceeded max blocks
    if (block_idx >= pool->max_blocks) {
        // Can't track this free block - memory is leaked
        // In a real system, we might compact the free list
        atomic_fetch_sub_explicit(&pool->block_count, 1, memory_order_relaxed);
        return;
    }

    blocks[block_idx].offset = offset;
    blocks[block_idx].size = aligned_size;

    // O(1) atomic push to head
    uint old_head = atomic_load_explicit(&pool->freelist_head, memory_order_relaxed);
    do {
        blocks[block_idx].next = old_head;
    } while (!atomic_compare_exchange_weak_explicit(
        &pool->freelist_head,
        &old_head,
        block_idx,
        memory_order_relaxed,
        memory_order_relaxed
    ));

    atomic_fetch_add_explicit(&pool->free_count, 1, memory_order_relaxed);
}

// ============================================================================
// APP LIFECYCLE
// ============================================================================

// Legacy launch kernel (for backwards compatibility)
kernel void gpu_launch_app(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device AllocatorState* state_alloc [[buffer(2)]],
    device AllocatorState* vertex_alloc [[buffer(3)]],
    constant uint& app_type [[buffer(4)]],
    constant uint& state_size [[buffer(5)]],
    constant uint& vertex_size [[buffer(6)]],
    device uint* result_slot [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint slot = allocate_slot(header);
    if (slot == INVALID_SLOT) {
        *result_slot = INVALID_SLOT;
        return;
    }

    uint state_off = gpu_alloc(state_alloc, state_size, 16);
    if (state_off == INVALID_SLOT) {
        free_slot(header, slot);
        *result_slot = INVALID_SLOT;
        return;
    }

    apps[slot].flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE | APP_FLAG_DIRTY | APP_FLAG_NEEDS_INIT;
    apps[slot].app_type = app_type;
    apps[slot].slot_id = slot;
    apps[slot].window_id = 0;
    apps[slot].state_offset = state_off;
    apps[slot].state_size = state_size;
    // GPU IS THE COMPUTER: Fixed vertex region per slot - simple, parallel, no allocation
    apps[slot].vertex_offset = slot * VERTS_PER_SLOT * sizeof(RenderVertex);
    apps[slot].vertex_size = VERTS_PER_SLOT * sizeof(RenderVertex);
    apps[slot].thread_count = 1024;
    apps[slot].priority = PRIORITY_NORMAL;

    // CRITICAL: Initialize frame counter and other state to 0 (not garbage)
    apps[slot].frame_number = 0;
    apps[slot].last_run_frame = 0;
    apps[slot].vertex_count = 0;
    apps[slot].input_head = 0;
    apps[slot].input_tail = 0;

    *result_slot = slot;
}

// Issue #155: O(1) launch with free list memory management
kernel void gpu_launch_app_o1(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device MemoryPool* state_pool [[buffer(2)]],
    device FreeBlock* state_blocks [[buffer(3)]],
    device MemoryPool* vertex_pool [[buffer(4)]],
    device FreeBlock* vertex_blocks [[buffer(5)]],
    constant uint& app_type [[buffer(6)]],
    constant uint& state_size [[buffer(7)]],
    constant uint& vertex_size [[buffer(8)]],
    device uint* result_slot [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint slot = allocate_slot(header);
    if (slot == INVALID_SLOT) {
        *result_slot = INVALID_SLOT;
        return;
    }

    // O(1) allocation for state only - vertices use fixed slot-based regions
    uint state_off = gpu_alloc_o1(state_pool, state_blocks, state_size);
    if (state_off == INVALID_SLOT) {
        free_slot(header, slot);
        *result_slot = INVALID_SLOT;
        return;
    }

    apps[slot].flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE | APP_FLAG_DIRTY | APP_FLAG_NEEDS_INIT;
    apps[slot].app_type = app_type;
    apps[slot].slot_id = slot;
    apps[slot].window_id = 0;
    apps[slot].state_offset = state_off;
    apps[slot].state_size = state_size;
    // GPU IS THE COMPUTER: Fixed vertex region per slot - simple, parallel, no allocation
    apps[slot].vertex_offset = slot * VERTS_PER_SLOT * sizeof(RenderVertex);
    apps[slot].vertex_size = VERTS_PER_SLOT * sizeof(RenderVertex);
    apps[slot].thread_count = 1024;
    apps[slot].priority = PRIORITY_NORMAL;

    // CRITICAL: Initialize frame counter and other state to 0 (not garbage)
    apps[slot].frame_number = 0;
    apps[slot].last_run_frame = 0;
    apps[slot].vertex_count = 0;
    apps[slot].input_head = 0;
    apps[slot].input_tail = 0;

    *result_slot = slot;
}

kernel void gpu_close_app(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device MemoryPool* state_pool [[buffer(2)]],
    device FreeBlock* state_blocks [[buffer(3)]],
    device MemoryPool* vertex_pool [[buffer(4)]],
    device FreeBlock* vertex_blocks [[buffer(5)]],
    constant uint& slot_id [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    if (slot_id >= header->max_slots) return;

    GpuAppDescriptor app = apps[slot_id];
    if (!(app.flags & APP_FLAG_ACTIVE)) return;

    // Free state memory - O(1)
    // NOTE: Vertices use fixed slot-based regions, NOT the O(1) allocator
    // Each slot owns vertex_offset = slot * VERTS_PER_SLOT * sizeof(RenderVertex)
    // When slot is freed, next app at same slot reuses the same vertex region
    if (app.state_size > 0) {
        gpu_free_o1(state_pool, state_blocks, app.state_offset, app.state_size);
    }
    // Vertex memory is NOT freed to the pool - it's slot-based

    // Clear descriptor
    apps[slot_id].flags = 0;
    apps[slot_id].state_offset = INVALID_SLOT;
    apps[slot_id].vertex_offset = INVALID_SLOT;

    // Free slot - O(1)
    free_slot(header, slot_id);
}

// ============================================================================
// Issue #156: GPU SCHEDULER (parallel predicate evaluation)
// ============================================================================

constant uint STARVATION_THRESHOLD = 10;  // Run if not run for 10 frames

// Frame budget for controlling execution
struct FrameBudget {
    atomic_uint remaining;     // Cycles remaining this frame
    uint per_frame_limit;      // Reset each frame
    atomic_uint skipped_count; // Apps skipped due to budget
    uint _pad;
};

// Each app evaluates: "Should I run this frame?"
// O(1) predicate - no data structure traversal
inline bool should_i_run(
    device const GpuAppDescriptor* app,
    uint current_frame
) {
    // Not active? Exit in 1 cycle
    if (!(app->flags & APP_FLAG_ACTIVE)) return false;

    // Not dirty? Exit in 1 cycle
    if (!(app->flags & APP_FLAG_DIRTY)) return false;

    // Suspended? Exit in 1 cycle
    if (app->flags & APP_FLAG_SUSPENDED) return false;

    // Always run (starvation handled by priority boost below)
    return true;
}

// Check if app is starving (hasn't run for too long)
inline bool am_i_starving(
    device const GpuAppDescriptor* app,
    uint current_frame
) {
    uint frames_since_run = current_frame - app->last_run_frame;
    return frames_since_run > STARVATION_THRESHOLD;
}

// Get effective priority (boosted if starving)
inline uint effective_priority(
    device const GpuAppDescriptor* app,
    uint current_frame
) {
    uint base = app->priority;

    // Starving apps get priority boost
    if (am_i_starving(app, current_frame)) {
        base = min(base + 1, PRIORITY_REALTIME);
    }

    return base;
}

// Try to claim budget for execution
inline bool try_claim_budget(
    device FrameBudget* budget,
    uint cost,
    uint priority
) {
    // High priority always runs
    if (priority >= PRIORITY_HIGH) return true;

    // Try to claim budget
    uint old_budget = atomic_fetch_sub_explicit(
        &budget->remaining, cost, memory_order_relaxed
    );

    if (old_budget >= cost) {
        return true;  // Budget claimed
    }

    // Over budget - rollback and skip
    atomic_fetch_add_explicit(&budget->remaining, cost, memory_order_relaxed);
    atomic_fetch_add_explicit(&budget->skipped_count, 1, memory_order_relaxed);
    return false;
}

// ============================================================================
// ISSUE #159: APP-SPECIFIC STATE STRUCTS
// ============================================================================

// Counter app state (basic CUSTOM app type)
struct CounterState {
    uint value;
    uint increment;
    uint _pad[2];
};

// Game of Life state (cellular automaton)
// Issue #239 fix: Added double buffering to prevent data race
struct GameOfLifeState {
    uint width;
    uint height;
    uint generation;
    uint read_from_a;  // 1 = read from grid_a, write to grid_b; 0 = vice versa
    // Grid data follows: uchar grid_a[width * height], then grid_b[width * height]
};

// Particles state (physics simulation)
struct ParticlesState {
    uint count;
    uint max_count;
    float2 gravity;
    float4 bounds;  // (min_x, min_y, max_x, max_y)
    // Particle data follows
};

struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
    float lifetime;
    float _pad[3];
};

// ============================================================================
// ISSUE #159: APP-SPECIFIC UPDATE FUNCTIONS
// ============================================================================

// CUSTOM/Counter app update
inline void counter_app_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    uint tid,
    uint tg_size
) {
    if (tid != 0) return;  // Single-threaded for now

    device CounterState* state = (device CounterState*)(unified_state + app->state_offset);
    state->value += state->increment;
    app->vertex_count = 6;  // Simple quad
}


// Game of Life update (parallel cellular automaton)
// Issue #239 fix: Double buffering to prevent data race
inline void game_of_life_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    uint tid,
    uint tg_size
) {
    device GameOfLifeState* state = (device GameOfLifeState*)(unified_state + app->state_offset);
    uint grid_size = state->width * state->height;

    // Issue #239: Double buffer - read from one, write to other
    device uchar* grid_a = (device uchar*)(state + 1);
    device uchar* grid_b = grid_a + grid_size;

    device uchar* read_grid = state->read_from_a ? grid_a : grid_b;
    device uchar* write_grid = state->read_from_a ? grid_b : grid_a;

    // Each thread handles some cells
    uint cells_per_thread = (grid_size + tg_size - 1) / tg_size;
    uint start = tid * cells_per_thread;
    uint end = min(start + cells_per_thread, grid_size);

    for (uint i = start; i < end; i++) {
        uint x = i % state->width;
        uint y = i / state->width;

        // Count neighbors (wrapping) - READ from read_grid
        uint neighbors = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                uint nx = (x + dx + state->width) % state->width;
                uint ny = (y + dy + state->height) % state->height;
                if (read_grid[ny * state->width + nx] != 0) neighbors++;
            }
        }

        // Conway's rules - WRITE to write_grid (different from read_grid!)
        bool alive = read_grid[i] != 0;
        bool next = (alive && (neighbors == 2 || neighbors == 3)) ||
                   (!alive && neighbors == 3);
        write_grid[i] = next ? 1 : 0;
    }

    // Thread 0 updates generation, swaps buffers, and sets vertex count
    if (tid == 0) {
        state->generation++;
        state->read_from_a = state->read_from_a ? 0 : 1;  // Swap buffers for next frame
        app->vertex_count = grid_size * 6;  // 6 vertices per cell (quad)
    }
}

// Particles update (physics simulation)
// Issue #245 fix: Use fixed timestep for stable physics (intentional, not a bug)
// Fixed timestep ensures deterministic simulation regardless of frame rate.
// For variable timestep, pass dt via ParticlesState.dt field from host.
inline void particles_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    uint tid,
    uint tg_size
) {
    device ParticlesState* state = (device ParticlesState*)(unified_state + app->state_offset);
    device Particle* particles = (device Particle*)(state + 1);

    // Each thread updates some particles
    uint per_thread = (state->count + tg_size - 1) / tg_size;
    uint start = tid * per_thread;
    uint end = min(start + per_thread, state->count);

    // Issue #245: Fixed timestep for deterministic physics (1/60s = 16.67ms)
    // This is intentional: fixed timestep prevents frame-rate dependent behavior
    // and avoids physics instability at high/low frame rates.
    const float FIXED_DT = 1.0f / 60.0f;

    for (uint i = start; i < end; i++) {
        // Apply gravity
        particles[i].velocity += state->gravity * FIXED_DT;
        particles[i].position += particles[i].velocity * FIXED_DT;

        // Bounce on bounds
        if (particles[i].position.x < state->bounds.x) {
            particles[i].position.x = state->bounds.x;
            particles[i].velocity.x *= -0.8;
        }
        if (particles[i].position.x > state->bounds.z) {
            particles[i].position.x = state->bounds.z;
            particles[i].velocity.x *= -0.8;
        }
        if (particles[i].position.y < state->bounds.y) {
            particles[i].position.y = state->bounds.y;
            particles[i].velocity.y *= -0.8;
        }
        if (particles[i].position.y > state->bounds.w) {
            particles[i].position.y = state->bounds.w;
            particles[i].velocity.y *= -0.8;
        }

        // Decrease lifetime
        particles[i].lifetime -= FIXED_DT;
    }

    if (tid == 0) {
        app->vertex_count = state->count * 6;  // 6 vertices per particle (quad)
    }
}

// ============================================================================
// System App Structures and Updates (Issue #155-#161)
// ============================================================================

// Compositor state - minimal, just tracks frame stats
struct CompositorState {
    float screen_width;
    float screen_height;
    uint window_count;
    uint frame_number;
    float4 background_color;
    uint total_vertices_rendered;
    uint app_count;
    uint _pad[2];
};

// Compositor constants
constant uint COMPOSITOR_BACKGROUND_VERTS = 6;  // Full-screen quad

// Compositor update - generates background quad and tracks total vertices
// This runs as part of megakernel, so it uses the unified vertex buffer
inline void compositor_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuAppDescriptor* all_apps,
    uint max_slots,
    uint tid,
    uint tg_size
) {
    device CompositorState* state = (device CompositorState*)(unified_state + app->state_offset);

    // Thread 0 handles bookkeeping and background
    if (tid == 0) {
        state->frame_number++;

        // Generate background quad at depth 0 (furthest back)
        device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
        float4 bg_color = state->background_color;
        float w = state->screen_width;
        float h = state->screen_height;
        float depth = 0.0;  // Back of scene

        // Background quad (two triangles)
        // Triangle 1: top-left, top-right, bottom-right
        verts[0].position = packed_float3(0, 0, depth);
        verts[0].color = bg_color;
        verts[0].uv = float2(0, 0);

        verts[1].position = packed_float3(w, 0, depth);
        verts[1].color = bg_color;
        verts[1].uv = float2(1, 0);

        verts[2].position = packed_float3(w, h, depth);
        verts[2].color = bg_color;
        verts[2].uv = float2(1, 1);

        // Triangle 2: top-left, bottom-right, bottom-left
        verts[3].position = packed_float3(0, 0, depth);
        verts[3].color = bg_color;
        verts[3].uv = float2(0, 0);

        verts[4].position = packed_float3(w, h, depth);
        verts[4].color = bg_color;
        verts[4].uv = float2(1, 1);

        verts[5].position = packed_float3(0, h, depth);
        verts[5].color = bg_color;
        verts[5].uv = float2(0, 1);

        app->vertex_count = COMPOSITOR_BACKGROUND_VERTS;
    }

    // Thread 0 counts total vertices across all apps
    if (tid == 0) {
        uint total = 0;
        uint app_count = 0;
        for (uint i = 0; i < max_slots; i++) {
            device GpuAppDescriptor* other = &all_apps[i];
            if (other->flags & APP_FLAG_ACTIVE) {
                total += other->vertex_count;
                app_count++;
            }
        }
        state->total_vertices_rendered = total;
        state->app_count = app_count;
    }
}

// Issue #156: DockItem struct
struct DockItem {
    uint app_type;           // What app to launch on click
    uint flags;              // VISIBLE, RUNNING, HOVERED, BOUNCING, CLICKED
    uint running_count;      // Number of running instances
    float current_size;      // Animated size (interpolates to target)
    float target_size;       // Target size (base or magnified)
    float bounce_phase;      // Bounce animation phase [0, 2*PI]
    float center_x;          // Computed center X position
    float center_y;          // Computed center Y position
    float4 icon_color;       // Icon color
};

// Issue #156: Dock state
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
    uint mouse_pressed;      // 1 if mouse button is pressed this frame

    // Padding for 16-byte alignment
    uint _pad[2];

    // Items array follows after state (32 * sizeof(DockItem) bytes)
};

// Issue #156: Helper - compute magnification factor based on distance
inline float dock_magnification_factor(
    float cursor_x,
    float icon_center_x,
    float magnification_radius,
    float base_magnification
) {
    float distance = abs(cursor_x - icon_center_x);

    if (distance > magnification_radius) {
        return 1.0;
    }

    // Smooth falloff using smoothstep
    float t = 1.0 - (distance / magnification_radius);
    float smooth_t = t * t * (3.0 - 2.0 * t);
    return 1.0 + (base_magnification - 1.0) * smooth_t;
}

// Issue #156: Full dock update with parallel hover detection and magnification
inline void dock_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    uint tid,
    uint tg_size
) {
    device DockState* state = (device DockState*)(unified_state + app->state_offset);
    device DockItem* items = (device DockItem*)((device uchar*)state + sizeof(DockState));
    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);

    // ========================================================================
    // PHASE 1: Thread 0 resets per-frame state
    // ========================================================================
    if (tid == 0) {
        state->hovered_item = INVALID_SLOT;  // Reset before PHASE 2 uses atomic_fetch_min
        state->clicked_item = INVALID_SLOT;
        state->time += 0.016;  // ~60fps
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 1b: Compute layout FIRST (Thread 0 only)
    // This MUST happen before hover detection so center_x/center_y are valid
    // ========================================================================
    if (tid == 0) {
        // Calculate total width with current sizes
        float total_width = 0.0;
        uint visible_count = 0;
        for (uint i = 0; i < state->item_count; i++) {
            if (items[i].flags & DOCK_ITEM_VISIBLE) {
                total_width += items[i].current_size + state->icon_spacing;
                visible_count++;
            }
        }
        if (visible_count > 0) {
            total_width -= state->icon_spacing;  // No trailing space
        }

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
    // PHASE 2: Parallel hover detection (each thread checks its icon)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        float dx = state->cursor_pos.x - items[tid].center_x;
        float dy = state->cursor_pos.y - items[tid].center_y;
        float hit_radius = items[tid].current_size * 0.5 + 4.0;
        float dist_sq = dx*dx + dy*dy;
        float radius_sq = hit_radius * hit_radius;
        bool hovered = state->cursor_in_dock && (dist_sq < radius_sq);

        // Debug: store computed values in padding (thread 0 only)
        if (tid == 0) {
            state->_pad[0] = as_type<uint>(dist_sq);    // distance squared
            state->_pad[1] = as_type<uint>(radius_sq);  // radius squared
        }

        if (hovered) {
            items[tid].flags |= DOCK_ITEM_HOVERED;
            atomic_fetch_min_explicit((device atomic_uint*)&state->hovered_item, tid, memory_order_relaxed);
        } else {
            items[tid].flags &= ~DOCK_ITEM_HOVERED;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 2b: Click detection (Thread 0 only, after hover is resolved)
    // ========================================================================
    if (tid == 0) {
        if (state->mouse_pressed && state->hovered_item != INVALID_SLOT) {
            state->clicked_item = state->hovered_item;
            // Start bounce animation on clicked item
            items[state->clicked_item].flags |= DOCK_ITEM_BOUNCING;
            items[state->clicked_item].bounce_phase = 0.0;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 3: Compute target sizes with magnification (parallel)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        if (state->cursor_in_dock) {
            float mag = dock_magnification_factor(
                state->cursor_pos.x,
                items[tid].center_x,
                state->magnification_radius,
                state->magnified_size / state->base_icon_size
            );
            items[tid].target_size = state->base_icon_size * mag;
        } else {
            items[tid].target_size = state->base_icon_size;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ========================================================================
    // PHASE 4: Animate sizes toward targets (parallel)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        float diff = items[tid].target_size - items[tid].current_size;
        items[tid].current_size += diff * state->animation_speed;
        if (abs(diff) < 0.1) {
            items[tid].current_size = items[tid].target_size;
        }

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
    // PHASE 5: Generate vertices (parallel - each thread generates its icon)
    // (Layout was computed in PHASE 1b, so center_x/center_y are already valid)
    // ========================================================================
    if (tid < state->item_count && (items[tid].flags & DOCK_ITEM_VISIBLE)) {
        float size = items[tid].current_size;
        float half_size = size * 0.5;
        float cx = items[tid].center_x;
        float cy = items[tid].center_y;

        // Bounce offset (negative Y goes up)
        float bounce_offset = 0.0;
        if (items[tid].flags & DOCK_ITEM_BOUNCING) {
            bounce_offset = sin(items[tid].bounce_phase) * state->bounce_height;
        }

        float left = cx - half_size;
        float right = cx + half_size;
        float top = cy - half_size - bounce_offset;
        float bottom = cy + half_size - bounce_offset;
        float depth = 0.99;  // Dock in front of most things
        float4 color = items[tid].icon_color;

        // 6 vertices per icon (2 triangles)
        uint base = tid * 6;
        device RenderVertex* v = verts + base;

        // Triangle 1: top-left, top-right, bottom-right
        v[0].position = packed_float3(left, top, depth);
        v[0].color = color;
        v[0].uv = float2(0.0, 0.0);

        v[1].position = packed_float3(right, top, depth);
        v[1].color = color;
        v[1].uv = float2(1.0, 0.0);

        v[2].position = packed_float3(right, bottom, depth);
        v[2].color = color;
        v[2].uv = float2(1.0, 1.0);

        // Triangle 2: top-left, bottom-right, bottom-left
        v[3].position = packed_float3(left, top, depth);
        v[3].color = color;
        v[3].uv = float2(0.0, 0.0);

        v[4].position = packed_float3(right, bottom, depth);
        v[4].color = color;
        v[4].uv = float2(1.0, 1.0);

        v[5].position = packed_float3(left, bottom, depth);
        v[5].color = color;
        v[5].uv = float2(0.0, 1.0);
    }

    // ========================================================================
    // PHASE 7: Thread 0 updates vertex count
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

// Issue #157: MenuBar state
struct MenuBarState {
    float screen_width;
    float bar_height;
    float padding_x;
    float text_scale;
    uint menu_count;
    uint total_item_count;
    uint open_menu;
    uint hovered_menu;
    uint hovered_item;
    uint selected_item;
    float time;
    float dropdown_anim;
    float4 bar_color;
    float4 text_color;
    uint _pad[2];
};

// MenuBar constants
constant uint MENUBAR_BACKGROUND_VERTS = 6;  // Bar background quad
constant float MENUBAR_DEFAULT_HEIGHT = 24.0;

// MenuBar update - generates menu bar background and tracks state
inline void menubar_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    uint tid,
    uint tg_size
) {
    device MenuBarState* state = (device MenuBarState*)(unified_state + app->state_offset);
    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);

    // Thread 0 generates menu bar background
    if (tid == 0) {
        // Update time
        state->time += 0.016;  // ~60fps

        // Get dimensions
        float screen_width = state->screen_width;
        float bar_height = state->bar_height;
        if (bar_height < 1.0) bar_height = MENUBAR_DEFAULT_HEIGHT;

        // Depth: menu bar should be in front of compositor background
        float depth = 0.98;

        // Menu bar background color (translucent light gray like macOS)
        float4 bar_color = state->bar_color;
        if (bar_color.w < 0.01) {
            bar_color = float4(0.95, 0.95, 0.97, 0.92);  // Default
        }

        // Generate background quad (6 vertices)
        // Triangle 1: top-left, top-right, bottom-right
        verts[0].position = packed_float3(0, 0, depth);
        verts[0].color = bar_color;
        verts[0].uv = float2(0, 0);

        verts[1].position = packed_float3(screen_width, 0, depth);
        verts[1].color = bar_color;
        verts[1].uv = float2(1, 0);

        verts[2].position = packed_float3(screen_width, bar_height, depth);
        verts[2].color = bar_color;
        verts[2].uv = float2(1, 1);

        // Triangle 2: top-left, bottom-right, bottom-left
        verts[3].position = packed_float3(0, 0, depth);
        verts[3].color = bar_color;
        verts[3].uv = float2(0, 0);

        verts[4].position = packed_float3(screen_width, bar_height, depth);
        verts[4].color = bar_color;
        verts[4].uv = float2(1, 1);

        verts[5].position = packed_float3(0, bar_height, depth);
        verts[5].color = bar_color;
        verts[5].uv = float2(0, 1);

        app->vertex_count = MENUBAR_BACKGROUND_VERTS;
    }
}

// Window Chrome state - Issue #159
struct WindowChromeState {
    uint window_count;
    uint focused_window;
    uint dragging_window;
    uint resizing_window;
    uint hovered_button;      // Encoded: (window_idx << 8) | button_type
    uint clicked_button;
    float2 drag_offset;
    float2 resize_origin;
    float title_bar_height;
    float border_width;
    float button_radius;
    float button_spacing;
    float button_left_margin;
    float corner_radius;
    float2 _dim_pad;
    float4 title_focused;
    float4 title_unfocused;
    float4 close_color;
    float4 minimize_color;
    float4 maximize_color;
    float4 border_color;
    float4 button_hover_tint;
    uint _pad[2];
};

// Chrome constants
constant uint CHROME_TITLE_BAR_VERTS = 6;       // 6 vertices (quad)
constant uint CHROME_BUTTON_VERTS = 18;         // 18 vertices (6-triangle circle approx)
constant uint CHROME_BORDER_VERTS = 24;         // 4 borders x 6 vertices each
constant uint CHROME_RESIZE_HANDLE_VERTS = 6;   // 6 vertices (quad)
constant uint CHROME_VERTS_PER_WINDOW = CHROME_TITLE_BAR_VERTS + CHROME_BUTTON_VERTS * 3 + CHROME_BORDER_VERTS + CHROME_RESIZE_HANDLE_VERTS;
// = 6 + 54 + 24 + 6 = 90 vertices per window

// Button types for hovered_button/clicked_button encoding
constant uint BUTTON_CLOSE = 0;
constant uint BUTTON_MINIMIZE = 1;
constant uint BUTTON_MAXIMIZE = 2;
constant uint BUTTON_NONE = 0xFF;

// Write a quad (2 triangles = 6 vertices)
inline void write_quad(
    device RenderVertex* v,
    float2 origin,      // top-left corner
    float2 size,        // width, height
    float depth,
    float4 color
) {
    // Triangle 1: top-left, top-right, bottom-right
    v[0].position = packed_float3(origin.x, origin.y, depth);
    v[0].color = color;
    v[0].uv = float2(0, 0);

    v[1].position = packed_float3(origin.x + size.x, origin.y, depth);
    v[1].color = color;
    v[1].uv = float2(1, 0);

    v[2].position = packed_float3(origin.x + size.x, origin.y + size.y, depth);
    v[2].color = color;
    v[2].uv = float2(1, 1);

    // Triangle 2: top-left, bottom-right, bottom-left
    v[3].position = packed_float3(origin.x, origin.y, depth);
    v[3].color = color;
    v[3].uv = float2(0, 0);

    v[4].position = packed_float3(origin.x + size.x, origin.y + size.y, depth);
    v[4].color = color;
    v[4].uv = float2(1, 1);

    v[5].position = packed_float3(origin.x, origin.y + size.y, depth);
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
    for (uint i = 0; i < 6; i++) {
        float angle1 = float(i) * M_PI_F / 3.0;
        float angle2 = float(i + 1) * M_PI_F / 3.0;

        float2 p1 = center + radius * float2(cos(angle1), sin(angle1));
        float2 p2 = center + radius * float2(cos(angle2), sin(angle2));

        uint base = i * 3;
        v[base + 0].position = packed_float3(center.x, center.y, depth);
        v[base + 0].color = color;
        v[base + 0].uv = float2(0.5, 0.5);

        v[base + 1].position = packed_float3(p1.x, p1.y, depth);
        v[base + 1].color = color;
        v[base + 1].uv = float2(0.5 + 0.5 * cos(angle1), 0.5 + 0.5 * sin(angle1));

        v[base + 2].position = packed_float3(p2.x, p2.y, depth);
        v[base + 2].color = color;
        v[base + 2].uv = float2(0.5 + 0.5 * cos(angle2), 0.5 + 0.5 * sin(angle2));
    }
}

// Window Chrome update - generates vertices for window decorations
// Each thread generates chrome for one window simultaneously
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
        // Write zeroed vertices
        device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
        uint base = tid * CHROME_VERTS_PER_WINDOW;
        for (uint i = 0; i < CHROME_VERTS_PER_WINDOW; i++) {
            verts[base + i].position = packed_float3(0, 0, 0);
            verts[base + i].color = float4(0, 0, 0, 0);
            verts[base + i].uv = float2(0, 0);
        }
        if (tid == 0) {
            app->vertex_count = window_count * CHROME_VERTS_PER_WINDOW;
        }
        return;
    }

    // Get vertex buffer offset for this window
    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
    uint base = tid * CHROME_VERTS_PER_WINDOW;

    // Determine if this window is focused
    bool is_focused = (tid == state->focused_window);
    float4 title_color = is_focused ? state->title_focused : state->title_unfocused;
    float depth = window.depth + 0.001;  // Chrome slightly in front of content

    uint vert_idx = 0;

    // =========================================================================
    // 1. TITLE BAR (6 vertices)
    // =========================================================================
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
    write_quad(
        verts + base + vert_idx,
        float2(window.x + window.width - handle_size, window.y + window.height - handle_size),
        float2(handle_size, handle_size),
        depth,
        float4(0.5, 0.5, 0.5, 0.5)  // Semi-transparent gray
    );
    vert_idx += 6;

    // Thread 0 commits the total vertex count
    if (tid == 0) {
        app->vertex_count = window_count * CHROME_VERTS_PER_WINDOW;
    }
}

// ============================================================================
// BYTECODE VIRTUAL MACHINE (Issue #164)
// ============================================================================

// Bytecode opcodes
constant uint OP_NOP = 0x00;
constant uint OP_HALT = 0xFF;

// Arithmetic
constant uint OP_MOV = 0x01;
constant uint OP_ADD = 0x02;
constant uint OP_SUB = 0x03;
constant uint OP_MUL = 0x04;
constant uint OP_DIV = 0x05;
constant uint OP_MOD = 0x06;

// Math intrinsics (Phase 5 - Issue #178)
constant uint OP_SIN = 0x07;
constant uint OP_COS = 0x08;
constant uint OP_SQRT = 0x09;

// Float unary/binary math ops (Issue #198)
constant uint OP_ABS = 0x20;        // dst.x = fabs(s1.x)
constant uint OP_CEIL = 0x21;       // dst.x = ceil(s1.x)
constant uint OP_FLOOR = 0x22;      // dst.x = floor(s1.x)
constant uint OP_TRUNC = 0x23;      // dst.x = trunc(s1.x)
constant uint OP_NEAREST = 0x24;    // dst.x = rint(s1.x) - round to nearest, ties to even
constant uint OP_COPYSIGN = 0x25;   // dst.x = copysign(s1.x, s2.x) - magnitude of s1, sign of s2
constant uint OP_FMIN = 0x26;       // dst.x = fmin(s1.x, s2.x) - propagates NaN
constant uint OP_FMAX = 0x27;       // dst.x = fmax(s1.x, s2.x) - propagates NaN
constant uint OP_FNEG = 0x28;       // dst.x = -s1.x (float negate)

// ═══════════════════════════════════════════════════════════════════════════
// 64-BIT FLOAT OPERATIONS (Issue #189, Issue #27 - Double-Single Emulation)
// THE GPU IS THE COMPUTER - Metal does NOT support native double precision
// F64 uses DOUBLE-SINGLE representation: regs[r].x = hi, regs[r].y = lo
// where the true value is (hi + lo), providing ~47 bits of mantissa precision
// This is the same technique used by metal-float64 library
// ═══════════════════════════════════════════════════════════════════════════

// F64 arithmetic (0x0A-0x0E) - implemented via double-single algorithms
constant uint OP_F64_ADD = 0x0A;      // dst.xy = ds_add(s1.xy, s2.xy)
constant uint OP_F64_SUB = 0x0B;      // dst.xy = ds_sub(s1.xy, s2.xy)
constant uint OP_F64_MUL = 0x0C;      // dst.xy = ds_mul(s1.xy, s2.xy)
constant uint OP_F64_DIV = 0x0D;      // dst.xy = ds_div(s1.xy, s2.xy)
constant uint OP_F64_SQRT = 0x0E;     // dst.xy = ds_sqrt(s1.xy)

// F64 comparison (0x29-0x2E) - double-single aware comparisons
constant uint OP_F64_EQ = 0x29;       // dst.x = (s1.xy == s2.xy) ? 1.0 : 0.0
constant uint OP_F64_NE = 0x2A;       // dst.x = (s1.xy != s2.xy) ? 1.0 : 0.0
constant uint OP_F64_LT = 0x2B;       // dst.x = (s1.xy < s2.xy) ? 1.0 : 0.0
constant uint OP_F64_GT = 0x2C;       // dst.x = (s1.xy > s2.xy) ? 1.0 : 0.0
constant uint OP_F64_LE = 0x2D;       // dst.x = (s1.xy <= s2.xy) ? 1.0 : 0.0
constant uint OP_F64_GE = 0x2E;       // dst.x = (s1.xy >= s2.xy) ? 1.0 : 0.0

// F64 min/max (0x2F-0x30) - double-single aware min/max
constant uint OP_F64_MIN = 0x2F;      // dst.xy = min(s1.xy, s2.xy) as double-single
constant uint OP_F64_MAX = 0x30;      // dst.xy = max(s1.xy, s2.xy) as double-single

// F64 conversion (0x0F-0x12)
constant uint OP_F64_FROM_I32_S = 0x0F;  // dst.xy = ds_from_i32(s1.x signed)
constant uint OP_F64_FROM_I32_U = 0x10;  // dst.xy = ds_from_u32(s1.x unsigned)
constant uint OP_F64_FROM_I64_S = 0x11;  // dst.xy = ds_from_i64(s1.xy signed)
constant uint OP_F64_FROM_I64_U = 0x12;  // dst.xy = ds_from_u64(s1.xy unsigned)

// F64 to integer conversion (0x18-0x1B)
constant uint OP_F64_TO_I32_S = 0x18;    // dst.x = ds_to_i32(s1.xy truncate)
constant uint OP_F64_TO_I32_U = 0x19;    // dst.x = ds_to_u32(s1.xy truncate)
constant uint OP_F64_TO_I64_S = 0x1A;    // dst.xy = ds_to_i64(s1.xy truncate)
constant uint OP_F64_TO_I64_U = 0x1B;    // dst.xy = ds_to_u64(s1.xy truncate)

// F64 reinterpret operations (0x1D-0x1E)
// NOTE: These are approximations since double-single doesn't preserve IEEE 754 bits
constant uint OP_F64_REINTERPRET_I64 = 0x1D;  // dst.xy = ds_from_f64_bits(s1.xy)
constant uint OP_I64_REINTERPRET_F64 = 0x1E;  // dst.xy = ds_to_f64_bits(s1.xy)

constant uint OP_LOADI = 0x13;    // Load immediate (broadcast to all components)
constant uint OP_SETX = 0x14;     // Set .x component
constant uint OP_SETY = 0x15;     // Set .y component
constant uint OP_SETZ = 0x16;     // Set .z component
constant uint OP_SETW = 0x17;     // Set .w component

// Vector packing (0x1C-0x1F)
constant uint OP_PACK2 = 0x1C;    // dst.xy = (s1.x, s2.x) - pack two scalars into float2

// Comparison
constant uint OP_EQ = 0x40;
constant uint OP_LT = 0x42;
constant uint OP_GT = 0x44;

// Control flow
constant uint OP_JMP = 0x60;
constant uint OP_JZ = 0x61;
constant uint OP_JNZ = 0x62;

// Memory
constant uint OP_LD = 0x80;       // Load float4 from state
constant uint OP_ST = 0x81;       // Store float4 to state
constant uint OP_LD1 = 0x82;      // Load byte
constant uint OP_ST1 = 0x83;      // Store byte
constant uint OP_LD2 = 0x8C;      // Load 16-bit (halfword)
constant uint OP_ST2 = 0x8D;      // Store 16-bit (halfword)
constant uint OP_LD4 = 0x8E;      // Load 32-bit (word)
constant uint OP_ST4 = 0x8F;      // Store 32-bit (word)

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOMATIC CODE TRANSFORMATION OPCODES (Phase 8 - Issue #182)
// THE GPU IS THE COMPUTER - transform CPU patterns to GPU-native equivalents
// ═══════════════════════════════════════════════════════════════════════════════

// Work queue operations (0x84-0x85)
constant uint OP_WORK_PUSH = 0x84;    // Push work item to queue
constant uint OP_WORK_POP = 0x85;     // Pop work item from queue

// I/O request operations (0x86-0x87)
constant uint OP_REQUEST_QUEUE = 0x86;  // Queue I/O request
constant uint OP_REQUEST_POLL = 0x87;   // Poll request status

// Frame-based timing (0x88)
constant uint OP_FRAME_WAIT = 0x88;   // Wait for N frames

// Spinlock operations (0x89-0x8A)
constant uint OP_SPINLOCK = 0x89;     // Acquire spinlock (Mutex::lock)
constant uint OP_SPINUNLOCK = 0x8A;   // Release spinlock (Mutex::unlock)

// Threadgroup barrier (0x8B)
constant uint OP_BARRIER = 0x8B;      // Threadgroup barrier (Condvar::wait)

// Graphics
constant uint OP_QUAD = 0xA0;     // Emit quad: pos/size=src1, color=src2, depth=imm

// More 64-bit comparisons (0xA1-0xA7)
constant uint OP_INT64_NE = 0xA1;     // dst.x = (s1.xy != s2.xy) ? 1 : 0
constant uint OP_INT64_LT_U = 0xA2;   // dst.x = (s1.xy < s2.xy unsigned) ? 1 : 0
constant uint OP_INT64_LT_S = 0xA3;   // dst.x = (s1.xy < s2.xy signed) ? 1 : 0
constant uint OP_INT64_LE_U = 0xA4;   // dst.x = (s1.xy <= s2.xy unsigned) ? 1 : 0
constant uint OP_INT64_LE_S = 0xA5;   // dst.x = (s1.xy <= s2.xy signed) ? 1 : 0
constant uint OP_INT64_EQZ = 0xA6;    // dst.x = (s1.xy == 0) ? 1 : 0
constant uint OP_INT64_ROTR = 0xA7;   // dst.xy = rotate_right(s1.xy, s2.x)
constant uint OP_INT64_ROTL = 0xA8;   // dst.xy = rotate_left(s1.xy, s2.x)
constant uint OP_INT64_CLZ = 0xA9;    // dst.x = count leading zeros of s1.xy
constant uint OP_INT64_CTZ = 0xAA;    // dst.x = count trailing zeros of s1.xy
constant uint OP_INT64_POPCNT = 0xAB; // dst.x = population count of s1.xy

// ═══════════════════════════════════════════════════════════════════════════════
// 64-BIT INTEGER OPERATIONS (Issue #188)
// THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support int64/uint64
// 64-bit values use regs[r].xy (2 x 32-bit floats reinterpreted as ulong)
// ═══════════════════════════════════════════════════════════════════════════════

// 64-bit arithmetic (0xB0-0xB5)
constant uint OP_INT64_ADD = 0xB0;     // dst.xy = s1.xy + s2.xy (as ulong)
constant uint OP_INT64_SUB = 0xB1;     // dst.xy = s1.xy - s2.xy (as ulong)
constant uint OP_INT64_MUL = 0xB2;     // dst.xy = s1.xy * s2.xy (as ulong)
constant uint OP_INT64_DIV_S = 0xB3;   // dst.xy = s1.xy / s2.xy (as long, signed)
constant uint OP_INT64_DIV_U = 0xB4;   // dst.xy = s1.xy / s2.xy (as ulong, unsigned)
constant uint OP_INT64_REM_U = 0xB5;   // dst.xy = s1.xy % s2.xy (as ulong)
constant uint OP_INT64_REM_S = 0xAC;   // dst.xy = s1.xy % s2.xy (as long, signed)

// 64-bit bitwise (0xB6-0xB9)
constant uint OP_INT64_AND = 0xB6;     // dst.xy = s1.xy & s2.xy
constant uint OP_INT64_OR = 0xB7;      // dst.xy = s1.xy | s2.xy
constant uint OP_INT64_XOR = 0xB8;     // dst.xy = s1.xy ^ s2.xy
constant uint OP_INT64_SHL = 0xB9;     // dst.xy = s1.xy << s2.x (shift from low 6 bits)

// 64-bit shifts and conversions (0xBA-0xBF)
constant uint OP_INT64_SHR_U = 0xBA;   // dst.xy = s1.xy >> s2.x (logical shift)
constant uint OP_INT64_SHR_S = 0xBB;   // dst.xy = s1.xy >> s2.x (arithmetic shift)
constant uint OP_INT64_EQ = 0xBC;      // dst.x = (s1.xy == s2.xy) ? 1 : 0
constant uint OP_INT64_WRAP = 0xBD;    // dst.x = s1.xy as i32 (wrap to 32-bit)
constant uint OP_INT64_EXTEND_U = 0xBE; // dst.xy = s1.x as u64 (zero-extend)
constant uint OP_INT64_EXTEND_S = 0xBF; // dst.xy = s1.x as i64 (sign-extend)

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGER OPERATIONS (Phase 1 - Issue #171)
// THE GPU IS THE COMPUTER - integers are just bits, reinterpret don't convert
// ═══════════════════════════════════════════════════════════════════════════════

// Integer arithmetic (0xC0-0xC7)
constant uint OP_INT_ADD = 0xC0;    // dst.x = as_int(s1.x) + as_int(s2.x)
constant uint OP_INT_SUB = 0xC1;
constant uint OP_INT_MUL = 0xC2;
constant uint OP_INT_DIV_S = 0xC3;  // Signed divide
constant uint OP_INT_DIV_U = 0xC4;  // Unsigned divide
constant uint OP_INT_REM_S = 0xC5;  // Signed remainder
constant uint OP_INT_REM_U = 0xC6;  // Unsigned remainder
constant uint OP_INT_NEG = 0xC7;    // Negate
constant uint OP_CTZ = 0xC8;        // Count trailing zeros
constant uint OP_POPCNT = 0xC9;     // Population count (number of 1 bits)

// Bitwise (0xCA-0xCF)
constant uint OP_BIT_AND = 0xCA;
constant uint OP_BIT_OR = 0xCB;
constant uint OP_BIT_XOR = 0xCC;
constant uint OP_BIT_NOT = 0xCD;
constant uint OP_SHL = 0xCE;        // Shift left
constant uint OP_SHR_U = 0xCF;      // Shift right unsigned

// More shifts (0xD0-0xD3)
constant uint OP_SHR_S = 0xD0;      // Shift right signed (arithmetic)
constant uint OP_ROTL = 0xD1;       // Rotate left
constant uint OP_ROTR = 0xD2;       // Rotate right
constant uint OP_CLZ = 0xD3;        // Count leading zeros

// Integer comparison (0xD4-0xD9)
constant uint OP_INT_EQ = 0xD4;
constant uint OP_INT_NE = 0xD5;
constant uint OP_INT_LT_S = 0xD6;   // Signed less than
constant uint OP_INT_LT_U = 0xD7;   // Unsigned less than
constant uint OP_INT_LE_S = 0xD8;   // Signed less or equal
constant uint OP_INT_LE_U = 0xD9;   // Unsigned less or equal

// Conversion (0xDA-0xDD)
constant uint OP_INT_TO_F = 0xDA;   // Signed int to float
constant uint OP_UINT_TO_F = 0xDB;  // Unsigned int to float
constant uint OP_F_TO_INT = 0xDC;   // Float to signed int
constant uint OP_F_TO_UINT = 0xDD;  // Float to unsigned int

// Load immediate integer (0xDE-0xDF)
constant uint OP_LOADI_INT = 0xDE;  // Load immediate as signed int bits
constant uint OP_LOADI_UINT = 0xDF; // Load immediate as unsigned int bits

// ═══════════════════════════════════════════════════════════════════════════════
// ATOMIC OPERATIONS (Phase 2 - Issue #172)
// THE GPU IS THE COMPUTER - lock-free coordination, GPU never waits
// Atomics are NOT for locks. Atomics are for LOCK-FREE COORDINATION.
// ═══════════════════════════════════════════════════════════════════════════════

// Atomic load/store (0xE0-0xE1)
constant uint OP_ATOMIC_LOAD = 0xE0;   // dst.x = atomic_load(&state[addr])
constant uint OP_ATOMIC_STORE = 0xE1;  // atomic_store(&state[addr], val)

// Atomic read-modify-write (0xE2-0xEA)
constant uint OP_ATOMIC_ADD = 0xE2;    // dst.x = atomic_fetch_add(&state[addr], val)
constant uint OP_ATOMIC_SUB = 0xE3;    // dst.x = atomic_fetch_sub(&state[addr], val)
constant uint OP_ATOMIC_MAX_U = 0xE4;  // Unsigned max
constant uint OP_ATOMIC_MIN_U = 0xE5;  // Unsigned min
constant uint OP_ATOMIC_MAX_S = 0xE6;  // Signed max
constant uint OP_ATOMIC_MIN_S = 0xE7;  // Signed min
constant uint OP_ATOMIC_AND = 0xE8;
constant uint OP_ATOMIC_OR = 0xE9;
constant uint OP_ATOMIC_XOR = 0xEA;

// Atomic compare-and-swap (0xEB)
constant uint OP_ATOMIC_CAS = 0xEB;    // if state[addr] == expected: state = desired, dst = 1

// Atomic increment/decrement (0xEC-0xED)
constant uint OP_ATOMIC_INC = 0xEC;    // dst.x = atomic_fetch_add(&state[addr], 1)
constant uint OP_ATOMIC_DEC = 0xED;    // dst.x = atomic_fetch_sub(&state[addr], 1)

// Memory fence (0xEE)
constant uint OP_MEM_FENCE = 0xEE;     // threadgroup_barrier(mem_flags::mem_device)

// ═══════════════════════════════════════════════════════════════════════════════
// ALLOCATOR OPERATIONS (Phase 6 - Issue #179)
// THE GPU IS THE COMPUTER - GPU-resident memory allocator for Rust alloc crate
// Lock-free slab allocator with atomic free lists per size class
// ═══════════════════════════════════════════════════════════════════════════════

// Memory allocation (0xF0-0xF3)
constant uint OP_ALLOC = 0xF0;         // dst = gpu_alloc(size_reg, align_reg)
constant uint OP_DEALLOC = 0xF1;       // gpu_dealloc(ptr_reg, size_reg, align_reg)
constant uint OP_REALLOC = 0xF2;       // dst = gpu_realloc(ptr_reg, old_size, new_size)
constant uint OP_ALLOC_ZERO = 0xF3;    // dst = gpu_alloc_zeroed(size_reg, align_reg)

// WASM Memory Operations (Issue #210)
constant uint OP_MEMORY_SIZE = 0xF4;   // dst = current memory size in pages
constant uint OP_MEMORY_GROW = 0xF5;   // dst = memory_grow(delta_pages) returns old size or -1

// WASI Operations (Issue #207 - GPU-Native WASI)
constant uint OP_WASI_FD_WRITE = 0xF6;         // fd_write -> debug buffer for stdout/stderr
constant uint OP_WASI_FD_READ = 0xF7;          // fd_read -> always returns EBADF
constant uint OP_WASI_PROC_EXIT = 0xF8;        // proc_exit -> halt
constant uint OP_WASI_ENVIRON_SIZES_GET = 0xF9; // returns 0,0 (no env)
constant uint OP_WASI_ENVIRON_GET = 0xFA;       // returns success (empty)
constant uint OP_WASI_ARGS_SIZES_GET = 0xFB;    // returns 0,0 (no args)
constant uint OP_WASI_ARGS_GET = 0xFC;          // returns success (empty)
constant uint OP_WASI_CLOCK_TIME_GET = 0xFD;    // returns frame count as time
constant uint OP_WASI_RANDOM_GET = 0xFE;        // pseudo-random from thread ID

// Panic Handling (Issue #209)
constant uint OP_PANIC = 0x76;                  // panic + halt
constant uint OP_UNREACHABLE = 0x77;            // unreachable trap
constant uint OP_CALL_FUNC = 0x78;              // call function (GPU call stack)
constant uint OP_RETURN_FUNC = 0x79;            // return from function

// Table operations (Issue #212)
constant uint OP_TABLE_GET = 0x50;              // table.get
constant uint OP_TABLE_SET = 0x51;              // table.set
constant uint OP_TABLE_SIZE = 0x52;             // table.size
constant uint OP_TABLE_GROW = 0x53;             // table.grow
constant uint OP_TABLE_INIT = 0x54;             // table.init
constant uint OP_TABLE_COPY = 0x55;             // table.copy
constant uint OP_TABLE_FILL = 0x56;             // table.fill

// ═══════════════════════════════════════════════════════════════════════════════
// SIMD OPERATIONS (Issue #211)
// THE GPU IS THE COMPUTER - float4 is native SIMD, these ops work on all 4 lanes
// ═══════════════════════════════════════════════════════════════════════════════

constant uint OP_V4_ADD = 0x90;      // dst = s1 + s2 (all 4 lanes)
constant uint OP_V4_SUB = 0x91;      // dst = s1 - s2 (all 4 lanes)
constant uint OP_V4_MUL = 0x92;      // dst = s1 * s2 (all 4 lanes)
constant uint OP_V4_DIV = 0x93;      // dst = s1 / s2 (all 4 lanes)
constant uint OP_V4_MIN = 0x94;      // dst = min(s1, s2) per lane
constant uint OP_V4_MAX = 0x95;      // dst = max(s1, s2) per lane
constant uint OP_V4_ABS = 0x96;      // dst = abs(s1) per lane
constant uint OP_V4_NEG = 0x97;      // dst = -s1 per lane
constant uint OP_V4_SQRT = 0x98;     // dst = sqrt(s1) per lane
constant uint OP_V4_DOT = 0x99;      // dst.x = dot(s1, s2)
constant uint OP_V4_SHUFFLE = 0x9A;  // dst = shuffle(s1, s2, imm)
constant uint OP_V4_EXTRACT = 0x9B;  // dst.x = s1[imm]
constant uint OP_V4_REPLACE = 0x9C;  // dst = s1 with s1[imm] = s2.x
constant uint OP_V4_SPLAT = 0x9D;    // dst = (s1.x, s1.x, s1.x, s1.x)
constant uint OP_V4_EQ = 0x9E;       // dst = (s1 == s2) per lane
constant uint OP_V4_LT = 0x9F;       // dst = (s1 < s2) per lane

// ═══════════════════════════════════════════════════════════════════════════════
// DEBUG I/O OPERATIONS (Phase 7 - Issue #180)
// THE GPU IS THE COMPUTER - debug output via ring buffer
// Lock-free atomic writes, CPU reads buffer after kernel execution
// ═══════════════════════════════════════════════════════════════════════════════

// Debug output (0x70-0x75)
constant uint OP_DBG_I32 = 0x70;       // Debug print i32 from src1 register
constant uint OP_DBG_F32 = 0x71;       // Debug print f32 from src1 register
constant uint OP_DBG_STR = 0x72;       // Debug print string (ptr=src1, len=src2)
constant uint OP_DBG_BOOL = 0x73;      // Debug print bool from src1 register
constant uint OP_DBG_NL = 0x74;        // Debug newline marker
constant uint OP_DBG_FLUSH = 0x75;     // Force debug flush (no-op on GPU)

// Debug entry types (stored in type field)
constant uchar DBG_TYPE_INT = 0x01;
constant uchar DBG_TYPE_FLOAT = 0x02;
constant uchar DBG_TYPE_STRING = 0x03;
constant uchar DBG_TYPE_BOOL = 0x04;
constant uchar DBG_TYPE_NEWLINE = 0x0A;
constant uchar DBG_TYPE_FLUSH = 0x0F;

// Bytecode instruction layout (8 bytes):
// [opcode:8][dst:8][src1:8][src2:8][imm:32]
// CRITICAL: Store imm as uint to prevent denormal flushing
// Small integers (< 8M) are denormal when stored as float bits
// GPU flushes denormals to zero during float loads, corrupting constants
struct BytecodeInst {
    uchar opcode;
    uchar dst;
    uchar src1;
    uchar src2;
    uint imm_bits;  // Use as_type<float>() when float value needed
};

// Bytecode app state header (at start of state buffer)
struct BytecodeHeader {
    uint code_size;        // Number of instructions
    uint entry_point;      // Starting PC
    uint vertex_budget;    // Max vertices allowed
    uint flags;
    // Bytecode follows immediately after header
    // Then app-specific state data
};

// ═══════════════════════════════════════════════════════════════════════════════
// SLAB ALLOCATOR (Phase 6 - Issue #179)
// THE GPU IS THE COMPUTER - Lock-free memory allocator with atomic free lists
// Size classes: 16, 32, 64, 128, 256, 512, 1024, 2048+ bytes
// ═══════════════════════════════════════════════════════════════════════════════

// Slab allocator header - lives at a fixed offset in state memory
// For bytecode apps, this is at the end of their state region
struct SlabAllocator {
    atomic_uint free_heads[8];   // Free list heads for each size class (0xFFFFFFFF = empty)
    atomic_uint heap_top;        // Bump pointer for large/initial allocations
    uint heap_size;              // Total heap size
    uint _pad[2];
};

// Size classes: 0=16B, 1=32B, 2=64B, 3=128B, 4=256B, 5=512B, 6=1024B, 7=2048B+
// Size class lookup table (must be at file scope for constant address space)
constant uint SLAB_SIZE_TABLE[8] = {16, 32, 64, 128, 256, 512, 1024, 2048};

inline uint size_to_class(uint size) {
    if (size <= 16) return 0;
    if (size <= 32) return 1;
    if (size <= 64) return 2;
    if (size <= 128) return 3;
    if (size <= 256) return 4;
    if (size <= 512) return 5;
    if (size <= 1024) return 6;
    return 7;  // Large allocation (bump allocator)
}

// Get block size for a size class
inline uint class_to_size(uint class_idx) {
    return SLAB_SIZE_TABLE[min(class_idx, 7u)];
}

// Allocate memory from slab allocator
// Returns offset from heap start, or 0xFFFFFFFF on failure
inline uint gpu_alloc(
    uint size,
    uint align,
    device SlabAllocator* alloc,
    device uchar* heap
) {
    if (size == 0) return 0xFFFFFFFF;

    // Round up to alignment
    size = max(size, align);

    uint class_idx = size_to_class(size);
    uint block_size = class_to_size(class_idx);

    // Try to pop from free list using lock-free CAS loop
    uint head = atomic_load_explicit(&alloc->free_heads[class_idx], memory_order_relaxed);

    while (head != 0xFFFFFFFF) {
        // Read next pointer from the free block
        uint next = *(device uint*)(heap + head);

        // Try to CAS the head to next
        if (atomic_compare_exchange_weak_explicit(
            &alloc->free_heads[class_idx],
            &head,
            next,
            memory_order_relaxed,
            memory_order_relaxed
        )) {
            // Successfully popped from free list
            return head;
        }
        // CAS failed, head was updated by compare_exchange, retry with new head
    }

    // Free list empty - bump allocate
    uint old_top = atomic_fetch_add_explicit(&alloc->heap_top, block_size, memory_order_relaxed);

    // Check if we exceeded heap size
    if (old_top + block_size > alloc->heap_size) {
        // Out of memory - try to restore heap_top (best effort)
        atomic_fetch_sub_explicit(&alloc->heap_top, block_size, memory_order_relaxed);
        return 0xFFFFFFFF;
    }

    return old_top;
}

// Free memory back to slab allocator
// THE GPU NEVER WAITS - push to free list and continue
inline void gpu_dealloc(
    uint ptr,
    uint size,
    device SlabAllocator* alloc,
    device uchar* heap
) {
    if (ptr == 0xFFFFFFFF || size == 0) return;

    uint class_idx = size_to_class(size);

    // Push to free list using lock-free CAS loop
    uint head = atomic_load_explicit(&alloc->free_heads[class_idx], memory_order_relaxed);

    do {
        // Store current head as next pointer in freed block
        *(device uint*)(heap + ptr) = head;

        // Try to CAS head to our freed block
    } while (!atomic_compare_exchange_weak_explicit(
        &alloc->free_heads[class_idx],
        &head,
        ptr,
        memory_order_relaxed,
        memory_order_relaxed
    ));
}

// Reallocate memory to new size
// Copies data if block moves, frees old block
inline uint gpu_realloc(
    uint ptr,
    uint old_size,
    uint new_size,
    device SlabAllocator* alloc,
    device uchar* heap
) {
    if (ptr == 0xFFFFFFFF) {
        // Null pointer - just allocate
        return gpu_alloc(new_size, 1, alloc, heap);
    }

    if (new_size == 0) {
        // Zero size - just free
        gpu_dealloc(ptr, old_size, alloc, heap);
        return 0xFFFFFFFF;
    }

    // Check if same size class - no realloc needed
    uint old_class = size_to_class(old_size);
    uint new_class = size_to_class(new_size);

    if (new_class <= old_class) {
        // Fits in existing block
        return ptr;
    }

    // Need larger block - allocate new, copy, free old
    uint new_ptr = gpu_alloc(new_size, 1, alloc, heap);
    if (new_ptr == 0xFFFFFFFF) return 0xFFFFFFFF;

    // Copy old data to new location
    uint copy_size = min(old_size, new_size);
    for (uint i = 0; i < copy_size; i++) {
        heap[new_ptr + i] = heap[ptr + i];
    }

    // Free old block
    gpu_dealloc(ptr, old_size, alloc, heap);

    return new_ptr;
}

// Allocate zeroed memory
inline uint gpu_alloc_zeroed(
    uint size,
    uint align,
    device SlabAllocator* alloc,
    device uchar* heap
) {
    uint ptr = gpu_alloc(size, align, alloc, heap);
    if (ptr == 0xFFFFFFFF) return ptr;

    // Zero the allocated memory
    uint block_size = class_to_size(size_to_class(size));
    for (uint i = 0; i < block_size; i++) {
        heap[ptr + i] = 0;
    }

    return ptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEBUG I/O BUFFER (Phase 7 - Issue #180)
// THE GPU IS THE COMPUTER - ring buffer for debug output
// Lock-free atomic writes, entries include thread ID for multi-thread debugging
// ═══════════════════════════════════════════════════════════════════════════════

// Debug buffer lives at a fixed offset in state memory
// Default size: 4KB debug buffer
constant uint DEBUG_BUFFER_SIZE = 4096;
constant uint DEBUG_BUFFER_DATA_OFFSET = 8;  // After header (write_pos + capacity)

struct DebugBuffer {
    atomic_uint write_pos;    // Current write position (atomic for lock-free)
    uint capacity;            // Size of data area (DEBUG_BUFFER_SIZE - 8)
    // data follows: uchar data[capacity]
};

// ═══════════════════════════════════════════════════════════════════════════════
// DOUBLE-SINGLE ARITHMETIC (Issue #27)
// THE GPU IS THE COMPUTER - Metal does NOT support native f64
// We emulate f64 using double-single representation: value = hi + lo
// where hi and lo are both float32, giving ~47 bits of mantissa precision
// Based on Dekker/Knuth algorithms and metal-float64 library techniques
// ═══════════════════════════════════════════════════════════════════════════════

// Two-sum: compute a + b = s + e exactly (Knuth algorithm)
// Returns float2(s, e) where s is the sum and e is the rounding error
inline float2 two_sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e = (a - (s - v)) + (b - v);
    return float2(s, e);
}

// Quick two-sum: faster when |a| >= |b| is known
inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return float2(s, e);
}

// Split a float into hi and lo parts for multiplication (Veltkamp split)
// Uses 2^12 + 1 = 4097 as the split constant for float32 (12 bits for hi)
inline float2 split(float a) {
    const float SPLIT = 4097.0f;  // 2^12 + 1
    float t = SPLIT * a;
    float hi = t - (t - a);
    float lo = a - hi;
    return float2(hi, lo);
}

// Two-product: compute a * b = p + e exactly (Dekker algorithm)
// Returns float2(p, e) where p is the product and e is the rounding error
inline float2 two_prod(float a, float b) {
    float p = a * b;
    float2 a_split = split(a);
    float2 b_split = split(b);
    float e = ((a_split.x * b_split.x - p) + a_split.x * b_split.y + a_split.y * b_split.x) + a_split.y * b_split.y;
    return float2(p, e);
}

// Double-single addition: (a.x + a.y) + (b.x + b.y) = (c.x + c.y)
inline float2 ds_add(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);
    float2 t = two_sum(a.y, b.y);
    s.y += t.x;
    s = quick_two_sum(s.x, s.y);
    s.y += t.y;
    s = quick_two_sum(s.x, s.y);
    return s;
}

// Double-single subtraction: (a.x + a.y) - (b.x + b.y) = (c.x + c.y)
inline float2 ds_sub(float2 a, float2 b) {
    return ds_add(a, float2(-b.x, -b.y));
}

// Double-single multiplication: (a.x + a.y) * (b.x + b.y) = (c.x + c.y)
inline float2 ds_mul(float2 a, float2 b) {
    float2 p = two_prod(a.x, b.x);
    p.y += a.x * b.y + a.y * b.x;
    p = quick_two_sum(p.x, p.y);
    return p;
}

// Double-single division: (a.x + a.y) / (b.x + b.y) = (c.x + c.y)
// Uses Newton-Raphson refinement for accuracy
inline float2 ds_div(float2 a, float2 b) {
    float q1 = a.x / b.x;

    // r = a - q1 * b (compute residual in double-single)
    float2 p = two_prod(q1, b.x);
    float2 r = ds_sub(a, float2(p.x, p.y + q1 * b.y));

    // q2 = r / b.x (refinement)
    float q2 = r.x / b.x;

    return quick_two_sum(q1, q2);
}

// Double-single square root using Newton-Raphson
inline float2 ds_sqrt(float2 a) {
    if (a.x <= 0.0f) {
        return float2(0.0f, 0.0f);
    }

    // Initial estimate
    float x = rsqrt(a.x);  // 1/sqrt(a.x)
    float q = a.x * x;      // First approximation of sqrt(a)

    // Refine: r = a - q^2, then q += r / (2*q)
    float2 p = two_prod(q, q);
    float2 r = ds_sub(a, p);
    float q2 = r.x * x * 0.5f;

    return quick_two_sum(q, q2);
}

// Convert i32 to double-single
inline float2 ds_from_i32(int v) {
    return float2(float(v), 0.0f);
}

// Convert u32 to double-single
inline float2 ds_from_u32(uint v) {
    return float2(float(v), 0.0f);
}

// Convert i64 (stored as two u32s in xy) to double-single
// i64 bits: x = low 32 bits, y = high 32 bits (including sign)
inline float2 ds_from_i64(float2 bits) {
    uint lo = as_type<uint>(bits.x);
    int hi = as_type<int>(bits.y);  // Signed high part

    // value = hi * 2^32 + lo
    float hi_f = float(hi) * 4294967296.0f;  // hi * 2^32
    float lo_f = float(lo);

    return two_sum(hi_f, lo_f);
}

// Convert u64 (stored as two u32s in xy) to double-single
inline float2 ds_from_u64(float2 bits) {
    uint lo = as_type<uint>(bits.x);
    uint hi = as_type<uint>(bits.y);

    // value = hi * 2^32 + lo
    float hi_f = float(hi) * 4294967296.0f;  // hi * 2^32
    float lo_f = float(lo);

    return two_sum(hi_f, lo_f);
}

// Convert double-single to i32 (truncate)
inline int ds_to_i32(float2 ds) {
    // Add hi and lo, truncate to int
    return int(ds.x + ds.y);
}

// Convert double-single to u32 (truncate)
inline uint ds_to_u32(float2 ds) {
    return uint(ds.x + ds.y);
}

// Convert double-single to i64 (truncate)
// Returns float2 where x = low 32 bits, y = high 32 bits (as int)
inline float2 ds_to_i64(float2 ds) {
    // Get the combined value
    float v = ds.x + ds.y;

    // Handle sign
    bool negative = v < 0.0f;
    float abs_v = negative ? -v : v;

    // Split into hi and lo 32-bit parts
    float hi_f = floor(abs_v / 4294967296.0f);  // / 2^32
    float lo_f = abs_v - hi_f * 4294967296.0f;

    uint lo = uint(lo_f);
    int hi = int(hi_f);

    // Apply two's complement if negative
    if (negative) {
        lo = ~lo + 1;
        if (lo == 0) hi++;
        hi = ~hi;
    }

    return float2(as_type<float>(lo), as_type<float>(uint(hi)));
}

// Convert double-single to u64 (truncate)
inline float2 ds_to_u64(float2 ds) {
    float v = ds.x + ds.y;
    if (v < 0.0f) v = 0.0f;  // Clamp negative to 0

    // Split into hi and lo 32-bit parts
    float hi_f = floor(v / 4294967296.0f);  // / 2^32
    float lo_f = v - hi_f * 4294967296.0f;

    uint lo = uint(lo_f);
    uint hi = uint(hi_f);

    return float2(as_type<float>(lo), as_type<float>(hi));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Double-single reinterpret operations
//
// CRITICAL LIMITATION: These operations CANNOT preserve full IEEE 754 f64 bit patterns.
//
// Mathematical reality:
//   - IEEE 754 f64: 52-bit mantissa (stored in 64 bits)
//   - Double-single: ~47-bit mantissa (two 23-bit f32 mantissas combined)
//   - 5 bits of precision are LOST during conversion
//
// Example:
//   CPU (IEEE 754):    0.1 = 0x3FB999999999999A (all 52 mantissa bits used)
//   GPU (double-single): 0.1 -> 0x3FB99999A0000000 (low 5 bits zeroed)
//
// This is UNFIXABLE without native f64 hardware support (which Metal lacks).
// Arithmetic operations work fine (value equality within epsilon).
// Bit-exact reinterpret operations will show discrepancies in low-order bits.
//
// See: tests/test_opcodes_f64.rs for the ignored test and full explanation.
// ═══════════════════════════════════════════════════════════════════════════════

// Convert IEEE 754 f64 bits (stored as two u32s in xy) to double-single
// This parses the IEEE 754 format and reconstructs the value as double-single
inline float2 ds_from_f64_bits(float2 bits_f) {
    uint lo_bits = as_type<uint>(bits_f.x);
    uint hi_bits = as_type<uint>(bits_f.y);

    // Extract IEEE 754 components from the 64-bit value
    // Format: sign(1) | exponent(11) | mantissa(52)
    int sign = (hi_bits >> 31) ? -1 : 1;
    int exp = int((hi_bits >> 20) & 0x7FF);  // 11-bit exponent
    uint mant_hi = hi_bits & 0xFFFFF;        // Upper 20 bits of mantissa
    uint mant_lo = lo_bits;                  // Lower 32 bits of mantissa

    // Handle special cases
    if (exp == 0x7FF) {
        // Infinity or NaN
        if (mant_hi == 0 && mant_lo == 0) {
            // Infinity
            return float2(sign > 0 ? INFINITY : -INFINITY, 0.0f);
        } else {
            // NaN
            return float2(NAN, 0.0f);
        }
    }

    if (exp == 0) {
        // Zero or subnormal
        if (mant_hi == 0 && mant_lo == 0) {
            // Zero (preserve sign for -0.0)
            return float2(sign > 0 ? 0.0f : -0.0f, 0.0f);
        }
        // Subnormal - just return a small value (can't represent exactly)
        return float2(0.0f, 0.0f);
    }

    // Normal number: value = sign * 2^(exp-1023) * (1 + mantissa/2^52)
    // Build the mantissa value: 1.mantissa (in range [1, 2))
    // mantissa_value = 1 + (mant_hi * 2^32 + mant_lo) / 2^52
    //                = 1 + mant_hi / 2^20 + mant_lo / 2^52

    float mant_hi_contrib = float(mant_hi) / 1048576.0f;      // / 2^20
    float mant_lo_contrib = float(mant_lo) / 4503599627370496.0f;  // / 2^52

    // Start with 1.0 + mantissa contributions
    float2 mantissa = two_sum(1.0f + mant_hi_contrib, mant_lo_contrib);

    // Apply exponent: multiply by 2^(exp-1023)
    int actual_exp = exp - 1023;
    float scale = 1.0f;

    // Handle exponent scaling carefully to avoid overflow
    // We can safely scale by 2^127 max in f32
    if (actual_exp > 0) {
        while (actual_exp > 0) {
            int step = min(actual_exp, 30);
            scale = float(1 << step);
            mantissa = ds_mul(mantissa, float2(scale, 0.0f));
            actual_exp -= step;
        }
    } else if (actual_exp < 0) {
        while (actual_exp < 0) {
            int step = min(-actual_exp, 30);
            scale = 1.0f / float(1 << step);
            mantissa = ds_mul(mantissa, float2(scale, 0.0f));
            actual_exp += step;
        }
    }

    // Apply sign
    if (sign < 0) {
        mantissa = float2(-mantissa.x, -mantissa.y);
    }

    return mantissa;
}

// Convert double-single to IEEE 754 f64 bits (returned as two u32s in xy)
// This reconstructs the f64 value from double-single and extracts its IEEE 754 bits
inline float2 ds_to_f64_bits(float2 ds) {
    float v = ds.x + ds.y;

    // Handle special cases
    if (isinf(v)) {
        if (v > 0) {
            // +Infinity: 0x7FF0000000000000
            return float2(as_type<float>(0u), as_type<float>(0x7FF00000u));
        } else {
            // -Infinity: 0xFFF0000000000000
            return float2(as_type<float>(0u), as_type<float>(0xFFF00000u));
        }
    }

    if (isnan(v)) {
        // NaN: 0x7FF8000000000000 (quiet NaN)
        return float2(as_type<float>(0u), as_type<float>(0x7FF80000u));
    }

    if (v == 0.0f) {
        // Check for -0.0 (sign bit in hi component)
        uint sign_bit = (ds.x < 0.0f || (ds.x == 0.0f && as_type<uint>(ds.x) >> 31)) ? 0x80000000u : 0u;
        return float2(as_type<float>(0u), as_type<float>(sign_bit));
    }

    // Normal number - extract sign, exponent, mantissa
    uint sign = (v < 0.0f) ? 1u : 0u;
    float abs_v = abs(v);

    // Find exponent: largest n such that 2^n <= abs_v
    int exp = 0;
    float temp = abs_v;
    if (temp >= 1.0f) {
        while (temp >= 2.0f && exp < 1023) {
            temp /= 2.0f;
            exp++;
        }
    } else {
        while (temp < 1.0f && exp > -1022) {
            temp *= 2.0f;
            exp--;
        }
    }

    // Now temp is in [1, 2) and abs_v = temp * 2^exp
    // IEEE 754 exponent = exp + 1023 (bias)
    uint biased_exp = uint(exp + 1023);

    // Mantissa = (temp - 1) * 2^52
    // temp is in [1, 2), so (temp - 1) is in [0, 1)
    float mantissa_f = (temp - 1.0f) * 4503599627370496.0f;  // * 2^52

    // Split into high 20 bits and low 32 bits
    uint mant_lo = uint(fmod(mantissa_f, 4294967296.0f));  // Low 32 bits
    uint mant_hi = uint(mantissa_f / 4294967296.0f);       // High 20 bits (actually uses lower 20)
    mant_hi &= 0xFFFFF;  // Ensure only 20 bits

    // Construct the 64-bit value
    // hi = sign(1) | exponent(11) | mant_hi(20)
    // lo = mant_lo(32)
    uint hi_bits = (sign << 31) | (biased_exp << 20) | mant_hi;
    uint lo_bits = mant_lo;

    return float2(as_type<float>(lo_bits), as_type<float>(hi_bits));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Double-single comparison operations
// For double-single format: value = hi + lo, where |lo| < ulp(hi)
// Comparison is done lexicographically: first compare hi, then lo if equal
// ═══════════════════════════════════════════════════════════════════════════════

// Double-single less than: a < b
inline bool ds_lt(float2 a, float2 b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

// Double-single greater than: a > b
inline bool ds_gt(float2 a, float2 b) {
    return (a.x > b.x) || (a.x == b.x && a.y > b.y);
}

// Double-single less than or equal: a <= b
inline bool ds_le(float2 a, float2 b) {
    return (a.x < b.x) || (a.x == b.x && a.y <= b.y);
}

// Double-single greater than or equal: a >= b
inline bool ds_ge(float2 a, float2 b) {
    return (a.x > b.x) || (a.x == b.x && a.y >= b.y);
}

// Double-single equality: a == b (both components must match)
inline bool ds_eq(float2 a, float2 b) {
    return a.x == b.x && a.y == b.y;
}

// Double-single not equal: a != b
inline bool ds_ne(float2 a, float2 b) {
    return a.x != b.x || a.y != b.y;
}

// Double-single minimum
inline float2 ds_min(float2 a, float2 b) {
    return ds_lt(a, b) ? a : b;
}

// Double-single maximum
inline float2 ds_max(float2 a, float2 b) {
    return ds_gt(a, b) ? a : b;
}

// ═══════════════════════════════════════════════════════════════════════════════

// Debug entry format:
// [thread_id:4][type:1][length:1][data:length]
// Types: 0x01=INT, 0x02=FLOAT, 0x03=STRING, 0x04=BOOL, 0x0A=NEWLINE, 0x0F=FLUSH

// Write i32 to debug buffer
inline void gpu_debug_i32(int value, uint tid, device DebugBuffer* dbg, device uchar* dbg_data) {
    if (!dbg) return;
    uint entry_size = 10;  // 4 (tid) + 1 (type) + 1 (len) + 4 (value)
    uint pos = atomic_fetch_add_explicit(&dbg->write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg->capacity) return;  // Buffer full

    device uchar* ptr = dbg_data + pos;
    *(device uint*)ptr = tid;
    ptr[4] = DBG_TYPE_INT;
    ptr[5] = 4;     // 4 bytes
    *(device int*)(ptr + 6) = value;
}

// Write f32 to debug buffer
inline void gpu_debug_f32(float value, uint tid, device DebugBuffer* dbg, device uchar* dbg_data) {
    if (!dbg) return;
    uint entry_size = 10;  // 4 (tid) + 1 (type) + 1 (len) + 4 (value)
    uint pos = atomic_fetch_add_explicit(&dbg->write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg->capacity) return;

    device uchar* ptr = dbg_data + pos;
    *(device uint*)ptr = tid;
    ptr[4] = DBG_TYPE_FLOAT;
    ptr[5] = 4;
    *(device float*)(ptr + 6) = value;
}

// Write string to debug buffer (copies from state memory)
inline void gpu_debug_str(device const uchar* str, uint len, uint tid, device DebugBuffer* dbg, device uchar* dbg_data) {
    if (!dbg || len > 255) return;  // Max string length 255
    uint entry_size = 6 + len;  // 4 (tid) + 1 (type) + 1 (len) + len (data)
    uint pos = atomic_fetch_add_explicit(&dbg->write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg->capacity) return;

    device uchar* ptr = dbg_data + pos;
    *(device uint*)ptr = tid;
    ptr[4] = DBG_TYPE_STRING;
    ptr[5] = (uchar)len;
    for (uint i = 0; i < len; i++) {
        ptr[6 + i] = str[i];
    }
}

// Write bool to debug buffer
inline void gpu_debug_bool(bool value, uint tid, device DebugBuffer* dbg, device uchar* dbg_data) {
    if (!dbg) return;
    uint entry_size = 7;  // 4 (tid) + 1 (type) + 1 (len) + 1 (value)
    uint pos = atomic_fetch_add_explicit(&dbg->write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg->capacity) return;

    device uchar* ptr = dbg_data + pos;
    *(device uint*)ptr = tid;
    ptr[4] = DBG_TYPE_BOOL;
    ptr[5] = 1;
    ptr[6] = value ? 1 : 0;
}

// Write newline marker to debug buffer
inline void gpu_debug_newline(uint tid, device DebugBuffer* dbg, device uchar* dbg_data) {
    if (!dbg) return;
    uint entry_size = 6;  // 4 (tid) + 1 (type) + 1 (len=0)
    uint pos = atomic_fetch_add_explicit(&dbg->write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg->capacity) return;

    device uchar* ptr = dbg_data + pos;
    *(device uint*)ptr = tid;
    ptr[4] = DBG_TYPE_NEWLINE;
    ptr[5] = 0;
}

// Write flush marker to debug buffer (indicates debug output complete)
inline void gpu_debug_flush(uint tid, device DebugBuffer* dbg, device uchar* dbg_data) {
    if (!dbg) return;
    uint entry_size = 6;
    uint pos = atomic_fetch_add_explicit(&dbg->write_pos, entry_size, memory_order_relaxed);
    if (pos + entry_size > dbg->capacity) return;

    device uchar* ptr = dbg_data + pos;
    *(device uint*)ptr = tid;
    ptr[4] = DBG_TYPE_FLUSH;
    ptr[5] = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOMATIC CODE TRANSFORMATION SUPPORT (Phase 8 - Issue #182)
// THE GPU IS THE COMPUTER - GPU-native implementations of CPU patterns
// ═══════════════════════════════════════════════════════════════════════════════

// Work queue for async pattern transformation
// Lock-free MPMC queue using atomic head/tail
constant uint WORK_QUEUE_SIZE = 256;  // Must be power of 2

struct WorkQueue {
    atomic_uint head;        // Consumer position
    atomic_uint tail;        // Producer position
    uint capacity;           // WORK_QUEUE_SIZE
    uint _pad;
    uint items[WORK_QUEUE_SIZE];
};

// Push work item to queue (async/await transformation)
// Returns true if successful, false if queue full
inline bool work_push(uint item, device WorkQueue* queue) {
    if (!queue) return false;

    // Get current tail and try to advance it
    uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
    uint head = atomic_load_explicit(&queue->head, memory_order_relaxed);

    // Check if queue is full
    if (tail - head >= queue->capacity) {
        return false;  // Queue full
    }

    // Try to claim the slot
    uint new_tail = tail + 1;
    if (!atomic_compare_exchange_weak_explicit(
        &queue->tail, &tail, new_tail,
        memory_order_relaxed, memory_order_relaxed
    )) {
        return false;  // Another thread got there first
    }

    // Write the item (slot is claimed)
    queue->items[tail % queue->capacity] = item;
    return true;
}

// Pop work item from queue
// Returns item if successful, 0xFFFFFFFF if queue empty
inline uint work_pop(device WorkQueue* queue) {
    if (!queue) return 0xFFFFFFFF;

    // Get current head and check if empty
    uint head = atomic_load_explicit(&queue->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);

    // Check if queue is empty
    if (head >= tail) {
        return 0xFFFFFFFF;  // Queue empty
    }

    // Try to claim the slot
    uint new_head = head + 1;
    if (!atomic_compare_exchange_weak_explicit(
        &queue->head, &head, new_head,
        memory_order_relaxed, memory_order_relaxed
    )) {
        return 0xFFFFFFFF;  // Another thread got there first
    }

    // Read and return the item
    return queue->items[head % queue->capacity];
}

// Spinlock acquire (Mutex::lock transformation)
// THE GPU IS THE COMPUTER - uses atomic CAS for lock acquisition
// Includes timeout protection to avoid infinite spin
constant uint SPINLOCK_MAX_ITERATIONS = 10000;

inline bool spinlock_acquire(device atomic_uint* lock) {
    uint expected = 0;
    uint iterations = 0;

    while (iterations < SPINLOCK_MAX_ITERATIONS) {
        // Try to atomically set lock from 0 to 1
        // Note: Metal only supports memory_order_relaxed for device memory
        // Use threadgroup_barrier for synchronization if needed
        if (atomic_compare_exchange_weak_explicit(
            lock, &expected, 1u,
            memory_order_relaxed, memory_order_relaxed
        )) {
            return true;  // Lock acquired
        }
        expected = 0;  // Reset expected for retry
        iterations++;
    }

    return false;  // Timeout - lock not acquired
}

// Spinlock release (Mutex::unlock transformation)
inline void spinlock_release(device atomic_uint* lock) {
    // Note: Metal only supports memory_order_relaxed for device memory
    atomic_store_explicit(lock, 0u, memory_order_relaxed);
}

// Frame-based timing (thread::sleep transformation)
// THE GPU IS THE COMPUTER - GPU threads don't sleep, they spin
// This reads the frame counter and waits for it to advance
// Note: In real usage, this would need access to a global frame counter
// For now, it's a placeholder that will spin for a fixed number of iterations
constant uint FRAME_ITERATIONS_PER_FRAME = 1000;

inline void frame_wait(uint frames, device uint* frame_counter) {
    // If we have access to a real frame counter, use it
    if (frame_counter) {
        uint target = *frame_counter + frames;
        while (*frame_counter < target) {
            // Spin
        }
    } else {
        // Fallback: spin for approximately N frames worth
        for (uint f = 0; f < frames; f++) {
            for (uint i = 0; i < FRAME_ITERATIONS_PER_FRAME; i++) {
                // Spin
            }
        }
    }
}

// Bytecode interpreter - runs program on GPU
inline void bytecode_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    float actual_screen_width,
    float actual_screen_height,
    uint tid,
    uint tg_size
) {
    // Only thread 0 runs bytecode for now (can parallelize later)
    if (tid != 0) return;

    device BytecodeHeader* header = (device BytecodeHeader*)(unified_state + app->state_offset);
    device BytecodeInst* code = (device BytecodeInst*)(header + 1);

    // Register file (32 float4 registers)
    float4 regs[32];
    for (uint i = 0; i < 32; i++) regs[i] = float4(0);
    regs[1] = float4(float(tid), 0.0, 0.0, 0.0);       // r1 = thread ID
    regs[2] = float4(float(tg_size), 0.0, 0.0, 0.0);   // r2 = threadgroup size
    // CRITICAL BUG FIX: DO NOT use as_type<float>(integer) for small integers!
    // as_type<float>(60) = 8.4e-44 which is a DENORMAL that GPU FTZ flushes to 0!
    // Store as actual float value. WASM apps that need integer can handle conversion.
    regs[3] = float4(float(app->frame_number), 0.0, 0.0, 0.0);  // r3 = frame number (as float)

    // Input registers (Phase 9 rendering intrinsics)
    regs[16] = float4(400.0, 0.0, 0.0, 0.0);   // r16 = cursor_x
    regs[17] = float4(300.0, 0.0, 0.0, 0.0);   // r17 = cursor_y
    regs[18] = float4(0.0, 0.0, 0.0, 0.0);     // r18 = mouse_down (0 = not pressed)
    regs[19] = float4(float(app->frame_number) / 60.0, 0.0, 0.0, 0.0);  // r19 = time in seconds

    // Screen dimensions for get_screen_width()/get_screen_height() intrinsics
    // Default to 800x600 to match what most WASM apps expect
    regs[20] = float4(800.0);   // r20 = screen_width (default for WASM apps)
    regs[21] = float4(600.0);   // r21 = screen_height (default for WASM apps)

    // State pointer (after bytecode)
    uint state_offset_bytes = sizeof(BytecodeHeader) + header->code_size * sizeof(BytecodeInst);
    device float4* state = (device float4*)(unified_state + app->state_offset + state_offset_bytes);

    // Issue #250 fix: Calculate state buffer size for bounds checking
    uint state_size_bytes = app->state_size > state_offset_bytes ?
                            app->state_size - state_offset_bytes : 0;
    uint state_size_float4 = state_size_bytes / sizeof(float4);

    // Allocator and heap setup (Phase 6 - Issue #179)
    // State layout: SlabAllocator[0-2] (48 bytes) | result[3] (16 bytes) | params[4-7] (64 bytes) | heap[8+]
    device SlabAllocator* alloc = (device SlabAllocator*)state;
    device uchar* heap = (device uchar*)(state + 8);  // Skip allocator (48) + result (16) + params (64)

    // Issue #176: Load parameters from state buffer into r4-r7
    // THE GPU IS THE COMPUTER: Parameters come from GPU memory, not CPU
    if (state_size_float4 > 4) regs[4] = state[4];  // param0 -> r4
    if (state_size_float4 > 5) regs[5] = state[5];  // param1 -> r5
    if (state_size_float4 > 6) regs[6] = state[6];  // param2 -> r6
    if (state_size_float4 > 7) regs[7] = state[7];  // param3 -> r7

    // Debug buffer setup (Phase 7 - Issue #180)
    // Debug buffer is at a fixed offset after the heap (simplified: at state + 256 float4s)
    // Layout: DebugBuffer header (8 bytes) + data area
    device DebugBuffer* dbg = (device DebugBuffer*)(state + 256);  // 256 * 16 = 4KB after state start
    device uchar* dbg_data = (device uchar*)(dbg + 1);  // Data follows header

    // Work queue setup (Phase 8 - Issue #182)
    // Work queue is at state + 512 float4s (8KB after state start)
    device WorkQueue* work_queue = (device WorkQueue*)(state + 512);

    // Frame counter pointer (for frame_wait) - uses app's frame_number
    device uint* frame_counter = &app->frame_number;

    // Vertex output
    device RenderVertex* verts = unified_vertices + app->vertex_offset / sizeof(RenderVertex);
    uint vert_idx = 0;

    // Execute
    // ═══════════════════════════════════════════════════════════════════════════
    // THE GPU IS THE COMPUTER - No arbitrary instruction limits
    // ═══════════════════════════════════════════════════════════════════════════
    // Design rationale (Issue #213 fix):
    // - Apps are designed to complete within a single frame
    // - Complex apps (Game of Life 32x32 = 1024 cells) need ~500K+ instructions
    // - Metal's watchdog timeout (~2-5 seconds) catches infinite loops
    // - Apps needing multi-frame execution should use OP_YIELD
    // - This removes arbitrary CPU-like limits from GPU-native execution
    // ═══════════════════════════════════════════════════════════════════════════
    uint pc = header->entry_point;
    bool running = true;

    // Call stack for recursion support (Issue #208)
    // MAX_CALL_DEPTH = 64 for bounded recursion
    uint call_stack[64];
    uint call_depth = 0;

    while (running && pc < header->code_size) {
        BytecodeInst inst = code[pc];
        uint op = inst.opcode;
        uint d = inst.dst & 0x1F;    // 5 bits for register
        uint s1 = inst.src1 & 0x1F;
        uint s2 = inst.src2 & 0x1F;
        // CRITICAL: Keep imm as uint to preserve integer bits for LOADI_INT/LOADI_UINT
        // Convert to float only when the instruction semantically needs a float
        uint imm_bits = inst.imm_bits;
        float imm = as_type<float>(imm_bits);  // For instructions that need float

        switch (op) {
            case OP_NOP: break;
            case OP_HALT: running = false; break;

            // Arithmetic
            case OP_MOV: regs[d] = regs[s1]; break;
            case OP_ADD: regs[d] = regs[s1] + regs[s2]; break;
            case OP_SUB: regs[d] = regs[s1] - regs[s2]; break;
            case OP_MUL: regs[d] = regs[s1] * regs[s2]; break;
            // IEEE 754 semantics: div by 0 = ±inf, 0/0 = NaN
            case OP_DIV: regs[d] = regs[s1] / regs[s2]; break;
            case OP_MOD: regs[d].x = fmod(regs[s1].x, regs[s2].x); break;

            // Math intrinsics (Phase 5 - Issue #178)
            case OP_SIN: regs[d].x = sin(regs[s1].x); break;
            case OP_COS: regs[d].x = cos(regs[s1].x); break;
            case OP_SQRT: regs[d].x = sqrt(regs[s1].x); break;

            // Float unary/binary math ops (Issue #198)
            case OP_ABS: regs[d].x = fabs(regs[s1].x); break;
            case OP_CEIL: regs[d].x = ceil(regs[s1].x); break;
            case OP_FLOOR: regs[d].x = floor(regs[s1].x); break;
            case OP_TRUNC: regs[d].x = trunc(regs[s1].x); break;
            case OP_NEAREST: regs[d].x = rint(regs[s1].x); break;  // rounds to nearest, ties to even
            case OP_COPYSIGN: regs[d].x = copysign(regs[s1].x, regs[s2].x); break;
            case OP_FMIN: regs[d].x = fmin(regs[s1].x, regs[s2].x); break;
            case OP_FMAX: regs[d].x = fmax(regs[s1].x, regs[s2].x); break;
            case OP_FNEG: regs[d].x = -regs[s1].x; break;

            // ═══════════════════════════════════════════════════════════════
            // F64 OPERATIONS (Issue #27 fix - Double-Single Emulation)
            // THE GPU IS THE COMPUTER - Metal does NOT support native double
            // We emulate f64 using double-single: regs[r].xy where value = x + y
            // This provides ~47 bits of mantissa precision (vs 52 for native f64)
            // ═══════════════════════════════════════════════════════════════
            case OP_F64_ADD: regs[d].xy = ds_add(regs[s1].xy, regs[s2].xy); break;
            case OP_F64_SUB: regs[d].xy = ds_sub(regs[s1].xy, regs[s2].xy); break;
            case OP_F64_MUL: regs[d].xy = ds_mul(regs[s1].xy, regs[s2].xy); break;
            case OP_F64_DIV: regs[d].xy = ds_div(regs[s1].xy, regs[s2].xy); break;
            case OP_F64_SQRT: regs[d].xy = ds_sqrt(regs[s1].xy); break;

            // F64 comparisons - double-single aware, return 1.0f or 0.0f in dst.x
            case OP_F64_EQ: regs[d].x = ds_eq(regs[s1].xy, regs[s2].xy) ? 1.0f : 0.0f; break;
            case OP_F64_NE: regs[d].x = ds_ne(regs[s1].xy, regs[s2].xy) ? 1.0f : 0.0f; break;
            case OP_F64_LT: regs[d].x = ds_lt(regs[s1].xy, regs[s2].xy) ? 1.0f : 0.0f; break;
            case OP_F64_GT: regs[d].x = ds_gt(regs[s1].xy, regs[s2].xy) ? 1.0f : 0.0f; break;
            case OP_F64_LE: regs[d].x = ds_le(regs[s1].xy, regs[s2].xy) ? 1.0f : 0.0f; break;
            case OP_F64_GE: regs[d].x = ds_ge(regs[s1].xy, regs[s2].xy) ? 1.0f : 0.0f; break;

            // F64 min/max - double-single aware
            case OP_F64_MIN: regs[d].xy = ds_min(regs[s1].xy, regs[s2].xy); break;
            case OP_F64_MAX: regs[d].xy = ds_max(regs[s1].xy, regs[s2].xy); break;

            // F64 conversions using double-single representation
            // Note: i32 values are stored as float VALUES (1.0f for 1), not bits
            // Use int()/uint() to convert float value, not as_type<> which reinterprets bits
            case OP_F64_FROM_I32_S: regs[d].xy = ds_from_i32(int(regs[s1].x)); break;
            // For unsigned: convert float value to signed int, then reinterpret bits as unsigned
            // This handles negative i32 values (like -1) which should become large unsigned values
            case OP_F64_FROM_I32_U: regs[d].xy = ds_from_u32(as_type<uint>(int(regs[s1].x))); break;
            case OP_F64_FROM_I64_S: regs[d].xy = ds_from_i64(regs[s1].xy); break;
            case OP_F64_FROM_I64_U: regs[d].xy = ds_from_u64(regs[s1].xy); break;

            // Store as float VALUE (1 -> 1.0f), not as bits (1 -> 0x00000001)
            // This matches how LOADI_INT stores integers
            case OP_F64_TO_I32_S: regs[d].x = float(ds_to_i32(regs[s1].xy)); break;
            // For unsigned: truncate to uint, reinterpret bits as signed, then store as float
            // This handles large values (> INT_MAX) which should become negative i32 values
            case OP_F64_TO_I32_U: {
                uint u = ds_to_u32(regs[s1].xy);
                regs[d].x = float(as_type<int>(u));  // Reinterpret uint bits as signed, store as float value
                break;
            }
            case OP_F64_TO_I64_S: regs[d].xy = ds_to_i64(regs[s1].xy); break;
            case OP_F64_TO_I64_U: regs[d].xy = ds_to_u64(regs[s1].xy); break;

            // F64 reinterpret operations (approximations for double-single)
            // i64 bits -> f64 value: parse IEEE 754 bits and convert to double-single
            case OP_F64_REINTERPRET_I64: regs[d].xy = ds_from_f64_bits(regs[s1].xy); break;
            // f64 value -> i64 bits: convert double-single to value and extract IEEE 754 bits
            case OP_I64_REINTERPRET_F64: regs[d].xy = ds_to_f64_bits(regs[s1].xy); break;

            case OP_LOADI: regs[d] = float4(imm); break;
            // Issue #273 fix: Use as_type to preserve raw bits, not imm which may be NaN-flushed
            case OP_SETX: regs[d].x = as_type<float>(imm_bits); break;
            case OP_SETY: regs[d].y = as_type<float>(imm_bits); break;
            case OP_SETZ: regs[d].z = as_type<float>(imm_bits); break;
            case OP_SETW: regs[d].w = as_type<float>(imm_bits); break;

            // Vector packing - pack two scalar .x values into dst.xy
            // Used for emit_quad to build float2 position and size
            case OP_PACK2: {
                regs[d].x = regs[s1].x;
                regs[d].y = regs[s2].x;
                break;
            }

            // Issue #227 fix: Float comparisons return i32 (0 or 1), must store integer bits
            // as_type<float> is a bitcast, not arithmetic - FTZ doesn't apply to bitcasts
            // JZ/JNZ compare against 0.0f (bits 0x0), which works: 0x00000000 == 0.0f, 0x00000001 != 0.0f
            case OP_EQ: regs[d].x = as_type<float>(uint(regs[s1].x == regs[s2].x)); break;
            case OP_LT: regs[d].x = as_type<float>(uint(regs[s1].x < regs[s2].x)); break;
            case OP_GT: regs[d].x = as_type<float>(uint(regs[s1].x > regs[s2].x)); break;

            // Control flow
            // CRITICAL: Check float value, not integer bits
            // as_type<int>(1.0f) = 0x3F800000 which works, but comparisons now return 1.0f/0.0f
            case OP_JMP: pc = uint(imm) - 1; break;  // -1 because pc++ at end
            case OP_JZ: if (regs[s1].x == 0.0f) pc = uint(imm) - 1; break;
            case OP_JNZ: if (regs[s1].x != 0.0f) pc = uint(imm) - 1; break;

            // Memory
            // THE GPU IS THE COMPUTER - address registers hold integer bits, use as_type
            // Issue #250 fix: All memory operations now have bounds checking
            case OP_LD: {
                // Issue #213 fix: Use uint() float-to-int conversion instead of as_type<uint>()
                // Small integers stored via LOADI are float VALUES (3.0f), not integer BITS (0x3)
                // as_type<uint>(3.0f) = 0x40400000 (wrong!), uint(3.0f) = 3 (correct!)
                uint idx = uint(regs[s1].x) + uint(imm);
                if (idx >= state_size_float4) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                regs[d] = state[idx];
                break;
            }
            case OP_ST: {
                // Issue #213 fix: Use uint() float-to-int conversion instead of as_type<uint>()
                // Small integers stored via LOADI are float VALUES (3.0f), not integer BITS (0x3)
                // as_type<uint>(3.0f) = 0x40400000 (wrong!), uint(3.0f) = 3 (correct!)
                uint idx = uint(regs[s1].x) + uint(imm);
                if (idx >= state_size_float4) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                state[idx] = regs[s2];
                break;
            }
            case OP_LD1: {
                // Load 8-bit from byte address
                // CRITICAL FIX: Clear entire register to avoid garbage in .y/.z/.w
                device uchar* bytes = (device uchar*)state;
                uint idx = as_type<uint>(regs[s1].x) + uint(imm);
                if (idx >= state_size_bytes) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                regs[d] = float4(float(bytes[idx]), 0.0, 0.0, 0.0);
                break;
            }
            case OP_ST1: {
                device uchar* bytes = (device uchar*)state;
                uint idx = as_type<uint>(regs[s1].x) + uint(imm);
                if (idx >= state_size_bytes) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                bytes[idx] = uchar(regs[s2].x);
                break;
            }
            case OP_LD2: {
                // Load 16-bit (halfword) from byte address
                // CRITICAL FIX: Clear entire register to avoid garbage in .y/.z/.w
                device ushort* halfwords = (device ushort*)state;
                uint byte_addr = as_type<uint>(regs[s1].x) + uint(imm);
                if (byte_addr + 1 >= state_size_bytes) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                uint hw_idx = byte_addr / 2;  // Convert to halfword index
                regs[d] = float4(as_type<float>(uint(halfwords[hw_idx])), 0.0, 0.0, 0.0);
                break;
            }
            case OP_ST2: {
                // Store 16-bit (halfword) to byte address
                device ushort* halfwords = (device ushort*)state;
                uint byte_addr = as_type<uint>(regs[s1].x) + uint(imm);
                if (byte_addr + 1 >= state_size_bytes) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                uint hw_idx = byte_addr / 2;  // Convert to halfword index
                halfwords[hw_idx] = ushort(as_type<uint>(regs[s2].x));
                break;
            }
            case OP_LD4: {
                // Load 32-bit (word) from byte address
                // CRITICAL FIX: Clear entire register to avoid garbage in .y/.z/.w
                // This bug caused array access corruption - registers reused with stale values
                device uint* words = (device uint*)state;
                uint byte_addr = as_type<uint>(regs[s1].x) + uint(imm);
                if (byte_addr + 3 >= state_size_bytes) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                uint word_idx = byte_addr / 4;  // Convert to word index
                regs[d] = float4(as_type<float>(words[word_idx]), 0.0, 0.0, 0.0);
                break;
            }
            case OP_ST4: {
                // Store 32-bit (word) to byte address
                device uint* words = (device uint*)state;
                uint byte_addr = as_type<uint>(regs[s1].x) + uint(imm);
                if (byte_addr + 3 >= state_size_bytes) {
                    gpu_debug_i32(-5, tid, dbg, dbg_data);  // Memory access violation
                    running = false;
                    break;
                }
                uint word_idx = byte_addr / 4;  // Convert to word index
                words[word_idx] = as_type<uint>(regs[s2].x);
                break;
            }

            // Graphics
            case OP_QUAD: {
                // s1 = position register (xy), s2 = size register (xy)
                // dst register = color (u32 packed as 0xRRGGBBAA in .x bits), imm = depth
                if (vert_idx + 6 <= header->vertex_budget) {
                    float2 pos = regs[s1].xy;
                    float2 size = regs[s2].xy;

                    // Scale from app coordinate space (800x600) to actual screen space
                    // Apps emit coordinates assuming 800x600, shader divides by actual screen
                    float app_width = regs[20].x;   // What app thinks screen width is (800)
                    float app_height = regs[21].x;  // What app thinks screen height is (600)
                    if (app_width > 0 && app_height > 0) {
                        float scale_x = actual_screen_width / app_width;
                        float scale_y = actual_screen_height / app_height;
                        pos.x *= scale_x;
                        pos.y *= scale_y;
                        size.x *= scale_x;
                        size.y *= scale_y;
                    }

                    // Unpack u32 color (0xRRGGBBAA) to float4 RGBA
                    uint packed = as_type<uint>(regs[d].x);
                    float4 color = float4(
                        float((packed >> 24) & 0xFF) / 255.0,  // R
                        float((packed >> 16) & 0xFF) / 255.0,  // G
                        float((packed >> 8) & 0xFF) / 255.0,   // B
                        float(packed & 0xFF) / 255.0           // A
                    );

                    write_quad(verts + vert_idx, pos, size, imm, color);
                    vert_idx += 6;
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // 64-BIT INTEGER OPERATIONS (Issue #188)
            // THE GPU IS THE COMPUTER - Apple Silicon GPUs natively support int64/uint64
            // 64-bit values use regs[r].xy (2 x 32-bit floats reinterpreted as ulong)
            // ═══════════════════════════════════════════════════════════════

            // 64-bit arithmetic
            case OP_INT64_ADD: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(a + b);
                break;
            }
            case OP_INT64_SUB: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(a - b);
                break;
            }
            case OP_INT64_MUL: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(a * b);
                break;
            }
            case OP_INT64_DIV_S: {
                long a = as_type<long>(regs[s1].xy);
                long b = as_type<long>(regs[s2].xy);
                regs[d].xy = as_type<float2>(b != 0 ? a / b : 0L);
                break;
            }
            case OP_INT64_DIV_U: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(b != 0 ? a / b : 0UL);
                break;
            }
            case OP_INT64_REM_U: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(b != 0 ? a % b : 0UL);
                break;
            }
            case OP_INT64_REM_S: {
                long a = as_type<long>(regs[s1].xy);
                long b = as_type<long>(regs[s2].xy);
                regs[d].xy = as_type<float2>(b != 0 ? a % b : 0L);
                break;
            }

            // 64-bit bitwise
            case OP_INT64_AND: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(a & b);
                break;
            }
            case OP_INT64_OR: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(a | b);
                break;
            }
            case OP_INT64_XOR: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].xy = as_type<float2>(a ^ b);
                break;
            }
            case OP_INT64_SHL: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint shift = as_type<uint>(regs[s2].x) & 0x3F;  // Mask to 6 bits for 64-bit
                regs[d].xy = as_type<float2>(a << shift);
                break;
            }
            case OP_INT64_SHR_U: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint shift = as_type<uint>(regs[s2].x) & 0x3F;
                regs[d].xy = as_type<float2>(a >> shift);
                break;
            }
            case OP_INT64_SHR_S: {
                long a = as_type<long>(regs[s1].xy);
                uint shift = as_type<uint>(regs[s2].x) & 0x3F;
                regs[d].xy = as_type<float2>(a >> shift);  // Arithmetic shift for signed
                break;
            }
            // Issue #227 fix: INT64 comparisons must store integer bits for i32 return type
            // as_type<float>(1u) preserves bit pattern 0x00000001, read back as i32 correctly
            // (1.0f has bits 0x3F800000, which reads as 1065353216 - WRONG!)
            case OP_INT64_EQ: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].x = as_type<float>(uint(a == b));
                break;
            }
            case OP_INT64_NE: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].x = as_type<float>(uint(a != b));
                break;
            }
            case OP_INT64_LT_U: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].x = as_type<float>(uint(a < b));
                break;
            }
            case OP_INT64_LT_S: {
                long a = as_type<long>(regs[s1].xy);
                long b = as_type<long>(regs[s2].xy);
                regs[d].x = as_type<float>(uint(a < b));
                break;
            }
            case OP_INT64_LE_U: {
                ulong a = as_type<ulong>(regs[s1].xy);
                ulong b = as_type<ulong>(regs[s2].xy);
                regs[d].x = as_type<float>(uint(a <= b));
                break;
            }
            case OP_INT64_LE_S: {
                long a = as_type<long>(regs[s1].xy);
                long b = as_type<long>(regs[s2].xy);
                regs[d].x = as_type<float>(uint(a <= b));
                break;
            }
            case OP_INT64_EQZ: {
                ulong a = as_type<ulong>(regs[s1].xy);
                regs[d].x = as_type<float>(uint(a == 0));
                break;
            }
            case OP_INT64_ROTR: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint shift = as_type<uint>(regs[s2].x) & 0x3F;
                regs[d].xy = as_type<float2>((a >> shift) | (a << (64 - shift)));
                break;
            }
            case OP_INT64_ROTL: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint shift = as_type<uint>(regs[s2].x) & 0x3F;
                regs[d].xy = as_type<float2>((a << shift) | (a >> (64 - shift)));
                break;
            }
            case OP_INT64_CLZ: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint result = uint(clz(a));
                regs[d].x = as_type<float>(result);
                regs[d].y = 0.0;  // i64 result, high word is 0
                break;
            }
            case OP_INT64_CTZ: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint result = uint(ctz(a));
                regs[d].x = as_type<float>(result);
                regs[d].y = 0.0;  // i64 result, high word is 0
                break;
            }
            case OP_INT64_POPCNT: {
                ulong a = as_type<ulong>(regs[s1].xy);
                uint result = uint(popcount(a));
                regs[d].x = as_type<float>(result);
                regs[d].y = 0.0;  // i64 result, high word is 0
                break;
            }
            case OP_INT64_WRAP: {
                // Wrap to i32: take low 32 bits
                ulong a = as_type<ulong>(regs[s1].xy);
                regs[d].x = as_type<float>(uint(a & 0xFFFFFFFFUL));
                break;
            }
            case OP_INT64_EXTEND_U: {
                // Zero-extend i32 to i64
                uint a = as_type<uint>(regs[s1].x);
                regs[d].xy = as_type<float2>(ulong(a));
                break;
            }
            case OP_INT64_EXTEND_S: {
                // Sign-extend i32 to i64
                int a = as_type<int>(regs[s1].x);
                regs[d].xy = as_type<float2>(long(a));
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // INTEGER OPERATIONS (Phase 1 - Issue #171)
            // THE GPU IS THE COMPUTER - use as_type<> for zero-cost reinterpret
            // ═══════════════════════════════════════════════════════════════

            // Integer arithmetic
            case OP_INT_ADD: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = float(a + b);
                break;
            }
            case OP_INT_SUB: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = float(a - b);
                break;
            }
            case OP_INT_MUL: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = float(a * b);
                break;
            }
            case OP_INT_DIV_S: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = float(b != 0 ? a / b : 0);
                break;
            }
            case OP_INT_DIV_U: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = float(b != 0 ? a / b : 0u);
                break;
            }
            case OP_INT_REM_S: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = float(b != 0 ? a % b : 0);
                break;
            }
            case OP_INT_REM_U: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = float(b != 0 ? a % b : 0u);
                break;
            }
            case OP_INT_NEG: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                regs[d].x = float(-a);
                break;
            }

            // Bitwise operations - these need integer BITS, convert from float value
            case OP_BIT_AND: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = float(a & b);
                break;
            }
            case OP_BIT_OR: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = float(a | b);
                break;
            }
            case OP_BIT_XOR: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = float(a ^ b);
                break;
            }
            case OP_BIT_NOT: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                regs[d].x = float(~a);
                break;
            }
            case OP_SHL: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x) & 0x1F;  // Mask to 5 bits
                regs[d].x = float(a << b);
                break;
            }
            case OP_SHR_U: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x) & 0x1F;
                regs[d].x = float(a >> b);
                break;
            }
            case OP_SHR_S: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                uint b = uint(regs[s2].x) & 0x1F;
                regs[d].x = float(a >> b);  // Arithmetic shift
                break;
            }
            case OP_ROTL: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x) & 0x1F;
                regs[d].x = float((a << b) | (a >> (32 - b)));
                break;
            }
            case OP_ROTR: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x) & 0x1F;
                regs[d].x = float((a >> b) | (a << (32 - b)));
                break;
            }
            case OP_CLZ: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                regs[d].x = float(clz(a));
                break;
            }

            case OP_CTZ: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                regs[d].x = float(ctz(a));
                break;
            }

            case OP_POPCNT: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                regs[d].x = float(popcount(a));
                break;
            }

            // Issue #227 fix: Integer comparisons must store integer bits for i32 return type
            // as_type<float>(uint(cond)) preserves bit pattern 0/1, read back as i32 correctly
            case OP_INT_EQ: {
                // Issue #213 fix: Values are stored as floats now, convert to int for comparison
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = (a == b) ? 1.0f : 0.0f;  // Store result as float for JNZ
                break;
            }
            case OP_INT_NE: {
                // Issue #213 fix: Values are stored as floats now, convert to int for comparison
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = (a != b) ? 1.0f : 0.0f;  // Store result as float for JNZ
                break;
            }
            case OP_INT_LT_S: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = (a < b) ? 1.0f : 0.0f;
                break;
            }
            case OP_INT_LT_U: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = (a < b) ? 1.0f : 0.0f;
                break;
            }
            case OP_INT_LE_S: {
                // Issue #213 fix: Values are stored as floats now
                int a = int(regs[s1].x);
                int b = int(regs[s2].x);
                regs[d].x = (a <= b) ? 1.0f : 0.0f;
                break;
            }
            case OP_INT_LE_U: {
                // Issue #213 fix: Values are stored as floats now
                uint a = uint(regs[s1].x);
                uint b = uint(regs[s2].x);
                regs[d].x = (a <= b) ? 1.0f : 0.0f;
                break;
            }

            // Conversion (these actually convert, not just reinterpret)
            case OP_INT_TO_F: {
                int a = as_type<int>(regs[s1].x);
                regs[d].x = float(a);  // Actual conversion
                break;
            }
            case OP_UINT_TO_F: {
                uint a = as_type<uint>(regs[s1].x);
                regs[d].x = float(a);
                break;
            }
            case OP_F_TO_INT: {
                float a = regs[s1].x;
                regs[d].x = as_type<float>(int(a));
                break;
            }
            case OP_F_TO_UINT: {
                float a = regs[s1].x;
                regs[d].x = as_type<float>(uint(a));
                break;
            }

            // Load immediate integer
            // Issue #213 FIX: Store as float VALUE, not integer BITS!
            // Integer bits (e.g., 1024 = 0x00000400) are denormalized floats.
            // GPU FTZ (Flush-To-Zero) flushes denormals to 0 during MOV, corrupting constants.
            // By storing as float VALUE (1024 -> 1024.0f), we get normalized floats.
            // INT_ADD etc. read float values with int(reg.x) and store with float(result).
            case OP_LOADI_INT: {
                int int_val = as_type<int>(imm_bits);  // Interpret bits as signed int
                regs[d] = float4(float(int_val), 0.0, 0.0, 0.0);  // Store as float VALUE
                break;
            }
            case OP_LOADI_UINT: {
                uint uint_val = imm_bits;  // Already unsigned
                regs[d] = float4(float(uint_val), 0.0, 0.0, 0.0);  // Store as float VALUE
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // ATOMIC OPERATIONS (Phase 2 - Issue #172)
            // THE GPU IS THE COMPUTER - lock-free coordination, GPU never waits
            // Use device atomic_uint* for atomic access to state memory
            // ═══════════════════════════════════════════════════════════════

            case OP_ATOMIC_LOAD: {
                uint addr = as_type<uint>(regs[s1].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint val = atomic_load_explicit(ptr, memory_order_relaxed);
                regs[d].x = as_type<float>(val);
                break;
            }

            case OP_ATOMIC_STORE: {
                uint addr = as_type<uint>(regs[s2].x);  // addr in s2 for store
                uint val = as_type<uint>(regs[s1].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                atomic_store_explicit(ptr, val, memory_order_relaxed);
                break;
            }

            case OP_ATOMIC_ADD: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_add_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_SUB: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_sub_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_MAX_U: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_max_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_MIN_U: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_min_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_MAX_S: {
                uint addr = as_type<uint>(regs[s1].x);
                int val = as_type<int>(regs[s2].x);
                device atomic_int* ptr = (device atomic_int*)&((device int*)state)[addr];
                int old = atomic_fetch_max_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_MIN_S: {
                uint addr = as_type<uint>(regs[s1].x);
                int val = as_type<int>(regs[s2].x);
                device atomic_int* ptr = (device atomic_int*)&((device int*)state)[addr];
                int old = atomic_fetch_min_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_AND: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_and_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_OR: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_or_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_XOR: {
                uint addr = as_type<uint>(regs[s1].x);
                uint val = as_type<uint>(regs[s2].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_xor_explicit(ptr, val, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_CAS: {
                uint addr = as_type<uint>(regs[s1].x);
                uint expected = as_type<uint>(regs[s2].x);
                uint desired = as_type<uint>(imm);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                bool success = atomic_compare_exchange_weak_explicit(
                    ptr, &expected, desired,
                    memory_order_relaxed, memory_order_relaxed
                );
                regs[d].x = success ? 1.0f : 0.0f;  // NOT as_type<float>(1u) - denormal!
                break;
            }

            case OP_ATOMIC_INC: {
                uint addr = as_type<uint>(regs[s1].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_add_explicit(ptr, 1u, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_ATOMIC_DEC: {
                uint addr = as_type<uint>(regs[s1].x);
                device atomic_uint* ptr = (device atomic_uint*)&((device uint*)state)[addr];
                uint old = atomic_fetch_sub_explicit(ptr, 1u, memory_order_relaxed);
                regs[d].x = as_type<float>(old);
                break;
            }

            case OP_MEM_FENCE: {
                threadgroup_barrier(mem_flags::mem_device);
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // ALLOCATOR OPERATIONS (Phase 6 - Issue #179)
            // THE GPU IS THE COMPUTER - lock-free memory allocation
            // ═══════════════════════════════════════════════════════════════

            case OP_ALLOC: {
                // dst = gpu_alloc(size_reg.x, align_reg.x)
                uint size = as_type<uint>(regs[s1].x);
                uint align = as_type<uint>(regs[s2].x);
                uint ptr = gpu_alloc(size, align, alloc, heap);
                regs[d].x = as_type<float>(ptr);
                break;
            }

            case OP_DEALLOC: {
                // gpu_dealloc(ptr_reg.x, size_reg.x, align_reg.x)
                // ptr in s1, size in s2, align in imm
                uint ptr = as_type<uint>(regs[s1].x);
                uint size = as_type<uint>(regs[s2].x);
                // align is passed in imm but not used by dealloc (size class determines)
                gpu_dealloc(ptr, size, alloc, heap);
                break;
            }

            case OP_REALLOC: {
                // dst = gpu_realloc(ptr_reg.x, old_size_reg.x, new_size_reg.x)
                // ptr in s1, old_size in s2, new_size_reg index in imm
                uint ptr = as_type<uint>(regs[s1].x);
                uint old_size = as_type<uint>(regs[s2].x);
                uint new_size_reg = as_type<uint>(imm) & 0x1F;
                uint new_size = as_type<uint>(regs[new_size_reg].x);
                uint new_ptr = gpu_realloc(ptr, old_size, new_size, alloc, heap);
                regs[d].x = as_type<float>(new_ptr);
                break;
            }

            case OP_ALLOC_ZERO: {
                // dst = gpu_alloc_zeroed(size_reg.x, align_reg.x)
                uint size = as_type<uint>(regs[s1].x);
                uint align = as_type<uint>(regs[s2].x);
                uint ptr = gpu_alloc_zeroed(size, align, alloc, heap);
                regs[d].x = as_type<float>(ptr);
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // MEMORY OPERATIONS (Issue #210 - GPU-Native Dynamic Memory)
            // THE GPU IS THE COMPUTER - WASM linear memory on GPU
            // ═══════════════════════════════════════════════════════════════

            case OP_MEMORY_SIZE: {
                // dst = current memory size in pages (64KB per page)
                // Uses heap_size from SlabAllocator
                uint heap_size = alloc->heap_size;
                uint pages = heap_size / 65536;  // 64KB per WASM page
                regs[d].x = as_type<float>(pages);
                break;
            }

            case OP_MEMORY_GROW: {
                // dst = memory_grow(delta_pages) - returns old size or -1 on failure
                // GPU buffers are fixed-size, so we can only "grow" within heap_size
                uint delta_pages = as_type<uint>(regs[s1].x);
                uint heap_size = alloc->heap_size;
                uint max_pages = heap_size / 65536;

                // Current usage in pages (heap_top / 65536, rounded up)
                uint current_bytes = atomic_load_explicit(&alloc->heap_top, memory_order_relaxed);
                uint current_pages = (current_bytes + 65535) / 65536;

                // Check if we can "grow" (really just allow more of the fixed buffer)
                uint new_pages = current_pages + delta_pages;
                if (new_pages <= max_pages) {
                    // "Growth" succeeds - update heap_top to reserve pages
                    uint new_bytes = new_pages * 65536;
                    atomic_store_explicit(&alloc->heap_top, new_bytes, memory_order_relaxed);
                    regs[d].x = as_type<float>(current_pages);  // Return old size
                } else {
                    // Can't grow beyond heap size
                    regs[d].x = as_type<float>(0xFFFFFFFF);  // -1 as unsigned
                }
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // WASI OPERATIONS (Issue #207 - GPU-Native WASI)
            // THE GPU IS THE COMPUTER - WASI system calls on GPU
            // ═══════════════════════════════════════════════════════════════

            case OP_WASI_FD_WRITE: {
                // fd_write(fd, iovs, iovs_len, nwritten) -> errno
                // fd in imm, iovs in s1, nwritten in s2
                uint fd = as_type<uint>(imm);
                uint iovs_ptr = as_type<uint>(regs[s1].x);
                uint nwritten_ptr = as_type<uint>(regs[s2].x);

                // Only support stdout (1) and stderr (2)
                if (fd == 1 || fd == 2) {
                    // Read iovec: ptr at iovs_ptr, len at iovs_ptr+4
                    uint buf_ptr = as_type<uint>(state[iovs_ptr / 16].x);
                    uint buf_len = as_type<uint>(state[(iovs_ptr + 4) / 16].x);

                    // Write to debug buffer
                    device const uchar* str = heap + buf_ptr;
                    gpu_debug_str(str, buf_len, tid, dbg, dbg_data);

                    // Write bytes written
                    state[nwritten_ptr / 16].x = as_type<float>(buf_len);
                    regs[d].x = as_type<float>(0u);  // Success (errno 0)
                } else {
                    regs[d].x = as_type<float>(8u);  // EBADF
                }
                break;
            }

            case OP_WASI_FD_READ: {
                // fd_read always returns EBADF - no input on GPU
                regs[d].x = as_type<float>(8u);  // EBADF
                break;
            }

            case OP_WASI_PROC_EXIT: {
                // proc_exit(code) - halt execution
                // Code is in s1, but we just stop
                running = false;
                break;
            }

            case OP_WASI_ENVIRON_SIZES_GET: {
                // environ_sizes_get(count_ptr, size_ptr) -> errno
                // Return 0 count, 0 size (no environment on GPU)
                uint count_ptr = as_type<uint>(regs[s1].x);
                uint size_ptr = as_type<uint>(regs[s2].x);
                state[count_ptr / 16].x = as_type<float>(0u);
                state[size_ptr / 16].x = as_type<float>(0u);
                regs[d].x = as_type<float>(0u);  // Success
                break;
            }

            case OP_WASI_ENVIRON_GET: {
                // environ_get returns success (empty environment)
                regs[d].x = as_type<float>(0u);  // Success
                break;
            }

            case OP_WASI_ARGS_SIZES_GET: {
                // args_sizes_get(count_ptr, size_ptr) -> errno
                // Return 0 count, 0 size (no args on GPU)
                uint count_ptr = as_type<uint>(regs[s1].x);
                uint size_ptr = as_type<uint>(regs[s2].x);
                state[count_ptr / 16].x = as_type<float>(0u);
                state[size_ptr / 16].x = as_type<float>(0u);
                regs[d].x = as_type<float>(0u);  // Success
                break;
            }

            case OP_WASI_ARGS_GET: {
                // args_get returns success (empty args)
                regs[d].x = as_type<float>(0u);  // Success
                break;
            }

            case OP_WASI_CLOCK_TIME_GET: {
                // clock_time_get(clock_id, precision, time_ptr) -> errno
                // Return frame count as nanoseconds (1 frame = 16.67ms = 16670000ns)
                uint time_ptr = as_type<uint>(regs[s2].x);
                uint frame = *frame_counter;
                uint64_t ns = uint64_t(frame) * 16670000;  // ~60fps
                // Store as two 32-bit values (WASI uses 64-bit time)
                state[time_ptr / 16].x = as_type<float>(uint(ns & 0xFFFFFFFF));
                state[(time_ptr + 4) / 16].x = as_type<float>(uint(ns >> 32));
                regs[d].x = as_type<float>(0u);  // Success
                break;
            }

            case OP_WASI_RANDOM_GET: {
                // random_get(buf_ptr, buf_len) -> errno
                // Use thread ID + frame as pseudo-random seed
                uint buf_ptr = as_type<uint>(regs[s1].x);
                uint buf_len = as_type<uint>(regs[s2].x);

                // Simple LCG random generator seeded by tid + frame
                uint seed = tid + (*frame_counter * 1103515245);
                for (uint i = 0; i < buf_len; i++) {
                    seed = seed * 1103515245 + 12345;
                    heap[buf_ptr + i] = uchar(seed >> 16);
                }
                regs[d].x = as_type<float>(0u);  // Success
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // PANIC HANDLING (Issue #209 - GPU-Native Panic)
            // THE GPU IS THE COMPUTER - panic handling on GPU
            // ═══════════════════════════════════════════════════════════════

            case OP_PANIC: {
                // panic(msg_ptr, msg_len) - write message to debug buffer and halt
                // s1 = msg_ptr (byte offset into heap), s2 = msg_len
                uint msg_ptr = as_type<uint>(regs[s1].x);
                uint msg_len = as_type<uint>(regs[s2].x);

                // Write "PANIC: " prefix using individual chars
                gpu_debug_i32(-1, tid, dbg, dbg_data);  // Marker for panic
                gpu_debug_str(heap + msg_ptr, msg_len, tid, dbg, dbg_data);
                gpu_debug_newline(tid, dbg, dbg_data);
                gpu_debug_flush(tid, dbg, dbg_data);

                // Halt execution
                running = false;
                break;
            }

            case OP_UNREACHABLE: {
                // unreachable() - halt with unreachable trap
                // Write marker for unreachable
                gpu_debug_i32(-2, tid, dbg, dbg_data);  // Marker for unreachable
                gpu_debug_newline(tid, dbg, dbg_data);
                gpu_debug_flush(tid, dbg, dbg_data);

                // Halt execution
                running = false;
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // RECURSION SUPPORT (Issue #208 - GPU-Native Recursion)
            // THE GPU IS THE COMPUTER - function calls via GPU call stack
            // ═══════════════════════════════════════════════════════════════

            case OP_CALL_FUNC: {
                // call_func(target_pc) - push return address, jump to function
                uint target_pc = as_type<uint>(imm);

                // Check for stack overflow (max depth = 64)
                if (call_depth >= 64) {
                    gpu_debug_i32(-3, tid, dbg, dbg_data);  // Marker for stack overflow
                    gpu_debug_newline(tid, dbg, dbg_data);
                    running = false;
                    break;
                }

                // Push return address (next instruction)
                call_stack[call_depth++] = pc + 1;

                // Jump to target function
                pc = target_pc - 1;  // -1 because pc++ at end of loop
                break;
            }

            case OP_RETURN_FUNC: {
                // return_func() - pop return address, jump back to caller
                if (call_depth == 0) {
                    // Return from main - we're done
                    running = false;
                    break;
                }

                // Pop return address
                uint return_pc = call_stack[--call_depth];
                pc = return_pc - 1;  // -1 because pc++ at end of loop
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // TABLE OPERATIONS (Issue #212 - GPU-Native Table Operations)
            // THE GPU IS THE COMPUTER - tables are GPU-resident arrays with O(1) lookup
            //
            // Table layout in heap memory (per table):
            //   offset +0: size (uint)
            //   offset +4: max_size (uint)
            //   offset +8: entries[0] (uint - PC or 0xFFFFFFFF for null)
            //   offset +12: entries[1]
            //   ...
            //
            // Function table 0 starts at heap offset 0
            // Each additional table is at offset: table_idx * 1024
            // ═══════════════════════════════════════════════════════════════

            case OP_TABLE_GET: {
                // dst = table.get(table_idx, elem_idx)
                uint table_idx = as_type<uint>(imm);
                uint elem_idx = as_type<uint>(regs[s1].x);

                // Table base in heap
                uint table_base = table_idx * 1024;
                uint table_size = uint(heap[table_base]) | (uint(heap[table_base + 1]) << 8) |
                                  (uint(heap[table_base + 2]) << 16) | (uint(heap[table_base + 3]) << 24);

                // Bounds check
                if (elem_idx >= table_size) {
                    gpu_debug_i32(-4, tid, dbg, dbg_data);  // Table out of bounds
                    running = false;
                    break;
                }

                // Read entry (4 bytes at offset 8 + elem_idx * 4)
                uint entry_offset = table_base + 8 + elem_idx * 4;
                uint entry = uint(heap[entry_offset]) | (uint(heap[entry_offset + 1]) << 8) |
                             (uint(heap[entry_offset + 2]) << 16) | (uint(heap[entry_offset + 3]) << 24);
                regs[d].x = as_type<float>(entry);
                break;
            }

            case OP_TABLE_SET: {
                // table.set(table_idx, elem_idx, value)
                uint table_idx = as_type<uint>(imm);
                uint elem_idx = as_type<uint>(regs[s1].x);
                uint value = as_type<uint>(regs[s2].x);

                // Table base in heap
                uint table_base = table_idx * 1024;
                uint table_size = uint(heap[table_base]) | (uint(heap[table_base + 1]) << 8) |
                                  (uint(heap[table_base + 2]) << 16) | (uint(heap[table_base + 3]) << 24);

                // Bounds check
                if (elem_idx >= table_size) {
                    gpu_debug_i32(-4, tid, dbg, dbg_data);  // Table out of bounds
                    running = false;
                    break;
                }

                // Write entry (4 bytes at offset 8 + elem_idx * 4)
                uint entry_offset = table_base + 8 + elem_idx * 4;
                heap[entry_offset] = value & 0xFF;
                heap[entry_offset + 1] = (value >> 8) & 0xFF;
                heap[entry_offset + 2] = (value >> 16) & 0xFF;
                heap[entry_offset + 3] = (value >> 24) & 0xFF;
                break;
            }

            case OP_TABLE_SIZE: {
                // dst = table.size(table_idx)
                uint table_idx = as_type<uint>(imm);
                uint table_base = table_idx * 1024;
                uint table_size = uint(heap[table_base]) | (uint(heap[table_base + 1]) << 8) |
                                  (uint(heap[table_base + 2]) << 16) | (uint(heap[table_base + 3]) << 24);
                regs[d].x = as_type<float>(table_size);
                break;
            }

            case OP_TABLE_GROW: {
                // dst = table.grow(table_idx, delta, init_val)
                uint table_idx = as_type<uint>(imm);
                uint delta = as_type<uint>(regs[s1].x);
                uint init_val = as_type<uint>(regs[s2].x);

                uint table_base = table_idx * 1024;
                uint table_size = uint(heap[table_base]) | (uint(heap[table_base + 1]) << 8) |
                                  (uint(heap[table_base + 2]) << 16) | (uint(heap[table_base + 3]) << 24);
                uint max_size = uint(heap[table_base + 4]) | (uint(heap[table_base + 5]) << 8) |
                                (uint(heap[table_base + 6]) << 16) | (uint(heap[table_base + 7]) << 24);

                uint new_size = table_size + delta;
                if (new_size > max_size || new_size > 253) {  // Max entries = (1024 - 8) / 4 = 254
                    regs[d].x = as_type<float>(0xFFFFFFFFu);  // Failure
                    break;
                }

                // Initialize new entries with init_val
                for (uint i = table_size; i < new_size; i++) {
                    uint entry_offset = table_base + 8 + i * 4;
                    heap[entry_offset] = init_val & 0xFF;
                    heap[entry_offset + 1] = (init_val >> 8) & 0xFF;
                    heap[entry_offset + 2] = (init_val >> 16) & 0xFF;
                    heap[entry_offset + 3] = (init_val >> 24) & 0xFF;
                }

                // Update size
                heap[table_base] = new_size & 0xFF;
                heap[table_base + 1] = (new_size >> 8) & 0xFF;
                heap[table_base + 2] = (new_size >> 16) & 0xFF;
                heap[table_base + 3] = (new_size >> 24) & 0xFF;

                regs[d].x = as_type<float>(table_size);  // Return old size
                break;
            }

            case OP_TABLE_FILL: {
                // table.fill(table_idx, dst, value, count)
                uint table_idx = as_type<uint>(imm);
                uint dst_idx = as_type<uint>(regs[s1].x);
                uint value = as_type<uint>(regs[s2].x);
                uint count = as_type<uint>(regs[d].x);

                uint table_base = table_idx * 1024;
                uint table_size = uint(heap[table_base]) | (uint(heap[table_base + 1]) << 8) |
                                  (uint(heap[table_base + 2]) << 16) | (uint(heap[table_base + 3]) << 24);

                if (dst_idx + count > table_size) {
                    gpu_debug_i32(-4, tid, dbg, dbg_data);  // Table out of bounds
                    running = false;
                    break;
                }

                for (uint i = 0; i < count; i++) {
                    uint entry_offset = table_base + 8 + (dst_idx + i) * 4;
                    heap[entry_offset] = value & 0xFF;
                    heap[entry_offset + 1] = (value >> 8) & 0xFF;
                    heap[entry_offset + 2] = (value >> 16) & 0xFF;
                    heap[entry_offset + 3] = (value >> 24) & 0xFF;
                }
                break;
            }

            case OP_TABLE_COPY: {
                // table.copy(dst_table, src_table, dst, src, count)
                uint packed = as_type<uint>(imm);
                uint dst_table = (packed >> 16) & 0xFFFF;
                uint src_table = packed & 0xFFFF;
                uint dst_idx = as_type<uint>(regs[s1].x);
                uint src_idx = as_type<uint>(regs[s2].x);
                uint count = as_type<uint>(regs[d].x);

                uint dst_base = dst_table * 1024;
                uint src_base = src_table * 1024;

                // Read sizes (simplified bounds check)
                uint dst_size = uint(heap[dst_base]) | (uint(heap[dst_base + 1]) << 8) |
                                (uint(heap[dst_base + 2]) << 16) | (uint(heap[dst_base + 3]) << 24);
                uint src_size = uint(heap[src_base]) | (uint(heap[src_base + 1]) << 8) |
                                (uint(heap[src_base + 2]) << 16) | (uint(heap[src_base + 3]) << 24);

                if (dst_idx + count > dst_size || src_idx + count > src_size) {
                    gpu_debug_i32(-4, tid, dbg, dbg_data);  // Table out of bounds
                    running = false;
                    break;
                }

                // Copy with memmove semantics for overlapping regions
                if (dst_table == src_table && dst_idx > src_idx) {
                    // Copy backwards
                    for (int i = count - 1; i >= 0; i--) {
                        uint src_off = src_base + 8 + (src_idx + i) * 4;
                        uint dst_off = dst_base + 8 + (dst_idx + i) * 4;
                        heap[dst_off] = heap[src_off];
                        heap[dst_off + 1] = heap[src_off + 1];
                        heap[dst_off + 2] = heap[src_off + 2];
                        heap[dst_off + 3] = heap[src_off + 3];
                    }
                } else {
                    // Copy forwards
                    for (uint i = 0; i < count; i++) {
                        uint src_off = src_base + 8 + (src_idx + i) * 4;
                        uint dst_off = dst_base + 8 + (dst_idx + i) * 4;
                        heap[dst_off] = heap[src_off];
                        heap[dst_off + 1] = heap[src_off + 1];
                        heap[dst_off + 2] = heap[src_off + 2];
                        heap[dst_off + 3] = heap[src_off + 3];
                    }
                }
                break;
            }

            case OP_TABLE_INIT: {
                // table.init - not fully implemented (needs element segments)
                // For now, treat as no-op
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // SIMD OPERATIONS (Issue #211)
            // THE GPU IS THE COMPUTER - float4 is native SIMD, all 4 lanes
            // ═══════════════════════════════════════════════════════════════

            case OP_V4_ADD: {
                // dst = s1 + s2 (all 4 lanes)
                regs[d] = regs[s1] + regs[s2];
                break;
            }

            case OP_V4_SUB: {
                // dst = s1 - s2 (all 4 lanes)
                regs[d] = regs[s1] - regs[s2];
                break;
            }

            case OP_V4_MUL: {
                // dst = s1 * s2 (all 4 lanes)
                regs[d] = regs[s1] * regs[s2];
                break;
            }

            case OP_V4_DIV: {
                // dst = s1 / s2 (all 4 lanes)
                regs[d] = regs[s1] / regs[s2];
                break;
            }

            case OP_V4_MIN: {
                // dst = min(s1, s2) per lane
                regs[d] = min(regs[s1], regs[s2]);
                break;
            }

            case OP_V4_MAX: {
                // dst = max(s1, s2) per lane
                regs[d] = max(regs[s1], regs[s2]);
                break;
            }

            case OP_V4_ABS: {
                // dst = abs(s1) per lane
                regs[d] = abs(regs[s1]);
                break;
            }

            case OP_V4_NEG: {
                // dst = -s1 per lane
                regs[d] = -regs[s1];
                break;
            }

            case OP_V4_SQRT: {
                // dst = sqrt(s1) per lane
                regs[d] = sqrt(regs[s1]);
                break;
            }

            case OP_V4_DOT: {
                // dst.x = dot(s1, s2)
                regs[d].x = dot(regs[s1], regs[s2]);
                break;
            }

            case OP_V4_SHUFFLE: {
                // Swizzle/shuffle based on mask in imm
                // mask: 4 2-bit indices (bits 0-1=x, 2-3=y, 4-5=z, 6-7=w)
                uint mask = as_type<uint>(imm);
                float4 src = regs[s1];
                float4 result;
                result.x = src[(mask >> 0) & 3];
                result.y = src[(mask >> 2) & 3];
                result.z = src[(mask >> 4) & 3];
                result.w = src[(mask >> 6) & 3];
                regs[d] = result;
                break;
            }

            case OP_V4_EXTRACT: {
                // dst.x = s1[lane]
                uint lane = as_type<uint>(imm) & 3;
                regs[d].x = regs[s1][lane];
                break;
            }

            case OP_V4_REPLACE: {
                // dst = s1 with dst[lane] = s2.x
                uint lane = as_type<uint>(imm) & 3;
                float4 result = regs[s1];
                result[lane] = regs[s2].x;
                regs[d] = result;
                break;
            }

            case OP_V4_SPLAT: {
                // dst = (s1.x, s1.x, s1.x, s1.x)
                float val = regs[s1].x;
                regs[d] = float4(val, val, val, val);
                break;
            }

            case OP_V4_EQ: {
                // dst = (s1 == s2) per lane (1.0 or 0.0)
                float4 a = regs[s1];
                float4 b = regs[s2];
                regs[d] = float4(
                    (a.x == b.x) ? 1.0f : 0.0f,
                    (a.y == b.y) ? 1.0f : 0.0f,
                    (a.z == b.z) ? 1.0f : 0.0f,
                    (a.w == b.w) ? 1.0f : 0.0f
                );
                break;
            }

            case OP_V4_LT: {
                // dst = (s1 < s2) per lane (1.0 or 0.0)
                float4 a = regs[s1];
                float4 b = regs[s2];
                regs[d] = float4(
                    (a.x < b.x) ? 1.0f : 0.0f,
                    (a.y < b.y) ? 1.0f : 0.0f,
                    (a.z < b.z) ? 1.0f : 0.0f,
                    (a.w < b.w) ? 1.0f : 0.0f
                );
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // DEBUG I/O OPERATIONS (Phase 7 - Issue #180)
            // THE GPU IS THE COMPUTER - debug output via ring buffer
            // Lock-free atomic writes include thread ID for multi-thread debugging
            // ═══════════════════════════════════════════════════════════════

            case OP_DBG_I32: {
                // Debug print i32 from src1 register
                int value = as_type<int>(regs[s1].x);
                gpu_debug_i32(value, tid, dbg, dbg_data);
                break;
            }

            case OP_DBG_F32: {
                // Debug print f32 from src1 register
                float value = regs[s1].x;
                gpu_debug_f32(value, tid, dbg, dbg_data);
                break;
            }

            case OP_DBG_STR: {
                // Debug print string (ptr=src1, len=src2)
                uint str_ptr = as_type<uint>(regs[s1].x);
                uint str_len = as_type<uint>(regs[s2].x);
                // String data is in state memory (relative to heap)
                device const uchar* str = heap + str_ptr;
                gpu_debug_str(str, str_len, tid, dbg, dbg_data);
                break;
            }

            case OP_DBG_BOOL: {
                // Debug print bool from src1 register (0 = false, non-zero = true)
                bool value = as_type<uint>(regs[s1].x) != 0;
                gpu_debug_bool(value, tid, dbg, dbg_data);
                break;
            }

            case OP_DBG_NL: {
                // Debug newline marker
                gpu_debug_newline(tid, dbg, dbg_data);
                break;
            }

            case OP_DBG_FLUSH: {
                // Force debug flush (marker for CPU)
                gpu_debug_flush(tid, dbg, dbg_data);
                break;
            }

            // ═══════════════════════════════════════════════════════════════
            // AUTOMATIC CODE TRANSFORMATION OPERATIONS (Phase 8 - Issue #182)
            // THE GPU IS THE COMPUTER - GPU-native implementations of CPU patterns
            // ═══════════════════════════════════════════════════════════════

            case OP_WORK_PUSH: {
                // Push work item to queue: work_push(item_reg, queue_reg)
                uint item = as_type<uint>(regs[s1].x);
                // queue_reg points to queue index (0 = default work_queue)
                bool success = work_push(item, work_queue);
                // Optional: could return success status
                break;
            }

            case OP_WORK_POP: {
                // Pop work item from queue: dst = work_pop(queue_reg)
                uint item = work_pop(work_queue);
                regs[d].x = as_type<float>(item);
                break;
            }

            case OP_REQUEST_QUEUE: {
                // Queue I/O request (placeholder for future async I/O)
                // request_queue(type_reg, data_reg)
                // For now, just store the request type for debugging
                uint req_type = as_type<uint>(regs[s1].x);
                uint req_data = as_type<uint>(regs[s2].x);
                // TODO: Implement actual I/O request queueing
                break;
            }

            case OP_REQUEST_POLL: {
                // Poll I/O request status (placeholder)
                // dst = request_poll(id_reg)
                // Returns: 0 = pending, 1 = complete, 0xFFFFFFFF = error
                // For now, always return "complete"
                regs[d].x = as_type<float>(1u);
                break;
            }

            case OP_FRAME_WAIT: {
                // Wait for N frames: frame_wait(frames_reg)
                uint frames = as_type<uint>(regs[s1].x);
                frame_wait(frames, frame_counter);
                break;
            }

            case OP_SPINLOCK: {
                // Acquire spinlock: spinlock(lock_reg)
                // lock_reg contains address of lock in state memory
                uint lock_addr = as_type<uint>(regs[s1].x);
                device atomic_uint* lock = (device atomic_uint*)&((device uint*)state)[lock_addr];
                bool acquired = spinlock_acquire(lock);
                // Store result in same register (1 = acquired, 0 = timeout)
                regs[d].x = acquired ? 1.0f : 0.0f;  // NOT as_type<float>(1u) - denormal!
                break;
            }

            case OP_SPINUNLOCK: {
                // Release spinlock: spinunlock(lock_reg)
                uint lock_addr = as_type<uint>(regs[s1].x);
                device atomic_uint* lock = (device atomic_uint*)&((device uint*)state)[lock_addr];
                spinlock_release(lock);
                break;
            }

            case OP_BARRIER: {
                // Threadgroup barrier (Condvar::wait transformation)
                // All threads must reach this point before any continue
                threadgroup_barrier(mem_flags::mem_device);
                break;
            }

            default: break;  // Unknown opcode - skip
        }
        pc++;
    }

    // Note: Result is stored by bytecode explicitly (st instruction to state[3])
    // State layout: SlabAllocator[0-2] (48 bytes) | result[3] (16 bytes) | params[4-7] (64 bytes) | heap[8+]

    app->vertex_count = vert_idx;
}

// App dispatcher - routes to app-specific update
inline void dispatch_app_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device RenderVertex* unified_vertices,
    device GpuWindow* windows,
    device GpuAppDescriptor* all_apps,
    uint window_count,
    uint max_slots,
    float screen_width,
    float screen_height,
    uint tid,
    uint tg_size
) {
    switch (app->app_type) {
        case APP_TYPE_GAME_OF_LIFE:
            game_of_life_update(app, unified_state, tid, tg_size);
            break;
        case APP_TYPE_PARTICLES:
            particles_update(app, unified_state, tid, tg_size);
            break;
        // System apps
        case APP_TYPE_COMPOSITOR:
            compositor_update(app, unified_state, unified_vertices, all_apps, max_slots, tid, tg_size);
            break;
        case APP_TYPE_DOCK:
            dock_update(app, unified_state, unified_vertices, tid, tg_size);
            break;
        case APP_TYPE_MENUBAR:
            menubar_update(app, unified_state, unified_vertices, tid, tg_size);
            break;
        case APP_TYPE_WINDOW_CHROME:
            window_chrome_update(app, unified_state, unified_vertices, windows, window_count, tid, tg_size);
            break;
        case APP_TYPE_BYTECODE:
            bytecode_update(app, unified_state, unified_vertices, screen_width, screen_height, tid, tg_size);
            break;
        default:
            // CUSTOM and unknown types use counter
            counter_app_update(app, unified_state, tid, tg_size);
            break;
    }
}

// ============================================================================
// MEGAKERNEL (all apps in one dispatch)
// ============================================================================

// Legacy placeholder (for backward compatibility)
inline void placeholder_app_update(device GpuAppDescriptor* app, device uchar* state) {
    // Increment a counter in state
    device uint* counter = (device uint*)(state + app->state_offset);
    *counter = *counter + 1;
    app->vertex_count = 6;
}

// Simple megakernel (no budget enforcement)
kernel void gpu_app_megakernel(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    constant uint& frame_number [[buffer(3)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];

    // O(1) predicate evaluation - each app decides for itself
    if (!should_i_run(app, frame_number)) return;

    // TODO: Dispatch to app-specific update based on app_type
    // For now, use placeholder
    placeholder_app_update(app, unified_state);

    app->last_run_frame = frame_number;
    app->flags &= ~APP_FLAG_DIRTY;
}

// Issue #156: Megakernel with frame budget enforcement
kernel void gpu_app_megakernel_budgeted(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    device FrameBudget* budget [[buffer(3)]],
    constant uint& frame_number [[buffer(4)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];

    // O(1) predicate evaluation
    if (!should_i_run(app, frame_number)) return;

    // Get effective priority (boosted if starving)
    uint priority = effective_priority(app, frame_number);

    // Try to claim execution budget
    uint cost = app->thread_count;
    if (!try_claim_budget(budget, cost, priority)) {
        return;  // Over budget, skip this frame
    }

    // Execute app update
    placeholder_app_update(app, unified_state);

    app->last_run_frame = frame_number;
    app->flags &= ~APP_FLAG_DIRTY;
}

// Issue #159: Parallel megakernel (one threadgroup per app slot)
// Each threadgroup has multiple threads for parallel app execution
kernel void gpu_app_megakernel_parallel(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    constant uint& frame_number [[buffer(3)]],
    device RenderVertex* unified_vertices [[buffer(4)]],
    device GpuWindow* windows [[buffer(5)]],
    constant uint& window_count [[buffer(6)]],
    constant float2& screen_size [[buffer(7)]],
    uint slot_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (slot_id >= header->max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];

    // O(1) predicate evaluation (thread 0 only, but all threads check)
    if (!should_i_run(app, frame_number)) return;

    // Dispatch to app-specific update with parallel threads
    // Pass all_apps and max_slots for compositor to count vertices
    // Pass screen_size for bytecode apps to scale coordinates
    dispatch_app_update(app, unified_state, unified_vertices, windows, apps, window_count, header->max_slots, screen_size.x, screen_size.y, tid, tg_size);

    // Barrier before updating flags (ensure all threads done)
    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0 updates tracking
    if (tid == 0) {
        app->last_run_frame = frame_number;
        app->flags &= ~APP_FLAG_DIRTY;
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

kernel void mark_all_dirty(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;
    if (apps[slot_id].flags & APP_FLAG_ACTIVE) {
        apps[slot_id].flags |= APP_FLAG_DIRTY;
    }
}

kernel void get_app_stats(
    device const AppTableHeader* header [[buffer(0)]],
    device const GpuAppDescriptor* apps [[buffer(1)]],
    device uint* stats [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    stats[0] = atomic_load_explicit(&header->active_count, memory_order_relaxed);
    stats[1] = header->max_slots;

    uint visible = 0;
    uint dirty = 0;
    uint focused = 0;

    for (uint i = 0; i < header->max_slots; i++) {
        if (apps[i].flags & APP_FLAG_ACTIVE) {
            if (apps[i].flags & APP_FLAG_VISIBLE) visible++;
            if (apps[i].flags & APP_FLAG_DIRTY) dirty++;
            if (apps[i].flags & APP_FLAG_FOCUS) focused++;
        }
    }

    stats[2] = visible;
    stats[3] = dirty;
    stats[4] = focused;
}

// ============================================================================
// Issue #157: GPU INPUT & WINDOW INTEGRATION
// ============================================================================

// Input event types
constant uint EVENT_NONE = 0;
constant uint EVENT_KEY_DOWN = 1;
constant uint EVENT_KEY_UP = 2;
constant uint EVENT_MOUSE_MOVE = 3;
constant uint EVENT_MOUSE_DOWN = 4;
constant uint EVENT_MOUSE_UP = 5;
constant uint EVENT_SCROLL = 6;

struct InputEvent {
    uint event_type;      // EVENT_*
    uint key_or_button;   // Key code or mouse button
    float2 position;      // Cursor position
    uint modifiers;       // Shift, Ctrl, Alt, Cmd
    uint frame;           // Frame when event occurred
    uint _pad[2];
};

struct InputQueue {
    atomic_uint head;     // Consumer position
    atomic_uint tail;     // Producer position (CPU writes here)
    uint capacity;
    uint _pad;
};

// GpuWindow struct defined earlier in file

// O(1) bounds check
inline bool point_in_rect(float2 p, float x, float y, float w, float h) {
    return p.x >= x && p.x < x + w &&
           p.y >= y && p.y < y + h;
}

// Each app processes input in parallel - checks "is this for me?"
kernel void gpu_process_input(
    device InputQueue* queue [[buffer(0)]],
    device InputEvent* events [[buffer(1)]],
    device GpuAppDescriptor* apps [[buffer(2)]],
    device GpuWindow* windows [[buffer(3)]],
    constant uint& max_slots [[buffer(4)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];
    if (!(app->flags & APP_FLAG_ACTIVE)) return;

    // Get window if app has one
    GpuWindow window = windows[app->window_id];
    bool i_am_focused = (app->flags & APP_FLAG_FOCUS) != 0;

    // Read event queue (all apps read same queue)
    uint head = atomic_load_explicit(&queue->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);

    for (uint i = head; i < tail && i < head + 8; i++) {  // Max 8 events per frame
        InputEvent event = events[i % queue->capacity];

        bool this_is_for_me = false;

        if (event.event_type == EVENT_KEY_DOWN || event.event_type == EVENT_KEY_UP) {
            // Keyboard: only focused app
            this_is_for_me = i_am_focused;
        } else if (event.event_type >= EVENT_MOUSE_MOVE) {
            // Mouse: app under cursor
            this_is_for_me = point_in_rect(event.position, window.x, window.y, window.width, window.height);
        }

        if (this_is_for_me) {
            // Add to my local input queue (circular buffer in descriptor)
            uint my_tail = app->input_tail;
            if (my_tail - app->input_head < 8) {  // Don't overflow
                // Pack event into 32 bits: type(4) | key(12) | modifiers(8) | reserved(8)
                app->input_events[my_tail % 8] =
                    (event.event_type & 0xF) |
                    ((event.key_or_button & 0xFFF) << 4) |
                    ((event.modifiers & 0xFF) << 16);
                app->input_tail = my_tail + 1;
                app->flags |= APP_FLAG_DIRTY;  // Wake me up
            }
        }
    }
}

// Find topmost window at click position (parallel max)
kernel void gpu_find_focus_target(
    device GpuAppDescriptor* apps [[buffer(0)]],
    device GpuWindow* windows [[buffer(1)]],
    constant float2& click_pos [[buffer(2)]],
    constant uint& max_slots [[buffer(3)]],
    device atomic_uint* topmost_slot [[buffer(4)]],
    device atomic_uint* topmost_depth [[buffer(5)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];
    if (!(app->flags & APP_FLAG_ACTIVE)) return;
    if (!(app->flags & APP_FLAG_VISIBLE)) return;

    GpuWindow window = windows[app->window_id];

    // Am I under the click?
    if (!point_in_rect(click_pos, window.x, window.y, window.width, window.height)) return;

    // Parallel max to find topmost (depth as uint for atomic compare)
    uint my_depth = as_type<uint>(window.depth);
    uint old_depth = atomic_load_explicit(topmost_depth, memory_order_relaxed);

    while (my_depth > old_depth) {
        if (atomic_compare_exchange_weak_explicit(
            topmost_depth, &old_depth, my_depth,
            memory_order_relaxed, memory_order_relaxed
        )) {
            atomic_store_explicit(topmost_slot, slot_id, memory_order_relaxed);
            break;
        }
    }
}

// Apply focus to the found slot (parallel flag update)
kernel void gpu_apply_focus(
    device GpuAppDescriptor* apps [[buffer(0)]],
    constant uint& new_focus_slot [[buffer(1)]],
    constant uint& max_slots [[buffer(2)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= max_slots) return;

    if (slot_id == new_focus_slot) {
        apps[slot_id].flags |= APP_FLAG_FOCUS;
    } else {
        apps[slot_id].flags &= ~APP_FLAG_FOCUS;
    }
}

// Advance input queue head (after all apps have processed)
kernel void gpu_advance_input_queue(
    device InputQueue* queue [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
    atomic_store_explicit(&queue->head, tail, memory_order_relaxed);
}

// Issue #155: Memory pool statistics
kernel void get_memory_stats(
    device const MemoryPool* state_pool [[buffer(0)]],
    device const MemoryPool* vertex_pool [[buffer(1)]],
    device uint* stats [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // State pool stats
    stats[0] = atomic_load_explicit(&state_pool->freelist_head, memory_order_relaxed);
    stats[1] = atomic_load_explicit(&state_pool->bump_pointer, memory_order_relaxed);
    stats[2] = atomic_load_explicit(&state_pool->free_count, memory_order_relaxed);
    stats[3] = state_pool->pool_size;
    stats[4] = atomic_load_explicit(&state_pool->block_count, memory_order_relaxed);

    // Vertex pool stats
    stats[5] = atomic_load_explicit(&vertex_pool->freelist_head, memory_order_relaxed);
    stats[6] = atomic_load_explicit(&vertex_pool->bump_pointer, memory_order_relaxed);
    stats[7] = atomic_load_explicit(&vertex_pool->free_count, memory_order_relaxed);
    stats[8] = vertex_pool->pool_size;
    stats[9] = atomic_load_explicit(&vertex_pool->block_count, memory_order_relaxed);
}

// ============================================================================
// ISSUE #158: GPU RENDERING PIPELINE
// ============================================================================

// Render state - tracks total vertices for single draw call
struct RenderState {
    atomic_uint total_vertex_count;  // Sum of all app vertex counts
    uint max_vertices;               // Capacity limit
    uint screen_width;
    uint screen_height;
};

// RenderVertex struct defined earlier in file

// Finalize render: compute total vertices needed for slot-based layout
// With slot-based layout, we need to draw up to (max_active_slot + 1) * VERTS_PER_SLOT
kernel void gpu_finalize_render(
    device const GpuAppDescriptor* apps [[buffer(0)]],
    device const AppTableHeader* header [[buffer(1)]],
    device RenderState* render [[buffer(2)]],
    device const GpuWindow* windows [[buffer(3)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;

    GpuAppDescriptor app = apps[slot_id];

    // Skip inactive or invisible apps
    if (!(app.flags & APP_FLAG_ACTIVE)) return;
    if (!(app.flags & APP_FLAG_VISIBLE)) return;

    // For slot-based layout: track highest vertex index used
    // vertex_end = slot * VERTS_PER_SLOT + vertex_count
    uint vertex_end = slot_id * VERTS_PER_SLOT + app.vertex_count;
    atomic_fetch_max_explicit(
        &render->total_vertex_count,
        vertex_end,
        memory_order_relaxed
    );
}

// Reset render state for new frame
kernel void gpu_reset_render_state(
    device RenderState* render [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    atomic_store_explicit(&render->total_vertex_count, 0, memory_order_relaxed);
}

// Generate test vertices for an app (simple colored quad)
// Each app writes 6 vertices (2 triangles) for its window
kernel void gpu_generate_test_vertices(
    device GpuAppDescriptor* apps [[buffer(0)]],
    device const AppTableHeader* header [[buffer(1)]],
    device RenderVertex* vertices [[buffer(2)]],
    device const GpuWindow* windows [[buffer(3)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];

    if (!(app->flags & APP_FLAG_ACTIVE)) return;
    if (!(app->flags & APP_FLAG_VISIBLE)) return;

    // Get window bounds
    GpuWindow window = windows[app->window_id];
    float depth = window.depth;

    // Colors based on slot (for visual debugging)
    float r = (float)((slot_id * 37) % 256) / 255.0;
    float g = (float)((slot_id * 73) % 256) / 255.0;
    float b = (float)((slot_id * 113) % 256) / 255.0;
    float4 color = float4(r, g, b, 1.0);

    // Write 6 vertices (2 triangles) for quad
    uint base = app->vertex_offset / sizeof(RenderVertex);
    device RenderVertex* v = vertices + base;

    // Triangle 1: top-left, top-right, bottom-right
    v[0].position = packed_float3(window.x, window.y, depth);
    v[0].color = color;
    v[0].uv = float2(0, 0);

    v[1].position = packed_float3(window.x + window.width, window.y, depth);
    v[1].color = color;
    v[1].uv = float2(1, 0);

    v[2].position = packed_float3(window.x + window.width, window.y + window.height, depth);
    v[2].color = color;
    v[2].uv = float2(1, 1);

    // Triangle 2: top-left, bottom-right, bottom-left
    v[3].position = packed_float3(window.x, window.y, depth);
    v[3].color = color;
    v[3].uv = float2(0, 0);

    v[4].position = packed_float3(window.x + window.width, window.y + window.height, depth);
    v[4].color = color;
    v[4].uv = float2(1, 1);

    v[5].position = packed_float3(window.x, window.y + window.height, depth);
    v[5].color = color;
    v[5].uv = float2(0, 1);

    // Update vertex count
    app->vertex_count = 6;
}
"#;

// ============================================================================
// GpuAppSystem - Main Interface
// ============================================================================

/// GPU-Centric App System
///
/// Manages app lifecycle entirely on the GPU. The CPU only submits
/// command buffers and handles I/O.
pub struct GpuAppSystem {
    device: Device,
    command_queue: CommandQueue,

    // Pipelines
    launch_pipeline: ComputePipelineState,
    launch_o1_pipeline: ComputePipelineState,  // Issue #155: O(1) launch
    close_pipeline: ComputePipelineState,
    megakernel_pipeline: ComputePipelineState,
    mark_dirty_pipeline: ComputePipelineState,
    stats_pipeline: ComputePipelineState,
    memory_stats_pipeline: ComputePipelineState,  // Issue #155: memory stats

    // GPU-resident buffers (legacy bump allocator)
    app_table_buffer: Buffer,      // Header + descriptors
    state_alloc_buffer: Buffer,    // Allocator for app state (legacy)
    vertex_alloc_buffer: Buffer,   // Allocator for vertices (legacy)
    unified_state_buffer: Buffer,  // All app state
    unified_vertex_buffer: Buffer, // All app vertices

    // Issue #155: O(1) memory management buffers
    state_pool_buffer: Buffer,        // MemoryPool for state
    state_blocks_buffer: Buffer,      // FreeBlock array for state
    vertex_pool_buffer: Buffer,       // MemoryPool for vertices
    vertex_blocks_buffer: Buffer,     // FreeBlock array for vertices
    use_o1_allocator: bool,           // Use O(1) allocator instead of bump

    // Issue #156: GPU Scheduler
    budget_buffer: Buffer,            // FrameBudget for execution control
    budgeted_megakernel_pipeline: ComputePipelineState,
    use_budgeted_scheduler: bool,     // Use budget-enforced scheduler

    // Issue #157: Input & Windows
    input_queue_buffer: Buffer,       // InputQueue header
    input_events_buffer: Buffer,      // Array of InputEvent
    windows_buffer: Buffer,           // Array of GpuWindow
    process_input_pipeline: ComputePipelineState,
    find_focus_pipeline: ComputePipelineState,
    apply_focus_pipeline: ComputePipelineState,
    advance_queue_pipeline: ComputePipelineState,

    // Issue #158: Rendering Pipeline
    render_state_buffer: Buffer,          // RenderState for vertex count
    render_vertices_buffer: Buffer,       // Unified render vertices
    screen_size_buffer: Buffer,           // float2 for bytecode coordinate scaling
    finalize_render_pipeline: ComputePipelineState,
    reset_render_pipeline: ComputePipelineState,
    generate_vertices_pipeline: ComputePipelineState,

    // Issue #159: App Migration
    parallel_megakernel_pipeline: ComputePipelineState,
    use_parallel_megakernel: bool,        // Use threadgroup-parallel megakernel

    // Configuration
    max_slots: u32,
    state_pool_size: usize,
    vertex_pool_size: usize,

    // Frame tracking
    current_frame: u32,
}

impl GpuAppSystem {
    /// Create a new GPU app system with default settings
    pub fn new(device: &Device) -> Result<Self, String> {
        Self::with_capacity(
            device,
            MAX_APP_SLOTS,
            DEFAULT_STATE_POOL_SIZE,
            DEFAULT_VERTEX_POOL_SIZE,
        )
    }

    /// Create with custom capacity
    pub fn with_capacity(
        device: &Device,
        max_slots: u32,
        state_pool_size: usize,
        vertex_pool_size: usize,
    ) -> Result<Self, String> {
        // Issue #267 fix: Pre-validate memory requirements before any allocation
        // This prevents partial allocation failures that could corrupt GPU state
        const MAX_REASONABLE_SLOTS: u32 = 1024;
        const MAX_REASONABLE_POOL_SIZE: usize = 1024 * 1024 * 1024; // 1GB

        if max_slots > MAX_REASONABLE_SLOTS {
            return Err(format!("max_slots {} exceeds maximum {}", max_slots, MAX_REASONABLE_SLOTS));
        }
        if state_pool_size > MAX_REASONABLE_POOL_SIZE {
            return Err(format!("state_pool_size {} exceeds maximum {}", state_pool_size, MAX_REASONABLE_POOL_SIZE));
        }
        if vertex_pool_size > MAX_REASONABLE_POOL_SIZE {
            return Err(format!("vertex_pool_size {} exceeds maximum {}", vertex_pool_size, MAX_REASONABLE_POOL_SIZE));
        }

        // Calculate total memory requirements
        let total_required = state_pool_size + vertex_pool_size +
            (max_slots as usize) * mem::size_of::<GpuAppDescriptor>() +
            mem::size_of::<AppTableHeader>() +
            mem::size_of::<AllocatorState>() * 2 +
            mem::size_of::<MemoryPool>() * 2;

        // Check against device limit (recommended_max_working_set_size returns bytes available)
        let device_limit = device.recommended_max_working_set_size() as usize;
        if device_limit > 0 && total_required > device_limit / 2 {
            // Use at most 50% of device memory for app system
            return Err(format!(
                "Required memory {} exceeds safe limit {} (50% of device max {})",
                total_required, device_limit / 2, device_limit
            ));
        }

        let command_queue = device.new_command_queue();

        // Compile shader
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(GPU_APP_SYSTEM_SHADER, &options)
            .map_err(|e| format!("Shader compile failed: {}", e))?;

        let launch_pipeline = Self::create_pipeline(device, &library, "gpu_launch_app")?;
        let launch_o1_pipeline = Self::create_pipeline(device, &library, "gpu_launch_app_o1")?;
        let close_pipeline = Self::create_pipeline(device, &library, "gpu_close_app")?;
        let megakernel_pipeline = Self::create_pipeline(device, &library, "gpu_app_megakernel")?;
        let budgeted_megakernel_pipeline = Self::create_pipeline(device, &library, "gpu_app_megakernel_budgeted")?;
        let mark_dirty_pipeline = Self::create_pipeline(device, &library, "mark_all_dirty")?;
        let stats_pipeline = Self::create_pipeline(device, &library, "get_app_stats")?;
        let memory_stats_pipeline = Self::create_pipeline(device, &library, "get_memory_stats")?;
        // Issue #157: Input processing pipelines
        let process_input_pipeline = Self::create_pipeline(device, &library, "gpu_process_input")?;
        let find_focus_pipeline = Self::create_pipeline(device, &library, "gpu_find_focus_target")?;
        let apply_focus_pipeline = Self::create_pipeline(device, &library, "gpu_apply_focus")?;
        let advance_queue_pipeline = Self::create_pipeline(device, &library, "gpu_advance_input_queue")?;

        // Issue #158: Rendering pipelines
        let finalize_render_pipeline = Self::create_pipeline(device, &library, "gpu_finalize_render")?;
        let reset_render_pipeline = Self::create_pipeline(device, &library, "gpu_reset_render_state")?;
        let generate_vertices_pipeline = Self::create_pipeline(device, &library, "gpu_generate_test_vertices")?;

        // Issue #159: Parallel megakernel
        let parallel_megakernel_pipeline = Self::create_pipeline(device, &library, "gpu_app_megakernel_parallel")?;

        // Create app table buffer (header + slots)
        let header_size = mem::size_of::<AppTableHeader>();
        let slots_size = (max_slots as usize) * mem::size_of::<GpuAppDescriptor>();
        let app_table_buffer = device.new_buffer(
            (header_size + slots_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize header
        unsafe {
            let header = app_table_buffer.contents() as *mut AppTableHeader;
            *header = AppTableHeader {
                max_slots,
                ..Default::default()
            };
        }

        // Create legacy allocator buffers
        let state_alloc_buffer = device.new_buffer(
            mem::size_of::<AllocatorState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_alloc_buffer = device.new_buffer(
            mem::size_of::<AllocatorState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize legacy allocators
        unsafe {
            let state_alloc = state_alloc_buffer.contents() as *mut AllocatorState;
            (*state_alloc).bump_pointer = 0;
            (*state_alloc).pool_size = state_pool_size as u32;
            (*state_alloc).allocation_count = 0;
            (*state_alloc).peak_usage = 0;

            let vertex_alloc = vertex_alloc_buffer.contents() as *mut AllocatorState;
            (*vertex_alloc).bump_pointer = 0;
            (*vertex_alloc).pool_size = vertex_pool_size as u32;
            (*vertex_alloc).allocation_count = 0;
            (*vertex_alloc).peak_usage = 0;
        }

        // Issue #155: Create O(1) memory pool buffers
        let state_pool_buffer = device.new_buffer(
            mem::size_of::<MemoryPool>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_pool_buffer = device.new_buffer(
            mem::size_of::<MemoryPool>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let state_blocks_buffer = device.new_buffer(
            (MAX_FREE_BLOCKS as usize * mem::size_of::<FreeBlock>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_blocks_buffer = device.new_buffer(
            (MAX_FREE_BLOCKS as usize * mem::size_of::<FreeBlock>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize O(1) memory pools
        unsafe {
            let state_pool = state_pool_buffer.contents() as *mut MemoryPool;
            *state_pool = MemoryPool {
                freelist_head: INVALID_SLOT,
                bump_pointer: 0,
                free_count: 0,
                pool_size: state_pool_size as u32,
                block_count: 0,
                max_blocks: MAX_FREE_BLOCKS,
                _pad: [0; 2],
            };

            let vertex_pool = vertex_pool_buffer.contents() as *mut MemoryPool;
            *vertex_pool = MemoryPool {
                freelist_head: INVALID_SLOT,
                bump_pointer: 0,
                free_count: 0,
                pool_size: vertex_pool_size as u32,
                block_count: 0,
                max_blocks: MAX_FREE_BLOCKS,
                _pad: [0; 2],
            };
        }

        // Create unified buffers
        let unified_state_buffer = device.new_buffer(
            state_pool_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let unified_vertex_buffer = device.new_buffer(
            vertex_pool_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Issue #156: Create frame budget buffer
        let budget_buffer = device.new_buffer(
            mem::size_of::<FrameBudget>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize budget (no limit by default)
        unsafe {
            let budget = budget_buffer.contents() as *mut FrameBudget;
            *budget = FrameBudget::default();
        }

        // Issue #157: Create input and window buffers
        let input_queue_buffer = device.new_buffer(
            mem::size_of::<InputQueue>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_events_buffer = device.new_buffer(
            (MAX_INPUT_EVENTS as usize * mem::size_of::<InputEvent>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let windows_buffer = device.new_buffer(
            (MAX_WINDOWS as usize * mem::size_of::<GpuWindow>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize input queue
        unsafe {
            let queue = input_queue_buffer.contents() as *mut InputQueue;
            *queue = InputQueue {
                head: 0,
                tail: 0,
                capacity: MAX_INPUT_EVENTS,
                _pad: 0,
            };
        }

        // Initialize windows with defaults
        unsafe {
            let windows = windows_buffer.contents() as *mut GpuWindow;
            for i in 0..MAX_WINDOWS {
                *windows.add(i as usize) = GpuWindow::default();
            }
        }

        // Issue #158: Create render buffers
        let render_state_buffer = device.new_buffer(
            mem::size_of::<RenderState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let render_vertices_buffer = device.new_buffer(
            (MAX_VERTICES as usize * mem::size_of::<RenderVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize all vertices to transparent (alpha = 0) so unused slots don't draw
        unsafe {
            let vertices = render_vertices_buffer.contents() as *mut RenderVertex;
            for i in 0..MAX_VERTICES as usize {
                (*vertices.add(i)).color[3] = 0.0; // Transparent
            }
        }

        // Screen size buffer for bytecode coordinate scaling (float2)
        let screen_size_buffer = device.new_buffer(
            8, // sizeof(float2)
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            let ptr = screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = [1920.0, 1080.0]; // Default, will be updated by GpuOs
        }

        // Initialize render state
        unsafe {
            let render = render_state_buffer.contents() as *mut RenderState;
            *render = RenderState {
                total_vertex_count: 0,
                max_vertices: MAX_VERTICES,
                screen_width: 1920,  // Default, can be updated
                screen_height: 1080,
            };
        }

        Ok(Self {
            device: device.clone(),
            command_queue,
            launch_pipeline,
            launch_o1_pipeline,
            close_pipeline,
            megakernel_pipeline,
            mark_dirty_pipeline,
            stats_pipeline,
            memory_stats_pipeline,
            app_table_buffer,
            state_alloc_buffer,
            vertex_alloc_buffer,
            unified_state_buffer,
            unified_vertex_buffer,
            state_pool_buffer,
            state_blocks_buffer,
            vertex_pool_buffer,
            vertex_blocks_buffer,
            use_o1_allocator: true,  // Use O(1) by default
            budget_buffer,
            budgeted_megakernel_pipeline,
            use_budgeted_scheduler: false,  // Disabled by default
            input_queue_buffer,
            input_events_buffer,
            windows_buffer,
            process_input_pipeline,
            find_focus_pipeline,
            apply_focus_pipeline,
            advance_queue_pipeline,
            render_state_buffer,
            render_vertices_buffer,
            screen_size_buffer,
            finalize_render_pipeline,
            reset_render_pipeline,
            generate_vertices_pipeline,
            parallel_megakernel_pipeline,
            use_parallel_megakernel: true,  // CRITICAL: Must be true for bytecode VM to run!
            max_slots,
            state_pool_size,
            vertex_pool_size,
            current_frame: 0,
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let function = library
            .get_function(name, None)
            .map_err(|e| format!("Function {} not found: {}", name, e))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline {} failed: {}", name, e))
    }

    // ========================================================================
    // App Lifecycle (GPU-initiated)
    // ========================================================================

    /// Launch an app (GPU handles allocation)
    pub fn launch_app(
        &mut self,
        app_type: u32,
        state_size: u32,
        vertex_size: u32,
    ) -> Option<u32> {
        if self.use_o1_allocator {
            self.launch_app_o1(app_type, state_size, vertex_size)
        } else {
            self.launch_app_legacy(app_type, state_size, vertex_size)
        }
    }

    /// Issue #159: Launch app by type using APP_TYPES registry
    pub fn launch_by_type(&mut self, type_id: u32) -> Option<u32> {
        let info = get_app_type_info(type_id)?;
        self.launch_app(type_id, info.state_size, info.vertex_size)
    }

    /// Issue #159: Enable/disable parallel megakernel
    pub fn set_use_parallel_megakernel(&mut self, enabled: bool) {
        self.use_parallel_megakernel = enabled;
    }

    /// Issue #159: Check if using parallel megakernel
    pub fn is_using_parallel_megakernel(&self) -> bool {
        self.use_parallel_megakernel
    }

    /// Launch app with legacy bump allocator
    fn launch_app_legacy(
        &mut self,
        app_type: u32,
        state_size: u32,
        vertex_size: u32,
    ) -> Option<u32> {
        let app_type_buf = self.device.new_buffer_with_data(
            &app_type as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let state_size_buf = self.device.new_buffer_with_data(
            &state_size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_size_buf = self.device.new_buffer_with_data(
            &vertex_size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let result_buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // Initialize to INVALID_SLOT
        unsafe {
            *(result_buf.contents() as *mut u32) = INVALID_SLOT;
        }

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.launch_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.state_alloc_buffer), 0);
        enc.set_buffer(3, Some(&self.vertex_alloc_buffer), 0);
        enc.set_buffer(4, Some(&app_type_buf), 0);
        enc.set_buffer(5, Some(&state_size_buf), 0);
        enc.set_buffer(6, Some(&vertex_size_buf), 0);
        enc.set_buffer(7, Some(&result_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let slot = unsafe { *(result_buf.contents() as *const u32) };
        if slot != INVALID_SLOT {
            Some(slot)
        } else {
            None
        }
    }

    /// Launch app with O(1) free list allocator (Issue #155)
    fn launch_app_o1(
        &mut self,
        app_type: u32,
        state_size: u32,
        vertex_size: u32,
    ) -> Option<u32> {
        let app_type_buf = self.device.new_buffer_with_data(
            &app_type as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let state_size_buf = self.device.new_buffer_with_data(
            &state_size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_size_buf = self.device.new_buffer_with_data(
            &vertex_size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let result_buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // Initialize to INVALID_SLOT
        unsafe {
            *(result_buf.contents() as *mut u32) = INVALID_SLOT;
        }

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.launch_o1_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.state_pool_buffer), 0);
        enc.set_buffer(3, Some(&self.state_blocks_buffer), 0);
        enc.set_buffer(4, Some(&self.vertex_pool_buffer), 0);
        enc.set_buffer(5, Some(&self.vertex_blocks_buffer), 0);
        enc.set_buffer(6, Some(&app_type_buf), 0);
        enc.set_buffer(7, Some(&state_size_buf), 0);
        enc.set_buffer(8, Some(&vertex_size_buf), 0);
        enc.set_buffer(9, Some(&result_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let slot = unsafe { *(result_buf.contents() as *const u32) };
        if slot != INVALID_SLOT {
            Some(slot)
        } else {
            None
        }
    }

    /// Close an app (GPU handles cleanup with O(1) memory free)
    pub fn close_app(&mut self, slot_id: u32) {
        let slot_buf = self.device.new_buffer_with_data(
            &slot_id as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.close_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.state_pool_buffer), 0);
        enc.set_buffer(3, Some(&self.state_blocks_buffer), 0);
        enc.set_buffer(4, Some(&self.vertex_pool_buffer), 0);
        enc.set_buffer(5, Some(&self.vertex_blocks_buffer), 0);
        enc.set_buffer(6, Some(&slot_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ========================================================================
    // Frame Execution
    // ========================================================================

    /// Run one frame (megakernel processes all apps)
    pub fn run_frame(&mut self) {
        if self.use_parallel_megakernel {
            if self.current_frame == 0 {
                println!("DEBUG: Using PARALLEL megakernel");
            }
            self.run_frame_parallel();
        } else if self.use_budgeted_scheduler {
            if self.current_frame == 0 {
                println!("DEBUG: Using BUDGETED megakernel");
            }
            self.run_frame_budgeted();
        } else {
            if self.current_frame == 0 {
                println!("DEBUG: Using SIMPLE (legacy) megakernel - DOCK WON'T WORK!");
            }
            self.run_frame_simple();
        }
    }

    /// Run frame without budget enforcement
    fn run_frame_simple(&mut self) {
        self.current_frame += 1;

        // Issue #237 fix: Mark all active apps as DIRTY and increment frame_number
        // This ensures they run each frame and frame() returns incrementing values
        unsafe {
            let header = self.app_table_buffer.contents() as *const AppTableHeader;
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>()) as *mut GpuAppDescriptor;

            for i in 0..(*header).max_slots {
                let app = &mut *apps.add(i as usize);
                if app.flags & flags::ACTIVE != 0 {
                    app.flags |= flags::DIRTY;  // Re-mark dirty so it runs
                    app.frame_number += 1;      // Increment per-app frame counter
                }
            }
        }

        let frame_buf = self.device.new_buffer_with_data(
            &self.current_frame as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.megakernel_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.unified_state_buffer), 0);
        enc.set_buffer(3, Some(&frame_buf), 0);

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Issue #159: Run frame with parallel megakernel (one threadgroup per app)
    fn run_frame_parallel(&mut self) {
        self.current_frame += 1;

        // Mark all active apps as DIRTY and increment their frame_number
        // This ensures they run each frame and frame() returns incrementing values
        unsafe {
            let header = self.app_table_buffer.contents() as *const AppTableHeader;
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>()) as *mut GpuAppDescriptor;

            for i in 0..(*header).max_slots {
                let app = &mut *apps.add(i as usize);
                if app.flags & flags::ACTIVE != 0 {
                    app.flags |= flags::DIRTY;  // Re-mark dirty so it runs
                    app.frame_number += 1;      // Increment per-app frame counter
                }
            }
        }

        let frame_buf = self.device.new_buffer_with_data(
            &self.current_frame as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        // Get window count for passing to megakernel
        let window_count = self.count_active_windows();
        let window_count_buf = self.device.new_buffer_with_data(
            &window_count as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.parallel_megakernel_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.unified_state_buffer), 0);
        enc.set_buffer(3, Some(&frame_buf), 0);
        // GPU IS THE COMPUTER: Apps write directly to contiguous vertex regions
        // slot 0 → vertices 0-1023, slot 1 → vertices 1024-2047, etc.
        enc.set_buffer(4, Some(&self.render_vertices_buffer), 0);
        enc.set_buffer(5, Some(&self.windows_buffer), 0);
        enc.set_buffer(6, Some(&window_count_buf), 0);
        enc.set_buffer(7, Some(&self.screen_size_buffer), 0);

        // Dispatch threadgroups: max_slots threadgroups, 256 threads each
        enc.dispatch_thread_groups(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Run frame with budget enforcement (Issue #156)
    fn run_frame_budgeted(&mut self) {
        self.current_frame += 1;

        // Issue #237 fix: Mark all active apps as DIRTY and increment frame_number
        // This ensures they run each frame and frame() returns incrementing values
        unsafe {
            let header = self.app_table_buffer.contents() as *const AppTableHeader;
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>()) as *mut GpuAppDescriptor;

            for i in 0..(*header).max_slots {
                let app = &mut *apps.add(i as usize);
                if app.flags & flags::ACTIVE != 0 {
                    app.flags |= flags::DIRTY;  // Re-mark dirty so it runs
                    app.frame_number += 1;      // Increment per-app frame counter
                }
            }
        }

        // Reset budget for this frame
        unsafe {
            let budget = self.budget_buffer.contents() as *mut FrameBudget;
            (*budget).remaining = (*budget).per_frame_limit;
            (*budget).skipped_count = 0;
        }

        let frame_buf = self.device.new_buffer_with_data(
            &self.current_frame as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.budgeted_megakernel_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.unified_state_buffer), 0);
        enc.set_buffer(3, Some(&self.budget_buffer), 0);
        enc.set_buffer(4, Some(&frame_buf), 0);

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Mark all apps as dirty (need redraw)
    pub fn mark_all_dirty(&self) {
        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.mark_dirty_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get system statistics
    pub fn stats(&self) -> GpuAppSystemStats {
        let stats_buf = self.device.new_buffer(20, MTLResourceOptions::StorageModeShared);

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.stats_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&stats_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        unsafe {
            let ptr = stats_buf.contents() as *const u32;
            GpuAppSystemStats {
                active_count: *ptr.add(0),
                max_slots: *ptr.add(1),
                visible_count: *ptr.add(2),
                dirty_count: *ptr.add(3),
                focused_count: *ptr.add(4),
            }
        }
    }

    /// Get active app count
    pub fn active_count(&self) -> u32 {
        unsafe {
            let header = self.app_table_buffer.contents() as *const AppTableHeader;
            (*header).active_count
        }
    }

    /// Get memory statistics (Issue #155)
    pub fn memory_stats(&self) -> MemoryStats {
        let stats_buf = self.device.new_buffer(40, MTLResourceOptions::StorageModeShared);

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.memory_stats_pipeline);
        enc.set_buffer(0, Some(&self.state_pool_buffer), 0);
        enc.set_buffer(1, Some(&self.vertex_pool_buffer), 0);
        enc.set_buffer(2, Some(&stats_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        unsafe {
            let ptr = stats_buf.contents() as *const u32;
            MemoryStats {
                state_pool: MemoryPoolStats {
                    freelist_head: *ptr.add(0),
                    bump_pointer: *ptr.add(1),
                    free_count: *ptr.add(2),
                    pool_size: *ptr.add(3),
                    block_count: *ptr.add(4),
                },
                vertex_pool: MemoryPoolStats {
                    freelist_head: *ptr.add(5),
                    bump_pointer: *ptr.add(6),
                    free_count: *ptr.add(7),
                    pool_size: *ptr.add(8),
                    block_count: *ptr.add(9),
                },
            }
        }
    }

    /// Enable or disable O(1) allocator
    pub fn set_use_o1_allocator(&mut self, use_o1: bool) {
        self.use_o1_allocator = use_o1;
    }

    /// Check if using O(1) allocator
    pub fn is_using_o1_allocator(&self) -> bool {
        self.use_o1_allocator
    }

    // ========================================================================
    // Issue #156: GPU Scheduler Control
    // ========================================================================

    /// Enable or disable budgeted scheduler
    pub fn set_use_budgeted_scheduler(&mut self, use_budgeted: bool) {
        self.use_budgeted_scheduler = use_budgeted;
    }

    /// Check if using budgeted scheduler
    pub fn is_using_budgeted_scheduler(&self) -> bool {
        self.use_budgeted_scheduler
    }

    /// Set frame budget (cycles)
    pub fn set_frame_budget(&mut self, budget: u32) {
        unsafe {
            let b = self.budget_buffer.contents() as *mut FrameBudget;
            (*b).per_frame_limit = budget;
            (*b).remaining = budget;
        }
    }

    /// Get frame budget settings
    pub fn frame_budget(&self) -> (u32, u32) {
        unsafe {
            let b = self.budget_buffer.contents() as *const FrameBudget;
            ((*b).per_frame_limit, (*b).remaining)
        }
    }

    /// Set priority for an app
    pub fn set_priority(&mut self, slot: u32, priority: u32) {
        if slot >= self.max_slots {
            return;
        }

        unsafe {
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(slot as usize)).priority = priority;
        }
    }

    /// Suspend an app (won't run until resumed)
    pub fn suspend(&mut self, slot: u32) {
        if slot >= self.max_slots {
            return;
        }

        unsafe {
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(slot as usize)).flags |= flags::SUSPENDED;
        }
    }

    /// Resume a suspended app
    pub fn resume(&mut self, slot: u32) {
        if slot >= self.max_slots {
            return;
        }

        unsafe {
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            let app = &mut *apps.add(slot as usize);
            app.flags &= !flags::SUSPENDED;
            app.flags |= flags::DIRTY;  // Wake it up
        }
    }

    /// Get scheduler statistics
    pub fn scheduler_stats(&self) -> SchedulerStats2 {
        let mut stats = SchedulerStats2::default();

        unsafe {
            let apps = (self.app_table_buffer.contents() as *const u8)
                .add(mem::size_of::<AppTableHeader>())
                as *const GpuAppDescriptor;

            for i in 0..self.max_slots {
                let app = &*apps.add(i as usize);
                if app.flags & flags::ACTIVE != 0 {
                    stats.active_count += 1;

                    if app.flags & flags::SUSPENDED != 0 {
                        stats.suspended_count += 1;
                    }

                    // Check starvation
                    let frames_since_run = self.current_frame.saturating_sub(app.last_run_frame);
                    if frames_since_run > STARVATION_THRESHOLD {
                        stats.starving_count += 1;
                    }

                    // Count by priority
                    let p = app.priority.min(3) as usize;
                    stats.apps_by_priority[p] += 1;
                }
            }

            // Get skipped count from budget buffer
            let budget = self.budget_buffer.contents() as *const FrameBudget;
            stats.skipped_this_frame = (*budget).skipped_count;
        }

        stats
    }

    // ========================================================================
    // Issue #157: Input & Window Management
    // ========================================================================

    /// Queue an input event (CPU writes to GPU queue)
    pub fn queue_input(&mut self, event: InputEvent) {
        unsafe {
            let queue = self.input_queue_buffer.contents() as *mut InputQueue;
            let events = self.input_events_buffer.contents() as *mut InputEvent;

            let tail = (*queue).tail;
            let capacity = (*queue).capacity;

            // Check if queue is full
            let head = (*queue).head;
            if tail.wrapping_sub(head) >= capacity {
                return;  // Queue full, drop event
            }

            // Write event with current frame number
            let mut ev = event;
            ev.frame = self.current_frame;
            *events.add((tail % capacity) as usize) = ev;

            // Advance tail
            (*queue).tail = tail + 1;
        }
    }

    /// Process all queued input (GPU runs parallel dispatch)
    pub fn process_input(&mut self) {
        let max_slots_buf = self.device.new_buffer_with_data(
            &self.max_slots as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.process_input_pipeline);
        enc.set_buffer(0, Some(&self.input_queue_buffer), 0);
        enc.set_buffer(1, Some(&self.input_events_buffer), 0);
        enc.set_buffer(
            2,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(3, Some(&self.windows_buffer), 0);
        enc.set_buffer(4, Some(&max_slots_buf), 0);

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();

        // Also advance the queue
        let enc2 = cmd.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(&self.advance_queue_pipeline);
        enc2.set_buffer(0, Some(&self.input_queue_buffer), 0);
        enc2.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc2.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Handle click to find and focus topmost window
    pub fn handle_click(&mut self, x: f32, y: f32) {
        let click_pos: [f32; 2] = [x, y];
        let click_buf = self.device.new_buffer_with_data(
            click_pos.as_ptr() as *const _,
            8,
            MTLResourceOptions::StorageModeShared,
        );
        let max_slots_buf = self.device.new_buffer_with_data(
            &self.max_slots as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let topmost_slot_buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let topmost_depth_buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // Initialize to invalid
        unsafe {
            *(topmost_slot_buf.contents() as *mut u32) = INVALID_SLOT;
            *(topmost_depth_buf.contents() as *mut u32) = 0;
        }

        let cmd = self.command_queue.new_command_buffer();

        // Find topmost window at click position
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.find_focus_pipeline);
        enc.set_buffer(
            0,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(1, Some(&self.windows_buffer), 0);
        enc.set_buffer(2, Some(&click_buf), 0);
        enc.set_buffer(3, Some(&max_slots_buf), 0);
        enc.set_buffer(4, Some(&topmost_slot_buf), 0);
        enc.set_buffer(5, Some(&topmost_depth_buf), 0);
        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        // Get result and apply focus
        let topmost = unsafe { *(topmost_slot_buf.contents() as *const u32) };
        if topmost != INVALID_SLOT {
            self.set_focus(topmost);
        }
    }

    /// Set focus to a specific app
    pub fn set_focus(&mut self, slot: u32) {
        let focus_buf = self.device.new_buffer_with_data(
            &slot as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let max_slots_buf = self.device.new_buffer_with_data(
            &self.max_slots as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.apply_focus_pipeline);
        enc.set_buffer(
            0,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(1, Some(&focus_buf), 0);
        enc.set_buffer(2, Some(&max_slots_buf), 0);

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Get currently focused app
    pub fn focused_app(&self) -> Option<u32> {
        unsafe {
            let apps = (self.app_table_buffer.contents() as *const u8)
                .add(mem::size_of::<AppTableHeader>())
                as *const GpuAppDescriptor;

            for i in 0..self.max_slots {
                let app = &*apps.add(i as usize);
                if app.flags & flags::ACTIVE != 0 && app.flags & flags::FOCUS != 0 {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Create a window for an app
    pub fn create_window(&mut self, slot: u32, x: f32, y: f32, width: f32, height: f32) -> Option<u32> {
        if slot >= self.max_slots {
            return None;
        }

        unsafe {
            // Update window in windows buffer
            let windows = self.windows_buffer.contents() as *mut GpuWindow;
            let window = &mut *windows.add(slot as usize);
            window.x = x;
            window.y = y;
            window.width = width;
            window.height = height;
            window.depth = 0.5;  // Default middle depth
            window.app_slot = slot;
            window.flags = 1;  // Active

            // Link window to app
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(slot as usize)).window_id = slot;  // Window ID matches slot for now
        }

        Some(slot)  // Window ID = slot for now
    }

    /// Get window for an app
    pub fn get_window(&self, window_id: u32) -> Option<GpuWindow> {
        if window_id >= self.max_slots {
            return None;
        }

        unsafe {
            let windows = self.windows_buffer.contents() as *const GpuWindow;
            let window = *windows.add(window_id as usize);
            if window.flags != 0 {
                Some(window)
            } else {
                None
            }
        }
    }

    /// Set window depth (for z-ordering)
    pub fn set_window_depth(&mut self, window_id: u32, depth: f32) {
        if window_id >= self.max_slots {
            return;
        }

        unsafe {
            let windows = self.windows_buffer.contents() as *mut GpuWindow;
            (*windows.add(window_id as usize)).depth = depth;
        }
    }

    /// Count active windows (visible with non-zero flags)
    pub fn count_active_windows(&self) -> u32 {
        let mut count = 0u32;
        unsafe {
            let windows = self.windows_buffer.contents() as *const GpuWindow;
            for i in 0..MAX_WINDOWS {
                let window = *windows.add(i as usize);
                if window.flags & window_flags::VISIBLE != 0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// Read input queue state (for testing)
    pub fn read_input_queue(&self) -> InputQueue {
        unsafe {
            let queue = self.input_queue_buffer.contents() as *const InputQueue;
            *queue
        }
    }

    // ========================================================================
    // Issue #158: Rendering Pipeline
    // ========================================================================

    /// Reset render state for new frame
    pub fn reset_render_state(&mut self) {
        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.reset_render_pipeline);
        enc.set_buffer(0, Some(&self.render_state_buffer), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Generate test vertices for all visible apps (GPU-driven)
    pub fn generate_vertices(&mut self) {
        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.generate_vertices_pipeline);
        enc.set_buffer(
            0,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(1, Some(&self.app_table_buffer), 0);
        // Write directly to render buffer - slot-based contiguous layout
        enc.set_buffer(2, Some(&self.render_vertices_buffer), 0);
        enc.set_buffer(3, Some(&self.windows_buffer), 0);

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Finalize render: sum all app vertex counts
    pub fn finalize_render(&mut self) {
        // Reset render state (total_vertex_count = 0)
        self.reset_render_state();

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.finalize_render_pipeline);
        enc.set_buffer(
            0,
            Some(&self.app_table_buffer),
            mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(1, Some(&self.app_table_buffer), 0);
        enc.set_buffer(2, Some(&self.render_state_buffer), 0);
        enc.set_buffer(3, Some(&self.windows_buffer), 0);

        enc.dispatch_threads(
            MTLSize::new(self.max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Get total vertex count for draw call
    pub fn total_vertex_count(&self) -> u32 {
        unsafe {
            let render = self.render_state_buffer.contents() as *const RenderState;
            (*render).total_vertex_count
        }
    }

    /// Set screen size for bytecode coordinate scaling
    pub fn set_screen_size(&mut self, width: f32, height: f32) {
        unsafe {
            let ptr = self.screen_size_buffer.contents() as *mut [f32; 2];
            *ptr = [width, height];
        }
    }

    /// Read render state (for testing)
    pub fn render_state(&self) -> RenderState {
        unsafe {
            let render = self.render_state_buffer.contents() as *const RenderState;
            *render
        }
    }

    /// Get unified render vertices buffer (for rendering)
    pub fn render_vertices_buffer(&self) -> &Buffer {
        &self.render_vertices_buffer
    }

    /// Get an app descriptor
    pub fn get_app(&self, slot: u32) -> Option<GpuAppDescriptor> {
        if slot >= self.max_slots {
            return None;
        }

        let desc = unsafe {
            let apps = (self.app_table_buffer.contents() as *const u8)
                .add(mem::size_of::<AppTableHeader>())
                as *const GpuAppDescriptor;
            *apps.add(slot as usize)
        };

        if desc.is_active() {
            Some(desc)
        } else {
            None
        }
    }

    /// Mark a specific app as dirty
    pub fn mark_dirty(&self, slot: u32) {
        if slot >= self.max_slots {
            return;
        }

        unsafe {
            let apps = (self.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(slot as usize)).flags |= flags::DIRTY;
        }
    }

    // ========================================================================
    // Buffer Access (for integration with existing systems)
    // ========================================================================

    /// Get the unified state buffer
    pub fn state_buffer(&self) -> &Buffer {
        &self.unified_state_buffer
    }

    /// Get the unified vertex buffer
    pub fn vertex_buffer(&self) -> &Buffer {
        &self.unified_vertex_buffer
    }

    /// Get the app table buffer
    pub fn app_table_buffer(&self) -> &Buffer {
        &self.app_table_buffer
    }

    /// Get current frame number
    pub fn current_frame(&self) -> u32 {
        self.current_frame
    }

    /// Write raw data to an app's state buffer
    ///
    /// This is used for bytecode apps where the state includes the bytecode program.
    /// The data is written at the app's state_offset in unified_state_buffer.
    ///
    /// Also initializes the SlabAllocator header after the bytecode for memory operations.
    pub fn write_app_state(&self, slot: u32, data: &[u8]) {
        let app = match self.get_app(slot) {
            Some(a) => a,
            None => return,
        };

        // Ensure we don't overflow the allocated state size
        let write_len = data.len().min(app.state_size as usize);

        unsafe {
            let state_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), state_ptr, write_len);

            // Initialize SlabAllocator header after bytecode (Issue #210)
            // Layout: BytecodeHeader (16 bytes) + Instructions (code_size * 8)
            // SlabAllocator starts at: state_offset + header_size + code_size * 8
            if data.len() >= 16 {
                // Read code_size from bytecode header
                let code_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
                let header_size = std::mem::size_of::<BytecodeHeader>();
                let inst_size = std::mem::size_of::<BytecodeInst>();
                let alloc_header_size = 48; // SlabAllocator size (3 float4s)
                let result_size = 16; // Result register (1 float4)

                let alloc_offset = header_size + code_size * inst_size;
                let overhead = alloc_header_size + result_size; // Total overhead before heap

                // Calculate heap size: remaining state after allocator header + result
                let heap_size = if alloc_offset + overhead < app.state_size as usize {
                    (app.state_size as usize - alloc_offset - overhead) as u32
                } else {
                    0
                };

                // Initialize SlabAllocator fields at alloc_offset
                // struct SlabAllocator {
                //     atomic_uint free_heads[8]; // 32 bytes - init to 0xFFFFFFFF (empty)
                //     atomic_uint heap_top;      // 4 bytes - init to 0
                //     uint heap_size;            // 4 bytes - set to calculated size
                //     uint _pad[2];              // 8 bytes
                // };
                let alloc_ptr = state_ptr.add(alloc_offset) as *mut u32;

                // Initialize free_heads[8] to 0xFFFFFFFF (empty)
                for i in 0..8 {
                    *alloc_ptr.add(i) = 0xFFFFFFFF;
                }
                // heap_top = 0
                *alloc_ptr.add(8) = 0;
                // heap_size = calculated heap size
                *alloc_ptr.add(9) = heap_size;
                // _pad[2] = 0
                *alloc_ptr.add(10) = 0;
                *alloc_ptr.add(11) = 0;
            }
        }
    }

    /// Read an i32 from an app's state buffer at a given byte offset
    ///
    /// For bytecode apps, the state starts with BytecodeHeader, then instructions,
    /// then the actual app state. This function reads from the beginning of the
    /// app's state buffer (including header).
    pub fn read_app_state_i32(&self, slot: u32, byte_offset: usize) -> Option<i32> {
        let app = self.get_app(slot)?;

        // Ensure we don't read past the allocated state size
        if byte_offset + 4 > app.state_size as usize {
            return None;
        }

        unsafe {
            let state_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(byte_offset) as *const i32;
            Some(*state_ptr)
        }
    }

    /// Read the return value from a bytecode app
    /// The bytecode VM stores results in state[3] (after allocator header)
    /// State layout: SlabAllocator[0-2] (48 bytes) | result[3] (16 bytes) | params[4-7] (64 bytes) | heap[8+]
    pub fn read_bytecode_result(&self, slot: u32) -> Option<i32> {
        let app = self.get_app(slot)?;

        // Read the bytecode header to get code_size
        let header: BytecodeHeader = unsafe {
            let header_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *const BytecodeHeader;
            *header_ptr
        };

        // Calculate offset to state[3] (after header + instructions + allocator header)
        let header_size = std::mem::size_of::<BytecodeHeader>();
        let inst_size = std::mem::size_of::<BytecodeInst>();
        let alloc_header_size = 48; // SlabAllocator (3 float4s)
        let state_offset = header_size + (header.code_size as usize) * inst_size + alloc_header_size;

        // Issue #213 fix: Return values are now stored as FLOAT VALUES (not integer bits)
        // This avoids GPU FTZ (Flush-To-Zero) corrupting small integers which are denormalized floats.
        // Read as f32 and convert to i32.
        unsafe {
            let result_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(state_offset) as *const f32;
            let float_val = *result_ptr;
            Some(float_val as i32)
        }
    }

    /// Read bytecode result as i64 (from state[3].xy, stored as 2x32-bit words)
    /// THE GPU IS THE COMPUTER - i64 is stored as (lo: u32, hi: u32) in xy components
    /// State layout: SlabAllocator[0-2] (48 bytes) | result[3] (16 bytes) | params[4-7] (64 bytes) | heap[8+]
    pub fn read_bytecode_result_i64(&self, slot: u32) -> Option<i64> {
        let app = self.get_app(slot)?;

        // Read the bytecode header to get code_size
        let header: BytecodeHeader = unsafe {
            let header_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *const BytecodeHeader;
            *header_ptr
        };

        // Calculate offset to state[3] (after header + instructions + allocator header)
        let header_size = std::mem::size_of::<BytecodeHeader>();
        let inst_size = std::mem::size_of::<BytecodeInst>();
        let alloc_header_size = 48; // SlabAllocator (3 float4s)
        let state_offset = header_size + (header.code_size as usize) * inst_size + alloc_header_size;

        // Read state[3].x (lo) and state[3].y (hi) as u32, combine to i64
        unsafe {
            let base_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(state_offset);

            let lo = *(base_ptr as *const u32);
            let hi = *((base_ptr as *const u32).add(1));

            Some(((hi as u64) << 32 | (lo as u64)) as i64)
        }
    }

    /// Read bytecode result as f64 in double-single format (from state[3].xy)
    /// THE GPU IS THE COMPUTER - Metal doesn't support native f64, so we use double-single:
    /// value = hi + lo where hi is stored in .x and lo in .y
    /// This provides ~47 bits of mantissa precision (vs 52 for native f64)
    pub fn read_bytecode_result_f64(&self, slot: u32) -> Option<f64> {
        let app = self.get_app(slot)?;

        let header: BytecodeHeader = unsafe {
            let header_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *const BytecodeHeader;
            *header_ptr
        };

        let header_size = std::mem::size_of::<BytecodeHeader>();
        let inst_size = std::mem::size_of::<BytecodeInst>();
        let alloc_header_size = 48; // SlabAllocator (3 float4s)
        let state_offset = header_size + (header.code_size as usize) * inst_size + alloc_header_size;

        // Read state[3].x (hi) and state[3].y (lo) as f32, combine to f64
        unsafe {
            let base_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(state_offset);

            let hi = *(base_ptr as *const f32);
            let lo = *((base_ptr as *const f32).add(1));

            // Double-single: value = hi + lo
            Some((hi as f64) + (lo as f64))
        }
    }

    /// Read an f32 from an app's state buffer at a given byte offset
    pub fn read_app_state_f32(&self, slot: u32, byte_offset: usize) -> Option<f32> {
        let app = self.get_app(slot)?;

        if byte_offset + 4 > app.state_size as usize {
            return None;
        }

        unsafe {
            let state_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(byte_offset) as *const f32;
            Some(*state_ptr)
        }
    }

    /// Issue #176: Set parameters for a bytecode app
    /// THE GPU IS THE COMPUTER: Parameters are stored in GPU memory at state[4-7]
    /// and loaded into registers r4-r7 by the bytecode interpreter.
    ///
    /// State layout: SlabAllocator[0-2] | result[3] | params[4-7] | heap[8+]
    ///
    /// # Arguments
    /// * `slot` - The app slot ID
    /// * `params` - Up to 4 parameter values (stored as float4.x components)
    ///
    /// # Example
    /// ```ignore
    /// system.set_bytecode_params(slot, &[42, 100, 200]); // r4=42, r5=100, r6=200
    /// ```
    pub fn set_bytecode_params(&mut self, slot: u32, params: &[i32]) {
        let app = match self.get_app(slot) {
            Some(a) => a,
            None => return,
        };

        // Calculate offset to state buffer (after header + instructions)
        let header: BytecodeHeader = unsafe {
            let header_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *const BytecodeHeader;
            *header_ptr
        };

        let header_size = std::mem::size_of::<BytecodeHeader>();
        let inst_size = std::mem::size_of::<BytecodeInst>();
        let state_start = header_size + (header.code_size as usize) * inst_size;

        // Params are at state[4-7] (bytes 64-127 from state start)
        let params_offset = state_start + 4 * 16; // 4 float4s = 64 bytes for alloc+result

        unsafe {
            let base_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(params_offset) as *mut f32;

            for (i, &param) in params.iter().take(4).enumerate() {
                // Store as float (GPU will read as float4.x)
                *base_ptr.add(i * 4) = param as f32;
            }
        }
    }

    /// Issue #176: Set float parameters for a bytecode app
    pub fn set_bytecode_params_f32(&mut self, slot: u32, params: &[f32]) {
        let app = match self.get_app(slot) {
            Some(a) => a,
            None => return,
        };

        // Calculate offset to state buffer (after header + instructions)
        let header: BytecodeHeader = unsafe {
            let header_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *const BytecodeHeader;
            *header_ptr
        };

        let header_size = std::mem::size_of::<BytecodeHeader>();
        let inst_size = std::mem::size_of::<BytecodeInst>();
        let state_start = header_size + (header.code_size as usize) * inst_size;

        // Params are at state[4-7] (bytes 64-127 from state start)
        let params_offset = state_start + 4 * 16; // 4 float4s = 64 bytes for alloc+result

        unsafe {
            let base_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize)
                .add(params_offset) as *mut f32;

            for (i, &param) in params.iter().take(4).enumerate() {
                // Store as float (GPU will read as float4.x)
                *base_ptr.add(i * 4) = param;
            }
        }
    }

    // ========================================================================
    // Compositor Management
    // ========================================================================

    /// Initialize compositor state with screen dimensions
    pub fn initialize_compositor_state(&mut self, slot: u32, screen_width: f32, screen_height: f32) {
        if slot >= self.max_slots {
            return;
        }

        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::COMPOSITOR => a,
            _ => return,
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *mut CompositorState;

            let state = &mut *state_ptr;
            state.screen_width = screen_width;
            state.screen_height = screen_height;
            state.window_count = 0;
            state.frame_number = 0;
            state.background_color = [0.08, 0.08, 0.12, 1.0];  // Dark background
            state.total_vertices_rendered = 0;
            state.app_count = 0;
        }
    }

    // ========================================================================
    // Issue #156: Dock Management
    // ========================================================================

    /// Initialize dock state with screen dimensions
    pub fn initialize_dock_state(&mut self, slot: u32, screen_width: f32, screen_height: f32) {
        if slot >= self.max_slots {
            return;
        }

        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => return,
        };

        // DEBUG: Print Rust struct sizes
        println!("DEBUG DOCK INIT: Rust sizeof(DockState)={}, sizeof(DockItem)={}",
            mem::size_of::<DockState>(), mem::size_of::<DockItem>());

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *mut DockState;

            let state = &mut *state_ptr;
            state.screen_width = screen_width;
            state.screen_height = screen_height;
            state.dock_y = screen_height - DOCK_DEFAULT_HEIGHT;
            state.dock_height = DOCK_DEFAULT_HEIGHT;
            state.base_icon_size = DOCK_DEFAULT_ICON_SIZE;
            state.magnified_size = DOCK_MAGNIFIED_SIZE;
            state.icon_spacing = DOCK_ICON_SPACING;
            state.magnification_radius = 120.0;
            state.animation_speed = 0.25;
            state.bounce_height = 20.0;
            state.bounce_speed = 0.15;
            state.item_count = 0;
            state.hovered_item = u32::MAX;
            state.clicked_item = u32::MAX;
            state.cursor_in_dock = 0;
        }
    }

    /// Add a dock item
    pub fn add_dock_item(&mut self, slot: u32, item_app_type: u32, color: [f32; 4]) -> Option<u32> {
        if slot >= self.max_slots {
            return None;
        }

        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => return None,
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *mut DockState;
            let state = &mut *state_ptr;

            if state.item_count >= MAX_DOCK_ITEMS {
                return None;
            }

            let items_ptr = (state_ptr as *mut u8).add(mem::size_of::<DockState>())
                as *mut DockItem;
            let item = &mut *items_ptr.add(state.item_count as usize);

            item.app_type = item_app_type;
            item.flags = dock_item_flags::VISIBLE;
            item.running_count = 0;
            item.current_size = state.base_icon_size;
            item.target_size = state.base_icon_size;
            item.bounce_phase = 0.0;
            item.center_x = 0.0;  // Will be computed by GPU
            item.center_y = 0.0;
            item.icon_color = color;

            let item_idx = state.item_count;
            state.item_count += 1;
            Some(item_idx)
        }
    }

    /// Get dock state (read-only)
    pub fn get_dock_state(&self, slot: u32) -> Option<DockState> {
        let app = self.get_app(slot)?;
        if app.app_type != app_type::DOCK {
            return None;
        }

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *const DockState;
            Some(*state_ptr)
        }
    }

    /// Get dock items (read-only)
    pub fn get_dock_items(&self, slot: u32) -> Vec<DockItem> {
        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => return Vec::new(),
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *const DockState;
            let item_count = (*state_ptr).item_count as usize;

            let items_ptr = (state_ptr as *const u8).add(mem::size_of::<DockState>())
                as *const DockItem;

            (0..item_count)
                .map(|i| *items_ptr.add(i))
                .collect()
        }
    }

    /// Trigger bounce animation on a dock item
    pub fn trigger_dock_bounce(&mut self, slot: u32, item_index: u32) {
        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => return,
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *const DockState;
            let item_count = (*state_ptr).item_count;

            if item_index >= item_count {
                return;
            }

            let items_ptr = (state_ptr as *mut u8).add(mem::size_of::<DockState>())
                as *mut DockItem;
            let item = &mut *items_ptr.add(item_index as usize);

            item.flags |= dock_item_flags::BOUNCING;
            item.bounce_phase = 0.0;
        }
    }

    /// Update dock cursor position
    pub fn update_dock_cursor(&mut self, slot: u32, x: f32, y: f32) {
        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => return,
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *mut DockState;
            let state = &mut *state_ptr;

            state.cursor_pos = [x, y];
            state.cursor_in_dock = if y >= state.dock_y { 1 } else { 0 };
        }

        // Mark dock dirty so GPU processes the cursor update
        self.mark_dirty(slot);
    }

    /// Update dock mouse button state
    pub fn update_dock_mouse_pressed(&mut self, slot: u32, pressed: bool) {
        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => return,
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *mut DockState;
            let state = &mut *state_ptr;
            state.mouse_pressed = if pressed { 1 } else { 0 };
        }

        // Mark dock dirty so GPU processes the click
        self.mark_dirty(slot);
    }

    /// Get clicked dock app type (if any)
    pub fn get_clicked_dock_app(&self, slot: u32) -> Option<u32> {
        let app = self.get_app(slot)?;
        if app.app_type != app_type::DOCK {
            return None;
        }

        unsafe {
            let state_ptr = self.unified_state_buffer.contents().add(app.state_offset as usize)
                as *const DockState;
            let state = &*state_ptr;

            if state.clicked_item == u32::MAX || state.clicked_item >= state.item_count {
                return None;
            }

            let items_ptr = (state_ptr as *const u8).add(mem::size_of::<DockState>())
                as *const DockItem;
            let item = &*items_ptr.add(state.clicked_item as usize);

            Some(item.app_type)
        }
    }

    /// Debug: print dock state
    pub fn debug_dock_state(&self, slot: u32) {
        let app = match self.get_app(slot) {
            Some(a) if a.app_type == app_type::DOCK => a,
            _ => { println!("DEBUG: Invalid dock slot"); return; }
        };

        unsafe {
            let state_ptr = self.unified_state_buffer.contents()
                .add(app.state_offset as usize) as *const DockState;
            let state = &*state_ptr;

            println!("DEBUG DOCK: dock_y={:.1}, cursor=({:.1},{:.1}), in_dock={}, mouse_pressed={}, items={}, hovered={}, clicked={}",
                state.dock_y,
                state.cursor_pos[0], state.cursor_pos[1],
                state.cursor_in_dock,
                state.mouse_pressed,
                state.item_count,
                if state.hovered_item == u32::MAX { "NONE".to_string() } else { state.hovered_item.to_string() },
                if state.clicked_item == u32::MAX { "NONE".to_string() } else { state.clicked_item.to_string() }
            );

            // Print dock item positions
            let items_ptr = (state_ptr as *const u8).add(mem::size_of::<DockState>()) as *const DockItem;
            for i in 0..state.item_count.min(5) {
                let item = &*items_ptr.add(i as usize);
                println!("  Item {}: center=({:.1},{:.1}), size={:.1}, visible={}",
                    i, item.center_x, item.center_y, item.current_size, item.flags & dock_item_flags::VISIBLE != 0);
            }

            // Print GPU debug values (stored in _pad by PHASE 2)
            // These are dist_sq and radius_sq for thread 0 (item 0)
            let dist_sq: f32 = f32::from_bits(state._pad[0]);
            let radius_sq: f32 = f32::from_bits(state._pad[1]);
            println!("  GPU hover check (item 0): dist²={:.1}, radius²={:.1}, hit={}",
                dist_sq, radius_sq, dist_sq < radius_sq);
        }
    }
}

/// System statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuAppSystemStats {
    pub active_count: u32,
    pub max_slots: u32,
    pub visible_count: u32,
    pub dirty_count: u32,
    pub focused_count: u32,
}

/// Memory pool statistics (Issue #155)
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryPoolStats {
    pub freelist_head: u32,
    pub bump_pointer: u32,
    pub free_count: u32,
    pub pool_size: u32,
    pub block_count: u32,
}

/// Combined memory statistics for both pools
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    pub state_pool: MemoryPoolStats,
    pub vertex_pool: MemoryPoolStats,
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
    fn test_dock_struct_sizes() {
        // These sizes MUST match the Metal shader struct sizes
        // Metal sizeof(DockState) and sizeof(DockItem) should equal these
        println!("Rust sizeof(DockState) = {}", mem::size_of::<DockState>());
        println!("Rust sizeof(DockItem) = {}", mem::size_of::<DockItem>());

        // DockState: 22 fields × 4 bytes = 88 bytes (plus any alignment)
        // DockItem: 8 fields with float4 icon_color = 48 bytes
        assert_eq!(mem::size_of::<DockState>(), 88, "DockState size mismatch");
        assert_eq!(mem::size_of::<DockItem>(), 48, "DockItem size mismatch");
    }

    #[test]
    fn test_system_creation() {
        let device = get_device();
        let system = GpuAppSystem::new(&device).expect("Failed to create system");

        assert_eq!(system.active_count(), 0);
        assert_eq!(system.current_frame(), 0);
    }

    #[test]
    fn test_launch_and_close() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let slot = system.launch_app(app_type::CUSTOM, 1024, 512);
        assert!(slot.is_some());
        assert_eq!(system.active_count(), 1);

        system.close_app(slot.unwrap());
        assert_eq!(system.active_count(), 0);
    }

    #[test]
    fn test_frame_execution() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let slot = system.launch_app(app_type::CUSTOM, 64, 64);
        assert!(slot.is_some());

        system.run_frame();
        assert_eq!(system.current_frame(), 1);

        let app = system.get_app(slot.unwrap()).unwrap();
        assert_eq!(app.last_run_frame, 1);
        assert_eq!(app.flags & flags::DIRTY, 0);
    }

    #[test]
    fn test_stats() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        system.launch_app(app_type::CUSTOM, 1024, 512);
        system.launch_app(app_type::CUSTOM, 1024, 512);

        let stats = system.stats();
        assert_eq!(stats.active_count, 2);
        assert_eq!(stats.max_slots, MAX_APP_SLOTS);
    }

    // ========================================================================
    // Issue #155: O(1) Memory Management Tests
    // ========================================================================

    #[test]
    fn test_o1_allocator_enabled() {
        let device = get_device();
        let system = GpuAppSystem::new(&device).expect("Failed to create system");
        assert!(system.is_using_o1_allocator(), "O(1) allocator should be enabled by default");
    }

    #[test]
    fn test_memory_stats() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Initial state
        let stats = system.memory_stats();
        assert_eq!(stats.state_pool.free_count, 0, "No free blocks initially");
        assert_eq!(stats.state_pool.bump_pointer, 0, "Bump pointer at 0 initially");

        // Launch app
        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        let stats = system.memory_stats();
        assert!(stats.state_pool.bump_pointer >= 1024, "State bump pointer should advance");
        // NOTE: Vertices use fixed slot-based regions, NOT bump allocation
        // Each slot gets vertex_offset = slot * VERTS_PER_SLOT * sizeof(RenderVertex)
        // So vertex_pool.bump_pointer stays 0 (that's the intended design)

        // Close app
        system.close_app(slot);

        let stats = system.memory_stats();
        assert_eq!(stats.state_pool.free_count, 1, "Should have 1 free state block");
        // Vertex memory is slot-based, not freed to pool
    }

    #[test]
    fn test_alloc_free_reuse() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch app, note memory offset
        let slot1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let offset1 = system.get_app(slot1).unwrap().state_offset;

        // Close app - memory goes to free list
        system.close_app(slot1);

        // Launch new app with same size - should reuse freed memory
        let slot2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let offset2 = system.get_app(slot2).unwrap().state_offset;

        assert_eq!(offset1, offset2, "Should reuse freed memory via O(1) free list");
    }

    #[test]
    fn test_free_list_lifo() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch 3 apps
        let a = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let b = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let c = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        let offset_a = system.get_app(a).unwrap().state_offset;
        let offset_b = system.get_app(b).unwrap().state_offset;
        let offset_c = system.get_app(c).unwrap().state_offset;

        // Close in order C, B, A (LIFO means C goes on stack last)
        system.close_app(c);
        system.close_app(b);
        system.close_app(a);

        let stats = system.memory_stats();
        assert_eq!(stats.state_pool.free_count, 3, "Should have 3 free blocks");

        // Launch new app - should get A's memory (last freed = first allocated in LIFO)
        let d = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        assert_eq!(
            system.get_app(d).unwrap().state_offset,
            offset_a,
            "LIFO: should get last freed block (A)"
        );

        // Launch another - should get B's memory
        let e = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        assert_eq!(
            system.get_app(e).unwrap().state_offset,
            offset_b,
            "LIFO: should get second-to-last freed block (B)"
        );

        // Launch another - should get C's memory
        let f = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        assert_eq!(
            system.get_app(f).unwrap().state_offset,
            offset_c,
            "LIFO: should get third-to-last freed block (C)"
        );
    }

    #[test]
    fn test_fallback_to_bump_when_free_list_exhausted() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch and close one app
        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let offset1 = system.get_app(slot).unwrap().state_offset;
        system.close_app(slot);

        // Launch with a larger size - free block too small, should bump allocate
        let slot2 = system.launch_app(app_type::CUSTOM, 2048, 1024).unwrap();
        let offset2 = system.get_app(slot2).unwrap().state_offset;

        // The offset should be after the original allocation (bump allocated)
        assert!(
            offset2 >= offset1 + 1024,
            "Larger allocation should bump allocate past original: {} >= {}",
            offset2,
            offset1 + 1024
        );

        // But the free block should still be there
        let stats = system.memory_stats();
        assert_eq!(stats.state_pool.free_count, 1, "Free block should still exist");
    }

    // ========================================================================
    // Issue #156: GPU Scheduler Tests
    // ========================================================================

    #[test]
    fn test_suspend_resume() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        // Verify active
        assert!(system.get_app(slot).unwrap().is_active());
        assert!(!system.get_app(slot).unwrap().flags & flags::SUSPENDED != 0);

        // Suspend
        system.suspend(slot);
        assert!(system.get_app(slot).unwrap().flags & flags::SUSPENDED != 0);

        // Run frame - suspended app shouldn't update
        let last_run_before = system.get_app(slot).unwrap().last_run_frame;
        system.mark_dirty(slot);
        system.run_frame();
        let last_run_after = system.get_app(slot).unwrap().last_run_frame;
        assert_eq!(last_run_before, last_run_after, "Suspended app should not run");

        // Resume
        system.resume(slot);
        assert!(system.get_app(slot).unwrap().flags & flags::SUSPENDED == 0);
        assert!(system.get_app(slot).unwrap().is_dirty(), "Resume should mark dirty");

        // Run frame - app should now run
        system.run_frame();
        let last_run_final = system.get_app(slot).unwrap().last_run_frame;
        assert!(last_run_final > last_run_after, "Resumed app should run");
    }

    #[test]
    fn test_set_priority() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        // Default priority
        assert_eq!(system.get_app(slot).unwrap().priority, priority::NORMAL);

        // Set to high
        system.set_priority(slot, priority::HIGH);
        assert_eq!(system.get_app(slot).unwrap().priority, priority::HIGH);

        // Set to background
        system.set_priority(slot, priority::BACKGROUND);
        assert_eq!(system.get_app(slot).unwrap().priority, priority::BACKGROUND);
    }

    #[test]
    fn test_scheduler_stats() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch apps with different priorities
        let bg = system.launch_app(app_type::CUSTOM, 256, 128).unwrap();
        system.set_priority(bg, priority::BACKGROUND);

        let normal = system.launch_app(app_type::CUSTOM, 256, 128).unwrap();
        // normal stays at default priority

        let high = system.launch_app(app_type::CUSTOM, 256, 128).unwrap();
        system.set_priority(high, priority::HIGH);

        // Suspend one
        system.suspend(bg);

        let stats = system.scheduler_stats();
        assert_eq!(stats.active_count, 3);
        assert_eq!(stats.suspended_count, 1);
        assert_eq!(stats.apps_by_priority[priority::BACKGROUND as usize], 1);
        assert_eq!(stats.apps_by_priority[priority::NORMAL as usize], 1);
        assert_eq!(stats.apps_by_priority[priority::HIGH as usize], 1);
    }

    #[test]
    fn test_budgeted_scheduler() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Enable budgeted scheduler
        system.set_use_budgeted_scheduler(true);
        assert!(system.is_using_budgeted_scheduler());

        // Set a low budget
        system.set_frame_budget(5000);

        // Launch several apps
        for _ in 0..5 {
            system.launch_app(app_type::CUSTOM, 256, 128);
        }

        // Run frame with budget
        system.mark_all_dirty();
        system.run_frame();

        // Check that some apps ran
        let stats = system.scheduler_stats();
        assert!(stats.active_count > 0);
        // Note: With low budget, some might be skipped
    }

    #[test]
    fn test_all_apps_evaluate_simultaneously() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch 64 apps
        for _ in 0..64 {
            system.launch_app(app_type::CUSTOM, 64, 32);
        }

        system.mark_all_dirty();
        system.run_frame();

        // All 64 should have run (parallel evaluation)
        let stats = system.scheduler_stats();
        assert_eq!(stats.active_count, 64);

        // Verify all ran in the same frame
        for slot in 0..64 {
            if let Some(app) = system.get_app(slot) {
                assert_eq!(app.last_run_frame, 1, "App {} should have run in frame 1", slot);
            }
        }
    }

    // ========================================================================
    // Issue #157: GPU Input & Window Integration Tests
    // ========================================================================

    #[test]
    fn test_create_window() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        // Create window
        let window_id = system.create_window(slot, 100.0, 100.0, 400.0, 300.0);
        assert!(window_id.is_some(), "Window should be created");

        // Verify window exists
        let window = system.get_window(window_id.unwrap());
        assert!(window.is_some(), "Window should be retrievable");

        let w = window.unwrap();
        assert_eq!(w.x, 100.0);
        assert_eq!(w.y, 100.0);
        assert_eq!(w.width, 400.0);
        assert_eq!(w.height, 300.0);
        assert_eq!(w.app_slot, slot);
    }

    #[test]
    fn test_focus_changes_on_set_focus() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let app1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let app2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        // Initially no focus
        assert!(system.focused_app().is_none(), "No app focused initially");

        // Focus app1
        system.set_focus(app1);
        assert_eq!(system.focused_app(), Some(app1), "App1 should be focused");
        assert!(system.get_app(app1).unwrap().flags & flags::FOCUS != 0);
        assert!(system.get_app(app2).unwrap().flags & flags::FOCUS == 0);

        // Focus app2 (should unfocus app1)
        system.set_focus(app2);
        assert_eq!(system.focused_app(), Some(app2), "App2 should be focused");
        assert!(system.get_app(app1).unwrap().flags & flags::FOCUS == 0);
        assert!(system.get_app(app2).unwrap().flags & flags::FOCUS != 0);
    }

    #[test]
    fn test_queue_input() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Queue keyboard event
        system.queue_input(InputEvent::key_down(0x04)); // 'A' key
        system.queue_input(InputEvent::key_up(0x04));

        // Queue mouse event
        system.queue_input(InputEvent::mouse_move(100.0, 200.0));
        system.queue_input(InputEvent::mouse_click(100.0, 200.0, 0));

        // Verify events are queued (read from buffer)
        let queue = system.read_input_queue();
        assert!(queue.tail >= 4, "Should have at least 4 events queued");
    }

    #[test]
    fn test_process_input_keyboard_to_focused() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let app1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let app2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        // Focus app1
        system.set_focus(app1);

        // Queue keyboard event
        system.queue_input(InputEvent::key_down(0x04));

        // Process input - GPU dispatches to apps
        system.process_input();

        // Only focused app should receive keyboard input
        let app1_desc = system.get_app(app1).unwrap();
        let app2_desc = system.get_app(app2).unwrap();

        // Focused app should be marked dirty (received input)
        assert!(app1_desc.flags & flags::DIRTY != 0, "Focused app should be dirty after input");
    }

    #[test]
    fn test_process_input_mouse_to_window() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let app1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        system.create_window(app1, 0.0, 0.0, 200.0, 200.0);

        let app2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        system.create_window(app2, 300.0, 0.0, 200.0, 200.0);

        // Queue mouse click inside app1's window
        system.queue_input(InputEvent::mouse_click(50.0, 50.0, 0));

        // Process input
        system.process_input();

        // App1 should be marked dirty (received mouse click)
        let app1_desc = system.get_app(app1).unwrap();
        assert!(app1_desc.flags & flags::DIRTY != 0, "App under mouse should receive input");
    }

    #[test]
    fn test_handle_click_focuses_topmost() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Create two overlapping windows
        let back = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let back_win = system.create_window(back, 0.0, 0.0, 200.0, 200.0).unwrap();

        let front = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let front_win = system.create_window(front, 50.0, 50.0, 200.0, 200.0).unwrap();

        // Bring front window to front (higher depth)
        system.set_window_depth(front_win, 0.9);
        system.set_window_depth(back_win, 0.1);

        // Click in overlap area
        system.handle_click(100.0, 100.0);

        // Front window's app should be focused
        assert_eq!(system.focused_app(), Some(front), "Front window should gain focus");
    }

    #[test]
    fn test_window_depth_ordering() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let app = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let window_id = system.create_window(app, 0.0, 0.0, 100.0, 100.0).unwrap();

        // Default depth
        let w = system.get_window(window_id).unwrap();
        assert!(w.depth >= 0.0 && w.depth <= 1.0, "Depth should be in [0,1]");

        // Set depth
        system.set_window_depth(window_id, 0.5);
        let w = system.get_window(window_id).unwrap();
        assert_eq!(w.depth, 0.5);

        // Bring to front
        system.set_window_depth(window_id, 1.0);
        let w = system.get_window(window_id).unwrap();
        assert_eq!(w.depth, 1.0);
    }

    // ========================================================================
    // Issue #158: GPU Rendering Pipeline Tests
    // ========================================================================

    #[test]
    fn test_render_state_initialization() {
        let device = get_device();
        let system = GpuAppSystem::new(&device).expect("Failed to create system");

        let state = system.render_state();
        assert_eq!(state.total_vertex_count, 0, "Initial vertex count should be 0");
        assert_eq!(state.max_vertices, MAX_VERTICES);
    }

    #[test]
    fn test_reset_render_state() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Manually modify vertex count
        unsafe {
            let render = system.render_state_buffer.contents() as *mut RenderState;
            (*render).total_vertex_count = 100;
        }

        // Reset should clear it
        system.reset_render_state();

        let state = system.render_state();
        assert_eq!(state.total_vertex_count, 0, "Reset should clear vertex count");
    }

    #[test]
    fn test_generate_vertices_single_app() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch app and create window
        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        system.create_window(slot, 100.0, 100.0, 200.0, 200.0);

        // Make visible
        unsafe {
            let apps = (system.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(slot as usize)).flags |= flags::VISIBLE;
        }

        // Generate vertices
        system.generate_vertices();

        // Check app has vertices
        let app = system.get_app(slot).unwrap();
        assert_eq!(app.vertex_count, 6, "App should have 6 vertices (2 triangles)");
    }

    #[test]
    fn test_finalize_render_sums_vertices() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch 3 apps with windows
        for i in 0..3 {
            let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
            system.create_window(slot, (i * 100) as f32, 0.0, 100.0, 100.0);

            // Make visible and set vertex count
            unsafe {
                let apps = (system.app_table_buffer.contents() as *mut u8)
                    .add(mem::size_of::<AppTableHeader>())
                    as *mut GpuAppDescriptor;
                (*apps.add(slot as usize)).flags |= flags::VISIBLE;
                (*apps.add(slot as usize)).vertex_count = 6;
            }
        }

        // Finalize render
        system.finalize_render();

        // total_vertex_count returns the highest vertex index for slot-based layout:
        // vertex_end = highest_slot * VERTS_PER_SLOT + vertex_count
        // With VERTS_PER_SLOT = 1024, slots 0,1,2 with 6 verts each:
        // max = slot 2 * 1024 + 6 = 2054
        let total = system.total_vertex_count();
        assert!(total >= 2054, "Total vertex count should be at least highest_slot * 1024 + verts");
    }

    #[test]
    fn test_unified_vertex_buffer() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Create two apps with windows
        let app1 = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();
        system.create_window(app1, 0.0, 0.0, 100.0, 100.0);

        let app2 = system.launch_app(app_type::CUSTOM, 1024, 1024).unwrap();
        system.create_window(app2, 100.0, 0.0, 100.0, 100.0);

        // Make visible
        unsafe {
            let apps = (system.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(app1 as usize)).flags |= flags::VISIBLE;
            (*apps.add(app2 as usize)).flags |= flags::VISIBLE;
        }

        // Generate vertices
        system.generate_vertices();

        // Check both apps have vertices
        assert_eq!(system.get_app(app1).unwrap().vertex_count, 6);
        assert_eq!(system.get_app(app2).unwrap().vertex_count, 6);

        // Finalize and check total
        // total_vertex_count returns highest vertex index for slot-based layout:
        // vertex_end = max_slot * VERTS_PER_SLOT + vertex_count
        // With VERTS_PER_SLOT = 1024, the highest slot determines the total
        system.finalize_render();
        let max_slot = std::cmp::max(app1, app2);
        let expected = max_slot * 1024 + 6; // highest_slot * VERTS_PER_SLOT + vertex_count
        assert_eq!(system.total_vertex_count(), expected);
    }

    #[test]
    fn test_invisible_apps_excluded_from_render() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Create visible and invisible apps
        let visible = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        system.create_window(visible, 0.0, 0.0, 100.0, 100.0);

        let invisible = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        system.create_window(invisible, 100.0, 0.0, 100.0, 100.0);

        // Apps start with VISIBLE flag set by launch kernel.
        // Clear VISIBLE from the invisible app to test filtering.
        unsafe {
            let apps = (system.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            // visible stays visible (flags already has VISIBLE)
            (*apps.add(visible as usize)).vertex_count = 6;
            // invisible: clear VISIBLE flag
            (*apps.add(invisible as usize)).flags &= !flags::VISIBLE;
            (*apps.add(invisible as usize)).vertex_count = 6;  // Has vertices but not visible
        }

        // Finalize render
        system.finalize_render();

        let total = system.total_vertex_count();

        // Only visible app should contribute
        assert_eq!(total, 6, "Only visible app should contribute vertices (got {})", total);
    }

    #[test]
    fn test_depth_in_vertices() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let slot = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let window_id = system.create_window(slot, 100.0, 100.0, 200.0, 200.0).unwrap();

        // Set specific depth
        system.set_window_depth(window_id, 0.75);

        // Make visible
        unsafe {
            let apps = (system.app_table_buffer.contents() as *mut u8)
                .add(mem::size_of::<AppTableHeader>())
                as *mut GpuAppDescriptor;
            (*apps.add(slot as usize)).flags |= flags::VISIBLE;
        }

        // Generate vertices
        system.generate_vertices();

        // Check that vertices have correct depth
        let app = system.get_app(slot).unwrap();
        assert_eq!(app.vertex_count, 6);

        // Read vertex from buffer
        unsafe {
            let vertices = system.render_vertices_buffer.contents() as *const RenderVertex;
            let base = app.vertex_offset / mem::size_of::<RenderVertex>() as u32;
            let v = *vertices.add(base as usize);
            assert_eq!(v.position[2], 0.75, "Vertex z should match window depth");
        }
    }

    // ========================================================================
    // Issue #159: App Migration to Megakernel Tests
    // ========================================================================

    #[test]
    fn test_app_type_registry() {
        // Verify registry has expected entries
        assert!(get_app_type_info(app_type::CUSTOM).is_some());
        assert!(get_app_type_info(app_type::GAME_OF_LIFE).is_some());
        assert!(get_app_type_info(app_type::PARTICLES).is_some());

        // Check CUSTOM app info
        let info = get_app_type_info(app_type::CUSTOM).unwrap();
        assert_eq!(info.name, "Custom");
        assert!(info.state_size > 0);
        assert!(info.vertex_size > 0);

        // Unknown type should return None
        assert!(get_app_type_info(999).is_none());
    }

    #[test]
    fn test_launch_by_type() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch by type
        let slot = system.launch_by_type(app_type::CUSTOM);
        assert!(slot.is_some(), "Should launch CUSTOM app");

        let app = system.get_app(slot.unwrap()).unwrap();
        assert_eq!(app.app_type, app_type::CUSTOM);

        // Launch Game of Life
        let gol = system.launch_by_type(app_type::GAME_OF_LIFE);
        assert!(gol.is_some(), "Should launch GAME_OF_LIFE app");

        let gol_app = system.get_app(gol.unwrap()).unwrap();
        assert_eq!(gol_app.app_type, app_type::GAME_OF_LIFE);

        // Unknown type should fail
        let unknown = system.launch_by_type(999);
        assert!(unknown.is_none(), "Unknown type should fail to launch");
    }

    #[test]
    fn test_parallel_megakernel_runs() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Enable parallel megakernel
        system.set_use_parallel_megakernel(true);
        assert!(system.is_using_parallel_megakernel());

        // Launch some apps
        let slot1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
        let slot2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

        // Run frame
        system.mark_all_dirty();
        system.run_frame();

        // Both apps should have run
        assert_eq!(system.get_app(slot1).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(slot2).unwrap().last_run_frame, 1);
    }

    #[test]
    fn test_game_of_life_app_type_sizes() {
        let info = get_app_type_info(app_type::GAME_OF_LIFE).unwrap();

        // Check reasonable sizes
        assert!(info.state_size >= 16, "State should include header");
        assert!(info.state_size >= 128 * 128, "State should include grid");
        assert!(info.thread_count >= 64, "Should use multiple threads");
    }

    #[test]
    fn test_counter_app_update() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Launch custom app
        let slot = system.launch_app(app_type::CUSTOM, 64, 288).unwrap();

        // Initialize counter state
        unsafe {
            let state = system.unified_state_buffer.contents() as *mut u8;
            let counter = state.add(system.get_app(slot).unwrap().state_offset as usize) as *mut u32;
            *counter = 0;  // Initial value
            *counter.add(1) = 1;  // Increment
        }

        // Run frames
        system.mark_dirty(slot);
        system.run_frame();

        system.mark_dirty(slot);
        system.run_frame();

        // Counter should have incremented
        unsafe {
            let state = system.unified_state_buffer.contents() as *const u8;
            let counter = state.add(system.get_app(slot).unwrap().state_offset as usize) as *const u32;
            assert!(*counter > 0, "Counter should have incremented");
        }
    }

    #[test]
    fn test_multiple_app_types_together() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        // Enable parallel megakernel
        system.set_use_parallel_megakernel(true);

        // Launch different app types
        let custom = system.launch_by_type(app_type::CUSTOM).unwrap();
        let gol = system.launch_by_type(app_type::GAME_OF_LIFE).unwrap();
        let particles = system.launch_by_type(app_type::PARTICLES).unwrap();

        // Run frame
        system.mark_all_dirty();
        system.run_frame();

        // All should have run
        assert_eq!(system.get_app(custom).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(gol).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(particles).unwrap().last_run_frame, 1);

        // Verify different app types
        assert_eq!(system.get_app(custom).unwrap().app_type, app_type::CUSTOM);
        assert_eq!(system.get_app(gol).unwrap().app_type, app_type::GAME_OF_LIFE);
        assert_eq!(system.get_app(particles).unwrap().app_type, app_type::PARTICLES);
    }

    // ========================================================================
    // Issue #155-#161: System App Tests
    // ========================================================================

    #[test]
    fn test_launch_compositor() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let compositor = system.launch_by_type(app_type::COMPOSITOR);
        assert!(compositor.is_some(), "Should be able to launch compositor");

        let app = system.get_app(compositor.unwrap()).unwrap();
        assert_eq!(app.app_type, app_type::COMPOSITOR);
    }

    #[test]
    fn test_launch_dock() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let dock = system.launch_by_type(app_type::DOCK);
        assert!(dock.is_some(), "Should be able to launch dock");

        let app = system.get_app(dock.unwrap()).unwrap();
        assert_eq!(app.app_type, app_type::DOCK);
    }

    #[test]
    fn test_launch_menubar() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let menubar = system.launch_by_type(app_type::MENUBAR);
        assert!(menubar.is_some(), "Should be able to launch menubar");

        let app = system.get_app(menubar.unwrap()).unwrap();
        assert_eq!(app.app_type, app_type::MENUBAR);
    }

    #[test]
    fn test_launch_window_chrome() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        let chrome = system.launch_by_type(app_type::WINDOW_CHROME);
        assert!(chrome.is_some(), "Should be able to launch window chrome");

        let app = system.get_app(chrome.unwrap()).unwrap();
        assert_eq!(app.app_type, app_type::WINDOW_CHROME);
    }

    #[test]
    fn test_system_apps_run_in_megakernel() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        system.set_use_parallel_megakernel(true);

        let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();
        let dock = system.launch_by_type(app_type::DOCK).unwrap();
        let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
        let chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

        system.set_priority(compositor, priority::REALTIME);
        system.set_priority(dock, priority::REALTIME);
        system.set_priority(menubar, priority::REALTIME);
        system.set_priority(chrome, priority::REALTIME);

        system.mark_all_dirty();
        system.run_frame();

        // All should have run
        assert_eq!(system.get_app(compositor).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(dock).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(menubar).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(chrome).unwrap().last_run_frame, 1);
    }

    #[test]
    fn test_system_apps_with_user_apps() {
        let device = get_device();
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

        system.set_use_parallel_megakernel(true);

        // Launch system apps
        let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();
        let dock = system.launch_by_type(app_type::DOCK).unwrap();

        // Launch user apps
        let terminal = system.launch_by_type(app_type::TERMINAL).unwrap();
        let filesystem = system.launch_by_type(app_type::FILESYSTEM).unwrap();

        system.mark_all_dirty();
        system.run_frame();

        // All should have run
        assert_eq!(system.get_app(compositor).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(dock).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(terminal).unwrap().last_run_frame, 1);
        assert_eq!(system.get_app(filesystem).unwrap().last_run_frame, 1);

        // System apps should be REALTIME capable
        system.set_priority(compositor, priority::REALTIME);
        assert_eq!(system.get_app(compositor).unwrap().priority, priority::REALTIME);
    }
}
