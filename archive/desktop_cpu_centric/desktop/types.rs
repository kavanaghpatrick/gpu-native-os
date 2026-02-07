//! GPU Desktop Core Window Data Structures
//!
//! Defines GPU-aligned data structures for window management.
//! All structs use #[repr(C)] with explicit padding for Metal alignment.

use std::mem;

// ============================================================================
// Window Flags
// ============================================================================

/// Window is visible on screen
pub const WINDOW_FLAG_VISIBLE: u32 = 1 << 0;
/// Window has keyboard focus
pub const WINDOW_FLAG_FOCUSED: u32 = 1 << 1;
/// Window is minimized to dock
pub const WINDOW_FLAG_MINIMIZED: u32 = 1 << 2;
/// Window is maximized (fills screen minus dock)
pub const WINDOW_FLAG_MAXIMIZED: u32 = 1 << 3;
/// Window is being dragged
pub const WINDOW_FLAG_DRAGGING: u32 = 1 << 4;
/// Window is being resized
pub const WINDOW_FLAG_RESIZING: u32 = 1 << 5;
/// Window has unsaved changes (shows dot in close button)
pub const WINDOW_FLAG_DIRTY: u32 = 1 << 6;
/// Window title bar is hidden
pub const WINDOW_FLAG_BORDERLESS: u32 = 1 << 7;
/// Window cannot be resized
pub const WINDOW_FLAG_FIXED_SIZE: u32 = 1 << 8;
/// Window is a modal dialog
pub const WINDOW_FLAG_MODAL: u32 = 1 << 9;
/// Window content needs redraw
pub const WINDOW_FLAG_NEEDS_REDRAW: u32 = 1 << 10;

// ============================================================================
// Resize Edge Flags
// ============================================================================

pub const RESIZE_NONE: u32 = 0;
pub const RESIZE_LEFT: u32 = 1 << 0;
pub const RESIZE_RIGHT: u32 = 1 << 1;
pub const RESIZE_TOP: u32 = 1 << 2;
pub const RESIZE_BOTTOM: u32 = 1 << 3;
pub const RESIZE_TOP_LEFT: u32 = RESIZE_TOP | RESIZE_LEFT;
pub const RESIZE_TOP_RIGHT: u32 = RESIZE_TOP | RESIZE_RIGHT;
pub const RESIZE_BOTTOM_LEFT: u32 = RESIZE_BOTTOM | RESIZE_LEFT;
pub const RESIZE_BOTTOM_RIGHT: u32 = RESIZE_BOTTOM | RESIZE_RIGHT;

// ============================================================================
// Constants
// ============================================================================

/// Maximum number of windows supported
pub const MAX_WINDOWS: usize = 64;

/// Maximum title length in bytes
pub const MAX_TITLE_LEN: usize = 64;

/// Title bar height in pixels
pub const TITLE_BAR_HEIGHT: f32 = 28.0;

/// Window button size (close, minimize, maximize)
pub const BUTTON_SIZE: f32 = 12.0;

/// Window button spacing
pub const BUTTON_SPACING: f32 = 8.0;

/// Window button inset from edge
pub const BUTTON_INSET: f32 = 8.0;

/// Window border/shadow size
pub const BORDER_SIZE: f32 = 1.0;

/// Window shadow blur radius
pub const SHADOW_RADIUS: f32 = 20.0;

/// Window corner radius
pub const CORNER_RADIUS: f32 = 10.0;

/// Minimum window size
pub const MIN_WINDOW_WIDTH: f32 = 200.0;
pub const MIN_WINDOW_HEIGHT: f32 = 100.0;

/// Resize handle size at window edges
pub const RESIZE_HANDLE_SIZE: f32 = 6.0;

// ============================================================================
// Window Structure
// ============================================================================

/// A window in the desktop environment
///
/// GPU-aligned structure (128 bytes total) for efficient GPU processing.
/// Uses packed arrays and explicit padding for Metal compatibility.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Window {
    // Frame (x, y, width, height) - 16 bytes
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,

    // Identifiers - 16 bytes
    pub id: u32,          // Unique window ID
    pub z_order: u32,     // Z-order (0 = bottom, higher = front)
    pub app_id: u32,      // Associated app ID
    pub flags: u32,       // Window flags (WINDOW_FLAG_*)

    // Content area offset (for rendering app content) - 16 bytes
    pub content_x: f32,
    pub content_y: f32,
    pub content_width: f32,
    pub content_height: f32,

    // Title (64 bytes, null-terminated UTF-8)
    pub title: [u8; MAX_TITLE_LEN],

    // Padding to reach 128 bytes (128 - 48 - 64 = 16 bytes)
    pub _padding: [f32; 4],
}

impl Default for Window {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 400.0,
            height: 300.0,
            id: 0,
            z_order: 0,
            app_id: 0,
            flags: WINDOW_FLAG_VISIBLE,
            content_x: 0.0,
            content_y: TITLE_BAR_HEIGHT,
            content_width: 400.0,
            content_height: 300.0 - TITLE_BAR_HEIGHT,
            title: [0u8; MAX_TITLE_LEN],
            _padding: [0.0; 4],
        }
    }
}

impl Window {
    /// Create a new window
    pub fn new(id: u32, title: &str, x: f32, y: f32, width: f32, height: f32) -> Self {
        let mut win = Self {
            id,
            x,
            y,
            width,
            height,
            ..Default::default()
        };
        win.set_title(title);
        win.update_content_area();
        win
    }

    /// Set window title
    pub fn set_title(&mut self, title: &str) {
        self.title = [0u8; MAX_TITLE_LEN];
        let bytes = title.as_bytes();
        let len = bytes.len().min(MAX_TITLE_LEN - 1);
        self.title[..len].copy_from_slice(&bytes[..len]);
    }

    /// Get window title as string
    pub fn get_title(&self) -> &str {
        let end = self.title.iter().position(|&b| b == 0).unwrap_or(MAX_TITLE_LEN);
        std::str::from_utf8(&self.title[..end]).unwrap_or("")
    }

    /// Update content area based on frame and title bar
    pub fn update_content_area(&mut self) {
        if self.flags & WINDOW_FLAG_BORDERLESS != 0 {
            self.content_x = 0.0;
            self.content_y = 0.0;
            self.content_width = self.width;
            self.content_height = self.height;
        } else {
            self.content_x = 0.0;
            self.content_y = TITLE_BAR_HEIGHT;
            self.content_width = self.width;
            self.content_height = self.height - TITLE_BAR_HEIGHT;
        }
    }

    /// Check if point is inside window frame
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width &&
        py >= self.y && py < self.y + self.height
    }

    /// Check if point is in title bar
    pub fn in_title_bar(&self, px: f32, py: f32) -> bool {
        if self.flags & WINDOW_FLAG_BORDERLESS != 0 {
            return false;
        }
        px >= self.x && px < self.x + self.width &&
        py >= self.y && py < self.y + TITLE_BAR_HEIGHT
    }

    /// Check if point is in close button
    pub fn in_close_button(&self, px: f32, py: f32) -> bool {
        let bx = self.x + BUTTON_INSET;
        let by = self.y + (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0;
        px >= bx && px < bx + BUTTON_SIZE &&
        py >= by && py < by + BUTTON_SIZE
    }

    /// Check if point is in minimize button
    pub fn in_minimize_button(&self, px: f32, py: f32) -> bool {
        let bx = self.x + BUTTON_INSET + BUTTON_SIZE + BUTTON_SPACING;
        let by = self.y + (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0;
        px >= bx && px < bx + BUTTON_SIZE &&
        py >= by && py < by + BUTTON_SIZE
    }

    /// Check if point is in maximize button
    pub fn in_maximize_button(&self, px: f32, py: f32) -> bool {
        let bx = self.x + BUTTON_INSET + 2.0 * (BUTTON_SIZE + BUTTON_SPACING);
        let by = self.y + (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0;
        px >= bx && px < bx + BUTTON_SIZE &&
        py >= by && py < by + BUTTON_SIZE
    }

    /// Get resize edge at point (returns RESIZE_* flags)
    pub fn resize_edge_at(&self, px: f32, py: f32) -> u32 {
        if self.flags & WINDOW_FLAG_FIXED_SIZE != 0 {
            return RESIZE_NONE;
        }

        let mut edge = RESIZE_NONE;

        // Check edges
        if px >= self.x && px < self.x + RESIZE_HANDLE_SIZE {
            edge |= RESIZE_LEFT;
        }
        if px > self.x + self.width - RESIZE_HANDLE_SIZE && px <= self.x + self.width {
            edge |= RESIZE_RIGHT;
        }
        if py >= self.y && py < self.y + RESIZE_HANDLE_SIZE {
            edge |= RESIZE_TOP;
        }
        if py > self.y + self.height - RESIZE_HANDLE_SIZE && py <= self.y + self.height {
            edge |= RESIZE_BOTTOM;
        }

        edge
    }

    /// Set window flag
    pub fn set_flag(&mut self, flag: u32, value: bool) {
        if value {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }

    /// Check if flag is set
    pub fn has_flag(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Check if window is visible
    pub fn is_visible(&self) -> bool {
        self.has_flag(WINDOW_FLAG_VISIBLE) && !self.has_flag(WINDOW_FLAG_MINIMIZED)
    }

    /// Check if window has focus
    pub fn is_focused(&self) -> bool {
        self.has_flag(WINDOW_FLAG_FOCUSED)
    }
}

// ============================================================================
// Drag State
// ============================================================================

/// State for window drag/resize operations
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct DragState {
    /// Window ID being dragged (0 if none)
    pub window_id: u32,
    /// Drag mode (0=none, 1=move, 2=resize)
    pub mode: u32,
    /// Resize edge flags
    pub resize_edge: u32,
    pub _pad0: u32,

    /// Drag start position (mouse)
    pub start_x: f32,
    pub start_y: f32,
    /// Window position at drag start
    pub window_start_x: f32,
    pub window_start_y: f32,

    /// Window size at drag start (for resize)
    pub window_start_width: f32,
    pub window_start_height: f32,
    pub _pad1: [f32; 2],
}

impl DragState {
    /// Start a move drag
    pub fn start_move(&mut self, window_id: u32, mouse_x: f32, mouse_y: f32, win_x: f32, win_y: f32) {
        self.window_id = window_id;
        self.mode = 1;
        self.resize_edge = RESIZE_NONE;
        self.start_x = mouse_x;
        self.start_y = mouse_y;
        self.window_start_x = win_x;
        self.window_start_y = win_y;
    }

    /// Start a resize drag
    pub fn start_resize(&mut self, window_id: u32, edge: u32, mouse_x: f32, mouse_y: f32,
                        win_x: f32, win_y: f32, win_w: f32, win_h: f32) {
        self.window_id = window_id;
        self.mode = 2;
        self.resize_edge = edge;
        self.start_x = mouse_x;
        self.start_y = mouse_y;
        self.window_start_x = win_x;
        self.window_start_y = win_y;
        self.window_start_width = win_w;
        self.window_start_height = win_h;
    }

    /// End dragging
    pub fn end_drag(&mut self) {
        self.window_id = 0;
        self.mode = 0;
        self.resize_edge = RESIZE_NONE;
    }

    /// Check if currently dragging
    pub fn is_dragging(&self) -> bool {
        self.mode != 0
    }

    /// Check if in move mode
    pub fn is_moving(&self) -> bool {
        self.mode == 1
    }

    /// Check if in resize mode
    pub fn is_resizing(&self) -> bool {
        self.mode == 2
    }
}

// ============================================================================
// Desktop State
// ============================================================================

/// Desktop environment state
///
/// Contains all windows and global desktop state.
/// GPU-aligned for efficient compute kernel access.
#[repr(C, align(16))]
pub struct DesktopState {
    /// All windows (MAX_WINDOWS slots)
    pub windows: [Window; MAX_WINDOWS],

    /// Number of active windows
    pub window_count: u32,
    /// ID of focused window (0 if none)
    pub focused_window: u32,
    /// Next window ID to assign
    pub next_window_id: u32,
    /// Screen dimensions
    pub screen_width: f32,

    pub screen_height: f32,
    /// Dock height
    pub dock_height: f32,
    /// Desktop background color (RGBA packed)
    pub background_color: u32,
    pub _pad0: u32,

    /// Current drag state
    pub drag: DragState,

    /// Mouse position
    pub mouse_x: f32,
    pub mouse_y: f32,
    /// Mouse button state (bit 0 = left, bit 1 = right, bit 2 = middle)
    pub mouse_buttons: u32,
    pub _pad1: u32,
}

impl Default for DesktopState {
    fn default() -> Self {
        Self {
            windows: [Window::default(); MAX_WINDOWS],
            window_count: 0,
            focused_window: 0,
            next_window_id: 1,
            screen_width: 1920.0,
            screen_height: 1080.0,
            dock_height: 70.0,
            background_color: 0x1a1a2eFF,  // Dark blue-gray
            _pad0: 0,
            drag: DragState::default(),
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: 0,
            _pad1: 0,
        }
    }
}

/// Placement constants
const PLACEMENT_PADDING: f32 = 10.0;   // Padding from screen edges
const PLACEMENT_GRID_STEP: f32 = 50.0; // Grid step for candidate positions

impl DesktopState {
    /// Create a new desktop state with given screen dimensions
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            screen_width: width,
            screen_height: height,
            ..Default::default()
        }
    }

    /// Create a new window and return its ID
    pub fn create_window(&mut self, title: &str, x: f32, y: f32, width: f32, height: f32) -> Option<u32> {
        if self.window_count as usize >= MAX_WINDOWS {
            return None;
        }

        let id = self.next_window_id;
        self.next_window_id += 1;

        let idx = self.window_count as usize;
        self.windows[idx] = Window::new(id, title, x, y, width, height);
        self.windows[idx].z_order = self.window_count;
        self.window_count += 1;

        // Focus the new window
        self.focus_window(id);

        Some(id)
    }

    /// Close a window by ID
    pub fn close_window(&mut self, id: u32) -> bool {
        if let Some(idx) = self.find_window_index(id) {
            // Remove window by swapping with last and decrementing count
            let last_idx = (self.window_count - 1) as usize;
            if idx != last_idx {
                self.windows[idx] = self.windows[last_idx];
            }
            self.window_count -= 1;

            // Update focus if needed
            if self.focused_window == id {
                self.focused_window = if self.window_count > 0 {
                    // Focus topmost window
                    self.get_topmost_window().map(|w| w.id).unwrap_or(0)
                } else {
                    0
                };
            }

            // Re-normalize z-order
            self.normalize_z_order();
            true
        } else {
            false
        }
    }

    /// Focus a window by ID
    pub fn focus_window(&mut self, id: u32) {
        if let Some(idx) = self.find_window_index(id) {
            // Unfocus previous
            if self.focused_window != 0 {
                if let Some(prev_idx) = self.find_window_index(self.focused_window) {
                    self.windows[prev_idx].flags &= !WINDOW_FLAG_FOCUSED;
                }
            }

            // Focus new
            self.windows[idx].flags |= WINDOW_FLAG_FOCUSED;
            self.focused_window = id;

            // Bring to front
            self.bring_to_front(id);
        }
    }

    /// Bring window to front
    pub fn bring_to_front(&mut self, id: u32) {
        if let Some(idx) = self.find_window_index(id) {
            let max_z = self.window_count;
            self.windows[idx].z_order = max_z;
            self.normalize_z_order();
        }
    }

    /// Normalize z-order values to be sequential
    fn normalize_z_order(&mut self) {
        // Sort windows by z-order
        let count = self.window_count as usize;
        let windows = &mut self.windows[..count];

        // Simple bubble sort (small number of windows)
        for i in 0..count {
            for j in 0..count - 1 - i {
                if windows[j].z_order > windows[j + 1].z_order {
                    windows.swap(j, j + 1);
                }
            }
        }

        // Assign sequential z-order
        for (i, win) in windows.iter_mut().enumerate() {
            win.z_order = i as u32;
        }
    }

    /// Find window at screen coordinates
    pub fn window_at(&self, x: f32, y: f32) -> Option<&Window> {
        // Search from front to back (highest z-order first)
        let mut result: Option<&Window> = None;
        let mut max_z = 0u32;

        for i in 0..self.window_count as usize {
            let win = &self.windows[i];
            if win.is_visible() && win.contains_point(x, y) && win.z_order >= max_z {
                max_z = win.z_order;
                result = Some(win);
            }
        }

        result
    }

    /// Find window at screen coordinates (mutable)
    pub fn window_at_mut(&mut self, x: f32, y: f32) -> Option<&mut Window> {
        let mut found_idx: Option<usize> = None;
        let mut max_z = 0u32;

        for i in 0..self.window_count as usize {
            let win = &self.windows[i];
            if win.is_visible() && win.contains_point(x, y) && win.z_order >= max_z {
                max_z = win.z_order;
                found_idx = Some(i);
            }
        }

        found_idx.map(|idx| &mut self.windows[idx])
    }

    /// Get topmost visible window
    pub fn get_topmost_window(&self) -> Option<&Window> {
        let mut result: Option<&Window> = None;
        let mut max_z = 0u32;

        for i in 0..self.window_count as usize {
            let win = &self.windows[i];
            if win.is_visible() && win.z_order >= max_z {
                max_z = win.z_order;
                result = Some(win);
            }
        }

        result
    }

    /// Get focused window
    pub fn get_focused_window(&self) -> Option<&Window> {
        if self.focused_window == 0 {
            return None;
        }
        self.find_window_index(self.focused_window)
            .map(|idx| &self.windows[idx])
    }

    /// Get focused window (mutable)
    pub fn get_focused_window_mut(&mut self) -> Option<&mut Window> {
        if self.focused_window == 0 {
            return None;
        }
        self.find_window_index(self.focused_window)
            .map(|idx| &mut self.windows[idx])
    }

    /// Get window by ID
    pub fn get_window(&self, id: u32) -> Option<&Window> {
        self.find_window_index(id).map(|idx| &self.windows[idx])
    }

    /// Get window by ID (mutable)
    pub fn get_window_mut(&mut self, id: u32) -> Option<&mut Window> {
        self.find_window_index(id).map(|idx| &mut self.windows[idx])
    }

    /// Find window index by ID
    fn find_window_index(&self, id: u32) -> Option<usize> {
        for i in 0..self.window_count as usize {
            if self.windows[i].id == id {
                return Some(i);
            }
        }
        None
    }

    /// Iterate over visible windows in z-order (back to front)
    pub fn visible_windows_z_order(&self) -> impl Iterator<Item = &Window> {
        let mut indices: Vec<usize> = (0..self.window_count as usize)
            .filter(|&i| self.windows[i].is_visible())
            .collect();
        indices.sort_by_key(|&i| self.windows[i].z_order);
        indices.into_iter().map(move |i| &self.windows[i])
    }

    /// Get usable screen area (excluding menu bar and dock)
    pub fn usable_area(&self) -> (f32, f32, f32, f32) {
        let menu_bar_height = 24.0;  // MENU_BAR_HEIGHT
        (0.0, menu_bar_height, self.screen_width, self.screen_height - self.dock_height - menu_bar_height)
    }

    /// Find a non-overlapping position for a new window
    ///
    /// Scans candidate positions and finds one with zero (or minimal) overlap
    /// with existing windows. Uses a grid-based search starting from top-left.
    pub fn find_non_overlapping_position(&self, width: f32, height: f32) -> (f32, f32) {
        let (usable_x, usable_y, usable_w, usable_h) = self.usable_area();

        // If no windows, place at top-left with padding
        if self.window_count == 0 {
            return (usable_x + PLACEMENT_PADDING, usable_y + PLACEMENT_PADDING);
        }

        let max_x = usable_x + usable_w - width - PLACEMENT_PADDING;
        let max_y = usable_y + usable_h - height - PLACEMENT_PADDING;

        // Clamp to ensure window fits
        let max_x = max_x.max(usable_x + PLACEMENT_PADDING);
        let max_y = max_y.max(usable_y + PLACEMENT_PADDING);

        let mut best_x = usable_x + PLACEMENT_PADDING;
        let mut best_y = usable_y + PLACEMENT_PADDING;
        let mut best_overlap = f32::MAX;

        // Scan grid of candidate positions
        let mut test_y = usable_y + PLACEMENT_PADDING;
        while test_y <= max_y {
            let mut test_x = usable_x + PLACEMENT_PADDING;
            while test_x <= max_x {
                let overlap = self.calculate_total_overlap(test_x, test_y, width, height);

                if overlap < best_overlap {
                    best_overlap = overlap;
                    best_x = test_x;
                    best_y = test_y;

                    // Found zero overlap - use it immediately
                    if overlap == 0.0 {
                        return (best_x, best_y);
                    }
                }

                test_x += PLACEMENT_GRID_STEP;
            }
            test_y += PLACEMENT_GRID_STEP;
        }

        // Also try positions adjacent to existing windows (right edge, bottom edge)
        for i in 0..self.window_count as usize {
            let win = &self.windows[i];
            if !win.is_visible() {
                continue;
            }

            // Try to the right of this window
            let right_x = win.x + win.width + PLACEMENT_PADDING;
            let right_y = win.y;
            if right_x + width <= usable_x + usable_w - PLACEMENT_PADDING {
                let overlap = self.calculate_total_overlap(right_x, right_y, width, height);
                if overlap < best_overlap {
                    best_overlap = overlap;
                    best_x = right_x;
                    best_y = right_y;
                    if overlap == 0.0 {
                        return (best_x, best_y);
                    }
                }
            }

            // Try below this window
            let below_x = win.x;
            let below_y = win.y + win.height + PLACEMENT_PADDING;
            if below_y + height <= usable_y + usable_h - PLACEMENT_PADDING {
                let overlap = self.calculate_total_overlap(below_x, below_y, width, height);
                if overlap < best_overlap {
                    best_overlap = overlap;
                    best_x = below_x;
                    best_y = below_y;
                    if overlap == 0.0 {
                        return (best_x, best_y);
                    }
                }
            }
        }

        (best_x, best_y)
    }

    /// Calculate total overlap area between a candidate rect and all existing windows
    fn calculate_total_overlap(&self, x: f32, y: f32, width: f32, height: f32) -> f32 {
        let mut total = 0.0;

        for i in 0..self.window_count as usize {
            let win = &self.windows[i];
            if !win.is_visible() {
                continue;
            }

            // Calculate intersection
            let left = x.max(win.x);
            let right = (x + width).min(win.x + win.width);
            let top = y.max(win.y);
            let bottom = (y + height).min(win.y + win.height);

            if left < right && top < bottom {
                let overlap_area = (right - left) * (bottom - top);
                total += overlap_area;
            }
        }

        total
    }
}

// ============================================================================
// Size Assertions
// ============================================================================

const _: () = {
    assert!(mem::size_of::<Window>() == 128, "Window must be 128 bytes");
    assert!(mem::align_of::<Window>() == 16, "Window must be 16-byte aligned");
    assert!(mem::size_of::<DragState>() == 48, "DragState must be 48 bytes");
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_creation() {
        let win = Window::new(1, "Test Window", 100.0, 100.0, 400.0, 300.0);
        assert_eq!(win.id, 1);
        assert_eq!(win.get_title(), "Test Window");
        assert_eq!(win.x, 100.0);
        assert_eq!(win.width, 400.0);
        assert!(win.is_visible());
    }

    #[test]
    fn test_window_hit_test() {
        let win = Window::new(1, "Test", 100.0, 100.0, 400.0, 300.0);

        // Inside window
        assert!(win.contains_point(200.0, 200.0));
        // Outside window
        assert!(!win.contains_point(50.0, 50.0));
        // In title bar
        assert!(win.in_title_bar(200.0, 110.0));
        // Not in title bar (below)
        assert!(!win.in_title_bar(200.0, 200.0));
    }

    #[test]
    fn test_desktop_create_close() {
        let mut desktop = DesktopState::new(1920.0, 1080.0);

        // Create windows
        let id1 = desktop.create_window("Window 1", 100.0, 100.0, 400.0, 300.0).unwrap();
        let id2 = desktop.create_window("Window 2", 200.0, 200.0, 400.0, 300.0).unwrap();

        assert_eq!(desktop.window_count, 2);
        assert_eq!(desktop.focused_window, id2);  // Last created is focused

        // Close first window
        assert!(desktop.close_window(id1));
        assert_eq!(desktop.window_count, 1);

        // Close second window
        assert!(desktop.close_window(id2));
        assert_eq!(desktop.window_count, 0);
    }

    #[test]
    fn test_desktop_z_order() {
        let mut desktop = DesktopState::new(1920.0, 1080.0);

        let id1 = desktop.create_window("Window 1", 100.0, 100.0, 400.0, 300.0).unwrap();
        let id2 = desktop.create_window("Window 2", 200.0, 200.0, 400.0, 300.0).unwrap();
        let id3 = desktop.create_window("Window 3", 300.0, 300.0, 400.0, 300.0).unwrap();

        // id3 should be on top (highest z_order)
        let topmost = desktop.get_topmost_window().unwrap();
        assert_eq!(topmost.id, id3);

        // Bring id1 to front
        desktop.bring_to_front(id1);
        let topmost = desktop.get_topmost_window().unwrap();
        assert_eq!(topmost.id, id1);
    }

    #[test]
    fn test_window_at_point() {
        let mut desktop = DesktopState::new(1920.0, 1080.0);

        let id1 = desktop.create_window("Window 1", 100.0, 100.0, 200.0, 200.0).unwrap();
        let id2 = desktop.create_window("Window 2", 150.0, 150.0, 200.0, 200.0).unwrap();

        // Point in overlap area should return id2 (on top)
        let win = desktop.window_at(200.0, 200.0).unwrap();
        assert_eq!(win.id, id2);

        // Point only in id1
        let win = desktop.window_at(110.0, 110.0).unwrap();
        assert_eq!(win.id, id1);

        // Point outside all windows
        assert!(desktop.window_at(50.0, 50.0).is_none());
    }
}
