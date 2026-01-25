//! GPU Window Manager Kernel
//!
//! Provides GPU-accelerated window management operations:
//! - Hit testing (find window at coordinates)
//! - Z-order management (bring to front)
//! - Window move/resize operations
//!
//! All operations are implemented as Metal compute kernels for parallel execution.

use metal::*;
use std::mem;

use super::types::*;

/// Uniforms passed to window manager kernels
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct WindowManagerUniforms {
    /// Mouse position
    pub mouse_x: f32,
    pub mouse_y: f32,
    /// Mouse delta (for move/resize)
    pub delta_x: f32,
    pub delta_y: f32,

    /// Number of windows
    pub window_count: u32,
    /// Target window ID (for operations on specific window)
    pub target_window: u32,
    /// Operation flags
    pub operation: u32,
    /// Resize edge flags
    pub resize_edge: u32,

    /// Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
    /// Dock height (for constraining windows)
    pub dock_height: f32,
    pub _pad: f32,
}

/// Result of hit test operation
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct HitTestResult {
    /// Window ID at point (0 if none)
    pub window_id: u32,
    /// Hit region (0=none, 1=content, 2=titlebar, 3=close, 4=minimize, 5=maximize, 6=resize)
    pub region: u32,
    /// Resize edge flags (if region == 6)
    pub resize_edge: u32,
    /// Window index in array
    pub window_index: u32,
}

/// Hit test region codes
pub const REGION_NONE: u32 = 0;
pub const REGION_CONTENT: u32 = 1;
pub const REGION_TITLEBAR: u32 = 2;
pub const REGION_CLOSE_BUTTON: u32 = 3;
pub const REGION_MINIMIZE_BUTTON: u32 = 4;
pub const REGION_MAXIMIZE_BUTTON: u32 = 5;
pub const REGION_RESIZE: u32 = 6;

/// GPU Window Manager
///
/// Manages window operations using Metal compute shaders
pub struct GpuWindowManager {
    device: Device,
    command_queue: CommandQueue,

    // Compute pipelines
    hit_test_pipeline: ComputePipelineState,
    move_window_pipeline: ComputePipelineState,
    resize_window_pipeline: ComputePipelineState,
    z_order_pipeline: ComputePipelineState,

    // Buffers
    uniforms_buffer: Buffer,
    result_buffer: Buffer,
}

impl GpuWindowManager {
    /// Create a new window manager
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = device.new_library_with_source(WINDOW_MANAGER_METAL, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile window manager shaders: {}", e))?;

        let hit_test_fn = library.get_function("hit_test_kernel", None)
            .map_err(|e| format!("Failed to get hit_test_kernel: {}", e))?;
        let move_window_fn = library.get_function("move_window_kernel", None)
            .map_err(|e| format!("Failed to get move_window_kernel: {}", e))?;
        let resize_window_fn = library.get_function("resize_window_kernel", None)
            .map_err(|e| format!("Failed to get resize_window_kernel: {}", e))?;
        let z_order_fn = library.get_function("update_z_order_kernel", None)
            .map_err(|e| format!("Failed to get update_z_order_kernel: {}", e))?;

        let hit_test_pipeline = device.new_compute_pipeline_state_with_function(&hit_test_fn)
            .map_err(|e| format!("Failed to create hit_test pipeline: {}", e))?;
        let move_window_pipeline = device.new_compute_pipeline_state_with_function(&move_window_fn)
            .map_err(|e| format!("Failed to create move_window pipeline: {}", e))?;
        let resize_window_pipeline = device.new_compute_pipeline_state_with_function(&resize_window_fn)
            .map_err(|e| format!("Failed to create resize_window pipeline: {}", e))?;
        let z_order_pipeline = device.new_compute_pipeline_state_with_function(&z_order_fn)
            .map_err(|e| format!("Failed to create z_order pipeline: {}", e))?;

        // Create buffers
        let uniforms_buffer = device.new_buffer(
            mem::size_of::<WindowManagerUniforms>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let result_buffer = device.new_buffer(
            mem::size_of::<HitTestResult>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device: device.clone(),
            command_queue,
            hit_test_pipeline,
            move_window_pipeline,
            resize_window_pipeline,
            z_order_pipeline,
            uniforms_buffer,
            result_buffer,
        })
    }

    /// Perform hit test at coordinates
    ///
    /// Returns the window ID and hit region at the given coordinates.
    /// Runs on GPU to parallelize across all windows.
    pub fn hit_test(
        &self,
        windows_buffer: &Buffer,
        window_count: u32,
        x: f32,
        y: f32,
    ) -> HitTestResult {
        if window_count == 0 {
            return HitTestResult::default();
        }

        // Update uniforms
        let uniforms = WindowManagerUniforms {
            mouse_x: x,
            mouse_y: y,
            window_count,
            ..Default::default()
        };

        let uniforms_ptr = self.uniforms_buffer.contents() as *mut WindowManagerUniforms;
        unsafe { *uniforms_ptr = uniforms; }

        // Clear result (atomic uint, not HitTestResult struct)
        let result_ptr = self.result_buffer.contents() as *mut u32;
        unsafe { *result_ptr = 0; }

        // Execute kernel
        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.hit_test_pipeline);
        encoder.set_buffer(0, Some(windows_buffer), 0);
        encoder.set_buffer(1, Some(&self.uniforms_buffer), 0);
        encoder.set_buffer(2, Some(&self.result_buffer), 0);

        // One thread per window
        let thread_count = window_count.max(1) as u64;
        let threads_per_group = self.hit_test_pipeline.thread_execution_width();
        let thread_groups = (thread_count + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Unpack result: (z_order << 24) | (region << 16) | (edge << 8) | index
        let packed = unsafe { *result_ptr };

        if packed == 0 {
            return HitTestResult::default();
        }

        let window_index = (packed & 0xFF) as u32;
        let resize_edge = ((packed >> 8) & 0xFF) as u32;
        let region = ((packed >> 16) & 0xFF) as u32;

        // Get window ID from the windows buffer
        let window_id = unsafe {
            let windows_ptr = windows_buffer.contents() as *const Window;
            (*windows_ptr.add(window_index as usize)).id
        };

        HitTestResult {
            window_id,
            region,
            resize_edge,
            window_index,
        }
    }

    /// Move a window by delta
    ///
    /// Updates window position on GPU.
    pub fn move_window(
        &self,
        windows_buffer: &Buffer,
        window_count: u32,
        window_id: u32,
        delta_x: f32,
        delta_y: f32,
        screen_width: f32,
        screen_height: f32,
        dock_height: f32,
    ) {
        let uniforms = WindowManagerUniforms {
            delta_x,
            delta_y,
            window_count,
            target_window: window_id,
            screen_width,
            screen_height,
            dock_height,
            ..Default::default()
        };

        let uniforms_ptr = self.uniforms_buffer.contents() as *mut WindowManagerUniforms;
        unsafe { *uniforms_ptr = uniforms; }

        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.move_window_pipeline);
        encoder.set_buffer(0, Some(windows_buffer), 0);
        encoder.set_buffer(1, Some(&self.uniforms_buffer), 0);

        // Only need one thread to move one window
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Resize a window
    pub fn resize_window(
        &self,
        windows_buffer: &Buffer,
        window_count: u32,
        window_id: u32,
        delta_x: f32,
        delta_y: f32,
        resize_edge: u32,
        screen_width: f32,
        screen_height: f32,
        dock_height: f32,
    ) {
        let uniforms = WindowManagerUniforms {
            delta_x,
            delta_y,
            window_count,
            target_window: window_id,
            resize_edge,
            screen_width,
            screen_height,
            dock_height,
            ..Default::default()
        };

        let uniforms_ptr = self.uniforms_buffer.contents() as *mut WindowManagerUniforms;
        unsafe { *uniforms_ptr = uniforms; }

        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.resize_window_pipeline);
        encoder.set_buffer(0, Some(windows_buffer), 0);
        encoder.set_buffer(1, Some(&self.uniforms_buffer), 0);

        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1, 1, 1),
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Bring window to front (update z-order)
    pub fn bring_to_front(
        &self,
        windows_buffer: &Buffer,
        window_count: u32,
        window_id: u32,
    ) {
        let uniforms = WindowManagerUniforms {
            window_count,
            target_window: window_id,
            ..Default::default()
        };

        let uniforms_ptr = self.uniforms_buffer.contents() as *mut WindowManagerUniforms;
        unsafe { *uniforms_ptr = uniforms; }

        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.z_order_pipeline);
        encoder.set_buffer(0, Some(windows_buffer), 0);
        encoder.set_buffer(1, Some(&self.uniforms_buffer), 0);

        // One thread per window to update z-order
        let thread_count = window_count.max(1) as u64;
        let threads_per_group = self.z_order_pipeline.thread_execution_width();
        let thread_groups = (thread_count + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Create a GPU buffer for windows
    pub fn create_windows_buffer(&self, capacity: usize) -> Buffer {
        self.device.new_buffer(
            (capacity * mem::size_of::<Window>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

/// Metal shader source for window manager kernels
const WINDOW_MANAGER_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Window flags
constant uint WINDOW_FLAG_VISIBLE = 1 << 0;
constant uint WINDOW_FLAG_MINIMIZED = 1 << 2;
constant uint WINDOW_FLAG_BORDERLESS = 1 << 7;
constant uint WINDOW_FLAG_FIXED_SIZE = 1 << 8;

// Window constants
constant float TITLE_BAR_HEIGHT = 28.0;
constant float BUTTON_SIZE = 12.0;
constant float BUTTON_SPACING = 8.0;
constant float BUTTON_INSET = 8.0;
constant float RESIZE_HANDLE_SIZE = 6.0;
constant float MIN_WINDOW_WIDTH = 200.0;
constant float MIN_WINDOW_HEIGHT = 100.0;

// Resize edge flags
constant uint RESIZE_LEFT = 1 << 0;
constant uint RESIZE_RIGHT = 1 << 1;
constant uint RESIZE_TOP = 1 << 2;
constant uint RESIZE_BOTTOM = 1 << 3;

// Hit region codes
constant uint REGION_NONE = 0;
constant uint REGION_CONTENT = 1;
constant uint REGION_TITLEBAR = 2;
constant uint REGION_CLOSE_BUTTON = 3;
constant uint REGION_MINIMIZE_BUTTON = 4;
constant uint REGION_MAXIMIZE_BUTTON = 5;
constant uint REGION_RESIZE = 6;

// Window structure (must match Rust)
struct Window {
    float x;
    float y;
    float width;
    float height;
    uint id;
    uint z_order;
    uint app_id;
    uint flags;
    float content_x;
    float content_y;
    float content_width;
    float content_height;
    char title[64];
    float _padding[4];
};

struct WindowManagerUniforms {
    float mouse_x;
    float mouse_y;
    float delta_x;
    float delta_y;
    uint window_count;
    uint target_window;
    uint operation;
    uint resize_edge;
    float screen_width;
    float screen_height;
    float dock_height;
    float _pad;
};

struct HitTestResult {
    uint window_id;
    uint region;
    uint resize_edge;
    uint window_index;
};

// Check if point is inside window
bool window_contains_point(device const Window& win, float px, float py) {
    return px >= win.x && px < win.x + win.width &&
           py >= win.y && py < win.y + win.height;
}

// Check if window is visible
bool window_is_visible(device const Window& win) {
    return (win.flags & WINDOW_FLAG_VISIBLE) != 0 &&
           (win.flags & WINDOW_FLAG_MINIMIZED) == 0;
}

// Get resize edge at point
uint get_resize_edge(device const Window& win, float px, float py) {
    if ((win.flags & WINDOW_FLAG_FIXED_SIZE) != 0) {
        return 0;
    }

    uint edge = 0;

    if (px >= win.x && px < win.x + RESIZE_HANDLE_SIZE) {
        edge |= RESIZE_LEFT;
    }
    if (px > win.x + win.width - RESIZE_HANDLE_SIZE && px <= win.x + win.width) {
        edge |= RESIZE_RIGHT;
    }
    if (py >= win.y && py < win.y + RESIZE_HANDLE_SIZE) {
        edge |= RESIZE_TOP;
    }
    if (py > win.y + win.height - RESIZE_HANDLE_SIZE && py <= win.y + win.height) {
        edge |= RESIZE_BOTTOM;
    }

    return edge;
}

// Get hit region at point
uint get_hit_region(device const Window& win, float px, float py, thread uint& resize_edge) {
    if (!window_contains_point(win, px, py)) {
        return REGION_NONE;
    }

    // Check resize edges first
    resize_edge = get_resize_edge(win, px, py);
    if (resize_edge != 0) {
        return REGION_RESIZE;
    }

    // Check if borderless
    if ((win.flags & WINDOW_FLAG_BORDERLESS) != 0) {
        return REGION_CONTENT;
    }

    // Check title bar
    if (py >= win.y && py < win.y + TITLE_BAR_HEIGHT) {
        // Check close button
        float bx = win.x + BUTTON_INSET;
        float by = win.y + (TITLE_BAR_HEIGHT - BUTTON_SIZE) / 2.0;
        if (px >= bx && px < bx + BUTTON_SIZE && py >= by && py < by + BUTTON_SIZE) {
            return REGION_CLOSE_BUTTON;
        }

        // Check minimize button
        bx = win.x + BUTTON_INSET + BUTTON_SIZE + BUTTON_SPACING;
        if (px >= bx && px < bx + BUTTON_SIZE && py >= by && py < by + BUTTON_SIZE) {
            return REGION_MINIMIZE_BUTTON;
        }

        // Check maximize button
        bx = win.x + BUTTON_INSET + 2.0 * (BUTTON_SIZE + BUTTON_SPACING);
        if (px >= bx && px < bx + BUTTON_SIZE && py >= by && py < by + BUTTON_SIZE) {
            return REGION_MAXIMIZE_BUTTON;
        }

        return REGION_TITLEBAR;
    }

    return REGION_CONTENT;
}

// Hit test kernel - find window at coordinates
// Uses atomic max to find topmost (highest z-order) window at point
kernel void hit_test_kernel(
    device Window* windows [[buffer(0)]],
    device const WindowManagerUniforms& uniforms [[buffer(1)]],
    device atomic_uint* result [[buffer(2)]],  // Packed: (z_order << 24) | (region << 16) | (edge << 8) | index
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uniforms.window_count) return;

    device const Window& win = windows[tid];

    if (!window_is_visible(win)) return;

    uint resize_edge = 0;
    uint region = get_hit_region(win, uniforms.mouse_x, uniforms.mouse_y, resize_edge);

    if (region == REGION_NONE) return;

    // Pack result: use z_order for atomic max comparison
    // Higher z_order wins (front-most window)
    uint packed = (win.z_order << 24) | (region << 16) | (resize_edge << 8) | (tid & 0xFF);

    atomic_fetch_max_explicit(result, packed, memory_order_relaxed);
}

// Move window kernel
kernel void move_window_kernel(
    device Window* windows [[buffer(0)]],
    device const WindowManagerUniforms& uniforms [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // Find target window
    for (uint i = 0; i < uniforms.window_count; i++) {
        if (windows[i].id == uniforms.target_window) {
            // Apply delta
            float new_x = windows[i].x + uniforms.delta_x;
            float new_y = windows[i].y + uniforms.delta_y;

            // Constrain to screen (allow partial off-screen but keep title bar visible)
            float min_x = -windows[i].width + 100.0;  // At least 100px visible
            float max_x = uniforms.screen_width - 100.0;
            float min_y = 0.0;  // Don't go above screen
            float max_y = uniforms.screen_height - uniforms.dock_height - TITLE_BAR_HEIGHT;

            windows[i].x = clamp(new_x, min_x, max_x);
            windows[i].y = clamp(new_y, min_y, max_y);

            return;
        }
    }
}

// Resize window kernel
kernel void resize_window_kernel(
    device Window* windows [[buffer(0)]],
    device const WindowManagerUniforms& uniforms [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    for (uint i = 0; i < uniforms.window_count; i++) {
        if (windows[i].id == uniforms.target_window) {
            device Window& win = windows[i];

            float new_x = win.x;
            float new_y = win.y;
            float new_w = win.width;
            float new_h = win.height;

            // Apply resize based on edge flags
            if ((uniforms.resize_edge & RESIZE_LEFT) != 0) {
                float dx = uniforms.delta_x;
                if (new_w - dx >= MIN_WINDOW_WIDTH) {
                    new_x += dx;
                    new_w -= dx;
                }
            }
            if ((uniforms.resize_edge & RESIZE_RIGHT) != 0) {
                new_w += uniforms.delta_x;
            }
            if ((uniforms.resize_edge & RESIZE_TOP) != 0) {
                float dy = uniforms.delta_y;
                if (new_h - dy >= MIN_WINDOW_HEIGHT) {
                    new_y += dy;
                    new_h -= dy;
                }
            }
            if ((uniforms.resize_edge & RESIZE_BOTTOM) != 0) {
                new_h += uniforms.delta_y;
            }

            // Enforce minimum size
            new_w = max(new_w, MIN_WINDOW_WIDTH);
            new_h = max(new_h, MIN_WINDOW_HEIGHT);

            // Apply
            win.x = new_x;
            win.y = new_y;
            win.width = new_w;
            win.height = new_h;

            // Update content area
            if ((win.flags & WINDOW_FLAG_BORDERLESS) != 0) {
                win.content_x = 0.0;
                win.content_y = 0.0;
                win.content_width = new_w;
                win.content_height = new_h;
            } else {
                win.content_x = 0.0;
                win.content_y = TITLE_BAR_HEIGHT;
                win.content_width = new_w;
                win.content_height = new_h - TITLE_BAR_HEIGHT;
            }

            return;
        }
    }
}

// Update z-order kernel - bring target window to front
kernel void update_z_order_kernel(
    device Window* windows [[buffer(0)]],
    device const WindowManagerUniforms& uniforms [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uniforms.window_count) return;

    device Window& win = windows[tid];

    if (win.id == uniforms.target_window) {
        // Bring to front: set z_order to window_count (highest)
        win.z_order = uniforms.window_count;
    } else if (win.z_order > 0) {
        // Shift others down
        win.z_order--;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_manager_creation() {
        let device = Device::system_default().expect("No Metal device");
        let manager = GpuWindowManager::new(&device);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_hit_test() {
        let device = Device::system_default().expect("No Metal device");
        let manager = GpuWindowManager::new(&device).expect("Failed to create manager");

        // Create a test window
        let windows_buffer = manager.create_windows_buffer(1);
        let mut win = Window::new(1, "Test", 100.0, 100.0, 400.0, 300.0);
        win.z_order = 0;

        let ptr = windows_buffer.contents() as *mut Window;
        unsafe { *ptr = win; }

        // Hit test inside window
        let result = manager.hit_test(&windows_buffer, 1, 200.0, 200.0);
        assert_eq!(result.window_id, 1);
        assert_eq!(result.region, REGION_CONTENT);

        // Hit test in title bar
        let result = manager.hit_test(&windows_buffer, 1, 200.0, 110.0);
        assert_eq!(result.window_id, 1);
        assert_eq!(result.region, REGION_TITLEBAR);
    }
}
