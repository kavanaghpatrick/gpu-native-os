//! GPU App Descriptor Format (Issue #148)
//!
//! Defines the GPU-resident data structure that describes a running app.
//! This replaces CPU-side `Box<dyn DesktopApp>`.

use metal::Buffer;

/// App descriptor flags
pub mod flags {
    pub const ACTIVE: u32 = 1 << 0;   // Slot in use
    pub const VISIBLE: u32 = 1 << 1;  // Should render
    pub const FOCUSED: u32 = 1 << 2;  // Has input focus
    pub const CLOSING: u32 = 1 << 3;  // Marked for close
    pub const DIRTY: u32 = 1 << 4;    // Needs redraw
}

/// Invalid slot marker
pub const INVALID_SLOT: u32 = 0xFFFFFFFF;

/// Maximum number of app slots
pub const MAX_APP_SLOTS: usize = 256;

/// GPU App Descriptor - 128 bytes, fits in 2 cache lines
///
/// This struct lives in GPU memory and describes everything needed
/// to run an app without CPU involvement.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuAppDescriptor {
    // Identity (16 bytes)
    pub slot_id: u32,           // Slot in app pool
    pub window_id: u32,         // Associated window
    pub app_type: u32,          // Index into pipeline table
    pub flags: u32,             // APP_ACTIVE, APP_VISIBLE, etc.

    // Buffer offsets into GPU memory pool (32 bytes)
    pub state_offset: u32,      // App-specific state
    pub state_size: u32,
    pub vertex_offset: u32,     // Vertex buffer
    pub vertex_size: u32,
    pub param_offset: u32,      // Per-frame params
    pub param_size: u32,
    pub extra_offset: u32,      // Additional buffers
    pub extra_size: u32,

    // Configuration (32 bytes)
    pub thread_count: u32,      // Compute dispatch size
    pub vertex_count: u32,      // Draw call vertex count
    pub clear_color: [f32; 4],  // Background color
    pub preferred_size: [f32; 2], // Window size hint
    pub compute_pipeline_handle: u32, // Handle into pipeline table
    pub render_pipeline_handle: u32,  // Handle into pipeline table

    // Runtime state (40 bytes)
    pub total_time: f32,        // Accumulated time
    pub frame_count: u32,       // Frames rendered
    pub mouse_x: f32,           // Cursor in app space
    pub mouse_y: f32,
    pub mouse_buttons: u32,     // Button state
    pub key_modifiers: u32,     // Modifier keys
    pub _runtime_pad: [u32; 4], // Pad to 40 bytes
}

// Compile-time size check
const _: () = assert!(std::mem::size_of::<GpuAppDescriptor>() == 128);

impl Default for GpuAppDescriptor {
    fn default() -> Self {
        Self {
            slot_id: INVALID_SLOT,
            window_id: 0,
            app_type: 0,
            flags: 0,
            state_offset: 0,
            state_size: 0,
            vertex_offset: 0,
            vertex_size: 0,
            param_offset: 0,
            param_size: 0,
            extra_offset: 0,
            extra_size: 0,
            thread_count: 0,
            vertex_count: 0,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            preferred_size: [800.0, 600.0],
            compute_pipeline_handle: INVALID_SLOT,
            render_pipeline_handle: INVALID_SLOT,
            total_time: 0.0,
            frame_count: 0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_buttons: 0,
            key_modifiers: 0,
            _runtime_pad: [0; 4],
        }
    }
}

impl GpuAppDescriptor {
    /// Check if slot is active
    pub fn is_active(&self) -> bool {
        self.flags & flags::ACTIVE != 0
    }

    /// Check if app is visible
    pub fn is_visible(&self) -> bool {
        self.flags & flags::VISIBLE != 0
    }

    /// Check if app has focus
    pub fn is_focused(&self) -> bool {
        self.flags & flags::FOCUSED != 0
    }

    /// Check if app is marked for closing
    pub fn is_closing(&self) -> bool {
        self.flags & flags::CLOSING != 0
    }

    /// Check if app needs redraw
    pub fn is_dirty(&self) -> bool {
        self.flags & flags::DIRTY != 0
    }

    /// Mark app for closing
    pub fn mark_closing(&mut self) {
        self.flags |= flags::CLOSING;
    }

    /// Mark app as dirty (needs redraw)
    pub fn mark_dirty(&mut self) {
        self.flags |= flags::DIRTY;
    }

    /// Clear dirty flag
    pub fn clear_dirty(&mut self) {
        self.flags &= !flags::DIRTY;
    }

    /// Activate the slot
    pub fn activate(&mut self, slot_id: u32, window_id: u32, app_type: u32) {
        self.slot_id = slot_id;
        self.window_id = window_id;
        self.app_type = app_type;
        self.flags = flags::ACTIVE | flags::VISIBLE | flags::DIRTY;
        self.total_time = 0.0;
        self.frame_count = 0;
    }

    /// Deactivate the slot
    pub fn deactivate(&mut self) {
        self.flags = 0;
        self.slot_id = INVALID_SLOT;
    }

    /// Set buffer locations
    pub fn set_buffers(
        &mut self,
        state_offset: u32,
        state_size: u32,
        vertex_offset: u32,
        vertex_size: u32,
    ) {
        self.state_offset = state_offset;
        self.state_size = state_size;
        self.vertex_offset = vertex_offset;
        self.vertex_size = vertex_size;
    }

    /// Verify buffer offsets don't overlap
    pub fn buffers_valid(&self) -> bool {
        // Check state and vertex don't overlap
        if self.state_size > 0 && self.vertex_size > 0 {
            let state_end = self.state_offset + self.state_size;
            let vertex_end = self.vertex_offset + self.vertex_size;

            // Either state ends before vertex starts, or vertex ends before state starts
            if !(state_end <= self.vertex_offset || vertex_end <= self.state_offset) {
                return false;
            }
        }
        true
    }
}

/// Metal shader header for GpuAppDescriptor
pub const APP_DESCRIPTOR_METAL_HEADER: &str = r#"
// GPU App Descriptor - 128 bytes
struct GpuAppDescriptor {
    // Identity (16 bytes)
    uint slot_id;
    uint window_id;
    uint app_type;
    uint flags;

    // Buffer offsets (32 bytes)
    uint state_offset;
    uint state_size;
    uint vertex_offset;
    uint vertex_size;
    uint param_offset;
    uint param_size;
    uint extra_offset;
    uint extra_size;

    // Configuration (32 bytes)
    uint thread_count;
    uint vertex_count;
    float4 clear_color;
    float2 preferred_size;
    uint compute_pipeline_handle;
    uint render_pipeline_handle;

    // Runtime (40 bytes)
    float total_time;
    uint frame_count;
    float mouse_x;
    float mouse_y;
    uint mouse_buttons;
    uint key_modifiers;
    uint _runtime_pad[4];
};

// App flags
#define APP_FLAG_ACTIVE   (1u << 0)
#define APP_FLAG_VISIBLE  (1u << 1)
#define APP_FLAG_FOCUSED  (1u << 2)
#define APP_FLAG_CLOSING  (1u << 3)
#define APP_FLAG_DIRTY    (1u << 4)

#define INVALID_SLOT 0xFFFFFFFF
#define MAX_APP_SLOTS 256

// Helper functions
inline bool app_is_active(device const GpuAppDescriptor* app) {
    return (app->flags & APP_FLAG_ACTIVE) != 0;
}

inline bool app_is_visible(device const GpuAppDescriptor* app) {
    return (app->flags & APP_FLAG_VISIBLE) != 0;
}

inline bool app_needs_render(device const GpuAppDescriptor* app) {
    return app_is_active(app) && app_is_visible(app) && (app->flags & APP_FLAG_DIRTY) != 0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_size() {
        assert_eq!(std::mem::size_of::<GpuAppDescriptor>(), 128);
    }

    #[test]
    fn test_descriptor_alignment() {
        // Should be at least 4-byte aligned for GPU
        assert!(std::mem::align_of::<GpuAppDescriptor>() >= 4);
    }

    #[test]
    fn test_flags() {
        let mut desc = GpuAppDescriptor::default();
        assert!(!desc.is_active());
        assert!(!desc.is_visible());
        assert!(!desc.is_focused());
        assert!(!desc.is_closing());

        desc.flags |= flags::ACTIVE;
        assert!(desc.is_active());

        desc.flags |= flags::VISIBLE;
        assert!(desc.is_visible());

        desc.mark_closing();
        assert!(desc.is_closing());
    }

    #[test]
    fn test_activate_deactivate() {
        let mut desc = GpuAppDescriptor::default();

        desc.activate(5, 100, 1);
        assert!(desc.is_active());
        assert!(desc.is_visible());
        assert!(desc.is_dirty());
        assert_eq!(desc.slot_id, 5);
        assert_eq!(desc.window_id, 100);
        assert_eq!(desc.app_type, 1);

        desc.deactivate();
        assert!(!desc.is_active());
        assert_eq!(desc.slot_id, INVALID_SLOT);
    }

    #[test]
    fn test_buffer_offsets() {
        let mut desc = GpuAppDescriptor::default();

        desc.set_buffers(1024, 256, 2048, 4096);

        assert_eq!(desc.state_offset, 1024);
        assert_eq!(desc.state_size, 256);
        assert_eq!(desc.vertex_offset, 2048);
        assert_eq!(desc.vertex_size, 4096);

        // Verify no overlap
        assert!(desc.buffers_valid());
    }

    #[test]
    fn test_buffer_overlap_detection() {
        let mut desc = GpuAppDescriptor::default();

        // Overlapping buffers
        desc.state_offset = 1000;
        desc.state_size = 500;
        desc.vertex_offset = 1200; // Overlaps with state
        desc.vertex_size = 500;

        assert!(!desc.buffers_valid());
    }

    #[test]
    fn test_default_values() {
        let desc = GpuAppDescriptor::default();

        assert_eq!(desc.slot_id, INVALID_SLOT);
        assert_eq!(desc.flags, 0);
        assert_eq!(desc.clear_color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(desc.preferred_size, [800.0, 600.0]);
        assert_eq!(desc.compute_pipeline_handle, INVALID_SLOT);
        assert_eq!(desc.render_pipeline_handle, INVALID_SLOT);
    }

    #[test]
    fn test_dirty_flag() {
        let mut desc = GpuAppDescriptor::default();

        assert!(!desc.is_dirty());

        desc.mark_dirty();
        assert!(desc.is_dirty());

        desc.clear_dirty();
        assert!(!desc.is_dirty());
    }
}
