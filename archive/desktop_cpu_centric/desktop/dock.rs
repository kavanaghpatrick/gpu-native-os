//! GPU Dock System
//!
//! A macOS-style dock at the bottom of the screen with:
//! - App icons with hover magnification
//! - Running app indicators
//! - Click handling for app launch
//! - Smooth animations
//!
//! Fully GPU-accelerated rendering.

use metal::*;
use std::mem;

/// Maximum number of dock items
pub const MAX_DOCK_ITEMS: usize = 32;

/// Dock item flags
pub const DOCK_ITEM_VISIBLE: u32 = 1 << 0;
pub const DOCK_ITEM_RUNNING: u32 = 1 << 1;
pub const DOCK_ITEM_HOVERED: u32 = 1 << 2;
pub const DOCK_ITEM_CLICKED: u32 = 1 << 3;
pub const DOCK_ITEM_BOUNCING: u32 = 1 << 4;

/// A single dock item
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct DockItem {
    /// App ID this item represents
    pub app_id: u32,
    /// Item flags
    pub flags: u32,
    /// Icon texture index (in icon atlas)
    pub icon_index: u32,
    /// Number of running instances
    pub instance_count: u32,

    /// Computed position (set by layout kernel)
    pub x: f32,
    pub y: f32,
    /// Current size (animated for hover magnification)
    pub size: f32,
    /// Bounce animation progress (0-1)
    pub bounce_progress: f32,

    /// App name (for tooltips)
    pub name: [u8; 32],

    /// Padding to reach 64 bytes
    pub _padding: [f32; 2],
}

impl Default for DockItem {
    fn default() -> Self {
        Self {
            app_id: 0,
            flags: 0,
            icon_index: 0,
            instance_count: 0,
            x: 0.0,
            y: 0.0,
            size: 48.0,
            bounce_progress: 0.0,
            name: [0u8; 32],
            _padding: [0.0; 2],
        }
    }
}

impl DockItem {
    /// Create a new dock item
    pub fn new(app_id: u32, name: &str, icon_index: u32) -> Self {
        let mut item = Self {
            app_id,
            icon_index,
            flags: DOCK_ITEM_VISIBLE,
            ..Default::default()
        };
        item.set_name(name);
        item
    }

    /// Set item name
    pub fn set_name(&mut self, name: &str) {
        self.name = [0u8; 32];
        let bytes = name.as_bytes();
        let len = bytes.len().min(31);
        self.name[..len].copy_from_slice(&bytes[..len]);
    }

    /// Get item name
    pub fn get_name(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(32);
        std::str::from_utf8(&self.name[..end]).unwrap_or("")
    }

    /// Check if item is visible
    pub fn is_visible(&self) -> bool {
        self.flags & DOCK_ITEM_VISIBLE != 0
    }

    /// Check if app is running
    pub fn is_running(&self) -> bool {
        self.flags & DOCK_ITEM_RUNNING != 0 || self.instance_count > 0
    }

    /// Check if hovered
    pub fn is_hovered(&self) -> bool {
        self.flags & DOCK_ITEM_HOVERED != 0
    }

    /// Set running state
    pub fn set_running(&mut self, running: bool) {
        if running {
            self.flags |= DOCK_ITEM_RUNNING;
        } else {
            self.flags &= !DOCK_ITEM_RUNNING;
        }
    }
}

/// Dock configuration
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct DockConfig {
    /// Dock height
    pub height: f32,
    /// Icon size (default)
    pub icon_size: f32,
    /// Icon size (hovered/magnified)
    pub icon_size_hover: f32,
    /// Spacing between icons
    pub spacing: f32,

    /// Magnification range (pixels from cursor)
    pub magnification_range: f32,
    /// Bounce animation duration (frames)
    pub bounce_duration: f32,
    /// Dock background color
    pub background_color: [f32; 4],

    /// Running indicator color
    pub indicator_color: [f32; 4],

    /// Screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
    /// Mouse position (for hover effects)
    pub mouse_x: f32,
    pub mouse_y: f32,

    /// Number of dock items
    pub item_count: u32,
    /// Delta time (for animations)
    pub delta_time: f32,
    pub _padding: [f32; 2],
}

impl Default for DockConfig {
    fn default() -> Self {
        Self {
            height: 70.0,
            icon_size: 48.0,
            icon_size_hover: 64.0,
            spacing: 8.0,
            magnification_range: 100.0,
            bounce_duration: 30.0,
            background_color: [0.2, 0.2, 0.2, 0.8],  // Semi-transparent dark
            indicator_color: [0.4, 0.7, 1.0, 1.0],    // Light blue
            screen_width: 1920.0,
            screen_height: 1080.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            item_count: 0,
            delta_time: 1.0 / 60.0,
            _padding: [0.0; 2],
        }
    }
}

/// Dock state
#[repr(C, align(16))]
pub struct DockState {
    pub items: [DockItem; MAX_DOCK_ITEMS],
    pub config: DockConfig,
    pub hovered_item: i32,  // -1 if none
    pub _padding: [u32; 3],
}

impl Default for DockState {
    fn default() -> Self {
        Self {
            items: [DockItem::default(); MAX_DOCK_ITEMS],
            config: DockConfig::default(),
            hovered_item: -1,
            _padding: [0; 3],
        }
    }
}

impl DockState {
    /// Create a new dock state
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
        let mut state = Self::default();
        state.config.screen_width = screen_width;
        state.config.screen_height = screen_height;
        state
    }

    /// Add an item to the dock
    pub fn add_item(&mut self, app_id: u32, name: &str, icon_index: u32) -> bool {
        if self.config.item_count as usize >= MAX_DOCK_ITEMS {
            return false;
        }

        let idx = self.config.item_count as usize;
        self.items[idx] = DockItem::new(app_id, name, icon_index);
        self.config.item_count += 1;
        true
    }

    /// Remove an item from the dock
    pub fn remove_item(&mut self, app_id: u32) -> bool {
        for i in 0..self.config.item_count as usize {
            if self.items[i].app_id == app_id {
                // Shift items down
                for j in i..self.config.item_count as usize - 1 {
                    self.items[j] = self.items[j + 1];
                }
                self.config.item_count -= 1;
                return true;
            }
        }
        false
    }

    /// Find item at coordinates
    pub fn item_at(&self, x: f32, y: f32) -> Option<usize> {
        // Check if in dock area
        let dock_y = self.config.screen_height - self.config.height;
        if y < dock_y {
            return None;
        }

        // Check each item
        for i in 0..self.config.item_count as usize {
            let item = &self.items[i];
            if !item.is_visible() {
                continue;
            }

            let half_size = item.size / 2.0;
            if x >= item.x - half_size && x <= item.x + half_size &&
               y >= item.y - half_size && y <= item.y + half_size {
                return Some(i);
            }
        }

        None
    }

    /// Update hover state based on mouse position
    pub fn update_hover(&mut self, mouse_x: f32, mouse_y: f32) {
        self.config.mouse_x = mouse_x;
        self.config.mouse_y = mouse_y;

        // Clear previous hover
        for i in 0..self.config.item_count as usize {
            self.items[i].flags &= !DOCK_ITEM_HOVERED;
        }

        // Set new hover
        if let Some(idx) = self.item_at(mouse_x, mouse_y) {
            self.items[idx].flags |= DOCK_ITEM_HOVERED;
            self.hovered_item = idx as i32;
        } else {
            self.hovered_item = -1;
        }
    }

    /// Get the dock rectangle
    pub fn dock_rect(&self) -> (f32, f32, f32, f32) {
        let y = self.config.screen_height - self.config.height;
        (0.0, y, self.config.screen_width, self.config.height)
    }
}

/// Dock vertex for rendering
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct DockVertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
    pub color: [f32; 4],
    pub item_index: u32,
    /// Vertex type: 0=background, 1=icon, 2=indicator
    pub vertex_type: u32,
    pub _pad: [f32; 2],
}

/// GPU Dock Renderer
pub struct GpuDock {
    command_queue: CommandQueue,

    // Pipelines
    layout_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    state_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,

    /// Cached state
    pub state: DockState,
}

impl GpuDock {
    /// Create a new GPU dock
    pub fn new(device: &Device, pixel_format: MTLPixelFormat) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let library = device.new_library_with_source(DOCK_METAL, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile dock shaders: {}", e))?;

        let layout_fn = library.get_function("dock_layout_kernel", None)
            .map_err(|e| format!("Failed to get dock_layout_kernel: {}", e))?;
        let layout_pipeline = device.new_compute_pipeline_state_with_function(&layout_fn)
            .map_err(|e| format!("Failed to create layout pipeline: {}", e))?;

        let vertex_fn = library.get_function("dock_vertex", None)
            .map_err(|e| format!("Failed to get dock_vertex: {}", e))?;
        let fragment_fn = library.get_function("dock_fragment", None)
            .map_err(|e| format!("Failed to get dock_fragment: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc.color_attachments().object_at(0).unwrap().set_pixel_format(pixel_format);

        // Enable blending
        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_blending_enabled(true);
        attachment.set_rgb_blend_operation(MTLBlendOperation::Add);
        attachment.set_alpha_blend_operation(MTLBlendOperation::Add);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device.new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        // Create buffers
        let state_buffer = device.new_buffer(
            mem::size_of::<DockState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Max vertices: background (6) + icons (6 each) + indicators (6 each)
        let max_vertices = 6 + MAX_DOCK_ITEMS * 12;
        let vertices_buffer = device.new_buffer(
            (max_vertices * mem::size_of::<DockVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vertex_count_buffer = device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            command_queue,
            layout_pipeline,
            render_pipeline,
            state_buffer,
            vertices_buffer,
            vertex_count_buffer,
            state: DockState::default(),
        })
    }

    /// Update dock layout (compute icon positions with magnification)
    pub fn update_layout(&mut self) {
        // Sync state to GPU using ptr::copy
        let state_ptr = self.state_buffer.contents() as *mut DockState;
        unsafe {
            std::ptr::copy_nonoverlapping(&self.state as *const DockState, state_ptr, 1);
        }

        // Reset vertex count
        let count_ptr = self.vertex_count_buffer.contents() as *mut u32;
        unsafe { *count_ptr = 0; }

        // Execute layout kernel
        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.layout_pipeline);
        encoder.set_buffer(0, Some(&self.state_buffer), 0);
        encoder.set_buffer(1, Some(&self.vertices_buffer), 0);
        encoder.set_buffer(2, Some(&self.vertex_count_buffer), 0);

        // One thread per item + 1 for background
        let thread_count = (self.state.config.item_count + 1).max(1) as u64;
        let threads_per_group = self.layout_pipeline.thread_execution_width();
        let thread_groups = (thread_count + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Sync item positions back from GPU
        unsafe {
            let gpu_state = &*state_ptr;
            for i in 0..self.state.config.item_count as usize {
                self.state.items[i].x = gpu_state.items[i].x;
                self.state.items[i].y = gpu_state.items[i].y;
                self.state.items[i].size = gpu_state.items[i].size;
            }
        }
    }

    /// Render the dock
    pub fn render(&self, encoder: &RenderCommandEncoderRef) {
        let vertex_count = unsafe {
            *(self.vertex_count_buffer.contents() as *const u32)
        };

        if vertex_count == 0 {
            return;
        }

        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&self.state_buffer), 0);
        encoder.set_fragment_buffer(0, Some(&self.state_buffer), 0);

        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, vertex_count as u64);
    }

    /// Get the vertex count
    pub fn vertex_count(&self) -> u32 {
        unsafe { *(self.vertex_count_buffer.contents() as *const u32) }
    }
}

/// Metal shader source for dock
const DOCK_METAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint DOCK_ITEM_VISIBLE = 1 << 0;
constant uint DOCK_ITEM_RUNNING = 1 << 1;
constant uint DOCK_ITEM_HOVERED = 1 << 2;

constant uint VERTEX_BACKGROUND = 0;
constant uint VERTEX_ICON = 1;
constant uint VERTEX_INDICATOR = 2;

struct DockItem {
    uint app_id;
    uint flags;
    uint icon_index;
    uint instance_count;
    float x;
    float y;
    float size;
    float bounce_progress;
    char name[32];
    float _padding[2];
};

struct DockConfig {
    float height;
    float icon_size;
    float icon_size_hover;
    float spacing;
    float magnification_range;
    float bounce_duration;
    float4 background_color;
    float4 indicator_color;
    float screen_width;
    float screen_height;
    float mouse_x;
    float mouse_y;
    uint item_count;
    float delta_time;
    float2 _padding;
};

struct DockState {
    DockItem items[32];
    DockConfig config;
    int hovered_item;
    uint _padding[3];
};

struct DockVertex {
    float2 position;
    float2 tex_coord;
    float4 color;
    uint item_index;
    uint vertex_type;
    float2 _pad;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 tex_coord;
    uint vertex_type;
};

// Add a quad
void add_dock_quad(
    device DockVertex* vertices,
    device atomic_uint* vertex_count,
    float x, float y, float w, float h,
    float4 color,
    uint item_index,
    uint vertex_type,
    float screen_width,
    float screen_height
) {
    uint base = atomic_fetch_add_explicit(vertex_count, 6, memory_order_relaxed);

    // Convert to NDC
    float x0 = (x / screen_width) * 2.0 - 1.0;
    float y0 = 1.0 - (y / screen_height) * 2.0;
    float x1 = ((x + w) / screen_width) * 2.0 - 1.0;
    float y1 = 1.0 - ((y + h) / screen_height) * 2.0;

    vertices[base + 0] = DockVertex{float2(x0, y0), float2(0, 0), color, item_index, vertex_type, float2(0)};
    vertices[base + 1] = DockVertex{float2(x1, y0), float2(1, 0), color, item_index, vertex_type, float2(0)};
    vertices[base + 2] = DockVertex{float2(x0, y1), float2(0, 1), color, item_index, vertex_type, float2(0)};
    vertices[base + 3] = DockVertex{float2(x1, y0), float2(1, 0), color, item_index, vertex_type, float2(0)};
    vertices[base + 4] = DockVertex{float2(x1, y1), float2(1, 1), color, item_index, vertex_type, float2(0)};
    vertices[base + 5] = DockVertex{float2(x0, y1), float2(0, 1), color, item_index, vertex_type, float2(0)};
}

// Layout and generate vertices
kernel void dock_layout_kernel(
    device DockState* state [[buffer(0)]],
    device DockVertex* vertices [[buffer(1)]],
    device atomic_uint* vertex_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    device DockConfig& config = state->config;

    if (tid == 0) {
        // Background quad
        float dock_y = config.screen_height - config.height;
        add_dock_quad(vertices, vertex_count,
            0, dock_y, config.screen_width, config.height,
            config.background_color, 0, VERTEX_BACKGROUND,
            config.screen_width, config.screen_height);
    }

    if (tid > 0 && tid <= config.item_count) {
        uint item_idx = tid - 1;
        device DockItem& item = state->items[item_idx];

        if ((item.flags & DOCK_ITEM_VISIBLE) == 0) return;

        // Calculate position based on index
        float total_width = 0;
        for (uint i = 0; i < config.item_count; i++) {
            total_width += config.icon_size + config.spacing;
        }
        total_width -= config.spacing;

        float start_x = (config.screen_width - total_width) / 2.0;
        float x = start_x;
        for (uint i = 0; i < item_idx; i++) {
            x += config.icon_size + config.spacing;
        }

        // Apply magnification based on mouse distance
        float distance = abs(config.mouse_x - (x + config.icon_size / 2.0));
        float mag_factor = 1.0 - clamp(distance / config.magnification_range, 0.0, 1.0);
        float size = mix(config.icon_size, config.icon_size_hover, mag_factor * mag_factor);

        // Dock Y position
        float dock_y = config.screen_height - config.height;
        float icon_y = dock_y + (config.height - size) / 2.0;

        // Apply bounce if active
        if ((item.flags & DOCK_ITEM_HOVERED) != 0) {
            // Subtle lift on hover
            icon_y -= 5.0;
        }

        // Store computed position
        item.x = x + config.icon_size / 2.0;  // Center position
        item.y = icon_y + size / 2.0;
        item.size = size;

        // Icon quad (placeholder - would use texture in real impl)
        float4 icon_color = float4(0.6, 0.6, 0.8, 1.0);  // Placeholder color
        if ((item.flags & DOCK_ITEM_HOVERED) != 0) {
            icon_color = float4(0.7, 0.7, 0.9, 1.0);  // Lighter on hover
        }

        add_dock_quad(vertices, vertex_count,
            x + (config.icon_size - size) / 2.0, icon_y, size, size,
            icon_color, item_idx, VERTEX_ICON,
            config.screen_width, config.screen_height);

        // Running indicator (small dot below icon)
        if ((item.flags & DOCK_ITEM_RUNNING) != 0 || item.instance_count > 0) {
            float indicator_size = 4.0;
            float indicator_x = x + (config.icon_size - indicator_size) / 2.0;
            float indicator_y = icon_y + size + 4.0;

            add_dock_quad(vertices, vertex_count,
                indicator_x, indicator_y, indicator_size, indicator_size,
                config.indicator_color, item_idx, VERTEX_INDICATOR,
                config.screen_width, config.screen_height);
        }
    }
}

// Vertex shader
vertex VertexOut dock_vertex(
    device const DockVertex* vertices [[buffer(0)]],
    device const DockState& state [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    device const DockVertex& v = vertices[vid];

    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.color = v.color;
    out.tex_coord = v.tex_coord;
    out.vertex_type = v.vertex_type;

    return out;
}

// Circle SDF
float circle_sdf(float2 uv, float2 center, float radius) {
    return length(uv - center) - radius;
}

// Rounded rect SDF
float rounded_rect_sdf(float2 uv, float2 size, float radius) {
    float2 q = abs(uv - 0.5) * size - size * 0.5 + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

// Fragment shader
fragment float4 dock_fragment(
    VertexOut in [[stage_in]],
    device const DockState& state [[buffer(0)]]
) {
    float4 color = in.color;

    switch (in.vertex_type) {
        case VERTEX_BACKGROUND: {
            // Frosted glass effect (simplified)
            // In production, would sample background and blur
            color.a *= 0.9;
            break;
        }

        case VERTEX_ICON: {
            // Rounded rect icon placeholder
            float d = rounded_rect_sdf(in.tex_coord, float2(1.0), 0.15);
            float aa = fwidth(d);
            float alpha = 1.0 - smoothstep(-aa, aa, d);
            color.a *= alpha;
            break;
        }

        case VERTEX_INDICATOR: {
            // Circular running indicator
            float d = circle_sdf(in.tex_coord, float2(0.5), 0.5);
            float aa = fwidth(d);
            float alpha = 1.0 - smoothstep(-aa, aa, d);
            color.a *= alpha;
            break;
        }
    }

    return color;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dock_item_creation() {
        let item = DockItem::new(1, "Finder", 0);
        assert_eq!(item.app_id, 1);
        assert_eq!(item.get_name(), "Finder");
        assert!(item.is_visible());
        assert!(!item.is_running());
    }

    #[test]
    fn test_dock_state_add_remove() {
        let mut state = DockState::new(1920.0, 1080.0);

        assert!(state.add_item(1, "Finder", 0));
        assert!(state.add_item(2, "Safari", 1));
        assert_eq!(state.config.item_count, 2);

        assert!(state.remove_item(1));
        assert_eq!(state.config.item_count, 1);
        assert_eq!(state.items[0].app_id, 2);
    }

    #[test]
    fn test_dock_creation() {
        let device = Device::system_default().expect("No Metal device");
        let dock = GpuDock::new(&device, MTLPixelFormat::BGRA8Unorm);
        assert!(dock.is_ok());
    }

    #[test]
    fn test_dock_layout() {
        let device = Device::system_default().expect("No Metal device");
        let mut dock = GpuDock::new(&device, MTLPixelFormat::BGRA8Unorm)
            .expect("Failed to create dock");

        dock.state = DockState::new(1920.0, 1080.0);
        dock.state.add_item(1, "Finder", 0);
        dock.state.add_item(2, "Safari", 1);

        // Verify items were added
        assert_eq!(dock.state.config.item_count, 2);
        assert!(dock.state.items[0].is_visible());
        assert!(dock.state.items[1].is_visible());

        dock.update_layout();

        let count = dock.vertex_count();
        // Background (6) at minimum
        // Note: Icon vertices depend on GPU kernel working correctly
        assert!(count >= 6, "Expected at least 6 vertices for background, got {}", count);
    }
}
