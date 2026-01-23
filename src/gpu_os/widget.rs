// Issue #15: Widget System - Compressed State & Branchless Dispatch
//
// Widget state management with compressed structures and branchless type dispatch
// for SIMD-friendly processing.

use super::memory::WidgetCompact;
use metal::*;
use std::mem;
#[allow(unused_imports)]
use std::time::Instant;

/// Widget types
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum WidgetType {
    Container = 0,
    Button = 1,
    Text = 2,
    Image = 3,
    Scroll = 4,
    Slider = 5,
    Checkbox = 6,
    Input = 7,
}

/// Widget state flags
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct WidgetFlags(pub u8);

impl WidgetFlags {
    pub const VISIBLE: u8 = 0x01;
    pub const ENABLED: u8 = 0x02;
    pub const FOCUSABLE: u8 = 0x04;
    pub const HOVERED: u8 = 0x08;
    pub const PRESSED: u8 = 0x10;
    pub const FOCUSED: u8 = 0x20;

    pub fn new() -> Self {
        Self(Self::VISIBLE | Self::ENABLED)
    }

    pub fn is_visible(&self) -> bool {
        self.0 & Self::VISIBLE != 0
    }

    pub fn is_enabled(&self) -> bool {
        self.0 & Self::ENABLED != 0
    }

    pub fn is_hovered(&self) -> bool {
        self.0 & Self::HOVERED != 0
    }

    pub fn is_pressed(&self) -> bool {
        self.0 & Self::PRESSED != 0
    }

    pub fn is_focused(&self) -> bool {
        self.0 & Self::FOCUSED != 0
    }
}

impl Default for WidgetFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// Hit test parameters for the compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct HitTestParams {
    widget_count: u32,
    cursor_x: f32,
    cursor_y: f32,
    _padding: u32,
}

/// Hit test result from GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct HitTestResult {
    hit_count: u32,
    topmost_widget: u32,
    topmost_z: u32,
    _padding: u32,
}

const WIDGET_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct WidgetCompact {
    ushort4 bounds;
    uint packed_colors;
    ushort packed_style;
    ushort parent_id;
    ushort first_child;
    ushort next_sibling;
    ushort z_order;
    ushort _padding;
};

struct HitTestParams {
    uint widget_count;
    float cursor_x;
    float cursor_y;
    uint _padding;
};

struct HitTestResult {
    atomic_uint hit_count;
    atomic_uint topmost_widget;
    atomic_uint topmost_z;
    uint _padding;
};

inline float f16_to_float(ushort bits) {
    return float(as_type<half>(bits));
}

inline bool is_visible(ushort packed_style) {
    return (packed_style & 0x1) != 0;
}

inline bool point_in_rect(float2 point, ushort4 bounds) {
    float x = f16_to_float(bounds.x);
    float y = f16_to_float(bounds.y);
    float w = f16_to_float(bounds.z);
    float h = f16_to_float(bounds.w);
    return point.x >= x && point.x <= x + w &&
           point.y >= y && point.y <= y + h;
}

kernel void hit_test_kernel(
    device WidgetCompact* widgets [[buffer(0)]],
    constant HitTestParams& params [[buffer(1)]],
    device HitTestResult* result [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Initialize result on thread 0
    if (tid == 0) {
        atomic_store_explicit(&result->hit_count, 0, memory_order_relaxed);
        atomic_store_explicit(&result->topmost_widget, 0xFFFFFFFF, memory_order_relaxed);
        // Use 0xFFFFFFFF as "no z-order set" sentinel so z_order=0 can still win
        atomic_store_explicit(&result->topmost_z, 0xFFFFFFFF, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid < params.widget_count) {
        WidgetCompact w = widgets[tid];

        if (is_visible(w.packed_style)) {
            float2 cursor = float2(params.cursor_x, params.cursor_y);
            bool hit = point_in_rect(cursor, w.bounds);

            if (hit) {
                atomic_fetch_add_explicit(&result->hit_count, 1, memory_order_relaxed);

                // Update topmost - first hit always wins, then higher z-order wins
                uint current_z = atomic_load_explicit(&result->topmost_z, memory_order_relaxed);
                uint my_z = uint(w.z_order);
                // If current_z is sentinel (0xFFFFFFFF) or my_z is higher, try to update
                while (current_z == 0xFFFFFFFF || my_z > current_z) {
                    if (atomic_compare_exchange_weak_explicit(
                        &result->topmost_z, &current_z, my_z,
                        memory_order_relaxed, memory_order_relaxed)) {
                        atomic_store_explicit(&result->topmost_widget, tid, memory_order_relaxed);
                        break;
                    }
                    // Reload and check again if exchange failed
                    if (current_z != 0xFFFFFFFF && my_z <= current_z) break;
                }
            }
        }
    }
}

// Sort kernel for z-order sorting (bitonic sort)
kernel void sort_by_z_kernel(
    device WidgetCompact* widgets [[buffer(0)]],
    constant uint& widget_count [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    constant uint& step [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid >= widget_count) return;

    uint partner = tid ^ step;

    if (partner > tid && partner < widget_count) {
        bool ascending = ((tid & stage) == 0);
        ushort z_a = widgets[tid].z_order;
        ushort z_b = widgets[partner].z_order;

        bool should_swap = ascending ? (z_a > z_b) : (z_a < z_b);

        if (should_swap) {
            WidgetCompact temp = widgets[tid];
            widgets[tid] = widgets[partner];
            widgets[partner] = temp;
        }
    }
}
"#;

/// Widget manager handles hit testing and state updates
pub struct WidgetManager {
    hit_test_pipeline: ComputePipelineState,
    sort_pipeline: ComputePipelineState,
    params_buffer: Buffer,
    result_buffer: Buffer,
    sort_params_buffer: Buffer,
}

impl WidgetManager {
    /// Create a new widget manager
    pub fn new(device: &Device) -> Result<Self, String> {
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(WIDGET_SHADER, &options)
            .map_err(|e| format!("Failed to compile widget shader: {}", e))?;

        let hit_test_fn = library
            .get_function("hit_test_kernel", None)
            .map_err(|e| format!("Failed to get hit_test_kernel: {}", e))?;

        let hit_test_pipeline = device
            .new_compute_pipeline_state_with_function(&hit_test_fn)
            .map_err(|e| format!("Failed to create hit test pipeline: {}", e))?;

        let sort_fn = library
            .get_function("sort_by_z_kernel", None)
            .map_err(|e| format!("Failed to get sort_by_z_kernel: {}", e))?;

        let sort_pipeline = device
            .new_compute_pipeline_state_with_function(&sort_fn)
            .map_err(|e| format!("Failed to create sort pipeline: {}", e))?;

        let params_buffer = device.new_buffer(
            mem::size_of::<HitTestParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let result_buffer = device.new_buffer(
            mem::size_of::<HitTestResult>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let sort_params_buffer = device.new_buffer(
            16, // 4 u32 values
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            hit_test_pipeline,
            sort_pipeline,
            params_buffer,
            result_buffer,
            sort_params_buffer,
        })
    }

    /// Perform hit testing for cursor position
    pub fn hit_test(
        &self,
        encoder: &ComputeCommandEncoderRef,
        widgets: &Buffer,
        widget_count: usize,
        cursor_x: f32,
        cursor_y: f32,
    ) {
        // Guard against empty widget list
        if widget_count == 0 {
            return;
        }
        // Update params
        unsafe {
            let ptr = self.params_buffer.contents() as *mut HitTestParams;
            *ptr = HitTestParams {
                widget_count: widget_count as u32,
                cursor_x,
                cursor_y,
                _padding: 0,
            };
        }

        encoder.set_compute_pipeline_state(&self.hit_test_pipeline);
        encoder.set_buffer(0, Some(widgets), 0);
        encoder.set_buffer(1, Some(&self.params_buffer), 0);
        encoder.set_buffer(2, Some(&self.result_buffer), 0);

        let threads = widget_count.min(1024);
        let threadgroup_size = MTLSize::new(threads as u64, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Hit test synchronously (for testing)
    pub fn hit_test_sync(
        &self,
        queue: &CommandQueue,
        widgets: &Buffer,
        widget_count: usize,
        cursor_x: f32,
        cursor_y: f32,
    ) -> (usize, Option<usize>) {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.hit_test(&encoder, widgets, widget_count, cursor_x, cursor_y);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        let result = unsafe { *(self.result_buffer.contents() as *const HitTestResult) };

        let topmost = if result.topmost_widget == 0xFFFFFFFF {
            None
        } else {
            Some(result.topmost_widget as usize)
        };

        (result.hit_count as usize, topmost)
    }

    /// Sort widgets by z-order for correct rendering
    pub fn sort_by_z_order(
        &self,
        encoder: &ComputeCommandEncoderRef,
        widgets: &Buffer,
        widget_count: usize,
    ) {
        // Bitonic sort requires multiple passes
        let mut k = 2u32;
        while k <= widget_count.next_power_of_two() as u32 {
            let mut j = k / 2;
            while j > 0 {
                unsafe {
                    let ptr = self.sort_params_buffer.contents() as *mut u32;
                    *ptr.add(0) = widget_count as u32;
                    *ptr.add(1) = k;
                    *ptr.add(2) = j;
                    *ptr.add(3) = 0;
                }

                encoder.set_compute_pipeline_state(&self.sort_pipeline);
                encoder.set_buffer(0, Some(widgets), 0);
                encoder.set_buffer(1, Some(&self.sort_params_buffer), 0);
                encoder.set_buffer(2, Some(&self.sort_params_buffer), 4);
                encoder.set_buffer(3, Some(&self.sort_params_buffer), 8);

                let threads = widget_count.min(1024);
                let threadgroup_size = MTLSize::new(threads as u64, 1, 1);
                let threadgroups = MTLSize::new(1, 1, 1);

                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);

                j /= 2;
            }
            k *= 2;
        }
    }

    /// Sort widgets by z-order synchronously (waits for each pass)
    pub fn sort_by_z_order_sync(
        &self,
        queue: &CommandQueue,
        widgets: &Buffer,
        widget_count: usize,
    ) {
        // Bitonic sort requires multiple passes with synchronization
        let mut k = 2u32;
        while k <= widget_count.next_power_of_two() as u32 {
            let mut j = k / 2;
            while j > 0 {
                unsafe {
                    let ptr = self.sort_params_buffer.contents() as *mut u32;
                    *ptr.add(0) = widget_count as u32;
                    *ptr.add(1) = k;
                    *ptr.add(2) = j;
                    *ptr.add(3) = 0;
                }

                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(&self.sort_pipeline);
                encoder.set_buffer(0, Some(widgets), 0);
                encoder.set_buffer(1, Some(&self.sort_params_buffer), 0);
                encoder.set_buffer(2, Some(&self.sort_params_buffer), 4);
                encoder.set_buffer(3, Some(&self.sort_params_buffer), 8);

                let threads = widget_count.min(1024);
                let threadgroup_size = MTLSize::new(threads as u64, 1, 1);
                let threadgroups = MTLSize::new(1, 1, 1);

                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                j /= 2;
            }
            k *= 2;
        }
    }
}

/// Widget builder for creating widgets with a fluent API
pub struct WidgetBuilder {
    widget: WidgetCompact,
}

impl WidgetBuilder {
    pub fn new(widget_type: WidgetType) -> Self {
        let mut widget = WidgetCompact::default();
        widget.set_widget_type(widget_type as u8);
        widget.set_flags(WidgetFlags::VISIBLE | WidgetFlags::ENABLED);
        Self { widget }
    }

    pub fn bounds(mut self, x: f32, y: f32, w: f32, h: f32) -> Self {
        self.widget.set_bounds(x, y, w, h);
        self
    }

    pub fn background_color(mut self, r: f32, g: f32, b: f32) -> Self {
        let border = self.widget.border_color();
        self.widget.set_colors([r, g, b], [border[0], border[1], border[2]]);
        self
    }

    pub fn border_color(mut self, r: f32, g: f32, b: f32) -> Self {
        let bg = self.widget.background_color();
        self.widget.set_colors([bg[0], bg[1], bg[2]], [r, g, b]);
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        // Convert to 0-15 range (0-60 pixels)
        let value = ((radius / 4.0).clamp(0.0, 15.0)) as u8;
        self.widget.set_corner_radius(value);
        self
    }

    pub fn border_width(mut self, width: f32) -> Self {
        let value = (width.clamp(0.0, 15.0)) as u8;
        self.widget.set_border_width(value);
        self
    }

    pub fn z_order(mut self, z: u16) -> Self {
        self.widget.z_order = z;
        self
    }

    pub fn parent(mut self, parent_id: u16) -> Self {
        self.widget.parent_id = parent_id;
        self
    }

    pub fn build(self) -> WidgetCompact {
        self.widget
    }
}

/// Extension trait for WidgetCompact
pub trait WidgetExt {
    fn widget_type(&self) -> WidgetType;
    fn flags(&self) -> WidgetFlags;
    fn corner_radius(&self) -> f32;
    fn border_width(&self) -> f32;
}

impl WidgetExt for WidgetCompact {
    fn widget_type(&self) -> WidgetType {
        match self.get_widget_type() {
            0 => WidgetType::Container,
            1 => WidgetType::Button,
            2 => WidgetType::Text,
            3 => WidgetType::Image,
            4 => WidgetType::Scroll,
            5 => WidgetType::Slider,
            6 => WidgetType::Checkbox,
            7 => WidgetType::Input,
            _ => WidgetType::Container,
        }
    }

    fn flags(&self) -> WidgetFlags {
        WidgetFlags(self.get_flags())
    }

    fn corner_radius(&self) -> f32 {
        self.get_corner_radius()
    }

    fn border_width(&self) -> f32 {
        self.get_border_width()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_builder() {
        let widget = WidgetBuilder::new(WidgetType::Button)
            .bounds(0.1, 0.1, 0.2, 0.05)
            .background_color(0.2, 0.4, 0.8)
            .corner_radius(4.0)
            .z_order(10)
            .build();

        assert_eq!(widget.z_order, 10);
        assert_eq!(widget.widget_type(), WidgetType::Button);
    }

    #[test]
    fn test_widget_flags() {
        let flags = WidgetFlags::new();
        assert!(flags.is_visible());
        assert!(flags.is_enabled());
        assert!(!flags.is_hovered());
    }
}
