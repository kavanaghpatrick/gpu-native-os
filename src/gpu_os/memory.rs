// Issue #12: Memory Architecture - Unified GPU Buffers
//
// This module defines all data structures and memory layout for the GPU-Native OS.
// All buffers use MTLStorageModeShared for unified memory access.

use half::f16;
use metal::*;
use std::mem;

/// Compact widget structure - 24 bytes
/// Stores all widget state in a GPU-efficient format
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct WidgetCompact {
    /// Bounds as half-precision floats (x, y, width, height)
    pub bounds: [u16; 4],           // 8 bytes
    /// Packed colors: bg[16 bits] | border[16 bits]
    pub packed_colors: u32,          // 4 bytes
    /// Packed style: border_width[4] | corner_radius[4] | type[4] | flags[4]
    pub packed_style: u16,           // 2 bytes
    /// Parent widget index (0 = root)
    pub parent_id: u16,              // 2 bytes
    /// First child widget index (0 = none)
    pub first_child: u16,            // 2 bytes
    /// Next sibling widget index (0 = none)
    pub next_sibling: u16,           // 2 bytes
    /// Z-order for overlap handling
    pub z_order: u16,                // 2 bytes
    /// Padding for alignment
    pub _padding: u16,               // 2 bytes
}

impl Default for WidgetCompact {
    fn default() -> Self {
        Self {
            bounds: [0; 4],
            packed_colors: 0,
            packed_style: 0x0003, // Type=0 (Container), Flags=3 (visible|enabled)
            parent_id: 0,
            first_child: 0,
            next_sibling: 0,
            z_order: 0,
            _padding: 0,
        }
    }
}

/// Input event from HID devices - 24 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct InputEvent {
    /// Event type (see InputEventType)
    pub event_type: u16,
    /// HID keycode or mouse button
    pub keycode: u16,
    /// Cursor position (normalized 0-1)
    pub position: [f32; 2],
    /// Movement delta
    pub delta: [f32; 2],
    /// Modifier keys (shift, ctrl, alt, cmd)
    pub modifiers: u32,
    /// Frame-relative timestamp
    pub timestamp: u32,
}

/// Input event types
#[repr(u16)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum InputEventType {
    None = 0,
    MouseMove = 1,
    MouseDown = 2,
    MouseUp = 3,
    MouseScroll = 4,
    KeyDown = 5,
    KeyUp = 6,
    KeyRepeat = 7,
    MouseDoubleClick = 8,  // Issue #122: Double-click detection
    MouseTripleClick = 9,  // For text selection (select paragraph)
}

/// Ring buffer for input events
#[repr(C)]
pub struct InputQueue {
    /// Write position (incremented by CPU/IOKit)
    pub head: u32,
    /// Read position (incremented by GPU)
    pub tail: u32,
    /// Padding for alignment
    _padding: [u32; 2],
    /// Ring buffer of events
    pub events: [InputEvent; 256],
}

/// Indirect draw arguments for GPU-driven rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DrawArguments {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

/// Per-frame state
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct FrameState {
    pub frame_number: u32,
    pub time: f32,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub focused_widget: u32,
    pub hovered_widget: u32,
    pub modifiers: u32,
    pub _padding: u32,
}

/// GPU Memory Manager - handles all buffer allocation
pub struct GpuMemory {
    pub widget_buffer: Buffer,
    pub input_queue_buffer: Buffer,
    pub vertex_buffer: Buffer,
    pub draw_args_buffer: Buffer,
    pub frame_state_buffer: Buffer,
    max_widgets: usize,
    current_widget_count: std::cell::Cell<usize>,
}

impl GpuMemory {
    /// Create a new GPU memory manager with all required buffers
    pub fn new(device: &Device, max_widgets: usize) -> Self {
        let widget_buffer_size = max_widgets * WidgetCompact::SIZE;
        let vertex_buffer_size = max_widgets * 6 * 64; // 6 vertices per widget, 64 bytes per vertex
        let input_queue_size = mem::size_of::<InputQueue>();
        let draw_args_size = mem::size_of::<DrawArguments>();
        let frame_state_size = mem::size_of::<FrameState>();

        // Use shared storage mode for unified memory (CPU and GPU accessible)
        let options = MTLResourceOptions::StorageModeShared;

        let widget_buffer = device.new_buffer(widget_buffer_size as u64, options);
        let input_queue_buffer = device.new_buffer(input_queue_size as u64, options);
        let vertex_buffer = device.new_buffer(vertex_buffer_size as u64, options);
        let draw_args_buffer = device.new_buffer(draw_args_size as u64, options);
        let frame_state_buffer = device.new_buffer(frame_state_size as u64, options);

        // Initialize input queue with zeros
        unsafe {
            let ptr = input_queue_buffer.contents() as *mut InputQueue;
            std::ptr::write_bytes(ptr, 0, 1);
        }

        // Initialize draw args
        unsafe {
            let ptr = draw_args_buffer.contents() as *mut DrawArguments;
            (*ptr) = DrawArguments {
                vertex_count: 0,
                instance_count: 1,
                vertex_start: 0,
                base_instance: 0,
            };
        }

        // Initialize frame state
        unsafe {
            let ptr = frame_state_buffer.contents() as *mut FrameState;
            std::ptr::write_bytes(ptr, 0, 1);
        }

        Self {
            widget_buffer,
            input_queue_buffer,
            vertex_buffer,
            draw_args_buffer,
            frame_state_buffer,
            max_widgets,
            current_widget_count: std::cell::Cell::new(0),
        }
    }

    /// Get current widget count
    pub fn widget_count(&self) -> usize {
        self.current_widget_count.get()
    }

    /// Get the total memory usage in bytes
    pub fn total_memory_usage(&self) -> usize {
        self.widget_buffer.length() as usize
            + self.input_queue_buffer.length() as usize
            + self.vertex_buffer.length() as usize
            + self.draw_args_buffer.length() as usize
            + self.frame_state_buffer.length() as usize
    }

    /// Write widget data to the GPU buffer
    pub fn write_widgets(&self, widgets: &[WidgetCompact]) {
        assert!(widgets.len() <= self.max_widgets, "Too many widgets");
        unsafe {
            let ptr = self.widget_buffer.contents() as *mut WidgetCompact;
            std::ptr::copy_nonoverlapping(widgets.as_ptr(), ptr, widgets.len());
        }
        self.current_widget_count.set(widgets.len());
    }

    /// Read widget data back from GPU buffer
    pub fn read_widgets(&self, count: usize) -> Vec<WidgetCompact> {
        let count = count.min(self.max_widgets);
        let mut widgets = Vec::with_capacity(count);
        unsafe {
            let ptr = self.widget_buffer.contents() as *const WidgetCompact;
            for i in 0..count {
                widgets.push(*ptr.add(i));
            }
        }
        widgets
    }

    /// Get a mutable reference to the input queue
    pub fn input_queue_mut(&self) -> &mut InputQueue {
        unsafe { &mut *(self.input_queue_buffer.contents() as *mut InputQueue) }
    }

    /// Get a reference to the input queue
    pub fn input_queue(&self) -> &InputQueue {
        unsafe { &*(self.input_queue_buffer.contents() as *const InputQueue) }
    }

    /// Get a mutable reference to the draw arguments
    pub fn draw_args_mut(&self) -> &mut DrawArguments {
        unsafe { &mut *(self.draw_args_buffer.contents() as *mut DrawArguments) }
    }

    /// Get the draw arguments
    pub fn draw_args(&self) -> DrawArguments {
        unsafe { *(self.draw_args_buffer.contents() as *const DrawArguments) }
    }

    /// Get a mutable reference to the frame state
    pub fn frame_state_mut(&self) -> &mut FrameState {
        unsafe { &mut *(self.frame_state_buffer.contents() as *mut FrameState) }
    }

    /// Get the frame state
    pub fn frame_state(&self) -> FrameState {
        unsafe { *(self.frame_state_buffer.contents() as *const FrameState) }
    }
}

impl WidgetCompact {
    pub const SIZE: usize = 24;

    /// Create a new widget with the given bounds (in normalized 0-1 coordinates)
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        let mut widget = Self::default();
        widget.set_bounds(x, y, width, height);
        widget
    }

    /// Set bounds using f32 values (converted to half-precision u16)
    pub fn set_bounds(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.bounds[0] = f32_to_f16(x);
        self.bounds[1] = f32_to_f16(y);
        self.bounds[2] = f32_to_f16(width);
        self.bounds[3] = f32_to_f16(height);
    }

    /// Get bounds as f32 values
    pub fn get_bounds(&self) -> [f32; 4] {
        [
            f16_to_f32(self.bounds[0]),
            f16_to_f32(self.bounds[1]),
            f16_to_f32(self.bounds[2]),
            f16_to_f32(self.bounds[3]),
        ]
    }

    /// Pack two RGB565 colors into packed_colors
    pub fn set_colors(&mut self, background: [f32; 3], border: [f32; 3]) {
        let bg_packed = rgb_to_rgb565(background);
        let border_packed = rgb_to_rgb565(border);
        self.packed_colors = ((bg_packed as u32) << 16) | (border_packed as u32);
    }

    /// Unpack background color from packed_colors
    pub fn background_color(&self) -> [f32; 4] {
        let bg_packed = (self.packed_colors >> 16) as u16;
        let [r, g, b] = rgb565_to_rgb(bg_packed);
        [r, g, b, 1.0]
    }

    /// Unpack border color from packed_colors
    pub fn border_color(&self) -> [f32; 4] {
        let border_packed = (self.packed_colors & 0xFFFF) as u16;
        let [r, g, b] = rgb565_to_rgb(border_packed);
        [r, g, b, 1.0]
    }

    /// Set the widget type (stored in packed_style bits 4-7)
    pub fn set_widget_type(&mut self, widget_type: u8) {
        self.packed_style = (self.packed_style & 0xFF0F) | ((widget_type as u16 & 0xF) << 4);
    }

    /// Get the widget type
    pub fn get_widget_type(&self) -> u8 {
        ((self.packed_style >> 4) & 0xF) as u8
    }

    /// Set flags (stored in packed_style bits 0-3)
    pub fn set_flags(&mut self, flags: u8) {
        self.packed_style = (self.packed_style & 0xFFF0) | (flags as u16 & 0xF);
    }

    /// Get flags
    pub fn get_flags(&self) -> u8 {
        (self.packed_style & 0xF) as u8
    }

    /// Set corner radius (stored in packed_style bits 8-11)
    pub fn set_corner_radius(&mut self, radius: u8) {
        self.packed_style = (self.packed_style & 0xF0FF) | ((radius as u16 & 0xF) << 8);
    }

    /// Get corner radius (0-15 maps to 0-60 pixels)
    pub fn get_corner_radius(&self) -> f32 {
        let raw = ((self.packed_style >> 8) & 0xF) as f32;
        raw * 4.0 // Scale: 0-15 → 0-60
    }

    /// Set border width (stored in packed_style bits 12-15)
    pub fn set_border_width(&mut self, width: u8) {
        self.packed_style = (self.packed_style & 0x0FFF) | ((width as u16 & 0xF) << 12);
    }

    /// Get border width (0-15 maps to 0-15 pixels)
    pub fn get_border_width(&self) -> f32 {
        ((self.packed_style >> 12) & 0xF) as f32
    }
}

impl InputEvent {
    // Note: PRD says 24 bytes but actual size is 28 due to alignment
    // (u16, u16, f32[2], f32[2], u32, u32) = 28 bytes with natural alignment
    pub const SIZE: usize = 28;

    /// Create a new mouse move event
    pub fn mouse_move(x: f32, y: f32, dx: f32, dy: f32) -> Self {
        Self {
            event_type: InputEventType::MouseMove as u16,
            keycode: 0,
            position: [x, y],
            delta: [dx, dy],
            modifiers: 0,
            timestamp: 0,
        }
    }

    /// Create a new mouse down event
    pub fn mouse_down(button: u16, x: f32, y: f32) -> Self {
        Self {
            event_type: InputEventType::MouseDown as u16,
            keycode: button,
            position: [x, y],
            delta: [0.0, 0.0],
            modifiers: 0,
            timestamp: 0,
        }
    }

    /// Create a new mouse up event
    pub fn mouse_up(button: u16, x: f32, y: f32) -> Self {
        Self {
            event_type: InputEventType::MouseUp as u16,
            keycode: button,
            position: [x, y],
            delta: [0.0, 0.0],
            modifiers: 0,
            timestamp: 0,
        }
    }

    /// Create a new key down event
    pub fn key_down(keycode: u16, modifiers: u32) -> Self {
        Self {
            event_type: InputEventType::KeyDown as u16,
            keycode,
            position: [0.0, 0.0],
            delta: [0.0, 0.0],
            modifiers,
            timestamp: 0,
        }
    }

    /// Create a new key up event
    pub fn key_up(keycode: u16, modifiers: u32) -> Self {
        Self {
            event_type: InputEventType::KeyUp as u16,
            keycode,
            position: [0.0, 0.0],
            delta: [0.0, 0.0],
            modifiers,
            timestamp: 0,
        }
    }

    /// Create a new double-click event (Issue #122)
    pub fn mouse_double_click(button: u16, x: f32, y: f32) -> Self {
        Self {
            event_type: InputEventType::MouseDoubleClick as u16,
            keycode: button,
            position: [x, y],
            delta: [0.0, 0.0],
            modifiers: 0,
            timestamp: 0,
        }
    }

    /// Create a new triple-click event (for text selection)
    pub fn mouse_triple_click(button: u16, x: f32, y: f32) -> Self {
        Self {
            event_type: InputEventType::MouseTripleClick as u16,
            keycode: button,
            position: [x, y],
            delta: [0.0, 0.0],
            modifiers: 0,
            timestamp: 0,
        }
    }
}

impl InputQueue {
    pub const CAPACITY: usize = 256;

    /// Create a new empty input queue
    pub fn new() -> Self {
        Self {
            head: 0,
            tail: 0,
            _padding: [0; 2],
            events: [InputEvent::default(); 256],
        }
    }

    /// Get the number of pending events
    pub fn pending_count(&self) -> usize {
        (self.head.wrapping_sub(self.tail)) as usize
    }

    /// Push an event to the queue (called from CPU/IOKit)
    pub fn push(&mut self, event: InputEvent) -> bool {
        let count = self.pending_count();
        if count >= Self::CAPACITY {
            return false; // Queue full
        }
        let slot = (self.head as usize) % Self::CAPACITY;
        self.events[slot] = event;
        self.head = self.head.wrapping_add(1);
        true
    }

    /// Pop an event from the queue
    pub fn pop(&mut self) -> Option<InputEvent> {
        if self.pending_count() == 0 {
            return None;
        }
        let slot = (self.tail as usize) % Self::CAPACITY;
        let event = self.events[slot];
        self.tail = self.tail.wrapping_add(1);
        Some(event)
    }

    /// Drain up to `max` events from the queue
    pub fn drain(&mut self, max: usize) -> Vec<InputEvent> {
        let count = self.pending_count().min(max);
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(event) = self.pop() {
                events.push(event);
            }
        }
        events
    }
}

impl Default for InputQueue {
    fn default() -> Self {
        Self::new()
    }
}

// Half-precision float conversion helpers

/// Convert f32 to IEEE 754 half-precision (f16) stored as u16
fn f32_to_f16(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

/// Convert IEEE 754 half-precision (f16) stored as u16 to f32
fn f16_to_f32(value: u16) -> f32 {
    f16::from_bits(value).to_f32()
}

/// Convert RGB floats (0-1) to RGB565
fn rgb_to_rgb565(rgb: [f32; 3]) -> u16 {
    let r = (rgb[0].clamp(0.0, 1.0) * 31.0) as u16;
    let g = (rgb[1].clamp(0.0, 1.0) * 63.0) as u16;
    let b = (rgb[2].clamp(0.0, 1.0) * 31.0) as u16;
    (r << 11) | (g << 5) | b
}

/// Convert RGB565 to RGB floats (0-1)
fn rgb565_to_rgb(packed: u16) -> [f32; 3] {
    let r = ((packed >> 11) & 0x1F) as f32 / 31.0;
    let g = ((packed >> 5) & 0x3F) as f32 / 63.0;
    let b = (packed & 0x1F) as f32 / 31.0;
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_size() {
        assert_eq!(mem::size_of::<WidgetCompact>(), 24);
    }

    #[test]
    fn test_input_event_size() {
        // InputEvent is 28 bytes due to alignment (4 bytes for timestamp alignment)
        assert_eq!(mem::size_of::<InputEvent>(), InputEvent::SIZE);
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0, 0.5, 1.0, -1.0, 0.123, 100.0, -50.0];
        for v in values {
            let f16 = f32_to_f16(v);
            let back = f16_to_f32(f16);
            assert!((v - back).abs() < 0.01, "f16 roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_rgb565_roundtrip() {
        let colors = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.5]];
        for c in colors {
            let packed = rgb_to_rgb565(c);
            let back = rgb565_to_rgb(packed);
            for i in 0..3 {
                assert!((c[i] - back[i]).abs() < 0.05, "RGB565 roundtrip failed");
            }
        }
    }

    #[test]
    fn test_f16_bit_patterns_ieee754() {
        // Verify the half crate produces IEEE 754 binary16 format
        // which is what Metal's as_type<half> expects

        // IEEE 754 binary16 format:
        // - Sign: 1 bit (bit 15)
        // - Exponent: 5 bits (bits 14-10), biased by 15
        // - Mantissa: 10 bits (bits 9-0), implicit leading 1
        // value = (-1)^s * 2^(e-15) * (1 + m/1024) for normal numbers

        // Test 0.5: s=0, e=14 (2^-1), m=0
        // Expected: 0 01110 0000000000 = 0x3800
        let half_bits = f32_to_f16(0.5);
        assert_eq!(half_bits, 0x3800,
            "f16(0.5) should be 0x3800, got 0x{:04X}", half_bits);

        // Test 1.0: s=0, e=15 (2^0), m=0
        // Expected: 0 01111 0000000000 = 0x3C00
        let one_bits = f32_to_f16(1.0);
        assert_eq!(one_bits, 0x3C00,
            "f16(1.0) should be 0x3C00, got 0x{:04X}", one_bits);

        // Test 0.0: all zeros
        // Expected: 0 00000 0000000000 = 0x0000
        let zero_bits = f32_to_f16(0.0);
        assert_eq!(zero_bits, 0x0000,
            "f16(0.0) should be 0x0000, got 0x{:04X}", zero_bits);

        // Test -1.0: s=1, e=15, m=0
        // Expected: 1 01111 0000000000 = 0xBC00
        let neg_one_bits = f32_to_f16(-1.0);
        assert_eq!(neg_one_bits, 0xBC00,
            "f16(-1.0) should be 0xBC00, got 0x{:04X}", neg_one_bits);

        // Test 0.25: s=0, e=13 (2^-2), m=0
        // Expected: 0 01101 0000000000 = 0x3400
        let quarter_bits = f32_to_f16(0.25);
        assert_eq!(quarter_bits, 0x3400,
            "f16(0.25) should be 0x3400, got 0x{:04X}", quarter_bits);

        println!("\n=== f16 IEEE 754 bit pattern verification ===");
        println!("f16(0.0)  = 0x{:04X} (expected 0x0000)", zero_bits);
        println!("f16(0.25) = 0x{:04X} (expected 0x3400)", quarter_bits);
        println!("f16(0.5)  = 0x{:04X} (expected 0x3800)", half_bits);
        println!("f16(1.0)  = 0x{:04X} (expected 0x3C00)", one_bits);
        println!("f16(-1.0) = 0x{:04X} (expected 0xBC00)", neg_one_bits);
        println!("\nAll bit patterns match IEEE 754 binary16.");
        println!("Metal's as_type<half> will interpret these correctly.");
    }

    #[test]
    fn test_f16_precision_for_widgets() {
        // Test precision for typical widget coordinate values
        let typical_values = [
            0.05_f32, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
        ];

        println!("\n=== f16 precision for widget coordinates ===");
        let mut max_error: f32 = 0.0;
        for v in typical_values {
            let bits = f32_to_f16(v);
            let back = f16_to_f32(bits);
            let error = (v - back).abs();
            max_error = max_error.max(error);
            println!("{:.2} -> 0x{:04X} -> {:.6} (error: {:.6})", v, bits, back, error);
        }

        println!("\nMax error: {:.6}", max_error);
        println!("For 2560px display: {:.2} pixels", max_error * 2560.0);

        // f16 has ~0.1% precision in the 0-1 range
        // That's about 2.5 pixels on a 2560px display - perfectly acceptable
        assert!(max_error < 0.002, "f16 precision error too high: {}", max_error);
    }
}
