// Issue #13: Input Pipeline - IOKit to GPU Ring Buffer
//
// Handles input from HID devices (keyboard, mouse) and writes to a GPU-accessible
// ring buffer that the compute kernel reads directly.

use super::memory::{InputEvent, InputEventType, InputQueue};
use metal::*;
use std::mem;
#[allow(unused_imports)]
use std::sync::atomic::{AtomicU32, Ordering};

/// Input handler that bridges IOKit events to GPU memory
pub struct InputHandler {
    queue_buffer: Buffer,
}

impl InputHandler {
    /// Create a new input handler with a GPU-accessible ring buffer
    pub fn new(device: &Device) -> Self {
        let queue_buffer = device.new_buffer(
            mem::size_of::<InputQueue>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize the queue
        unsafe {
            let ptr = queue_buffer.contents() as *mut InputQueue;
            std::ptr::write(ptr, InputQueue::new());
        }

        Self { queue_buffer }
    }

    /// Get the ring buffer for binding to compute kernel
    pub fn buffer(&self) -> &Buffer {
        &self.queue_buffer
    }

    /// Get mutable access to the input queue
    fn queue_mut(&self) -> &mut InputQueue {
        unsafe { &mut *(self.queue_buffer.contents() as *mut InputQueue) }
    }

    /// Get read access to the input queue
    fn queue(&self) -> &InputQueue {
        unsafe { &*(self.queue_buffer.contents() as *const InputQueue) }
    }

    /// Push an event to the queue (thread-safe via atomic head)
    fn push_event(&self, event: InputEvent) -> bool {
        let queue = self.queue_mut();
        queue.push(event)
    }

    /// Push a mouse move event to the queue
    pub fn push_mouse_move(&self, x: f32, y: f32, dx: f32, dy: f32) {
        let event = InputEvent::mouse_move(x, y, dx, dy);
        self.push_event(event);
    }

    /// Push a mouse button event to the queue
    pub fn push_mouse_button(&self, button: u16, pressed: bool, x: f32, y: f32) {
        let event = if pressed {
            InputEvent::mouse_down(button, x, y)
        } else {
            InputEvent::mouse_up(button, x, y)
        };
        self.push_event(event);
    }

    /// Push a key event to the queue
    pub fn push_key(&self, keycode: u16, pressed: bool, modifiers: u32) {
        let event = if pressed {
            InputEvent::key_down(keycode, modifiers)
        } else {
            InputEvent::key_up(keycode, modifiers)
        };
        self.push_event(event);
    }

    /// Push a scroll event to the queue
    pub fn push_scroll(&self, x: f32, y: f32, dx: f32, dy: f32) {
        let event = InputEvent {
            event_type: InputEventType::MouseScroll as u16,
            keycode: 0,
            position: [x, y],
            delta: [dx, dy],
            modifiers: 0,
            timestamp: 0,
        };
        self.push_event(event);
    }

    /// Get the number of pending events in the queue
    pub fn pending_count(&self) -> usize {
        self.queue().pending_count()
    }

    /// Clear all pending events (for testing)
    pub fn clear(&self) {
        let queue = self.queue_mut();
        queue.head = 0;
        queue.tail = 0;
    }

    /// Drain events from the queue (for debugging/testing)
    pub fn drain_events(&self, max: usize) -> Vec<InputEvent> {
        self.queue_mut().drain(max)
    }
}

/// Simulated input source for testing
pub struct TestInputSource {
    handler: InputHandler,
}

impl TestInputSource {
    pub fn new(device: &Device) -> Self {
        Self {
            handler: InputHandler::new(device),
        }
    }

    /// Simulate a burst of N mouse move events
    pub fn simulate_mouse_moves(&self, count: usize) {
        for i in 0..count {
            let t = i as f32 / count as f32;
            self.handler.push_mouse_move(
                t,           // x: 0.0 to 1.0
                t * 0.5,     // y: 0.0 to 0.5
                0.01,        // dx
                0.005,       // dy
            );
        }
    }

    /// Simulate typing a string
    pub fn simulate_typing(&self, text: &str) {
        for c in text.chars() {
            // Convert char to keycode (simplified - ASCII only)
            let keycode = c as u16;
            let modifiers = if c.is_uppercase() { 0x01 } else { 0 }; // Shift

            // Key down
            self.handler.push_key(keycode, true, modifiers);
            // Key up
            self.handler.push_key(keycode, false, modifiers);
        }
    }

    /// Simulate a mouse click at position
    pub fn simulate_click(&self, x: f32, y: f32, button: u16) {
        self.handler.push_mouse_button(button, true, x, y);
        self.handler.push_mouse_button(button, false, x, y);
    }

    /// Simulate a drag from one point to another
    pub fn simulate_drag(&self, from_x: f32, from_y: f32, to_x: f32, to_y: f32, steps: usize) {
        // Mouse down
        self.handler.push_mouse_button(0, true, from_x, from_y);

        // Move events
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let x = from_x + (to_x - from_x) * t;
            let y = from_y + (to_y - from_y) * t;
            let dx = (to_x - from_x) / steps as f32;
            let dy = (to_y - from_y) / steps as f32;
            self.handler.push_mouse_move(x, y, dx, dy);
        }

        // Mouse up
        self.handler.push_mouse_button(0, false, to_x, to_y);
    }

    pub fn handler(&self) -> &InputHandler {
        &self.handler
    }
}

/// Modifier key flags (matching HID conventions)
pub mod modifiers {
    pub const SHIFT: u32 = 0x01;
    pub const CONTROL: u32 = 0x02;
    pub const ALT: u32 = 0x04;
    pub const COMMAND: u32 = 0x08;
    pub const CAPS_LOCK: u32 = 0x10;
}

/// Common HID keycodes
pub mod keycodes {
    pub const KEY_A: u16 = 0x04;
    pub const KEY_B: u16 = 0x05;
    pub const KEY_C: u16 = 0x06;
    pub const KEY_ENTER: u16 = 0x28;
    pub const KEY_ESCAPE: u16 = 0x29;
    pub const KEY_BACKSPACE: u16 = 0x2A;
    pub const KEY_TAB: u16 = 0x2B;
    pub const KEY_SPACE: u16 = 0x2C;
    pub const KEY_LEFT: u16 = 0x50;
    pub const KEY_RIGHT: u16 = 0x4F;
    pub const KEY_UP: u16 = 0x52;
    pub const KEY_DOWN: u16 = 0x51;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_handler_creation() {
        let device = Device::system_default().expect("No Metal device");
        let handler = InputHandler::new(&device);
        assert_eq!(handler.pending_count(), 0);
    }

    #[test]
    fn test_push_and_count() {
        let device = Device::system_default().expect("No Metal device");
        let handler = InputHandler::new(&device);

        handler.push_mouse_move(0.5, 0.5, 0.01, 0.01);
        assert_eq!(handler.pending_count(), 1);

        handler.push_key(keycodes::KEY_A, true, 0);
        assert_eq!(handler.pending_count(), 2);
    }

    #[test]
    fn test_clear() {
        let device = Device::system_default().expect("No Metal device");
        let handler = InputHandler::new(&device);

        handler.push_mouse_move(0.5, 0.5, 0.01, 0.01);
        handler.push_mouse_move(0.6, 0.6, 0.01, 0.01);
        assert_eq!(handler.pending_count(), 2);

        handler.clear();
        assert_eq!(handler.pending_count(), 0);
    }
}
