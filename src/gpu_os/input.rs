// Issue #13: Input Pipeline - IOKit to GPU Ring Buffer
//
// Handles input from HID devices (keyboard, mouse) and writes to a GPU-accessible
// ring buffer that the compute kernel reads directly.

use super::memory::{InputEvent, InputEventType, InputQueue};
use metal::*;
use std::mem;
#[allow(unused_imports)]
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

/// Double-click detection parameters
pub const DOUBLE_CLICK_TIME_MS: u64 = 500;    // Max time between clicks
pub const DOUBLE_CLICK_DISTANCE: f32 = 5.0;    // Max pixel distance between clicks
pub const TRIPLE_CLICK_TIME_MS: u64 = 500;     // Max time after double-click for triple

/// Click state tracking for multi-click detection
#[derive(Clone)]
struct ClickState {
    last_click_time: Option<Instant>,
    last_click_pos: [f32; 2],
    last_button: u16,
    click_count: u8,  // 1 = single, 2 = double, 3 = triple
}

impl Default for ClickState {
    fn default() -> Self {
        Self {
            last_click_time: None,
            last_click_pos: [0.0, 0.0],
            last_button: 0,
            click_count: 0,
        }
    }
}

/// Input handler that bridges IOKit events to GPU memory
pub struct InputHandler {
    queue_buffer: Buffer,
}

/// Input handler with multi-click detection (Issue #122)
/// Wraps InputHandler and automatically detects double/triple clicks
pub struct ClickDetectingInputHandler {
    inner: InputHandler,
    click_state: std::cell::RefCell<ClickState>,
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

impl ClickDetectingInputHandler {
    /// Create a new click-detecting input handler
    pub fn new(device: &Device) -> Self {
        Self {
            inner: InputHandler::new(device),
            click_state: std::cell::RefCell::new(ClickState::default()),
        }
    }

    /// Get the underlying buffer for GPU binding
    pub fn buffer(&self) -> &Buffer {
        self.inner.buffer()
    }

    /// Push a mouse move event
    pub fn push_mouse_move(&self, x: f32, y: f32, dx: f32, dy: f32) {
        self.inner.push_mouse_move(x, y, dx, dy);
    }

    /// Push a mouse button event with automatic multi-click detection
    pub fn push_mouse_button(&self, button: u16, pressed: bool, x: f32, y: f32) {
        if pressed {
            // Check for multi-click
            let mut state = self.click_state.borrow_mut();
            let now = Instant::now();

            let is_multi_click = if let Some(last_time) = state.last_click_time {
                let elapsed_ms = now.duration_since(last_time).as_millis() as u64;
                let dx = x - state.last_click_pos[0];
                let dy = y - state.last_click_pos[1];
                let distance = (dx * dx + dy * dy).sqrt();

                elapsed_ms <= DOUBLE_CLICK_TIME_MS
                    && distance <= DOUBLE_CLICK_DISTANCE
                    && state.last_button == button
            } else {
                false
            };

            if is_multi_click {
                state.click_count = (state.click_count + 1).min(3);
            } else {
                state.click_count = 1;
            }

            state.last_click_time = Some(now);
            state.last_click_pos = [x, y];
            state.last_button = button;

            let click_count = state.click_count;
            drop(state); // Release borrow before pushing events

            // Push the base mouse down event
            self.inner.push_mouse_button(button, true, x, y);

            // Push multi-click events
            match click_count {
                2 => {
                    let event = InputEvent::mouse_double_click(button, x, y);
                    self.inner.push_event(event);
                }
                3 => {
                    let event = InputEvent::mouse_triple_click(button, x, y);
                    self.inner.push_event(event);
                }
                _ => {}
            }
        } else {
            // Mouse up - just forward
            self.inner.push_mouse_button(button, false, x, y);
        }
    }

    /// Push a key event
    pub fn push_key(&self, keycode: u16, pressed: bool, modifiers: u32) {
        self.inner.push_key(keycode, pressed, modifiers);
    }

    /// Push a scroll event
    pub fn push_scroll(&self, x: f32, y: f32, dx: f32, dy: f32) {
        self.inner.push_scroll(x, y, dx, dy);
    }

    /// Get the number of pending events
    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    /// Clear all pending events
    pub fn clear(&self) {
        self.inner.clear();
        *self.click_state.borrow_mut() = ClickState::default();
    }

    /// Drain events from the queue
    pub fn drain_events(&self, max: usize) -> Vec<InputEvent> {
        self.inner.drain_events(max)
    }

    /// Get the current click count (1, 2, or 3)
    pub fn current_click_count(&self) -> u8 {
        self.click_state.borrow().click_count
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

    // Issue #122: Double-click detection tests
    #[test]
    fn test_single_click() {
        let device = Device::system_default().expect("No Metal device");
        let handler = ClickDetectingInputHandler::new(&device);

        handler.push_mouse_button(0, true, 100.0, 100.0);
        handler.push_mouse_button(0, false, 100.0, 100.0);

        // Single click: MouseDown + MouseUp = 2 events
        assert_eq!(handler.pending_count(), 2);
        assert_eq!(handler.current_click_count(), 1);
    }

    #[test]
    fn test_double_click() {
        let device = Device::system_default().expect("No Metal device");
        let handler = ClickDetectingInputHandler::new(&device);

        // First click
        handler.push_mouse_button(0, true, 100.0, 100.0);
        handler.push_mouse_button(0, false, 100.0, 100.0);

        // Second click (same position, quick)
        handler.push_mouse_button(0, true, 100.0, 100.0);
        handler.push_mouse_button(0, false, 100.0, 100.0);

        // Double click: 2x(MouseDown + MouseUp) + 1 DoubleClick = 5 events
        assert_eq!(handler.pending_count(), 5);
        assert_eq!(handler.current_click_count(), 2);

        // Verify we have the double-click event
        let events = handler.drain_events(10);
        let double_clicks: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == InputEventType::MouseDoubleClick as u16)
            .collect();
        assert_eq!(double_clicks.len(), 1);
    }

    #[test]
    fn test_triple_click() {
        let device = Device::system_default().expect("No Metal device");
        let handler = ClickDetectingInputHandler::new(&device);

        // Three rapid clicks
        for _ in 0..3 {
            handler.push_mouse_button(0, true, 100.0, 100.0);
            handler.push_mouse_button(0, false, 100.0, 100.0);
        }

        // Triple click: 3x(MouseDown + MouseUp) + 1 DoubleClick + 1 TripleClick = 8 events
        assert_eq!(handler.pending_count(), 8);
        assert_eq!(handler.current_click_count(), 3);

        let events = handler.drain_events(10);
        let triple_clicks: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == InputEventType::MouseTripleClick as u16)
            .collect();
        assert_eq!(triple_clicks.len(), 1);
    }

    #[test]
    fn test_double_click_distance_threshold() {
        let device = Device::system_default().expect("No Metal device");
        let handler = ClickDetectingInputHandler::new(&device);

        // First click
        handler.push_mouse_button(0, true, 100.0, 100.0);
        handler.push_mouse_button(0, false, 100.0, 100.0);

        // Second click too far away (> 5 pixels)
        handler.push_mouse_button(0, true, 110.0, 100.0);  // 10 pixels away
        handler.push_mouse_button(0, false, 110.0, 100.0);

        // Should NOT count as double-click
        assert_eq!(handler.current_click_count(), 1);

        // 4 events: 2 per click, no double-click event
        assert_eq!(handler.pending_count(), 4);
    }

    #[test]
    fn test_different_buttons_no_multi_click() {
        let device = Device::system_default().expect("No Metal device");
        let handler = ClickDetectingInputHandler::new(&device);

        // Left click
        handler.push_mouse_button(0, true, 100.0, 100.0);
        handler.push_mouse_button(0, false, 100.0, 100.0);

        // Right click (different button)
        handler.push_mouse_button(1, true, 100.0, 100.0);
        handler.push_mouse_button(1, false, 100.0, 100.0);

        // Should NOT count as double-click
        assert_eq!(handler.current_click_count(), 1);
    }
}
