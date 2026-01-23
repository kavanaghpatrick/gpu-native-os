// Tests for Issue #13: Input Pipeline - IOKit to GPU Ring Buffer
//
// These tests verify the input handling system.
// Run with: cargo test --test test_issue_13_input

use metal::Device;
use rust_experiment::gpu_os::input::*;
use rust_experiment::gpu_os::memory::InputEventType;

fn setup() -> (Device, InputHandler) {
    let device = Device::system_default().expect("No Metal device");
    let handler = InputHandler::new(&device);
    (device, handler)
}

#[test]
fn test_input_handler_creates_buffer() {
    let (device, handler) = setup();

    let buffer = handler.buffer();
    assert!(buffer.length() > 0, "Input buffer must be allocated");
}

#[test]
fn test_push_mouse_move() {
    let (_, handler) = setup();

    handler.push_mouse_move(0.5, 0.5, 0.01, 0.0);

    assert_eq!(
        handler.pending_count(), 1,
        "Should have 1 pending event after push"
    );
}

#[test]
fn test_push_mouse_button() {
    let (_, handler) = setup();

    handler.push_mouse_button(0, true, 0.5, 0.5);  // Left click down
    handler.push_mouse_button(0, false, 0.5, 0.5); // Left click up

    assert_eq!(
        handler.pending_count(), 2,
        "Should have 2 pending events"
    );
}

#[test]
fn test_push_key() {
    let (_, handler) = setup();

    handler.push_key(0x00, true, 0);  // 'A' key down
    handler.push_key(0x00, false, 0); // 'A' key up

    assert_eq!(handler.pending_count(), 2);
}

#[test]
fn test_clear_events() {
    let (_, handler) = setup();

    // Add some events
    for _ in 0..10 {
        handler.push_mouse_move(0.5, 0.5, 0.01, 0.0);
    }

    assert_eq!(handler.pending_count(), 10);

    handler.clear();

    assert_eq!(
        handler.pending_count(), 0,
        "Queue should be empty after clear"
    );
}

#[test]
fn test_burst_64_events() {
    let (_, handler) = setup();

    // Push 64 events rapidly (simulates burst input)
    for i in 0..64 {
        handler.push_mouse_move(i as f32 / 64.0, 0.5, 0.01, 0.0);
    }

    assert_eq!(
        handler.pending_count(), 64,
        "Should handle burst of 64 events"
    );
}

#[test]
fn test_queue_wraparound() {
    let (_, handler) = setup();

    // Fill queue past capacity to test wraparound
    for i in 0..300 {
        handler.push_mouse_move(i as f32 / 300.0, 0.5, 0.01, 0.0);
    }

    // Queue should handle wraparound correctly
    // Oldest events may be overwritten, but no crash
    let count = handler.pending_count();
    assert!(count <= 256, "Queue should not exceed capacity");
}

#[test]
fn test_simulate_mouse_moves() {
    let device = Device::system_default().expect("No Metal device");
    let source = TestInputSource::new(&device);

    source.simulate_mouse_moves(100);

    assert_eq!(
        source.handler().pending_count(), 100,
        "Should have 100 simulated mouse moves"
    );
}

#[test]
fn test_simulate_typing() {
    let device = Device::system_default().expect("No Metal device");
    let source = TestInputSource::new(&device);

    source.simulate_typing("Hello");

    // "Hello" = 5 key down + 5 key up = 10 events
    assert_eq!(
        source.handler().pending_count(), 10,
        "Typing 'Hello' should produce 10 key events"
    );
}
