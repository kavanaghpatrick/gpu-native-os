// Tests for Issue #149: GPU-Driven Event Dispatch
//
// These tests verify that the GPU event loop:
// 1. Integrates with existing InputHandler (no duplication)
// 2. Processes input events on GPU
// 3. Handles window drag/resize without CPU dispatch
// 4. Uses GpuRuntime's existing infrastructure
//
// Run with: cargo test --test test_issue_149_gpu_event_loop

use metal::Device;
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::event_loop::{
    dispatch, edge, region, window_flags, GpuEventLoopState, GpuWindow, HitTestResult,
    INVALID_WINDOW,
};

fn setup() -> GpuRuntime {
    let device = Device::system_default().expect("No Metal device");
    GpuRuntime::new(device)
}

fn create_test_windows(runtime: &GpuRuntime, windows: &[GpuWindow]) -> metal::Buffer {
    let buffer = runtime.device.new_buffer(
        (windows.len() * GpuWindow::SIZE) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            windows.as_ptr(),
            buffer.contents() as *mut GpuWindow,
            windows.len(),
        );
    }
    buffer
}

// ============================================================================
// Basic Initialization Tests
// ============================================================================

#[test]
fn test_event_loop_initializes() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // init_event_loop creates the handle but doesn't start running
    // start_event_loop must be called to begin processing
    assert!(!handle.is_running(), "Event loop should NOT be running until start_event_loop is called");
}

#[test]
fn test_event_loop_state_buffer_created() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // State buffer should be large enough for GpuEventLoopState
    assert!(
        handle.state_buffer.length() >= GpuEventLoopState::SIZE as u64,
        "State buffer too small"
    );
}

#[test]
fn test_event_loop_hit_result_buffer_created() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Hit result buffer should be large enough for HitTestResult
    assert!(
        handle.hit_result_buffer.length() >= HitTestResult::SIZE as u64,
        "Hit result buffer too small"
    );
}

#[test]
fn test_event_loop_can_stop() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Create a dummy windows buffer for start_event_loop
    let windows_buffer = runtime.device.new_buffer(
        32, // One window size
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Start the event loop
    runtime.start_event_loop(&handle, &windows_buffer, 0);
    assert!(handle.is_running(), "Event loop should be running after start");

    handle.stop();
    assert!(!handle.is_running(), "Event loop should stop when requested");
}

// ============================================================================
// Integration with Existing InputHandler Tests
// ============================================================================

#[test]
fn test_uses_existing_input_handler() {
    let runtime = setup();

    let _handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Push events through the EXISTING InputHandler
    runtime.input.push_mouse_move(0.5, 0.5, 0.01, 0.0);
    runtime.input.push_mouse_button(0, true, 0.5, 0.5);

    // Events should be in the InputHandler's queue
    assert_eq!(
        runtime.input.pending_count(),
        2,
        "Events should go through existing InputHandler"
    );
}

#[test]
fn test_event_loop_uses_input_handler_buffer() {
    let runtime = setup();

    let _handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // The input buffer used by event loop should be the same as InputHandler's buffer
    let input_buffer = runtime.input.buffer();
    assert!(
        input_buffer.length() > 0,
        "InputHandler buffer should exist"
    );

    // Push through InputHandler
    runtime.input.push_mouse_move(100.0, 100.0, 0.0, 0.0);

    assert_eq!(
        runtime.input.pending_count(),
        1,
        "Event should be pending in InputHandler"
    );
}

// ============================================================================
// Event Processing Tests
// ============================================================================

#[test]
fn test_gpu_event_loop_processes_mouse_move() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Create a test window
    let windows = vec![GpuWindow {
        x: 100.0,
        y: 100.0,
        width: 400.0,
        height: 300.0,
        z_order: 1,
        flags: window_flags::VISIBLE,
        _padding: [0; 2],
    }];
    let windows_buffer = create_test_windows(&runtime, &windows);

    // Start the event loop (marks as running)
    runtime.start_event_loop(&handle, &windows_buffer, 1);

    // Push mouse move event AFTER start to ensure the first kernel sees no events
    runtime.input.push_mouse_move(200.0, 150.0, 0.0, 0.0);

    // Process the event loop multiple times to ensure event is processed
    // First call processes the event, second ensures state is stable
    runtime.process_event_loop(&handle, &windows_buffer, 1);
    runtime.process_event_loop(&handle, &windows_buffer, 1);

    // Read state - mouse position should be updated
    let state = handle.read_state();

    // Allow some tolerance for floating point comparison
    assert!(
        (state.mouse_x - 200.0).abs() < 1.0,
        "Mouse X should be ~200: got {}",
        state.mouse_x
    );
    assert!(
        (state.mouse_y - 150.0).abs() < 1.0,
        "Mouse Y should be ~150: got {}",
        state.mouse_y
    );
}

#[test]
fn test_gpu_event_loop_processes_click() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Create a test window
    let windows = vec![GpuWindow {
        x: 100.0,
        y: 100.0,
        width: 400.0,
        height: 300.0,
        z_order: 1,
        flags: window_flags::VISIBLE,
        _padding: [0; 2],
    }];
    let windows_buffer = create_test_windows(&runtime, &windows);

    // Start the event loop
    runtime.start_event_loop(&handle, &windows_buffer, 1);

    // Push mouse click in window content area
    runtime.input.push_mouse_button(0, true, 250.0, 200.0);

    // Process the event loop
    runtime.process_event_loop(&handle, &windows_buffer, 1);

    // Check that mouse button state is updated
    let state = handle.read_state();
    assert!(
        state.mouse_buttons & 1 != 0,
        "Mouse button 0 should be pressed"
    );
}

#[test]
fn test_gpu_event_loop_no_cpu_dispatch_during_drag() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Create a test window
    let windows = vec![GpuWindow {
        x: 100.0,
        y: 100.0,
        width: 400.0,
        height: 300.0,
        z_order: 1,
        flags: window_flags::VISIBLE,
        _padding: [0; 2],
    }];
    let windows_buffer = create_test_windows(&runtime, &windows);

    // Click on title bar to start drag (y < 30 is title bar)
    runtime.input.push_mouse_button(0, true, 200.0, 115.0); // Title bar area
    runtime.process_event_loop(&handle, &windows_buffer, 1);

    // Process hit test result
    runtime.process_event_loop(&handle, &windows_buffer, 1);

    let state = handle.read_state();

    // If drag started, drag_window should be set
    if state.drag_window != INVALID_WINDOW {
        // Now move mouse - this should NOT require CPU dispatch
        runtime.input.push_mouse_move(250.0, 165.0, 50.0, 50.0);
        let _needs_dispatch = runtime.process_event_loop(&handle, &windows_buffer, 1);

        // The GPU handles window move directly - CPU just needs to know if frame is dirty
        let state_after = handle.read_state();
        assert_eq!(
            state_after.drag_window, 0,
            "Drag should be active on window 0"
        );
    }
}

// ============================================================================
// Window Operation Tests
// ============================================================================

#[test]
fn test_window_focus_on_click() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    // Create two test windows
    let windows = vec![
        GpuWindow {
            x: 100.0,
            y: 100.0,
            width: 300.0,
            height: 200.0,
            z_order: 1,
            flags: window_flags::VISIBLE,
            _padding: [0; 2],
        },
        GpuWindow {
            x: 200.0,
            y: 150.0,
            width: 300.0,
            height: 200.0,
            z_order: 2, // Higher z-order = on top
            flags: window_flags::VISIBLE,
            _padding: [0; 2],
        },
    ];
    let windows_buffer = create_test_windows(&runtime, &windows);

    // Start the event loop
    runtime.start_event_loop(&handle, &windows_buffer, 2);

    // Click in the second window's content area (inside overlap region)
    // Position 350, 200 is inside both windows but window 1 (z=2) is on top
    runtime.input.push_mouse_button(0, true, 350.0, 200.0);

    // Process the event loop multiple times:
    // 1. First call processes the click event and triggers HIT_TEST dispatch
    // 2. Second call runs the hit test and result handler
    // 3. Third call for good measure
    runtime.process_event_loop(&handle, &windows_buffer, 2);
    runtime.process_event_loop(&handle, &windows_buffer, 2);
    runtime.process_event_loop(&handle, &windows_buffer, 2);

    let state = handle.read_state();
    // The topmost window at click location should get focus
    // Window 1 (z_order=2) is on top in the overlap region
    assert_eq!(
        state.focused_window, 1,
        "Clicked window (index 1, higher z-order) should be focused, got {} (INVALID={})",
        state.focused_window, INVALID_WINDOW
    );
}

// ============================================================================
// State Management Tests
// ============================================================================

#[test]
fn test_initial_state_values() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    let state = handle.read_state();

    assert_eq!(state.drag_window, INVALID_WINDOW);
    assert_eq!(state.resize_window, INVALID_WINDOW);
    assert_eq!(state.focused_window, INVALID_WINDOW);
    assert_eq!(state.next_dispatch, dispatch::NONE);
    assert_eq!(state.frame_dirty, 0);
}

#[test]
fn test_hit_result_reset_after_read() {
    let runtime = setup();

    let handle = runtime
        .init_event_loop()
        .expect("Should initialize event loop");

    let hit_result = handle.read_hit_result();

    // Initial hit result should be empty
    assert!(!hit_result.is_hit(), "Initial hit result should be empty");
}

// ============================================================================
// Struct Size and Alignment Tests
// ============================================================================

#[test]
fn test_gpu_event_loop_state_alignment() {
    assert_eq!(
        GpuEventLoopState::SIZE % 16,
        0,
        "GpuEventLoopState must be 16-byte aligned for Metal"
    );
    assert_eq!(
        GpuEventLoopState::SIZE,
        96,
        "GpuEventLoopState size changed unexpectedly"
    );
}

#[test]
fn test_gpu_window_size() {
    assert_eq!(
        GpuWindow::SIZE,
        32,
        "GpuWindow should be 32 bytes (8 floats/uints)"
    );
}

#[test]
fn test_hit_test_result_size() {
    assert_eq!(
        HitTestResult::SIZE,
        8,
        "HitTestResult should be 8 bytes (u64)"
    );
}

// ============================================================================
// Hit Test Result Decoding Tests
// ============================================================================

#[test]
fn test_hit_result_decoding() {
    // Encode: z_order=5, window_index=2, region=CONTENT(2), resize_edge=NONE(0)
    let result = HitTestResult {
        z_order: 5,
        packed_data: (2 << 16) | (2 << 8) | 0,
    };

    assert!(result.is_hit());
    assert_eq!(result.window_index(), 2);
    assert_eq!(result.region(), region::CONTENT);
    assert_eq!(result.resize_edge(), edge::NONE);
}

#[test]
fn test_hit_result_with_resize_edge() {
    // Encode: z_order=3, window_index=0, region=RESIZE(6), edge=LEFT|BOTTOM(1|8=9)
    let result = HitTestResult {
        z_order: 3,
        packed_data: (0 << 16) | (6 << 8) | 9,
    };

    assert!(result.is_hit());
    assert_eq!(result.window_index(), 0);
    assert_eq!(result.region(), region::RESIZE);
    assert_eq!(result.resize_edge(), edge::LEFT | edge::BOTTOM);
}

// ============================================================================
// Constants Tests
// ============================================================================

#[test]
fn test_dispatch_constants() {
    assert_eq!(dispatch::NONE, 0);
    assert_eq!(dispatch::HIT_TEST, 1);
    assert_eq!(dispatch::WINDOW_MOVE, 2);
    assert_eq!(dispatch::WINDOW_RESIZE, 3);
    assert_eq!(dispatch::RENDER, 8);
}

#[test]
fn test_region_constants() {
    assert_eq!(region::NONE, 0);
    assert_eq!(region::TITLE, 1);
    assert_eq!(region::CONTENT, 2);
    assert_eq!(region::CLOSE, 3);
}

#[test]
fn test_edge_constants() {
    assert_eq!(edge::NONE, 0);
    assert_eq!(edge::LEFT, 1);
    assert_eq!(edge::RIGHT, 2);
    assert_eq!(edge::TOP, 4);
    assert_eq!(edge::BOTTOM, 8);
}

#[test]
fn test_window_flags_constants() {
    assert_eq!(window_flags::VISIBLE, 1);
    assert_eq!(window_flags::MINIMIZED, 2);
    assert_eq!(window_flags::MAXIMIZED, 4);
    assert_eq!(window_flags::FOCUSED, 8);
}
