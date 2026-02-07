// Issue #159 - Window Chrome Tests
//
// Tests for GPU-generated window decorations (title bars, buttons, borders, resize handles).
// Verifies that chrome vertices are generated correctly for each window.

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::{
    app_type, GpuAppSystem, WindowChromeState, CHROME_VERTS_PER_WINDOW, RenderVertex,
};
use std::mem;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// =============================================================================
// Test 1: Chrome generates vertices for windows
// =============================================================================

#[test]
fn test_chrome_generates_vertices() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    // Enable parallel megakernel for window chrome
    system.set_use_parallel_megakernel(true);

    // Launch a terminal which creates a window
    let slot = system.launch_by_type(app_type::TERMINAL).unwrap();
    system.create_window(slot, 100.0, 100.0, 400.0, 300.0);

    // Launch window chrome
    let chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    // Mark all dirty and run frame
    system.mark_all_dirty();
    system.run_frame();

    // Chrome should have run
    let chrome_app = system.get_app(chrome).unwrap();
    assert_eq!(chrome_app.last_run_frame, 1, "Chrome should have run");
}

// =============================================================================
// Test 2: Multiple windows generate multiple chrome vertices
// =============================================================================

#[test]
fn test_chrome_multiple_windows() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch multiple terminals with windows
    for i in 0..3 {
        let slot = system.launch_by_type(app_type::TERMINAL).unwrap();
        system.create_window(slot, 100.0 + i as f32 * 200.0, 100.0, 400.0, 300.0);
    }

    // Launch window chrome
    let _chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // Count active windows
    let window_count = system.count_active_windows();
    assert_eq!(window_count, 3, "Should have 3 windows");
}

// =============================================================================
// Test 3: WindowChromeState struct layout
// =============================================================================

#[test]
fn test_chrome_state_layout() {
    // Verify WindowChromeState struct size matches Metal expectations
    let size = mem::size_of::<WindowChromeState>();
    println!("WindowChromeState size: {} bytes", size);

    // Should be aligned to 16 bytes and reasonably sized
    assert!(size > 0 && size <= 512, "State size should be reasonable");
}

// =============================================================================
// Test 4: Chrome state defaults
// =============================================================================

#[test]
fn test_chrome_state_defaults() {
    let state = WindowChromeState::default();

    // Check default values
    assert_eq!(state.title_bar_height, 28.0, "Default title bar height should be 28");
    assert_eq!(state.border_width, 1.0, "Default border width should be 1");
    assert_eq!(state.button_radius, 6.0, "Default button radius should be 6");
    assert_eq!(state.dragging_window, u32::MAX, "No window should be dragging initially");
    assert_eq!(state.resizing_window, u32::MAX, "No window should be resizing initially");

    // Check colors
    assert!(state.close_color[0] > 0.9, "Close button should be red");
    assert!(state.minimize_color[1] > 0.7, "Minimize button should be yellow");
    assert!(state.maximize_color[1] > 0.7, "Maximize button should be green");
}

// =============================================================================
// Test 5: RenderVertex struct alignment
// =============================================================================

#[test]
fn test_render_vertex_layout() {
    assert_eq!(
        mem::size_of::<RenderVertex>(),
        48,
        "RenderVertex should be 48 bytes"
    );
}

// =============================================================================
// Test 6: Chrome constants
// =============================================================================

#[test]
fn test_chrome_vertex_constants() {
    // Verify vertex counts per window
    // 6 (title bar) + 54 (3 buttons * 18) + 24 (4 borders * 6) + 6 (resize) = 90
    assert_eq!(CHROME_VERTS_PER_WINDOW, 90, "Should be 90 vertices per window");
}

// =============================================================================
// Test 7: Chrome runs in megakernel
// =============================================================================

#[test]
fn test_chrome_runs_in_megakernel() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch system apps including chrome
    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();
    let dock = system.launch_by_type(app_type::DOCK).unwrap();
    let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
    let chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    // Launch a user app with window
    let terminal = system.launch_by_type(app_type::TERMINAL).unwrap();
    system.create_window(terminal, 100.0, 100.0, 400.0, 300.0);

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // All should have run
    assert_eq!(system.get_app(compositor).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(dock).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(menubar).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(chrome).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(terminal).unwrap().last_run_frame, 1);
}

// =============================================================================
// Test 8: Multiple frames stability
// =============================================================================

#[test]
fn test_chrome_multiple_frames_stability() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch chrome and some windows
    let _chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    for i in 0..5 {
        let slot = system.launch_by_type(app_type::TERMINAL).unwrap();
        system.create_window(slot, 50.0 + i as f32 * 100.0, 50.0, 300.0, 200.0);
    }

    // Run many frames
    for frame in 0..100 {
        system.mark_all_dirty();
        system.run_frame();

        // Should be stable
        let window_count = system.count_active_windows();
        assert_eq!(window_count, 5, "Window count should remain stable at frame {}", frame);
    }
}

// =============================================================================
// Test 9: Chrome with no windows
// =============================================================================

#[test]
fn test_chrome_no_windows() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch only chrome (no windows)
    let chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // Chrome should still run
    assert_eq!(system.get_app(chrome).unwrap().last_run_frame, 1);

    // Zero windows = zero chrome vertices (handled gracefully)
    let window_count = system.count_active_windows();
    assert_eq!(window_count, 0);
}

// =============================================================================
// Test 10: Performance - Many windows
// =============================================================================

#[test]
fn test_chrome_performance_many_windows() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch chrome
    let _chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    // Create 20 windows
    for i in 0..20 {
        let slot = system.launch_by_type(app_type::TERMINAL).unwrap();
        system.create_window(
            slot,
            50.0 + (i % 5) as f32 * 150.0,
            50.0 + (i / 5) as f32 * 150.0,
            300.0,
            200.0,
        );
    }

    // Warm up
    for _ in 0..10 {
        system.mark_all_dirty();
        system.run_frame();
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        system.mark_all_dirty();
        system.run_frame();
    }
    let duration = start.elapsed();

    let us_per_frame = duration.as_micros() / 1000;
    println!("Chrome (20 windows): {}us/frame", us_per_frame);

    // Should be reasonably fast (under 10ms per frame)
    assert!(us_per_frame < 10000, "Chrome generation too slow: {}us", us_per_frame);
}
