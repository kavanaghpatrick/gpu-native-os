// Issue #158 - GPU Compositor Integration Tests
//
// Tests for the GPU compositor which generates background quad and tracks total vertices.

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::{
    app_type, GpuAppSystem, CompositorState, COMPOSITOR_BACKGROUND_VERTS,
};
use std::mem;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// =============================================================================
// Test 1: Compositor generates background
// =============================================================================

#[test]
fn test_compositor_generates_background() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch compositor
    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // Compositor should have run and generated background vertices
    let app = system.get_app(compositor).unwrap();
    assert_eq!(app.last_run_frame, 1, "Compositor should have run");
    assert_eq!(
        app.vertex_count,
        COMPOSITOR_BACKGROUND_VERTS,
        "Compositor should generate {} background vertices",
        COMPOSITOR_BACKGROUND_VERTS
    );
}

// =============================================================================
// Test 2: Compositor state layout
// =============================================================================

#[test]
fn test_compositor_state_layout() {
    let size = mem::size_of::<CompositorState>();
    println!("CompositorState size: {} bytes", size);

    // Should be reasonable
    assert!(size > 0 && size <= 256, "State size should be reasonable");
}

// =============================================================================
// Test 3: Compositor state defaults
// =============================================================================

#[test]
fn test_compositor_state_defaults() {
    let state = CompositorState::default();

    assert_eq!(state.screen_width, 1280.0, "Default width should be 1280");
    assert_eq!(state.screen_height, 720.0, "Default height should be 720");
    assert_eq!(state.frame_number, 0, "Initial frame should be 0");

    // Dark background
    assert!(state.background_color[0] < 0.2, "Background should be dark");
    assert!(state.background_color[1] < 0.2, "Background should be dark");
    assert!(state.background_color[2] < 0.2, "Background should be dark");
}

// =============================================================================
// Test 4: Compositor runs in megakernel with other apps
// =============================================================================

#[test]
fn test_compositor_runs_with_system_apps() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch all system apps
    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();
    let dock = system.launch_by_type(app_type::DOCK).unwrap();
    let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
    let chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // All should have run
    assert_eq!(system.get_app(compositor).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(dock).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(menubar).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(chrome).unwrap().last_run_frame, 1);
}

// =============================================================================
// Test 5: Compositor with user apps
// =============================================================================

#[test]
fn test_compositor_with_user_apps() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch compositor and user apps
    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();
    let terminal = system.launch_by_type(app_type::TERMINAL).unwrap();
    let filesystem = system.launch_by_type(app_type::FILESYSTEM).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // All should have run
    assert_eq!(system.get_app(compositor).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(terminal).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(filesystem).unwrap().last_run_frame, 1);
}

// =============================================================================
// Test 6: Background constant
// =============================================================================

#[test]
fn test_background_vertex_constant() {
    // Background quad should be 6 vertices (2 triangles)
    assert_eq!(COMPOSITOR_BACKGROUND_VERTS, 6, "Background should be 6 vertices");
}

// =============================================================================
// Test 7: Multiple frames stability
// =============================================================================

#[test]
fn test_compositor_multiple_frames() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    // Run many frames
    for frame in 1..=100u32 {
        system.mark_all_dirty();
        system.run_frame();

        let app = system.get_app(compositor).unwrap();
        assert_eq!(app.last_run_frame, frame, "Frame {} should match", frame);
    }
}

// =============================================================================
// Test 8: Compositor is first in render order (depth 0)
// =============================================================================

#[test]
fn test_compositor_at_back() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch compositor
    let _compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    // Launch app with window
    let terminal = system.launch_by_type(app_type::TERMINAL).unwrap();
    system.create_window(terminal, 100.0, 100.0, 400.0, 300.0);

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // Window should be in front of compositor background
    // (depth is handled by z-order of vertices)
    let window = system.get_window(terminal).unwrap();
    assert!(window.depth > 0.0, "Window should be in front of background");
}

// =============================================================================
// Test 9: Compositor vertex count
// =============================================================================

#[test]
fn test_compositor_vertex_count() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // Verify vertex count
    let app = system.get_app(compositor).unwrap();
    assert_eq!(app.vertex_count, 6, "Background should be 6 vertices");
}

// =============================================================================
// Test 10: Performance test
// =============================================================================

#[test]
fn test_compositor_performance() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch compositor and many apps
    let _compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    for _ in 0..10 {
        let slot = system.launch_by_type(app_type::TERMINAL).unwrap();
        system.create_window(slot, 0.0, 0.0, 200.0, 150.0);
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
    println!("Compositor (10 apps): {}us/frame", us_per_frame);

    // Should be reasonably fast
    assert!(us_per_frame < 10000, "Compositor too slow: {}us", us_per_frame);
}
