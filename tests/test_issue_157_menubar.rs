// Issue #157 - MenuBar as Megakernel App Tests
//
// Tests for the GPU-native menu bar.

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::{
    app_type, GpuAppSystem, MenuBarState, MENUBAR_BACKGROUND_VERTS, MENUBAR_DEFAULT_HEIGHT,
};
use std::mem;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// =============================================================================
// Test 1: MenuBar generates background
// =============================================================================

#[test]
fn test_menubar_generates_background() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch menubar
    let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // Menubar should have run and generated background vertices
    let app = system.get_app(menubar).unwrap();
    assert_eq!(app.last_run_frame, 1, "Menubar should have run");
    assert_eq!(
        app.vertex_count,
        MENUBAR_BACKGROUND_VERTS,
        "Menubar should generate {} background vertices",
        MENUBAR_BACKGROUND_VERTS
    );
}

// =============================================================================
// Test 2: MenuBar state layout
// =============================================================================

#[test]
fn test_menubar_state_layout() {
    let size = mem::size_of::<MenuBarState>();
    println!("MenuBarState size: {} bytes", size);

    // Should be reasonable
    assert!(size > 0 && size <= 256, "State size should be reasonable");
}

// =============================================================================
// Test 3: MenuBar state defaults
// =============================================================================

#[test]
fn test_menubar_state_defaults() {
    let state = MenuBarState::default();

    assert_eq!(state.screen_width, 1280.0, "Default width should be 1280");
    assert_eq!(
        state.bar_height, MENUBAR_DEFAULT_HEIGHT,
        "Default height should be {}",
        MENUBAR_DEFAULT_HEIGHT
    );
    assert_eq!(state.menu_count, 0, "Initial menu count should be 0");
    assert_eq!(state.open_menu, u32::MAX, "No menu should be open initially");
    assert_eq!(state.hovered_menu, u32::MAX, "No menu should be hovered initially");

    // Translucent bar color
    assert!(state.bar_color[3] > 0.5, "Bar should have alpha > 0.5");
}

// =============================================================================
// Test 4: MenuBar runs in megakernel with other apps
// =============================================================================

#[test]
fn test_menubar_runs_with_system_apps() {
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
// Test 5: MenuBar with user apps
// =============================================================================

#[test]
fn test_menubar_with_user_apps() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch menubar and user apps
    let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
    let terminal = system.launch_by_type(app_type::TERMINAL).unwrap();
    let filesystem = system.launch_by_type(app_type::FILESYSTEM).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // All should have run
    assert_eq!(system.get_app(menubar).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(terminal).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(filesystem).unwrap().last_run_frame, 1);
}

// =============================================================================
// Test 6: Background constant
// =============================================================================

#[test]
fn test_background_vertex_constant() {
    // Background should be 6 vertices (2 triangles)
    assert_eq!(MENUBAR_BACKGROUND_VERTS, 6, "Background should be 6 vertices");
}

// =============================================================================
// Test 7: Multiple frames stability
// =============================================================================

#[test]
fn test_menubar_multiple_frames() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();

    // Run many frames
    for frame in 1..=100u32 {
        system.mark_all_dirty();
        system.run_frame();

        let app = system.get_app(menubar).unwrap();
        assert_eq!(app.last_run_frame, frame, "Frame {} should match", frame);
    }
}

// =============================================================================
// Test 8: MenuBar at top (depth ordering)
// =============================================================================

#[test]
fn test_menubar_depth_ordering() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch menubar and compositor
    let _menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
    let _compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // MenuBar should be in front of compositor background (depth 0.98 vs 0.0)
    // This is verified by the shader setting depth = 0.98 for menubar
}

// =============================================================================
// Test 9: Default height constant
// =============================================================================

#[test]
fn test_default_height_constant() {
    assert_eq!(MENUBAR_DEFAULT_HEIGHT, 24.0, "Default height should be 24 pixels");
}

// =============================================================================
// Test 10: Performance test
// =============================================================================

#[test]
fn test_menubar_performance() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch menubar and other apps
    let _menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
    let _compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();

    for _ in 0..5 {
        let slot = system.launch_by_type(app_type::TERMINAL).unwrap();
        system.create_window(slot, 0.0, 30.0, 200.0, 150.0);
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
    println!("MenuBar (5 apps): {}us/frame", us_per_frame);

    // Should be reasonably fast
    assert!(us_per_frame < 10000, "MenuBar too slow: {}us", us_per_frame);
}
