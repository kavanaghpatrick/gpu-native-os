// Issue #156 - Dock as Megakernel App Tests
//
// Tests for the GPU-native dock with parallel hover detection,
// magnification effect, and bounce animations.

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::{
    app_type, GpuAppSystem, DockItem, DockState,
    DOCK_DEFAULT_HEIGHT, DOCK_DEFAULT_ICON_SIZE,
};
use std::mem;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// =============================================================================
// Test 1: Dock launches as megakernel app
// =============================================================================

#[test]
fn test_dock_launches_as_megakernel_app() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK);
    assert!(dock_slot.is_some(), "Should be able to launch dock");

    let app = system.get_app(dock_slot.unwrap()).unwrap();
    assert_eq!(app.app_type, app_type::DOCK);
}

// =============================================================================
// Test 2: Dock state initialization
// =============================================================================

#[test]
fn test_dock_state_initialization() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(dock_slot, 1920.0, 1080.0);

    let state = system.get_dock_state(dock_slot).unwrap();
    assert_eq!(state.screen_width, 1920.0);
    assert_eq!(state.screen_height, 1080.0);
    assert_eq!(state.dock_height, DOCK_DEFAULT_HEIGHT);
    assert_eq!(state.base_icon_size, DOCK_DEFAULT_ICON_SIZE);
}

// =============================================================================
// Test 3: DockState struct layout
// =============================================================================

#[test]
fn test_dock_state_layout() {
    let size = mem::size_of::<DockState>();
    println!("DockState size: {} bytes", size);

    // Should be reasonable (Metal requires field alignment, not struct alignment)
    assert!(size > 0 && size <= 256, "State size should be reasonable");
}

// =============================================================================
// Test 4: DockItem struct layout
// =============================================================================

#[test]
fn test_dock_item_layout() {
    let size = mem::size_of::<DockItem>();
    println!("DockItem size: {} bytes", size);

    // Should be reasonable
    assert!(size > 0 && size <= 64, "Item size should be reasonable");
}

// =============================================================================
// Test 5: Add dock items
// =============================================================================

#[test]
fn test_dock_add_items() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(dock_slot, 1920.0, 1080.0);

    // Add items
    let idx0 = system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);
    let idx1 = system.add_dock_item(dock_slot, app_type::FILESYSTEM, [0.0, 1.0, 0.0, 1.0]);
    let idx2 = system.add_dock_item(dock_slot, app_type::DOCUMENT, [0.0, 0.0, 1.0, 1.0]);

    assert_eq!(idx0, Some(0));
    assert_eq!(idx1, Some(1));
    assert_eq!(idx2, Some(2));

    let state = system.get_dock_state(dock_slot).unwrap();
    assert_eq!(state.item_count, 3, "Should have 3 dock items");
}

// =============================================================================
// Test 6: Dock runs in megakernel
// =============================================================================

#[test]
fn test_dock_runs_in_megakernel() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(dock_slot, 1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    system.mark_dirty(dock_slot);
    system.run_frame();

    let app = system.get_app(dock_slot).unwrap();
    assert_eq!(app.last_run_frame, 1, "Dock should have run in frame 1");
    assert!(app.vertex_count > 0, "Dock should have generated vertices");
}

// =============================================================================
// Test 7: Dock generates correct vertex count
// =============================================================================

#[test]
fn test_dock_vertex_count() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(dock_slot, 1920.0, 1080.0);

    // Add 5 items
    for i in 0..5 {
        let hue = (i as f32) / 5.0;
        system.add_dock_item(dock_slot, app_type::CUSTOM, [hue, 0.5, 0.5, 1.0]);
    }

    system.mark_dirty(dock_slot);
    system.run_frame();

    let app = system.get_app(dock_slot).unwrap();
    assert_eq!(app.vertex_count, 5 * 6, "5 icons * 6 vertices = 30");
}

// =============================================================================
// Test 8: Dock with system apps
// =============================================================================

#[test]
fn test_dock_with_system_apps() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    // Launch all system apps
    let compositor = system.launch_by_type(app_type::COMPOSITOR).unwrap();
    let dock = system.launch_by_type(app_type::DOCK).unwrap();
    let menubar = system.launch_by_type(app_type::MENUBAR).unwrap();
    let chrome = system.launch_by_type(app_type::WINDOW_CHROME).unwrap();

    system.initialize_dock_state(dock, 1920.0, 1080.0);
    system.add_dock_item(dock, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);

    system.mark_all_dirty();
    system.run_frame();

    // All should have run
    assert_eq!(system.get_app(compositor).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(dock).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(menubar).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(chrome).unwrap().last_run_frame, 1);
}

// =============================================================================
// Test 9: Dock multiple frames stability
// =============================================================================

#[test]
fn test_dock_multiple_frames() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(dock_slot, 1920.0, 1080.0);
    system.add_dock_item(dock_slot, app_type::TERMINAL, [1.0, 0.0, 0.0, 1.0]);
    system.add_dock_item(dock_slot, app_type::FILESYSTEM, [0.0, 1.0, 0.0, 1.0]);

    // Run many frames
    for frame in 1..=100u32 {
        system.mark_all_dirty();
        system.run_frame();

        let app = system.get_app(dock_slot).unwrap();
        assert_eq!(app.last_run_frame, frame, "Frame {} should match", frame);
    }
}

// =============================================================================
// Test 10: Performance benchmark
// =============================================================================

#[test]
fn test_dock_performance() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    system.set_use_parallel_megakernel(true);

    let dock_slot = system.launch_by_type(app_type::DOCK).unwrap();
    system.initialize_dock_state(dock_slot, 1920.0, 1080.0);

    // Add 16 dock items
    for i in 0..16 {
        let hue = (i as f32) / 16.0;
        system.add_dock_item(dock_slot, app_type::CUSTOM, [hue, 0.5, 0.5, 1.0]);
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
    println!("Dock (16 items): {}us/frame", us_per_frame);

    // Should be reasonably fast
    assert!(us_per_frame < 10000, "Dock too slow: {}us", us_per_frame);
}
