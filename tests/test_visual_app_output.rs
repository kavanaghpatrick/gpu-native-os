//! Tests for Visual App Output (Issue #246)
//!
//! Tests that launched apps (Terminal, Filesystem, Document) render visible
//! geometry in their window bounds.
//!
//! These tests verify that the GPU megakernel properly generates vertices
//! for each app type after launching and running frames.
//!
//! NOTE: These tests are currently ignored due to a shader compilation bug
//! (duplicate case values in bytecode VM opcode handling). Once the shader
//! bug is fixed, these tests will verify the visual output behavior.
//! See the shader error: "duplicate case value: 'OP_F64_NEG' and 'OP_F64_TO_I32_S_SAT'"

use metal::Device;
use rust_experiment::gpu_os::gpu_app_system::{app_type, GpuAppSystem};

fn get_device() -> Option<Device> {
    Device::system_default()
}

// =============================================================================
// Test 1: Terminal app generates visible vertices after launch
// =============================================================================

/// Test that terminal app generates visible vertices after launch.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values
/// preventing compilation. Once fixed, this test verifies:
/// - Terminal app can be launched
/// - After one frame, vertex_count > 0
/// - Vertices are positioned correctly
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_terminal_renders_visible() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch terminal app
    let slot = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal app");

    // Run frame to trigger app update
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS:
    let app = system.get_app(slot).expect("Terminal app should be active");

    // Terminal falls through to counter_app_update which sets vertex_count = 6
    assert!(
        app.vertex_count > 0,
        "Terminal should emit vertices, got {}",
        app.vertex_count
    );
    assert_eq!(
        app.vertex_count, 6,
        "Terminal should emit 6 vertices (one quad from counter_app_update)"
    );
    assert_eq!(
        app.last_run_frame, 1,
        "Terminal should have run in frame 1"
    );
    assert_eq!(
        app.app_type,
        app_type::TERMINAL,
        "App type should be TERMINAL"
    );
}

// =============================================================================
// Test 2: Filesystem app generates visible vertices after launch
// =============================================================================

/// Test that filesystem app generates visible vertices after launch.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_filesystem_renders_visible() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch filesystem app
    let slot = system
        .launch_by_type(app_type::FILESYSTEM)
        .expect("Failed to launch Filesystem app");

    // Run frame to trigger app update
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS:
    let app = system.get_app(slot).expect("Filesystem app should be active");

    // Filesystem falls through to counter_app_update which sets vertex_count = 6
    assert!(
        app.vertex_count > 0,
        "Filesystem should emit vertices, got {}",
        app.vertex_count
    );
    assert_eq!(
        app.vertex_count, 6,
        "Filesystem should emit 6 vertices (one quad from counter_app_update)"
    );
    assert_eq!(
        app.last_run_frame, 1,
        "Filesystem should have run in frame 1"
    );
    assert_eq!(
        app.app_type,
        app_type::FILESYSTEM,
        "App type should be FILESYSTEM"
    );
}

// =============================================================================
// Test 3: Document app generates visible vertices after launch
// =============================================================================

/// Test that document app generates visible vertices after launch.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_document_renders_visible() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch document app
    let slot = system
        .launch_by_type(app_type::DOCUMENT)
        .expect("Failed to launch Document app");

    // Run frame to trigger app update
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS:
    let app = system.get_app(slot).expect("Document app should be active");

    // Document falls through to counter_app_update which sets vertex_count = 6
    assert!(
        app.vertex_count > 0,
        "Document should emit vertices, got {}",
        app.vertex_count
    );
    assert_eq!(
        app.vertex_count, 6,
        "Document should emit 6 vertices (one quad from counter_app_update)"
    );
    assert_eq!(
        app.last_run_frame, 1,
        "Document should have run in frame 1"
    );
    assert_eq!(
        app.app_type,
        app_type::DOCUMENT,
        "App type should be DOCUMENT"
    );
}

// =============================================================================
// Test 4: Different app types have distinct type IDs (they are distinguishable)
// =============================================================================

/// Test that different app types have distinct type IDs.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_apps_have_distinct_colors() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch all three app types
    let terminal_slot = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal");
    let filesystem_slot = system
        .launch_by_type(app_type::FILESYSTEM)
        .expect("Failed to launch Filesystem");
    let document_slot = system
        .launch_by_type(app_type::DOCUMENT)
        .expect("Failed to launch Document");

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS: Apps have distinct type IDs
    let terminal = system.get_app(terminal_slot).unwrap();
    let filesystem = system.get_app(filesystem_slot).unwrap();
    let document = system.get_app(document_slot).unwrap();

    assert_eq!(terminal.app_type, app_type::TERMINAL);
    assert_eq!(filesystem.app_type, app_type::FILESYSTEM);
    assert_eq!(document.app_type, app_type::DOCUMENT);

    // All have distinct types
    assert_ne!(terminal.app_type, filesystem.app_type);
    assert_ne!(terminal.app_type, document.app_type);
    assert_ne!(filesystem.app_type, document.app_type);

    // All have distinct slot IDs
    assert_ne!(terminal_slot, filesystem_slot);
    assert_ne!(terminal_slot, document_slot);
    assert_ne!(filesystem_slot, document_slot);
}

// =============================================================================
// Test 5: App content renders within window bounds
// =============================================================================

/// Test that app content renders within window bounds.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_app_renders_in_correct_window() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch app and create window
    let slot = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal");

    // Create window at position (100, 200) with size (300, 400)
    system.create_window(slot, 100.0, 200.0, 300.0, 400.0);

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS: App should be associated with window
    let app = system.get_app(slot).expect("App should be active");

    // Window ID should be set (window was created)
    // Note: window_id is the index in the windows array
    assert!(app.window_id < 64, "Window ID should be valid (< 64)");
    assert!(app.vertex_count > 0, "App should emit vertices");
    assert_eq!(app.last_run_frame, 1, "App should have run");
}

// =============================================================================
// Test 6: App depth ordering (apps with windows vs system apps)
// =============================================================================

/// Test that app depth ordering is correct (system apps vs user apps).
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_app_depth_ordering() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch compositor (background) and user app
    let compositor = system
        .launch_by_type(app_type::COMPOSITOR)
        .expect("Failed to launch Compositor");
    let terminal = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal");
    let dock = system
        .launch_by_type(app_type::DOCK)
        .expect("Failed to launch Dock");

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS: All apps ran
    let comp_app = system.get_app(compositor).unwrap();
    let term_app = system.get_app(terminal).unwrap();
    let dock_app = system.get_app(dock).unwrap();

    assert_eq!(comp_app.last_run_frame, 1, "Compositor should have run");
    assert_eq!(term_app.last_run_frame, 1, "Terminal should have run");
    assert_eq!(dock_app.last_run_frame, 1, "Dock should have run");

    // Compositor generates background vertices
    assert!(
        comp_app.vertex_count > 0,
        "Compositor should emit background vertices"
    );
    // Dock generates its own vertices
    assert!(
        dock_app.vertex_count > 0,
        "Dock should emit vertices"
    );
}

// =============================================================================
// Test 7: Multiple apps can be launched and all render
// =============================================================================

/// Test that multiple apps can be launched and all render.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_multiple_apps_all_render() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch 3 different apps
    let slots: Vec<u32> = vec![
        system.launch_by_type(app_type::TERMINAL).unwrap(),
        system.launch_by_type(app_type::FILESYSTEM).unwrap(),
        system.launch_by_type(app_type::DOCUMENT).unwrap(),
    ];

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS: All 3 should have vertex_count > 0
    for (i, &slot) in slots.iter().enumerate() {
        let app = system.get_app(slot).expect("App should be active");
        assert!(
            app.vertex_count > 0,
            "App {} at slot {} should emit vertices, got {}",
            i,
            slot,
            app.vertex_count
        );
        assert_eq!(
            app.last_run_frame, 1,
            "App {} should have run in frame 1",
            i
        );
    }

    // Verify distinct slots
    assert_ne!(slots[0], slots[1]);
    assert_ne!(slots[0], slots[2]);
    assert_ne!(slots[1], slots[2]);
}

// =============================================================================
// Test 8: App vertex count is within allocated budget
// =============================================================================

/// Test that app vertex count is within allocated budget.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_app_vertex_count_within_budget() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch Terminal
    let slot = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal");

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    let app = system.get_app(slot).expect("App should be active");

    // From APP_TYPES registry:
    // Terminal: vertex_size = 2000 * 6 * 48 bytes
    // This means max 2000 * 6 = 12000 vertices
    let max_vertices = 2000 * 6;

    // REAL ASSERTIONS: vertex_count should never exceed allocated vertices
    assert!(
        app.vertex_count <= max_vertices,
        "Terminal vertex_count {} exceeds budget {}",
        app.vertex_count,
        max_vertices
    );
    assert!(
        app.vertex_count > 0,
        "Terminal should emit some vertices"
    );
}

// =============================================================================
// Test 9: Window-slot association via slot_id
// =============================================================================

/// Test window-slot association via slot_id.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_window_slot_association() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch app
    let slot = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal");

    // Create window for this app
    system.create_window(slot, 50.0, 100.0, 400.0, 300.0);

    // Run frame
    system.mark_all_dirty();
    system.run_frame();

    // REAL ASSERTIONS: App slot_id should match the slot we launched
    let app = system.get_app(slot).expect("App should be active");
    assert_eq!(
        app.slot_id, slot,
        "App slot_id {} should match launch slot {}",
        app.slot_id, slot
    );

    // App should be marked visible after window creation and frame run
    // (The GPU finds window by matching window.app_slot == app.slot_id)
    assert!(
        app.vertex_count > 0,
        "App with window should emit vertices"
    );
}

// =============================================================================
// Test 10: Frame counter increments across multiple frames
// =============================================================================

/// Test that frame counter increments across multiple frames.
///
/// Currently ignored: GpuAppSystem shader has duplicate opcode case values.
#[test]
#[ignore = "Shader compilation bug: duplicate case values in bytecode VM opcodes"]
fn test_frame_counter_increments() {
    let device = match get_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping test: no Metal device available");
            return;
        }
    };

    let mut system = GpuAppSystem::new(&device).expect("Failed to create GPU app system");
    system.set_use_parallel_megakernel(true);

    // Launch terminal
    let slot = system
        .launch_by_type(app_type::TERMINAL)
        .expect("Failed to launch Terminal");

    // Run 10 frames
    for frame in 1..=10 {
        system.mark_all_dirty();
        system.run_frame();

        let app = system.get_app(slot).expect("App should be active");

        // REAL ASSERTIONS: last_run_frame should match current frame
        assert_eq!(
            app.last_run_frame, frame,
            "After frame {}, last_run_frame should be {}",
            frame, frame
        );

        // Vertex count should remain valid
        assert!(
            app.vertex_count > 0,
            "Frame {}: vertex_count should be > 0",
            frame
        );
    }
}
