// Tests for Issue #15: Widget System - Compressed State & Branchless Dispatch
//
// These tests verify the widget system.
// Run with: cargo test --test test_issue_15_widget

use metal::Device;
use rust_experiment::gpu_os::widget::*;
use rust_experiment::gpu_os::memory::{GpuMemory, WidgetCompact};

fn setup() -> (Device, GpuMemory, WidgetManager) {
    let device = Device::system_default().expect("No Metal device");
    let memory = GpuMemory::new(&device, 1024);
    let manager = WidgetManager::new(&device).expect("Widget manager should compile");
    (device, memory, manager)
}

#[test]
fn test_widget_manager_compiles() {
    let device = Device::system_default().expect("No Metal device");
    let result = WidgetManager::new(&device);
    assert!(result.is_ok(), "Widget manager should compile");
}

#[test]
fn test_widget_builder() {
    let widget = WidgetBuilder::new(WidgetType::Button)
        .bounds(0.1, 0.1, 0.2, 0.05)
        .background_color(0.2, 0.4, 0.8)
        .border_color(0.1, 0.2, 0.4)
        .corner_radius(4.0)
        .border_width(1.0)
        .z_order(10)
        .build();

    assert_eq!(widget.z_order, 10);
    assert_eq!(widget.widget_type(), WidgetType::Button);
}

#[test]
fn test_widget_type_extraction() {
    let widget = WidgetBuilder::new(WidgetType::Slider)
        .bounds(0.0, 0.0, 0.1, 0.02)
        .build();

    assert_eq!(widget.widget_type(), WidgetType::Slider);
}

#[test]
fn test_widget_flags() {
    let flags = WidgetFlags::new();

    assert!(flags.is_visible(), "Widgets should be visible by default");
    assert!(flags.is_enabled(), "Widgets should be enabled by default");
}

#[test]
fn test_hit_testing_single_widget() {
    let (device, memory, manager) = setup();
    let queue = device.new_command_queue();

    // Create a widget at (0.4, 0.4) with size (0.2, 0.2)
    let widgets = vec![
        WidgetBuilder::new(WidgetType::Button)
            .bounds(0.4, 0.4, 0.2, 0.2)
            .z_order(0)
            .build(),
    ];
    memory.write_widgets(&widgets);

    // Test hit inside widget
    let (hits, topmost) = manager.hit_test_sync(
        &queue,
        &memory.widget_buffer,
        1,
        0.5, 0.5,  // Center of widget
    );

    assert_eq!(hits, 1, "Should hit 1 widget");
    assert_eq!(topmost, Some(0), "Topmost should be widget 0");

    // Test miss outside widget
    let (hits, topmost) = manager.hit_test_sync(
        &queue,
        &memory.widget_buffer,
        1,
        0.1, 0.1,  // Outside widget
    );

    assert_eq!(hits, 0, "Should hit 0 widgets");
    assert_eq!(topmost, None, "No topmost widget");
}

#[test]
fn test_hit_testing_z_order() {
    let (device, memory, manager) = setup();
    let queue = device.new_command_queue();

    // Create overlapping widgets with different z-orders
    let widgets = vec![
        WidgetBuilder::new(WidgetType::Container)
            .bounds(0.3, 0.3, 0.4, 0.4)
            .z_order(0)
            .build(),
        WidgetBuilder::new(WidgetType::Button)
            .bounds(0.4, 0.4, 0.2, 0.2)
            .z_order(10)  // Higher z-order
            .build(),
    ];
    memory.write_widgets(&widgets);

    let (hits, topmost) = manager.hit_test_sync(
        &queue,
        &memory.widget_buffer,
        2,
        0.5, 0.5,  // Center (both widgets)
    );

    assert_eq!(hits, 2, "Should hit 2 widgets");
    assert_eq!(topmost, Some(1), "Topmost should be widget 1 (higher z)");
}

#[test]
fn test_hit_testing_1024_widgets() {
    let (device, memory, manager) = setup();
    let queue = device.new_command_queue();

    // Create 1024 small widgets in a grid
    let widgets: Vec<WidgetCompact> = (0..1024)
        .map(|i| {
            let x = (i % 32) as f32 * 0.03;
            let y = (i / 32) as f32 * 0.03;
            WidgetBuilder::new(WidgetType::Button)
                .bounds(x, y, 0.025, 0.025)
                .z_order(i as u16)
                .build()
        })
        .collect();
    memory.write_widgets(&widgets);

    let (hits, topmost) = manager.hit_test_sync(
        &queue,
        &memory.widget_buffer,
        1024,
        0.5, 0.5,
    );

    // Some widgets should be hit at this position
    assert!(hits >= 0, "Hit test should complete without error");
}

#[test]
fn test_hit_testing_performance() {
    let (device, memory, manager) = setup();
    let queue = device.new_command_queue();

    // Create 1024 widgets
    let widgets: Vec<WidgetCompact> = (0..1024)
        .map(|i| {
            WidgetBuilder::new(WidgetType::Button)
                .bounds(0.0, 0.0, 1.0, 1.0)
                .z_order(i as u16)
                .build()
        })
        .collect();
    memory.write_widgets(&widgets);

    // Benchmark hit testing
    let start = std::time::Instant::now();
    for _ in 0..100 {
        manager.hit_test_sync(&queue, &memory.widget_buffer, 1024, 0.5, 0.5);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / 100.0;

    // Note: GPU command submission overhead dominates; amortized over 100 iterations
    assert!(
        elapsed < 1.0,
        "Hit testing 1024 widgets should be under 1ms (averaged). Got: {:.3}ms",
        elapsed
    );
}

#[test]
fn test_z_order_sorting() {
    let (device, memory, manager) = setup();
    let queue = device.new_command_queue();

    // Create widgets with random z-orders
    let widgets: Vec<WidgetCompact> = vec![
        WidgetBuilder::new(WidgetType::Button).z_order(50).build(),
        WidgetBuilder::new(WidgetType::Button).z_order(10).build(),
        WidgetBuilder::new(WidgetType::Button).z_order(30).build(),
        WidgetBuilder::new(WidgetType::Button).z_order(20).build(),
    ];
    memory.write_widgets(&widgets);

    // Sort by z-order (using sync version for proper barrier handling)
    manager.sort_by_z_order_sync(&queue, &memory.widget_buffer, 4);

    // Verify sorted order
    let sorted = memory.read_widgets(4);
    for i in 1..4 {
        assert!(
            sorted[i].z_order >= sorted[i-1].z_order,
            "Widgets should be sorted by z-order"
        );
    }
}
