// Tests for Issue #17: Hybrid Rendering - Compute Logic + Fragment Pixels
//
// These tests verify the hybrid rendering pipeline.
// Run with: cargo test --test test_issue_17_render

use metal::Device;
use rust_experiment::gpu_os::render::*;
use rust_experiment::gpu_os::memory::{GpuMemory, DrawArguments};
use rust_experiment::gpu_os::widget::{WidgetBuilder, WidgetType};

fn setup() -> (Device, GpuMemory, HybridRenderer) {
    let device = Device::system_default().expect("No Metal device");
    let memory = GpuMemory::new(&device, 1024);
    let renderer = HybridRenderer::new(&device).expect("Renderer should compile");
    (device, memory, renderer)
}

#[test]
fn test_widget_vertex_size() {
    assert_eq!(
        WidgetVertex::SIZE, 64,
        "WidgetVertex must be 64 bytes"
    );
}

#[test]
fn test_hybrid_renderer_compiles() {
    let device = Device::system_default().expect("No Metal device");
    let result = HybridRenderer::new(&device);

    assert!(result.is_ok(), "Hybrid renderer should compile");
}

#[test]
fn test_vertex_buffer_size_calculation() {
    // 1024 widgets × 6 vertices × 64 bytes = 393216 bytes
    let size = HybridRenderer::vertex_buffer_size(1024);
    assert_eq!(size, 1024 * 6 * 64);
}

#[test]
fn test_generate_vertices_single_widget() {
    let (device, memory, renderer) = setup();
    let queue = device.new_command_queue();

    // Create one widget
    let widgets = vec![
        WidgetBuilder::new(WidgetType::Button)
            .bounds(0.1, 0.1, 0.2, 0.1)
            .background_color(0.2, 0.4, 0.8)
            .corner_radius(4.0)
            .build(),
    ];
    memory.write_widgets(&widgets);

    let vertex_buffer = device.new_buffer(
        HybridRenderer::vertex_buffer_size(1) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let (draw_args, time_ms) = renderer.generate_vertices_sync(
        &queue,
        &memory.widget_buffer,
        1,
        &vertex_buffer,
        &memory.draw_args_buffer,
    );

    assert_eq!(
        draw_args.vertex_count, 6,
        "Single widget should produce 6 vertices"
    );
    assert_eq!(draw_args.instance_count, 1);
}

#[test]
fn test_generate_vertices_100_widgets() {
    let (device, memory, renderer) = setup();
    let queue = device.new_command_queue();

    // Create 100 widgets
    let widgets: Vec<_> = (0..100)
        .map(|i| {
            WidgetBuilder::new(WidgetType::Button)
                .bounds(
                    (i % 10) as f32 * 0.1,
                    (i / 10) as f32 * 0.1,
                    0.08,
                    0.08,
                )
                .z_order(i as u16)
                .build()
        })
        .collect();
    memory.write_widgets(&widgets);

    let vertex_buffer = device.new_buffer(
        HybridRenderer::vertex_buffer_size(100) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let (draw_args, _) = renderer.generate_vertices_sync(
        &queue,
        &memory.widget_buffer,
        100,
        &vertex_buffer,
        &memory.draw_args_buffer,
    );

    assert_eq!(
        draw_args.vertex_count, 600,
        "100 widgets should produce 600 vertices"
    );
}

#[test]
fn test_vertex_generation_performance() {
    let (device, memory, renderer) = setup();
    let queue = device.new_command_queue();

    // Create 1024 widgets
    let widgets: Vec<_> = (0..1024)
        .map(|i| {
            WidgetBuilder::new(WidgetType::Button)
                .bounds(0.0, 0.0, 0.01, 0.01)
                .z_order(i as u16)
                .build()
        })
        .collect();
    memory.write_widgets(&widgets);

    let vertex_buffer = device.new_buffer(
        HybridRenderer::vertex_buffer_size(1024) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Warm up
    for _ in 0..5 {
        renderer.generate_vertices_sync(
            &queue,
            &memory.widget_buffer,
            1024,
            &vertex_buffer,
            &memory.draw_args_buffer,
        );
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..100 {
        let (_, time_ms) = renderer.generate_vertices_sync(
            &queue,
            &memory.widget_buffer,
            1024,
            &vertex_buffer,
            &memory.draw_args_buffer,
        );
        times.push(time_ms);
    }

    let avg_ms = times.iter().sum::<f64>() / times.len() as f64;

    assert!(
        avg_ms < 0.5,
        "Vertex generation for 1024 widgets should be under 0.5ms. Got: {:.3}ms",
        avg_ms
    );
}

#[test]
fn test_draw_args_correctness() {
    let (device, memory, renderer) = setup();
    let queue = device.new_command_queue();

    let widgets: Vec<_> = (0..50)
        .map(|i| {
            WidgetBuilder::new(WidgetType::Container)
                .z_order(i as u16)
                .build()
        })
        .collect();
    memory.write_widgets(&widgets);

    let vertex_buffer = device.new_buffer(
        HybridRenderer::vertex_buffer_size(50) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let (draw_args, _) = renderer.generate_vertices_sync(
        &queue,
        &memory.widget_buffer,
        50,
        &vertex_buffer,
        &memory.draw_args_buffer,
    );

    assert_eq!(draw_args.vertex_count, 300); // 50 × 6
    assert_eq!(draw_args.instance_count, 1);
    assert_eq!(draw_args.vertex_start, 0);
    assert_eq!(draw_args.base_instance, 0);
}

#[test]
fn test_frame_renderer_creation() {
    let device = Device::system_default().expect("No Metal device");
    let result = FrameRenderer::new(&device);

    assert!(result.is_ok(), "Frame renderer should compile");
}

#[test]
fn test_full_frame_under_5ms() {
    let device = Device::system_default().expect("No Metal device");
    let memory = GpuMemory::new(&device, 1024);
    let frame_renderer = FrameRenderer::new(&device).expect("Frame renderer");
    let font_atlas = rust_experiment::gpu_os::text::FontAtlas::create_default(&device)
        .expect("Font atlas");
    let queue = device.new_command_queue();

    // Create widgets
    let widgets: Vec<_> = (0..256)
        .map(|i| {
            WidgetBuilder::new(WidgetType::Button)
                .bounds(
                    (i % 16) as f32 * 0.06,
                    (i / 16) as f32 * 0.06,
                    0.05,
                    0.05,
                )
                .z_order(i as u16)
                .build()
        })
        .collect();
    memory.write_widgets(&widgets);

    let stats = frame_renderer.render_frame_with_stats(&queue, &memory, &font_atlas);

    assert!(
        stats.total_time_ms < 5.0,
        "Full frame should be under 5ms. Got: {:.3}ms",
        stats.total_time_ms
    );

    println!("Frame stats: compute={:.3}ms render={:.3}ms total={:.3}ms",
        stats.compute_time_ms, stats.render_time_ms, stats.total_time_ms);
}
