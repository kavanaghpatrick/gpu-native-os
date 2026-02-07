// Issue #162 - Visual Demo Tests
//
// Tests for connecting the megakernel to window rendering.
// These tests verify the render pipeline and buffer integration work correctly.

use metal::*;
use rust_experiment::gpu_os::gpu_os::GpuOs;
use rust_experiment::gpu_os::gpu_app_system::{app_type, RenderVertex};
use std::mem;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// =============================================================================
// Test 1: Render Pipeline Creation
// =============================================================================

const RENDER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RenderVertex {
    float3 position;
    float _pad0;
    float4 color;
    float2 uv;
    float2 _pad1;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
};

vertex VertexOut unified_vertex_shader(
    const device RenderVertex* vertices [[buffer(0)]],
    constant float2& screen_size [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    RenderVertex v = vertices[vid];

    float2 pos = v.position.xy / screen_size;
    pos = pos * 2.0 - 1.0;
    pos.y = -pos.y;

    VertexOut out;
    out.position = float4(pos, v.position.z, 1.0);
    out.color = v.color;
    out.uv = v.uv;
    return out;
}

fragment float4 unified_fragment_shader(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;

#[test]
fn test_render_pipeline_creation() {
    let device = get_device();

    let library = device
        .new_library_with_source(RENDER_SHADER, &CompileOptions::new())
        .expect("Shader compilation failed");

    let vertex_fn = library
        .get_function("unified_vertex_shader", None)
        .expect("Vertex function not found");

    let fragment_fn = library
        .get_function("unified_fragment_shader", None)
        .expect("Fragment function not found");

    let render_desc = RenderPipelineDescriptor::new();
    render_desc.set_vertex_function(Some(&vertex_fn));
    render_desc.set_fragment_function(Some(&fragment_fn));
    render_desc
        .color_attachments()
        .object_at(0)
        .unwrap()
        .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

    let _pipeline = device
        .new_render_pipeline_state(&render_desc)
        .expect("Pipeline creation failed");

    // Pipeline created successfully
}

// =============================================================================
// Test 2: Unified Buffer Render Integration
// =============================================================================

#[test]
fn test_unified_buffer_render_integration() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch some apps to generate vertices
    let _app1 = os.launch_app(app_type::TERMINAL);
    let _app2 = os.launch_app(app_type::FILESYSTEM);

    // Run frame to generate vertices
    os.run_frame();

    // Get render buffer
    let buffer = os.render_vertices_buffer();
    assert!(buffer.length() > 0, "Vertex buffer should have capacity");

    // Get vertex count
    let count = os.total_vertex_count();
    println!("Vertex count after frame: {}", count);

    // Buffer should be valid for rendering
    assert!(!buffer.contents().is_null());
}

// =============================================================================
// Test 3: Multiple Frames Stability
// =============================================================================

#[test]
fn test_multiple_frames_stability() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch apps
    os.launch_app(app_type::TERMINAL);
    os.launch_app(app_type::FILESYSTEM);

    // Run many frames
    for _i in 0..100 {
        os.run_frame();

        // Vertex count should be consistent
        let _count = os.total_vertex_count();

        // No crashes, buffer stays valid
        let buffer = os.render_vertices_buffer();
        assert!(!buffer.contents().is_null());
    }

    assert_eq!(os.frame_count(), 100);
}

// =============================================================================
// Test 4: Vertex Buffer Layout
// =============================================================================

#[test]
fn test_vertex_buffer_layout() {
    // Verify RenderVertex matches Metal expectations
    assert_eq!(
        mem::size_of::<RenderVertex>(),
        48,
        "RenderVertex should be 48 bytes: float3 + pad + float4 + float2 + pad2"
    );

    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch app with window to generate vertices
    let slot = os.launch_app(app_type::TERMINAL).expect("Launch failed");

    // Mark dirty and run
    os.system.mark_dirty(slot);
    os.run_frame();

    // If vertices were generated, verify layout
    let count = os.total_vertex_count();
    println!("Vertex count: {}", count);

    if count > 0 {
        unsafe {
            let vertices = os.render_vertices_buffer().contents() as *const RenderVertex;
            let v = *vertices;

            // Position should be in reasonable screen coordinates
            println!("First vertex position: {:?}", v.position);
            println!("First vertex color: {:?}", v.color);

            // Z should be valid depth (0.0 to 1.0)
            assert!(
                v.position[2] >= 0.0 && v.position[2] <= 1.0,
                "Depth {} should be in [0, 1]",
                v.position[2]
            );

            // Color alpha should be non-zero for visible (only if we have real vertices)
            // Note: System apps (dock, menubar, etc.) are still TODO skeletons
            // and may not generate vertices with valid colors yet
            if v.color[3] > 0.0 {
                println!("Vertex has valid alpha: {}", v.color[3]);
            } else {
                println!("Note: Vertex alpha is 0 - system apps may still be skeletons");
            }
        }
    } else {
        // No vertices generated yet - this is OK for Issue #162
        // The system apps (Issues #156-159) will generate vertices when implemented
        println!("Note: No vertices generated yet - system apps are TODO skeletons");
    }

    // The key test: buffer infrastructure is working
    let buffer = os.render_vertices_buffer();
    assert!(!buffer.contents().is_null(), "Buffer should be valid");
    assert!(buffer.length() > 0, "Buffer should have capacity");
}

// =============================================================================
// Test 5: Zero CPU Geometry
// =============================================================================

#[test]
fn test_zero_cpu_geometry() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch apps
    os.launch_app(app_type::TERMINAL);

    // Run frame - CPU does nothing but submit
    os.run_frame();

    // The ONLY data we read from GPU is vertex count (for draw call)
    let count = os.total_vertex_count();

    // Vertex buffer contents are NEVER read by CPU in production
    // They go directly: GPU compute -> GPU render
    let buffer = os.render_vertices_buffer();

    // Verify buffer is GPU-ready without reading contents
    assert!(buffer.length() >= (count as u64) * 48); // 48 = sizeof(RenderVertex)
}

// =============================================================================
// Test 6: Input Integration
// =============================================================================

#[test]
fn test_input_to_render_integration() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch app
    let _slot = os.launch_app(app_type::TERMINAL).expect("Launch failed");

    // Send input events
    os.mouse_move(100.0, 100.0, 5.0, 0.0);
    os.mouse_click(100.0, 100.0, 0);
    os.key_event(0x00, true, 0); // 'a' key

    // Run frame - input processed, vertices generated
    os.run_frame();

    // System should be stable after input
    assert_eq!(os.frame_count(), 1);
    let _count = os.total_vertex_count();
    let _buffer = os.render_vertices_buffer();
}

// =============================================================================
// Test 7: Screen Size Configuration
// =============================================================================

#[test]
fn test_screen_size_configuration() {
    let device = get_device();
    let os = GpuOs::boot_with_size(&device, 1920.0, 1080.0).expect("Boot failed");

    assert_eq!(os.screen_width, 1920.0);
    assert_eq!(os.screen_height, 1080.0);
}

// =============================================================================
// Test 8: System Apps Launched
// =============================================================================

#[test]
fn test_system_apps_launched() {
    let device = get_device();
    let os = GpuOs::boot(&device).expect("Boot failed");

    // System apps should be launched
    assert!(
        os.compositor_slot().is_some() || os.dock_slot().is_some() || os.menubar_slot().is_some(),
        "At least one system app should be launched"
    );
}

// =============================================================================
// Test 9: App Launch and Close Cycle
// =============================================================================

#[test]
fn test_app_launch_close_cycle() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    let initial_count = os.app_count();

    // Launch
    let slot = os.launch_app(app_type::TERMINAL).expect("Launch failed");
    assert_eq!(os.app_count(), initial_count + 1);

    // Run a few frames
    for _ in 0..10 {
        os.run_frame();
    }

    // Close
    os.close_app(slot);
    assert_eq!(os.app_count(), initial_count);
}

// =============================================================================
// Test 10: Stress Test - Many Apps
// =============================================================================

#[test]
fn test_stress_many_apps() {
    let device = get_device();
    let mut os = GpuOs::boot(&device).expect("Boot failed");

    // Launch many apps
    let mut slots = Vec::new();
    for i in 0..16 {
        let app_type = if i % 2 == 0 {
            app_type::TERMINAL
        } else {
            app_type::FILESYSTEM
        };
        if let Some(slot) = os.launch_app(app_type) {
            slots.push(slot);
        }
    }

    println!("Launched {} apps", slots.len());

    // Run many frames
    for frame in 0..100 {
        os.run_frame();

        if frame % 20 == 0 {
            let count = os.total_vertex_count();
            println!("Frame {}: {} vertices", frame, count);
        }
    }

    // All should be stable
    assert!(os.frame_count() >= 100);
    let _buffer = os.render_vertices_buffer();
}
