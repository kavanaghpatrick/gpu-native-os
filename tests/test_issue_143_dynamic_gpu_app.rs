//! Tests for Issue #143: DynamicGpuApp Implementation
//!
//! Tests loading and running GPU apps from .gpuapp bundles.

use rust_experiment::gpu_os::dynamic_app::DynamicGpuApp;
use rust_experiment::gpu_os::app::GpuApp;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn create_minimal_bundle(dir: &Path) -> std::path::PathBuf {
    let bundle = dir.join("minimal.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Minimal Test"

[shaders]
compute = "test_compute"
vertex = "test_vertex"
fragment = "test_fragment"

[config]
thread_count = 64
vertex_count = 6
"#,
    )
    .unwrap();

    fs::write(
        bundle.join("main.metal"),
        r#"
kernel void test_compute(
    device float4* vertices [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < 6) {
        float2 positions[6] = {
            float2(-1, -1), float2(1, -1), float2(-1, 1),
            float2(-1, 1), float2(1, -1), float2(1, 1)
        };
        vertices[tid] = float4(positions[tid], 0, 1);
    }
}

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut test_vertex(
    device float4* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = vertices[vid];
    out.color = float4(0.2, 0.5, 1.0, 1.0);
    return out;
}

fragment float4 test_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
"#,
    )
    .unwrap();

    bundle
}

#[test]
fn test_load_minimal_bundle() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_minimal_bundle(dir.path());

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();

    assert_eq!(app.name(), "Minimal Test");
    assert_eq!(app.thread_count(), 64);
    assert_eq!(app.vertex_count(), 6);
}

#[test]
fn test_pipelines_created() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_minimal_bundle(dir.path());

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();

    // These should not panic - pipelines were created
    let _ = app.compute_pipeline();
    let _ = app.render_pipeline();
}

#[test]
fn test_buffers_created() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_minimal_bundle(dir.path());

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();

    // Verify buffers exist and have non-zero length
    assert!(app.vertices_buffer().length() > 0);
    assert!(app.params_buffer().length() > 0);
}

#[test]
fn test_app_buffers_from_manifest() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("buffered.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Buffered"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[buffers]
params = { size = 128, slot = 2 }
data_a = { size = 1024, slot = 3 }
data_b = { size = 2048, slot = 4 }
"#,
    )
    .unwrap();

    fs::write(
        bundle.join("main.metal"),
        r#"
kernel void k() {}
struct V { float4 p [[position]]; };
vertex V v() { V o; o.p = float4(0); return o; }
fragment float4 f() { return float4(1); }
"#,
    )
    .unwrap();

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();

    // Should have 2 app buffers (data_a, data_b)
    assert_eq!(app.app_buffers().len(), 2);
}

#[test]
fn test_invalid_shader_error() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("bad.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Bad"

[shaders]
compute = "missing_function"
vertex = "v"
fragment = "f"
"#,
    )
    .unwrap();

    fs::write(bundle.join("main.metal"), "// empty shader").unwrap();

    let result = DynamicGpuApp::load(&bundle, &device);
    assert!(result.is_err());
}

#[test]
fn test_clear_color_from_manifest() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("colored.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Colored"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
clear_color = [1.0, 0.0, 0.0, 1.0]
"#,
    )
    .unwrap();

    fs::write(
        bundle.join("main.metal"),
        r#"
kernel void k() {}
struct V { float4 p [[position]]; };
vertex V v() { V o; o.p = float4(0); return o; }
fragment float4 f() { return float4(1); }
"#,
    )
    .unwrap();

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let color = app.clear_color();

    assert!((color.red - 1.0).abs() < 0.001);
    assert!((color.green - 0.0).abs() < 0.001);
}

#[test]
fn test_preferred_size() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("sized.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Sized"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
preferred_size = [640.0, 480.0]
"#,
    )
    .unwrap();

    fs::write(
        bundle.join("main.metal"),
        r#"
kernel void k() {}
struct V { float4 p [[position]]; };
vertex V v() { V o; o.p = float4(0); return o; }
fragment float4 f() { return float4(1); }
"#,
    )
    .unwrap();

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();
    assert_eq!(app.preferred_size(), (640.0, 480.0));
}

#[test]
fn test_default_preferred_size() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_minimal_bundle(dir.path());

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();
    assert_eq!(app.preferred_size(), (800.0, 600.0)); // Default
}

#[test]
fn test_manifest_accessor() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_minimal_bundle(dir.path());

    let app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let manifest = app.manifest();

    assert_eq!(manifest.app.name, "Minimal Test");
    assert_eq!(manifest.shaders.compute, "test_compute");
}

#[test]
fn test_missing_bundle_error() {
    let device = metal::Device::system_default().expect("No Metal device");
    let result = DynamicGpuApp::load(&std::path::PathBuf::from("/nonexistent.gpuapp"), &device);
    assert!(result.is_err());
}

#[test]
fn test_missing_shader_file_error() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("no_shader.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "No Shader"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#,
    )
    .unwrap();
    // Don't create main.metal

    let result = DynamicGpuApp::load(&bundle, &device);
    assert!(result.is_err());
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("Expected error"),
    };
    assert!(err.contains("Failed to read shader"));
}

#[test]
fn test_shader_syntax_error() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("syntax_error.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Syntax Error"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#,
    )
    .unwrap();

    fs::write(
        bundle.join("main.metal"),
        r#"
// This has a syntax error
kernel void k( {{{invalid syntax
"#,
    )
    .unwrap();

    let result = DynamicGpuApp::load(&bundle, &device);
    assert!(result.is_err());
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("Expected error"),
    };
    assert!(err.contains("compile") || err.contains("Failed"), "Error should mention compilation: {}", err);
}

#[test]
fn test_uses_app_shader_header() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("uses_header.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Uses Header"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#,
    )
    .unwrap();

    // This shader uses FrameState from APP_SHADER_HEADER
    fs::write(
        bundle.join("main.metal"),
        r#"
// This uses FrameState from APP_SHADER_HEADER
kernel void k(device FrameState& frame [[buffer(0)]]) {
    // Access frame.time, etc.
    float t = frame.time;
}
struct V { float4 p [[position]]; };
vertex V v() { V o; o.p = float4(0); return o; }
fragment float4 f() { return float4(1); }
"#,
    )
    .unwrap();

    // Should compile successfully because APP_SHADER_HEADER is prepended
    let result = DynamicGpuApp::load(&bundle, &device);
    assert!(result.is_ok(), "Should compile with APP_SHADER_HEADER types");
}
