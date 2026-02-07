//! Tests for Issue #144: DynamicDesktopApp Wrapper
//!
//! Tests wrapping DynamicGpuApp in a DesktopApp for windowed execution.

use rust_experiment::gpu_os::dynamic_app::{DynamicGpuApp, DynamicDesktopApp};
use rust_experiment::gpu_os::desktop::app::DesktopApp;
use std::fs;
use tempfile::TempDir;

fn create_test_bundle(dir: &std::path::Path) -> std::path::PathBuf {
    let bundle = dir.join("test.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Desktop Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
thread_count = 64
vertex_count = 6
preferred_size = [640.0, 480.0]
"#,
    )
    .unwrap();

    fs::write(
        bundle.join("main.metal"),
        r#"
kernel void k(device float4* verts [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid < 6) {
        float2 p[6] = {float2(-1,-1),float2(1,-1),float2(-1,1),float2(-1,1),float2(1,-1),float2(1,1)};
        verts[tid] = float4(p[tid], 0, 1);
    }
}
struct V { float4 p [[position]]; float4 c; };
vertex V v(device float4* verts [[buffer(0)]], uint vid [[vertex_id]]) {
    V o; o.p = verts[vid]; o.c = float4(0.5, 0.8, 1.0, 1.0); return o;
}
fragment float4 f(V in [[stage_in]]) { return in.c; }
"#,
    )
    .unwrap();

    bundle
}

#[test]
fn test_create_desktop_wrapper() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    assert_eq!(desktop_app.name(), "Desktop Test");
}

#[test]
fn test_preferred_size_from_manifest() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    let (w, h) = desktop_app.preferred_size();
    assert!((w - 640.0).abs() < 0.1);
    assert!((h - 480.0).abs() < 0.1);
}

#[test]
fn test_update_advances_time() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let mut desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    desktop_app.update(0.016);
    desktop_app.update(0.016);
    desktop_app.update(0.016);

    // Time should have advanced (we can't access private fields, but update shouldn't panic)
}

#[test]
fn test_init_succeeds() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let mut desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    // Init should succeed (already initialized in new())
    assert!(desktop_app.init(&device).is_ok());
}

#[test]
fn test_icon_index() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    // Default icon index for dynamic apps
    assert_eq!(desktop_app.icon_index(), 0);
}

#[test]
fn test_wrapper_creation_with_different_sizes() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();

    // Create bundle with different size
    let bundle = dir.path().join("large.gpuapp");
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Large App"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
preferred_size = [1920.0, 1080.0]
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

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    let (w, h) = desktop_app.preferred_size();
    assert!((w - 1920.0).abs() < 0.1);
    assert!((h - 1080.0).abs() < 0.1);
}

#[test]
fn test_should_close_default() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let mut desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    // Default should allow closing
    assert!(desktop_app.should_close());
}

#[test]
fn test_has_unsaved_changes_default() {
    let device = metal::Device::system_default().expect("No Metal device");
    let dir = TempDir::new().unwrap();
    let bundle = create_test_bundle(dir.path());

    let gpu_app = DynamicGpuApp::load(&bundle, &device).unwrap();
    let desktop_app = DynamicDesktopApp::new(gpu_app, &device).unwrap();

    // Default should have no unsaved changes
    assert!(!desktop_app.has_unsaved_changes());
}
