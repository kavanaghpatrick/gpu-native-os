//! Tests for Issue #146: Test Bundles and End-to-End Verification
//!
//! Verifies the complete dynamic app loading flow works.
//! Note: Desktop wrapper tests archived - need rewrite for megakernel architecture.

use rust_experiment::gpu_os::dynamic_app::{AppDiscovery, DynamicGpuApp};
use rust_experiment::gpu_os::app::GpuApp;
use std::path::Path;

#[test]
fn test_hello_gpu_bundle_exists() {
    let bundle = Path::new("./apps/hello_gpu.gpuapp");

    if !bundle.exists() {
        println!("Skipping - hello_gpu.gpuapp bundle not found at {:?}", bundle);
        return;
    }

    assert!(bundle.exists(), "hello_gpu.gpuapp bundle should exist");
    assert!(bundle.join("manifest.toml").exists(), "manifest.toml should exist");
    assert!(bundle.join("main.metal").exists(), "main.metal should exist");
}

#[test]
fn test_hello_gpu_manifest_valid() {
    let bundle = Path::new("./apps/hello_gpu.gpuapp");

    if !bundle.exists() {
        println!("Skipping - bundle not found");
        return;
    }

    let manifest_content = std::fs::read_to_string(bundle.join("manifest.toml")).unwrap();
    let manifest: rust_experiment::gpu_os::dynamic_app::AppManifest =
        toml::from_str(&manifest_content).expect("Manifest should parse");

    assert_eq!(manifest.app.name, "Hello GPU");
    assert_eq!(manifest.shaders.compute, "hello_compute");
    assert_eq!(manifest.shaders.vertex, "hello_vertex");
    assert_eq!(manifest.shaders.fragment, "hello_fragment");
}

#[test]
fn test_hello_gpu_loads() {
    let bundle = Path::new("./apps/hello_gpu.gpuapp");

    if !bundle.exists() {
        println!("Skipping - bundle not found");
        return;
    }

    let device = metal::Device::system_default().expect("No Metal device");
    let gpu_app = DynamicGpuApp::load(bundle, &device).expect("Failed to load hello_gpu");

    assert_eq!(gpu_app.name(), "Hello GPU");
    assert_eq!(gpu_app.thread_count(), 6);
    assert_eq!(gpu_app.vertex_count(), 6);
}

#[test]
fn test_discovery_finds_hello_gpu() {
    let bundle = Path::new("./apps/hello_gpu.gpuapp");

    if !bundle.exists() {
        println!("Skipping - bundle not found");
        return;
    }

    let discovery = AppDiscovery::new();
    let found = discovery.find_by_name("hello_gpu");

    assert!(found.is_some(), "Discovery should find hello_gpu");
}

#[test]
fn test_discovery_lists_hello_gpu() {
    let bundle = Path::new("./apps/hello_gpu.gpuapp");

    if !bundle.exists() {
        println!("Skipping - bundle not found");
        return;
    }

    let discovery = AppDiscovery::new();
    let apps = discovery.scan();

    let names: Vec<_> = apps.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"hello_gpu"), "Should list hello_gpu in scan results");
}

#[test]
fn test_gpu_app_pipelines_valid() {
    let bundle = Path::new("./apps/hello_gpu.gpuapp");

    if !bundle.exists() {
        println!("Skipping - bundle not found");
        return;
    }

    let device = metal::Device::system_default().expect("No Metal device");

    // 1. Discover the app
    let discovery = AppDiscovery::new();
    let bundle_path = discovery.find_by_name("hello_gpu").expect("Should find hello_gpu");

    // 2. Load as GPU app
    let gpu_app = DynamicGpuApp::load(&bundle_path, &device).expect("Should load");

    // 3. Verify pipelines exist
    let _ = gpu_app.compute_pipeline();
    let _ = gpu_app.render_pipeline();
    let _ = gpu_app.vertices_buffer();
    let _ = gpu_app.params_buffer();

    println!("GPU app pipeline test passed for hello_gpu!");
}

// TODO: Add tests for DynamicGpuApp integration with GpuAppSystem megakernel
// The old DynamicDesktopApp wrapper was archived - needs rewrite for megakernel
