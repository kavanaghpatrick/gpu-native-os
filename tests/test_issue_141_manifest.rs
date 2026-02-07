//! Tests for Issue #141: Dynamic App Manifest Parsing
//!
//! Tests the TOML manifest parsing for .gpuapp bundles.

use rust_experiment::gpu_os::dynamic_app::manifest::*;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_parse_minimal_manifest() {
    let toml = r#"
[app]
name = "Test App"

[shaders]
compute = "test_kernel"
vertex = "vertex_main"
fragment = "fragment_main"
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    assert_eq!(manifest.app.name, "Test App");
    assert_eq!(manifest.app.version, "1.0"); // default
    assert_eq!(manifest.shaders.compute, "test_kernel");
    assert_eq!(manifest.shaders.vertex, "vertex_main");
    assert_eq!(manifest.shaders.fragment, "fragment_main");
    assert_eq!(manifest.config.thread_count, 1024); // default
}

#[test]
fn test_parse_full_manifest() {
    let toml = r#"
[app]
name = "Game of Life"
version = "2.0"

[shaders]
compute = "compute_life"
vertex = "vertex_quad"
fragment = "fragment_color"

[config]
thread_count = 2048
vertex_count = 6
clear_color = [0.0, 0.0, 0.0, 1.0]
preferred_size = [640.0, 480.0]

[buffers]
params = { size = 64, slot = 2 }
grid_a = { size = 4096, slot = 3 }
grid_b = { size = 4096, slot = 4 }
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    assert_eq!(manifest.app.name, "Game of Life");
    assert_eq!(manifest.app.version, "2.0");
    assert_eq!(manifest.config.thread_count, 2048);
    assert_eq!(manifest.buffers.len(), 3);
    assert_eq!(manifest.buffers["grid_a"].slot, 3);
    assert_eq!(manifest.buffers["grid_a"].size, 4096);
    assert_eq!(manifest.preferred_size(), (640.0, 480.0));
}

#[test]
fn test_load_from_bundle() {
    let dir = TempDir::new().unwrap();
    let bundle = dir.path().join("test.gpuapp");
    std::fs::create_dir(&bundle).unwrap();

    std::fs::write(
        bundle.join("manifest.toml"),
        r#"
[app]
name = "Bundle Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#,
    )
    .unwrap();

    let manifest = AppManifest::load(&bundle).unwrap();
    assert_eq!(manifest.app.name, "Bundle Test");
}

#[test]
fn test_shader_path() {
    let toml = r#"
[app]
name = "Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    let bundle_path = PathBuf::from("/some/path/app.gpuapp");
    let shader_path = manifest.shader_path(&bundle_path);
    assert_eq!(shader_path, PathBuf::from("/some/path/app.gpuapp/main.metal"));
}

#[test]
fn test_app_buffer_configs_sorted() {
    let toml = r#"
[app]
name = "Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[buffers]
params = { size = 64, slot = 2 }
z_last = { size = 100, slot = 5 }
a_first = { size = 100, slot = 3 }
m_middle = { size = 100, slot = 4 }
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    let configs = manifest.app_buffer_configs();

    // Should be sorted by slot, excluding params
    assert_eq!(configs.len(), 3);
    assert_eq!(configs[0].1.slot, 3);
    assert_eq!(configs[1].1.slot, 4);
    assert_eq!(configs[2].1.slot, 5);
}

#[test]
fn test_params_config() {
    let toml = r#"
[app]
name = "Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[buffers]
params = { size = 128, slot = 2 }
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    let params = manifest.params_config().unwrap();
    assert_eq!(params.size, 128);
    assert_eq!(params.slot, 2);
}

#[test]
fn test_dynamic_vertex_count() {
    let toml = r#"
[app]
name = "Dynamic"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
vertex_count = { dynamic = true, offset = 0 }
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    assert!(manifest.config.vertex_count.is_dynamic());
    assert!(manifest.config.vertex_count.static_count().is_none());
}

#[test]
fn test_static_vertex_count() {
    let toml = r#"
[app]
name = "Static"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"

[config]
vertex_count = 12
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    assert!(!manifest.config.vertex_count.is_dynamic());
    assert_eq!(manifest.config.vertex_count.static_count(), Some(12));
}

#[test]
fn test_invalid_manifest_error() {
    let result: Result<AppManifest, _> = toml::from_str("invalid toml {{{");
    assert!(result.is_err());
}

#[test]
fn test_missing_required_field() {
    // Missing shaders section
    let toml = r#"
[app]
name = "Test"
"#;
    let result: Result<AppManifest, _> = toml::from_str(toml);
    assert!(result.is_err());
}

#[test]
fn test_load_nonexistent_bundle() {
    let result = AppManifest::load(&PathBuf::from("/nonexistent/path.gpuapp"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read manifest"));
}

#[test]
fn test_default_clear_color() {
    let toml = r#"
[app]
name = "Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    // Default clear color is dark blue-gray
    assert_eq!(manifest.config.clear_color, [0.1, 0.1, 0.15, 1.0]);
}

#[test]
fn test_default_preferred_size() {
    let toml = r#"
[app]
name = "Test"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#;
    let manifest: AppManifest = toml::from_str(toml).unwrap();
    assert_eq!(manifest.preferred_size(), (800.0, 600.0));
}
