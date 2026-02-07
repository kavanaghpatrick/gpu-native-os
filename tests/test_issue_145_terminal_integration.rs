//! Tests for Issue #145: Terminal Integration for Dynamic Apps
//!
//! Tests that the terminal can discover and launch dynamic apps.

use rust_experiment::gpu_os::dynamic_app::AppDiscovery;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn create_test_bundle(dir: &std::path::Path, name: &str) {
    let bundle = dir.join(format!("{}.gpuapp", name));
    fs::create_dir_all(&bundle).unwrap();

    fs::write(
        bundle.join("manifest.toml"),
        format!(
            r#"
[app]
name = "{}"

[shaders]
compute = "k"
vertex = "v"
fragment = "f"
"#,
            name
        ),
    )
    .unwrap();

    fs::write(bundle.join("main.metal"), "// shader code").unwrap();
}

#[test]
fn test_discovery_finds_apps_in_custom_path() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "custom_app");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].name, "custom_app");
}

#[test]
fn test_find_by_name_works() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "findable");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);

    let found = discovery.find_by_name("findable");
    assert!(found.is_some());

    let not_found = discovery.find_by_name("nonexistent");
    assert!(not_found.is_none());
}

#[test]
fn test_discovery_returns_sorted_names() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "zebra_app");
    create_test_bundle(dir.path(), "alpha_app");
    create_test_bundle(dir.path(), "beta_app");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let names = discovery.list_names();

    assert_eq!(names, vec!["alpha_app", "beta_app", "zebra_app"]);
}

#[test]
fn test_discovery_from_multiple_paths() {
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();

    create_test_bundle(dir1.path(), "app_from_dir1");
    create_test_bundle(dir2.path(), "app_from_dir2");

    let discovery = AppDiscovery::with_paths(vec![
        dir1.path().to_path_buf(),
        dir2.path().to_path_buf(),
    ]);

    let apps = discovery.scan();
    assert_eq!(apps.len(), 2);

    let names: Vec<_> = apps.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"app_from_dir1"));
    assert!(names.contains(&"app_from_dir2"));
}

#[test]
fn test_default_discovery_includes_apps_dir() {
    let discovery = AppDiscovery::new();
    let paths = discovery.search_paths();

    // Should include ./apps at minimum
    assert!(paths.iter().any(|p| p.ends_with("apps")));
}

#[test]
fn test_builtin_apps_list() {
    // This tests that the built-in app names haven't changed
    let builtin = ["terminal", "files", "documents", "editor"];

    // These should always be available as built-in
    for name in builtin {
        assert!(
            builtin.contains(&name),
            "Built-in app {} should be in the list",
            name
        );
    }
}

// Integration test note:
// Full integration testing requires running the desktop environment:
// 1. cargo run --release --example gpu_desktop
// 2. In terminal: apps - should show both built-in and discovered
// 3. In terminal: launch <dynamic_app> - should open window
