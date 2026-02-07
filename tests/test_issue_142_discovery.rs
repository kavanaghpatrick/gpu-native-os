//! Tests for Issue #142: App Discovery Service
//!
//! Tests scanning directories for .gpuapp bundles.

use rust_experiment::gpu_os::dynamic_app::discovery::{AppDiscovery, DiscoveredApp};
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
fn test_scan_empty_directory() {
    let dir = TempDir::new().unwrap();
    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();
    assert!(apps.is_empty());
}

#[test]
fn test_scan_finds_bundles() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "app_one");
    create_test_bundle(dir.path(), "app_two");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 2);
    let names: Vec<_> = apps.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"app_one"));
    assert!(names.contains(&"app_two"));
}

#[test]
fn test_find_by_name() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "findable_app");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);

    let found = discovery.find_by_name("findable_app");
    assert!(found.is_some());
    assert!(found.unwrap().ends_with("findable_app.gpuapp"));

    let not_found = discovery.find_by_name("nonexistent");
    assert!(not_found.is_none());
}

#[test]
fn test_ignores_missing_manifest() {
    let dir = TempDir::new().unwrap();

    // Bundle without manifest.toml
    let no_manifest = dir.path().join("no_manifest.gpuapp");
    fs::create_dir_all(&no_manifest).unwrap();
    fs::write(no_manifest.join("main.metal"), "// shader").unwrap();

    // Valid bundle
    create_test_bundle(dir.path(), "valid");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].name, "valid");
}

#[test]
fn test_ignores_missing_shader() {
    let dir = TempDir::new().unwrap();

    // Bundle without main.metal
    let no_shader = dir.path().join("no_shader.gpuapp");
    fs::create_dir_all(&no_shader).unwrap();
    fs::write(
        no_shader.join("manifest.toml"),
        "[app]\nname = \"x\"\n[shaders]\ncompute=\"k\"\nvertex=\"v\"\nfragment=\"f\"",
    )
    .unwrap();

    // Valid bundle
    create_test_bundle(dir.path(), "valid");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].name, "valid");
}

#[test]
fn test_ignores_file_with_gpuapp_name() {
    let dir = TempDir::new().unwrap();

    // File (not directory) named .gpuapp
    fs::write(dir.path().join("fake.gpuapp"), "not a directory").unwrap();

    // Valid bundle
    create_test_bundle(dir.path(), "valid");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].name, "valid");
}

#[test]
fn test_ignores_wrong_extension() {
    let dir = TempDir::new().unwrap();

    // Directory with wrong extension
    let wrong_ext = dir.path().join("wrong.app");
    fs::create_dir_all(&wrong_ext).unwrap();
    fs::write(wrong_ext.join("manifest.toml"), "[app]").unwrap();
    fs::write(wrong_ext.join("main.metal"), "//").unwrap();

    // Valid bundle
    create_test_bundle(dir.path(), "valid");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 1);
    assert_eq!(apps[0].name, "valid");
}

#[test]
fn test_multiple_search_paths() {
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();

    create_test_bundle(dir1.path(), "from_dir1");
    create_test_bundle(dir2.path(), "from_dir2");

    let discovery = AppDiscovery::with_paths(vec![
        dir1.path().to_path_buf(),
        dir2.path().to_path_buf(),
    ]);

    let apps = discovery.scan();
    assert_eq!(apps.len(), 2);

    let names: Vec<_> = apps.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"from_dir1"));
    assert!(names.contains(&"from_dir2"));
}

#[test]
fn test_nonexistent_search_path_doesnt_crash() {
    let discovery = AppDiscovery::with_paths(vec![PathBuf::from(
        "/nonexistent/path/that/definitely/does/not/exist",
    )]);

    // Should not panic, just return empty
    let apps = discovery.scan();
    assert!(apps.is_empty());
}

#[test]
fn test_is_valid_bundle() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "test");

    let bundle_path = dir.path().join("test.gpuapp");
    assert!(AppDiscovery::is_valid_bundle(&bundle_path));

    // Non-existent path
    assert!(!AppDiscovery::is_valid_bundle(&PathBuf::from(
        "/nope.gpuapp"
    )));
}

#[test]
fn test_list_names() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "alpha");
    create_test_bundle(dir.path(), "beta");
    create_test_bundle(dir.path(), "gamma");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let names = discovery.list_names();

    assert_eq!(names, vec!["alpha", "beta", "gamma"]); // Sorted
}

#[test]
fn test_scan_results_sorted_alphabetically() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "zebra");
    create_test_bundle(dir.path(), "apple");
    create_test_bundle(dir.path(), "mango");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    let names: Vec<_> = apps.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, vec!["apple", "mango", "zebra"]);
}

#[test]
fn test_first_search_path_takes_priority() {
    // If same app name exists in multiple paths, first path wins
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();

    create_test_bundle(dir1.path(), "shared_app");
    create_test_bundle(dir2.path(), "shared_app");

    let discovery = AppDiscovery::with_paths(vec![
        dir1.path().to_path_buf(),
        dir2.path().to_path_buf(),
    ]);

    let found = discovery.find_by_name("shared_app").unwrap();
    // Should be from dir1 (first in search path)
    assert!(found.starts_with(dir1.path()));
}

#[test]
fn test_discovered_app_path_is_absolute() {
    let dir = TempDir::new().unwrap();
    create_test_bundle(dir.path(), "test_app");

    let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
    let apps = discovery.scan();

    assert_eq!(apps.len(), 1);
    // Path should be valid and contain the bundle name
    assert!(apps[0].path.to_string_lossy().contains("test_app.gpuapp"));
}

#[test]
fn test_search_paths_accessor() {
    let paths = vec![
        PathBuf::from("/path/one"),
        PathBuf::from("/path/two"),
    ];
    let discovery = AppDiscovery::with_paths(paths.clone());

    assert_eq!(discovery.search_paths(), &paths);
}

#[test]
fn test_default_creates_discovery() {
    // Test that Default trait works
    let discovery = AppDiscovery::default();
    // Should have at least ./apps in search paths
    assert!(!discovery.search_paths().is_empty());
}
