//! App Discovery Service (Issue #142)
//!
//! Scans directories for .gpuapp bundles and provides lookup by name.
//!
//! # Search Paths
//! 1. `./apps/` - Current working directory
//! 2. `~/apps/` - User's home directory
//! 3. `GPU_APP_PATH` environment variable (colon-separated)

use std::env;
use std::path::{Path, PathBuf};

/// Information about a discovered app bundle
#[derive(Debug, Clone)]
pub struct DiscoveredApp {
    /// App name (derived from directory name without .gpuapp)
    pub name: String,
    /// Full path to the .gpuapp bundle directory
    pub path: PathBuf,
}

/// Service for discovering .gpuapp bundles
pub struct AppDiscovery {
    search_paths: Vec<PathBuf>,
}

impl AppDiscovery {
    /// Create discovery with default search paths:
    /// - ./apps/
    /// - ~/apps/
    /// - GPU_APP_PATH environment variable
    pub fn new() -> Self {
        let mut paths = Vec::new();

        // ./apps/ (relative to working directory)
        paths.push(PathBuf::from("./apps"));

        // ~/apps/ (user's home directory)
        if let Some(home) = dirs::home_dir() {
            paths.push(home.join("apps"));
        }

        // GPU_APP_PATH environment variable (colon-separated)
        if let Ok(custom) = env::var("GPU_APP_PATH") {
            for p in custom.split(':') {
                if !p.is_empty() {
                    paths.push(PathBuf::from(p));
                }
            }
        }

        Self { search_paths: paths }
    }

    /// Create discovery with custom search paths (for testing)
    pub fn with_paths(paths: Vec<PathBuf>) -> Self {
        Self { search_paths: paths }
    }

    /// Get the search paths being used
    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    /// Scan all search paths for .gpuapp bundles
    pub fn scan(&self) -> Vec<DiscoveredApp> {
        let mut apps = Vec::new();

        for search_path in &self.search_paths {
            if !search_path.exists() {
                continue;
            }

            if let Ok(entries) = std::fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if Self::is_valid_bundle(&path) {
                        // Extract name from directory (without .gpuapp extension)
                        if let Some(name) = path.file_stem() {
                            apps.push(DiscoveredApp {
                                name: name.to_string_lossy().to_string(),
                                path,
                            });
                        }
                    }
                }
            }
        }

        // Sort by name for consistent ordering
        apps.sort_by(|a, b| a.name.cmp(&b.name));
        apps
    }

    /// Find a specific app by name
    ///
    /// Searches through all search paths and returns the first match.
    pub fn find_by_name(&self, name: &str) -> Option<PathBuf> {
        let bundle_name = format!("{}.gpuapp", name);

        for search_path in &self.search_paths {
            let candidate = search_path.join(&bundle_name);
            if Self::is_valid_bundle(&candidate) {
                return Some(candidate);
            }
        }

        None
    }

    /// Check if a path is a valid .gpuapp bundle
    ///
    /// A valid bundle must:
    /// - Be a directory
    /// - Have .gpuapp extension
    /// - Contain manifest.toml
    /// - Contain main.metal
    pub fn is_valid_bundle(path: &Path) -> bool {
        path.is_dir()
            && path.extension().map(|e| e == "gpuapp").unwrap_or(false)
            && path.join("manifest.toml").exists()
            && path.join("main.metal").exists()
    }

    /// List all app names (convenience method)
    pub fn list_names(&self) -> Vec<String> {
        self.scan().into_iter().map(|a| a.name).collect()
    }
}

impl Default for AppDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_bundle(dir: &Path, name: &str) {
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
        fs::write(bundle.join("main.metal"), "// shader").unwrap();
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
        create_test_bundle(dir.path(), "my_app");

        let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);

        assert!(discovery.find_by_name("my_app").is_some());
        assert!(discovery.find_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_ignores_invalid_bundles() {
        let dir = TempDir::new().unwrap();

        // Missing manifest
        let no_manifest = dir.path().join("no_manifest.gpuapp");
        fs::create_dir_all(&no_manifest).unwrap();
        fs::write(no_manifest.join("main.metal"), "// shader").unwrap();

        // Missing shader
        let no_shader = dir.path().join("no_shader.gpuapp");
        fs::create_dir_all(&no_shader).unwrap();
        fs::write(no_shader.join("manifest.toml"), "[app]\nname = \"x\"").unwrap();

        // Not a directory (file with .gpuapp name)
        fs::write(dir.path().join("file.gpuapp"), "not a dir").unwrap();

        // Wrong extension
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
    fn test_nonexistent_search_path() {
        let discovery = AppDiscovery::with_paths(vec![
            PathBuf::from("/nonexistent/path/that/does/not/exist"),
        ]);

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
        assert!(!AppDiscovery::is_valid_bundle(&PathBuf::from("/nope.gpuapp")));
    }

    #[test]
    fn test_list_names() {
        let dir = TempDir::new().unwrap();
        create_test_bundle(dir.path(), "alpha");
        create_test_bundle(dir.path(), "beta");

        let discovery = AppDiscovery::with_paths(vec![dir.path().to_path_buf()]);
        let names = discovery.list_names();

        assert_eq!(names, vec!["alpha", "beta"]); // Sorted
    }

    #[test]
    fn test_scan_sorts_by_name() {
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
    fn test_first_match_wins() {
        // If same app name exists in multiple paths, first path wins
        let dir1 = TempDir::new().unwrap();
        let dir2 = TempDir::new().unwrap();

        create_test_bundle(dir1.path(), "shared");
        create_test_bundle(dir2.path(), "shared");

        let discovery = AppDiscovery::with_paths(vec![
            dir1.path().to_path_buf(),
            dir2.path().to_path_buf(),
        ]);

        let found = discovery.find_by_name("shared").unwrap();
        // Should be from dir1 (first in search path)
        assert!(found.starts_with(dir1.path()));
    }
}
