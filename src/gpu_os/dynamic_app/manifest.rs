//! App Manifest Parsing (Issue #141)
//!
//! Parses TOML manifests that describe GPU app configuration, shader entry points,
//! and buffer requirements.
//!
//! # Bundle Format
//! ```text
//! app_name.gpuapp/
//!   manifest.toml
//!   main.metal
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Complete app manifest parsed from manifest.toml
#[derive(Debug, Clone, Deserialize)]
pub struct AppManifest {
    pub app: AppInfo,
    pub shaders: ShaderConfig,
    #[serde(default)]
    pub config: AppConfig,
    #[serde(default)]
    pub buffers: HashMap<String, BufferConfig>,
}

/// Basic app information
#[derive(Debug, Clone, Deserialize)]
pub struct AppInfo {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1.0".to_string()
}

/// Shader function names
#[derive(Debug, Clone, Deserialize)]
pub struct ShaderConfig {
    pub compute: String,
    pub vertex: String,
    pub fragment: String,
}

/// App runtime configuration
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_thread_count")]
    pub thread_count: usize,
    #[serde(default = "default_vertex_count")]
    pub vertex_count: VertexCount,
    #[serde(default = "default_clear_color")]
    pub clear_color: [f32; 4],
    #[serde(default)]
    pub preferred_size: Option<[f32; 2]>,
}

fn default_thread_count() -> usize {
    1024
}

fn default_vertex_count() -> VertexCount {
    VertexCount::Static(6)
}

fn default_clear_color() -> [f32; 4] {
    [0.1, 0.1, 0.15, 1.0]
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            thread_count: default_thread_count(),
            vertex_count: default_vertex_count(),
            clear_color: default_clear_color(),
            preferred_size: None,
        }
    }
}

/// Vertex count can be static or read from buffer at runtime
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum VertexCount {
    Static(usize),
    Dynamic { dynamic: bool, offset: usize },
}

impl Default for VertexCount {
    fn default() -> Self {
        VertexCount::Static(6)
    }
}

impl VertexCount {
    pub fn static_count(&self) -> Option<usize> {
        match self {
            VertexCount::Static(n) => Some(*n),
            VertexCount::Dynamic { .. } => None,
        }
    }

    pub fn is_dynamic(&self) -> bool {
        matches!(self, VertexCount::Dynamic { .. })
    }
}

/// Buffer configuration
#[derive(Debug, Clone, Deserialize)]
pub struct BufferConfig {
    pub size: usize,
    pub slot: u32,
}

impl AppManifest {
    /// Load manifest from a .gpuapp bundle directory
    pub fn load(bundle_path: &Path) -> Result<Self, String> {
        let manifest_path = bundle_path.join("manifest.toml");
        let content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read manifest at {}: {}", manifest_path.display(), e))?;

        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse manifest: {}", e))
    }

    /// Get path to the main shader file
    pub fn shader_path(&self, bundle_path: &Path) -> PathBuf {
        bundle_path.join("main.metal")
    }

    /// Get preferred window size or default
    pub fn preferred_size(&self) -> (f32, f32) {
        self.config.preferred_size
            .map(|[w, h]| (w, h))
            .unwrap_or((800.0, 600.0))
    }

    /// Get app buffers sorted by slot (excluding params at slot 2)
    pub fn app_buffer_configs(&self) -> Vec<(&String, &BufferConfig)> {
        let mut configs: Vec<_> = self.buffers.iter()
            .filter(|(name, _)| *name != "params")
            .collect();
        configs.sort_by_key(|(_, cfg)| cfg.slot);
        configs
    }

    /// Get params buffer config if specified
    pub fn params_config(&self) -> Option<&BufferConfig> {
        self.buffers.get("params")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(manifest.preferred_size(), (640.0, 480.0));
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
        assert_eq!(configs[0].1.slot, 3); // a_first
        assert_eq!(configs[1].1.slot, 4); // m_middle
        assert_eq!(configs[2].1.slot, 5); // z_last
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
}
