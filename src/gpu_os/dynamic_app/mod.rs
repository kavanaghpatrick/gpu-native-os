//! Dynamic App Loading (Issues #141-#146)
//!
//! Load and run GPU apps from .gpuapp bundles at runtime.
//!
//! # Bundle Format
//! ```text
//! app_name.gpuapp/
//!   manifest.toml    # App configuration
//!   main.metal       # Shader source
//! ```
//!
//! # Example
//! ```ignore
//! use rust_experiment::gpu_os::dynamic_app::{AppDiscovery, DynamicGpuApp};
//!
//! let discovery = AppDiscovery::new();
//! for app in discovery.scan() {
//!     println!("Found: {}", app.name);
//! }
//!
//! if let Some(path) = discovery.find_by_name("my_app") {
//!     let gpu_app = DynamicGpuApp::load(&path, &device)?;
//!     // Run with GpuRuntime...
//! }
//! ```

pub mod manifest;
pub mod discovery;
pub mod app;
// ARCHIVED: desktop_wrapper - Used old CPU-centric desktop, needs rewrite for megakernel

pub use manifest::{AppManifest, AppInfo, ShaderConfig, AppConfig, BufferConfig, VertexCount};
pub use discovery::{AppDiscovery, DiscoveredApp};
pub use app::DynamicGpuApp;
