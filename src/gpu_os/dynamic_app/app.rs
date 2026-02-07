//! DynamicGpuApp Implementation (Issue #143)
//!
//! A generic GpuApp implementation that runs shaders loaded from .gpuapp bundles.
//!
//! # Example
//! ```ignore
//! let gpu_app = DynamicGpuApp::load(&bundle_path, &device)?;
//! runtime.run_frame(&mut gpu_app, &drawable);
//! ```

use metal::*;
use std::path::Path;

use crate::gpu_os::app::{GpuApp, AppBuilder, PipelineMode, APP_SHADER_HEADER};
use crate::gpu_os::memory::{FrameState, InputEvent};
use crate::gpu_os::vsync::FrameTiming;
use super::manifest::{AppManifest, VertexCount};

/// Maximum vertices a dynamic app can produce
const MAX_VERTICES: usize = 65536;

/// Size of vertex data in bytes (position float4 + color float4)
const VERTEX_STRIDE: usize = 32;

/// A GPU app loaded dynamically from a .gpuapp bundle
pub struct DynamicGpuApp {
    // Identity
    name: String,
    manifest: AppManifest,

    // Pipelines (created by AppBuilder)
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    vertices_buffer: Buffer,
    params_buffer: Buffer,
    app_buffers: Vec<Buffer>,

    // Configuration from manifest
    thread_count: usize,
    vertex_count: usize,
    clear_color: MTLClearColor,

    // Dynamic vertex count support
    dynamic_vertex_count: bool,
    vertex_count_offset: usize,
}

impl DynamicGpuApp {
    /// Load a GPU app from a .gpuapp bundle directory
    ///
    /// # Arguments
    /// * `bundle_path` - Path to the .gpuapp directory
    /// * `device` - Metal device to create resources on
    ///
    /// # Returns
    /// A DynamicGpuApp ready to run, or an error string
    pub fn load(bundle_path: &Path, device: &Device) -> Result<Self, String> {
        // 1. Load and parse manifest
        let manifest = AppManifest::load(bundle_path)?;

        // 2. Load shader source
        let shader_path = manifest.shader_path(bundle_path);
        let shader_source = std::fs::read_to_string(&shader_path)
            .map_err(|e| format!("Failed to read shader at {}: {}", shader_path.display(), e))?;

        // 3. Prepend APP_SHADER_HEADER for OS types and helpers
        let full_source = format!("{}\n\n// === App Shader ===\n{}", APP_SHADER_HEADER, shader_source);

        // 4. Compile using AppBuilder
        let builder = AppBuilder::new(device, &manifest.app.name);
        let library = builder.compile_library(&full_source)?;

        let compute_pipeline = builder.create_compute_pipeline(
            &library,
            &manifest.shaders.compute,
        )?;

        let render_pipeline = builder.create_render_pipeline(
            &library,
            &manifest.shaders.vertex,
            &manifest.shaders.fragment,
        )?;

        // 5. Create vertex buffer (large enough for max vertices)
        let vertices_buffer = device.new_buffer(
            (MAX_VERTICES * VERTEX_STRIDE) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // 6. Create params buffer
        let params_size = manifest.params_config()
            .map(|c| c.size)
            .unwrap_or(256);
        let params_buffer = device.new_buffer(
            params_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // 7. Create app buffers (slots 3+) based on manifest
        let buffer_configs = manifest.app_buffer_configs();
        let mut app_buffers = Vec::with_capacity(buffer_configs.len());

        for (_name, config) in buffer_configs {
            let buf = device.new_buffer(
                config.size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            app_buffers.push(buf);
        }

        // 8. Extract configuration
        let clear = manifest.config.clear_color;
        let clear_color = MTLClearColor::new(
            clear[0] as f64,
            clear[1] as f64,
            clear[2] as f64,
            clear[3] as f64,
        );

        let (vertex_count, dynamic_vertex_count, vertex_count_offset) = match &manifest.config.vertex_count {
            VertexCount::Static(n) => (*n, false, 0),
            VertexCount::Dynamic { offset, .. } => (0, true, *offset),
        };

        // Extract values before moving manifest
        let name = manifest.app.name.clone();
        let thread_count = manifest.config.thread_count;

        Ok(Self {
            name,
            manifest,
            compute_pipeline,
            render_pipeline,
            vertices_buffer,
            params_buffer,
            app_buffers,
            thread_count,
            vertex_count,
            clear_color,
            dynamic_vertex_count,
            vertex_count_offset,
        })
    }

    /// Get the manifest for this app
    pub fn manifest(&self) -> &AppManifest {
        &self.manifest
    }

    /// Get preferred window size from manifest
    pub fn preferred_size(&self) -> (f32, f32) {
        self.manifest.preferred_size()
    }
}

impl GpuApp for DynamicGpuApp {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_pipeline(&self) -> &ComputePipelineState {
        &self.compute_pipeline
    }

    fn render_pipeline(&self) -> &RenderPipelineState {
        &self.render_pipeline
    }

    fn vertices_buffer(&self) -> &Buffer {
        &self.vertices_buffer
    }

    fn vertex_count(&self) -> usize {
        if self.dynamic_vertex_count {
            // Read vertex count from params buffer at specified offset
            unsafe {
                let ptr = self.params_buffer.contents() as *const u8;
                let count_ptr = ptr.add(self.vertex_count_offset) as *const u32;
                (*count_ptr) as usize
            }
        } else {
            self.vertex_count
        }
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        // Slot 3 is always vertices_buffer (for compute to write, render to read)
        // Slots 4+ are app-specific buffers from manifest
        let mut buffers = vec![&self.vertices_buffer];
        buffers.extend(self.app_buffers.iter());
        buffers
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn thread_count(&self) -> usize {
        self.thread_count
    }

    fn clear_color(&self) -> MTLClearColor {
        self.clear_color
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::LowLatency
    }

    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32) {
        // Write standard frame info to params buffer
        // Apps expect this layout at the start of params:
        // offset 0: delta_time (f32)
        // offset 4: time (f32)
        // offset 8: cursor_x (f32)
        // offset 12: cursor_y (f32)
        // offset 16: frame_number (u32)
        // offset 20: mouse_buttons (u32)
        // offset 24+: app-specific data
        unsafe {
            let ptr = self.params_buffer.contents() as *mut f32;
            *ptr.add(0) = delta_time;
            *ptr.add(1) = frame_state.time;
            *ptr.add(2) = frame_state.cursor_x;
            *ptr.add(3) = frame_state.cursor_y;
            *(ptr.add(4) as *mut u32) = frame_state.frame_number;
            *(ptr.add(5) as *mut u32) = frame_state.modifiers;
        }
    }

    fn handle_input(&mut self, _event: &InputEvent) {
        // Dynamic apps receive input via InputQueue buffer (slot 1)
        // The GPU kernel processes events directly
    }

    fn post_frame(&mut self, _timing: &FrameTiming) {
        // Could add profiling here in the future
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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
        let device = Device::system_default().expect("No Metal device");
        let dir = TempDir::new().unwrap();
        let bundle = create_minimal_bundle(dir.path());

        let app = DynamicGpuApp::load(&bundle, &device).unwrap();

        assert_eq!(app.name(), "Minimal Test");
        assert_eq!(app.thread_count(), 64);
        assert_eq!(app.vertex_count(), 6);
    }

    #[test]
    fn test_pipelines_created() {
        let device = Device::system_default().expect("No Metal device");
        let dir = TempDir::new().unwrap();
        let bundle = create_minimal_bundle(dir.path());

        let app = DynamicGpuApp::load(&bundle, &device).unwrap();

        // These should not panic - pipelines exist
        let _ = app.compute_pipeline();
        let _ = app.render_pipeline();
        let _ = app.vertices_buffer();
        let _ = app.params_buffer();
    }

    #[test]
    fn test_app_buffers_from_manifest() {
        let device = Device::system_default().expect("No Metal device");
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

[config]

[buffers]
params = { size = 128, slot = 2 }
data_a = { size = 1024, slot = 3 }
data_b = { size = 2048, slot = 4 }
"#,
        )
        .unwrap();

        // Minimal valid shader
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

        // Should have 3 app buffers: vertices (slot 3) + data_a (slot 4) + data_b (slot 5)
        assert_eq!(app.app_buffers().len(), 3);
    }

    #[test]
    fn test_invalid_shader_error() {
        let device = Device::system_default().expect("No Metal device");
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

        fs::write(bundle.join("main.metal"), "// empty shader - no functions").unwrap();

        let result = DynamicGpuApp::load(&bundle, &device);
        match result {
            Ok(_) => panic!("Expected error for missing function"),
            Err(e) => assert!(e.contains("missing_function"), "Error should mention missing_function: {}", e),
        }
    }

    #[test]
    fn test_clear_color_from_manifest() {
        let device = Device::system_default().expect("No Metal device");
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
        assert!((color.blue - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_preferred_size() {
        let device = Device::system_default().expect("No Metal device");
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
    fn test_missing_bundle() {
        let device = Device::system_default().expect("No Metal device");
        let result = DynamicGpuApp::load(&std::path::PathBuf::from("/nonexistent.gpuapp"), &device);
        assert!(result.is_err());
    }
}
