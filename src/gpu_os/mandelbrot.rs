// Mandelbrot Fractal Viewer - GPU-Native App Implementation
//
// Uses a fullscreen quad with fragment shader computing the Mandelbrot set.
// This is more efficient than per-pixel compute as the fragment shader
// runs in parallel across all pixels with proper GPU utilization.

use super::app::{GpuApp, AppBuilder, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

// We use 6 vertices for a fullscreen quad (2 triangles)
pub const VERTICES_PER_QUAD: usize = 6;

// Default parameters - Start at Seahorse Valley (one of the most beautiful areas)
pub const DEFAULT_CENTER_X: f64 = -0.745;
pub const DEFAULT_CENTER_Y: f64 = 0.186;
pub const DEFAULT_ZOOM: f64 = 1.5;
pub const DEFAULT_MAX_ITERATIONS: u32 = 256;

// Zoom and pan speeds
pub const ZOOM_FACTOR: f64 = 1.3;
pub const PAN_SPEED: f64 = 0.1;

// Zoom limits - f32 precision breaks down around 10^6
pub const MAX_ZOOM: f64 = 1e6;
pub const MIN_ZOOM: f64 = 0.1;

// ============================================================================
// Interesting Locations in the Mandelbrot Set
// ============================================================================

/// Famous locations in the Mandelbrot set that are visually interesting
pub struct MandelbrotLocation {
    pub name: &'static str,
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    pub iterations: u32,
}

pub const PRESET_LOCATIONS: &[MandelbrotLocation] = &[
    // 1: Seahorse Valley - beautiful spiraling patterns
    MandelbrotLocation {
        name: "Seahorse Valley",
        center_x: -0.745,
        center_y: 0.186,
        zoom: 1.5,
        iterations: 256,
    },
    // 2: Double Spiral - intricate double spiral formation
    MandelbrotLocation {
        name: "Double Spiral",
        center_x: -0.7435669,
        center_y: 0.1314023,
        zoom: 200.0,
        iterations: 512,
    },
    // 3: Lightning - electric branching patterns
    MandelbrotLocation {
        name: "Lightning",
        center_x: -0.170337,
        center_y: -1.06506,
        zoom: 50.0,
        iterations: 256,
    },
    // 4: Mini Mandelbrot - a tiny copy of the whole set
    MandelbrotLocation {
        name: "Mini Mandelbrot",
        center_x: -1.7490863748,
        center_y: 0.0,
        zoom: 1000.0,
        iterations: 512,
    },
    // 5: Elephant Valley - elephant trunk shapes
    MandelbrotLocation {
        name: "Elephant Valley",
        center_x: 0.275,
        center_y: 0.0,
        zoom: 2.0,
        iterations: 256,
    },
    // 6: Starfish - star-like patterns
    MandelbrotLocation {
        name: "Starfish",
        center_x: -0.374004139,
        center_y: 0.659792175,
        zoom: 500.0,
        iterations: 512,
    },
    // 7: Tendril - delicate spiral arms
    MandelbrotLocation {
        name: "Tendril",
        center_x: -0.8,
        center_y: 0.156,
        zoom: 50.0,
        iterations: 256,
    },
    // 8: Julia Island - another mini Mandelbrot
    MandelbrotLocation {
        name: "Julia Island",
        center_x: -0.235125,
        center_y: 0.827215,
        zoom: 200.0,
        iterations: 512,
    },
    // 9: Quad Spiral - four-armed spiral
    MandelbrotLocation {
        name: "Quad Spiral",
        center_x: 0.281717921930775,
        center_y: 0.5771052841488505,
        zoom: 1000.0,
        iterations: 1024,
    },
    // 0: Overview - full set view
    MandelbrotLocation {
        name: "Full View",
        center_x: -0.5,
        center_y: 0.0,
        zoom: 0.35,
        iterations: 128,
    },
];

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Mandelbrot parameters - passed to shader each frame
/// Using f32 for GPU compatibility, but keeping f64 on CPU for precision
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MandelbrotParams {
    pub center_x: f32,
    pub center_y: f32,
    pub zoom: f32,
    pub max_iterations: u32,
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub time: f32,
    pub _padding: u32,
}

impl Default for MandelbrotParams {
    fn default() -> Self {
        Self {
            center_x: DEFAULT_CENTER_X as f32,
            center_y: DEFAULT_CENTER_Y as f32,
            zoom: DEFAULT_ZOOM as f32,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            viewport_width: 800.0,
            viewport_height: 800.0,
            time: 0.0,
            _padding: 0,
        }
    }
}

/// Fullscreen quad vertex
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct QuadVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

// ============================================================================
// Shader Source
// ============================================================================

fn shader_source() -> String {
    format!(r#"
{header}

// ============================================================================
// App-Specific Structures
// ============================================================================

struct MandelbrotParams {{
    float center_x;
    float center_y;
    float zoom;
    uint max_iterations;
    float viewport_width;
    float viewport_height;
    float time;
    uint _padding;
}};

struct QuadVertex {{
    float2 position;
    float2 uv;
}};

// ============================================================================
// Compute Kernel - Generates fullscreen quad
// ============================================================================

// Fullscreen quad data - constant memory at file scope
constant float2 quad_positions[6] = {{
    float2(-1.0, -1.0),  // Bottom-left
    float2( 1.0, -1.0),  // Bottom-right
    float2( 1.0,  1.0),  // Top-right
    float2(-1.0, -1.0),  // Bottom-left
    float2( 1.0,  1.0),  // Top-right
    float2(-1.0,  1.0),  // Top-left
}};

constant float2 quad_uvs[6] = {{
    float2(0.0, 1.0),
    float2(1.0, 1.0),
    float2(1.0, 0.0),
    float2(0.0, 1.0),
    float2(1.0, 0.0),
    float2(0.0, 0.0),
}};

kernel void mandelbrot_setup(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant MandelbrotParams& params [[buffer(2)]],
    device QuadVertex* vertices [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    // Only threads 0-5 generate the fullscreen quad
    if (tid >= 6) return;

    vertices[tid].position = quad_positions[tid];
    vertices[tid].uv = quad_uvs[tid];
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 uv;
}};

vertex VertexOut mandelbrot_vertex(
    const device QuadVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    QuadVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    return out;
}}

// ============================================================================
// Color Palette Functions
// ============================================================================

// Beautiful color gradient based on iteration count
float3 mandelbrot_color(float t, float time) {{
    // Multiple color bands with smooth transitions
    // Using trigonometric functions for smooth gradients

    // Add subtle time-based animation to make it feel alive
    float phase = time * 0.1;

    // Create a rich, vibrant palette
    float3 a = float3(0.5, 0.5, 0.5);
    float3 b = float3(0.5, 0.5, 0.5);
    float3 c = float3(1.0, 1.0, 1.0);
    float3 d = float3(0.00, 0.33, 0.67);

    // Offset d slightly with time for subtle color shifting
    d += float3(phase * 0.1, phase * 0.05, phase * 0.02);

    return a + b * cos(6.28318 * (c * t + d));
}}

// Alternative: Fire palette
float3 fire_palette(float t) {{
    return float3(
        min(1.0, t * 3.0),
        max(0.0, min(1.0, t * 3.0 - 1.0)),
        max(0.0, t * 3.0 - 2.0)
    );
}}

// Alternative: Ocean palette
float3 ocean_palette(float t) {{
    float3 deep = float3(0.0, 0.05, 0.2);
    float3 mid = float3(0.0, 0.3, 0.6);
    float3 shallow = float3(0.3, 0.8, 0.9);
    float3 foam = float3(1.0, 1.0, 1.0);

    if (t < 0.33) return mix(deep, mid, t * 3.0);
    if (t < 0.66) return mix(mid, shallow, (t - 0.33) * 3.0);
    return mix(shallow, foam, (t - 0.66) * 3.0);
}}

// ============================================================================
// Fragment Shader - Mandelbrot computation per pixel
// ============================================================================

fragment float4 mandelbrot_fragment(
    VertexOut in [[stage_in]],
    constant MandelbrotParams& params [[buffer(0)]]
) {{
    // Convert UV to complex plane coordinates
    float aspect = params.viewport_width / params.viewport_height;

    // Map UV [0,1] to centered coordinates [-1,1] with aspect correction
    float2 uv = in.uv * 2.0 - 1.0;
    uv.x *= aspect;

    // Apply zoom and center
    float2 c = float2(
        uv.x / params.zoom + params.center_x,
        uv.y / params.zoom + params.center_y
    );

    // Mandelbrot iteration: z = z^2 + c, starting with z = 0
    float2 z = float2(0.0, 0.0);
    uint iter = 0;

    // Use escape radius of 4 (squared = 16 for efficiency)
    // But use larger radius for smooth coloring
    float escape_radius_sq = 256.0;

    for (uint i = 0; i < params.max_iterations; i++) {{
        // z^2 = (a + bi)^2 = a^2 - b^2 + 2abi
        float z_real_sq = z.x * z.x;
        float z_imag_sq = z.y * z.y;

        // Check escape condition
        if (z_real_sq + z_imag_sq > escape_radius_sq) {{
            break;
        }}

        // z = z^2 + c
        z = float2(
            z_real_sq - z_imag_sq + c.x,
            2.0 * z.x * z.y + c.y
        );
        iter++;
    }}

    // Color the pixel
    if (iter == params.max_iterations) {{
        // Inside the set - black with subtle gradient
        return float4(0.0, 0.0, 0.0, 1.0);
    }}

    // Smooth coloring using escape-time algorithm with continuous potential
    // This avoids banding artifacts
    float log_zn = log(z.x * z.x + z.y * z.y) / 2.0;
    float nu = log(log_zn / log(2.0)) / log(2.0);
    float smooth_iter = float(iter) + 1.0 - nu;

    // Normalize to [0, 1] range for color lookup
    float t = smooth_iter / float(params.max_iterations);

    // Apply logarithmic scaling for better color distribution
    t = log(1.0 + t * 20.0) / log(21.0);

    // Get color from palette
    float3 color = mandelbrot_color(t, params.time);

    // Add subtle glow effect near the set boundary
    float glow = exp(-smooth_iter * 0.05);
    color += float3(0.1, 0.05, 0.15) * glow;

    return float4(color, 1.0);
}}
"#, header = APP_SHADER_HEADER)
}

// ============================================================================
// MandelbrotViewer App
// ============================================================================

pub struct MandelbrotViewer {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // App-specific buffers
    params_buffer: Buffer,
    vertices_buffer: Buffer,

    // High-precision state (CPU side)
    center_x: f64,
    center_y: f64,
    zoom: f64,
    max_iterations: u32,
    viewport_width: f32,
    viewport_height: f32,
    time: f32,

    // Mouse state for drag panning
    drag_start: Option<(f32, f32)>,
    drag_center_start: (f64, f64),
    last_mouse_pos: (f32, f32),
    mouse_down: bool,
}

impl MandelbrotViewer {
    /// Create a new Mandelbrot viewer app
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "MandelbrotViewer");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create pipelines
        let compute_pipeline = builder.create_compute_pipeline(&library, "mandelbrot_setup")?;

        // Create custom render pipeline with params buffer in fragment shader
        let render_pipeline = Self::create_render_pipeline(device, &library)?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<MandelbrotParams>());
        let vertices_buffer = builder.create_buffer(VERTICES_PER_QUAD * mem::size_of::<QuadVertex>());

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            vertices_buffer,
            center_x: DEFAULT_CENTER_X,
            center_y: DEFAULT_CENTER_Y,
            zoom: DEFAULT_ZOOM,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            viewport_width: 800.0,
            viewport_height: 800.0,
            time: 0.0,
            drag_start: None,
            drag_center_start: (DEFAULT_CENTER_X, DEFAULT_CENTER_Y),
            last_mouse_pos: (0.5, 0.5),
            mouse_down: false,
        })
    }

    /// Create render pipeline with fragment shader params binding
    fn create_render_pipeline(device: &Device, library: &Library) -> Result<RenderPipelineState, String> {
        let vertex_function = library
            .get_function("mandelbrot_vertex", None)
            .map_err(|e| format!("Failed to get mandelbrot_vertex: {}", e))?;

        let fragment_function = library
            .get_function("mandelbrot_fragment", None)
            .map_err(|e| format!("Failed to get mandelbrot_fragment: {}", e))?;

        let desc = RenderPipelineDescriptor::new();
        desc.set_vertex_function(Some(&vertex_function));
        desc.set_fragment_function(Some(&fragment_function));

        let attachment = desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // No blending needed for fullscreen fractal

        device
            .new_render_pipeline_state(&desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))
    }

    /// Zoom in by a factor
    pub fn zoom_in(&mut self) {
        let new_zoom = self.zoom * ZOOM_FACTOR;
        if new_zoom <= MAX_ZOOM {
            self.zoom = new_zoom;
            self.print_status();
        } else {
            println!("MAX ZOOM reached (f32 precision limit)");
        }
    }

    /// Zoom out by a factor
    pub fn zoom_out(&mut self) {
        let new_zoom = self.zoom / ZOOM_FACTOR;
        if new_zoom >= MIN_ZOOM {
            self.zoom = new_zoom;
            self.print_status();
        } else {
            println!("MIN ZOOM reached");
        }
    }

    /// Zoom at a specific point (for mouse scroll)
    pub fn zoom_at(&mut self, screen_x: f32, screen_y: f32, zoom_in: bool) {
        let aspect = self.viewport_width / self.viewport_height;

        // Convert screen position to complex plane position
        let uv_x = (screen_x * 2.0 - 1.0) * aspect as f32;
        let uv_y = screen_y * 2.0 - 1.0;

        let target_x = (uv_x as f64) / self.zoom + self.center_x;
        let target_y = (uv_y as f64) / self.zoom + self.center_y;

        // Apply zoom with limits
        let factor = if zoom_in { ZOOM_FACTOR } else { 1.0 / ZOOM_FACTOR };
        let new_zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);

        if (zoom_in && new_zoom >= MAX_ZOOM) || (!zoom_in && new_zoom <= MIN_ZOOM) {
            if zoom_in {
                println!("MAX ZOOM reached (f32 precision limit)");
            }
            return;
        }

        // Adjust center to keep target point in same screen position
        self.center_x = target_x - (uv_x as f64) / new_zoom;
        self.center_y = target_y - (uv_y as f64) / new_zoom;
        self.zoom = new_zoom;

        self.print_status();
    }

    /// Pan in a direction (normalized)
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let scale = PAN_SPEED / self.zoom;
        self.center_x += dx * scale;
        self.center_y += dy * scale;
        self.print_status();
    }

    /// Reset to default view
    pub fn reset(&mut self) {
        self.center_x = DEFAULT_CENTER_X;
        self.center_y = DEFAULT_CENTER_Y;
        self.zoom = DEFAULT_ZOOM;
        self.max_iterations = DEFAULT_MAX_ITERATIONS;
        println!("VIEW RESET");
        self.print_status();
    }

    /// Increase max iterations
    pub fn increase_iterations(&mut self) {
        self.max_iterations = (self.max_iterations + 64).min(2048);
        println!("Max iterations: {}", self.max_iterations);
    }

    /// Decrease max iterations
    pub fn decrease_iterations(&mut self) {
        self.max_iterations = (self.max_iterations.saturating_sub(64)).max(64);
        println!("Max iterations: {}", self.max_iterations);
    }

    /// Set viewport size
    pub fn set_viewport(&mut self, width: f32, height: f32) {
        self.viewport_width = width;
        self.viewport_height = height;
    }

    /// Get current zoom level
    pub fn zoom_level(&self) -> f64 {
        self.zoom
    }

    /// Get current center
    pub fn center(&self) -> (f64, f64) {
        (self.center_x, self.center_y)
    }

    /// Print current status to console
    fn print_status(&self) {
        println!(
            "Zoom: {:.2e}  Center: ({:.10}, {:.10})  Iterations: {}",
            self.zoom, self.center_x, self.center_y, self.max_iterations
        );
    }

    /// Get the params buffer for fragment shader binding
    pub fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    /// Jump to a preset location (0-9)
    pub fn goto_preset(&mut self, index: usize) {
        if index < PRESET_LOCATIONS.len() {
            let loc = &PRESET_LOCATIONS[index];
            self.center_x = loc.center_x;
            self.center_y = loc.center_y;
            self.zoom = loc.zoom;
            self.max_iterations = loc.iterations;
            println!("GOTO: {} (press {} again to explore)", loc.name, index);
            self.print_status();
        }
    }
}

impl GpuApp for MandelbrotViewer {
    fn name(&self) -> &str {
        "Mandelbrot Viewer"
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
        VERTICES_PER_QUAD
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![&self.vertices_buffer]  // slot 3 - vertices for compute to write
    }

    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32) {
        self.time += delta_time;

        // Handle ongoing drag
        if self.mouse_down {
            if let Some((start_x, start_y)) = self.drag_start {
                let dx = frame_state.cursor_x - start_x;
                let dy = frame_state.cursor_y - start_y;

                // Convert screen delta to complex plane delta
                let aspect = self.viewport_width / self.viewport_height;
                let scale = 2.0 / self.zoom;  // 2.0 because UV goes from -1 to 1

                self.center_x = self.drag_center_start.0 - (dx as f64) * scale * (aspect as f64);
                self.center_y = self.drag_center_start.1 - (dy as f64) * scale;
            }
        }

        // Build params
        let params = MandelbrotParams {
            center_x: self.center_x as f32,
            center_y: self.center_y as f32,
            zoom: self.zoom as f32,
            max_iterations: self.max_iterations,
            viewport_width: self.viewport_width,
            viewport_height: self.viewport_height,
            time: self.time,
            _padding: 0,
        };

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut MandelbrotParams;
            *ptr = params;
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            t if t == InputEventType::MouseDown as u16 => {
                self.mouse_down = true;
                self.drag_start = Some((event.position[0], event.position[1]));
                self.drag_center_start = (self.center_x, self.center_y);
            }
            t if t == InputEventType::MouseUp as u16 => {
                self.mouse_down = false;
                self.drag_start = None;
            }
            t if t == InputEventType::MouseMove as u16 => {
                self.last_mouse_pos = (event.position[0], event.position[1]);
            }
            t if t == InputEventType::MouseScroll as u16 => {
                // delta[1] is the scroll amount (positive = scroll up = zoom in)
                let zoom_in = event.delta[1] > 0.0;
                self.zoom_at(event.position[0], event.position[1], zoom_in);
            }
            _ => {}
        }
    }

    fn post_frame(&mut self, _timing: &FrameTiming) {
        // Nothing needed
    }

    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(0.0, 0.0, 0.0, 1.0)
    }

    fn thread_count(&self) -> usize {
        // Only need a few threads to generate the fullscreen quad
        64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // Ensure proper alignment for GPU
        assert_eq!(mem::size_of::<MandelbrotParams>(), 32);
        assert_eq!(mem::size_of::<QuadVertex>(), 16);
    }

    #[test]
    fn test_default_params() {
        let params = MandelbrotParams::default();
        assert_eq!(params.center_x, DEFAULT_CENTER_X as f32);
        assert_eq!(params.center_y, DEFAULT_CENTER_Y as f32);
        assert_eq!(params.zoom, DEFAULT_ZOOM as f32);
        assert_eq!(params.max_iterations, DEFAULT_MAX_ITERATIONS);
    }
}
