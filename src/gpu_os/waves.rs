// Wave Simulation - GPU-Native App Implementation
//
// 256x256 grid of height values simulating 2D wave propagation.
// Uses the wave equation: new_h = 2*h - prev_h + c*(neighbors - 4*h)
// with damping for natural decay.
//
// Click to create ripples!

use super::app::{GpuApp, AppBuilder, PipelineMode, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

pub const GRID_SIZE: usize = 256;
pub const CELL_COUNT: usize = GRID_SIZE * GRID_SIZE;
pub const VERTICES_PER_CELL: usize = 6;
pub const TOTAL_VERTICES: usize = CELL_COUNT * VERTICES_PER_CELL;

// Thread configuration - use 256 threads per group, multiple dispatches
pub const THREADS_PER_GROUP: usize = 256;
pub const THREAD_GROUPS: usize = (CELL_COUNT + THREADS_PER_GROUP - 1) / THREADS_PER_GROUP;

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Wave simulation parameters (slot 2) - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct WaveParams {
    pub delta_time: f32,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_down: u32,
    pub damping: f32,
    pub wave_speed: f32,
    pub frame_number: u32,
    pub ripple_strength: f32,
}

impl Default for WaveParams {
    fn default() -> Self {
        Self {
            delta_time: 1.0 / 120.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_down: 0,
            damping: 0.995,
            wave_speed: 0.25,
            frame_number: 0,
            ripple_strength: 0.5,
        }
    }
}

/// Cell vertex - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct WaveVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
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

struct WaveParams {{
    float delta_time;
    float mouse_x;
    float mouse_y;
    uint mouse_down;
    float damping;
    float wave_speed;
    uint frame_number;
    float ripple_strength;
}};

struct WaveVertex {{
    float2 position;
    float2 uv;
    float4 color;
}};

constant uint GRID_SIZE = 256;
constant uint CELL_COUNT = GRID_SIZE * GRID_SIZE;

// ============================================================================
// Wave Compute Kernel
// ============================================================================

kernel void wave_compute(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant WaveParams& params [[buffer(2)]],
    device float* height_a [[buffer(3)]],      // Buffer A
    device float* height_b [[buffer(4)]],      // Buffer B
    device WaveVertex* vertices [[buffer(5)]], // Output vertices
    device atomic_uint* vertex_count [[buffer(6)]], // Vertex count
    uint tid [[thread_position_in_grid]]
) {{
    // Use frame parity to alternate buffers (no separate swap pass needed)
    bool even_frame = (params.frame_number % 2) == 0;
    device float* current_buf = even_frame ? height_a : height_b;
    device float* prev_buf = even_frame ? height_b : height_a;
    device float* write_buf = prev_buf;  // Write to the "previous" buffer, it becomes current next frame

    // Each thread processes multiple cells (64 cells per thread with 1024 threads)
    for (uint cell_idx = tid; cell_idx < CELL_COUNT; cell_idx += 1024) {{

    // Cell coordinates
    uint cell_x = cell_idx % GRID_SIZE;
    uint cell_y = cell_idx / GRID_SIZE;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: HANDLE MOUSE INPUT (create ripples)
    // ═══════════════════════════════════════════════════════════════════

    // Calculate cell bounds (grid fills 0.0-1.0 of screen)
    float cell_size = 1.0 / float(GRID_SIZE);
    float cell_center_x = (float(cell_x) + 0.5) * cell_size;
    float cell_center_y = (float(cell_y) + 0.5) * cell_size;

    // Distance from mouse to cell center
    float dx = params.mouse_x - cell_center_x;
    float dy = params.mouse_y - cell_center_y;
    float dist = sqrt(dx * dx + dy * dy);

    // Get current and previous heights using frame parity
    float h_current = current_buf[cell_idx];
    float h_prev = prev_buf[cell_idx];

    // Add ripple on mouse down
    float ripple_radius = 0.05;
    if (params.mouse_down != 0 && dist < ripple_radius) {{
        // Gaussian-like ripple
        float strength = exp(-dist * dist / (ripple_radius * ripple_radius * 0.3));

        // Left click = positive ripple, right = negative
        float sign = (params.mouse_down == 1) ? 1.0 : -1.0;
        h_current += sign * strength * params.ripple_strength;
    }}

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: WAVE EQUATION
    // ═══════════════════════════════════════════════════════════════════

    // Get neighbor indices with wrapping
    uint left = (cell_x == 0) ? cell_idx + GRID_SIZE - 1 : cell_idx - 1;
    uint right = (cell_x == GRID_SIZE - 1) ? cell_idx - GRID_SIZE + 1 : cell_idx + 1;
    uint up = (cell_y == 0) ? cell_idx + CELL_COUNT - GRID_SIZE : cell_idx - GRID_SIZE;
    uint down = (cell_y == GRID_SIZE - 1) ? cell_idx - CELL_COUNT + GRID_SIZE : cell_idx + GRID_SIZE;

    // Get neighbor heights from current buffer
    float h_left = current_buf[left];
    float h_right = current_buf[right];
    float h_up = current_buf[up];
    float h_down = current_buf[down];

    // Wave equation: new_h = 2*h - prev_h + c*(neighbors - 4*h)
    float laplacian = h_left + h_right + h_up + h_down - 4.0 * h_current;
    float h_new = 2.0 * h_current - h_prev + params.wave_speed * laplacian;

    // Apply damping
    h_new *= params.damping;

    // Clamp to prevent instability
    h_new = clamp(h_new, -2.0f, 2.0f);

    // Write to the buffer that will be "current" next frame
    write_buf[cell_idx] = h_new;
    // Note: We'll swap buffer pointers on CPU side, or use frame parity

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: GENERATE VERTICES
    // ═══════════════════════════════════════════════════════════════════

    // Calculate cell screen position
    float x0 = float(cell_x) * cell_size;
    float y0 = float(cell_y) * cell_size;
    float x1 = x0 + cell_size;
    float y1 = y0 + cell_size;

    // Color based on height - beautiful blue gradient
    // Height range roughly -1 to 1
    float h = h_new;

    // Base color: deep blue ocean
    float3 deep_blue = float3(0.05, 0.15, 0.35);
    float3 light_blue = float3(0.3, 0.6, 0.9);
    float3 white_foam = float3(0.9, 0.95, 1.0);

    // Color interpolation based on height
    float3 color;
    if (h < 0.0) {{
        // Negative: deeper blue (troughs)
        float t = clamp(-h, 0.0f, 1.0f);
        color = mix(light_blue, deep_blue, t);
    }} else {{
        // Positive: lighter blue to white foam (peaks)
        float t = clamp(h, 0.0f, 1.0f);
        color = mix(light_blue, white_foam, t * t);
    }}

    // Add subtle highlight based on height gradient (fake lighting)
    float gradient_x = h_right - h_left;
    float gradient_y = h_down - h_up;
    float highlight = gradient_x * 0.3 + gradient_y * 0.3;
    color += float3(highlight * 0.2, highlight * 0.15, highlight * 0.1);
    color = clamp(color, 0.0f, 1.0f);

    float4 final_color = float4(color, 1.0);

    // Generate 6 vertices for cell quad
    uint base = cell_idx * 6;

    // Triangle 1: TL -> BL -> BR
    vertices[base + 0] = WaveVertex{{float2(x0, y0), float2(0, 0), final_color}};
    vertices[base + 1] = WaveVertex{{float2(x0, y1), float2(0, 1), final_color}};
    vertices[base + 2] = WaveVertex{{float2(x1, y1), float2(1, 1), final_color}};

    // Triangle 2: TL -> BR -> TR
    vertices[base + 3] = WaveVertex{{float2(x0, y0), float2(0, 0), final_color}};
    vertices[base + 4] = WaveVertex{{float2(x1, y1), float2(1, 1), final_color}};
    vertices[base + 5] = WaveVertex{{float2(x1, y0), float2(1, 0), final_color}};

    }} // end for loop over cells
}}

// Second pass: copy new heights back (double-buffer swap)
kernel void wave_swap(
    device float* height_a [[buffer(3)]],
    device float* height_b [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {{
    // Each thread processes multiple cells
    for (uint cell_idx = tid; cell_idx < CELL_COUNT; cell_idx += 1024) {{
        // Swap: copy b back to a for next frame
        height_a[cell_idx] = height_b[cell_idx];
    }}
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 uv;
    float4 color;
}};

vertex VertexOut wave_vertex(
    const device WaveVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    WaveVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position * 2.0 - 1.0, 0.0, 1.0);
    out.position.y = -out.position.y;
    out.uv = v.uv;
    out.color = v.color;
    return out;
}}

// ============================================================================
// Fragment Shader
// ============================================================================

fragment float4 wave_fragment(VertexOut in [[stage_in]]) {{
    return in.color;
}}
"#, header = APP_SHADER_HEADER)
}

// ============================================================================
// WaveSimulation App
// ============================================================================

pub struct WaveSimulation {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    swap_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    params_buffer: Buffer,
    height_a_buffer: Buffer,  // Current heights
    height_b_buffer: Buffer,  // Previous heights
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,

    // Current params
    current_params: WaveParams,

    // Mouse state
    mouse_down: bool,
    last_mouse_x: f32,
    last_mouse_y: f32,
    mouse_button: u32,  // 1 = left (positive), 2 = right (negative)

    // Frame counter for buffer swap
    frame_count: u32,
}

impl WaveSimulation {
    /// Create a new wave simulation
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "WaveSimulation");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create pipelines
        let compute_pipeline = builder.create_compute_pipeline(&library, "wave_compute")?;
        let swap_pipeline = builder.create_compute_pipeline(&library, "wave_swap")?;
        let render_pipeline = builder.create_render_pipeline(&library, "wave_vertex", "wave_fragment")?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<WaveParams>());
        let height_a_buffer = builder.create_buffer(CELL_COUNT * mem::size_of::<f32>());
        let height_b_buffer = builder.create_buffer(CELL_COUNT * mem::size_of::<f32>());
        let vertices_buffer = builder.create_buffer(TOTAL_VERTICES * mem::size_of::<WaveVertex>());
        let vertex_count_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Initialize heights to zero
        unsafe {
            let ptr_a = height_a_buffer.contents() as *mut f32;
            let ptr_b = height_b_buffer.contents() as *mut f32;
            std::ptr::write_bytes(ptr_a, 0, CELL_COUNT);
            std::ptr::write_bytes(ptr_b, 0, CELL_COUNT);
        }

        // Initialize params
        let params = WaveParams::default();
        unsafe {
            let ptr = params_buffer.contents() as *mut WaveParams;
            *ptr = params;
        }

        Ok(Self {
            compute_pipeline,
            swap_pipeline,
            render_pipeline,
            params_buffer,
            height_a_buffer,
            height_b_buffer,
            vertices_buffer,
            vertex_count_buffer,
            current_params: params,
            mouse_down: false,
            last_mouse_x: 0.5,
            last_mouse_y: 0.5,
            mouse_button: 1,
            frame_count: 0,
        })
    }

    /// Get the swap pipeline for double-buffering
    pub fn swap_pipeline(&self) -> &ComputePipelineState {
        &self.swap_pipeline
    }

    /// Adjust damping factor
    pub fn adjust_damping(&mut self, delta: f32) {
        self.current_params.damping = (self.current_params.damping + delta).clamp(0.9, 0.9999);
    }

    /// Adjust wave speed
    pub fn adjust_wave_speed(&mut self, delta: f32) {
        self.current_params.wave_speed = (self.current_params.wave_speed + delta).clamp(0.05, 0.5);
    }

    /// Adjust ripple strength
    pub fn adjust_ripple_strength(&mut self, delta: f32) {
        self.current_params.ripple_strength = (self.current_params.ripple_strength + delta).clamp(0.1, 2.0);
    }

    /// Reset simulation
    pub fn reset(&mut self) {
        unsafe {
            let ptr_a = self.height_a_buffer.contents() as *mut f32;
            let ptr_b = self.height_b_buffer.contents() as *mut f32;
            std::ptr::write_bytes(ptr_a, 0, CELL_COUNT);
            std::ptr::write_bytes(ptr_b, 0, CELL_COUNT);
        }
    }

    /// Get current params for display
    pub fn params(&self) -> &WaveParams {
        &self.current_params
    }
}

impl GpuApp for WaveSimulation {
    fn name(&self) -> &str {
        "Wave Simulation"
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
        TOTAL_VERTICES
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.height_a_buffer,    // slot 3
            &self.height_b_buffer,    // slot 4
            &self.vertices_buffer,    // slot 5
            &self.vertex_count_buffer, // slot 6
        ]
    }

    fn update_params(&mut self, _frame_state: &FrameState, delta_time: f32) {
        self.frame_count += 1;

        // Update params
        self.current_params.delta_time = delta_time;
        self.current_params.mouse_x = self.last_mouse_x;
        self.current_params.mouse_y = self.last_mouse_y;
        self.current_params.mouse_down = if self.mouse_down { self.mouse_button } else { 0 };
        self.current_params.frame_number = self.frame_count;

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut WaveParams;
            *ptr = self.current_params;
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            t if t == InputEventType::MouseMove as u16 => {
                self.last_mouse_x = event.position[0];
                self.last_mouse_y = event.position[1];
            }
            t if t == InputEventType::MouseDown as u16 => {
                self.mouse_down = true;
                // Button 0 = left = positive ripple, Button 1 = right = negative
                self.mouse_button = if event.keycode == 0 { 1 } else { 2 };
            }
            t if t == InputEventType::MouseUp as u16 => {
                self.mouse_down = false;
            }
            _ => {}
        }
    }

    fn thread_count(&self) -> usize {
        1024  // Standard threadgroup size, each thread processes multiple cells
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::HighThroughput  // Simulation benefits from frame overlap
    }

    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(0.02, 0.05, 0.15, 1.0)  // Deep ocean background
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
        assert_eq!(mem::size_of::<WaveParams>(), 32);
        assert_eq!(mem::size_of::<WaveVertex>(), 32);
    }

    #[test]
    fn test_constants() {
        assert_eq!(GRID_SIZE, 256);
        assert_eq!(CELL_COUNT, 65536);
        assert_eq!(TOTAL_VERTICES, 393216);
    }
}
