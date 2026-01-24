// Game of Life - GPU-Native App Implementation
//
// 32x32 grid = 1024 cells = 1024 threads (1:1 mapping)
// Now uses the GpuApp framework for OS integration.

use super::app::{GpuApp, AppBuilder, PipelineMode, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

pub const CELL_COUNT: usize = 1024;
pub const GRID_SIZE: usize = 32;
pub const VERTICES_PER_CELL: usize = 6;
pub const TOTAL_VERTICES: usize = CELL_COUNT * VERTICES_PER_CELL;

// Action codes
pub const ACTION_NONE: u32 = 0;
pub const ACTION_CLEAR: u32 = 1;
pub const ACTION_RANDOM: u32 = 2;
pub const ACTION_GLIDER: u32 = 3;
pub const ACTION_STEP: u32 = 4;

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Cell state - 2 bytes per cell
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CellState {
    pub alive: u8,
    pub age: u8,
}

/// Simulation state - 32 bytes (lives in GPU memory)
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimulationState {
    pub generation: u32,
    pub population: u32,
    pub running: u32,
    pub speed: f32,
    pub accumulator: f32,
    pub cursor_cell: u32,
    pub pending_toggle: u32,
    pub pending_action: u32,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            generation: 0,
            population: 0,
            running: 0,
            speed: 10.0,
            accumulator: 0.0,
            cursor_cell: 0xFFFFFFFF,
            pending_toggle: 0xFFFFFFFF,
            pending_action: ACTION_NONE,
        }
    }
}

/// App parameters passed each frame (slot 2)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AppParams {
    pub delta_time: f32,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub mouse_clicked: u32,
}

/// Cell vertex - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CellVertex {
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

struct CellState {{
    uchar alive;
    uchar age;
}};

struct SimulationState {{
    uint generation;
    uint population;
    uint running;
    float speed;
    float accumulator;
    uint cursor_cell;
    uint pending_toggle;
    uint pending_action;
}};

struct AppParams {{
    float delta_time;
    float cursor_x;
    float cursor_y;
    uint mouse_clicked;
}};

struct CellVertex {{
    float2 position;
    float2 uv;
    float4 color;
}};

struct DrawArguments {{
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
}};

// Action constants
constant uint ACTION_NONE = 0;
constant uint ACTION_CLEAR = 1;
constant uint ACTION_RANDOM = 2;
constant uint ACTION_GLIDER = 3;
constant uint ACTION_STEP = 4;

// ============================================================================
// Main Compute Kernel
// ============================================================================

kernel void game_of_life_kernel(
    constant FrameState& frame [[buffer(0)]],       // OS: frame state
    device InputQueue* input_queue [[buffer(1)]],   // OS: input queue (unused here, we use AppParams)
    constant AppParams& params [[buffer(2)]],       // App: per-frame params
    device CellState* cells [[buffer(3)]],          // App: cell state
    device SimulationState* sim [[buffer(4)]],      // App: simulation state
    device CellVertex* vertices [[buffer(5)]],      // App: output vertices
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {{
    // Thread identity -> cell coordinates
    uint cell_x = tid % 32;
    uint cell_y = tid / 32;

    // Threadgroup shared memory
    threadgroup bool current_alive[1024];
    threadgroup uint simd_population[32];
    threadgroup uint hit_cell;
    threadgroup bool do_step;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: HIT TESTING (cursor → cell)
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {{
        hit_cell = 0xFFFFFFFF;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate cell bounds (grid fills 0.1-0.9 of screen)
    float grid_start = 0.1;
    float grid_size = 0.8;
    float cell_size = grid_size / 32.0;

    float cell_left = grid_start + float(cell_x) * cell_size;
    float cell_top = grid_start + float(cell_y) * cell_size;

    // Use cursor from params (forwarded from OS FrameState)
    bool cursor_in_cell =
        params.cursor_x >= cell_left &&
        params.cursor_x < cell_left + cell_size &&
        params.cursor_y >= cell_top &&
        params.cursor_y < cell_top + cell_size;

    if (cursor_in_cell) {{
        atomic_store_explicit((threadgroup atomic_uint*)&hit_cell, tid, memory_order_relaxed);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Update cursor_cell and handle click
    if (tid == 0) {{
        sim->cursor_cell = hit_cell;
        if (params.mouse_clicked != 0 && hit_cell != 0xFFFFFFFF) {{
            sim->pending_toggle = hit_cell;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: PROCESS ACTIONS
    // ═══════════════════════════════════════════════════════════════════

    uint action = sim->pending_action;

    if (action == ACTION_CLEAR) {{
        cells[tid].alive = 0;
        cells[tid].age = 0;
    }}
    else if (action == ACTION_RANDOM) {{
        uint seed = tid ^ (sim->generation * 1664525u + 1013904223u);
        seed = seed * 1664525u + 1013904223u;
        cells[tid].alive = (seed % 100) < 30 ? 1 : 0;
        cells[tid].age = 0;
    }}
    else if (action == ACTION_GLIDER) {{
        cells[tid].alive = 0;
        cells[tid].age = 0;
        if (tid == (14*32 + 15)) cells[tid].alive = 1;
        if (tid == (15*32 + 16)) cells[tid].alive = 1;
        if (tid == (16*32 + 14)) cells[tid].alive = 1;
        if (tid == (16*32 + 15)) cells[tid].alive = 1;
        if (tid == (16*32 + 16)) cells[tid].alive = 1;
    }}

    if (tid == 0) {{
        sim->pending_action = ACTION_NONE;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: CELL TOGGLE (from click)
    // ═══════════════════════════════════════════════════════════════════

    if (sim->pending_toggle == tid) {{
        cells[tid].alive = 1 - cells[tid].alive;
        cells[tid].age = 0;
    }}

    if (tid == 0) {{
        sim->pending_toggle = 0xFFFFFFFF;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: SIMULATION STEP
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {{
        do_step = false;
        if (sim->running != 0 || action == ACTION_STEP) {{
            sim->accumulator += params.delta_time;
            float step_interval = 1.0 / sim->speed;
            if (sim->accumulator >= step_interval) {{
                sim->accumulator -= step_interval;
                do_step = true;
            }}
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (do_step) {{
        // Snapshot current state
        current_alive[tid] = cells[tid].alive != 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Count neighbors with wrap-around
        uint left_x = (cell_x == 0) ? 31 : cell_x - 1;
        uint right_x = (cell_x == 31) ? 0 : cell_x + 1;
        uint up_y = (cell_y == 0) ? 31 : cell_y - 1;
        uint down_y = (cell_y == 31) ? 0 : cell_y + 1;

        uint count = 0;
        count += current_alive[up_y * 32 + left_x] ? 1 : 0;
        count += current_alive[up_y * 32 + cell_x] ? 1 : 0;
        count += current_alive[up_y * 32 + right_x] ? 1 : 0;
        count += current_alive[cell_y * 32 + left_x] ? 1 : 0;
        count += current_alive[cell_y * 32 + right_x] ? 1 : 0;
        count += current_alive[down_y * 32 + left_x] ? 1 : 0;
        count += current_alive[down_y * 32 + cell_x] ? 1 : 0;
        count += current_alive[down_y * 32 + right_x] ? 1 : 0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply Game of Life rules
        bool was_alive = current_alive[tid];
        bool now_alive = (count == 3) || (was_alive && count == 2);

        cells[tid].alive = now_alive ? 1 : 0;

        if (was_alive != now_alive) {{
            cells[tid].age = 0;
        }} else if (cells[tid].age < 255) {{
            cells[tid].age++;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Count population using SIMD reduction
        uint my_alive = cells[tid].alive;
        uint row_sum = simd_sum(my_alive);

        if (simd_lane == 0) {{
            simd_population[simd_id] = row_sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {{
            uint total = 0;
            for (uint i = 0; i < 32; i++) {{
                total += simd_population[i];
            }}
            sim->population = total;
            sim->generation++;
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: GEOMETRY GENERATION
    // ═══════════════════════════════════════════════════════════════════

    bool alive = cells[tid].alive != 0;
    uchar age = cells[tid].age;
    bool hovered = (tid == sim->cursor_cell);

    // Color based on state and age
    float4 color;
    if (alive) {{
        float brightness = 1.0 - (float(age) / 255.0) * 0.3;
        color = float4(0.2 * brightness, 0.9 * brightness, 0.3 * brightness, 1.0);
    }} else {{
        float ghost = (age < 20) ? (1.0 - float(age) / 20.0) * 0.3 : 0.0;
        color = float4(0.15 + ghost, 0.15 + ghost * 0.5, 0.18 + ghost * 0.3, 1.0);
    }}

    if (hovered) {{
        color.rgb += float3(0.15, 0.15, 0.15);
    }}

    // Generate 6 vertices for cell quad
    float padding = 0.001;
    float x0 = cell_left + padding;
    float y0 = cell_top + padding;
    float x1 = cell_left + cell_size - padding;
    float y1 = cell_top + cell_size - padding;

    uint base = tid * 6;

    // Triangle 1: TL -> BL -> BR
    vertices[base + 0] = CellVertex{{float2(x0, y0), float2(0, 0), color}};
    vertices[base + 1] = CellVertex{{float2(x0, y1), float2(0, 1), color}};
    vertices[base + 2] = CellVertex{{float2(x1, y1), float2(1, 1), color}};

    // Triangle 2: TL -> BR -> TR
    vertices[base + 3] = CellVertex{{float2(x0, y0), float2(0, 0), color}};
    vertices[base + 4] = CellVertex{{float2(x1, y1), float2(1, 1), color}};
    vertices[base + 5] = CellVertex{{float2(x1, y0), float2(1, 0), color}};
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 uv;
    float4 color;
}};

vertex VertexOut gol_vertex(
    const device CellVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    CellVertex v = vertices[vid];
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

fragment float4 gol_fragment(VertexOut in [[stage_in]]) {{
    // SDF rounded corners
    float2 p = in.uv - 0.5;
    float corner_radius = 0.12;
    float2 q = abs(p) - (0.5 - corner_radius);
    float d = length(max(q, 0.0)) - corner_radius;

    float alpha = smoothstep(0.02, -0.02, d);
    float glow = smoothstep(0.15, -0.1, d) * 0.08 * in.color.g;
    float3 final_color = in.color.rgb + glow;

    return float4(final_color, in.color.a * alpha);
}}
"#, header = APP_SHADER_HEADER)
}

// ============================================================================
// GameOfLife App
// ============================================================================

pub struct GameOfLife {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // App-specific buffers
    params_buffer: Buffer,
    cells_buffer: Buffer,
    simulation_buffer: Buffer,
    vertices_buffer: Buffer,

    // Current params (updated each frame)
    current_params: AppParams,

    // Mouse state tracking
    mouse_was_down: bool,
    mouse_clicked: bool,
}

impl GameOfLife {
    /// Create a new Game of Life app
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "GameOfLife");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create pipelines
        let compute_pipeline = builder.create_compute_pipeline(&library, "game_of_life_kernel")?;
        let render_pipeline = builder.create_render_pipeline(&library, "gol_vertex", "gol_fragment")?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<AppParams>());
        let cells_buffer = builder.create_buffer(CELL_COUNT * mem::size_of::<CellState>());
        let simulation_buffer = builder.create_buffer(mem::size_of::<SimulationState>());
        let vertices_buffer = builder.create_buffer(TOTAL_VERTICES * mem::size_of::<CellVertex>());

        // Initialize cells to empty
        unsafe {
            let ptr = cells_buffer.contents() as *mut CellState;
            std::ptr::write_bytes(ptr, 0, CELL_COUNT);
        }

        // Initialize simulation state
        unsafe {
            let ptr = simulation_buffer.contents() as *mut SimulationState;
            *ptr = SimulationState::default();
        }

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            cells_buffer,
            simulation_buffer,
            vertices_buffer,
            current_params: AppParams::default(),
            mouse_was_down: false,
            mouse_clicked: false,
        })
    }

    /// Get simulation state (for reading stats)
    pub fn simulation_state(&self) -> SimulationState {
        unsafe { *(self.simulation_buffer.contents() as *const SimulationState) }
    }

    /// Get mutable simulation state (for setting actions)
    pub fn simulation_state_mut(&self) -> &mut SimulationState {
        unsafe { &mut *(self.simulation_buffer.contents() as *mut SimulationState) }
    }

    /// Set a pending action
    pub fn set_action(&self, action: u32) {
        self.simulation_state_mut().pending_action = action;
    }

    /// Toggle running state
    pub fn toggle_running(&self) {
        let sim = self.simulation_state_mut();
        sim.running = 1 - sim.running;
    }

    /// Adjust speed
    pub fn adjust_speed(&self, delta: f32) {
        let sim = self.simulation_state_mut();
        sim.speed = (sim.speed + delta).clamp(1.0, 60.0);
    }
}

impl GpuApp for GameOfLife {
    fn name(&self) -> &str {
        "Game of Life"
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
            &self.cells_buffer,      // slot 3
            &self.simulation_buffer, // slot 4
            &self.vertices_buffer,   // slot 5
        ]
    }

    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32) {
        // Build params from OS state
        self.current_params = AppParams {
            delta_time,
            cursor_x: frame_state.cursor_x,
            cursor_y: frame_state.cursor_y,
            mouse_clicked: if self.mouse_clicked { 1 } else { 0 },
        };

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut AppParams;
            *ptr = self.current_params;
        }

        // Clear click flag after sending to GPU
        self.mouse_clicked = false;
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            t if t == InputEventType::MouseDown as u16 => {
                self.mouse_was_down = true;
            }
            t if t == InputEventType::MouseUp as u16 => {
                if self.mouse_was_down {
                    self.mouse_clicked = true;
                }
                self.mouse_was_down = false;
            }
            _ => {}
        }
    }

    fn post_frame(&mut self, _timing: &FrameTiming) {
        // Could log stats here if needed
    }

    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(0.05, 0.05, 0.08, 1.0)
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::HighThroughput  // Cellular automaton benefits from frame overlap
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
        assert_eq!(mem::size_of::<CellState>(), 2);
        assert_eq!(mem::size_of::<SimulationState>(), 32);
        assert_eq!(mem::size_of::<CellVertex>(), 32);
        assert_eq!(mem::size_of::<AppParams>(), 16);
    }
}
