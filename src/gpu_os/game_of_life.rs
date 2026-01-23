// Game of Life Demo - GPU-Native Implementation
//
// 32x32 grid = 1024 cells = 1024 threads (1:1 mapping)
// All simulation, input handling, and rendering runs on GPU

use metal::*;
use std::mem;

/// Metal shader source for Game of Life
pub const GOL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Data Structures
// ============================================================================

struct CellState {
    uchar alive;    // 0 or 1
    uchar age;      // generations since state change
};

struct SimulationState {
    uint generation;
    uint population;
    uint running;
    float speed;
    float accumulator;
    uint cursor_cell;
    uint pending_toggle;
    uint pending_action;
};

struct CellVertex {
    float2 position;
    float2 uv;
    float4 color;
};

struct DrawArguments {
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
};

struct FrameParams {
    float delta_time;
    float cursor_x;
    float cursor_y;
    uint mouse_down;
    uint mouse_clicked;
};

// Action constants
constant uint ACTION_NONE = 0;
constant uint ACTION_CLEAR = 1;
constant uint ACTION_RANDOM = 2;
constant uint ACTION_GLIDER = 3;
constant uint ACTION_STEP = 4;

// ============================================================================
// Main Game of Life Kernel
// ============================================================================

kernel void game_of_life_kernel(
    device CellState* cells [[buffer(0)]],
    device SimulationState* sim [[buffer(1)]],
    device CellVertex* vertices [[buffer(2)]],
    device DrawArguments* draw_args [[buffer(3)]],
    constant FrameParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Thread identity -> cell coordinates
    uint cell_x = tid % 32;
    uint cell_y = tid / 32;

    // Threadgroup shared memory
    threadgroup bool current_alive[1024];
    threadgroup uint simd_population[32];
    threadgroup float shared_cursor_x;
    threadgroup float shared_cursor_y;
    threadgroup uint hit_cell;
    threadgroup bool do_step;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: INPUT & HIT TESTING
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {
        shared_cursor_x = params.cursor_x;
        shared_cursor_y = params.cursor_y;
        hit_cell = 0xFFFFFFFF;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate cell bounds (grid fills 0.1-0.9 of screen)
    float grid_start = 0.1;
    float grid_size = 0.8;
    float cell_size = grid_size / 32.0;

    float cell_left = grid_start + float(cell_x) * cell_size;
    float cell_top = grid_start + float(cell_y) * cell_size;

    bool cursor_in_cell =
        shared_cursor_x >= cell_left &&
        shared_cursor_x < cell_left + cell_size &&
        shared_cursor_y >= cell_top &&
        shared_cursor_y < cell_top + cell_size;

    // Find hit cell using SIMD ballot
    if (cursor_in_cell) {
        atomic_store_explicit((threadgroup atomic_uint*)&hit_cell, tid, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Update cursor_cell for hover highlight
    if (tid == 0) {
        sim->cursor_cell = hit_cell;

        // Handle click -> toggle
        if (params.mouse_clicked && hit_cell != 0xFFFFFFFF) {
            sim->pending_toggle = hit_cell;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: PROCESS ACTIONS
    // ═══════════════════════════════════════════════════════════════════

    uint action = sim->pending_action;

    if (action == ACTION_CLEAR) {
        cells[tid].alive = 0;
        cells[tid].age = 0;
    }
    else if (action == ACTION_RANDOM) {
        // LCG pseudo-random
        uint seed = tid ^ (sim->generation * 1664525u + 1013904223u);
        seed = seed * 1664525u + 1013904223u;
        cells[tid].alive = (seed % 100) < 30 ? 1 : 0;
        cells[tid].age = 0;
    }
    else if (action == ACTION_GLIDER) {
        // Spawn glider at center
        cells[tid].alive = 0;
        cells[tid].age = 0;
        if (tid == (14*32 + 15)) cells[tid].alive = 1;  // .#.
        if (tid == (15*32 + 16)) cells[tid].alive = 1;  // ..#
        if (tid == (16*32 + 14)) cells[tid].alive = 1;  // ###
        if (tid == (16*32 + 15)) cells[tid].alive = 1;
        if (tid == (16*32 + 16)) cells[tid].alive = 1;
    }

    if (tid == 0) {
        sim->pending_action = ACTION_NONE;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: CELL TOGGLE (from click)
    // ═══════════════════════════════════════════════════════════════════

    if (sim->pending_toggle == tid) {
        cells[tid].alive = 1 - cells[tid].alive;
        cells[tid].age = 0;
    }

    if (tid == 0) {
        sim->pending_toggle = 0xFFFFFFFF;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: SIMULATION STEP
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {
        do_step = false;
        if (sim->running != 0 || action == ACTION_STEP) {
            sim->accumulator += params.delta_time;
            float step_interval = 1.0 / sim->speed;
            if (sim->accumulator >= step_interval) {
                sim->accumulator -= step_interval;
                do_step = true;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (do_step) {
        // 4a. Snapshot current state
        current_alive[tid] = cells[tid].alive != 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4b. Count neighbors with wrap-around
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

        // 4c. Apply Game of Life rules
        bool was_alive = current_alive[tid];
        bool now_alive = (count == 3) || (was_alive && count == 2);

        cells[tid].alive = now_alive ? 1 : 0;

        if (was_alive != now_alive) {
            cells[tid].age = 0;
        } else if (cells[tid].age < 255) {
            cells[tid].age++;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4d. Count population using SIMD reduction
        uint my_alive = cells[tid].alive;
        uint row_sum = simd_sum(my_alive);

        if (simd_lane == 0) {
            simd_population[simd_id] = row_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint total = 0;
            for (uint i = 0; i < 32; i++) {
                total += simd_population[i];
            }
            sim->population = total;
            sim->generation++;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: GEOMETRY GENERATION
    // ═══════════════════════════════════════════════════════════════════

    bool alive = cells[tid].alive != 0;
    uchar age = cells[tid].age;
    bool hovered = (tid == sim->cursor_cell);

    // Color based on state and age
    float4 color;
    if (alive) {
        float brightness = 1.0 - (float(age) / 255.0) * 0.3;
        color = float4(0.2 * brightness, 0.9 * brightness, 0.3 * brightness, 1.0);
    } else {
        float ghost = (age < 20) ? (1.0 - float(age) / 20.0) * 0.3 : 0.0;
        color = float4(0.15 + ghost, 0.15 + ghost * 0.5, 0.18 + ghost * 0.3, 1.0);
    }

    if (hovered) {
        color.rgb += float3(0.15, 0.15, 0.15);
    }

    // Generate 6 vertices for cell quad
    float padding = 0.001;
    float x0 = cell_left + padding;
    float y0 = cell_top + padding;
    float x1 = cell_left + cell_size - padding;
    float y1 = cell_top + cell_size - padding;

    uint base = tid * 6;

    // Triangle 1: TL -> BL -> BR
    vertices[base + 0] = CellVertex{float2(x0, y0), float2(0, 0), color};
    vertices[base + 1] = CellVertex{float2(x0, y1), float2(0, 1), color};
    vertices[base + 2] = CellVertex{float2(x1, y1), float2(1, 1), color};

    // Triangle 2: TL -> BR -> TR
    vertices[base + 3] = CellVertex{float2(x0, y0), float2(0, 0), color};
    vertices[base + 4] = CellVertex{float2(x1, y1), float2(1, 1), color};
    vertices[base + 5] = CellVertex{float2(x1, y0), float2(1, 0), color};

    // Thread 0 writes draw arguments
    if (tid == 0) {
        draw_args->vertex_count = 1024 * 6;
        draw_args->instance_count = 1;
        draw_args->vertex_start = 0;
        draw_args->base_instance = 0;
    }
}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut gol_vertex(
    const device CellVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    CellVertex v = vertices[vid];
    VertexOut out;

    // Convert from [0,1] to clip space [-1,1]
    out.position = float4(v.position * 2.0 - 1.0, 0.0, 1.0);
    out.position.y = -out.position.y;
    out.uv = v.uv;
    out.color = v.color;

    return out;
}

// ============================================================================
// Fragment Shader
// ============================================================================

fragment float4 gol_fragment(VertexOut in [[stage_in]]) {
    // SDF rounded corners
    float2 p = in.uv - 0.5;
    float corner_radius = 0.12;
    float2 q = abs(p) - (0.5 - corner_radius);
    float d = length(max(q, 0.0)) - corner_radius;

    float alpha = smoothstep(0.02, -0.02, d);

    // Subtle glow for bright cells
    float glow = smoothstep(0.15, -0.1, d) * 0.08 * in.color.g;
    float3 final_color = in.color.rgb + glow;

    return float4(final_color, in.color.a * alpha);
}
"#;

/// Cell state - 2 bytes per cell
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CellState {
    pub alive: u8,
    pub age: u8,
}

/// Simulation state - 32 bytes
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
            speed: 10.0, // 10 generations per second
            accumulator: 0.0,
            cursor_cell: 0xFFFFFFFF,
            pending_toggle: 0xFFFFFFFF,
            pending_action: 0,
        }
    }
}

/// Frame parameters passed each frame
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct FrameParams {
    pub delta_time: f32,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub mouse_down: u32,
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

/// Draw arguments for indirect draw
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DrawArguments {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

/// Action constants
pub const ACTION_NONE: u32 = 0;
pub const ACTION_CLEAR: u32 = 1;
pub const ACTION_RANDOM: u32 = 2;
pub const ACTION_GLIDER: u32 = 3;
pub const ACTION_STEP: u32 = 4;

/// Game of Life GPU resources
pub struct GameOfLife {
    pub compute_pipeline: ComputePipelineState,
    pub render_pipeline: RenderPipelineState,
    pub cells_buffer: Buffer,
    pub simulation_buffer: Buffer,
    pub vertices_buffer: Buffer,
    pub draw_args_buffer: Buffer,
    pub params_buffer: Buffer,
    pub command_queue: CommandQueue,
}

impl GameOfLife {
    pub const CELL_COUNT: usize = 1024;
    pub const GRID_SIZE: usize = 32;

    /// Create a new Game of Life instance
    pub fn new(device: &Device) -> Result<Self, String> {
        // Compile shaders
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(GOL_SHADER_SOURCE, &options)
            .map_err(|e| format!("Failed to compile shaders: {}", e))?;

        // Create compute pipeline
        let kernel_fn = library
            .get_function("game_of_life_kernel", None)
            .map_err(|e| format!("Failed to get kernel function: {}", e))?;

        let compute_pipeline = device
            .new_compute_pipeline_state_with_function(&kernel_fn)
            .map_err(|e| format!("Failed to create compute pipeline: {}", e))?;

        // Create render pipeline
        let vertex_fn = library
            .get_function("gol_vertex", None)
            .map_err(|e| format!("Failed to get vertex function: {}", e))?;

        let fragment_fn = library
            .get_function("gol_fragment", None)
            .map_err(|e| format!("Failed to get fragment function: {}", e))?;

        let pipeline_desc = RenderPipelineDescriptor::new();
        pipeline_desc.set_vertex_function(Some(&vertex_fn));
        pipeline_desc.set_fragment_function(Some(&fragment_fn));

        let color_attachment = pipeline_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        color_attachment.set_blending_enabled(true);
        color_attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        color_attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        color_attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        color_attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device
            .new_render_pipeline_state(&pipeline_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        // Create buffers
        let cells_size = Self::CELL_COUNT * mem::size_of::<CellState>();
        let cells_buffer = device.new_buffer(cells_size as u64, MTLResourceOptions::StorageModeShared);

        let simulation_buffer = device.new_buffer(
            mem::size_of::<SimulationState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let vertices_size = Self::CELL_COUNT * 6 * mem::size_of::<CellVertex>();
        let vertices_buffer = device.new_buffer(vertices_size as u64, MTLResourceOptions::StorageModeShared);

        let draw_args_buffer = device.new_buffer(
            mem::size_of::<DrawArguments>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = device.new_buffer(
            mem::size_of::<FrameParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize cells to empty
        unsafe {
            let ptr = cells_buffer.contents() as *mut CellState;
            std::ptr::write_bytes(ptr, 0, Self::CELL_COUNT);
        }

        // Initialize simulation state
        unsafe {
            let ptr = simulation_buffer.contents() as *mut SimulationState;
            *ptr = SimulationState::default();
        }

        // Initialize draw args
        unsafe {
            let ptr = draw_args_buffer.contents() as *mut DrawArguments;
            *ptr = DrawArguments {
                vertex_count: (Self::CELL_COUNT * 6) as u32,
                instance_count: 1,
                vertex_start: 0,
                base_instance: 0,
            };
        }

        let command_queue = device.new_command_queue();

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            cells_buffer,
            simulation_buffer,
            vertices_buffer,
            draw_args_buffer,
            params_buffer,
            command_queue,
        })
    }

    /// Update frame parameters
    pub fn update_params(&self, params: &FrameParams) {
        unsafe {
            let ptr = self.params_buffer.contents() as *mut FrameParams;
            *ptr = *params;
        }
    }

    /// Get simulation state
    pub fn simulation_state(&self) -> SimulationState {
        unsafe { *(self.simulation_buffer.contents() as *const SimulationState) }
    }

    /// Get mutable simulation state
    pub fn simulation_state_mut(&self) -> &mut SimulationState {
        unsafe { &mut *(self.simulation_buffer.contents() as *mut SimulationState) }
    }

    /// Run compute and render passes
    pub fn render(&self, drawable: &MetalDrawableRef) {
        let command_buffer = self.command_queue.new_command_buffer();

        // Compute pass
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&self.compute_pipeline);
        compute_encoder.set_buffer(0, Some(&self.cells_buffer), 0);
        compute_encoder.set_buffer(1, Some(&self.simulation_buffer), 0);
        compute_encoder.set_buffer(2, Some(&self.vertices_buffer), 0);
        compute_encoder.set_buffer(3, Some(&self.draw_args_buffer), 0);
        compute_encoder.set_buffer(4, Some(&self.params_buffer), 0);

        // Dispatch single threadgroup of 1024 threads
        let threadgroup_size = MTLSize::new(1024, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);
        compute_encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        compute_encoder.end_encoding();

        // Render pass
        let render_pass_desc = RenderPassDescriptor::new();
        let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.05, 0.05, 0.08, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let render_encoder = command_buffer.new_render_command_encoder(&render_pass_desc);
        render_encoder.set_render_pipeline_state(&self.render_pipeline);
        render_encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);

        // Draw cells
        render_encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            (Self::CELL_COUNT * 6) as u64,
        );

        render_encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        assert_eq!(mem::size_of::<CellState>(), 2);
        assert_eq!(mem::size_of::<SimulationState>(), 32);
        assert_eq!(mem::size_of::<CellVertex>(), 32);
        assert_eq!(mem::size_of::<DrawArguments>(), 16);
        assert_eq!(mem::size_of::<FrameParams>(), 20);
    }
}
