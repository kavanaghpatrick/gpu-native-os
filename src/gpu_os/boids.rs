// Boids Flocking Simulation - GPU-Native App Implementation
//
// 1024 boids = 1024 threads (1:1 mapping)
// Each boid is rendered as a small triangle pointing in velocity direction.
// Classic boid rules: separation, alignment, cohesion
// Toroidal wrapping at screen edges
// Mouse interaction (attractor/repeller)

use super::app::{AppBuilder, GpuApp, PipelineMode, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

pub const BOID_COUNT: usize = 1024;
pub const VERTICES_PER_BOID: usize = 3; // Triangle per boid
pub const TOTAL_VERTICES: usize = BOID_COUNT * VERTICES_PER_BOID;

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Boid state - 16 bytes per boid
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Boid {
    pub position: [f32; 2], // Normalized 0-1
    pub velocity: [f32; 2], // Normalized direction * speed
}

/// Boid simulation parameters - 48 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BoidParams {
    pub delta_time: f32,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub mouse_down: u32,
    // Rule weights
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub mouse_weight: f32,
    // Rule distances
    pub visual_range: f32,      // How far boids can see
    pub separation_dist: f32,   // Minimum comfortable distance
    pub max_speed: f32,
    pub time: f32, // For color effects
}

impl Default for BoidParams {
    fn default() -> Self {
        Self {
            delta_time: 1.0 / 120.0,
            mouse_x: 0.5,
            mouse_y: 0.5,
            mouse_down: 0,
            separation_weight: 0.05,
            alignment_weight: 0.05,
            cohesion_weight: 0.005,
            mouse_weight: 0.0003,
            visual_range: 0.08,
            separation_dist: 0.025,
            max_speed: 0.4,
            time: 0.0,
        }
    }
}

/// Boid vertex - 32 bytes (position, color)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct BoidVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

// ============================================================================
// Shader Source
// ============================================================================

fn shader_source() -> String {
    format!(
        r#"
{header}

// ============================================================================
// App-Specific Structures
// ============================================================================

struct Boid {{
    float2 position;
    float2 velocity;
}};

struct BoidParams {{
    float delta_time;
    float mouse_x;
    float mouse_y;
    uint mouse_down;
    float separation_weight;
    float alignment_weight;
    float cohesion_weight;
    float mouse_weight;
    float visual_range;
    float separation_dist;
    float max_speed;
    float time;
}};

struct BoidVertex {{
    float2 position;
    float2 uv;
    float4 color;
}};

// ============================================================================
// Helper Functions
// ============================================================================

// Toroidal distance (handles wrap-around)
float2 toroidal_delta(float2 from, float2 to) {{
    float2 delta = to - from;
    // Wrap around if closer through the edge
    if (delta.x > 0.5) delta.x -= 1.0;
    if (delta.x < -0.5) delta.x += 1.0;
    if (delta.y > 0.5) delta.y -= 1.0;
    if (delta.y < -0.5) delta.y += 1.0;
    return delta;
}}

float toroidal_distance(float2 a, float2 b) {{
    float2 delta = toroidal_delta(a, b);
    return length(delta);
}}

// Limit vector magnitude
float2 limit_magnitude(float2 v, float max_mag) {{
    float mag = length(v);
    if (mag > max_mag && mag > 0.0001) {{
        return v * (max_mag / mag);
    }}
    return v;
}}

// ============================================================================
// Main Compute Kernel
// ============================================================================

kernel void boids_kernel(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant BoidParams& params [[buffer(2)]],
    device Boid* boids [[buffer(3)]],
    device BoidVertex* vertices [[buffer(4)]],
    device atomic_uint* vertex_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {{
    // This boid's current state
    Boid my_boid = boids[tid];
    float2 my_pos = my_boid.position;
    float2 my_vel = my_boid.velocity;

    // Threadgroup memory for efficient neighbor queries
    threadgroup float2 tg_positions[1024];
    threadgroup float2 tg_velocities[1024];

    // Share positions for neighbor queries
    tg_positions[tid] = my_pos;
    tg_velocities[tid] = my_vel;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: NEIGHBOR ANALYSIS (Boid Rules)
    // ═══════════════════════════════════════════════════════════════════

    float2 separation = float2(0.0);
    float2 alignment = float2(0.0);
    float2 cohesion_center = float2(0.0);
    uint neighbor_count = 0;
    uint close_count = 0;

    // Check all other boids
    for (uint i = 0; i < 1024; i++) {{
        if (i == tid) continue;

        float2 other_pos = tg_positions[i];
        float2 other_vel = tg_velocities[i];

        float dist = toroidal_distance(my_pos, other_pos);

        if (dist < params.visual_range) {{
            float2 delta = toroidal_delta(my_pos, other_pos);

            // Cohesion: accumulate neighbor positions
            cohesion_center += delta;

            // Alignment: accumulate neighbor velocities
            alignment += other_vel;

            neighbor_count++;

            // Separation: stronger push for very close boids
            if (dist < params.separation_dist && dist > 0.0001) {{
                // Push away, stronger when closer
                separation -= delta * (1.0 - dist / params.separation_dist);
                close_count++;
            }}
        }}
    }}

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: APPLY BOID RULES
    // ═══════════════════════════════════════════════════════════════════

    float2 acceleration = float2(0.0);

    if (neighbor_count > 0) {{
        // Cohesion: steer toward center of neighbors
        float2 cohesion_dir = cohesion_center / float(neighbor_count);
        acceleration += cohesion_dir * params.cohesion_weight;

        // Alignment: match average velocity of neighbors
        float2 avg_vel = alignment / float(neighbor_count);
        float2 alignment_steer = avg_vel - my_vel;
        acceleration += alignment_steer * params.alignment_weight;
    }}

    // Separation: always apply if there are close boids
    if (close_count > 0) {{
        acceleration += separation * params.separation_weight;
    }}

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: MOUSE INTERACTION
    // ═══════════════════════════════════════════════════════════════════

    float2 mouse_pos = float2(params.mouse_x, params.mouse_y);
    float2 to_mouse = toroidal_delta(my_pos, mouse_pos);
    float mouse_dist = length(to_mouse);

    if (mouse_dist > 0.01 && mouse_dist < 0.3) {{
        float mouse_strength = params.mouse_weight / (mouse_dist * mouse_dist + 0.001);

        if (params.mouse_down != 0) {{
            // Attract to mouse when pressed
            acceleration += to_mouse * mouse_strength * 50.0;
        }} else {{
            // Gentle avoidance when not pressed
            acceleration -= to_mouse * mouse_strength * 5.0;
        }}
    }}

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: UPDATE VELOCITY AND POSITION
    // ═══════════════════════════════════════════════════════════════════

    // Apply acceleration
    my_vel += acceleration;

    // Limit speed
    my_vel = limit_magnitude(my_vel, params.max_speed);

    // Ensure minimum speed (boids should always be moving)
    float speed = length(my_vel);
    if (speed < 0.05) {{
        // Add some random direction if too slow
        uint seed = tid + uint(params.time * 1000.0);
        float angle = random_float(seed) * 6.28318;
        my_vel = float2(cos(angle), sin(angle)) * 0.1;
    }}

    // Update position
    my_pos += my_vel * params.delta_time;

    // Toroidal wrapping
    my_pos = fract(my_pos + 1.0);

    // Write back to global memory
    boids[tid].position = my_pos;
    boids[tid].velocity = my_vel;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: GENERATE TRIANGLE GEOMETRY
    // ═══════════════════════════════════════════════════════════════════

    // Calculate heading angle from velocity
    float angle = atan2(my_vel.y, my_vel.x);

    // Triangle size based on speed
    float base_size = 0.008;
    float size = base_size * (0.7 + 0.3 * (speed / params.max_speed));

    // Triangle vertices (pointing in velocity direction)
    float2 forward = float2(cos(angle), sin(angle));
    float2 right = float2(-forward.y, forward.x);

    float2 tip = my_pos + forward * size * 2.0;
    float2 left_wing = my_pos - forward * size * 0.5 + right * size;
    float2 right_wing = my_pos - forward * size * 0.5 - right * size;

    // Color based on local flock density and velocity
    float density = float(neighbor_count) / 50.0; // 0-1 based on neighbors
    float speed_ratio = speed / params.max_speed;

    // Beautiful color gradient:
    // - Hue based on heading direction (creates rainbow flocking patterns)
    // - Saturation based on density (more saturated in denser areas)
    // - Value based on speed
    float hue = fmod((angle / 6.28318) * 360.0 + 180.0, 360.0);
    float saturation = 0.6 + density * 0.4;
    float value = 0.7 + speed_ratio * 0.3;

    float3 color = hsv_to_rgb(hue, saturation, value);

    // Add subtle shimmer based on time
    float shimmer = sin(params.time * 3.0 + float(tid) * 0.1) * 0.05;
    color = clamp(color + shimmer, 0.0, 1.0);

    // Write triangle vertices
    uint base = tid * 3;

    vertices[base + 0].position = tip * 2.0 - 1.0; // Convert to clip space
    vertices[base + 0].position.y *= -1.0;
    vertices[base + 0].uv = float2(0.5, 0.0);
    vertices[base + 0].color = float4(color * 1.2, 1.0); // Tip is brighter

    vertices[base + 1].position = left_wing * 2.0 - 1.0;
    vertices[base + 1].position.y *= -1.0;
    vertices[base + 1].uv = float2(0.0, 1.0);
    vertices[base + 1].color = float4(color * 0.8, 1.0); // Wings darker

    vertices[base + 2].position = right_wing * 2.0 - 1.0;
    vertices[base + 2].position.y *= -1.0;
    vertices[base + 2].uv = float2(1.0, 1.0);
    vertices[base + 2].color = float4(color * 0.8, 1.0);

    // Reset vertex count on thread 0 (written once)
    if (tid == 0) {{
        atomic_store_explicit(vertex_count, 1024 * 3, memory_order_relaxed);
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

vertex VertexOut boid_vertex(
    const device BoidVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    BoidVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}}

// ============================================================================
// Fragment Shader
// ============================================================================

fragment float4 boid_fragment(VertexOut in [[stage_in]]) {{
    // Simple gradient from tip to base
    float alpha = 1.0 - in.uv.y * 0.3;

    // Soft glow effect
    float glow = max(in.color.r, max(in.color.g, in.color.b)) * 0.15;
    float3 final_color = in.color.rgb + glow;

    return float4(final_color, alpha);
}}
"#,
        header = APP_SHADER_HEADER
    )
}

// ============================================================================
// BoidsSimulation App
// ============================================================================

pub struct BoidsSimulation {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    params_buffer: Buffer,
    boids_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,

    // Current params
    current_params: BoidParams,

    // Input state
    mouse_x: f32,
    mouse_y: f32,
    mouse_down: bool,
    time: f32,
}

impl BoidsSimulation {
    /// Create a new Boids simulation
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "BoidsSimulation");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create pipelines
        let compute_pipeline = builder.create_compute_pipeline(&library, "boids_kernel")?;
        let render_pipeline =
            builder.create_render_pipeline(&library, "boid_vertex", "boid_fragment")?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<BoidParams>());
        let boids_buffer = builder.create_buffer(BOID_COUNT * mem::size_of::<Boid>());
        let vertices_buffer = builder.create_buffer(TOTAL_VERTICES * mem::size_of::<BoidVertex>());
        let vertex_count_buffer = builder.create_buffer(mem::size_of::<u32>());

        // Initialize boids with random positions and velocities
        unsafe {
            let ptr = boids_buffer.contents() as *mut Boid;
            for i in 0..BOID_COUNT {
                // Use a deterministic seed for reproducible patterns
                let seed = i as u32;
                let px = Self::hash_to_float(seed);
                let py = Self::hash_to_float(seed.wrapping_add(12345));
                let angle = Self::hash_to_float(seed.wrapping_add(67890)) * std::f32::consts::TAU;
                let speed = 0.1 + Self::hash_to_float(seed.wrapping_add(11111)) * 0.1;

                *ptr.add(i) = Boid {
                    position: [px, py],
                    velocity: [angle.cos() * speed, angle.sin() * speed],
                };
            }
        }

        // Initialize params
        let current_params = BoidParams::default();
        unsafe {
            let ptr = params_buffer.contents() as *mut BoidParams;
            *ptr = current_params;
        }

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            boids_buffer,
            vertices_buffer,
            vertex_count_buffer,
            current_params,
            mouse_x: 0.5,
            mouse_y: 0.5,
            mouse_down: false,
            time: 0.0,
        })
    }

    // Simple hash function for initialization
    fn hash_to_float(x: u32) -> f32 {
        let mut h = x;
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2ae35);
        h ^= h >> 16;
        (h as f32) / (u32::MAX as f32)
    }

    /// Get current params for reading
    pub fn params(&self) -> &BoidParams {
        &self.current_params
    }

    /// Adjust separation weight
    pub fn adjust_separation(&mut self, delta: f32) {
        self.current_params.separation_weight =
            (self.current_params.separation_weight + delta).clamp(0.0, 0.2);
    }

    /// Adjust alignment weight
    pub fn adjust_alignment(&mut self, delta: f32) {
        self.current_params.alignment_weight =
            (self.current_params.alignment_weight + delta).clamp(0.0, 0.2);
    }

    /// Adjust cohesion weight
    pub fn adjust_cohesion(&mut self, delta: f32) {
        self.current_params.cohesion_weight =
            (self.current_params.cohesion_weight + delta).clamp(0.0, 0.02);
    }

    /// Adjust visual range
    pub fn adjust_visual_range(&mut self, delta: f32) {
        self.current_params.visual_range =
            (self.current_params.visual_range + delta).clamp(0.02, 0.2);
    }

    /// Reset to default parameters
    pub fn reset_params(&mut self) {
        self.current_params = BoidParams::default();
    }

    /// Scatter boids (randomize positions)
    pub fn scatter(&mut self) {
        unsafe {
            let ptr = self.boids_buffer.contents() as *mut Boid;
            for i in 0..BOID_COUNT {
                let seed = i as u32 + (self.time * 1000.0) as u32;
                let px = Self::hash_to_float(seed);
                let py = Self::hash_to_float(seed.wrapping_add(12345));
                let boid = ptr.add(i);
                (*boid).position = [px, py];
            }
        }
    }
}

impl GpuApp for BoidsSimulation {
    fn name(&self) -> &str {
        "Boids Simulation"
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
            &self.boids_buffer,        // slot 3
            &self.vertices_buffer,     // slot 4
            &self.vertex_count_buffer, // slot 5
        ]
    }

    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32) {
        self.time += delta_time;

        // Update params from OS state
        self.current_params.delta_time = delta_time;
        self.current_params.mouse_x = frame_state.cursor_x;
        self.current_params.mouse_y = frame_state.cursor_y;
        self.current_params.mouse_down = if self.mouse_down { 1 } else { 0 };
        self.current_params.time = self.time;

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut BoidParams;
            *ptr = self.current_params;
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            t if t == InputEventType::MouseMove as u16 => {
                self.mouse_x = event.position[0];
                self.mouse_y = event.position[1];
            }
            t if t == InputEventType::MouseDown as u16 => {
                self.mouse_down = true;
            }
            t if t == InputEventType::MouseUp as u16 => {
                self.mouse_down = false;
            }
            _ => {}
        }
    }

    fn post_frame(&mut self, _timing: &FrameTiming) {
        // Could log stats here
    }

    fn clear_color(&self) -> MTLClearColor {
        // Dark navy blue background
        MTLClearColor::new(0.02, 0.02, 0.06, 1.0)
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::HighThroughput  // Flocking simulation benefits from frame overlap
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
        assert_eq!(mem::size_of::<Boid>(), 16);
        assert_eq!(mem::size_of::<BoidParams>(), 48);
        assert_eq!(mem::size_of::<BoidVertex>(), 32);
    }

    #[test]
    fn test_default_params() {
        let params = BoidParams::default();
        assert!(params.separation_weight > 0.0);
        assert!(params.alignment_weight > 0.0);
        assert!(params.cohesion_weight > 0.0);
        assert!(params.visual_range > 0.0);
    }
}
