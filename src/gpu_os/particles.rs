// Particle System - GPU-Native App Implementation
//
// 10,000+ particles with physics simulation running entirely on the GPU.
// Uses the GpuApp framework for OS integration.
//
// Features:
// - Gravity, mouse attraction, slight randomness
// - Particles fade out as life decreases, respawn when dead
// - Rainbow colors based on velocity
// - Click and drag to attract particles toward cursor

use super::app::{AppBuilder, GpuApp, PipelineMode, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

pub const PARTICLE_COUNT: usize = 16384; // 16K particles for impressive visuals
pub const VERTICES_PER_PARTICLE: usize = 6; // 2 triangles per particle quad
pub const TOTAL_VERTICES: usize = PARTICLE_COUNT * VERTICES_PER_PARTICLE;
pub const THREAD_COUNT: usize = 1024; // Threads per dispatch
pub const PARTICLES_PER_THREAD: usize = (PARTICLE_COUNT + THREAD_COUNT - 1) / THREAD_COUNT;

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Particle state - 48 bytes per particle
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Particle {
    pub position: [f32; 2],  // 8 bytes
    pub velocity: [f32; 2],  // 8 bytes
    pub color: [f32; 4],     // 16 bytes
    pub life: f32,           // 4 bytes - 0 to 1, decreases over time
    pub max_life: f32,       // 4 bytes - for fade calculation
    pub size: f32,           // 4 bytes - particle size
    pub _padding: f32,       // 4 bytes - alignment
}

/// App parameters passed each frame (slot 2) - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ParticleParams {
    pub mouse_x: f32,         // 4 bytes
    pub mouse_y: f32,         // 4 bytes
    pub delta_time: f32,      // 4 bytes
    pub mouse_down: u32,      // 4 bytes - 0 or 1
    pub time: f32,            // 4 bytes - total elapsed time
    pub frame_number: u32,    // 4 bytes - for randomness seeding
    pub attraction_strength: f32, // 4 bytes
    pub gravity_strength: f32,    // 4 bytes
}

/// Particle vertex - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ParticleVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// Vertex count - for atomic counter
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct VertexCount {
    pub count: u32,
    pub _padding: [u32; 3],
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

struct Particle {{
    float2 position;
    float2 velocity;
    float4 color;
    float life;
    float max_life;
    float size;
    float _padding;
}};

struct ParticleParams {{
    float mouse_x;
    float mouse_y;
    float delta_time;
    uint mouse_down;
    float time;
    uint frame_number;
    float attraction_strength;
    float gravity_strength;
}};

struct ParticleVertex {{
    float2 position;
    float2 uv;
    float4 color;
}};

struct VertexCount {{
    atomic_uint count;
    uint _padding[3];
}};

// ============================================================================
// Constants
// ============================================================================

constant uint PARTICLE_COUNT = {particle_count};
constant uint PARTICLES_PER_THREAD = {particles_per_thread};

// ============================================================================
// Particle Physics Helpers
// ============================================================================

// Respawn a particle with random properties
void respawn_particle(thread Particle& p, uint seed, float time) {{
    // Random position - start from edges or center
    uint edge = hash(seed) % 4;
    float rand1 = random_float(hash(seed + 1));
    float rand2 = random_float(hash(seed + 2));

    if (edge == 0) {{ // Top
        p.position = float2(rand1, 0.0);
        p.velocity = float2(random_range(hash(seed + 3), -0.1, 0.1), random_range(hash(seed + 4), 0.1, 0.3));
    }} else if (edge == 1) {{ // Bottom
        p.position = float2(rand1, 1.0);
        p.velocity = float2(random_range(hash(seed + 3), -0.1, 0.1), random_range(hash(seed + 4), -0.3, -0.1));
    }} else if (edge == 2) {{ // Left
        p.position = float2(0.0, rand1);
        p.velocity = float2(random_range(hash(seed + 3), 0.1, 0.3), random_range(hash(seed + 4), -0.1, 0.1));
    }} else {{ // Right
        p.position = float2(1.0, rand1);
        p.velocity = float2(random_range(hash(seed + 3), -0.3, -0.1), random_range(hash(seed + 4), -0.1, 0.1));
    }}

    // Random life span (2-5 seconds worth at 60fps = 120-300 frames)
    p.max_life = random_range(hash(seed + 5), 2.0, 5.0);
    p.life = p.max_life;

    // Random size
    p.size = random_range(hash(seed + 6), 0.003, 0.012);

    // Initial color based on spawn position (will be updated based on velocity)
    float hue = random_range(hash(seed + 7), 0.0, 360.0);
    p.color = float4(hsv_to_rgb(hue, 0.9, 1.0), 1.0);
}}

// ============================================================================
// Main Compute Kernel
// ============================================================================

kernel void particle_kernel(
    constant FrameState& frame [[buffer(0)]],       // OS: frame state
    device InputQueue* input_queue [[buffer(1)]],   // OS: input queue (unused)
    constant ParticleParams& params [[buffer(2)]],  // App: per-frame params
    device Particle* particles [[buffer(3)]],       // App: particle array
    device ParticleVertex* vertices [[buffer(4)]],  // App: output vertices
    device VertexCount* vertex_count [[buffer(5)]], // App: vertex count
    uint tid [[thread_index_in_threadgroup]]
) {{
    // Reset vertex count at thread 0
    if (tid == 0) {{
        atomic_store_explicit(&vertex_count->count, 0, memory_order_relaxed);
    }}
    threadgroup_barrier(mem_flags::mem_device);

    // Each thread processes multiple particles
    for (uint i = 0; i < PARTICLES_PER_THREAD; i++) {{
        uint pid = tid * PARTICLES_PER_THREAD + i;
        if (pid >= PARTICLE_COUNT) break;

        Particle p = particles[pid];

        // Initialize particles on first frame
        if (params.frame_number == 0) {{
            uint seed = pid * 12345u + 67890u;
            respawn_particle(p, seed, params.time);
        }}

        // ═══════════════════════════════════════════════════════════════════
        // PHYSICS SIMULATION
        // ═══════════════════════════════════════════════════════════════════

        float dt = params.delta_time;

        // Gravity (downward force)
        p.velocity.y += params.gravity_strength * dt;

        // Mouse attraction when clicked
        if (params.mouse_down != 0) {{
            float2 mouse_pos = float2(params.mouse_x, params.mouse_y);
            float2 to_mouse = mouse_pos - p.position;
            float dist = length(to_mouse);

            if (dist > 0.01) {{
                float2 dir = normalize(to_mouse);
                // Inverse square attraction with soft falloff
                float strength = params.attraction_strength / (dist * dist + 0.1);
                strength = min(strength, 2.0); // Cap maximum force
                p.velocity += dir * strength * dt;
            }}
        }}

        // Add slight turbulence/randomness
        uint noise_seed = pid + params.frame_number * 7919u;
        float noise_x = random_range(hash(noise_seed), -1.0, 1.0);
        float noise_y = random_range(hash(noise_seed + 1), -1.0, 1.0);
        p.velocity += float2(noise_x, noise_y) * 0.05 * dt;

        // Damping (air resistance)
        p.velocity *= (1.0 - 0.5 * dt);

        // Update position
        p.position += p.velocity * dt;

        // Bounce off edges with energy loss
        float bounce = 0.6;
        if (p.position.x < 0.0) {{
            p.position.x = 0.0;
            p.velocity.x *= -bounce;
        }}
        if (p.position.x > 1.0) {{
            p.position.x = 1.0;
            p.velocity.x *= -bounce;
        }}
        if (p.position.y < 0.0) {{
            p.position.y = 0.0;
            p.velocity.y *= -bounce;
        }}
        if (p.position.y > 1.0) {{
            p.position.y = 1.0;
            p.velocity.y *= -bounce;
        }}

        // Decrease life
        p.life -= dt;

        // Respawn if dead
        if (p.life <= 0.0) {{
            uint seed = pid * 31337u + params.frame_number * 7919u;
            respawn_particle(p, seed, params.time);
        }}

        // ═══════════════════════════════════════════════════════════════════
        // COLOR UPDATE (rainbow based on velocity)
        // ═══════════════════════════════════════════════════════════════════

        float speed = length(p.velocity);
        float hue = fmod(speed * 500.0 + params.time * 30.0 + float(pid) * 0.1, 360.0);
        float saturation = 0.8 + 0.2 * (p.life / p.max_life); // More saturated when young
        float brightness = 0.7 + 0.3 * speed; // Brighter when moving fast
        brightness = min(brightness, 1.0);

        p.color.rgb = hsv_to_rgb(hue, saturation, brightness);
        p.color.a = p.life / p.max_life; // Fade out as life decreases

        // Write back to global memory
        particles[pid] = p;

        // ═══════════════════════════════════════════════════════════════════
        // GEOMETRY GENERATION
        // ═══════════════════════════════════════════════════════════════════

        // Scale size based on life (shrink as dying)
        float life_scale = 0.5 + 0.5 * (p.life / p.max_life);
        float size = p.size * life_scale;

        // Also scale up slightly based on speed for motion blur effect
        float speed_scale = 1.0 + min(speed * 2.0, 1.0);
        size *= speed_scale;

        // Quad corners
        float x0 = p.position.x - size;
        float y0 = p.position.y - size;
        float x1 = p.position.x + size;
        float y1 = p.position.y + size;

        // Atomically allocate 6 vertices
        uint base = atomic_fetch_add_explicit(&vertex_count->count, 6, memory_order_relaxed);

        float4 color = p.color;

        // Triangle 1: TL -> BL -> BR
        vertices[base + 0] = ParticleVertex{{float2(x0, y0), float2(0, 0), color}};
        vertices[base + 1] = ParticleVertex{{float2(x0, y1), float2(0, 1), color}};
        vertices[base + 2] = ParticleVertex{{float2(x1, y1), float2(1, 1), color}};

        // Triangle 2: TL -> BR -> TR
        vertices[base + 3] = ParticleVertex{{float2(x0, y0), float2(0, 0), color}};
        vertices[base + 4] = ParticleVertex{{float2(x1, y1), float2(1, 1), color}};
        vertices[base + 5] = ParticleVertex{{float2(x1, y0), float2(1, 0), color}};
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

vertex VertexOut particle_vertex(
    const device ParticleVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    ParticleVertex v = vertices[vid];
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

fragment float4 particle_fragment(VertexOut in [[stage_in]]) {{
    // Radial gradient for soft particles
    float2 center = in.uv - 0.5;
    float dist = length(center) * 2.0;

    // Soft circle with glow
    float alpha = smoothstep(1.0, 0.3, dist);
    float glow = exp(-dist * 2.0) * 0.5;

    float3 color = in.color.rgb * (1.0 + glow);
    float final_alpha = (alpha + glow * 0.3) * in.color.a;

    return float4(color, final_alpha);
}}
"#,
        header = APP_SHADER_HEADER,
        particle_count = PARTICLE_COUNT,
        particles_per_thread = PARTICLES_PER_THREAD
    )
}

// ============================================================================
// ParticleSystem App
// ============================================================================

pub struct ParticleSystem {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // App-specific buffers
    params_buffer: Buffer,
    particles_buffer: Buffer,
    vertices_buffer: Buffer,
    vertex_count_buffer: Buffer,

    // Current params (updated each frame)
    current_params: ParticleParams,

    // Mouse state tracking
    mouse_down: bool,
    mouse_x: f32,
    mouse_y: f32,

    // Statistics
    total_time: f32,
}

impl ParticleSystem {
    /// Create a new Particle System app
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "ParticleSystem");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create pipelines
        let compute_pipeline = builder.create_compute_pipeline(&library, "particle_kernel")?;
        let render_pipeline =
            builder.create_render_pipeline(&library, "particle_vertex", "particle_fragment")?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<ParticleParams>());
        let particles_buffer = builder.create_buffer(PARTICLE_COUNT * mem::size_of::<Particle>());
        let vertices_buffer =
            builder.create_buffer(TOTAL_VERTICES * mem::size_of::<ParticleVertex>());
        let vertex_count_buffer = builder.create_buffer(mem::size_of::<VertexCount>());

        // Initialize particles to zero (they'll be initialized on first frame in shader)
        unsafe {
            let ptr = particles_buffer.contents() as *mut Particle;
            std::ptr::write_bytes(ptr, 0, PARTICLE_COUNT);
        }

        // Initialize vertex count
        unsafe {
            let ptr = vertex_count_buffer.contents() as *mut VertexCount;
            *ptr = VertexCount::default();
        }

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            particles_buffer,
            vertices_buffer,
            vertex_count_buffer,
            current_params: ParticleParams {
                attraction_strength: 3.0,
                gravity_strength: 0.3,
                ..Default::default()
            },
            mouse_down: false,
            mouse_x: 0.5,
            mouse_y: 0.5,
            total_time: 0.0,
        })
    }

    /// Get current vertex count from GPU buffer
    pub fn read_vertex_count(&self) -> u32 {
        unsafe {
            let ptr = self.vertex_count_buffer.contents() as *const VertexCount;
            (*ptr).count
        }
    }

    /// Get statistics string
    pub fn stats(&self) -> String {
        format!(
            "Particles: {} | Mouse: ({:.2}, {:.2}) | {}",
            PARTICLE_COUNT,
            self.mouse_x,
            self.mouse_y,
            if self.mouse_down {
                "ATTRACTING"
            } else {
                "idle"
            }
        )
    }

    /// Adjust attraction strength
    pub fn adjust_attraction(&mut self, delta: f32) {
        self.current_params.attraction_strength =
            (self.current_params.attraction_strength + delta).clamp(0.5, 10.0);
    }

    /// Adjust gravity strength
    pub fn adjust_gravity(&mut self, delta: f32) {
        self.current_params.gravity_strength =
            (self.current_params.gravity_strength + delta).clamp(0.0, 2.0);
    }

    /// Get attraction strength
    pub fn attraction_strength(&self) -> f32 {
        self.current_params.attraction_strength
    }

    /// Get gravity strength
    pub fn gravity_strength(&self) -> f32 {
        self.current_params.gravity_strength
    }
}

impl GpuApp for ParticleSystem {
    fn name(&self) -> &str {
        "Particle System"
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
        // Read from GPU buffer - this is the actual count set by compute shader
        self.read_vertex_count() as usize
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.particles_buffer,    // slot 3
            &self.vertices_buffer,     // slot 4
            &self.vertex_count_buffer, // slot 5
        ]
    }

    fn update_params(&mut self, frame_state: &FrameState, delta_time: f32) {
        self.total_time += delta_time;

        // Build params from current state
        self.current_params.mouse_x = self.mouse_x;
        self.current_params.mouse_y = self.mouse_y;
        self.current_params.delta_time = delta_time;
        self.current_params.mouse_down = if self.mouse_down { 1 } else { 0 };
        self.current_params.time = self.total_time;
        self.current_params.frame_number = frame_state.frame_number;

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut ParticleParams;
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
        // Could log stats here if needed
    }

    fn thread_count(&self) -> usize {
        THREAD_COUNT
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::HighThroughput  // Particle system benefits from frame overlap
    }

    fn clear_color(&self) -> MTLClearColor {
        // Dark blue-black background for contrast
        MTLClearColor::new(0.02, 0.02, 0.05, 1.0)
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
        assert_eq!(mem::size_of::<Particle>(), 48);
        assert_eq!(mem::size_of::<ParticleParams>(), 32);
        assert_eq!(mem::size_of::<ParticleVertex>(), 32);
        assert_eq!(mem::size_of::<VertexCount>(), 16);
    }

    #[test]
    fn test_particle_count() {
        assert!(PARTICLE_COUNT >= 10000, "Should have at least 10K particles");
        assert_eq!(
            PARTICLES_PER_THREAD * THREAD_COUNT >= PARTICLE_COUNT,
            true
        );
    }
}
