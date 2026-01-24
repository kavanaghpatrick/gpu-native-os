// Metaballs - GPU-Native App Implementation
//
// Beautiful organic blobby shapes computed entirely on the GPU.
// Uses implicit surface rendering: each pixel computes the sum of
// radius^2/distance^2 for each ball center.
//
// Implements the GpuApp framework for OS integration.

use super::app::{GpuApp, AppBuilder, PipelineMode, APP_SHADER_HEADER};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;
use std::mem;

// ============================================================================
// Constants
// ============================================================================

pub const NUM_BALLS: usize = 12;
pub const VERTICES_PER_QUAD: usize = 6;  // Fullscreen quad = 2 triangles

// ============================================================================
// Data Structures (match shader)
// ============================================================================

/// Per-frame parameters passed to the GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetaballParams {
    pub time: f32,
    pub mouse_x: f32,
    pub mouse_y: f32,
    pub num_balls: u32,
    pub mouse_attract: u32,  // 1 = attract balls to mouse
    pub threshold: f32,      // Field threshold for surface
    pub _padding: [f32; 2],
}

/// A single metaball
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Ball {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub radius: f32,
    pub hue: f32,           // HSV hue for coloring
    pub _padding: [f32; 2],
}

impl Default for Ball {
    fn default() -> Self {
        Self {
            position: [0.5, 0.5],
            velocity: [0.0, 0.0],
            radius: 0.08,
            hue: 0.0,
            _padding: [0.0, 0.0],
        }
    }
}

/// Vertex for fullscreen quad
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

struct MetaballParams {{
    float time;
    float mouse_x;
    float mouse_y;
    uint num_balls;
    uint mouse_attract;
    float threshold;
    float2 _padding;
}};

struct Ball {{
    float2 position;
    float2 velocity;
    float radius;
    float hue;
    float2 _padding;
}};

struct QuadVertex {{
    float2 position;
    float2 uv;
}};

// ============================================================================
// Compute Kernel - Update ball physics
// ============================================================================

kernel void metaballs_compute(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant MetaballParams& params [[buffer(2)]],
    device Ball* balls [[buffer(3)]],
    device QuadVertex* vertices [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {{
    // Only NUM_BALLS threads do ball physics
    if (tid < params.num_balls) {{
        Ball ball = balls[tid];

        float dt = 0.016; // ~60fps timestep

        // Apply velocity
        ball.position += ball.velocity * dt;

        // Mouse attraction (when enabled)
        if (params.mouse_attract != 0) {{
            float2 mouse = float2(params.mouse_x, params.mouse_y);
            float2 to_mouse = mouse - ball.position;
            float dist = length(to_mouse);
            if (dist > 0.01) {{
                float attract_strength = 2.0 / (dist * dist + 0.5);
                ball.velocity += normalize(to_mouse) * attract_strength * dt;
            }}
        }}

        // Add some gentle floating motion
        float phase = params.time * 0.5 + float(tid) * 0.7;
        ball.velocity.x += sin(phase) * 0.3 * dt;
        ball.velocity.y += cos(phase * 1.3) * 0.3 * dt;

        // Damping
        ball.velocity *= 0.99;

        // Bounce off edges with padding for the ball radius
        float margin = ball.radius * 0.5;

        if (ball.position.x < margin) {{
            ball.position.x = margin;
            ball.velocity.x = abs(ball.velocity.x) * 0.8;
        }}
        if (ball.position.x > 1.0 - margin) {{
            ball.position.x = 1.0 - margin;
            ball.velocity.x = -abs(ball.velocity.x) * 0.8;
        }}
        if (ball.position.y < margin) {{
            ball.position.y = margin;
            ball.velocity.y = abs(ball.velocity.y) * 0.8;
        }}
        if (ball.position.y > 1.0 - margin) {{
            ball.position.y = 1.0 - margin;
            ball.velocity.y = -abs(ball.velocity.y) * 0.8;
        }}

        // Speed limit
        float speed = length(ball.velocity);
        if (speed > 1.5) {{
            ball.velocity = normalize(ball.velocity) * 1.5;
        }}

        balls[tid] = ball;
    }}

    // Thread 0 generates the fullscreen quad vertices
    if (tid == 0) {{
        // Triangle 1: TL -> BL -> BR
        vertices[0] = QuadVertex{{float2(0.0, 0.0), float2(0.0, 0.0)}};
        vertices[1] = QuadVertex{{float2(0.0, 1.0), float2(0.0, 1.0)}};
        vertices[2] = QuadVertex{{float2(1.0, 1.0), float2(1.0, 1.0)}};

        // Triangle 2: TL -> BR -> TR
        vertices[3] = QuadVertex{{float2(0.0, 0.0), float2(0.0, 0.0)}};
        vertices[4] = QuadVertex{{float2(1.0, 1.0), float2(1.0, 1.0)}};
        vertices[5] = QuadVertex{{float2(1.0, 0.0), float2(1.0, 0.0)}};
    }}
}}

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexOut {{
    float4 position [[position]];
    float2 uv;
}};

vertex VertexOut metaballs_vertex(
    const device QuadVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {{
    QuadVertex v = vertices[vid];
    VertexOut out;
    // Convert [0,1] to clip space [-1,1], flip Y
    out.position = float4(v.position.x * 2.0 - 1.0, -(v.position.y * 2.0 - 1.0), 0.0, 1.0);
    out.uv = v.uv;
    return out;
}}

// ============================================================================
// Fragment Shader - Per-pixel metaball field computation
// ============================================================================

fragment float4 metaballs_fragment(
    VertexOut in [[stage_in]],
    constant MetaballParams& params [[buffer(0)]],
    constant Ball* balls [[buffer(1)]]
) {{
    float2 uv = in.uv;

    // Compute metaball field: sum of (radius^2 / distance^2) for each ball
    float field = 0.0;
    float3 weighted_color = float3(0.0);
    float total_weight = 0.0;

    for (uint i = 0; i < params.num_balls; i++) {{
        Ball ball = balls[i];
        float2 diff = uv - ball.position;
        float dist_sq = dot(diff, diff);

        // Avoid division by zero
        float contribution = (ball.radius * ball.radius) / max(dist_sq, 0.0001);
        field += contribution;

        // Weight color by contribution for smooth blending
        float3 ball_color = hsv_to_rgb(ball.hue * 360.0, 0.8, 1.0);
        weighted_color += ball_color * contribution;
        total_weight += contribution;
    }}

    // Normalize weighted color
    float3 base_color = weighted_color / max(total_weight, 0.001);

    // Threshold for surface
    float threshold = params.threshold;

    if (field > threshold) {{
        // Inside the blob
        // Create gradient based on field strength
        float intensity = smoothstep(threshold, threshold * 3.0, field);

        // Add some depth/glow effect
        float glow = smoothstep(threshold, threshold * 1.5, field);

        // Darken edges, brighten center
        float3 color = base_color * (0.6 + 0.4 * intensity);

        // Add white highlight for 3D effect
        float highlight = smoothstep(threshold * 2.0, threshold * 4.0, field);
        color = mix(color, float3(1.0), highlight * 0.3);

        // Add rim lighting effect
        float rim = smoothstep(threshold * 1.2, threshold, field);
        color += float3(0.2, 0.3, 0.4) * rim;

        return float4(color, 1.0);
    }} else {{
        // Outside - subtle glow
        float glow = smoothstep(0.0, threshold, field);
        float3 bg_color = float3(0.02, 0.02, 0.04);
        float3 glow_color = base_color * glow * 0.3;

        return float4(bg_color + glow_color, 1.0);
    }}
}}
"#, header = APP_SHADER_HEADER)
}

// ============================================================================
// Metaballs App
// ============================================================================

pub struct Metaballs {
    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    params_buffer: Buffer,
    balls_buffer: Buffer,
    vertices_buffer: Buffer,

    // Current params
    current_params: MetaballParams,

    // Input state
    mouse_down: bool,
}

impl Metaballs {
    /// Create a new Metaballs app
    pub fn new(device: &Device) -> Result<Self, String> {
        let builder = AppBuilder::new(device, "Metaballs");

        // Compile shaders
        let source = shader_source();
        let library = builder.compile_library(&source)?;

        // Create compute pipeline
        let compute_pipeline = builder.create_compute_pipeline(&library, "metaballs_compute")?;

        // Create render pipeline (need custom one with fragment buffer access)
        let render_pipeline = Self::create_render_pipeline(device, &library)?;

        // Create buffers
        let params_buffer = builder.create_buffer(mem::size_of::<MetaballParams>());
        let balls_buffer = builder.create_buffer(NUM_BALLS * mem::size_of::<Ball>());
        let vertices_buffer = builder.create_buffer(VERTICES_PER_QUAD * mem::size_of::<QuadVertex>());

        // Initialize balls with random positions and velocities
        Self::initialize_balls(&balls_buffer);

        // Initialize params
        let current_params = MetaballParams {
            time: 0.0,
            mouse_x: 0.5,
            mouse_y: 0.5,
            num_balls: NUM_BALLS as u32,
            mouse_attract: 0,
            threshold: 1.0,
            _padding: [0.0, 0.0],
        };

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            params_buffer,
            balls_buffer,
            vertices_buffer,
            current_params,
            mouse_down: false,
        })
    }

    fn create_render_pipeline(device: &Device, library: &Library) -> Result<RenderPipelineState, String> {
        let vertex_function = library
            .get_function("metaballs_vertex", None)
            .map_err(|e| format!("Failed to get metaballs_vertex: {}", e))?;

        let fragment_function = library
            .get_function("metaballs_fragment", None)
            .map_err(|e| format!("Failed to get metaballs_fragment: {}", e))?;

        let desc = RenderPipelineDescriptor::new();
        desc.set_vertex_function(Some(&vertex_function));
        desc.set_fragment_function(Some(&fragment_function));

        let attachment = desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        device
            .new_render_pipeline_state(&desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))
    }

    fn initialize_balls(buffer: &Buffer) {
        unsafe {
            let ptr = buffer.contents() as *mut Ball;

            // Create balls with varied sizes and colors
            for i in 0..NUM_BALLS {
                // Use simple pseudo-random positioning based on index
                let seed = i as f32;
                let angle = seed * 2.39996; // Golden angle
                let r = 0.2 + (seed * 0.31415).sin().abs() * 0.3;

                let ball = Ball {
                    position: [
                        0.5 + r * angle.cos(),
                        0.5 + r * angle.sin(),
                    ],
                    velocity: [
                        (seed * 1.7).sin() * 0.3,
                        (seed * 2.3).cos() * 0.3,
                    ],
                    radius: 0.06 + (seed * 0.5).sin().abs() * 0.06,
                    hue: (i as f32) / (NUM_BALLS as f32), // Spread hues evenly
                    _padding: [0.0, 0.0],
                };

                *ptr.add(i) = ball;
            }
        }
    }

    /// Get the balls buffer (for fragment shader binding)
    pub fn balls_buffer(&self) -> &Buffer {
        &self.balls_buffer
    }

    /// Toggle mouse attraction
    pub fn toggle_attract(&mut self) {
        self.current_params.mouse_attract = if self.current_params.mouse_attract != 0 { 0 } else { 1 };
    }

    /// Reset balls to initial state
    pub fn reset(&self) {
        Self::initialize_balls(&self.balls_buffer);
    }

    /// Adjust threshold
    pub fn adjust_threshold(&mut self, delta: f32) {
        self.current_params.threshold = (self.current_params.threshold + delta).clamp(0.5, 2.5);
    }
}

impl GpuApp for Metaballs {
    fn name(&self) -> &str {
        "Metaballs"
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
        vec![
            &self.balls_buffer,    // slot 3
            &self.vertices_buffer, // slot 4
        ]
    }

    fn update_params(&mut self, frame_state: &FrameState, _delta_time: f32) {
        self.current_params.time = frame_state.time;
        self.current_params.mouse_x = frame_state.cursor_x;
        self.current_params.mouse_y = frame_state.cursor_y;

        // Enable attraction when mouse is down
        if self.mouse_down {
            self.current_params.mouse_attract = 1;
        }

        // Write to buffer
        unsafe {
            let ptr = self.params_buffer.contents() as *mut MetaballParams;
            *ptr = self.current_params;
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            t if t == InputEventType::MouseDown as u16 => {
                self.mouse_down = true;
            }
            t if t == InputEventType::MouseUp as u16 => {
                self.mouse_down = false;
                // Keep attraction enabled if it was toggled on
            }
            _ => {}
        }
    }

    fn post_frame(&mut self, _timing: &FrameTiming) {
        // Could log stats here
    }

    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(0.02, 0.02, 0.04, 1.0)
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::HighThroughput  // Metaballs animation benefits from frame overlap
    }

    fn thread_count(&self) -> usize {
        // Only need NUM_BALLS threads for physics, but use 32 for alignment
        32
    }
}

// ============================================================================
// Custom render method for fragment buffer binding
// ============================================================================

impl Metaballs {
    /// Custom render that binds the params and balls buffers to fragment shader
    pub fn render_with_buffers(&self, encoder: &RenderCommandEncoderRef) {
        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);

        // Bind buffers to fragment shader
        encoder.set_fragment_buffer(0, Some(&self.params_buffer), 0);
        encoder.set_fragment_buffer(1, Some(&self.balls_buffer), 0);

        encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            VERTICES_PER_QUAD as u64,
        );
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
        assert_eq!(mem::size_of::<MetaballParams>(), 32);
        assert_eq!(mem::size_of::<Ball>(), 32);
        assert_eq!(mem::size_of::<QuadVertex>(), 16);
    }

    #[test]
    fn test_constants() {
        assert!(NUM_BALLS >= 8 && NUM_BALLS <= 16);
        assert_eq!(VERTICES_PER_QUAD, 6);
    }
}
