// Ball Physics Demo - 1024 balls simulated entirely on GPU
//
// Each of 1024 threads owns exactly one ball. All physics (collision detection,
// response, integration) happens on the GPU in a single compute dispatch.

use metal::*;
use std::ffi::c_void;
use std::mem;

// Constants
pub const NUM_BALLS: usize = 1024;
pub const VERTICES_PER_BALL: usize = 6; // 2 triangles = 6 vertices
pub const TOTAL_VERTICES: usize = NUM_BALLS * VERTICES_PER_BALL;

// Action codes for user input
pub const ACTION_NONE: u32 = 0;
pub const ACTION_RESET: u32 = 1;
pub const ACTION_SCATTER: u32 = 2;
pub const ACTION_IMPULSE: u32 = 3;

// Ball state - 32 bytes per ball
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Ball {
    pub position: [f32; 2],  // 8 bytes - center x, y (normalized 0-1)
    pub velocity: [f32; 2],  // 8 bytes - velocity x, y (units/second)
    pub radius: f32,         // 4 bytes - ball radius (normalized)
    pub color: u32,          // 4 bytes - packed RGBA8
    pub mass: f32,           // 4 bytes - for collision response
    pub flags: u32,          // 4 bytes - state flags
}

impl Ball {
    pub const SIZE: usize = 32;
}

// Physics parameters - matches shader
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PhysicsParams {
    pub gravity: [f32; 2],   // 8 bytes - gravity vector
    pub bounce: f32,         // 4 bytes - coefficient of restitution
    pub friction: f32,       // 4 bytes - velocity damping per frame
    pub dt: f32,             // 4 bytes - timestep
    pub ball_radius: f32,    // 4 bytes - default ball radius
    pub num_balls: u32,      // 4 bytes
    pub _padding: u32,       // 4 bytes - align to 32 bytes
}

impl PhysicsParams {
    pub const SIZE: usize = 32;
}

// Input state from CPU
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InputState {
    pub gravity_x: f32,      // 4 bytes - gravity direction (from tilt/keys)
    pub gravity_y: f32,      // 4 bytes
    pub cursor_x: f32,       // 4 bytes - cursor position (normalized 0-1)
    pub cursor_y: f32,       // 4 bytes
    pub mouse_down: u32,     // 4 bytes - mouse button state
    pub mouse_clicked: u32,  // 4 bytes - click event
    pub pending_action: u32, // 4 bytes - action code
    pub impulse_strength: f32, // 4 bytes - impulse magnitude
}

impl InputState {
    pub const SIZE: usize = 32;
}

// Vertex format for rendering - 32 bytes per vertex
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BallVertex {
    pub position: [f32; 2],    // 8 bytes - screen position
    pub uv: [f32; 2],          // 8 bytes - texture coords for SDF
    pub color: u32,            // 4 bytes - packed RGBA8
    pub ball_center: [f32; 2], // 8 bytes - for fragment shader SDF
    pub radius: f32,           // 4 bytes - for fragment shader SDF
}

impl BallVertex {
    pub const SIZE: usize = 32;
}

// Draw arguments for indirect rendering
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DrawArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

// Metal shader source
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Constants
constant uint NUM_BALLS = 1024;
constant uint VERTICES_PER_BALL = 6;

// Ball state structure
struct Ball {
    float2 position;     // center x, y (normalized 0-1)
    float2 velocity;     // velocity x, y
    float radius;        // ball radius
    uint color;          // packed RGBA8
    float mass;          // for collision response
    uint flags;          // state flags
};

// Physics parameters
struct PhysicsParams {
    float2 gravity;      // gravity vector
    float bounce;        // coefficient of restitution
    float friction;      // velocity damping
    float dt;            // timestep
    float ball_radius;   // default radius
    uint num_balls;
    uint _padding;
};

// Input state
struct InputState {
    float gravity_x;
    float gravity_y;
    float cursor_x;
    float cursor_y;
    uint mouse_down;
    uint mouse_clicked;
    uint pending_action;
    float impulse_strength;
};

// Vertex output
struct BallVertex {
    float2 position;
    float2 uv;
    uint color;
    float2 ball_center;
    float radius;
};

// Draw arguments
struct DrawArgs {
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
};

// Action codes
constant uint ACTION_NONE = 0;
constant uint ACTION_RESET = 1;
constant uint ACTION_SCATTER = 2;
constant uint ACTION_IMPULSE = 3;

// Random number generator (simple LCG)
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

float random_float(uint seed) {
    return float(hash(seed)) / float(0xFFFFFFFFu);
}

// HSV to RGB conversion
float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float hp = h / 60.0;
    float x = c * (1.0 - abs(fmod(hp, 2.0) - 1.0));
    float3 rgb;
    if (hp < 1) rgb = float3(c, x, 0);
    else if (hp < 2) rgb = float3(x, c, 0);
    else if (hp < 3) rgb = float3(0, c, x);
    else if (hp < 4) rgb = float3(0, x, c);
    else if (hp < 5) rgb = float3(x, 0, c);
    else rgb = float3(c, 0, x);
    float m = v - c;
    return rgb + m;
}

// Pack RGB to RGBA8
uint pack_color(float3 rgb) {
    uint r = uint(clamp(rgb.r, 0.0, 1.0) * 255.0);
    uint g = uint(clamp(rgb.g, 0.0, 1.0) * 255.0);
    uint b = uint(clamp(rgb.b, 0.0, 1.0) * 255.0);
    return (255u << 24) | (b << 16) | (g << 8) | r;  // ABGR for Metal
}

// Main physics kernel
kernel void ball_physics_kernel(
    device Ball* balls [[buffer(0)]],
    constant PhysicsParams& params [[buffer(1)]],
    constant InputState& input [[buffer(2)]],
    device BallVertex* vertices [[buffer(3)]],
    device atomic_uint* draw_count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Thread-local state
    Ball my_ball = balls[tid];
    float2 my_pos = my_ball.position;
    float2 my_vel = my_ball.velocity;
    float my_radius = my_ball.radius;
    float my_mass = my_ball.mass;

    // Threadgroup shared gravity
    threadgroup float2 tg_gravity;
    threadgroup uint tg_action;
    threadgroup float2 tg_cursor;
    threadgroup float tg_impulse;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: INPUT - Read input state
    // ═══════════════════════════════════════════════════════════════
    if (tid == 0) {
        tg_gravity = float2(input.gravity_x, input.gravity_y);
        tg_action = input.pending_action;
        tg_cursor = float2(input.cursor_x, input.cursor_y);
        tg_impulse = input.impulse_strength;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: PROCESS ACTIONS
    // ═══════════════════════════════════════════════════════════════
    if (tg_action == ACTION_RESET) {
        // Reset to grid pattern
        uint row = tid / 32;
        uint col = tid % 32;
        my_pos = float2(
            0.1 + (col / 32.0) * 0.8,
            0.1 + (row / 32.0) * 0.8
        );
        my_vel = float2(0.0);

        // Rainbow color by position
        float hue = float(tid) / float(NUM_BALLS) * 360.0;
        my_ball.color = pack_color(hsv_to_rgb(hue, 0.8, 0.95));
    }
    else if (tg_action == ACTION_SCATTER) {
        // Random scatter with random velocities
        uint seed = tid * 12345u + hash(uint(my_pos.x * 1000.0));
        my_pos = float2(
            0.1 + random_float(seed) * 0.8,
            0.1 + random_float(seed + 1u) * 0.8
        );
        my_vel = float2(
            (random_float(seed + 2u) - 0.5) * 0.5,
            (random_float(seed + 3u) - 0.5) * 0.5
        );
    }
    else if (tg_action == ACTION_IMPULSE && input.mouse_clicked != 0) {
        // Apply radial impulse from cursor
        float2 dir = my_pos - tg_cursor;
        float dist = length(dir);
        if (dist > 0.001 && dist < 0.3) {
            float2 impulse = normalize(dir) * tg_impulse * (0.3 - dist) / 0.3;
            my_vel += impulse;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: COLLISION DETECTION & RESPONSE
    // ═══════════════════════════════════════════════════════════════

    // Ball-ball collisions (O(n²) for simplicity - GPU can handle it)
    float2 total_impulse = float2(0.0);
    float2 total_separation = float2(0.0);

    for (uint other = 0; other < NUM_BALLS; other++) {
        if (other == tid) continue;

        Ball other_ball = balls[other];
        float2 diff = my_pos - other_ball.position;
        float dist_sq = dot(diff, diff);
        float min_dist = my_radius + other_ball.radius;

        if (dist_sq < min_dist * min_dist && dist_sq > 0.0001) {
            float dist = sqrt(dist_sq);
            float2 normal = diff / dist;
            float overlap = min_dist - dist;

            // Collision response (elastic)
            float2 rel_vel = my_vel - other_ball.velocity;
            float vel_along_normal = dot(rel_vel, normal);

            if (vel_along_normal < 0) {
                // Balls approaching - apply impulse
                float e = params.bounce;
                float j = -(1.0 + e) * vel_along_normal;
                j /= (1.0 / my_mass) + (1.0 / other_ball.mass);

                total_impulse += (j / my_mass) * normal;
            }

            // Position separation (push balls apart)
            total_separation += normal * (overlap * 0.5);
        }
    }

    // Apply collision results
    my_vel += total_impulse;
    my_pos += total_separation;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: WALL COLLISIONS
    // ═══════════════════════════════════════════════════════════════

    // Left wall
    if (my_pos.x - my_radius < 0.0) {
        my_pos.x = my_radius;
        my_vel.x = abs(my_vel.x) * params.bounce;
    }
    // Right wall
    if (my_pos.x + my_radius > 1.0) {
        my_pos.x = 1.0 - my_radius;
        my_vel.x = -abs(my_vel.x) * params.bounce;
    }
    // Top wall
    if (my_pos.y - my_radius < 0.0) {
        my_pos.y = my_radius;
        my_vel.y = abs(my_vel.y) * params.bounce;
    }
    // Bottom wall
    if (my_pos.y + my_radius > 1.0) {
        my_pos.y = 1.0 - my_radius;
        my_vel.y = -abs(my_vel.y) * params.bounce;
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 5: INTEGRATION
    // ═══════════════════════════════════════════════════════════════

    // Apply gravity
    my_vel += tg_gravity * params.dt;

    // Apply friction (air resistance)
    my_vel *= params.friction;

    // Clamp velocity to prevent instability
    float speed = length(my_vel);
    if (speed > 2.0) {
        my_vel = normalize(my_vel) * 2.0;
    }

    // Update position
    my_pos += my_vel * params.dt;

    // Write back to device memory
    my_ball.position = my_pos;
    my_ball.velocity = my_vel;
    balls[tid] = my_ball;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 6: VERTEX GENERATION
    // ═══════════════════════════════════════════════════════════════

    // Convert to clip space (-1 to 1)
    float2 clip_pos = my_pos * 2.0 - 1.0;
    clip_pos.y = -clip_pos.y;  // Flip Y for Metal
    float clip_radius = my_radius * 2.0;

    // Quad corners
    float2 corners[4] = {
        clip_pos + float2(-clip_radius, -clip_radius),  // TL
        clip_pos + float2( clip_radius, -clip_radius),  // TR
        clip_pos + float2( clip_radius,  clip_radius),  // BR
        clip_pos + float2(-clip_radius,  clip_radius),  // BL
    };

    float2 uvs[4] = {
        float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 1)
    };

    // Triangle indices (CCW winding)
    int indices[6] = {0, 3, 2, 0, 2, 1};

    uint base = tid * VERTICES_PER_BALL;
    for (int i = 0; i < 6; i++) {
        int idx = indices[i];
        vertices[base + i].position = corners[idx];
        vertices[base + i].uv = uvs[idx];
        vertices[base + i].color = my_ball.color;
        vertices[base + i].ball_center = clip_pos;
        vertices[base + i].radius = clip_radius;
    }

    // Thread 0 updates draw count
    if (tid == 0) {
        atomic_store_explicit(draw_count, NUM_BALLS * VERTICES_PER_BALL, memory_order_relaxed);
    }
}

// ═══════════════════════════════════════════════════════════════════
// RENDER PIPELINE
// ═══════════════════════════════════════════════════════════════════

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
    float2 ball_center;
    float radius;
};

vertex VertexOut ball_vertex_shader(
    const device BallVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    BallVertex v = vertices[vid];

    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;

    // Unpack color (ABGR)
    out.color = float4(
        float((v.color >>  0) & 0xFF) / 255.0,
        float((v.color >>  8) & 0xFF) / 255.0,
        float((v.color >> 16) & 0xFF) / 255.0,
        float((v.color >> 24) & 0xFF) / 255.0
    );

    out.ball_center = v.ball_center;
    out.radius = v.radius;

    return out;
}

fragment float4 ball_fragment_shader(
    VertexOut in [[stage_in]]
) {
    // SDF circle rendering
    float2 uv = in.uv * 2.0 - 1.0;  // Map to -1..1
    float dist = length(uv);

    // Smooth edge with anti-aliasing
    float edge_width = fwidth(dist) * 1.5;
    float alpha = 1.0 - smoothstep(1.0 - edge_width, 1.0, dist);

    // Discard outside circle
    if (alpha < 0.01) discard_fragment();

    // Apply color with depth shading
    float4 color = in.color;
    color.rgb *= (1.0 - dist * 0.3);  // Slight radial gradient
    color.a *= alpha;

    return color;
}
"#;

pub struct BallPhysics {
    pub compute_pipeline: ComputePipelineState,
    pub render_pipeline: RenderPipelineState,
    pub balls_buffer: Buffer,
    pub params_buffer: Buffer,
    pub input_buffer: Buffer,
    pub vertices_buffer: Buffer,
    pub draw_args_buffer: Buffer,
    pub command_queue: CommandQueue,
}

impl BallPhysics {
    pub fn new(device: &Device) -> Result<Self, String> {
        // Compile shaders
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &compile_options)
            .map_err(|e| format!("Shader compile error: {}", e))?;

        // Create compute pipeline
        let compute_fn = library
            .get_function("ball_physics_kernel", None)
            .map_err(|e| format!("Failed to get compute function: {}", e))?;
        let compute_pipeline = device
            .new_compute_pipeline_state_with_function(&compute_fn)
            .map_err(|e| format!("Failed to create compute pipeline: {}", e))?;

        // Create render pipeline
        let vertex_fn = library
            .get_function("ball_vertex_shader", None)
            .map_err(|e| format!("Failed to get vertex function: {}", e))?;
        let fragment_fn = library
            .get_function("ball_fragment_shader", None)
            .map_err(|e| format!("Failed to get fragment function: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        // Enable blending for smooth edges
        let attachment = render_desc.color_attachments().object_at(0).unwrap();
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        // Initialize ball data
        let mut balls = vec![Ball {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            radius: 0.0,
            color: 0,
            mass: 0.0,
            flags: 0,
        }; NUM_BALLS];

        // Initialize balls in a grid with rainbow colors
        for i in 0..NUM_BALLS {
            let row = i / 32;
            let col = i % 32;
            balls[i].position = [
                0.1 + (col as f32 / 32.0) * 0.8,
                0.1 + (row as f32 / 32.0) * 0.8,
            ];
            balls[i].velocity = [0.0, 0.0];
            balls[i].radius = 0.008;  // Small balls
            balls[i].mass = 1.0;
            balls[i].flags = 0;

            // Rainbow color
            let hue = (i as f32 / NUM_BALLS as f32) * 360.0;
            balls[i].color = hsv_to_rgba8(hue, 0.8, 0.95);
        }

        let balls_buffer = device.new_buffer_with_data(
            balls.as_ptr() as *const c_void,
            (NUM_BALLS * Ball::SIZE) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize physics params
        let params = PhysicsParams {
            gravity: [0.0, 0.5],  // Default gravity pointing down
            bounce: 0.85,
            friction: 0.995,
            dt: 1.0 / 120.0,
            ball_radius: 0.008,
            num_balls: NUM_BALLS as u32,
            _padding: 0,
        };

        let params_buffer = device.new_buffer_with_data(
            &params as *const _ as *const c_void,
            PhysicsParams::SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize input state
        let input = InputState {
            gravity_x: 0.0,
            gravity_y: 0.5,
            cursor_x: 0.5,
            cursor_y: 0.5,
            mouse_down: 0,
            mouse_clicked: 0,
            pending_action: ACTION_NONE,
            impulse_strength: 0.5,
        };

        let input_buffer = device.new_buffer_with_data(
            &input as *const _ as *const c_void,
            InputState::SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Vertex buffer
        let vertices_buffer = device.new_buffer(
            (TOTAL_VERTICES * BallVertex::SIZE) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Draw arguments buffer
        let draw_args = DrawArgs {
            vertex_count: TOTAL_VERTICES as u32,
            instance_count: 1,
            vertex_start: 0,
            base_instance: 0,
        };

        let draw_args_buffer = device.new_buffer_with_data(
            &draw_args as *const _ as *const c_void,
            mem::size_of::<DrawArgs>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_queue = device.new_command_queue();

        Ok(Self {
            compute_pipeline,
            render_pipeline,
            balls_buffer,
            params_buffer,
            input_buffer,
            vertices_buffer,
            draw_args_buffer,
            command_queue,
        })
    }

    pub fn input_state_mut(&self) -> &mut InputState {
        unsafe { &mut *(self.input_buffer.contents() as *mut InputState) }
    }

    pub fn update_gravity(&self, x: f32, y: f32) {
        let input = self.input_state_mut();
        input.gravity_x = x;
        input.gravity_y = y;
    }

    pub fn update_cursor(&self, x: f32, y: f32) {
        let input = self.input_state_mut();
        input.cursor_x = x;
        input.cursor_y = y;
    }

    pub fn set_action(&self, action: u32) {
        let input = self.input_state_mut();
        input.pending_action = action;
    }

    pub fn set_mouse_state(&self, down: bool, clicked: bool) {
        let input = self.input_state_mut();
        input.mouse_down = if down { 1 } else { 0 };
        input.mouse_clicked = if clicked { 1 } else { 0 };
    }

    pub fn render(&self, drawable: &MetalDrawableRef) {
        let command_buffer = self.command_queue.new_command_buffer();

        // Compute pass - physics simulation
        let compute_encoder = command_buffer.new_compute_command_encoder();
        compute_encoder.set_compute_pipeline_state(&self.compute_pipeline);
        compute_encoder.set_buffer(0, Some(&self.balls_buffer), 0);
        compute_encoder.set_buffer(1, Some(&self.params_buffer), 0);
        compute_encoder.set_buffer(2, Some(&self.input_buffer), 0);
        compute_encoder.set_buffer(3, Some(&self.vertices_buffer), 0);
        compute_encoder.set_buffer(4, Some(&self.draw_args_buffer), 0);

        // Dispatch 1024 threads in single threadgroup
        compute_encoder.dispatch_threads(
            MTLSize::new(NUM_BALLS as u64, 1, 1),
            MTLSize::new(NUM_BALLS as u64, 1, 1),
        );
        compute_encoder.end_encoding();

        // Clear pending action after dispatch
        let input = self.input_state_mut();
        input.pending_action = ACTION_NONE;
        input.mouse_clicked = 0;

        // Render pass
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        color_attachment.set_clear_color(MTLClearColor::new(0.05, 0.05, 0.1, 1.0));

        let render_encoder = command_buffer.new_render_command_encoder(&render_desc);
        render_encoder.set_render_pipeline_state(&self.render_pipeline);
        render_encoder.set_vertex_buffer(0, Some(&self.vertices_buffer), 0);
        render_encoder.draw_primitives(
            MTLPrimitiveType::Triangle,
            0,
            TOTAL_VERTICES as u64,
        );
        render_encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}

// Helper: HSV to packed RGBA8
fn hsv_to_rgba8(h: f32, s: f32, v: f32) -> u32 {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());

    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = v - c;
    let r = ((r + m) * 255.0) as u32;
    let g = ((g + m) * 255.0) as u32;
    let b = ((b + m) * 255.0) as u32;

    // ABGR format for Metal
    (255 << 24) | (b << 16) | (g << 8) | r
}
