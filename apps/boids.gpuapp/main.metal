// Boids Flocking Simulation
//
// 1024 boids exhibiting emergent flocking behavior.
// Each boid follows three simple rules: separation, alignment, cohesion.
//
// Mouse interaction: move to repel, click to attract.

// App-Specific Structures

struct Boid {
    float2 position;
    float2 velocity;
};

struct BoidParams {
    float delta_time;
    float time;
    float mouse_x;
    float mouse_y;
    uint frame_number;
    uint mouse_down;
    float separation_weight;
    float alignment_weight;
    float cohesion_weight;
    float mouse_weight;
    float visual_range;
    float separation_dist;
    float max_speed;
    float _pad;
};

struct BoidVertex {
    float2 position;
    float2 uv;
    float4 color;
};

// Helper Functions

float2 toroidal_delta(float2 from, float2 to) {
    float2 delta = to - from;
    if (delta.x > 0.5) delta.x -= 1.0;
    if (delta.x < -0.5) delta.x += 1.0;
    if (delta.y > 0.5) delta.y -= 1.0;
    if (delta.y < -0.5) delta.y += 1.0;
    return delta;
}

float toroidal_distance(float2 a, float2 b) {
    float2 delta = toroidal_delta(a, b);
    return length(delta);
}

float2 limit_magnitude(float2 v, float max_mag) {
    float mag = length(v);
    if (mag > max_mag && mag > 0.0001) {
        return v * (max_mag / mag);
    }
    return v;
}

// Main Compute Kernel
kernel void boids_kernel(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant BoidParams& params [[buffer(2)]],
    device BoidVertex* vertices [[buffer(3)]],
    device Boid* boids [[buffer(4)]],
    device atomic_uint* vertex_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Initialize on first frame
    if (params.frame_number == 0) {
        uint seed = tid * 12345u + 67890u;
        float px = random_float(hash(seed));
        float py = random_float(hash(seed + 1));
        float angle = random_float(hash(seed + 2)) * 6.28318;
        float speed = 0.1 + random_float(hash(seed + 3)) * 0.1;

        boids[tid].position = float2(px, py);
        boids[tid].velocity = float2(cos(angle), sin(angle)) * speed;
    }

    // This boid's current state
    Boid my_boid = boids[tid];
    float2 my_pos = my_boid.position;
    float2 my_vel = my_boid.velocity;

    // Threadgroup memory for efficient neighbor queries
    threadgroup float2 tg_positions[1024];
    threadgroup float2 tg_velocities[1024];

    tg_positions[tid] = my_pos;
    tg_velocities[tid] = my_vel;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // PHASE 1: NEIGHBOR ANALYSIS
    float2 separation = float2(0.0);
    float2 alignment = float2(0.0);
    float2 cohesion_center = float2(0.0);
    uint neighbor_count = 0;
    uint close_count = 0;

    // Default parameters (params buffer layout may differ)
    float visual_range = 0.08;
    float separation_dist = 0.025;
    float separation_weight = 0.05;
    float alignment_weight = 0.05;
    float cohesion_weight = 0.005;
    float mouse_weight = 0.0003;
    float max_speed = 0.4;

    for (uint i = 0; i < 1024; i++) {
        if (i == tid) continue;

        float2 other_pos = tg_positions[i];
        float2 other_vel = tg_velocities[i];
        float dist = toroidal_distance(my_pos, other_pos);

        if (dist < visual_range) {
            float2 delta = toroidal_delta(my_pos, other_pos);
            cohesion_center += delta;
            alignment += other_vel;
            neighbor_count++;

            if (dist < separation_dist && dist > 0.0001) {
                separation -= delta * (1.0 - dist / separation_dist);
                close_count++;
            }
        }
    }

    // PHASE 2: APPLY BOID RULES
    float2 acceleration = float2(0.0);

    if (neighbor_count > 0) {
        float2 cohesion_dir = cohesion_center / float(neighbor_count);
        acceleration += cohesion_dir * cohesion_weight;

        float2 avg_vel = alignment / float(neighbor_count);
        float2 alignment_steer = avg_vel - my_vel;
        acceleration += alignment_steer * alignment_weight;
    }

    if (close_count > 0) {
        acceleration += separation * separation_weight;
    }

    // PHASE 3: MOUSE INTERACTION
    float2 mouse_pos = float2(frame.cursor_x, frame.cursor_y);
    float2 to_mouse = toroidal_delta(my_pos, mouse_pos);
    float mouse_dist = length(to_mouse);

    if (mouse_dist > 0.01 && mouse_dist < 0.3) {
        float mouse_strength = mouse_weight / (mouse_dist * mouse_dist + 0.001);

        if (frame.modifiers != 0) {
            // Attract when mouse button pressed
            acceleration += to_mouse * mouse_strength * 50.0;
        } else {
            // Gentle avoidance
            acceleration -= to_mouse * mouse_strength * 5.0;
        }
    }

    // PHASE 4: UPDATE VELOCITY AND POSITION
    my_vel += acceleration;
    my_vel = limit_magnitude(my_vel, max_speed);

    float speed = length(my_vel);
    if (speed < 0.05) {
        uint seed = tid + uint(frame.time * 1000.0);
        float angle = random_float(seed) * 6.28318;
        my_vel = float2(cos(angle), sin(angle)) * 0.1;
    }

    my_pos += my_vel * params.delta_time;
    my_pos = fract(my_pos + 1.0);

    boids[tid].position = my_pos;
    boids[tid].velocity = my_vel;

    // PHASE 5: GENERATE TRIANGLE GEOMETRY
    float angle = atan2(my_vel.y, my_vel.x);
    float base_size = 0.008;
    float size = base_size * (0.7 + 0.3 * (speed / max_speed));

    float2 forward = float2(cos(angle), sin(angle));
    float2 right = float2(-forward.y, forward.x);

    float2 tip = my_pos + forward * size * 2.0;
    float2 left_wing = my_pos - forward * size * 0.5 + right * size;
    float2 right_wing = my_pos - forward * size * 0.5 - right * size;

    // Color based on heading (rainbow)
    float density = float(neighbor_count) / 50.0;
    float speed_ratio = speed / max_speed;
    float hue = fmod((angle / 6.28318) * 360.0 + 180.0, 360.0);
    float saturation = 0.6 + density * 0.4;
    float value = 0.7 + speed_ratio * 0.3;
    float3 color = hsv_to_rgb(hue, saturation, value);

    float shimmer = sin(frame.time * 3.0 + float(tid) * 0.1) * 0.05;
    color = clamp(color + shimmer, 0.0, 1.0);

    uint base = tid * 3;

    vertices[base + 0].position = tip * 2.0 - 1.0;
    vertices[base + 0].position.y *= -1.0;
    vertices[base + 0].uv = float2(0.5, 0.0);
    vertices[base + 0].color = float4(color * 1.2, 1.0);

    vertices[base + 1].position = left_wing * 2.0 - 1.0;
    vertices[base + 1].position.y *= -1.0;
    vertices[base + 1].uv = float2(0.0, 1.0);
    vertices[base + 1].color = float4(color * 0.8, 1.0);

    vertices[base + 2].position = right_wing * 2.0 - 1.0;
    vertices[base + 2].position.y *= -1.0;
    vertices[base + 2].uv = float2(1.0, 1.0);
    vertices[base + 2].color = float4(color * 0.8, 1.0);

    if (tid == 0) {
        atomic_store_explicit(vertex_count, 1024 * 3, memory_order_relaxed);
    }
}

// Vertex Shader
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut boid_vertex(
    const device BoidVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    BoidVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

// Fragment Shader
fragment float4 boid_fragment(VertexOut in [[stage_in]]) {
    float alpha = 1.0 - in.uv.y * 0.3;
    float glow = max(in.color.r, max(in.color.g, in.color.b)) * 0.15;
    float3 final_color = in.color.rgb + glow;
    return float4(final_color, alpha);
}
