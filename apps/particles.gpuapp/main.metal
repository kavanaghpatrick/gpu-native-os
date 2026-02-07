// Particle System
//
// 16K particles with physics simulation.
// Click and drag to attract particles.

constant uint PARTICLE_COUNT = 16384;
constant uint PARTICLES_PER_THREAD = 16;

struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
    float life;
    float max_life;
    float size;
    float _padding;
};

struct ParticleParams {
    float delta_time;
    float time;
    float mouse_x;
    float mouse_y;
    uint frame_number;
    uint mouse_down;
    float attraction_strength;
    float gravity_strength;
};

struct ParticleVertex {
    float2 position;
    float2 uv;
    float4 color;
};

struct VertexCount {
    atomic_uint count;
    uint _padding[3];
};

// Respawn a particle
void respawn_particle(thread Particle& p, uint seed, float time) {
    uint edge = hash(seed) % 4;
    float rand1 = random_float(hash(seed + 1));

    if (edge == 0) {
        p.position = float2(rand1, 0.0);
        p.velocity = float2(random_range(hash(seed + 3), -0.1, 0.1), random_range(hash(seed + 4), 0.1, 0.3));
    } else if (edge == 1) {
        p.position = float2(rand1, 1.0);
        p.velocity = float2(random_range(hash(seed + 3), -0.1, 0.1), random_range(hash(seed + 4), -0.3, -0.1));
    } else if (edge == 2) {
        p.position = float2(0.0, rand1);
        p.velocity = float2(random_range(hash(seed + 3), 0.1, 0.3), random_range(hash(seed + 4), -0.1, 0.1));
    } else {
        p.position = float2(1.0, rand1);
        p.velocity = float2(random_range(hash(seed + 3), -0.3, -0.1), random_range(hash(seed + 4), -0.1, 0.1));
    }

    p.max_life = random_range(hash(seed + 5), 2.0, 5.0);
    p.life = p.max_life;
    p.size = random_range(hash(seed + 6), 0.003, 0.012);

    float hue = random_range(hash(seed + 7), 0.0, 360.0);
    p.color = float4(hsv_to_rgb(hue, 0.9, 1.0), 1.0);
}

kernel void particle_kernel(
    constant FrameState& frame [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    constant ParticleParams& params [[buffer(2)]],
    device ParticleVertex* vertices [[buffer(3)]],
    device Particle* particles [[buffer(4)]],
    device VertexCount* vertex_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        atomic_store_explicit(&vertex_count->count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    float attraction = 3.0;
    float gravity = 0.3;

    for (uint i = 0; i < PARTICLES_PER_THREAD; i++) {
        uint pid = tid * PARTICLES_PER_THREAD + i;
        if (pid >= PARTICLE_COUNT) break;

        Particle p = particles[pid];

        // Initialize on first frame
        if (params.frame_number == 0) {
            uint seed = pid * 12345u + 67890u;
            respawn_particle(p, seed, params.time);
        }

        float dt = params.delta_time;

        // Gravity
        p.velocity.y += gravity * dt;

        // Mouse attraction
        if (frame.modifiers != 0) {
            float2 mouse_pos = float2(frame.cursor_x, frame.cursor_y);
            float2 to_mouse = mouse_pos - p.position;
            float dist = length(to_mouse);

            if (dist > 0.01) {
                float2 dir = normalize(to_mouse);
                float strength = attraction / (dist * dist + 0.1);
                strength = min(strength, 2.0);
                p.velocity += dir * strength * dt;
            }
        }

        // Turbulence
        uint noise_seed = pid + params.frame_number * 7919u;
        float noise_x = random_range(hash(noise_seed), -1.0, 1.0);
        float noise_y = random_range(hash(noise_seed + 1), -1.0, 1.0);
        p.velocity += float2(noise_x, noise_y) * 0.05 * dt;

        // Damping
        p.velocity *= (1.0 - 0.5 * dt);

        // Update position
        p.position += p.velocity * dt;

        // Bounce off edges
        float bounce = 0.6;
        if (p.position.x < 0.0) { p.position.x = 0.0; p.velocity.x *= -bounce; }
        if (p.position.x > 1.0) { p.position.x = 1.0; p.velocity.x *= -bounce; }
        if (p.position.y < 0.0) { p.position.y = 0.0; p.velocity.y *= -bounce; }
        if (p.position.y > 1.0) { p.position.y = 1.0; p.velocity.y *= -bounce; }

        // Decrease life
        p.life -= dt;

        // Respawn if dead
        if (p.life <= 0.0) {
            uint seed = pid * 31337u + params.frame_number * 7919u;
            respawn_particle(p, seed, params.time);
        }

        // Update color based on velocity
        float speed = length(p.velocity);
        float hue = fmod(speed * 500.0 + params.time * 30.0 + float(pid) * 0.1, 360.0);
        float saturation = 0.8 + 0.2 * (p.life / p.max_life);
        float brightness = min(0.7 + 0.3 * speed, 1.0);
        p.color.rgb = hsv_to_rgb(hue, saturation, brightness);
        p.color.a = p.life / p.max_life;

        particles[pid] = p;

        // Generate quad geometry
        float life_scale = 0.5 + 0.5 * (p.life / p.max_life);
        float size = p.size * life_scale * (1.0 + min(speed * 2.0, 1.0));

        float x0 = p.position.x - size;
        float y0 = p.position.y - size;
        float x1 = p.position.x + size;
        float y1 = p.position.y + size;

        uint base = atomic_fetch_add_explicit(&vertex_count->count, 6, memory_order_relaxed);
        float4 color = p.color;

        vertices[base + 0] = ParticleVertex{float2(x0, y0), float2(0, 0), color};
        vertices[base + 1] = ParticleVertex{float2(x0, y1), float2(0, 1), color};
        vertices[base + 2] = ParticleVertex{float2(x1, y1), float2(1, 1), color};
        vertices[base + 3] = ParticleVertex{float2(x0, y0), float2(0, 0), color};
        vertices[base + 4] = ParticleVertex{float2(x1, y1), float2(1, 1), color};
        vertices[base + 5] = ParticleVertex{float2(x1, y0), float2(1, 0), color};
    }
}

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut particle_vertex(
    const device ParticleVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    ParticleVertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position * 2.0 - 1.0, 0.0, 1.0);
    out.position.y = -out.position.y;
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

fragment float4 particle_fragment(VertexOut in [[stage_in]]) {
    float2 center = in.uv - 0.5;
    float dist = length(center) * 2.0;

    float alpha = smoothstep(1.0, 0.3, dist);
    float glow = exp(-dist * 2.0) * 0.5;

    float3 color = in.color.rgb * (1.0 + glow);
    float final_alpha = (alpha + glow * 0.3) * in.color.a;

    return float4(color, final_alpha);
}
