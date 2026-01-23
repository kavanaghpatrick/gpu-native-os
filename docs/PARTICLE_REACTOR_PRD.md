# PRD: Generative Art Particle Reactor

**Version**: 1.0
**Date**: 2026-01-23
**Status**: Ready for Implementation
**Platform**: Apple Silicon (M1/M2/M3/M4) / Metal

---

## 1. Overview

### What It Is

The **Generative Art Particle Reactor** is a GPU-native demo application where a single threadgroup of 1024 threads simulates a living, breathing particle ecosystem. Each thread owns exactly one particle, creating emergent flocking behavior through simple local rules. Particles leave persistent trails that fade over time, creating luminous ribbons of motion across the screen.

### Why It's Impressive

**Emergent Beauty from Simple Rules**: Each particle follows only three local rules (separation, alignment, cohesion), yet the collective behavior produces mesmerizing, organic patterns—schools of fish, murmurations of starlings, flowing rivers of light.

**Technical Innovation**:
- **Zero CPU physics**: All simulation runs entirely on GPU
- **Perfect synchronization**: Single threadgroup enables `threadgroup_barrier()` for lock-step neighbor queries
- **Real-time interactivity**: Touch input creates attractors/repulsors that warp the flow
- **Persistent state**: Trail buffer accumulates history across frames without CPU intervention

**Visual Impact**:
- Bioluminescent trails that glow and fade
- Organic, unpredictable yet coherent motion
- Interactive—users shape the art with their fingers

---

## 2. User Experience

### What the User Sees

On launch, 1024 particles drift across a dark canvas. Initially scattered, they begin to cluster, swirl, and flow as flocking rules take effect. Each particle leaves a fading phosphorescent trail, creating ribbons of light that paint the history of motion.

The visual evolves continuously:
- Particles form schools that flow together
- Vortices emerge spontaneously then dissolve
- Tendrils split and merge organically
- Colors shift based on velocity and neighbors

### User Interactions

| Gesture | Effect | Visual Feedback |
|---------|--------|-----------------|
| **Tap** | Creates attractor point | Particles spiral inward, trails converge |
| **Tap + Hold** | Creates repulsor point | Particles explode outward, trails radiate |
| **Drag/Swipe** | Adds wind force in drag direction | Particles stream with the wind, trails align |
| **Pinch In** | Creates gravity well (black hole) | Particles orbit, trails form spiral arms |
| **Pinch Out** | Creates explosion burst | Particles scatter, trails form starburst |
| **Two-finger rotate** | Adds global rotation force | Entire system spins, trails form vortex |

### Modes

1. **Zen Mode** (default): Slow, calm flocking with soft colors
2. **Chaos Mode**: Stronger forces, faster particles, vivid colors
3. **Firefly Mode**: Particles blink on/off, create constellation patterns

---

## 3. Technical Architecture

### 3.1 Thread Assignment

All 1024 threads participate in all phases (Unified Worker Model):

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SINGLE THREADGROUP (1024 threads)                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: NEIGHBOR QUERY          All threads read particle positions│
│  ─────────────────────────────────into threadgroup memory           │
│                                                                      │
│  PHASE 2: FLOCKING FORCES         Each thread computes forces for   │
│  ─────────────────────────────────its own particle from neighbors   │
│                                                                      │
│  PHASE 3: EXTERNAL FORCES         Each thread adds touch attractors,│
│  ─────────────────────────────────wind forces, gravity wells        │
│                                                                      │
│  PHASE 4: INTEGRATION             Each thread updates its particle's│
│  ─────────────────────────────────velocity and position             │
│                                                                      │
│  PHASE 5: TRAIL UPDATE            Each thread writes its particle's │
│  ─────────────────────────────────current position to trail buffer  │
│                                                                      │
│  PHASE 6: RENDER COMPOSITE        Threads collaborate to blend      │
│  ─────────────────────────────────particles + trails to output      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DEVICE MEMORY (Persistent)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PARTICLE STATE BUFFER (48KB)                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Particle[1024] - position, velocity, color, age, flags         │ │
│  │ Size: 48 bytes × 1024 = 49,152 bytes                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  TRAIL BUFFER (Ring Buffer, 4MB)                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ TrailPoint[1024 × 256] - 256 history frames per particle       │ │
│  │ Size: 16 bytes × 262,144 = 4,194,304 bytes                     │ │
│  │ Ring index advances each frame (modulo 256)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  FORCE FIELD BUFFER (1KB)                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ ForcePoint[16] - active attractors, repulsors, wind vectors    │ │
│  │ Size: 64 bytes × 16 = 1,024 bytes                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  INPUT QUEUE (2KB)                                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ TouchEvent[64] - ring buffer of touch inputs                   │ │
│  │ Size: 32 bytes × 64 = 2,048 bytes                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  FRAME STATE (256B)                                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ frame_number, time, delta_time, trail_head, mode, parameters   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  OUTPUT TEXTURE (33MB @ 4K)                                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ RGBA8 texture for final composited output                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  TRAIL ACCUMULATION TEXTURE (67MB @ 4K)                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ RGBA16F for HDR trail accumulation with fade                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Total GPU Memory: ~105MB
```

### 3.3 Threadgroup Memory (32KB on-chip)

```metal
struct ThreadgroupMemory {
    // All particle positions (for neighbor queries)
    float2 positions[1024];              // 8KB

    // All particle velocities (for alignment)
    float2 velocities[1024];             // 8KB

    // Spatial hash buckets for O(1) neighbor lookup
    uint16_t bucket_start[256];          // 512B (16×16 grid)
    uint16_t bucket_count[256];          // 512B
    uint16_t sorted_indices[1024];       // 2KB

    // Active force points (copied from device)
    ForcePoint forces[16];               // 1KB
    uint force_count;                    // 4B

    // Frame parameters
    float delta_time;                    // 4B
    uint trail_head;                     // 4B
    uint mode;                           // 4B

    // SIMD scratch space
    float simd_scratch[32][8];           // 1KB

    // Remaining: ~11KB for expansion
};
```

### 3.4 Per-Frame Pipeline

```
Frame N begins (VSync triggers)
│
├─ PHASE 1: LOAD & SPATIAL HASH (0.2ms)
│  ├─ All threads load their particle position to threadgroup memory
│  ├─ Parallel radix sort into spatial hash buckets
│  └─ threadgroup_barrier()
│
├─ PHASE 2: FLOCKING FORCES (0.8ms)
│  ├─ Each thread queries neighbors via spatial hash
│  ├─ Compute separation force (steer away from too-close neighbors)
│  ├─ Compute alignment force (match average neighbor velocity)
│  ├─ Compute cohesion force (steer toward center of neighbors)
│  └─ threadgroup_barrier()
│
├─ PHASE 3: EXTERNAL FORCES (0.2ms)
│  ├─ Each thread iterates force points
│  ├─ Add attractor/repulsor forces (inverse square)
│  ├─ Add wind forces (directional)
│  ├─ Add boundary forces (soft walls)
│  └─ threadgroup_barrier()
│
├─ PHASE 4: INTEGRATION (0.1ms)
│  ├─ Apply forces to velocity (clamped to max speed)
│  ├─ Apply velocity to position
│  ├─ Handle boundary wrapping/bouncing
│  └─ Write updated state to device memory
│
├─ PHASE 5: TRAIL UPDATE (0.3ms)
│  ├─ Each thread writes position to trail buffer [tid][trail_head]
│  ├─ Fade trail accumulation texture (* 0.98)
│  ├─ Draw new trail segment (current pos → previous pos)
│  └─ threadgroup_barrier()
│
├─ PHASE 6: RENDER (2.0ms)
│  ├─ Clear output or blend with trail texture
│  ├─ Each thread renders its particle as glowing point
│  ├─ Composite particles over trail texture
│  └─ Apply bloom/glow post-process
│
└─ Frame N complete (~3.6ms total, 120Hz budget is 8.3ms)
```

---

## 4. Data Structures

### 4.1 Particle State

```metal
struct Particle {
    float2 position;        // 8 bytes  - screen coordinates [0,1]
    float2 velocity;        // 8 bytes  - units per second
    half4 color;            // 8 bytes  - RGBA (half precision)
    float age;              // 4 bytes  - seconds since spawn
    float phase;            // 4 bytes  - animation phase [0,2π]
    uint32_t flags;         // 4 bytes  - bit flags (alive, visible, etc.)
    float size;             // 4 bytes  - radius multiplier
    float trail_intensity;  // 4 bytes  - how bright the trail is
    uint32_t _padding;      // 4 bytes  - align to 48 bytes
};  // Total: 48 bytes
```

### 4.2 Trail Point

```metal
struct TrailPoint {
    half2 position;         // 4 bytes  - position at this frame
    half intensity;         // 2 bytes  - brightness at this point
    half age;               // 2 bytes  - how old this point is
    half4 color;            // 8 bytes  - color at this point
};  // Total: 16 bytes
```

### 4.3 Force Point

```metal
struct ForcePoint {
    float2 position;        // 8 bytes  - screen position [0,1]
    float2 direction;       // 8 bytes  - for directional forces (wind)
    float strength;         // 4 bytes  - force magnitude
    float radius;           // 4 bytes  - falloff radius
    uint32_t type;          // 4 bytes  - ATTRACTOR, REPULSOR, WIND, VORTEX
    float lifetime;         // 4 bytes  - remaining time (0 = permanent)
    float falloff;          // 4 bytes  - 1=linear, 2=quadratic
    uint32_t _padding[7];   // 28 bytes - align to 64 bytes
};  // Total: 64 bytes

// Force types
constant uint FORCE_NONE = 0;
constant uint FORCE_ATTRACTOR = 1;
constant uint FORCE_REPULSOR = 2;
constant uint FORCE_WIND = 3;
constant uint FORCE_VORTEX = 4;
constant uint FORCE_GRAVITY_WELL = 5;
```

### 4.4 Touch Event

```metal
struct TouchEvent {
    float2 position;        // 8 bytes  - touch position [0,1]
    float2 velocity;        // 8 bytes  - drag velocity
    uint32_t type;          // 4 bytes  - BEGAN, MOVED, ENDED
    uint32_t touch_id;      // 4 bytes  - for multi-touch tracking
    float pressure;         // 4 bytes  - 3D touch pressure
    float timestamp;        // 4 bytes  - event time
};  // Total: 32 bytes
```

### 4.5 Frame State

```metal
struct FrameState {
    uint64_t frame_number;  // 8 bytes
    float time;             // 4 bytes  - total elapsed time
    float delta_time;       // 4 bytes  - time since last frame
    uint32_t trail_head;    // 4 bytes  - ring buffer write index
    uint32_t mode;          // 4 bytes  - ZEN, CHAOS, FIREFLY

    // Flocking parameters (tunable)
    float separation_radius;    // 4 bytes
    float separation_weight;    // 4 bytes
    float alignment_radius;     // 4 bytes
    float alignment_weight;     // 4 bytes
    float cohesion_radius;      // 4 bytes
    float cohesion_weight;      // 4 bytes
    float max_speed;            // 4 bytes
    float max_force;            // 4 bytes

    // Visual parameters
    float trail_fade_rate;      // 4 bytes
    float glow_intensity;       // 4 bytes
    float particle_size;        // 4 bytes

    uint32_t _padding[5];       // 20 bytes - align to 128 bytes
};  // Total: 128 bytes
```

---

## 5. Shader Pseudocode

### 5.1 Main Kernel

```metal
kernel void particle_reactor(
    device Particle* particles [[buffer(0)]],
    device TrailPoint* trails [[buffer(1)]],
    device ForcePoint* forces [[buffer(2)]],
    device FrameState* state [[buffer(3)]],
    texture2d<half, access::read_write> trail_texture [[texture(0)]],
    texture2d<half, access::write> output [[texture(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup ThreadgroupMemory shared;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: LOAD POSITIONS TO SHARED MEMORY
    // ═══════════════════════════════════════════════════════════════

    Particle my_particle = particles[tid];
    shared.positions[tid] = my_particle.position;
    shared.velocities[tid] = my_particle.velocity;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build spatial hash (parallel bucket sort)
    uint2 cell = uint2(my_particle.position * 16.0);  // 16×16 grid
    uint bucket = cell.y * 16 + cell.x;
    // ... parallel counting sort into buckets ...

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: COMPUTE FLOCKING FORCES
    // ═══════════════════════════════════════════════════════════════

    float2 separation = float2(0);
    float2 alignment = float2(0);
    float2 cohesion = float2(0);
    uint neighbor_count = 0;

    // Query nearby buckets (3×3 neighborhood)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 neighbor_cell = int2(cell) + int2(dx, dy);
            if (neighbor_cell.x < 0 || neighbor_cell.x >= 16) continue;
            if (neighbor_cell.y < 0 || neighbor_cell.y >= 16) continue;

            uint neighbor_bucket = neighbor_cell.y * 16 + neighbor_cell.x;
            uint start = shared.bucket_start[neighbor_bucket];
            uint count = shared.bucket_count[neighbor_bucket];

            for (uint i = 0; i < count; i++) {
                uint other_idx = shared.sorted_indices[start + i];
                if (other_idx == tid) continue;

                float2 other_pos = shared.positions[other_idx];
                float2 other_vel = shared.velocities[other_idx];
                float2 diff = my_particle.position - other_pos;
                float dist = length(diff);

                // Separation: steer away from close neighbors
                if (dist < state->separation_radius && dist > 0.001) {
                    separation += normalize(diff) / dist;
                }

                // Alignment: match neighbor velocities
                if (dist < state->alignment_radius) {
                    alignment += other_vel;
                    neighbor_count++;
                }

                // Cohesion: steer toward neighbor center
                if (dist < state->cohesion_radius) {
                    cohesion += other_pos;
                }
            }
        }
    }

    // Normalize and weight forces
    separation *= state->separation_weight;

    if (neighbor_count > 0) {
        alignment = (alignment / float(neighbor_count) - my_particle.velocity);
        alignment *= state->alignment_weight;

        cohesion = (cohesion / float(neighbor_count) - my_particle.position);
        cohesion *= state->cohesion_weight;
    }

    float2 flock_force = separation + alignment + cohesion;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: EXTERNAL FORCES
    // ═══════════════════════════════════════════════════════════════

    float2 external_force = float2(0);

    // Load force points to shared memory (thread 0)
    if (tid == 0) {
        shared.force_count = 0;
        for (uint i = 0; i < 16; i++) {
            if (forces[i].type != FORCE_NONE) {
                shared.forces[shared.force_count++] = forces[i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads process force points
    for (uint i = 0; i < shared.force_count; i++) {
        ForcePoint f = shared.forces[i];
        float2 to_force = f.position - my_particle.position;
        float dist = length(to_force);

        if (dist < f.radius && dist > 0.001) {
            float falloff = pow(1.0 - dist / f.radius, f.falloff);

            switch (f.type) {
                case FORCE_ATTRACTOR:
                    external_force += normalize(to_force) * f.strength * falloff;
                    break;

                case FORCE_REPULSOR:
                    external_force -= normalize(to_force) * f.strength * falloff;
                    break;

                case FORCE_WIND:
                    external_force += f.direction * f.strength * falloff;
                    break;

                case FORCE_VORTEX:
                    float2 tangent = float2(-to_force.y, to_force.x);
                    external_force += normalize(tangent) * f.strength * falloff;
                    break;

                case FORCE_GRAVITY_WELL:
                    // Orbital mechanics: strong pull + tangential velocity
                    external_force += normalize(to_force) * f.strength * falloff * 2.0;
                    break;
            }
        }
    }

    // Boundary forces (soft walls)
    float margin = 0.05;
    if (my_particle.position.x < margin)
        external_force.x += (margin - my_particle.position.x) * 10.0;
    if (my_particle.position.x > 1.0 - margin)
        external_force.x -= (my_particle.position.x - (1.0 - margin)) * 10.0;
    if (my_particle.position.y < margin)
        external_force.y += (margin - my_particle.position.y) * 10.0;
    if (my_particle.position.y > 1.0 - margin)
        external_force.y -= (my_particle.position.y - (1.0 - margin)) * 10.0;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: INTEGRATION
    // ═══════════════════════════════════════════════════════════════

    float2 total_force = flock_force + external_force;

    // Clamp force magnitude
    float force_mag = length(total_force);
    if (force_mag > state->max_force) {
        total_force = normalize(total_force) * state->max_force;
    }

    // Update velocity
    my_particle.velocity += total_force * state->delta_time;

    // Clamp speed
    float speed = length(my_particle.velocity);
    if (speed > state->max_speed) {
        my_particle.velocity = normalize(my_particle.velocity) * state->max_speed;
    }

    // Update position
    float2 prev_position = my_particle.position;
    my_particle.position += my_particle.velocity * state->delta_time;

    // Wrap at boundaries (or bounce)
    my_particle.position = fract(my_particle.position);  // wrap

    // Update age and phase
    my_particle.age += state->delta_time;
    my_particle.phase = fract(my_particle.phase + state->delta_time * 0.5);

    // Color based on velocity direction and magnitude
    float hue = atan2(my_particle.velocity.y, my_particle.velocity.x) / (2.0 * M_PI_F) + 0.5;
    float saturation = saturate(speed / state->max_speed);
    my_particle.color = half4(hsv_to_rgb(hue, saturation, 1.0), 1.0);

    // Write back to device memory
    particles[tid] = my_particle;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 5: TRAIL UPDATE
    // ═══════════════════════════════════════════════════════════════

    // Write current position to trail ring buffer
    uint trail_idx = tid * 256 + state->trail_head;
    trails[trail_idx].position = half2(my_particle.position);
    trails[trail_idx].intensity = half(my_particle.trail_intensity);
    trails[trail_idx].age = half(0);
    trails[trail_idx].color = my_particle.color;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 6: RENDER (cooperative across all threads)
    // ═══════════════════════════════════════════════════════════════

    // Fade existing trail texture
    // Each thread handles a tile of pixels
    uint2 texture_size = uint2(output.get_width(), output.get_height());
    uint pixels_per_thread = (texture_size.x * texture_size.y) / 1024;
    uint pixel_start = tid * pixels_per_thread;

    for (uint p = 0; p < pixels_per_thread; p++) {
        uint pixel_idx = pixel_start + p;
        uint2 coord = uint2(pixel_idx % texture_size.x, pixel_idx / texture_size.x);

        half4 trail_color = trail_texture.read(coord);
        trail_color *= half(state->trail_fade_rate);  // Fade
        trail_texture.write(trail_color, coord);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Draw trail segment (line from prev to current position)
    // Using Bresenham or DDA line algorithm
    uint2 p0 = uint2(prev_position * float2(texture_size));
    uint2 p1 = uint2(my_particle.position * float2(texture_size));
    draw_line_additive(trail_texture, p0, p1, my_particle.color * half(my_particle.trail_intensity));

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Composite: trails + particles + glow
    for (uint p = 0; p < pixels_per_thread; p++) {
        uint pixel_idx = pixel_start + p;
        uint2 coord = uint2(pixel_idx % texture_size.x, pixel_idx / texture_size.x);

        half4 trail_color = trail_texture.read(coord);
        half4 final_color = trail_color;

        // Add glow from nearby particles (simplified - just this thread's particle)
        float2 pixel_pos = float2(coord) / float2(texture_size);
        float dist_to_particle = distance(pixel_pos, my_particle.position);
        float glow = exp(-dist_to_particle * 100.0) * state->glow_intensity;
        final_color += my_particle.color * half(glow);

        output.write(saturate(final_color), coord);
    }

    // Thread 0 updates frame state
    if (tid == 0) {
        state->frame_number++;
        state->trail_head = (state->trail_head + 1) % 256;
    }
}
```

### 5.2 Trail Accumulation (Alternative Approach)

```metal
// Instead of per-pixel fade, use texture blend in fragment shader
fragment half4 trail_fragment(
    VertexOut in [[stage_in]],
    texture2d<half> prev_frame [[texture(0)]]
) {
    half4 prev = prev_frame.sample(sampler, in.uv);
    half4 new_color = in.color;

    // Additive blend with fade
    return prev * 0.98h + new_color;
}
```

---

## 6. Input Handling

### 6.1 Touch Event Processing

```swift
// CPU-side: Write touch events to shared buffer
class InputHandler {
    var inputBuffer: MTLBuffer  // Shared with GPU

    func handleTouch(_ touch: UITouch, view: UIView, type: TouchType) {
        let queue = inputBuffer.contents().assumingMemoryBound(to: InputQueue.self)
        let head = queue.pointee.head
        let slot = head % 64

        let location = touch.location(in: view)
        let normalized = CGPoint(
            x: location.x / view.bounds.width,
            y: location.y / view.bounds.height
        )

        var event = TouchEvent()
        event.position = simd_float2(Float(normalized.x), Float(normalized.y))
        event.velocity = simd_float2(0, 0)  // Calculated from delta
        event.type = type.rawValue
        event.touch_id = UInt32(touch.hash & 0xFFFFFFFF)
        event.pressure = Float(touch.force / touch.maximumPossibleForce)
        event.timestamp = Float(CACurrentMediaTime())

        queue.pointee.events[Int(slot)] = event
        queue.pointee.head = head + 1
    }
}
```

### 6.2 Gesture → Force Mapping

```metal
// GPU-side: Convert touch events to force points
void process_input(
    device InputQueue* input,
    device ForcePoint* forces,
    threadgroup ThreadgroupMemory& shared,
    uint tid
) {
    if (tid != 0) return;  // Single-threaded input processing

    uint head = atomic_load(&input->head);
    uint tail = atomic_load(&input->tail);

    while (tail < head) {
        TouchEvent event = input->events[tail % 64];

        switch (event.type) {
            case TOUCH_BEGAN:
                // Find empty force slot
                for (uint i = 0; i < 16; i++) {
                    if (forces[i].type == FORCE_NONE) {
                        forces[i].type = FORCE_ATTRACTOR;
                        forces[i].position = event.position;
                        forces[i].strength = 2.0 + event.pressure * 3.0;
                        forces[i].radius = 0.2;
                        forces[i].falloff = 2.0;  // Quadratic
                        forces[i].lifetime = 0;   // Until touch ends
                        break;
                    }
                }
                break;

            case TOUCH_MOVED:
                // Update force position, add wind in drag direction
                // ... find force by touch_id, update position
                // If drag velocity high enough, add wind force
                if (length(event.velocity) > 0.5) {
                    add_wind_force(forces, event.position,
                                   normalize(event.velocity),
                                   length(event.velocity));
                }
                break;

            case TOUCH_ENDED:
                // Remove force or convert to decaying burst
                // ... find force by touch_id
                // Convert to repulsor burst if quick tap
                break;
        }

        tail++;
    }

    atomic_store(&input->tail, tail);

    // Decay temporary forces
    for (uint i = 0; i < 16; i++) {
        if (forces[i].lifetime > 0) {
            forces[i].lifetime -= shared.delta_time;
            if (forces[i].lifetime <= 0) {
                forces[i].type = FORCE_NONE;
            }
        }
    }
}
```

### 6.3 Gesture Recognition

| Gesture | Detection | Force Created |
|---------|-----------|---------------|
| Tap | BEGAN + ENDED < 0.3s, distance < 10px | Attractor (0.5s decay) |
| Long Press | BEGAN + no MOVED for > 0.5s | Repulsor (while held) |
| Drag | MOVED events with velocity | Wind (direction of drag) |
| Pinch In | Two touches moving closer | Gravity Well at center |
| Pinch Out | Two touches moving apart | Explosion burst |
| Rotate | Two touches rotating | Vortex at center |

---

## 7. Visual Design

### 7.1 Particle Appearance

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PARTICLE VISUAL                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Core:     Bright white/colored center (2-4 pixels)                │
│   Glow:     Gaussian falloff halo (8-16 pixel radius)               │
│   Color:    HSV based on velocity direction (hue) and speed (sat)   │
│                                                                      │
│   Velocity → Color mapping:                                         │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  ↑ North = Cyan        ↗ NE = Blue                       │      │
│   │  ← West = Green        → East = Red                      │      │
│   │  ↙ SW = Yellow         ↓ South = Orange                  │      │
│   │  ↖ NW = Teal           ↘ SE = Magenta                    │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│   Speed → Saturation:                                               │
│   • Slow = desaturated (white/gray)                                 │
│   • Fast = vivid saturated color                                    │
│                                                                      │
│   Speed → Size:                                                     │
│   • Slow = larger, softer glow                                      │
│   • Fast = smaller, intense point                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Trail Appearance

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAIL VISUAL                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Persistence: 256 frames (~2 seconds at 120fps)                    │
│   Fade: Exponential decay (0.98 per frame = 50% after 34 frames)    │
│                                                                      │
│   Width: Proportional to particle speed                             │
│   • Fast = thin, sharp trail                                        │
│   • Slow = thick, diffuse trail                                     │
│                                                                      │
│   Color: Inherits from particle at moment of emission               │
│   • Creates rainbow ribbons when direction changes                  │
│                                                                      │
│   Blend Mode: Additive                                              │
│   • Overlapping trails create bright intersections                  │
│   • Creates "heat map" effect where particles congregate            │
│                                                                      │
│   Visual Effect:                                                    │
│   ════════════════╗                                                 │
│                   ║  ← Current position (bright)                    │
│   ════════════════╝                                                 │
│        ↑                                                            │
│   Fading trail (older = dimmer)                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Glow / Bloom Effect

```metal
// Simple bloom: blur bright areas and add back
half4 apply_bloom(texture2d<half> input, uint2 coord, float intensity) {
    half4 center = input.read(coord);

    // Only bloom bright pixels
    half brightness = dot(center.rgb, half3(0.299h, 0.587h, 0.114h));
    if (brightness < 0.5h) return center;

    // Box blur (5x5)
    half4 blur = half4(0);
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            blur += input.read(coord + uint2(x, y));
        }
    }
    blur /= 25.0h;

    // Add bloom
    return center + blur * half(intensity);
}
```

### 7.4 Color Palettes by Mode

| Mode | Palette | Trail Fade | Particle Glow |
|------|---------|------------|---------------|
| **Zen** | Cool blues, teals, soft whites | 0.99 (slow) | 0.3 (subtle) |
| **Chaos** | Hot reds, oranges, magentas | 0.95 (fast) | 0.8 (intense) |
| **Firefly** | Warm yellows, greens | 0.97 + blink | 1.0 (when on) |

---

## 8. Performance Targets

### 8.1 Frame Budget (120Hz = 8.33ms)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRAME BUDGET BREAKDOWN                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  0.0ms ────────────────────────────────────────────────────── 8.3ms │
│  │                                                                │  │
│  ├─ CPU Dispatch (0.3ms) ─────────────────────────────────────────┤  │
│  │  └─ Touch handling + command buffer setup                      │  │
│  │                                                                │  │
│  ├─ GPU Compute (3.5ms) ──────────────────────────────────────────┤  │
│  │  ├─ Phase 1: Spatial Hash (0.2ms)                              │  │
│  │  ├─ Phase 2: Flocking (0.8ms)                                  │  │
│  │  ├─ Phase 3: External Forces (0.2ms)                           │  │
│  │  ├─ Phase 4: Integration (0.1ms)                               │  │
│  │  ├─ Phase 5: Trail Update (0.3ms)                              │  │
│  │  └─ Phase 6: Render (1.9ms)                                    │  │
│  │                                                                │  │
│  ├─ Display Present (0.5ms) ──────────────────────────────────────┤  │
│  │                                                                │  │
│  ├─ Headroom (4.0ms) ─────────────────────────────────────────────┤  │
│  │  └─ Available for: spikes, complex scenes, thermal throttling  │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Performance Metrics

| Metric | Target | Maximum | Notes |
|--------|--------|---------|-------|
| Frame Rate | 120 fps | 120 fps | VSync locked |
| Particle Count | 1024 | 1024 | Thread count is limit |
| Trail History | 256 frames | 512 frames | Ring buffer size |
| Active Forces | 8 | 16 | Force point slots |
| Resolution | 4K (3840×2160) | 4K | Fragment shader handles |
| GPU Compute Time | 3.5ms | 5.0ms | Leave headroom |
| Input Latency | <16ms | <24ms | Touch to visual response |
| Memory Usage | 105MB | 150MB | GPU memory |

### 8.3 Power Efficiency

| State | Power Draw | Percentage of Frame |
|-------|------------|---------------------|
| GPU Compute | ~1.5W | 42% (3.5ms) |
| GPU Render | ~2.0W | 24% (2.0ms) |
| GPU Idle | ~0W | 34% (2.8ms) |
| Display | 2.0W | 100% |
| **Average** | **~5.5W** | - |

Estimated battery life: MacBook Air (53.8Wh) / 5.5W = **~9.8 hours**

---

## 9. Implementation Milestones

### Phase 1: Foundation (Days 1-3)

**Goal**: Basic particle system with simple movement

- [ ] Metal compute pipeline setup (single threadgroup)
- [ ] Particle buffer allocation (1024 particles)
- [ ] Simple integration (velocity → position)
- [ ] Basic rendering (points as colored dots)
- [ ] Frame loop with VSync

**Deliverable**: 1024 particles moving randomly, rendered as points

### Phase 2: Flocking (Days 4-6)

**Goal**: Implement Craig Reynolds' boids algorithm

- [ ] Threadgroup memory for position sharing
- [ ] Spatial hash for neighbor queries
- [ ] Separation force implementation
- [ ] Alignment force implementation
- [ ] Cohesion force implementation
- [ ] Parameter tuning for organic behavior

**Deliverable**: Particles exhibiting flocking/schooling behavior

### Phase 3: Trails (Days 7-9)

**Goal**: Persistent trail rendering

- [ ] Trail ring buffer (256 frames history)
- [ ] Trail accumulation texture
- [ ] Per-frame fade shader
- [ ] Line drawing for trail segments
- [ ] Additive blending

**Deliverable**: Particles leaving glowing trails that fade over time

### Phase 4: Input (Days 10-12)

**Goal**: Touch interactivity

- [ ] Touch event buffer (CPU → GPU)
- [ ] Attractor force (tap)
- [ ] Repulsor force (long press)
- [ ] Wind force (drag)
- [ ] Force decay over time

**Deliverable**: User can influence particle flow with touch

### Phase 5: Polish (Days 13-15)

**Goal**: Visual refinement

- [ ] Particle glow shader
- [ ] Bloom post-processing
- [ ] Velocity-based coloring
- [ ] Mode switching (Zen/Chaos/Firefly)
- [ ] Performance optimization

**Deliverable**: Visually stunning, smooth 120fps experience

### Phase 6: Advanced Features (Days 16-20)

**Goal**: Extended interactions

- [ ] Pinch gesture → gravity well
- [ ] Two-finger rotate → vortex
- [ ] Shake → randomize
- [ ] Background music reactivity (optional)
- [ ] Save/share still images

**Deliverable**: Complete interactive art experience

---

## 10. Future Enhancements

### 10.1 Particle Count Scaling

**Multi-threadgroup approach** (if needed for >1024 particles):
- Split into spatial regions, each handled by one threadgroup
- Boundary particles shared via device memory
- Requires careful synchronization at region edges

### 10.2 3D Extension

- Z-coordinate for depth
- Perspective projection
- Depth-based blur (bokeh)
- 3D flocking in volumetric space

### 10.3 Audio Reactivity

- FFT of microphone input
- Bass → attractor strength
- Treble → particle speed
- Beat detection → color pulse

### 10.4 Particle Types

- Different particle species with different behaviors
- Predator/prey dynamics
- Symbiotic relationships
- Food chain emergence

### 10.5 Environmental Effects

- Obstacles (particles flow around shapes)
- Currents (persistent flow fields)
- Temperature zones (affect speed/color)
- Day/night cycle (changes palette)

### 10.6 Multi-User

- Shared canvas over network
- Each user has different colored particles
- Collaborative art creation

### 10.7 Export/Sharing

- Record video of simulation
- Export high-resolution stills
- Share parameter presets
- Generate NFT-style unique artworks

---

## Appendix A: Flocking Algorithm Reference

### Craig Reynolds' Boids (1987)

Three simple rules create emergent flocking:

1. **Separation**: Steer to avoid crowding local flockmates
   ```
   force = sum of (my_position - neighbor_position) / distance
   ```

2. **Alignment**: Steer toward average heading of local flockmates
   ```
   force = average_neighbor_velocity - my_velocity
   ```

3. **Cohesion**: Steer toward average position of local flockmates
   ```
   force = average_neighbor_position - my_position
   ```

### Tuning Parameters

| Parameter | Zen Mode | Chaos Mode | Effect |
|-----------|----------|------------|--------|
| separation_radius | 0.03 | 0.05 | Personal space |
| separation_weight | 1.5 | 2.5 | Avoidance strength |
| alignment_radius | 0.08 | 0.12 | Matching range |
| alignment_weight | 1.0 | 0.5 | Conformity |
| cohesion_radius | 0.10 | 0.08 | Grouping range |
| cohesion_weight | 0.8 | 0.3 | Togetherness |
| max_speed | 0.3 | 0.8 | Top velocity |
| max_force | 0.1 | 0.3 | Acceleration limit |

---

## Appendix B: Metal Shader Utilities

### HSV to RGB Conversion

```metal
float3 hsv_to_rgb(float h, float s, float v) {
    float3 rgb = saturate(abs(fmod(h * 6.0 + float3(0, 4, 2), 6.0) - 3.0) - 1.0);
    return v * mix(float3(1.0), rgb, s);
}
```

### Soft Circle SDF

```metal
float circle_sdf(float2 p, float2 center, float radius) {
    return length(p - center) - radius;
}

half4 soft_circle(float2 uv, float2 center, float radius, half4 color, float softness) {
    float d = circle_sdf(uv, center, radius);
    float alpha = 1.0 - smoothstep(-softness, softness, d);
    return color * half(alpha);
}
```

### Line Drawing (DDA)

```metal
void draw_line_additive(
    texture2d<half, access::read_write> tex,
    uint2 p0, uint2 p1,
    half4 color
) {
    int2 d = int2(p1) - int2(p0);
    int steps = max(abs(d.x), abs(d.y));
    if (steps == 0) return;

    float2 inc = float2(d) / float(steps);
    float2 p = float2(p0);

    for (int i = 0; i <= steps; i++) {
        uint2 coord = uint2(p);
        half4 existing = tex.read(coord);
        tex.write(existing + color, coord);
        p += inc;
    }
}
```

---

## Appendix C: Default Parameters

```metal
constant FrameState DEFAULT_STATE = {
    .frame_number = 0,
    .time = 0.0,
    .delta_time = 1.0 / 120.0,
    .trail_head = 0,
    .mode = MODE_ZEN,

    // Flocking (Zen mode defaults)
    .separation_radius = 0.03,
    .separation_weight = 1.5,
    .alignment_radius = 0.08,
    .alignment_weight = 1.0,
    .cohesion_radius = 0.10,
    .cohesion_weight = 0.8,
    .max_speed = 0.3,
    .max_force = 0.1,

    // Visual
    .trail_fade_rate = 0.98,
    .glow_intensity = 0.3,
    .particle_size = 1.0,
};
```

---

*This PRD defines a complete GPU-native generative art application that demonstrates the power of single-threadgroup parallel programming. The emergent beauty of flocking behavior, combined with interactive touch input and persistent trails, creates a mesmerizing visual experience that runs entirely on the GPU.*
