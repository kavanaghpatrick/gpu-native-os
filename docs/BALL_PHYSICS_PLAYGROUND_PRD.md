# PRD: 1024-Ball Physics Playground

**Version**: 1.0
**Date**: 2026-01-23
**Status**: Ready for Implementation
**Target Platform**: Apple Silicon (M1/M2/M3/M4) with Metal

---

## 1. Overview

### What It Is

The **1024-Ball Physics Playground** is a showcase demo for the GPU-Native OS architecture. A single threadgroup of 1024 GPU threads simulates 1024 bouncing balls in real-time, where **each thread owns exactly one ball**. The demo demonstrates:

- **Perfect parallelism**: 1 thread = 1 ball = 1 entity ownership
- **GPU-native physics**: All collision detection and response on GPU
- **Interactive containers**: Widget containers act as collision geometry
- **Tilt-to-pour**: Accelerometer input changes gravity, pouring balls between containers

### Why It's Impressive

| Traditional Approach | GPU-Native Approach |
|---------------------|---------------------|
| CPU calculates physics | GPU calculates physics |
| CPU sends positions to GPU | Positions stay on GPU |
| ~100 balls at 60fps | **1024 balls at 120fps** |
| O(n^2) collision on CPU | Parallel collision in threadgroup |
| Latency: ~16-32ms | Latency: **<8ms** |

**Key Demo Points**:
1. **Zero CPU physics code** - All 1024 balls simulated on GPU
2. **Guaranteed synchronization** - `threadgroup_barrier()` ensures correct physics
3. **Interactive** - Users can tilt device, tap containers, change parameters
4. **Visual proof** - Smooth 120fps with 1024 distinct balls proves the architecture

---

## 2. User Experience

### What the User Sees

```
┌──────────────────────────────────────────────────────────────┐
│  1024-Ball Physics Playground            FPS: 120   Balls: 1024  │
├──────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────────────┐     ┌─────────────────────┐           │
│   │    Container A      │     │    Container B      │           │
│   │  ┌───────────────┐  │     │  ┌───────────────┐  │           │
│   │  │ ○ ○ ○ ○ ○ ○ ○│  │     │  │               │  │           │
│   │  │○ ○ ○ ○ ○ ○ ○ │  │     │  │               │  │           │
│   │  │ ○ ○ ○ ○ ○ ○ ○│  │     │  │               │  │           │
│   │  │○ ○ ○ ○ ○ ○ ○ │  │     │  │               │  │           │
│   │  │ ○ ○ ○ ○ ○ ○ ○│  │     │  │               │  │           │
│   │  └───────────────┘  │     │  └───────────────┘  │           │
│   │     Count: 768      │     │     Count: 256      │           │
│   └─────────────────────┘     └─────────────────────┘           │
│                                                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     Shared Pool                           │   │
│   │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│   [Gravity: 9.8]  [Bounce: 0.8]  [Friction: 0.98]  [Reset]      │
└──────────────────────────────────────────────────────────────────┘
```

### User Interactions

| Input | Action | Response |
|-------|--------|----------|
| **Tilt device** | Changes gravity direction | Balls roll/pour toward low point |
| **Tap container** | Toggle open/closed | Balls can flow in/out when open |
| **Drag slider** | Adjust physics params | Real-time parameter changes |
| **Tap Reset** | Reset ball positions | Balls redistribute evenly |
| **Pinch** | Zoom in/out | See balls in detail or overview |
| **Swipe** | Apply impulse | Push balls in swipe direction |

### Demo Scenarios

1. **Hourglass Mode**: Tilt device 90 degrees, watch all balls pour from one container to another
2. **Shake Mode**: Rapid tilts cause chaotic bouncing (stress test collision)
3. **Calm Pool**: Lay device flat, balls settle into stable positions
4. **Cascade**: Open containers in sequence, watch balls flow through

---

## 3. Technical Architecture

### Thread Assignment Model

```
1024 Threads in Single Threadgroup
├── Thread 0-1023: Each owns Ball[tid]
│
│   Per-Frame Phases (ALL threads participate in ALL phases):
│   ├── Phase 1: Input Collection (read accelerometer, touch)
│   ├── Phase 2: Gravity Update (apply accelerometer to velocity)
│   ├── Phase 3: Collision Detection (ball-ball and ball-wall)
│   ├── Phase 4: Collision Response (update velocities)
│   ├── Phase 5: Integration (update positions)
│   ├── Phase 6: Container Assignment (which container is each ball in?)
│   └── Phase 7: Rendering Prep (generate vertex data)
```

### Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVICE MEMORY (Unified)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BALL STATE (40KB)                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Ball[1024] - 40 bytes each                                  │ │
│  │ position(8) + velocity(8) + radius(4) + color(4) +          │ │
│  │ container_id(2) + flags(2) + mass(4) + _pad(8)              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  CONTAINER STATE (2KB)                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Container[32] - 64 bytes each                               │ │
│  │ bounds(16) + walls[4](32) + flags(4) + ball_count(4) +      │ │
│  │ color(4) + _pad(4)                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PHYSICS PARAMS (64B)                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ gravity(8) + bounce(4) + friction(4) + dt(4) + ...          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  INPUT STATE (32B)                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ accel(12) + touch_pos(8) + touch_active(4) + swipe_vel(8)   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  SPATIAL HASH (128KB) - for O(1) neighbor lookup                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ cell_start[4096] + cell_count[4096] + sorted_balls[1024]    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  VERTEX BUFFER (48KB)                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Vertex[6 * 1024] - 6 vertices per ball quad, 8 bytes each   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total Device Memory: ~220KB
```

### Threadgroup Memory (32KB on-chip)

```
┌─────────────────────────────────────────────────────────────────┐
│               THREADGROUP MEMORY (Fast On-Chip)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COLLISION PAIRS (8KB)                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ collision_pairs[1024] - potential collision partner per ball │ │
│  │ collision_count[32] - per-SIMD collision counts             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  SPATIAL HASH SCRATCH (16KB)                                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ cell_keys[1024] - hash key for each ball                    │ │
│  │ cell_indices[1024] - sorted index for each ball             │ │
│  │ prefix_sums[1024] - for parallel bucket counting            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  SIMD SCRATCH (4KB)                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ simd_reduction[32][32] - for parallel reductions            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  SYNC COUNTERS (64B)                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ atomic phase_counter, collision_count, etc.                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  REMAINING: ~4KB for expansion                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Frame Pipeline

```
Frame Start (VSync trigger)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: INPUT (0.05ms)                                          │
│ ─────────────────────────────────────────────────────────────── │
│ Thread 0: Read accelerometer → threadgroup gravity              │
│ Thread 0-31: Read touch events → threadgroup touch state        │
│ ALL: barrier()                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: SPATIAL HASH BUILD (0.3ms)                              │
│ ─────────────────────────────────────────────────────────────── │
│ ALL: Compute cell key from position → cell_keys[tid]            │
│ ALL: barrier()                                                   │
│ ALL: Parallel prefix sum to get bucket offsets                  │
│ ALL: barrier()                                                   │
│ ALL: Write ball index to sorted position                        │
│ ALL: barrier()                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: COLLISION DETECTION (0.8ms)                             │
│ ─────────────────────────────────────────────────────────────── │
│ ALL: Query spatial hash for neighbors (same + adjacent cells)   │
│ ALL: Test ball[tid] against each neighbor                       │
│ ALL: Record collision pairs in threadgroup memory               │
│ ALL: barrier()                                                   │
│ ALL: Test ball[tid] against container walls                     │
│ ALL: barrier()                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: COLLISION RESPONSE (0.3ms)                              │
│ ─────────────────────────────────────────────────────────────── │
│ ALL: For each collision pair, compute impulse                   │
│ ALL: Apply impulse to velocity (atomics for symmetric pairs)    │
│ ALL: barrier()                                                   │
│ ALL: Apply wall bounces                                         │
│ ALL: barrier()                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: INTEGRATION (0.1ms)                                     │
│ ─────────────────────────────────────────────────────────────── │
│ ALL: Apply gravity to velocity                                  │
│ ALL: Apply friction to velocity                                 │
│ ALL: Update position from velocity                              │
│ ALL: Clamp position to world bounds                             │
│ ALL: Write back to device memory                                │
│ ALL: barrier()                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: CONTAINER ASSIGNMENT (0.1ms)                            │
│ ─────────────────────────────────────────────────────────────── │
│ ALL: Determine which container ball[tid] is in                  │
│ ALL: Update container ball counts (atomic)                      │
│ ALL: barrier()                                                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 7: VERTEX GENERATION (0.3ms)                               │
│ ─────────────────────────────────────────────────────────────── │
│ ALL: Generate 6 vertices for ball[tid] quad                     │
│ ALL: Write to vertex_buffer[tid * 6 ... tid * 6 + 5]            │
│ ALL: barrier()                                                   │
│ Thread 0: Update draw arguments                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
COMPUTE PASS COMPLETE (~2.0ms)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ RENDER PASS (Fragment Shader) (~1.5ms)                           │
│ ─────────────────────────────────────────────────────────────── │
│ Hardware rasterizes ball quads                                  │
│ Fragment shader applies SDF circle for smooth edges             │
│ Per-ball color from ball state                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Frame Complete (~3.5ms total, well under 8.33ms budget)
```

---

## 4. Data Structures

### Ball State (Rust/Metal)

```rust
// Rust host-side definition
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Ball {
    pub position: [f32; 2],      // 8 bytes - center x, y (normalized 0-1)
    pub velocity: [f32; 2],      // 8 bytes - velocity x, y (units/second)
    pub radius: f32,             // 4 bytes - ball radius (normalized)
    pub color: u32,              // 4 bytes - packed RGBA8
    pub container_id: u16,       // 2 bytes - which container (0 = none)
    pub flags: u16,              // 2 bytes - state flags
    pub mass: f32,               // 4 bytes - for collision response
    pub _padding: [f32; 2],      // 8 bytes - align to 40 bytes
}

impl Ball {
    pub const SIZE: usize = 40;
}

// Ball flags
pub mod ball_flags {
    pub const ACTIVE: u16    = 0x0001;  // Ball is simulated
    pub const SLEEPING: u16  = 0x0002;  // Ball at rest (skip physics)
    pub const COLLIDING: u16 = 0x0004;  // Currently colliding
    pub const GRABBED: u16   = 0x0008;  // User is dragging
}
```

```metal
// Metal shader definition
struct Ball {
    float2 position;        // 8 bytes
    float2 velocity;        // 8 bytes
    float radius;           // 4 bytes
    uint color;             // 4 bytes (packed RGBA8)
    ushort container_id;    // 2 bytes
    ushort flags;           // 2 bytes
    float mass;             // 4 bytes
    float2 _padding;        // 8 bytes
};  // Total: 40 bytes
```

### Container State

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Container {
    pub bounds: [f32; 4],        // 16 bytes - x, y, width, height
    pub walls: [Wall; 4],        // 32 bytes - top, right, bottom, left walls
    pub flags: u32,              // 4 bytes - open/closed, etc.
    pub ball_count: u32,         // 4 bytes - current balls inside
    pub color: u32,              // 4 bytes - container border color
    pub _padding: u32,           // 4 bytes - align to 64 bytes
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Wall {
    pub p1: [f32; 2],            // 4 bytes - wall start point
    pub p2: [f32; 2],            // 4 bytes - wall end point (implicit normal)
}

pub mod container_flags {
    pub const TOP_OPEN: u32    = 0x0001;
    pub const RIGHT_OPEN: u32  = 0x0002;
    pub const BOTTOM_OPEN: u32 = 0x0004;
    pub const LEFT_OPEN: u32   = 0x0008;
    pub const VISIBLE: u32     = 0x0010;
}
```

### Physics Parameters

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PhysicsParams {
    pub gravity: [f32; 2],       // 8 bytes - gravity vector (affected by accelerometer)
    pub bounce_coeff: f32,       // 4 bytes - coefficient of restitution (0-1)
    pub friction: f32,           // 4 bytes - velocity damping per frame (0.95-0.99)
    pub dt: f32,                 // 4 bytes - timestep (1/120 for 120Hz)
    pub ball_radius: f32,        // 4 bytes - default ball radius
    pub min_velocity: f32,       // 4 bytes - below this, ball sleeps
    pub collision_iterations: u32, // 4 bytes - solver iterations
    pub spatial_cell_size: f32,  // 4 bytes - size of spatial hash cells
    pub _padding: [f32; 5],      // 20 bytes - align to 64 bytes
}
```

### Input State

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InputState {
    pub accelerometer: [f32; 3], // 12 bytes - device tilt (x, y, z)
    pub touch_position: [f32; 2], // 8 bytes - current touch (normalized)
    pub touch_active: u32,       // 4 bytes - is touching?
    pub swipe_velocity: [f32; 2], // 8 bytes - velocity of swipe gesture
}
```

### Spatial Hash Cell

```rust
// In threadgroup memory - built fresh each frame
#[repr(C)]
struct SpatialHashCell {
    start_index: u32,    // Index into sorted ball list
    count: u32,          // Number of balls in this cell
}

// Grid: 64x64 = 4096 cells covering the 0-1 normalized space
// Cell size: ~0.0156 (1/64)
// Max balls per cell: ~16 (typical)
```

### Vertex Format

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BallVertex {
    pub position: [f32; 2],      // 8 bytes - screen position
    pub uv: [f32; 2],            // 8 bytes - texture coords for SDF
    pub color: u32,              // 4 bytes - packed RGBA8
    pub ball_center: [f32; 2],   // 8 bytes - for fragment shader SDF
    pub radius: f32,             // 4 bytes - for fragment shader SDF
    pub _padding: f32,           // 4 bytes - align to 32 bytes
}
```

---

## 5. Shader Pseudocode

### Main Physics Kernel

```metal
kernel void ball_physics_kernel(
    device Ball* balls [[buffer(0)]],
    device Container* containers [[buffer(1)]],
    constant PhysicsParams& params [[buffer(2)]],
    constant InputState& input [[buffer(3)]],
    device BallVertex* vertices [[buffer(4)]],
    device atomic_uint* draw_count [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // ═══════════════════════════════════════════════════════════════
    // THREAD-LOCAL STATE (registers)
    // ═══════════════════════════════════════════════════════════════
    Ball my_ball = balls[tid];
    float2 my_pos = my_ball.position;
    float2 my_vel = my_ball.velocity;
    float my_radius = my_ball.radius;
    float my_mass = my_ball.mass;

    // ═══════════════════════════════════════════════════════════════
    // THREADGROUP MEMORY
    // ═══════════════════════════════════════════════════════════════
    threadgroup float2 tg_gravity;
    threadgroup uint cell_keys[1024];
    threadgroup uint sorted_indices[1024];
    threadgroup uint cell_start[4096];
    threadgroup atomic_uint cell_count[64];  // Reduced for memory

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: INPUT - Read accelerometer, update gravity
    // ═══════════════════════════════════════════════════════════════
    if (tid == 0) {
        // Map accelerometer to gravity direction
        // input.accelerometer.xy is device tilt
        tg_gravity = float2(
            input.accelerometer.x * params.gravity.x,
            input.accelerometer.y * params.gravity.y + 9.8
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: SPATIAL HASH BUILD
    // ═══════════════════════════════════════════════════════════════

    // Each thread computes its cell key
    float cell_size = params.spatial_cell_size;  // e.g., 1/64
    uint cell_x = uint(clamp(my_pos.x / cell_size, 0.0, 63.0));
    uint cell_y = uint(clamp(my_pos.y / cell_size, 0.0, 63.0));
    uint my_cell = cell_y * 64 + cell_x;
    cell_keys[tid] = my_cell;

    // Clear cell counts (threads 0-63)
    if (tid < 64) {
        atomic_store_explicit(&cell_count[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count balls per cell
    atomic_fetch_add_explicit(&cell_count[my_cell % 64], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Prefix sum for cell_start (simplified - each SIMD handles 2 cells)
    // [Full implementation would use parallel prefix sum]

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: COLLISION DETECTION
    // ═══════════════════════════════════════════════════════════════

    float2 total_impulse = float2(0.0);

    // Check neighbors in same and adjacent cells (9-cell query)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = int(cell_x) + dx;
            int ny = int(cell_y) + dy;
            if (nx < 0 || nx >= 64 || ny < 0 || ny >= 64) continue;

            uint neighbor_cell = uint(ny * 64 + nx);

            // Iterate through balls in this cell
            // [Simplified - real impl uses cell_start/count]
            for (uint other_tid = 0; other_tid < 1024; other_tid++) {
                if (other_tid == tid) continue;
                if (cell_keys[other_tid] != neighbor_cell) continue;

                Ball other = balls[other_tid];
                float2 diff = my_pos - other.position;
                float dist_sq = dot(diff, diff);
                float min_dist = my_radius + other.radius;

                if (dist_sq < min_dist * min_dist && dist_sq > 0.0001) {
                    // Collision detected!
                    float dist = sqrt(dist_sq);
                    float2 normal = diff / dist;
                    float overlap = min_dist - dist;

                    // Compute impulse (elastic collision)
                    float2 rel_vel = my_vel - other.velocity;
                    float vel_along_normal = dot(rel_vel, normal);

                    if (vel_along_normal < 0) {
                        // Balls approaching
                        float e = params.bounce_coeff;
                        float j = -(1 + e) * vel_along_normal;
                        j /= (1.0 / my_mass) + (1.0 / other.mass);

                        total_impulse += (j / my_mass) * normal;

                        // Separate balls (position correction)
                        my_pos += normal * (overlap * 0.5);
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: WALL COLLISIONS
    // ═══════════════════════════════════════════════════════════════

    // World bounds (0-1 normalized)
    if (my_pos.x - my_radius < 0.0) {
        my_pos.x = my_radius;
        my_vel.x = abs(my_vel.x) * params.bounce_coeff;
    }
    if (my_pos.x + my_radius > 1.0) {
        my_pos.x = 1.0 - my_radius;
        my_vel.x = -abs(my_vel.x) * params.bounce_coeff;
    }
    if (my_pos.y - my_radius < 0.0) {
        my_pos.y = my_radius;
        my_vel.y = abs(my_vel.y) * params.bounce_coeff;
    }
    if (my_pos.y + my_radius > 1.0) {
        my_pos.y = 1.0 - my_radius;
        my_vel.y = -abs(my_vel.y) * params.bounce_coeff;
    }

    // Container wall collisions
    for (uint c = 0; c < MAX_CONTAINERS; c++) {
        Container container = containers[c];
        if ((container.flags & CONTAINER_VISIBLE) == 0) continue;

        // Test against each wall (if not open)
        // [Wall collision code similar to world bounds]
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 5: INTEGRATION
    // ═══════════════════════════════════════════════════════════════

    // Apply collision impulses
    my_vel += total_impulse;

    // Apply gravity
    my_vel += tg_gravity * params.dt;

    // Apply friction (air resistance)
    my_vel *= params.friction;

    // Update position
    my_pos += my_vel * params.dt;

    // Sleep detection
    float speed_sq = dot(my_vel, my_vel);
    if (speed_sq < params.min_velocity * params.min_velocity) {
        my_ball.flags |= BALL_SLEEPING;
        my_vel = float2(0.0);
    } else {
        my_ball.flags &= ~BALL_SLEEPING;
    }

    // Write back to device memory
    my_ball.position = my_pos;
    my_ball.velocity = my_vel;
    balls[tid] = my_ball;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 6: VERTEX GENERATION
    // ═══════════════════════════════════════════════════════════════

    // Generate 6 vertices for ball quad (2 triangles)
    uint base = tid * 6;

    // Quad corners: TL, TR, BR, BL
    float2 corners[4] = {
        my_pos + float2(-my_radius, -my_radius),  // TL
        my_pos + float2( my_radius, -my_radius),  // TR
        my_pos + float2( my_radius,  my_radius),  // BR
        my_pos + float2(-my_radius,  my_radius),  // BL
    };

    float2 uvs[4] = {
        float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 1)
    };

    // Triangle 1: TL, BL, BR (CCW)
    // Triangle 2: TL, BR, TR (CCW)
    int indices[6] = {0, 3, 2, 0, 2, 1};

    for (int i = 0; i < 6; i++) {
        int idx = indices[i];
        vertices[base + i].position = corners[idx];
        vertices[base + i].uv = uvs[idx];
        vertices[base + i].color = my_ball.color;
        vertices[base + i].ball_center = my_pos;
        vertices[base + i].radius = my_radius;
    }

    // Thread 0 sets draw count
    if (tid == 0) {
        atomic_store_explicit(draw_count, 1024 * 6, memory_order_relaxed);
    }
}
```

### Fragment Shader (SDF Circle)

```metal
fragment float4 ball_fragment(
    BallVertexOut in [[stage_in]]
) {
    // SDF circle rendering for smooth edges
    float2 uv = in.uv * 2.0 - 1.0;  // Map to -1..1
    float dist = length(uv);

    // Smooth edge with anti-aliasing
    float edge_width = fwidth(dist) * 1.5;
    float alpha = 1.0 - smoothstep(1.0 - edge_width, 1.0, dist);

    // Discard outside circle
    if (alpha < 0.01) discard_fragment();

    // Unpack color
    float4 color = unpack_rgba8(in.color);
    color.a *= alpha;

    // Optional: Add slight gradient for depth effect
    color.rgb *= (1.0 - dist * 0.3);

    return color;
}
```

---

## 6. Collision Detection Strategy

### Spatial Hashing vs Brute Force Analysis

| Approach | Complexity | 1024 Balls | Best When |
|----------|-----------|------------|-----------|
| Brute Force | O(n^2) | 1M checks | n < 100 |
| Spatial Hash | O(n * k) | ~10K checks | n > 100, uniform distribution |

**Decision**: Use **Spatial Hashing** for:
- 64x64 grid (4096 cells)
- Average ~0.25 balls/cell when spread out
- 9-cell query = ~2.25 balls average to check
- Worst case (all balls in one cell): falls back to O(n^2) for that cell

### Spatial Hash Implementation

```metal
// Step 1: Each thread computes its cell
uint cell = compute_cell(my_position, cell_size);
cell_keys[tid] = cell;
barrier();

// Step 2: Parallel counting (atomic)
atomic_add(&cell_counts[cell], 1);
barrier();

// Step 3: Prefix sum for cell_start indices
// Uses SIMD prefix sum + cross-SIMD reduction
parallel_prefix_sum(cell_counts, cell_start, 4096);
barrier();

// Step 4: Scatter balls to sorted positions
uint my_sorted_index = atomic_add(&cell_write_ptr[cell], 1);
sorted_balls[cell_start[cell] + my_sorted_index] = tid;
barrier();

// Step 5: Query - each thread checks 9 neighboring cells
for (dy = -1; dy <= 1; dy++) {
    for (dx = -1; dx <= 1; dx++) {
        uint neighbor_cell = compute_neighbor_cell(cell, dx, dy);
        for (i = cell_start[neighbor_cell]; i < cell_start[neighbor_cell + 1]; i++) {
            uint other_tid = sorted_balls[i];
            if (other_tid != tid) {
                test_collision(tid, other_tid);
            }
        }
    }
}
```

---

## 7. Widget Integration

### Containers as Collision Geometry

Containers from the GPU-Native OS widget system become physics collision boundaries.

```rust
// Convert widget bounds to physics container
fn widget_to_container(widget: &WidgetCompact) -> Container {
    let bounds = widget.get_bounds();
    Container {
        bounds: [bounds[0], bounds[1], bounds[2], bounds[3]],
        walls: [
            Wall::horizontal(bounds[0], bounds[0] + bounds[2], bounds[1]),  // Top
            Wall::vertical(bounds[0] + bounds[2], bounds[1], bounds[1] + bounds[3]),  // Right
            Wall::horizontal(bounds[0], bounds[0] + bounds[2], bounds[1] + bounds[3]),  // Bottom
            Wall::vertical(bounds[0], bounds[1], bounds[1] + bounds[3]),  // Left
        ],
        flags: container_flags::VISIBLE,
        ball_count: 0,
        color: widget.packed_colors,
        _padding: 0,
    }
}
```

### Opening/Closing Container Walls

```metal
// In collision phase - skip wall if open
if ((container.flags & TOP_OPEN) == 0) {
    test_wall_collision(my_pos, my_vel, container.walls[0], &impulse);
}
if ((container.flags & BOTTOM_OPEN) == 0) {
    test_wall_collision(my_pos, my_vel, container.walls[2], &impulse);
}
// ... similar for left/right
```

### Ball Count Display

Each container tracks how many balls are inside:

```metal
// After position update
threadgroup atomic_uint container_counts[MAX_CONTAINERS];

// Reset counts (thread 0 for each container)
if (tid < num_containers) {
    atomic_store_explicit(&container_counts[tid], 0, memory_order_relaxed);
}
barrier();

// Each ball increments its container's count
uint my_container = determine_container(my_pos);
if (my_container < num_containers) {
    atomic_fetch_add_explicit(&container_counts[my_container], 1, memory_order_relaxed);
}
barrier();

// Thread 0 writes counts to device memory
if (tid < num_containers) {
    containers[tid].ball_count = atomic_load_explicit(&container_counts[tid], memory_order_relaxed);
}
```

---

## 8. Visual Design

### Color Scheme

```rust
// Ball colors - rainbow gradient by initial position
fn ball_color(index: usize) -> u32 {
    let hue = (index as f32 / 1024.0) * 360.0;
    hsv_to_rgba8(hue, 0.8, 0.95)
}

// Container colors
const CONTAINER_BORDER: u32 = 0x4A5568FF;  // Gray-600
const CONTAINER_FILL: u32 = 0x2D374850;    // Gray-800 with alpha
const CONTAINER_OPEN_INDICATOR: u32 = 0x48BB78FF;  // Green-400

// Background
const BACKGROUND_COLOR: u32 = 0x1A202CFF;  // Gray-900
```

### Ball Rendering Effects

1. **SDF Circle**: Smooth anti-aliased edges via signed distance field
2. **Depth Shading**: Slight radial gradient gives 3D appearance
3. **Velocity Trail**: Optional motion blur based on velocity magnitude
4. **Sleep Indicator**: Sleeping balls slightly dimmed (80% brightness)
5. **Collision Flash**: Brief brightness pulse on collision

```metal
// Fragment shader effects
float4 render_ball(BallVertexOut in) {
    float dist = length(in.uv * 2.0 - 1.0);

    // Base color
    float4 color = unpack_rgba8(in.color);

    // Depth shading
    color.rgb *= 1.0 - dist * 0.3;

    // Edge anti-aliasing
    float alpha = smoothstep(1.0, 0.95, dist);

    // Sleeping balls dimmed
    if (in.flags & BALL_SLEEPING) {
        color.rgb *= 0.8;
    }

    return float4(color.rgb, color.a * alpha);
}
```

### Container Rendering

```
┌───────────────────────────────────┐
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← 2px border (solid when closed)
│▓                               ▓│
│▓   ╭─────────────────────────╮ ▓│ ← Inner area with rounded corners
│▓   │                         │ ▓│
│▓   │                         │ ▓│
│▓   │                         │ ▓│
│▓   │                         │ ▓│
│▓   ╰─────────────────────────╯ ▓│
│▓▓▓▓▓▓▓▓▓▓▓ OPEN ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← Green indicator when open
└───────────────────────────────────┘
     Count: 512                      ← Ball count label
```

---

## 9. Performance Targets

### Frame Budget (120Hz = 8.33ms)

| Component | Target | Max | Notes |
|-----------|--------|-----|-------|
| Input processing | 0.05ms | 0.1ms | Read accelerometer, touch |
| Spatial hash build | 0.3ms | 0.5ms | 1024 balls, 4096 cells |
| Collision detection | 0.8ms | 1.5ms | ~10K checks average |
| Collision response | 0.3ms | 0.5ms | Impulse application |
| Integration | 0.1ms | 0.2ms | Position/velocity update |
| Container assignment | 0.1ms | 0.2ms | Ball counting |
| Vertex generation | 0.3ms | 0.5ms | 6144 vertices |
| **Compute Total** | **2.0ms** | **3.5ms** | |
| Fragment rendering | 1.5ms | 2.5ms | SDF circles |
| **Frame Total** | **3.5ms** | **6.0ms** | 2.33ms headroom |

### Collision Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Ball-ball checks per frame | < 20K | With spatial hash |
| Ball-wall checks per frame | 4K | 1024 balls x 4 walls |
| Collision iterations | 1-2 | More for stacking stability |
| Penetration resolution | < 0.5px | No visible overlap |

### Memory Bandwidth

| Operation | Size | Frequency | Bandwidth |
|-----------|------|-----------|-----------|
| Ball state read | 40KB | 1/frame | 40KB |
| Ball state write | 40KB | 1/frame | 40KB |
| Vertex write | 192KB | 1/frame | 192KB |
| **Total/frame** | | | ~272KB |
| **At 120Hz** | | | ~32.6 MB/s |

Apple Silicon unified memory bandwidth: ~200 GB/s - well within budget.

---

## 10. Implementation Milestones

### Milestone 1: Static Balls (Week 1)
- [ ] Create Ball struct and buffer
- [ ] Initialize 1024 balls with random positions
- [ ] Basic vertex generation kernel
- [ ] Fragment shader with SDF circle
- [ ] Render static balls on screen
- **Deliverable**: 1024 colored circles displayed

### Milestone 2: Gravity & Integration (Week 1)
- [ ] Add velocity to balls
- [ ] Implement integration (position += velocity * dt)
- [ ] Add constant gravity
- [ ] World boundary collisions (bounce off edges)
- **Deliverable**: Balls fall and bounce off screen edges

### Milestone 3: Ball-Ball Collisions - Brute Force (Week 2)
- [ ] Implement O(n^2) collision detection
- [ ] Elastic collision response
- [ ] Position separation for penetration
- [ ] Verify physics correctness
- **Deliverable**: Balls bounce off each other (may be slow)

### Milestone 4: Spatial Hashing (Week 2)
- [ ] Implement spatial hash build kernel
- [ ] Parallel prefix sum for cell offsets
- [ ] 9-cell neighbor query
- [ ] Profile and verify speedup
- **Deliverable**: 120fps with 1024 balls

### Milestone 5: Containers (Week 3)
- [ ] Container data structure
- [ ] Container wall collisions
- [ ] Open/close wall mechanics
- [ ] Ball count tracking per container
- [ ] Container rendering
- **Deliverable**: Balls contained in boxes

### Milestone 6: Input Integration (Week 3)
- [ ] Accelerometer input (or simulated via keyboard)
- [ ] Map tilt to gravity direction
- [ ] Touch input for container interaction
- [ ] Swipe-to-impulse gesture
- **Deliverable**: Interactive ball pouring

### Milestone 7: Polish (Week 4)
- [ ] Sleep/wake optimization
- [ ] Visual effects (trails, glow)
- [ ] UI controls (sliders, reset)
- [ ] Performance profiling
- [ ] Demo mode (auto-tilt patterns)
- **Deliverable**: Complete demo application

---

## 11. Future Enhancements

### Visual Enhancements
- **Ball shadows**: Project circles onto "floor"
- **Glow effects**: Balls glow based on velocity
- **Particle trails**: Small particles behind fast-moving balls
- **Ball textures**: Replace solid colors with texture atlas

### Physics Enhancements
- **Variable ball sizes**: Mix of radii (still 1024 total)
- **Ball splitting**: Large balls split on impact
- **Ball merging**: Small balls combine
- **Magnetic balls**: Attract/repel forces
- **Fluid simulation**: SPH-style interactions

### Interaction Enhancements
- **Multi-touch**: Multiple simultaneous impulses
- **Ball grabbing**: Drag individual balls
- **Drawing walls**: User draws temporary barriers
- **Gravity wells**: Tap to create attraction points

### Gamification
- **Timer challenges**: Pour all balls to other side in X seconds
- **Precision mode**: Get exactly N balls in each container
- **Puzzle mode**: Navigate balls through maze of containers
- **Multiplayer**: Split screen, compete for balls

### Performance Scaling
- **4096 balls**: Use 4 threadgroups with shared spatial hash
- **16K balls**: Hierarchical spatial hash, multi-frame collision
- **Variable timestep**: Adaptive dt based on frame budget

---

## Appendix A: Key Code Patterns

### A1. SIMD Prefix Sum

```metal
// Compute exclusive prefix sum within SIMD group (32 threads)
float simd_prefix_exclusive_sum(float value, uint simd_lane) {
    float sum = value;
    if (simd_lane >= 1)  sum += simd_shuffle_up(sum, 1);
    if (simd_lane >= 2)  sum += simd_shuffle_up(sum, 2);
    if (simd_lane >= 4)  sum += simd_shuffle_up(sum, 4);
    if (simd_lane >= 8)  sum += simd_shuffle_up(sum, 8);
    if (simd_lane >= 16) sum += simd_shuffle_up(sum, 16);
    return sum - value;  // Exclusive (doesn't include self)
}
```

### A2. Elastic Collision Response

```metal
void elastic_collision(
    float2 pos_a, float2 vel_a, float mass_a,
    float2 pos_b, float2 vel_b, float mass_b,
    float restitution,
    thread float2& impulse_a,
    thread float2& impulse_b
) {
    float2 normal = normalize(pos_a - pos_b);
    float2 rel_vel = vel_a - vel_b;
    float vel_along_normal = dot(rel_vel, normal);

    // Don't resolve if separating
    if (vel_along_normal > 0) {
        impulse_a = float2(0);
        impulse_b = float2(0);
        return;
    }

    float j = -(1 + restitution) * vel_along_normal;
    j /= (1 / mass_a) + (1 / mass_b);

    impulse_a = (j / mass_a) * normal;
    impulse_b = -(j / mass_b) * normal;
}
```

### A3. Branchless Container Test

```metal
// Test if point is inside container, branchless for SIMD efficiency
bool point_in_container(float2 point, float4 bounds) {
    float4 tests = float4(
        point.x - bounds.x,          // x >= left
        bounds.x + bounds.z - point.x,  // x <= right
        point.y - bounds.y,          // y >= top
        bounds.y + bounds.w - point.y   // y <= bottom
    );
    return all(tests >= 0);
}
```

---

## Appendix B: Debug Visualizations

For development, add these debug overlays (toggle with key press):

1. **Spatial Hash Grid**: Draw cell boundaries, color by occupancy
2. **Velocity Vectors**: Line from ball center in velocity direction
3. **Collision Pairs**: Lines between colliding balls
4. **Sleep State**: Different color for sleeping balls
5. **Container Bounds**: Wireframe of collision geometry
6. **Frame Timing**: Per-phase timing breakdown graph

---

*This PRD provides a complete blueprint for implementing a visually impressive, technically sophisticated physics demo that showcases the GPU-Native OS architecture's parallel processing capabilities.*
