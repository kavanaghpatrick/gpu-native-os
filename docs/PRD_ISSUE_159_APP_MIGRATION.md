# PRD: App Migration to Megakernel (Issue #159)

## Overview

Port existing apps to megakernel format. Each app becomes a function called from the unified kernel.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| Virtual function dispatch | Switch statement (static dispatch) |
| Per-app state objects | Offsets into unified buffer |
| Dynamic pipeline binding | All code in one kernel |

## The GPU Insight

The megakernel pattern avoids dynamic dispatch:

```metal
// WRONG: Dynamic dispatch (requires CPU)
pipeline = get_pipeline_for_app(app_type);  // CPU lookup
encoder.set_pipeline(pipeline);              // CPU call

// RIGHT: Static dispatch in shader
switch (app.app_type) {
    case GAME_OF_LIFE: game_of_life_update(); break;
    case PARTICLES: particles_update(); break;
    // All code compiled into one kernel
}
```

### No Warp Divergence Concern

Each threadgroup processes ONE app slot. All threads in the threadgroup run the same app type → no divergence.

```metal
// Dispatch: 64 threadgroups, 256 threads each
// Threadgroup 0: all 256 threads run app in slot 0
// Threadgroup 1: all 256 threads run app in slot 1
// ...
// NO divergence within threadgroup
```

## Migration Pattern

### 1. Define State Struct

```metal
// Each app's state at its offset in unified buffer
struct GameOfLifeState {
    uint width;
    uint height;
    uint generation;
    uint _pad;
    // uchar grid[width * height] follows
};

struct ParticlesState {
    uint count;
    float2 gravity;
    float4 bounds;
    // Particle particles[count] follows
};

struct Particle {
    float2 position;
    float2 velocity;
    float4 color;
    float lifetime;
    float _pad[3];
};
```

### 2. Implement App Function

```metal
inline void game_of_life_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device Vertex* unified_vertices,
    uint tid,
    uint thread_count
) {
    // Get my state
    device GameOfLifeState* state =
        (device GameOfLifeState*)(unified_state + app->state_offset);
    device uchar* grid = (device uchar*)(state + 1);

    uint grid_size = state->width * state->height;

    // Parallel update - each thread handles some cells
    uint cells_per_thread = (grid_size + thread_count - 1) / thread_count;
    uint start = tid * cells_per_thread;
    uint end = min(start + cells_per_thread, grid_size);

    for (uint i = start; i < end; i++) {
        uint x = i % state->width;
        uint y = i / state->width;
        uint neighbors = count_neighbors(grid, x, y, state->width, state->height);

        // Conway's rules
        bool alive = grid[i] != 0;
        bool next = (alive && (neighbors == 2 || neighbors == 3)) ||
                   (!alive && neighbors == 3);
        grid[i] = next ? 1 : 0;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Thread 0 generates vertices
    if (tid == 0) {
        state->generation++;
        device Vertex* my_verts = unified_vertices + app->vertex_offset / sizeof(Vertex);
        app->vertex_count = generate_grid_vertices(state, grid, my_verts, app->window);
    }
}

inline void particles_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state,
    device Vertex* unified_vertices,
    uint tid,
    uint thread_count
) {
    device ParticlesState* state =
        (device ParticlesState*)(unified_state + app->state_offset);
    device Particle* particles = (device Particle*)(state + 1);

    // Each thread updates some particles
    uint per_thread = (state->count + thread_count - 1) / thread_count;
    uint start = tid * per_thread;
    uint end = min(start + per_thread, state->count);

    for (uint i = start; i < end; i++) {
        // Physics
        particles[i].velocity += state->gravity * 0.016;
        particles[i].position += particles[i].velocity * 0.016;

        // Bounce
        if (particles[i].position.x < state->bounds.x) {
            particles[i].position.x = state->bounds.x;
            particles[i].velocity.x *= -0.8;
        }
        // ... other bounds

        particles[i].lifetime -= 0.016;
    }

    // Generate vertices (each thread generates for its particles)
    if (tid < state->count) {
        device Vertex* my_verts = unified_vertices + app->vertex_offset / sizeof(Vertex);
        generate_particle_quad(particles[tid], my_verts + tid * 6);
    }

    if (tid == 0) {
        app->vertex_count = state->count * 6;
    }
}
```

### 3. Add to Megakernel Switch

```metal
kernel void gpu_app_megakernel(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    device Vertex* unified_vertices [[buffer(3)]],
    device GpuWindow* windows [[buffer(4)]],
    constant uint& frame_number [[buffer(5)]],
    uint slot_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (slot_id >= header->max_slots) return;

    device GpuAppDescriptor* app = &apps[slot_id];

    if (!should_i_run(app, frame_number)) return;

    // Static dispatch - no divergence within threadgroup
    switch (app->app_type) {
        case APP_TYPE_GAME_OF_LIFE:
            game_of_life_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_PARTICLES:
            particles_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_MANDELBROT:
            mandelbrot_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_BOIDS:
            boids_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_METABALLS:
            metaballs_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_WAVES:
            waves_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_TEXT_EDITOR:
            text_editor_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_FILESYSTEM:
            filesystem_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_TERMINAL:
            terminal_update(app, unified_state, unified_vertices, tid, tg_size);
            break;

        case APP_TYPE_DOCUMENT:
            document_update(app, unified_state, unified_vertices, tid, tg_size);
            break;
    }

    // Update frame tracking
    if (tid == 0) {
        app->last_run_frame = frame_number;
        app->flags &= ~APP_FLAG_DIRTY;
    }
}
```

### 4. App Type Registry (Rust)

```rust
pub struct AppTypeInfo {
    pub type_id: u32,
    pub name: &'static str,
    pub state_size: u32,
    pub vertex_size: u32,
    pub thread_count: u32,
}

pub const APP_TYPES: &[AppTypeInfo] = &[
    AppTypeInfo {
        type_id: app_type::GAME_OF_LIFE,
        name: "Game of Life",
        state_size: 16 + 128 * 128,  // Header + grid
        vertex_size: 128 * 128 * 6 * 32,
        thread_count: 256,
    },
    AppTypeInfo {
        type_id: app_type::PARTICLES,
        name: "Particles",
        state_size: 32 + 10000 * 32,  // Header + particles
        vertex_size: 10000 * 6 * 32,
        thread_count: 256,
    },
    // ... other apps
];

impl GpuAppSystem {
    pub fn launch_by_type(&mut self, type_id: u32) -> Option<u32> {
        let info = APP_TYPES.iter().find(|t| t.type_id == type_id)?;
        self.launch_app(type_id, info.state_size, info.vertex_size)
    }
}
```

## Migration Checklist

For each app:

- [ ] Define state struct in Metal
- [ ] Implement update function
- [ ] Add to megakernel switch
- [ ] Add to APP_TYPES registry
- [ ] Write migration test
- [ ] Verify output matches original

## Tests

```rust
#[test]
fn test_game_of_life_runs() {
    let mut system = GpuAppSystem::new(&device)?;

    let slot = system.launch_by_type(app_type::GAME_OF_LIFE).unwrap();
    system.create_window(slot, Rect::new(0.0, 0.0, 200.0, 200.0));

    for _ in 0..10 {
        system.mark_dirty(slot);
        system.run_frame();
    }

    let app = system.get_app(slot).unwrap();
    assert!(app.vertex_count > 0);
    assert_eq!(app.last_run_frame, 10);
}

#[test]
fn test_multiple_app_types_together() {
    let mut system = GpuAppSystem::new(&device)?;

    let gol = system.launch_by_type(app_type::GAME_OF_LIFE).unwrap();
    let particles = system.launch_by_type(app_type::PARTICLES).unwrap();
    let mandelbrot = system.launch_by_type(app_type::MANDELBROT).unwrap();

    system.mark_all_dirty();
    system.run_frame();

    // All should have run
    assert_eq!(system.get_app(gol).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(particles).unwrap().last_run_frame, 1);
    assert_eq!(system.get_app(mandelbrot).unwrap().last_run_frame, 1);
}

#[test]
fn test_text_editor_input() {
    let mut system = GpuAppSystem::new(&device)?;

    let editor = system.launch_by_type(app_type::TEXT_EDITOR).unwrap();
    system.set_focus(editor);

    // Type some text
    system.queue_input(InputEvent::key_down(KEY_H));
    system.queue_input(InputEvent::key_down(KEY_I));
    system.process_input();
    system.run_frame();

    // Verify text was added (read state)
    let state = system.read_app_state::<TextEditorState>(editor);
    assert!(state.buffer_size >= 2);
}
```

## Success Metrics

1. **All apps ported**: 11/11
2. **Feature parity**: 100%
3. **No CPU dispatch**: Zero per-app CPU involvement
4. **Performance**: >= original (usually better)
5. **Code size**: Megakernel < sum of individual kernels (shared utilities)
