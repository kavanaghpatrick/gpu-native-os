# PRD: GPU-Native OS Demo - 32x32 Game of Life

**Version**: 1.0
**Date**: 2026-01-23
**Status**: Ready for Implementation

---

## 1. Overview

### What It Is

A **Conway's Game of Life** implementation running entirely within a single GPU threadgroup of 1024 threads. Each thread owns exactly one cell in a 32x32 grid (32 x 32 = 1024), creating a 1:1 mapping between threads and cells.

### Why It's Impressive

This demo showcases the GPU-Native OS architecture's core strengths:

| Traditional Approach | GPU-Native Approach |
|---------------------|---------------------|
| CPU calculates neighbor counts | Each thread reads 8 neighbors via SIMD shuffle |
| CPU updates cells sequentially | All 1024 cells update simultaneously |
| CPU copies state to GPU | State never leaves GPU memory |
| CPU handles mouse clicks | GPU performs hit-testing in parallel |
| Multiple render passes | Single compute kernel + fragment pass |

**Key Innovations Demonstrated**:
1. **Perfect Parallelism**: 1024 threads = 1024 cells = 100% utilization
2. **Zero CPU Game Logic**: All simulation runs on GPU
3. **SIMD Neighbor Access**: `simd_shuffle` for same-row neighbors, threadgroup memory for vertical
4. **Integrated Input**: Click-to-toggle cells without CPU round-trip
5. **Real-time Visualization**: Smooth animations, color gradients, generation counter

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Frame Rate | 120 FPS | VSync locked |
| Generations/sec | 10-60 | User-adjustable speed |
| Input Latency | <16ms | Click → visual feedback |
| GPU Memory | <64KB | Minimal footprint |
| CPU Usage | <1% | Dispatch only |

---

## 2. User Experience

### What the User Sees

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                                                      │    │
│   │     ██  ██                                          │    │
│   │       ████                    ████                  │    │
│   │       ██                      █  █                  │    │
│   │                               ████                  │    │
│   │                                                      │    │
│   │            ███                                       │    │
│   │            █                                         │    │
│   │             █                                        │    │
│   │                                                      │    │
│   │                                                      │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
│   [▶ Play]  [⏸ Pause]  [↻ Step]  Speed: [████░░░░░░]         │
│   [Clear]   [Random]   [Glider]  Gen: 1,247  Pop: 42         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### User Interactions

| Input | Action | Response Time |
|-------|--------|---------------|
| **Click cell** | Toggle alive/dead | Immediate (<8ms) |
| **Click Play** | Start simulation | Next frame |
| **Click Pause** | Stop simulation | Next frame |
| **Click Step** | Advance one generation | Next frame |
| **Drag Speed slider** | Adjust simulation speed | Continuous |
| **Click Clear** | Kill all cells | Next frame |
| **Click Random** | Randomize ~30% alive | Next frame |
| **Click Glider** | Spawn glider at center | Next frame |
| **Hover cell** | Highlight cell | Immediate |

### Visual Feedback

1. **Alive cells**: Solid color with subtle glow
2. **Dead cells**: Dark with faint grid lines
3. **Hovered cell**: Bright outline
4. **Recently born**: Fade-in animation (0.1s)
5. **Recently died**: Fade-out to ghost (0.2s)
6. **Active buttons**: Highlight on hover, depress on click

---

## 3. Technical Architecture

### Thread Assignment

Every thread has a fixed identity based on its thread index:

```metal
// Thread ID → Cell Coordinates
uint tid = thread_index_in_threadgroup;  // 0-1023
uint cell_x = tid % 32;                   // 0-31 (column)
uint cell_y = tid / 32;                   // 0-31 (row)

// SIMD Group Identity
uint simd_id = tid / 32;                  // 0-31 (which row)
uint simd_lane = tid % 32;                // 0-31 (which column in row)
```

This creates a natural mapping where:
- Each **SIMD group** (32 threads) owns one **row** of the grid
- Each **thread** within a SIMD group owns one **cell** in that row
- Horizontal neighbors are accessible via **`simd_shuffle`**
- Vertical neighbors require **threadgroup memory** access

### Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEVICE MEMORY (Persistent)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CELL STATE BUFFER (2KB)                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ CellState cells[1024]  // 2 bytes per cell                 │ │
│  │   - alive: u8          // 0 or 1                           │ │
│  │   - age: u8            // generations since state change   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  SIMULATION STATE (64B)                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ generation: u32        // current generation count         │ │
│  │ population: u32        // number of alive cells            │ │
│  │ running: u32           // 0=paused, 1=running              │ │
│  │ speed: f32             // generations per second (1-60)    │ │
│  │ accumulator: f32       // time since last generation       │ │
│  │ cursor_cell: u32       // cell under cursor (or 0xFFFF)    │ │
│  │ pending_toggle: u32    // cell to toggle (or 0xFFFF)       │ │
│  │ pending_action: u32    // CLEAR, RANDOM, GLIDER, etc.      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  INPUT QUEUE (4KB) - Ring buffer from IOKit                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ head: atomic_uint                                           │ │
│  │ tail: atomic_uint                                           │ │
│  │ events[256]: InputEvent                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  WIDGET STATE (1KB) - UI buttons and controls                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ widgets[16]: WidgetCompact  // Play, Pause, Step, etc.     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  VERTEX BUFFER (48KB) - Generated geometry for rendering        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ cell_vertices[1024 * 6]: Vertex  // 6 verts per cell quad  │ │
│  │ widget_vertices[16 * 6]: Vertex  // 6 verts per widget     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  DRAW ARGUMENTS (32B) - Indirect draw parameters                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ cell_draw: DrawArguments                                    │ │
│  │ widget_draw: DrawArguments                                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Total Device Memory: ~56KB                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   THREADGROUP MEMORY (On-Chip)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CELL SNAPSHOT (1KB) - Current generation state                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ current_alive[1024]: bool  // packed as 128 bytes          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  NEIGHBOR COUNTS (1KB)                                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ neighbors[1024]: u8  // 0-8 neighbor count per cell        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  INPUT STAGING (256B)                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ pending_events[8]: InputEvent  // events this frame        │ │
│  │ event_count: u32                                            │ │
│  │ cursor_x: f32                                               │ │
│  │ cursor_y: f32                                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  REDUCTION SCRATCH (128B)                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ simd_population[32]: u32  // per-row alive counts          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Total Threadgroup Memory: ~2.5KB (well under 32KB limit)       │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Frame Pipeline

```
Frame Start (VSync @ 120Hz)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: INPUT COLLECTION (0.05ms)                              │
│ ─────────────────────────────────────────────────────────────── │
│ Threads 0-7: Read up to 8 events from input queue               │
│ Thread 0: Update cursor position, detect pending toggle         │
│ All threads: threadgroup_barrier()                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: HIT TESTING (0.05ms)                                   │
│ ─────────────────────────────────────────────────────────────── │
│ Threads 0-1023: Each checks if cursor is in its cell            │
│ Threads 0-15: Each checks if cursor is in its widget            │
│ SIMD ballot → find topmost hit                                  │
│ All threads: threadgroup_barrier()                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: PROCESS ACTIONS (0.02ms)                               │
│ ─────────────────────────────────────────────────────────────── │
│ Thread 0: Check pending_action (CLEAR, RANDOM, GLIDER)          │
│ All threads: Apply action if pending (parallel cell updates)    │
│ All threads: threadgroup_barrier()                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: CELL TOGGLE (0.01ms)                                   │
│ ─────────────────────────────────────────────────────────────── │
│ If mouse clicked on cell: toggle that cell's state              │
│ Reset age counter for toggled cell                              │
│ All threads: threadgroup_barrier()                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: SIMULATION STEP (0.1ms) - if running & time elapsed    │
│ ─────────────────────────────────────────────────────────────── │
│ 5a. Snapshot current state to threadgroup memory                │
│     Each thread: shared.current_alive[tid] = cells[tid].alive   │
│     threadgroup_barrier()                                       │
│                                                                  │
│ 5b. Count neighbors (parallel)                                  │
│     Each thread reads 8 neighbors:                              │
│       - Left/Right: simd_shuffle_xor or wrap                    │
│       - Up/Down: shared.current_alive[tid ± 32]                 │
│       - Diagonals: combination of above                         │
│     threadgroup_barrier()                                       │
│                                                                  │
│ 5c. Apply Game of Life rules (parallel)                         │
│     alive = (neighbors == 3) || (alive && neighbors == 2)       │
│     Update cells[tid].alive and cells[tid].age                  │
│     threadgroup_barrier()                                       │
│                                                                  │
│ 5d. Count population (SIMD reduction)                           │
│     simd_sum within each row → shared.simd_population[simd_id]  │
│     Thread 0: sum all 32 rows → simulation.population           │
│     Thread 0: increment generation counter                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: GEOMETRY GENERATION (0.3ms)                            │
│ ─────────────────────────────────────────────────────────────── │
│ Each thread generates 6 vertices for its cell:                  │
│   - Position based on cell_x, cell_y                            │
│   - Color based on alive state and age                          │
│   - Highlight if hovered                                        │
│                                                                  │
│ Threads 0-15: Generate widget quads                             │
│ Thread 0: Write draw argument counts                            │
│ All threads: threadgroup_barrier()                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPUTE PASS COMPLETE - Total: ~0.5ms                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ RENDER PASS (Fragment Shaders) - ~1.0ms                         │
│ ─────────────────────────────────────────────────────────────── │
│ Draw 1: Cell quads (indirect draw, 1024 * 6 vertices)           │
│   - SDF rounded corners                                         │
│   - Glow effect for alive cells                                 │
│   - Grid lines between cells                                    │
│                                                                  │
│ Draw 2: Widget quads (indirect draw, 16 * 6 vertices)           │
│   - Button styling                                              │
│   - Text rendering (generation count, population)               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRESENT - Display current frame                                  │
│ Total Frame Time: ~1.5ms (budget: 8.33ms at 120Hz)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Structures

### Rust Structs (must match Metal)

```rust
/// Per-cell state - 2 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CellState {
    pub alive: u8,      // 0 = dead, 1 = alive
    pub age: u8,        // generations since last state change (capped at 255)
}

/// Simulation control state - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimulationState {
    pub generation: u32,        // current generation count
    pub population: u32,        // number of alive cells
    pub running: u32,           // 0 = paused, 1 = running
    pub speed: f32,             // generations per second (1.0 - 60.0)
    pub accumulator: f32,       // time accumulated since last step
    pub cursor_cell: u32,       // cell index under cursor (0xFFFFFFFF = none)
    pub pending_toggle: u32,    // cell to toggle on click (0xFFFFFFFF = none)
    pub pending_action: u32,    // ACTION_NONE, ACTION_CLEAR, ACTION_RANDOM, etc.
}

/// Action constants
pub const ACTION_NONE: u32 = 0;
pub const ACTION_CLEAR: u32 = 1;
pub const ACTION_RANDOM: u32 = 2;
pub const ACTION_GLIDER: u32 = 3;
pub const ACTION_TOGGLE_RUN: u32 = 4;
pub const ACTION_STEP: u32 = 5;

/// Vertex for cell rendering - 32 bytes
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CellVertex {
    pub position: [f32; 2],     // 8 bytes - screen position
    pub uv: [f32; 2],           // 8 bytes - texture coordinates (for SDF)
    pub color: [f32; 4],        // 16 bytes - RGBA color
}

/// Memory buffer layout
pub struct GameOfLifeMemory {
    pub cells: Buffer,          // CellState[1024] = 2KB
    pub simulation: Buffer,     // SimulationState = 32B
    pub vertices: Buffer,       // CellVertex[1024*6 + 16*6] = ~200KB
    pub input_queue: Buffer,    // InputQueue = 4KB
    pub widgets: Buffer,        // WidgetCompact[16] = 384B
    pub draw_args: Buffer,      // DrawArguments[2] = 32B
}

impl GameOfLifeMemory {
    pub const CELL_COUNT: usize = 1024;
    pub const GRID_SIZE: usize = 32;
    pub const WIDGET_COUNT: usize = 16;

    pub fn total_size() -> usize {
        2048 + 32 + 200704 + 4096 + 384 + 32  // ~207KB
    }
}
```

### Metal Structs

```metal
// Cell state - 2 bytes
struct CellState {
    uchar alive;        // 0 or 1
    uchar age;          // 0-255
};

// Simulation state - 32 bytes
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

// Vertex output - 32 bytes
struct CellVertex {
    float2 position;
    float2 uv;
    float4 color;
};

// Threadgroup shared memory
struct SharedMemory {
    bool current_alive[1024];       // 1024 bytes (could pack to 128)
    uchar neighbor_count[1024];     // 1024 bytes
    uint simd_population[32];       // 128 bytes
    float cursor_x;                 // 4 bytes
    float cursor_y;                 // 4 bytes
    uint event_count;               // 4 bytes
    uint hit_cell;                  // 4 bytes (cell under cursor)
    uint hit_widget;                // 4 bytes (widget under cursor)
};
```

### Memory Size Summary

| Buffer | Size | Purpose |
|--------|------|---------|
| cells | 2 KB | 1024 cell states |
| simulation | 32 B | Generation, speed, flags |
| vertices | 200 KB | Cell + widget geometry |
| input_queue | 4 KB | Mouse/keyboard events |
| widgets | 384 B | UI button state |
| draw_args | 32 B | Indirect draw params |
| **Total Device** | **~207 KB** | |
| **Threadgroup** | **~2.5 KB** | Scratch space |

---

## 5. Shader Pseudocode

### Main Compute Kernel

```metal
kernel void game_of_life_kernel(
    device CellState* cells [[buffer(0)]],
    device SimulationState* sim [[buffer(1)]],
    device CellVertex* vertices [[buffer(2)]],
    device InputQueue* input [[buffer(3)]],
    device DrawArguments* draw_args [[buffer(4)]],
    constant float& delta_time [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Thread identity
    uint cell_x = tid % 32;
    uint cell_y = tid / 32;

    threadgroup SharedMemory shared;

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: INPUT COLLECTION
    // ═══════════════════════════════════════════════════════════════════

    if (tid == 0) {
        uint head = atomic_load(&input->head);
        uint tail = atomic_load(&input->tail);
        shared.event_count = min(head - tail, 8u);
        shared.hit_cell = 0xFFFFFFFF;
        shared.hit_widget = 0xFFFFFFFF;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read events and update cursor
    if (tid < shared.event_count) {
        InputEvent ev = input->events[(input->tail + tid) % 256];
        if (ev.type == EVENT_MOUSE_MOVE) {
            shared.cursor_x = ev.position.x;
            shared.cursor_y = ev.position.y;
        }
        if (ev.type == EVENT_MOUSE_DOWN && tid == 0) {
            // Will resolve hit in next phase
            sim->pending_toggle = 0xFFFFFFFE;  // Signal: resolve hit
        }
    }

    if (tid == 0) {
        atomic_store(&input->tail, input->tail + shared.event_count);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: HIT TESTING
    // ═══════════════════════════════════════════════════════════════════

    // Calculate cell bounds (assuming grid fills 0.1-0.9 of screen)
    float grid_start = 0.1;
    float grid_size = 0.8;
    float cell_size = grid_size / 32.0;

    float cell_left = grid_start + cell_x * cell_size;
    float cell_top = grid_start + cell_y * cell_size;

    bool cursor_in_cell =
        shared.cursor_x >= cell_left &&
        shared.cursor_x < cell_left + cell_size &&
        shared.cursor_y >= cell_top &&
        shared.cursor_y < cell_top + cell_size;

    // SIMD ballot to find hit cell
    simd_vote hits = simd_ballot(cursor_in_cell);
    if (cursor_in_cell && simd_ctz(hits) == simd_lane) {
        // This thread is the first hit in its SIMD group
        atomic_store(&shared.hit_cell, tid);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Resolve pending click
    if (tid == 0 && sim->pending_toggle == 0xFFFFFFFE) {
        sim->pending_toggle = shared.hit_cell;
    }

    // Update cursor_cell for hover highlight
    if (tid == 0) {
        sim->cursor_cell = shared.hit_cell;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: PROCESS ACTIONS
    // ═══════════════════════════════════════════════════════════════════

    uint action = sim->pending_action;

    if (action == ACTION_CLEAR) {
        cells[tid].alive = 0;
        cells[tid].age = 0;
    }
    else if (action == ACTION_RANDOM) {
        // Use thread ID and generation as seed for pseudo-random
        uint seed = tid ^ (sim->generation * 1664525u + 1013904223u);
        seed = seed * 1664525u + 1013904223u;
        cells[tid].alive = (seed % 100) < 30 ? 1 : 0;  // ~30% alive
        cells[tid].age = 0;
    }
    else if (action == ACTION_GLIDER) {
        // Spawn glider at center (cells 15,15 area)
        // Glider pattern:   .#.
        //                   ..#
        //                   ###
        cells[tid].alive = 0;
        if (tid == (14*32 + 15)) cells[tid].alive = 1;  // .#.
        if (tid == (15*32 + 16)) cells[tid].alive = 1;  // ..#
        if (tid == (16*32 + 14)) cells[tid].alive = 1;  // ###
        if (tid == (16*32 + 15)) cells[tid].alive = 1;
        if (tid == (16*32 + 16)) cells[tid].alive = 1;
        cells[tid].age = 0;
    }

    if (tid == 0) {
        sim->pending_action = ACTION_NONE;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 4: CELL TOGGLE (from click)
    // ═══════════════════════════════════════════════════════════════════

    if (sim->pending_toggle == tid) {
        cells[tid].alive = 1 - cells[tid].alive;  // Toggle
        cells[tid].age = 0;
    }

    if (tid == 0) {
        sim->pending_toggle = 0xFFFFFFFF;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 5: SIMULATION STEP (if running)
    // ═══════════════════════════════════════════════════════════════════

    bool should_step = false;
    if (tid == 0 && sim->running) {
        sim->accumulator += delta_time;
        float step_interval = 1.0 / sim->speed;
        if (sim->accumulator >= step_interval) {
            sim->accumulator -= step_interval;
            should_step = true;
        }
    }

    // Broadcast should_step to all threads
    threadgroup bool do_step;
    if (tid == 0) do_step = should_step;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (do_step) {
        // 5a. Snapshot current state
        shared.current_alive[tid] = cells[tid].alive;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5b. Count neighbors
        uint count = 0;

        // Horizontal neighbors (same row) - use simd_shuffle
        bool left_alive = simd_shuffle_down(shared.current_alive[tid], 1);
        bool right_alive = simd_shuffle_up(shared.current_alive[tid], 1);

        // Handle wrap-around for edges
        if (cell_x == 0) left_alive = shared.current_alive[tid + 31];
        if (cell_x == 31) right_alive = shared.current_alive[tid - 31];

        // Vertical neighbors (different rows) - use threadgroup memory
        uint up_idx = (cell_y == 0) ? tid + 992 : tid - 32;    // Wrap top
        uint down_idx = (cell_y == 31) ? tid - 992 : tid + 32;  // Wrap bottom

        bool up_alive = shared.current_alive[up_idx];
        bool down_alive = shared.current_alive[down_idx];

        // Diagonal neighbors
        uint up_left = (cell_y == 0) ? up_idx + 31 : up_idx;
        uint up_right = (cell_y == 0) ? up_idx - 31 : up_idx;
        if (cell_x == 0) up_left = up_idx + 31; else up_left = up_idx - 1;
        if (cell_x == 31) up_right = up_idx - 31; else up_right = up_idx + 1;

        uint down_left = (cell_y == 31) ? down_idx + 31 : down_idx;
        uint down_right = (cell_y == 31) ? down_idx - 31 : down_idx;
        if (cell_x == 0) down_left = down_idx + 31; else down_left = down_idx - 1;
        if (cell_x == 31) down_right = down_idx - 31; else down_right = down_idx + 1;

        count = uint(left_alive) + uint(right_alive) +
                uint(up_alive) + uint(down_alive) +
                uint(shared.current_alive[up_left]) +
                uint(shared.current_alive[up_right]) +
                uint(shared.current_alive[down_left]) +
                uint(shared.current_alive[down_right]);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5c. Apply Game of Life rules
        bool was_alive = shared.current_alive[tid];
        bool now_alive = (count == 3) || (was_alive && count == 2);

        cells[tid].alive = now_alive ? 1 : 0;

        if (was_alive != now_alive) {
            cells[tid].age = 0;  // Reset age on state change
        } else if (cells[tid].age < 255) {
            cells[tid].age++;    // Increment age (capped)
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5d. Count population using SIMD reduction
        uint my_alive = cells[tid].alive;
        uint row_sum = simd_sum(my_alive);

        if (simd_lane == 0) {
            shared.simd_population[simd_id] = row_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint total = 0;
            for (uint i = 0; i < 32; i++) {
                total += shared.simd_population[i];
            }
            sim->population = total;
            sim->generation++;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 6: GEOMETRY GENERATION
    // ═══════════════════════════════════════════════════════════════════

    // Calculate cell visual properties
    bool alive = cells[tid].alive;
    uchar age = cells[tid].age;
    bool hovered = (tid == sim->cursor_cell);

    // Color based on state and age
    float4 color;
    if (alive) {
        // Alive: bright green, fading slightly with age
        float brightness = 1.0 - (float(age) / 255.0) * 0.3;
        color = float4(0.2 * brightness, 0.9 * brightness, 0.3 * brightness, 1.0);
    } else {
        // Dead: dark gray, briefly shows "ghost" after death
        float ghost = (age < 20) ? (1.0 - float(age) / 20.0) * 0.3 : 0.0;
        color = float4(0.1 + ghost, 0.1 + ghost * 0.5, 0.1 + ghost * 0.3, 1.0);
    }

    // Highlight on hover
    if (hovered) {
        color.rgb += float3(0.2, 0.2, 0.2);
    }

    // Generate 6 vertices for cell quad (2 triangles)
    float padding = 0.002;  // Small gap between cells
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
        draw_args[0].vertex_count = 1024 * 6;
        draw_args[0].instance_count = 1;
        draw_args[0].vertex_start = 0;
        draw_args[0].base_instance = 0;
    }
}
```

### Fragment Shader

```metal
struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut cell_vertex(
    const device CellVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    CellVertex v = vertices[vid];
    VertexOut out;

    // Convert from normalized [0,1] to clip space [-1,1]
    out.position = float4(v.position * 2.0 - 1.0, 0.0, 1.0);
    out.position.y = -out.position.y;  // Flip Y for Metal coordinates
    out.uv = v.uv;
    out.color = v.color;

    return out;
}

fragment float4 cell_fragment(VertexOut in [[stage_in]]) {
    // Optional: SDF rounded corners
    float2 p = in.uv - 0.5;
    float corner_radius = 0.15;
    float2 q = abs(p) - (0.5 - corner_radius);
    float d = length(max(q, 0.0)) - corner_radius;

    float alpha = smoothstep(0.02, -0.02, d);

    // Optional: subtle glow for alive cells
    float glow = smoothstep(0.2, -0.1, d) * 0.1;
    float3 final_color = in.color.rgb + glow;

    return float4(final_color, in.color.a * alpha);
}
```

---

## 6. Widget Integration

### Widget Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Widget ID  │  Type    │  Bounds (x,y,w,h)   │  Action           │
├─────────────┼──────────┼─────────────────────┼───────────────────┤
│  0          │  Button  │  0.10, 0.92, 0.08, 0.05  │  Toggle Play/Pause │
│  1          │  Button  │  0.20, 0.92, 0.08, 0.05  │  Step              │
│  2          │  Slider  │  0.35, 0.92, 0.20, 0.05  │  Speed (1-60)      │
│  3          │  Button  │  0.60, 0.92, 0.08, 0.05  │  Clear             │
│  4          │  Button  │  0.70, 0.92, 0.08, 0.05  │  Random            │
│  5          │  Button  │  0.80, 0.92, 0.08, 0.05  │  Glider            │
│  6          │  Label   │  0.10, 0.05, 0.15, 0.04  │  "Gen: X"          │
│  7          │  Label   │  0.30, 0.05, 0.15, 0.04  │  "Pop: X"          │
└─────────────┴──────────┴─────────────────────┴───────────────────┘
```

### Widget Hit Testing (in kernel)

```metal
// After cell hit testing, test widgets (threads 0-15)
if (tid < 8) {
    Widget w = widgets[tid];
    bool hit = point_in_rect(float2(shared.cursor_x, shared.cursor_y), w.bounds);

    if (hit) {
        atomic_store(&shared.hit_widget, tid);
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);

// Process widget click
if (tid == 0 && mouse_clicked && shared.hit_widget != 0xFFFFFFFF) {
    uint widget = shared.hit_widget;

    switch (widget) {
        case 0:  // Play/Pause
            sim->running = 1 - sim->running;
            break;
        case 1:  // Step
            sim->pending_action = ACTION_STEP;
            break;
        case 3:  // Clear
            sim->pending_action = ACTION_CLEAR;
            break;
        case 4:  // Random
            sim->pending_action = ACTION_RANDOM;
            break;
        case 5:  // Glider
            sim->pending_action = ACTION_GLIDER;
            break;
    }
}
```

---

## 7. Visual Design

### Color Palette

| Element | Color (RGB) | Hex | Notes |
|---------|-------------|-----|-------|
| Background | (15, 15, 20) | `#0F0F14` | Near-black with blue tint |
| Grid lines | (30, 30, 40) | `#1E1E28` | Subtle grid |
| Dead cell | (25, 25, 30) | `#19191E` | Slightly lighter than bg |
| Alive cell | (50, 230, 80) | `#32E650` | Bright green |
| Alive (aged) | (35, 180, 60) | `#23B43C` | Slightly dimmer |
| Ghost (just died) | (80, 60, 50) | `#503C32` | Warm fade-out |
| Hover highlight | +30% brightness | - | Added to base color |
| Button normal | (40, 40, 50) | `#282832` | Dark gray-blue |
| Button hover | (60, 60, 75) | `#3C3C4B` | Lighter |
| Button active | (80, 80, 100) | `#505064` | Brightest |
| Text | (200, 200, 210) | `#C8C8D2` | Off-white |

### Animation Timings

| Animation | Duration | Easing |
|-----------|----------|--------|
| Cell birth | 100ms | ease-out |
| Cell death (ghost) | 200ms | linear |
| Button hover | 50ms | linear |
| Button press | 30ms | ease-in |
| Speed slider | instant | - |

### Grid Appearance

```
Cell Layout (32x32 grid in 80% of screen):

├── 10% margin ──├──────────── 80% grid ────────────├── 10% margin ──┤

Each cell: 2.5% of screen width/height
Cell padding: 0.2% (creates visible grid lines)
Corner radius: 15% of cell size (subtle rounding)
```

---

## 8. Performance Targets

### Frame Budget (8.33ms at 120Hz)

| Phase | Target Time | Maximum | Notes |
|-------|-------------|---------|-------|
| Input collection | 0.05ms | 0.1ms | Trivial |
| Hit testing | 0.05ms | 0.1ms | SIMD ballot |
| Actions | 0.02ms | 0.05ms | Rare |
| Cell toggle | 0.01ms | 0.02ms | Rare |
| Simulation step | 0.10ms | 0.2ms | Only when running |
| Geometry gen | 0.30ms | 0.5ms | Main compute cost |
| **Compute total** | **0.53ms** | **1.0ms** | |
| Fragment render | 1.0ms | 2.0ms | 1024 quads + effects |
| **Frame total** | **1.5ms** | **3.0ms** | 5.3ms headroom |

### Scaling Behavior

| Scenario | Expected FPS | Notes |
|----------|--------------|-------|
| Idle (paused, no input) | 120 | Minimal work |
| Running, sparse | 120 | ~10% cells alive |
| Running, dense | 120 | ~50% cells alive |
| Running, 60 gen/sec | 120 | Max simulation speed |
| Continuous clicking | 120 | Rapid toggle |

### Memory Bandwidth

| Operation | Data Size | Frequency |
|-----------|-----------|-----------|
| Read cell state | 2KB | Every sim step |
| Write cell state | 2KB | Every sim step |
| Write vertices | 200KB | Every frame |
| Read input queue | ~224B | Every frame (8 events max) |
| **Total per frame** | ~205KB | Well within GPU bandwidth |

---

## 9. Implementation Milestones

### Milestone 1: Static Grid (Day 1)

**Goal**: Render a 32x32 grid of cells with no interaction

- [ ] Create `CellState` buffer (1024 cells)
- [ ] Create compute kernel that generates cell vertices
- [ ] Create fragment shader for cell quads
- [ ] Verify all 1024 threads execute (debug output)
- [ ] Render static pattern (checkerboard)

**Deliverable**: Window showing colored 32x32 grid

### Milestone 2: Simulation Logic (Day 2)

**Goal**: Game of Life rules running on GPU

- [ ] Implement neighbor counting with wrap-around
- [ ] Implement birth/death rules
- [ ] Add simulation state (generation, running flag)
- [ ] Add speed control (accumulator timing)
- [ ] Test with known patterns (glider, blinker)

**Deliverable**: Self-running simulation with correct behavior

### Milestone 3: Input Handling (Day 3)

**Goal**: Click to toggle cells

- [ ] Integrate input queue from existing GPU-OS code
- [ ] Implement cell hit testing (cursor → cell index)
- [ ] Implement click → toggle pipeline
- [ ] Add hover highlight visual

**Deliverable**: Interactive cell toggling

### Milestone 4: UI Widgets (Day 4)

**Goal**: Buttons for Play/Pause/Step/Clear/Random/Glider

- [ ] Add widget buffer (8 button widgets)
- [ ] Implement widget hit testing
- [ ] Implement widget actions (pending_action system)
- [ ] Generate widget geometry
- [ ] Add widget visual states (normal/hover/active)

**Deliverable**: Fully interactive demo with UI

### Milestone 5: Polish (Day 5)

**Goal**: Production-quality visuals

- [ ] Add cell fade-in animation (birth)
- [ ] Add ghost fade-out animation (death)
- [ ] Add SDF rounded corners
- [ ] Add subtle glow effect for alive cells
- [ ] Add generation/population text labels
- [ ] Performance profiling and optimization

**Deliverable**: Complete, polished demo

---

## 10. Future Enhancements

### Phase 2 Features (Not in Initial Scope)

| Feature | Complexity | Value | Notes |
|---------|------------|-------|-------|
| **Larger grids** | Medium | High | 64x64 needs 4 dispatches or multi-threadgroup |
| **Pattern library** | Low | Medium | Presets: Gosper gun, spaceship, oscillators |
| **Brush painting** | Low | Medium | Drag to draw lines of cells |
| **Toroidal vs bounded** | Low | Low | Already toroidal (wrap-around) |
| **Cell trails** | Low | Medium | Cells leave fading path |
| **Color by age** | Low | Low | Rainbow gradient based on survival time |
| **Statistics graph** | Medium | Medium | Population over time chart |
| **Save/load patterns** | Medium | Medium | RLE format import/export |
| **Speed presets** | Low | Low | Pause, 1x, 10x, 60x buttons |
| **Zoom/pan** | High | Medium | Requires viewport transform |
| **64-bit patterns** | High | High | Use u64 packed rows for speed |

### Technical Improvements

1. **Packed cell storage**: Store 32 cells per u32 (bit packing)
2. **Change detection**: Skip simulation if no cells changed
3. **Dirty rectangles**: Only regenerate changed cell vertices
4. **Compute-to-render fence**: Proper synchronization for indirect draw
5. **Double buffering**: Ping-pong cell buffers to avoid barriers

### Integration with GPU-OS

This demo serves as a **proof-of-concept** for the GPU-Native OS:

- Demonstrates single-threadgroup viability
- Tests input pipeline latency
- Validates SIMD neighbor access patterns
- Benchmarks geometry generation throughput
- Provides template for future demos (falling sand, fluid sim, etc.)

---

## Appendix A: Game of Life Rules Reference

Conway's Game of Life rules:

1. **Birth**: A dead cell with exactly 3 neighbors becomes alive
2. **Survival**: An alive cell with 2 or 3 neighbors stays alive
3. **Death by isolation**: An alive cell with <2 neighbors dies
4. **Death by overcrowding**: An alive cell with >3 neighbors dies

Truth table:
```
Current State | Neighbors | Next State
--------------+-----------+-----------
Dead          | 0-2       | Dead
Dead          | 3         | Alive (birth)
Dead          | 4-8       | Dead
Alive         | 0-1       | Dead (lonely)
Alive         | 2-3       | Alive (survival)
Alive         | 4-8       | Dead (crowded)
```

Metal implementation:
```metal
bool next_alive = (neighbors == 3) || (current_alive && neighbors == 2);
```

---

## Appendix B: Classic Patterns

### Glider (5 cells, moves diagonally)
```
.#.
..#
###
```
Cell indices (from top-left of pattern): (0,1), (1,2), (2,0), (2,1), (2,2)

### Blinker (3 cells, period-2 oscillator)
```
###  →  .#.  →  ###
         #
         #
```

### Block (4 cells, stable)
```
##
##
```

### Beacon (6 cells, period-2 oscillator)
```
##..
##..
..##
..##
```

### Glider Gun (36 cells, emits gliders)
Too large to show - load from pattern library.

---

*This PRD defines a complete, implementation-ready Game of Life demo that showcases the GPU-Native OS architecture's capability to run interactive applications entirely within a single GPU threadgroup.*
