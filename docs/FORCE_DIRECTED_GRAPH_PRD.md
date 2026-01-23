# PRD: Force-Directed Graph Visualizer

**Version**: 1.0
**Date**: 2026-01-23
**Status**: Implementation Ready
**Demo Type**: GPU-Native OS Showcase

---

## 1. Overview

### 1.1 What It Is

A real-time force-directed graph visualization running entirely within a **single GPU threadgroup of 1024 threads**. Each thread owns one node, computing repulsion forces against all other nodes (N-body simulation), spring forces for connected edges, and rendering the final visualization - all in shared memory with perfect synchronization via `threadgroup_barrier()`.

### 1.2 Why It's Impressive

| Aspect | Traditional Approach | This Demo |
|--------|---------------------|-----------|
| **N-body computation** | CPU O(N^2) or GPU multi-dispatch | Single threadgroup, shared memory |
| **Synchronization** | Complex cross-threadgroup atomics | Perfect `threadgroup_barrier()` |
| **Memory latency** | Device memory (~400 cycles) | Shared memory (~1 cycle) |
| **Force calculations** | 1M/frame typical | 1M+ in shared memory |
| **State persistence** | Separate compute/render passes | Unified pipeline |

**The Hook**: Watch 1024 nodes self-organize into clusters in real-time. Drag any node and see the entire graph respond instantly. This is N-body physics running at 120fps in 32KB of shared memory.

### 1.3 Target Platform

- Apple Silicon (M1/M2/M3/M4)
- Metal 3.0
- macOS 13.0+
- 120Hz ProMotion display

---

## 2. User Experience

### 2.1 What the User Sees

```
┌─────────────────────────────────────────────────────────────────┐
│  Force-Directed Graph Visualizer              120 FPS  █████   │
│  Nodes: 1024  Edges: 4096  Forces: 1,048,576/frame             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ●───●───●                                    │
│                   /│   │   │\                                   │
│              ●───● │   ●   │ ●───●                              │
│               \  │ │  /│\  │ │  /                               │
│                ●─●─●─● │ ●─●─●─●                                │
│                  \│ │ \│/ │ │/                                  │
│                   ●─●──●──●─●                                   │
│                    \│ /│\ │/                                    │
│                     ●──●──●                                     │
│                      \ │ /                                      │
│                        ●   ← Dragging (highlighted)             │
│                                                                 │
│  [+] Add Node  [─] Add Edge  [🗑] Delete  [⟳] Reset  [⏸] Pause  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Visual Elements

1. **Nodes**: Circles colored by cluster (community detection via edge density)
2. **Edges**: Lines with thickness proportional to weight, alpha based on length
3. **Dragged Node**: Glowing highlight, larger radius
4. **Hovered Node**: Subtle highlight, shows degree count
5. **Selected Edge**: Highlighted in accent color
6. **Background**: Dark gradient, subtle grid for depth perception

### 2.3 Interactions

| Action | Input | Effect |
|--------|-------|--------|
| **Pan** | Two-finger drag / Middle-click | Move viewport |
| **Zoom** | Pinch / Scroll wheel | Scale view centered on cursor |
| **Drag Node** | Click + drag on node | Pin node to cursor, release to unpin |
| **Select Node** | Single click | Highlight node and its edges |
| **Add Edge** | Shift+click two nodes | Create spring between them |
| **Delete Node** | Right-click node | Remove node and all its edges |
| **Add Random Nodes** | Press 'N' | Add 10 random nodes with random edges |
| **Reset Layout** | Press 'R' | Randomize positions, restart simulation |
| **Pause/Resume** | Spacebar | Freeze/unfreeze physics simulation |

---

## 3. Technical Architecture

### 3.1 Single Threadgroup Model

All 1024 threads participate in every phase. Thread ID = Node ID.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SINGLE THREADGROUP (1024 threads)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Thread 0 ──── Node 0 data ──── Force 0 ──── Velocity 0        │
│  Thread 1 ──── Node 1 data ──── Force 1 ──── Velocity 1        │
│  Thread 2 ──── Node 2 data ──── Force 2 ──── Velocity 2        │
│    ...          ...              ...           ...              │
│  Thread 1023 ── Node 1023 ───── Force 1023 ── Velocity 1023    │
│                                                                 │
│  ═══════════════════ THREADGROUP MEMORY (32KB) ═══════════════ │
│  │ positions[1024] │ velocities[1024] │ forces[1024] │ edges │ │
│  │     8KB         │      8KB         │     8KB      │  8KB  │ │
│  ══════════════════════════════════════════════════════════════ │
│                                                                 │
│  Device Memory: Edge CSR (read-only), Vertex Buffer (write)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Thread Assignment Per Phase

| Phase | Threads Used | Work Per Thread |
|-------|--------------|-----------------|
| **1. Input** | 0-31 | Load/process input events |
| **2. Repulsion** | 0-1023 | N-body: each thread vs all others |
| **3. Attraction** | 0-1023 | Spring forces from edges |
| **4. Integration** | 0-1023 | Velocity + position update |
| **5. Centering** | 0-1023 | SIMD reduction for centroid |
| **6. Node Render** | 0-1023 | Generate 6 vertices per node |
| **7. Edge Render** | 0-MAX_EDGES | Generate 6 vertices per edge |

### 3.3 Memory Layout

#### Threadgroup Memory (32KB on-chip, single-cycle access)

```metal
struct ThreadgroupMemory {
    // Node state (updated every frame)
    float2 positions[1024];        // 8KB - x,y per node
    float2 velocities[1024];       // 8KB - vx,vy per node
    float2 forces[1024];           // 8KB - accumulated force

    // Edge cache (hot edges for current batch)
    uint2 edge_cache[512];         // 4KB - (src, dst) pairs
    half edge_weights[512];        // 1KB - spring constants

    // Reduction scratch
    float2 simd_scratch[32];       // 256B - per-SIMD reduction

    // Frame state
    uint node_count;               // Active nodes
    uint edge_count;               // Active edges
    uint hovered_node;             // Node under cursor
    uint dragged_node;             // Node being dragged
    float2 drag_target;            // Where dragged node should be
    float2 center_of_mass;         // For centering force

    // Remaining: ~2KB for expansion
};
```

#### Device Memory (Unified, GPU-accessible)

```metal
struct DeviceMemory {
    // Graph structure (CSR format for edges)
    uint edge_offsets[1025];       // 4KB - start index per node
    uint edge_targets[MAX_EDGES];  // Edge destination nodes
    half edge_weights[MAX_EDGES];  // Spring constants

    // Node properties (persistent across frames)
    uint node_colors[1024];        // Cluster color per node
    uint node_flags[1024];         // Selected, hovered, dragged

    // Vertex output buffer (for fragment shader)
    Vertex vertices[MAX_VERTICES]; // Node quads + edge quads
    uint vertex_count;             // Indirect draw count

    // Frame state (persistent)
    FrameState frame_state;        // Time, frame number, camera

    // Input queue (ring buffer from CPU)
    InputQueue input;              // Mouse/keyboard events
};
```

### 3.4 Per-Frame Pipeline

```
Frame Start (VSync trigger)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: INPUT COLLECTION (SIMD 0, 32 threads)                 │
│                                                                 │
│   - Read input events from ring buffer                          │
│   - Update cursor position, button state                        │
│   - Detect hovered node (point-in-circle test)                  │
│   - Handle drag start/end                                       │
│                                                                 │
│   threadgroup_barrier(mem_flags::mem_threadgroup);              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: REPULSION (ALL 1024 threads)                          │
│                                                                 │
│   for each thread tid:                                          │
│       my_pos = positions[tid]                                   │
│       force = (0, 0)                                            │
│       for i in 0..node_count:                                   │
│           if i != tid:                                          │
│               delta = my_pos - positions[i]                     │
│               dist = length(delta)                              │
│               force += normalize(delta) * k_repel / (dist^2)    │
│       forces[tid] = force                                       │
│                                                                 │
│   threadgroup_barrier(mem_flags::mem_threadgroup);              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: ATTRACTION (ALL 1024 threads)                         │
│                                                                 │
│   for each thread tid:                                          │
│       start = edge_offsets[tid]                                 │
│       end = edge_offsets[tid + 1]                               │
│       for e in start..end:                                      │
│           neighbor = edge_targets[e]                            │
│           weight = edge_weights[e]                              │
│           delta = positions[neighbor] - positions[tid]          │
│           dist = length(delta)                                  │
│           forces[tid] += normalize(delta) * k_spring * dist     │
│                                                                 │
│   threadgroup_barrier(mem_flags::mem_threadgroup);              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: DAMPING + CENTERING (ALL 1024 threads)                │
│                                                                 │
│   // Parallel reduction for center of mass                      │
│   simd_sum = simd_sum(positions[tid])                           │
│   if simd_is_first():                                           │
│       simd_scratch[simd_id] = simd_sum                          │
│   threadgroup_barrier()                                         │
│                                                                 │
│   if tid == 0:                                                  │
│       center = sum(simd_scratch) / node_count                   │
│       center_of_mass = center                                   │
│   threadgroup_barrier()                                         │
│                                                                 │
│   // Apply centering force                                      │
│   forces[tid] -= (positions[tid] - center_of_mass) * k_center   │
│   // Apply damping                                              │
│   forces[tid] -= velocities[tid] * damping                      │
│                                                                 │
│   threadgroup_barrier(mem_flags::mem_threadgroup);              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: INTEGRATION (ALL 1024 threads)                        │
│                                                                 │
│   // Semi-implicit Euler                                        │
│   if tid != dragged_node:                                       │
│       velocities[tid] += forces[tid] * dt                       │
│       velocities[tid] = clamp(velocities[tid], -max_vel, max_vel)
│       positions[tid] += velocities[tid] * dt                    │
│       positions[tid] = clamp(positions[tid], bounds)            │
│   else:                                                         │
│       // Dragged node follows cursor                            │
│       positions[tid] = drag_target                              │
│       velocities[tid] = (0, 0)                                  │
│                                                                 │
│   threadgroup_barrier(mem_flags::mem_threadgroup);              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: NODE RENDERING (ALL 1024 threads)                     │
│                                                                 │
│   if tid < node_count:                                          │
│       pos = positions[tid]                                      │
│       radius = NODE_RADIUS * (1.0 + 0.3 * is_hovered)           │
│       color = node_colors[tid]                                  │
│       generate_circle_quad(pos, radius, color, tid * 6)         │
│                                                                 │
│   threadgroup_barrier(mem_flags::mem_threadgroup);              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 7: EDGE RENDERING (threads 0..edge_count)                │
│                                                                 │
│   // Multiple passes if edge_count > 1024                       │
│   for batch in 0..(edge_count / 1024 + 1):                      │
│       edge_idx = batch * 1024 + tid                             │
│       if edge_idx < edge_count:                                 │
│           src = edge_sources[edge_idx]                          │
│           dst = edge_targets[edge_idx]                          │
│           p1 = positions[src]                                   │
│           p2 = positions[dst]                                   │
│           generate_line_quad(p1, p2, edge_color, vertex_offset) │
│       threadgroup_barrier()                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 8: WRITEBACK TO DEVICE MEMORY                            │
│                                                                 │
│   // Persist positions and velocities for next frame            │
│   device_positions[tid] = positions[tid]                        │
│   device_velocities[tid] = velocities[tid]                      │
│   if tid == 0:                                                  │
│       device_vertex_count = computed_vertex_count               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Fragment Shader (Hardware Rasterization)
    │
    ▼
Present to Display
```

### 3.5 Force Model

#### Repulsion (Coulomb's Law)

```
F_repel = k_r * (1 / d^2) * direction

Where:
  k_r = 10000.0 (repulsion constant)
  d = distance between nodes (clamped to min 1.0)
  direction = normalized vector from other node to this node
```

#### Attraction (Hooke's Law / Spring)

```
F_attract = k_s * (d - rest_length) * direction * weight

Where:
  k_s = 0.01 (spring constant)
  d = distance between connected nodes
  rest_length = 50.0 (natural spring length)
  weight = edge weight (0.5 - 2.0)
  direction = normalized vector toward neighbor
```

#### Damping (Velocity Decay)

```
F_damp = -damping * velocity

Where:
  damping = 0.9 (per-frame velocity retention)
```

#### Centering (Prevent Drift)

```
F_center = -k_c * (position - center_of_mass)

Where:
  k_c = 0.001 (centering strength)
  center_of_mass = average of all node positions
```

#### Integration

```
// Semi-implicit Euler (stable for springs)
velocity += force * dt
velocity = clamp(velocity, -max_velocity, max_velocity)
position += velocity * dt
position = clamp(position, world_bounds)
```

---

## 4. Data Structures

### 4.1 Rust Host-Side Structs

```rust
/// Node data for GPU upload
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub position: [f32; 2],      // x, y
    pub velocity: [f32; 2],      // vx, vy
    pub color: u32,              // packed RGBA
    pub flags: u32,              // SELECTED | HOVERED | DRAGGED | PINNED
}  // 24 bytes

/// Edge in CSR format
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub target: u32,             // destination node index
    pub weight: f16,             // spring constant multiplier (half precision)
    pub _padding: u16,
}  // 8 bytes

/// Graph structure (CSR - Compressed Sparse Row)
#[repr(C)]
pub struct GraphCSR {
    pub node_count: u32,
    pub edge_count: u32,
    pub offsets: [u32; 1025],    // offsets[i] = first edge index for node i
    pub edges: [Edge; 8192],     // edge list (targets + weights)
}

/// Frame constants
#[repr(C)]
pub struct FrameUniforms {
    pub time: f32,
    pub delta_time: f32,
    pub node_count: u32,
    pub edge_count: u32,

    pub k_repel: f32,            // Repulsion constant
    pub k_spring: f32,           // Spring constant
    pub k_center: f32,           // Centering force
    pub damping: f32,            // Velocity damping

    pub cursor_pos: [f32; 2],    // Mouse position in world space
    pub cursor_buttons: u32,     // Bitmask: LEFT | RIGHT | MIDDLE
    pub hovered_node: u32,       // Node under cursor (0xFFFFFFFF if none)
    pub dragged_node: u32,       // Currently dragged node
    pub _padding: [u32; 3],      // Align to 64 bytes
}  // 64 bytes

/// Vertex for rendering (node circles and edge lines)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub position: [f32; 2],      // x, y in clip space
    pub uv: [f32; 2],            // texture coords / SDF coords
    pub color: [f32; 4],         // RGBA
}  // 32 bytes
```

### 4.2 Metal Shader Structs

```metal
struct Node {
    float2 position;
    float2 velocity;
    uint color;
    uint flags;
};  // 24 bytes, matches Rust

struct Edge {
    uint target;
    half weight;
    ushort _padding;
};  // 8 bytes

struct FrameUniforms {
    float time;
    float delta_time;
    uint node_count;
    uint edge_count;

    float k_repel;
    float k_spring;
    float k_center;
    float damping;

    float2 cursor_pos;
    uint cursor_buttons;
    uint hovered_node;
    uint dragged_node;
    uint3 _padding;
};  // 64 bytes

struct Vertex {
    float2 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
    float4 color [[attribute(2)]];
};

// Threadgroup shared memory layout
struct ThreadgroupData {
    float2 positions[1024];      // 8KB
    float2 velocities[1024];     // 8KB
    float2 forces[1024];         // 8KB

    // Reduction scratch
    float2 simd_scratch[32];     // 256 bytes

    // Interaction state
    atomic_uint vertex_count;
    uint hovered_node;
    uint dragged_node;
    float2 drag_target;
    float2 center_of_mass;
};
```

### 4.3 CSR Edge Format

The Compressed Sparse Row format stores edges efficiently for graph traversal:

```
Node 0: edges to [1, 3]
Node 1: edges to [0, 2, 4]
Node 2: edges to [1]
Node 3: edges to [0]
Node 4: edges to [1]

offsets = [0, 2, 5, 6, 7, 8]
edges   = [(1), (3), (0), (2), (4), (1), (0), (1)]

To iterate edges of node i:
  for e in offsets[i]..offsets[i+1]:
      neighbor = edges[e].target
```

Benefits:
- O(1) access to edge list start
- O(degree) iteration per node
- Cache-friendly sequential access
- Compact memory footprint

---

## 5. Shader Pseudocode

### 5.1 Main Compute Kernel

```metal
kernel void force_directed_graph(
    device Node* nodes [[buffer(0)]],
    device uint* edge_offsets [[buffer(1)]],
    device Edge* edges [[buffer(2)]],
    device Vertex* vertex_buffer [[buffer(3)]],
    device atomic_uint* vertex_count [[buffer(4)]],
    constant FrameUniforms& uniforms [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Shared memory declaration
    threadgroup float2 tg_positions[1024];
    threadgroup float2 tg_velocities[1024];
    threadgroup float2 tg_forces[1024];
    threadgroup float2 tg_simd_scratch[32];
    threadgroup atomic_uint tg_hovered;
    threadgroup uint tg_dragged;
    threadgroup float2 tg_drag_target;
    threadgroup float2 tg_center;

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: LOAD STATE FROM DEVICE MEMORY
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        tg_positions[tid] = nodes[tid].position;
        tg_velocities[tid] = nodes[tid].velocity;
    }
    tg_forces[tid] = float2(0.0);

    if (tid == 0) {
        atomic_store_explicit(&tg_hovered, 0xFFFFFFFF, memory_order_relaxed);
        tg_dragged = uniforms.dragged_node;
        tg_drag_target = uniforms.cursor_pos;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: HIT TESTING (find hovered node)
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        float2 delta = tg_positions[tid] - uniforms.cursor_pos;
        float dist_sq = dot(delta, delta);
        float radius = 10.0;  // Node radius in world units

        if (dist_sq < radius * radius) {
            // Atomic min to get frontmost (closest) node
            atomic_fetch_min_explicit(&tg_hovered, tid, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: REPULSION FORCES (N-body, O(N^2))
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        float2 my_pos = tg_positions[tid];
        float2 repel_force = float2(0.0);

        for (uint i = 0; i < uniforms.node_count; i++) {
            if (i != tid) {
                float2 delta = my_pos - tg_positions[i];
                float dist_sq = max(dot(delta, delta), 1.0);  // Prevent division by zero
                float dist = sqrt(dist_sq);

                // Coulomb repulsion: F = k / d^2
                float magnitude = uniforms.k_repel / dist_sq;
                repel_force += (delta / dist) * magnitude;
            }
        }

        tg_forces[tid] += repel_force;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: ATTRACTION FORCES (springs from edges)
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        float2 my_pos = tg_positions[tid];
        float2 spring_force = float2(0.0);

        uint edge_start = edge_offsets[tid];
        uint edge_end = edge_offsets[tid + 1];

        for (uint e = edge_start; e < edge_end; e++) {
            uint neighbor = edges[e].target;
            float weight = float(edges[e].weight);

            float2 delta = tg_positions[neighbor] - my_pos;
            float dist = length(delta);
            float rest_length = 50.0;  // Natural spring length

            // Hooke's law: F = k * (d - rest) * direction
            if (dist > 0.001) {
                float displacement = dist - rest_length;
                float magnitude = uniforms.k_spring * displacement * weight;
                spring_force += normalize(delta) * magnitude;
            }
        }

        tg_forces[tid] += spring_force;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 5: CENTER OF MASS (parallel reduction)
    // ═══════════════════════════════════════════════════════════════
    float2 my_pos = (tid < uniforms.node_count) ? tg_positions[tid] : float2(0.0);
    float2 simd_sum_pos = simd_sum(my_pos);

    if (simd_lane == 0) {
        tg_simd_scratch[simd_id] = simd_sum_pos;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float2 total = float2(0.0);
        for (uint s = 0; s < 32; s++) {
            total += tg_simd_scratch[s];
        }
        tg_center = total / float(uniforms.node_count);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 6: CENTERING + DAMPING
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        // Centering force (pull toward center of mass)
        float2 to_center = tg_center - tg_positions[tid];
        tg_forces[tid] += to_center * uniforms.k_center;

        // Damping force (friction)
        tg_forces[tid] -= tg_velocities[tid] * (1.0 - uniforms.damping);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 7: INTEGRATION (update velocity and position)
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        bool is_dragged = (tid == tg_dragged);

        if (!is_dragged) {
            // Semi-implicit Euler integration
            float2 new_vel = tg_velocities[tid] + tg_forces[tid] * uniforms.delta_time;

            // Clamp velocity to prevent explosion
            float max_vel = 500.0;
            float vel_mag = length(new_vel);
            if (vel_mag > max_vel) {
                new_vel = normalize(new_vel) * max_vel;
            }

            tg_velocities[tid] = new_vel;
            tg_positions[tid] += new_vel * uniforms.delta_time;

            // Clamp to world bounds
            float2 bounds = float2(1000.0, 800.0);
            tg_positions[tid] = clamp(tg_positions[tid], -bounds, bounds);
        } else {
            // Dragged node follows cursor
            tg_positions[tid] = tg_drag_target;
            tg_velocities[tid] = float2(0.0);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 8: GENERATE NODE VERTICES
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        float2 pos = tg_positions[tid];
        uint hovered = atomic_load_explicit(&tg_hovered, memory_order_relaxed);
        bool is_hovered = (tid == hovered);
        bool is_dragged = (tid == tg_dragged);

        float radius = 8.0;
        if (is_hovered) radius = 12.0;
        if (is_dragged) radius = 14.0;

        // Unpack node color
        uint packed_color = nodes[tid].color;
        float4 color = float4(
            float((packed_color >> 0) & 0xFF) / 255.0,
            float((packed_color >> 8) & 0xFF) / 255.0,
            float((packed_color >> 16) & 0xFF) / 255.0,
            float((packed_color >> 24) & 0xFF) / 255.0
        );

        if (is_hovered || is_dragged) {
            color = mix(color, float4(1.0, 1.0, 0.5, 1.0), 0.3);  // Highlight
        }

        // Generate quad (2 triangles, 6 vertices)
        uint base = tid * 6;

        // Quad corners: TL, TR, BR, BL
        float2 tl = pos + float2(-radius, -radius);
        float2 tr = pos + float2(+radius, -radius);
        float2 br = pos + float2(+radius, +radius);
        float2 bl = pos + float2(-radius, +radius);

        // Triangle 1: TL -> BL -> BR
        vertex_buffer[base + 0] = Vertex{tl, float2(-1, -1), color};
        vertex_buffer[base + 1] = Vertex{bl, float2(-1, +1), color};
        vertex_buffer[base + 2] = Vertex{br, float2(+1, +1), color};

        // Triangle 2: TL -> BR -> TR
        vertex_buffer[base + 3] = Vertex{tl, float2(-1, -1), color};
        vertex_buffer[base + 4] = Vertex{br, float2(+1, +1), color};
        vertex_buffer[base + 5] = Vertex{tr, float2(+1, -1), color};
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 9: GENERATE EDGE VERTICES
    // ═══════════════════════════════════════════════════════════════
    uint node_vertices = uniforms.node_count * 6;

    // Each thread handles multiple edges if needed
    uint edges_per_thread = (uniforms.edge_count + 1023) / 1024;

    for (uint i = 0; i < edges_per_thread; i++) {
        uint edge_idx = tid * edges_per_thread + i;

        if (edge_idx < uniforms.edge_count) {
            // Find which node owns this edge
            uint src_node = 0;
            for (uint n = 0; n < uniforms.node_count; n++) {
                if (edge_idx >= edge_offsets[n] && edge_idx < edge_offsets[n + 1]) {
                    src_node = n;
                    break;
                }
            }

            uint dst_node = edges[edge_idx].target;

            float2 p1 = tg_positions[src_node];
            float2 p2 = tg_positions[dst_node];

            // Line as thin quad
            float2 dir = normalize(p2 - p1);
            float2 perp = float2(-dir.y, dir.x) * 1.0;  // Line thickness

            float4 edge_color = float4(0.5, 0.5, 0.5, 0.6);  // Gray, semi-transparent

            uint base = node_vertices + edge_idx * 6;

            vertex_buffer[base + 0] = Vertex{p1 - perp, float2(0, 0), edge_color};
            vertex_buffer[base + 1] = Vertex{p1 + perp, float2(0, 1), edge_color};
            vertex_buffer[base + 2] = Vertex{p2 + perp, float2(1, 1), edge_color};

            vertex_buffer[base + 3] = Vertex{p1 - perp, float2(0, 0), edge_color};
            vertex_buffer[base + 4] = Vertex{p2 + perp, float2(1, 1), edge_color};
            vertex_buffer[base + 5] = Vertex{p2 - perp, float2(1, 0), edge_color};
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 10: WRITEBACK STATE
    // ═══════════════════════════════════════════════════════════════
    if (tid < uniforms.node_count) {
        nodes[tid].position = tg_positions[tid];
        nodes[tid].velocity = tg_velocities[tid];
    }

    if (tid == 0) {
        uint total_vertices = node_vertices + uniforms.edge_count * 6;
        atomic_store_explicit(vertex_count, total_vertices, memory_order_relaxed);
    }
}
```

### 5.2 Fragment Shader (Node Circles)

```metal
fragment float4 node_fragment(
    VertexOut in [[stage_in]],
    constant FrameUniforms& uniforms [[buffer(0)]]
) {
    // SDF circle: distance from center
    float dist = length(in.uv);

    // Anti-aliased edge
    float edge = 0.9;  // Edge at 90% of radius
    float aa_width = 0.05;
    float alpha = 1.0 - smoothstep(edge - aa_width, edge + aa_width, dist);

    // Subtle border
    float border_start = 0.75;
    float border_end = 0.85;
    float border = smoothstep(border_start, border_end, dist) *
                   (1.0 - smoothstep(edge - aa_width, edge, dist));

    float4 fill_color = in.color;
    float4 border_color = fill_color * 0.7;  // Darker border

    float4 final_color = mix(fill_color, border_color, border);
    final_color.a *= alpha;

    // Discard fully transparent pixels
    if (final_color.a < 0.01) discard_fragment();

    return final_color;
}
```

### 5.3 Fragment Shader (Edges)

```metal
fragment float4 edge_fragment(
    VertexOut in [[stage_in]]
) {
    // Simple anti-aliased line
    float dist_from_center = abs(in.uv.y - 0.5) * 2.0;
    float alpha = 1.0 - smoothstep(0.8, 1.0, dist_from_center);

    float4 color = in.color;
    color.a *= alpha * 0.6;  // Semi-transparent edges

    return color;
}
```

---

## 6. Interaction Model

### 6.1 Node Dragging

```
State Machine:
┌─────────┐  mouse_down     ┌─────────────┐
│  IDLE   │ ───────────────▶│  DRAGGING   │
└────┬────┘  on node        └──────┬──────┘
     │                              │
     │  mouse_move                  │ mouse_move
     │  (update hover)              │ (move node)
     │                              │
     └──────────◀───────────────────┤
               mouse_up             │
                                    ▼
                              node.position = cursor
                              node.velocity = 0
                              forces[node] ignored
```

### 6.2 Edge Creation

```
1. User shift-clicks Node A → mark as "edge_source"
2. Visual feedback: Node A highlighted, dashed line to cursor
3. User shift-clicks Node B → create edge (A, B)
4. Edge added to CSR (requires recomputing offsets)
5. Clear edge_source state
```

### 6.3 Viewport Controls

```
Pan:
  - Two-finger drag or middle-click
  - Update camera.offset += drag_delta / camera.zoom

Zoom:
  - Pinch or scroll wheel
  - Zoom centered on cursor position:
    world_cursor = screen_to_world(cursor)
    camera.zoom *= scale_factor
    camera.offset += world_cursor * (1 - 1/scale_factor)
```

---

## 7. Visual Design

### 7.1 Node Appearance

| State | Radius | Color | Border |
|-------|--------|-------|--------|
| Default | 8px | Cluster color | 1px darker |
| Hovered | 12px | Cluster + highlight | 2px accent |
| Dragged | 14px | Cluster + glow | 2px + shadow |
| Selected | 10px | Cluster + ring | 3px accent |

### 7.2 Cluster Colors

Nodes colored by community/cluster using spectral palette:

```rust
const CLUSTER_COLORS: [u32; 8] = [
    0xFF6B9EE8,  // Blue
    0xFFE86B6B,  // Red
    0xFF6BE88A,  // Green
    0xFFE8D56B,  // Yellow
    0xFFB06BE8,  // Purple
    0xFF6BE8E8,  // Cyan
    0xFFE8A06B,  // Orange
    0xFFE86BB0,  // Pink
];
```

Cluster assignment via label propagation or Louvain community detection (can be precomputed).

### 7.3 Edge Appearance

| Property | Value |
|----------|-------|
| Default color | #808080 (gray) at 60% alpha |
| Thickness | 1-4px based on weight |
| Selected | Accent color, 100% alpha |
| Fading | Alpha decreases with edge length |

### 7.4 Background

- Dark gradient: #1a1a2e to #16213e
- Subtle grid: #ffffff at 5% alpha, 50px spacing
- Vignette: darker at edges

---

## 8. Performance Targets

### 8.1 Frame Budget (8.33ms at 120Hz)

| Phase | Target | Maximum |
|-------|--------|---------|
| Input processing | 0.05ms | 0.1ms |
| Force computation | 2.0ms | 3.0ms |
| Integration | 0.2ms | 0.5ms |
| Vertex generation | 0.5ms | 1.0ms |
| Fragment rendering | 2.0ms | 3.0ms |
| **Total GPU** | **4.75ms** | **7.6ms** |
| CPU dispatch | 0.3ms | 0.5ms |
| **Headroom** | **3.28ms** | **0.23ms** |

### 8.2 Computational Load

| Metric | Value | Notes |
|--------|-------|-------|
| Nodes | 1024 | One per thread |
| Max edges | 8192 | ~8 edges per node average |
| Force calculations | 1,048,576/frame | N^2 repulsion |
| Vertices generated | ~55,000 | 6 per node + 6 per edge |
| Memory bandwidth | ~50 MB/frame | Positions + edges + vertices |

### 8.3 Scaling Behavior

| Nodes | Force Calc | Time Est. | Status |
|-------|------------|-----------|--------|
| 256 | 65,536 | 0.5ms | Excellent |
| 512 | 262,144 | 1.2ms | Good |
| 1024 | 1,048,576 | 2.5ms | Target |
| 2048 | 4,194,304 | 10ms+ | Exceeds budget |

For >1024 nodes: Use Barnes-Hut approximation (O(N log N)) or multi-dispatch.

---

## 9. Implementation Milestones

### Milestone 1: Static Graph Rendering (Days 1-2)
- [ ] Metal setup with single threadgroup dispatch
- [ ] Load test graph (CSR format)
- [ ] Render nodes as colored quads
- [ ] Render edges as lines
- [ ] Basic camera (fixed viewport)

**Deliverable**: Static graph visible on screen

### Milestone 2: Force Simulation (Days 3-4)
- [ ] Implement repulsion forces (N-body)
- [ ] Implement spring attraction
- [ ] Implement velocity integration
- [ ] Add damping and centering
- [ ] Verify stability (no explosions)

**Deliverable**: Graph self-organizes from random positions

### Milestone 3: Interaction (Days 5-6)
- [ ] Input event pipeline (mouse position, buttons)
- [ ] Hit testing for node hover
- [ ] Node dragging (pin to cursor)
- [ ] Viewport pan and zoom
- [ ] Node selection highlight

**Deliverable**: User can drag nodes, pan/zoom view

### Milestone 4: Polish (Days 7-8)
- [ ] SDF circle rendering with anti-aliasing
- [ ] Edge thickness based on weight
- [ ] Cluster coloring
- [ ] Stats overlay (FPS, node count, forces)
- [ ] Pause/resume simulation
- [ ] Add/remove nodes and edges

**Deliverable**: Complete demo ready for presentation

### Milestone 5: Optimization (Days 9-10)
- [ ] Profile with Metal System Trace
- [ ] Optimize force loop (SIMD intrinsics)
- [ ] Reduce memory bandwidth
- [ ] Test on various Apple Silicon chips
- [ ] Stress test with 1024 fully-connected nodes

**Deliverable**: Consistent 120fps across scenarios

---

## 10. Future Enhancements

### 10.1 Algorithmic Improvements

| Enhancement | Benefit | Complexity |
|-------------|---------|------------|
| Barnes-Hut tree | O(N log N) forces | High |
| GPU-parallel Louvain | Real-time clustering | Medium |
| Multi-level layout | Handle 10K+ nodes | High |
| Incremental edge updates | Dynamic graphs | Medium |

### 10.2 Visual Features

| Feature | Description |
|---------|-------------|
| Edge bundling | Reduce visual clutter |
| Node labels | Text rendering per node |
| Animated transitions | Smooth add/remove |
| Heat map overlay | Show force magnitudes |
| 3D mode | Z-axis with rotation |

### 10.3 Interaction Extensions

| Feature | Description |
|---------|-------------|
| Box selection | Select multiple nodes |
| Node grouping | Collapse clusters |
| Edge weights UI | Slider to adjust springs |
| Force parameter tuning | Real-time adjustment |
| Graph import/export | Load from file |

### 10.4 Multi-Graph Support

Run multiple independent graphs in separate dispatches, or use threadgroup batching for smaller graphs.

---

## Appendix A: Test Graphs

### Small Test (64 nodes)
```rust
fn create_grid_graph(rows: usize, cols: usize) -> Graph {
    // Create rows x cols grid with edges to neighbors
    // Good for testing spring forces
}
```

### Medium Test (256 nodes)
```rust
fn create_scale_free_graph(n: usize, m: usize) -> Graph {
    // Barabasi-Albert preferential attachment
    // Creates realistic network topology
}
```

### Large Test (1024 nodes)
```rust
fn create_clustered_graph(clusters: usize, per_cluster: usize) -> Graph {
    // Dense intra-cluster, sparse inter-cluster edges
    // Tests cluster detection visually
}
```

---

## Appendix B: Force Parameter Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `k_repel` | 10000 | 1000-50000 | Higher = nodes spread more |
| `k_spring` | 0.01 | 0.001-0.1 | Higher = tighter clusters |
| `k_center` | 0.001 | 0.0001-0.01 | Higher = nodes pulled to center |
| `damping` | 0.9 | 0.8-0.99 | Higher = slower settling |
| `rest_length` | 50 | 20-200 | Natural edge length |
| `max_velocity` | 500 | 100-1000 | Prevents explosion |

---

## Appendix C: Known Limitations

1. **1024 node limit**: Fundamental to single-threadgroup model
2. **O(N^2) scaling**: Force computation doesn't scale beyond 2048 nodes
3. **No persistence**: Graph state lost on quit (could add file save)
4. **Simple clustering**: Color assignment is static, not real-time detection
5. **Edge ordering**: CSR format makes edge insertion expensive

---

*This PRD defines a complete, implementation-ready force-directed graph visualizer that demonstrates the power of GPU-native architecture with perfect synchronization in shared memory.*
