# PRD: Fractal Terrain Generator with Erosion

**Version**: 1.0
**Date**: 2026-01-23
**Status**: Ready for Implementation
**Demo Type**: GPU-Native OS Application

---

## 1. Overview

### What It Is

A real-time procedural terrain generator running entirely within a single GPU threadgroup (1024 threads). The demo generates a 32x32 heightmap where each thread owns one cell, performing noise generation, hydraulic erosion simulation, and 3D-style rendering in a unified compute pass.

### Why It's Impressive

| Traditional Approach | GPU-Native OS Approach |
|---------------------|------------------------|
| CPU generates noise (~50ms) | GPU generates noise (<1ms) |
| CPU transfers to GPU (~2ms) | Zero transfer (unified memory) |
| Separate erosion pass (~100ms) | Integrated erosion (<0.5ms/iteration) |
| Multiple draw calls | Single compute + render pass |
| **Total: 150ms+ regeneration** | **Total: <2ms regeneration** |

**Key Demo Points**:
1. **Instant regeneration**: Slider changes reflect in <16ms (same frame)
2. **Unified execution**: Noise, erosion, and render in one kernel dispatch
3. **Thread ownership**: Each thread "owns" its terrain cell permanently
4. **SIMD coherence**: Adjacent threads process adjacent cells (cache-friendly)

---

## 2. User Experience

### Visual Layout

```
+------------------------------------------------------------------+
|                    Fractal Terrain Generator                       |
+------------------------------------------------------------------+
|                                                                    |
|     +------------------------------------------+                   |
|     |                                          |                   |
|     |         3D Terrain Visualization         |                   |
|     |         (Height-colored mesh with        |                   |
|     |          basic directional lighting)     |                   |
|     |                                          |                   |
|     +------------------------------------------+                   |
|                                                                    |
|  +-----------------+  +-----------------+  +-----------------+     |
|  | Octaves    [4]  |  | Persistence    |  | Lacunarity     |     |
|  | [====|=======]  |  | [===|========] |  | [=======|====] |     |
|  | 1          8    |  | 0.3      0.7   |  | 1.5      2.5   |     |
|  +-----------------+  +-----------------+  +-----------------+     |
|                                                                    |
|  +-----------------+  +-----------------+  +-----------------+     |
|  | Sea Level       |  | Erosion Iters  |  | Rain Amount    |     |
|  | [==|==========] |  | [====|=======] |  | [=====|======] |     |
|  | 0.0      0.5    |  | 0         50   |  | 0.0      1.0   |     |
|  +-----------------+  +-----------------+  +-----------------+     |
|                                                                    |
|  [Regenerate Seed]    [Toggle Erosion: ON]    FPS: 120            |
+------------------------------------------------------------------+
```

### User Interactions

| Action | Result | Latency |
|--------|--------|---------|
| Drag any slider | Terrain regenerates instantly | <16ms |
| Click "Regenerate Seed" | New random terrain | <16ms |
| Toggle erosion | Starts/stops erosion animation | Immediate |
| Hold erosion ON | Terrain smooths over ~50 frames | 50 iterations at 120fps |

### Visual Feedback

- Sliders show current value numerically
- Terrain updates in real-time as sliders move (no "Apply" button)
- Erosion shows visible smoothing when enabled
- FPS counter confirms 120fps maintained during interaction

---

## 3. Technical Architecture

### Thread Assignment (32x32 = 1024 threads)

```
Thread ID Layout (matches heightmap cells):
+---+---+---+---+---+---+---+---+     +----+----+----+----+
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ... | 24 | 25 | 26 | 27 |  <- Row 0
+---+---+---+---+---+---+---+---+     +----+----+----+----+
| 32| 33| 34| 35| 36| 37| 38| 39| ... | 56 | 57 | 58 | 59 |  <- Row 1
+---+---+---+---+---+---+---+---+     +----+----+----+----+
         ...                                   ...
+----+----+----+----+----+----+----+     +------+------+
|992 |993 |994 |995 |996 |997 |998 | ... | 1022 | 1023 |     <- Row 31
+----+----+----+----+----+----+----+     +------+------+

tid -> (x, y):  x = tid % 32,  y = tid / 32
(x, y) -> tid:  tid = y * 32 + x
```

### Memory Layout

```
+------------------------------------------------------------------+
|                    GPU UNIFIED MEMORY                              |
+------------------------------------------------------------------+
|  HEIGHTMAP (4KB)                                                  |
|  +------------------------------------------------------------+   |
|  | float[1024] - height values in range [0.0, 1.0]            |   |
|  | Layout: row-major, 32x32 grid                              |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  WATER MAP (4KB) - For erosion simulation                         |
|  +------------------------------------------------------------+   |
|  | float[1024] - water depth at each cell                     |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  SEDIMENT MAP (4KB) - For erosion simulation                      |
|  +------------------------------------------------------------+   |
|  | float[1024] - suspended sediment at each cell              |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  TERRAIN PARAMS (64B)                                             |
|  +------------------------------------------------------------+   |
|  | octaves: u32, persistence: f32, lacunarity: f32,           |   |
|  | sea_level: f32, erosion_iters: u32, rain_amount: f32,      |   |
|  | seed: u32, flags: u32 (bit 0 = erosion_enabled)            |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  VERTEX BUFFER (72KB) - Pre-allocated for 32x32x2 triangles       |
|  +------------------------------------------------------------+   |
|  | Vertex[6144] - 2 triangles per cell, 3 vertices each       |   |
|  | Each vertex: position (float3) + normal (float3) + color   |   |
|  +------------------------------------------------------------+   |
|                                                                    |
|  WIDGET STATE (1KB) - 6 sliders + 2 buttons                       |
|  +------------------------------------------------------------+   |
|  | WidgetCompact[8] - UI controls                             |   |
|  +------------------------------------------------------------+   |
+------------------------------------------------------------------+
Total: ~85KB GPU memory
```

### Per-Frame Pipeline

```
Frame N:
+------------------------------------------------------------------+
| PHASE 1: INPUT (All 1024 threads)                    [0.1ms]      |
|   - Thread 0-7 read input events from queue                       |
|   - All threads check for slider value changes                    |
|   - If params changed: set regenerate flag                        |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| PHASE 2: TERRAIN GENERATION (if regenerate flag)     [0.8ms]      |
|   - Each thread computes simplex FBM for its cell                 |
|   - Parallel: all 1024 cells computed simultaneously              |
|   - Write height to heightmap[tid]                                |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| PHASE 3: EROSION STEP (if erosion enabled)           [0.5ms]      |
|   - Substep A: Add rain (water += rain_amount)                    |
|   - Substep B: Calculate flow to neighbors                        |
|   - Substep C: Move water and sediment                            |
|   - Substep D: Evaporation and deposition                         |
|   - Barrier between each substep                                  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| PHASE 4: GEOMETRY GENERATION (All 1024 threads)      [0.3ms]      |
|   - Each thread generates 6 vertices (2 triangles) for its cell   |
|   - Sample neighbor heights for normal calculation                |
|   - Apply height-based coloring                                   |
|   - Write to vertex_buffer[tid * 6 ... tid * 6 + 5]              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| RENDER PASS (Hardware rasterizer)                    [2.0ms]      |
|   - Draw 6144 vertices as triangles                               |
|   - Basic directional lighting in fragment shader                 |
|   - UI overlay rendered on top                                    |
+------------------------------------------------------------------+

Total per-frame: ~3.7ms (273 fps theoretical, vsync-locked to 120)
```

---

## 4. Data Structures

### Rust Definitions

```rust
/// Terrain generation parameters - sent to GPU each frame
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TerrainParams {
    pub octaves: u32,           // 1-8, number of noise layers
    pub persistence: f32,       // 0.3-0.7, amplitude decay per octave
    pub lacunarity: f32,        // 1.5-2.5, frequency multiplier per octave
    pub sea_level: f32,         // 0.0-0.5, water cutoff height
    pub erosion_iters: u32,     // 0-50, erosion steps per frame
    pub rain_amount: f32,       // 0.0-1.0, water added per iteration
    pub seed: u32,              // Random seed for noise
    pub flags: u32,             // Bit 0: erosion_enabled
}  // 32 bytes

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            sea_level: 0.2,
            erosion_iters: 0,
            rain_amount: 0.3,
            seed: 42,
            flags: 0,
        }
    }
}

/// Vertex for terrain mesh
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct TerrainVertex {
    pub position: [f32; 3],     // x, y (height), z
    pub normal: [f32; 3],       // Surface normal for lighting
    pub color: [f32; 4],        // RGBA based on height
}  // 40 bytes

/// Terrain buffers
pub struct TerrainBuffers {
    pub heightmap: Buffer,      // float[1024]
    pub water: Buffer,          // float[1024]
    pub sediment: Buffer,       // float[1024]
    pub params: Buffer,         // TerrainParams
    pub vertices: Buffer,       // TerrainVertex[6144]
}

impl TerrainBuffers {
    pub fn new(device: &Device) -> Self {
        Self {
            heightmap: device.new_buffer(
                (1024 * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            water: device.new_buffer(
                (1024 * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            sediment: device.new_buffer(
                (1024 * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            params: device.new_buffer(
                std::mem::size_of::<TerrainParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            vertices: device.new_buffer(
                (6144 * std::mem::size_of::<TerrainVertex>()) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
        }
    }
}

/// Slider widget with value binding
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SliderWidget {
    pub base: WidgetCompact,
    pub min_value: f32,
    pub max_value: f32,
    pub current_value: f32,
    pub step: f32,
}  // 40 bytes
```

### Metal Struct Definitions

```metal
struct TerrainParams {
    uint octaves;
    float persistence;
    float lacunarity;
    float sea_level;
    uint erosion_iters;
    float rain_amount;
    uint seed;
    uint flags;
};

struct TerrainVertex {
    float3 position;
    float3 normal;
    float4 color;
};

// Threadgroup shared memory for erosion
struct ErosionShared {
    float heights[1024];        // Working copy of heightmap
    float water[1024];          // Water depth
    float sediment[1024];       // Suspended sediment
    float outflow[1024][4];     // Flow to 4 neighbors (N,E,S,W)
};  // ~24KB threadgroup memory
```

---

## 5. Shader Pseudocode

### Simplex Noise (2D)

```metal
// Simplex noise implementation (Ken Perlin's improved algorithm)
// Returns value in range [-1, 1]

constant float2 grad2[] = {
    float2(1,1), float2(-1,1), float2(1,-1), float2(-1,-1),
    float2(1,0), float2(-1,0), float2(0,1), float2(0,-1)
};

inline float2 hash2(float2 p, uint seed) {
    // Simple hash function for gradient selection
    uint h = uint(p.x * 374761393.0 + p.y * 668265263.0) + seed;
    h = (h ^ (h >> 13)) * 1274126177u;
    return grad2[h & 7];
}

float simplex2d(float2 p, uint seed) {
    // Skew to simplex grid
    const float F2 = 0.366025403784;  // (sqrt(3) - 1) / 2
    const float G2 = 0.211324865405;  // (3 - sqrt(3)) / 6

    float s = (p.x + p.y) * F2;
    float2 pi = floor(p + s);
    float t = (pi.x + pi.y) * G2;
    float2 p0 = p - (pi - t);

    // Determine which simplex triangle
    float2 o1 = (p0.x > p0.y) ? float2(1, 0) : float2(0, 1);

    float2 p1 = p0 - o1 + G2;
    float2 p2 = p0 - 1.0 + 2.0 * G2;

    // Calculate contributions from corners
    float n0 = 0, n1 = 0, n2 = 0;

    float t0 = 0.5 - dot(p0, p0);
    if (t0 > 0) {
        t0 *= t0;
        n0 = t0 * t0 * dot(hash2(pi, seed), p0);
    }

    float t1 = 0.5 - dot(p1, p1);
    if (t1 > 0) {
        t1 *= t1;
        n1 = t1 * t1 * dot(hash2(pi + o1, seed), p1);
    }

    float t2 = 0.5 - dot(p2, p2);
    if (t2 > 0) {
        t2 *= t2;
        n2 = t2 * t2 * dot(hash2(pi + 1.0, seed), p2);
    }

    // Scale to [-1, 1]
    return 70.0 * (n0 + n1 + n2);
}
```

### Fractal Brownian Motion (FBM)

```metal
float fbm(float2 p, uint octaves, float persistence, float lacunarity, uint seed) {
    float value = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float max_value = 0.0;

    for (uint i = 0; i < octaves; i++) {
        value += amplitude * simplex2d(p * frequency, seed + i * 1000);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    // Normalize to [0, 1]
    return (value / max_value + 1.0) * 0.5;
}
```

### Erosion Step (Hydraulic Erosion)

```metal
void erosion_step(
    uint tid,
    threadgroup float* heights,
    threadgroup float* water,
    threadgroup float* sediment,
    float rain_amount,
    float evaporation_rate,    // 0.1
    float sediment_capacity,   // 0.5
    float erosion_rate,        // 0.3
    float deposition_rate      // 0.3
) {
    uint x = tid % 32;
    uint y = tid / 32;

    // =========================================================================
    // SUBSTEP A: Add rain
    // =========================================================================
    water[tid] += rain_amount * 0.01;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // SUBSTEP B: Calculate outflow to neighbors
    // =========================================================================
    float h = heights[tid] + water[tid];
    float outflow[4] = {0, 0, 0, 0};  // N, E, S, W

    // North neighbor
    if (y > 0) {
        uint n = tid - 32;
        float hn = heights[n] + water[n];
        outflow[0] = max(0.0, h - hn) * 0.25;
    }
    // East neighbor
    if (x < 31) {
        uint n = tid + 1;
        float hn = heights[n] + water[n];
        outflow[1] = max(0.0, h - hn) * 0.25;
    }
    // South neighbor
    if (y < 31) {
        uint n = tid + 32;
        float hn = heights[n] + water[n];
        outflow[2] = max(0.0, h - hn) * 0.25;
    }
    // West neighbor
    if (x > 0) {
        uint n = tid - 1;
        float hn = heights[n] + water[n];
        outflow[3] = max(0.0, h - hn) * 0.25;
    }

    float total_out = outflow[0] + outflow[1] + outflow[2] + outflow[3];
    if (total_out > water[tid]) {
        float scale = water[tid] / total_out;
        outflow[0] *= scale;
        outflow[1] *= scale;
        outflow[2] *= scale;
        outflow[3] *= scale;
    }

    // Store outflow in threadgroup memory for neighbor access
    threadgroup float tg_outflow[1024][4];
    tg_outflow[tid][0] = outflow[0];
    tg_outflow[tid][1] = outflow[1];
    tg_outflow[tid][2] = outflow[2];
    tg_outflow[tid][3] = outflow[3];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // SUBSTEP C: Update water based on inflow/outflow
    // =========================================================================
    float inflow = 0;
    if (y > 0)  inflow += tg_outflow[tid - 32][2];  // From north's south
    if (x < 31) inflow += tg_outflow[tid + 1][3];   // From east's west
    if (y < 31) inflow += tg_outflow[tid + 32][0];  // From south's north
    if (x > 0)  inflow += tg_outflow[tid - 1][1];   // From west's east

    water[tid] = water[tid] - total_out + inflow;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // SUBSTEP D: Erosion and deposition
    // =========================================================================
    float capacity = sediment_capacity * water[tid];

    if (sediment[tid] < capacity) {
        // Erode terrain
        float erode_amount = min(erosion_rate * (capacity - sediment[tid]), heights[tid] * 0.1);
        heights[tid] -= erode_amount;
        sediment[tid] += erode_amount;
    } else {
        // Deposit sediment
        float deposit_amount = deposition_rate * (sediment[tid] - capacity);
        heights[tid] += deposit_amount;
        sediment[tid] -= deposit_amount;
    }

    // Evaporation
    water[tid] *= (1.0 - evaporation_rate);
    sediment[tid] *= 0.99;  // Slow sediment settling

    threadgroup_barrier(mem_flags::mem_threadgroup);
}
```

### Geometry Generation

```metal
void generate_terrain_vertex(
    uint tid,
    threadgroup float* heights,
    device TerrainVertex* vertices,
    constant TerrainParams& params
) {
    uint x = tid % 32;
    uint y = tid / 32;

    // Cell corner positions (normalized to [-1, 1] range for rendering)
    float cell_size = 2.0 / 32.0;  // Full terrain spans [-1, 1]
    float px = -1.0 + x * cell_size;
    float pz = -1.0 + y * cell_size;

    // Sample heights at corners
    float h00 = heights[tid];
    float h10 = (x < 31) ? heights[tid + 1] : h00;
    float h01 = (y < 31) ? heights[tid + 32] : h00;
    float h11 = (x < 31 && y < 31) ? heights[tid + 33] : h00;

    // Calculate normal from height differences
    float3 normal = normalize(float3(
        (h00 - h10 + h01 - h11) * 16.0,  // X component
        2.0,                               // Y component (up)
        (h00 + h10 - h01 - h11) * 16.0   // Z component
    ));

    // Height-based coloring
    float4 color = height_to_color(h00, params.sea_level);

    // Generate 2 triangles (6 vertices) for this cell
    // Triangle 1: TL -> BL -> BR
    uint base = tid * 6;

    vertices[base + 0] = TerrainVertex{
        float3(px, h00 * 0.5, pz),
        normal,
        color
    };
    vertices[base + 1] = TerrainVertex{
        float3(px, h01 * 0.5, pz + cell_size),
        normal,
        color
    };
    vertices[base + 2] = TerrainVertex{
        float3(px + cell_size, h11 * 0.5, pz + cell_size),
        normal,
        height_to_color(h11, params.sea_level)
    };

    // Triangle 2: TL -> BR -> TR
    vertices[base + 3] = TerrainVertex{
        float3(px, h00 * 0.5, pz),
        normal,
        color
    };
    vertices[base + 4] = TerrainVertex{
        float3(px + cell_size, h11 * 0.5, pz + cell_size),
        normal,
        height_to_color(h11, params.sea_level)
    };
    vertices[base + 5] = TerrainVertex{
        float3(px + cell_size, h10 * 0.5, pz),
        normal,
        height_to_color(h10, params.sea_level)
    };
}

float4 height_to_color(float height, float sea_level) {
    // Color bands based on elevation
    if (height < sea_level) {
        // Water: deep blue to light blue
        float t = height / sea_level;
        return mix(float4(0.0, 0.1, 0.4, 1.0), float4(0.2, 0.5, 0.8, 1.0), t);
    }

    float land_height = (height - sea_level) / (1.0 - sea_level);

    if (land_height < 0.1) {
        // Sand/beach
        return float4(0.76, 0.70, 0.50, 1.0);
    } else if (land_height < 0.4) {
        // Grass
        float t = (land_height - 0.1) / 0.3;
        return mix(float4(0.2, 0.6, 0.2, 1.0), float4(0.3, 0.5, 0.2, 1.0), t);
    } else if (land_height < 0.7) {
        // Rock
        float t = (land_height - 0.4) / 0.3;
        return mix(float4(0.4, 0.4, 0.35, 1.0), float4(0.5, 0.5, 0.45, 1.0), t);
    } else {
        // Snow
        float t = (land_height - 0.7) / 0.3;
        return mix(float4(0.7, 0.7, 0.7, 1.0), float4(1.0, 1.0, 1.0, 1.0), t);
    }
}
```

---

## 6. Widget Integration

### Slider Definitions

```rust
pub fn create_terrain_sliders(device: &Device) -> Vec<SliderWidget> {
    vec![
        SliderWidget {
            base: WidgetBuilder::new(WidgetType::Slider)
                .bounds(0.05, 0.75, 0.25, 0.04)
                .build(),
            min_value: 1.0,
            max_value: 8.0,
            current_value: 4.0,  // octaves
            step: 1.0,
        },
        SliderWidget {
            base: WidgetBuilder::new(WidgetType::Slider)
                .bounds(0.35, 0.75, 0.25, 0.04)
                .build(),
            min_value: 0.3,
            max_value: 0.7,
            current_value: 0.5,  // persistence
            step: 0.05,
        },
        SliderWidget {
            base: WidgetBuilder::new(WidgetType::Slider)
                .bounds(0.65, 0.75, 0.25, 0.04)
                .build(),
            min_value: 1.5,
            max_value: 2.5,
            current_value: 2.0,  // lacunarity
            step: 0.1,
        },
        SliderWidget {
            base: WidgetBuilder::new(WidgetType::Slider)
                .bounds(0.05, 0.85, 0.25, 0.04)
                .build(),
            min_value: 0.0,
            max_value: 0.5,
            current_value: 0.2,  // sea_level
            step: 0.05,
        },
        SliderWidget {
            base: WidgetBuilder::new(WidgetType::Slider)
                .bounds(0.35, 0.85, 0.25, 0.04)
                .build(),
            min_value: 0.0,
            max_value: 50.0,
            current_value: 0.0,  // erosion_iters
            step: 1.0,
        },
        SliderWidget {
            base: WidgetBuilder::new(WidgetType::Slider)
                .bounds(0.65, 0.85, 0.25, 0.04)
                .build(),
            min_value: 0.0,
            max_value: 1.0,
            current_value: 0.3,  // rain_amount
            step: 0.1,
        },
    ]
}
```

### Slider Interaction (GPU)

```metal
void update_slider(
    device SliderWidget* slider,
    float2 cursor,
    bool mouse_down,
    device TerrainParams* params,
    uint param_index
) {
    // Check if cursor is within slider bounds
    if (!point_in_rect(cursor, slider->base.bounds)) return;

    if (mouse_down) {
        // Calculate value from cursor position
        float4 bounds = unpack_bounds(slider->base.bounds);
        float t = (cursor.x - bounds.x) / bounds.z;
        t = saturate(t);

        float value = mix(slider->min_value, slider->max_value, t);

        // Snap to step
        value = round(value / slider->step) * slider->step;

        slider->current_value = value;

        // Update corresponding param
        switch (param_index) {
            case 0: params->octaves = uint(value); break;
            case 1: params->persistence = value; break;
            case 2: params->lacunarity = value; break;
            case 3: params->sea_level = value; break;
            case 4: params->erosion_iters = uint(value); break;
            case 5: params->rain_amount = value; break;
        }
    }
}
```

---

## 7. Visual Design

### Color Palette

| Terrain Type | Height Range | RGB Values |
|--------------|--------------|------------|
| Deep Water | 0.0 - sea_level*0.5 | (0, 26, 102) #001A66 |
| Shallow Water | sea_level*0.5 - sea_level | (51, 128, 204) #3380CC |
| Sand/Beach | sea_level - sea_level+0.1 | (194, 178, 128) #C2B280 |
| Grass Low | sea_level+0.1 - sea_level+0.3 | (51, 153, 51) #339933 |
| Grass High | sea_level+0.3 - 0.6 | (76, 128, 51) #4C8033 |
| Rock | 0.6 - 0.8 | (102, 102, 89) #666659 |
| Snow | 0.8 - 1.0 | (255, 255, 255) #FFFFFF |

### Lighting

```metal
// Simple directional lighting for terrain
float3 light_direction = normalize(float3(0.5, 1.0, 0.3));
float ambient = 0.3;
float diffuse = max(0.0, dot(normal, light_direction));
float3 lit_color = base_color.rgb * (ambient + diffuse * 0.7);
```

### Camera Setup

```metal
// Isometric-style view matrix (fixed camera)
constant float4x4 view_matrix = {
    float4(0.866, -0.354, 0.354, 0.0),
    float4(0.0,   0.707,  0.707, 0.0),
    float4(-0.5, -0.612, 0.612, 0.0),
    float4(0.0,   0.0,   -3.0,  1.0)
};

// Orthographic projection
float4x4 ortho_projection(float size, float aspect) {
    return float4x4(
        float4(1.0 / (size * aspect), 0, 0, 0),
        float4(0, 1.0 / size, 0, 0),
        float4(0, 0, -0.01, 0),
        float4(0, 0, 0, 1)
    );
}
```

---

## 8. Performance Targets

### Frame Budget (8.33ms at 120Hz)

| Phase | Target | Maximum | Notes |
|-------|--------|---------|-------|
| Input processing | 0.1ms | 0.2ms | Slider updates |
| Noise generation | 0.8ms | 1.5ms | Full 32x32 FBM |
| Erosion (1 iter) | 0.5ms | 0.8ms | Skip if disabled |
| Geometry generation | 0.3ms | 0.5ms | 6144 vertices |
| Render (fragment) | 2.0ms | 3.0ms | Terrain + UI |
| **Total** | **3.7ms** | **6.0ms** | **Headroom: 2.3ms** |

### Scaling Targets

| Metric | Target | Maximum | Bottleneck |
|--------|--------|---------|------------|
| Heightmap resolution | 32x32 | 32x32 | Thread count |
| Erosion iterations/frame | 1 | 10 | Frame time |
| FPS | 120 | 120 | VSync locked |
| Slider update latency | <16ms | <16ms | Same-frame response |
| Regeneration time | <2ms | <5ms | Noise + geometry |

### Memory Targets

| Buffer | Size | Notes |
|--------|------|-------|
| Heightmap | 4KB | float[1024] |
| Water map | 4KB | float[1024] |
| Sediment map | 4KB | float[1024] |
| Vertex buffer | 240KB | 6144 * 40 bytes |
| Threadgroup memory | 24KB | Erosion working set |
| **Total GPU** | **~280KB** | Well under limits |

---

## 9. Implementation Milestones

### Milestone 1: Static Terrain (Day 1-2)

- [ ] Create `TerrainParams` and `TerrainBuffers` structs
- [ ] Implement simplex noise in Metal shader
- [ ] Implement FBM with configurable octaves
- [ ] Generate heightmap (all 1024 threads parallel)
- [ ] Verify output with solid color based on height

### Milestone 2: Terrain Rendering (Day 3-4)

- [ ] Implement `generate_terrain_vertex` kernel
- [ ] Set up vertex buffer and draw call
- [ ] Implement height-based coloring
- [ ] Add basic directional lighting
- [ ] Set up isometric camera view

### Milestone 3: UI Integration (Day 5-6)

- [ ] Create 6 slider widgets for parameters
- [ ] Implement slider hit testing and value update
- [ ] Wire slider values to `TerrainParams`
- [ ] Trigger terrain regeneration on value change
- [ ] Add "Regenerate Seed" button

### Milestone 4: Erosion Simulation (Day 7-8)

- [ ] Implement water and sediment maps
- [ ] Implement rain addition phase
- [ ] Implement water flow calculation
- [ ] Implement erosion and deposition
- [ ] Add erosion toggle button
- [ ] Tune erosion parameters for visual appeal

### Milestone 5: Polish (Day 9-10)

- [ ] Optimize shader for 120fps
- [ ] Add FPS counter display
- [ ] Fine-tune color palette
- [ ] Add parameter labels to UI
- [ ] Test on target hardware (M4)
- [ ] Profile and optimize hot paths

---

## 10. Future Enhancements

### Near-Term (Post-MVP)

| Enhancement | Complexity | Value |
|-------------|------------|-------|
| 3D rotation with mouse drag | Medium | High |
| Height-based shadow casting | Medium | High |
| Animated water shader | Low | Medium |
| Biome-based coloring | Low | Medium |
| Export heightmap to file | Low | Low |

### Long-Term (v2.0)

| Enhancement | Complexity | Value |
|-------------|------------|-------|
| 64x64 heightmap (4 threadgroups) | High | High |
| LOD for distant terrain | High | Medium |
| Procedural vegetation | High | High |
| Thermal erosion | Medium | Medium |
| Real-time sculpting with brush | High | High |

### Multi-Threadgroup Scaling

```
Future 64x64 terrain would require 4096 threads (4 threadgroups):
- Threadgroup 0: cells (0,0) to (31,31)
- Threadgroup 1: cells (32,0) to (63,31)
- Threadgroup 2: cells (0,32) to (31,63)
- Threadgroup 3: cells (32,32) to (63,63)

Challenge: Erosion across threadgroup boundaries requires
device-memory synchronization between kernel dispatches.
```

---

## Appendix A: Complete Main Kernel

```metal
kernel void terrain_kernel(
    device float* heightmap [[buffer(0)]],
    device float* water [[buffer(1)]],
    device float* sediment [[buffer(2)]],
    constant TerrainParams& params [[buffer(3)]],
    device TerrainVertex* vertices [[buffer(4)]],
    device SliderWidget* sliders [[buffer(5)]],
    device InputQueue* input [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup ErosionShared shared;

    // =========================================================================
    // PHASE 1: INPUT PROCESSING
    // =========================================================================
    // (Simplified - full input handling as in main kernel)
    bool params_changed = false;
    // ... check sliders, update params_changed

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 2: TERRAIN GENERATION (if params changed or first frame)
    // =========================================================================
    if (params_changed || params.flags & 0x2) {  // Bit 1 = needs_regeneration
        uint x = tid % 32;
        uint y = tid / 32;
        float2 pos = float2(x, y) / 32.0 * 4.0;  // Scale for noise frequency

        float height = fbm(pos, params.octaves, params.persistence,
                          params.lacunarity, params.seed);
        heightmap[tid] = height;
        water[tid] = 0.0;
        sediment[tid] = 0.0;
    }

    // Copy heightmap to shared memory for erosion
    shared.heights[tid] = heightmap[tid];
    shared.water[tid] = water[tid];
    shared.sediment[tid] = sediment[tid];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 3: EROSION (if enabled)
    // =========================================================================
    if (params.flags & 0x1) {  // Bit 0 = erosion_enabled
        for (uint i = 0; i < params.erosion_iters; i++) {
            erosion_step(tid, shared.heights, shared.water, shared.sediment,
                        params.rain_amount, 0.1, 0.5, 0.3, 0.3);
        }

        // Write back to device memory
        heightmap[tid] = shared.heights[tid];
        water[tid] = shared.water[tid];
        sediment[tid] = shared.sediment[tid];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 4: GEOMETRY GENERATION
    // =========================================================================
    generate_terrain_vertex(tid, shared.heights, vertices, params);
}
```

---

## Appendix B: Simplex Noise Reference

The simplex noise implementation is based on Ken Perlin's improved noise (2001), adapted for GPU:

1. **Skew input**: Transform (x,y) to simplex grid
2. **Find simplex**: Determine which triangle contains the point
3. **Gradient selection**: Hash corner coordinates to gradient vectors
4. **Contribution sum**: Weight gradients by distance falloff

Key optimization: Use `simd_shuffle` for gradient lookups when adjacent threads sample nearby coordinates.

---

## Appendix C: Erosion Algorithm Reference

The hydraulic erosion is based on the pipe model (O'Brien & Hodgins):

1. **Rain**: Add water uniformly
2. **Flow**: Calculate outflow based on height difference
3. **Transport**: Move water and sediment to neighbors
4. **Erosion/Deposition**: Erode if under capacity, deposit if over
5. **Evaporation**: Reduce water volume

The algorithm is parallelized by computing all cells simultaneously with threadgroup barriers between dependent phases.

---

*This PRD provides a complete specification for implementing a fractal terrain generator as a GPU-Native OS demo application, showcasing instant procedural generation and real-time parameter adjustment within the single-threadgroup architecture.*
