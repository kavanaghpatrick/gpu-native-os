# PRD: GPU Compositor Integration (Issue #158)

## Overview

The GPU compositor is the final rendering stage that orchestrates z-sorting, occlusion culling, and vertex compaction entirely on the GPU. It runs LAST each frame, consuming the unified vertex buffer from all apps and producing a compact, depth-sorted vertex stream for a single draw call.

**Key Insight**: Traditional compositors are CPU-bound (sorting windows, managing draw order). This compositor runs as a GPU compute kernel, leveraging parallel algorithms for O(log N) sorting and O(N) work with O(log N) depth for compaction.

## Goal

Create a GPU-native compositor that:
1. **Z-sorts all app vertices by depth** using parallel bitonic sort
2. **Culls occluded vertices** (fully hidden behind opaque windows)
3. **Compacts the vertex buffer** to eliminate gaps from culled vertices
4. **Outputs a single contiguous vertex stream** for one draw call

All operations run on GPU. CPU only submits the command buffer.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| qsort() on window list | Bitonic sort on GPU (O(log^2 N) depth) |
| CPU visibility raycast | Per-vertex occlusion test (parallel) |
| memcpy to compact buffer | Prefix sum + scatter (O(log N) depth) |
| Sequential draw per window | Single draw of compacted buffer |
| CPU depth buffer readback | GPU-resident depth pyramid |

## Existing Infrastructure to REUSE

### 1. Unified Vertex Buffer (from gpu_app_system.rs)

```rust
// Already exists - apps write vertices with depth to unified buffer
pub const DEFAULT_VERTEX_POOL_SIZE: usize = 16 * 1024 * 1024;  // 16MB

pub struct GpuAppDescriptor {
    pub vertex_offset: u32,    // Offset in unified vertex buffer
    pub vertex_size: u32,      // Bytes allocated
    pub vertex_count: u32,     // Actual vertices written
    // ...
}
```

### 2. Z-Order in Widgets (from memory.rs, widget.rs)

```rust
#[repr(C)]
pub struct WidgetCompact {
    // ...
    pub z_order: u16,  // Already exists!
    // ...
}
```

### 3. Bitonic Sort (from benchmark_visual.rs)

```metal
// Existing bitonic sort in threadgroup memory
void gpu_sort_widgets(device BenchWidget* widgets, uint count, uint tid,
                      threadgroup BenchWidget* tg_widgets) {
    // Bitonic sort (limited to 1024 - Metal max threadgroup size)
    for (uint k = 2; k <= 1024; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            uint ixj = tid ^ j;
            if (ixj > tid && tid < count && ixj < count) {
                bool ascending = ((tid & k) == 0);
                BenchWidget a = tg_widgets[tid];
                BenchWidget b = tg_widgets[ixj];
                bool should_swap = ascending ? (a.z_order > b.z_order) : (a.z_order < b.z_order);
                if (should_swap) {
                    tg_widgets[tid] = b;
                    tg_widgets[ixj] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
```

### 4. Prefix Sum (from parallel_compact.rs)

```metal
// Blelloch parallel prefix sum - already implemented
kernel void prefix_sum_upsweep(device uint* data, constant uint& n, constant uint& stride, uint tid);
kernel void prefix_sum_downsweep(device uint* data, constant uint& n, constant uint& stride, uint tid);
```

### 5. CompositorState (from gpu_app_system.rs)

```metal
struct CompositorState {
    float screen_width;
    float screen_height;
    uint window_count;
    uint frame_number;
    float4 background_color;
    uint total_vertices_rendered;
    uint _pad[3];
};
```

## Architecture

### Data Flow

```
Apps write vertices    GPU Compositor Pipeline              Final Render
     |                        |                                  |
     v                        v                                  v
[Unified Vertex     [1. Mark Visible]    [Compacted      [Single Draw
 Buffer with    -->  [2. Bitonic Sort] -> Vertex      -->  Call with
 z_order per        [3. Prefix Sum]       Buffer]         Depth Test]
 vertex]            [4. Scatter]
```

### Compositor Pipeline Stages

| Stage | Kernel | Complexity | Description |
|-------|--------|------------|-------------|
| 1 | `mark_visible` | O(1) per vertex | Set visibility flag based on bounds |
| 2 | `bitonic_sort_pass` | O(log^2 N) passes | Sort by z_order (front-to-back) |
| 3 | `prefix_sum` | O(log N) passes | Compute output offsets |
| 4 | `scatter_compact` | O(1) per vertex | Copy visible vertices to output |

## Metal Shader Pseudocode

### 1. Render Vertex with Z-Order

```metal
// Vertex structure with depth for compositor
struct CompositorVertex {
    float3 position;      // x, y, z (z = normalized depth from z_order)
    float4 color;
    float2 uv;
    uint window_id;       // For occlusion grouping
    uint flags;           // bit 0: visible, bit 1: opaque
};
```

### 2. Parallel Bitonic Sort by Z-Order

```metal
// Large-scale bitonic sort using multiple dispatch passes
// Handles > 1024 elements by working in device memory

struct SortParams {
    uint count;           // Number of vertices
    uint k;               // Current k value (power of 2)
    uint j;               // Current j value
};

kernel void bitonic_sort_pass(
    device CompositorVertex* vertices [[buffer(0)]],
    constant SortParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint count = params.count;
    uint k = params.k;
    uint j = params.j;

    if (tid >= count) return;

    uint ixj = tid ^ j;

    // Only process if ixj > tid (avoid double swaps)
    if (ixj > tid && ixj < count) {
        CompositorVertex a = vertices[tid];
        CompositorVertex b = vertices[ixj];

        // Ascending if (tid & k) == 0, descending otherwise
        bool ascending = ((tid & k) == 0);

        // Compare by z (depth) - lower z = closer to camera = rendered last
        bool should_swap = ascending
            ? (a.position.z > b.position.z)
            : (a.position.z < b.position.z);

        if (should_swap) {
            vertices[tid] = b;
            vertices[ixj] = a;
        }
    }
}

// Host dispatches multiple passes:
// for k in [2, 4, 8, ..., next_power_of_2(count)]:
//     for j in [k/2, k/4, ..., 1]:
//         dispatch(bitonic_sort_pass, {count, k, j})
```

### 3. Visibility Culling with Depth Pyramid

```metal
// Per-vertex visibility test using hierarchical Z-buffer (HZB)
struct CullingParams {
    uint vertex_count;
    uint window_count;
    float2 screen_size;
};

kernel void mark_visibility(
    device CompositorVertex* vertices [[buffer(0)]],
    device atomic_uint* visible_counts [[buffer(1)]],  // Per-window visible count
    texture2d<float, access::read> depth_pyramid [[texture(0)]],
    constant CullingParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.vertex_count) return;

    CompositorVertex v = vertices[tid];

    // Convert position to screen space
    float2 screen_pos = float2(
        (v.position.x + 1.0) * 0.5 * params.screen_size.x,
        (1.0 - v.position.y) * 0.5 * params.screen_size.y
    );

    // Sample depth pyramid at appropriate mip level
    // Use conservative bounds of the triangle containing this vertex
    uint mip = 0;  // Could compute based on projected area
    float stored_depth = depth_pyramid.read(uint2(screen_pos), mip).r;

    // Vertex is visible if its depth is closer than stored depth
    // (assuming depth 0 = near, 1 = far)
    bool is_visible = v.position.z <= stored_depth + 0.0001;

    // Mark visibility in flags
    v.flags = (v.flags & ~1u) | (is_visible ? 1u : 0u);
    vertices[tid] = v;

    // Count visible vertices per window for stats
    if (is_visible) {
        atomic_fetch_add_explicit(&visible_counts[v.window_id], 1, memory_order_relaxed);
    }
}
```

### 4. Prefix Sum for Output Offsets

```metal
// Reuse existing Blelloch scan from parallel_compact.rs
// Input: visibility flags (1 = visible, 0 = culled)
// Output: exclusive scan giving output index for each visible vertex

kernel void compute_visibility_prefix(
    device uint* visibility_flags [[buffer(0)]],   // 1 per vertex
    device uint* output_indices [[buffer(1)]],     // Exclusive scan result
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Copy visibility to output for in-place scan
    if (tid < count) {
        output_indices[tid] = visibility_flags[tid];
    }
}

// Then dispatch prefix_sum_upsweep and prefix_sum_downsweep passes
```

### 5. Vertex Compaction via Scatter

```metal
struct CompactParams {
    uint vertex_count;
    uint total_visible;  // From last element of prefix sum
};

kernel void scatter_compact(
    device CompositorVertex* input_vertices [[buffer(0)]],
    device CompositorVertex* output_vertices [[buffer(1)]],
    device uint* output_indices [[buffer(2)]],  // From prefix sum
    constant CompactParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.vertex_count) return;

    CompositorVertex v = input_vertices[tid];

    // Only copy visible vertices
    if (v.flags & 1u) {
        uint output_idx = output_indices[tid];
        output_vertices[output_idx] = v;
    }
}
```

### 6. Full Compositor Kernel (Single Dispatch for Simple Case)

```metal
// Optimized compositor for <= 1024 vertices (fits in threadgroup)
// Combines sort + cull + compact in one dispatch

kernel void compositor_full(
    device CompositorVertex* input_vertices [[buffer(0)]],
    device CompositorVertex* output_vertices [[buffer(1)]],
    device atomic_uint* output_count [[buffer(2)]],
    constant uint& vertex_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup CompositorVertex tg_vertices[1024];
    threadgroup uint tg_visible[1024];
    threadgroup uint tg_scan[1024];

    // Load into threadgroup memory
    if (tid < vertex_count) {
        tg_vertices[tid] = input_vertices[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: Bitonic sort by depth
    // ═══════════════════════════════════════════════════════════════
    for (uint k = 2; k <= 1024; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            if (tid < vertex_count) {
                uint ixj = tid ^ j;
                if (ixj > tid && ixj < vertex_count) {
                    bool ascending = ((tid & k) == 0);
                    CompositorVertex a = tg_vertices[tid];
                    CompositorVertex b = tg_vertices[ixj];

                    bool should_swap = ascending
                        ? (a.position.z > b.position.z)
                        : (a.position.z < b.position.z);

                    if (should_swap) {
                        tg_vertices[tid] = b;
                        tg_vertices[ixj] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Mark visibility (simplified - all visible for now)
    // ═══════════════════════════════════════════════════════════════
    if (tid < vertex_count) {
        tg_visible[tid] = 1;  // All visible (occlusion culling optional)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Prefix sum for compaction offsets
    // ═══════════════════════════════════════════════════════════════
    if (tid < vertex_count) {
        tg_scan[tid] = tg_visible[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch up-sweep
    for (uint stride = 1; stride < 1024; stride *= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 1024 && idx - stride < vertex_count) {
            tg_scan[idx] += tg_scan[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear last for exclusive scan
    if (tid == 0) {
        tg_scan[1023] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch down-sweep
    for (uint stride = 512; stride >= 1; stride /= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 1024) {
            uint temp = tg_scan[idx - stride];
            tg_scan[idx - stride] = tg_scan[idx];
            tg_scan[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: Scatter to output buffer
    // ═══════════════════════════════════════════════════════════════
    if (tid < vertex_count && tg_visible[tid]) {
        uint output_idx = tg_scan[tid];
        output_vertices[output_idx] = tg_vertices[tid];
    }

    // Thread 0 stores total count
    if (tid == 0) {
        uint total = tg_scan[vertex_count - 1] + tg_visible[vertex_count - 1];
        atomic_store_explicit(output_count, total, memory_order_relaxed);
    }
}
```

## Rust Implementation

### CompositorPipeline

```rust
pub struct CompositorPipeline {
    device: Device,

    // Pipelines
    sort_pipeline: ComputePipelineState,
    visibility_pipeline: ComputePipelineState,
    prefix_sum_upsweep: ComputePipelineState,
    prefix_sum_downsweep: ComputePipelineState,
    scatter_pipeline: ComputePipelineState,
    full_compositor_pipeline: ComputePipelineState,

    // Buffers
    staging_buffer: Buffer,       // For multi-pass sort
    visibility_buffer: Buffer,    // Visibility flags
    scan_buffer: Buffer,          // Prefix sum workspace
    output_buffer: Buffer,        // Compacted output

    // State
    max_vertices: usize,
}

impl CompositorPipeline {
    pub fn new(device: &Device, max_vertices: usize) -> Result<Self, String> {
        // Compile shaders, create pipelines, allocate buffers
        // ...
    }

    /// Run full compositor pipeline
    pub fn composite(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_vertices: &Buffer,
        vertex_count: usize,
        output_count: &Buffer,
    ) -> &Buffer {
        if vertex_count <= 1024 {
            // Use optimized single-dispatch kernel
            self.composite_small(encoder, input_vertices, vertex_count, output_count)
        } else {
            // Use multi-pass pipeline
            self.composite_large(encoder, input_vertices, vertex_count, output_count)
        }
    }

    fn composite_small(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_vertices: &Buffer,
        vertex_count: usize,
        output_count: &Buffer,
    ) -> &Buffer {
        encoder.set_compute_pipeline_state(&self.full_compositor_pipeline);
        encoder.set_buffer(0, Some(input_vertices), 0);
        encoder.set_buffer(1, Some(&self.output_buffer), 0);
        encoder.set_buffer(2, Some(output_count), 0);
        encoder.set_bytes(3, 4, &(vertex_count as u32) as *const _ as *const _);

        encoder.dispatch_threads(
            MTLSize::new(vertex_count as u64, 1, 1),
            MTLSize::new(1024, 1, 1),
        );

        &self.output_buffer
    }

    fn composite_large(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_vertices: &Buffer,
        vertex_count: usize,
        output_count: &Buffer,
    ) -> &Buffer {
        // 1. Bitonic sort passes
        let n = vertex_count.next_power_of_two();
        let mut k = 2u32;
        while k <= n as u32 {
            let mut j = k / 2;
            while j > 0 {
                encoder.set_compute_pipeline_state(&self.sort_pipeline);
                encoder.set_buffer(0, Some(input_vertices), 0);
                // Set params: count, k, j
                encoder.dispatch_threads(
                    MTLSize::new(vertex_count as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                j /= 2;
            }
            k *= 2;
        }

        // 2. Mark visibility
        encoder.set_compute_pipeline_state(&self.visibility_pipeline);
        encoder.set_buffer(0, Some(input_vertices), 0);
        encoder.set_buffer(1, Some(&self.visibility_buffer), 0);
        encoder.dispatch_threads(
            MTLSize::new(vertex_count as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        // 3. Prefix sum (multiple passes)
        self.run_prefix_sum(encoder, vertex_count);

        // 4. Scatter compact
        encoder.set_compute_pipeline_state(&self.scatter_pipeline);
        encoder.set_buffer(0, Some(input_vertices), 0);
        encoder.set_buffer(1, Some(&self.output_buffer), 0);
        encoder.set_buffer(2, Some(&self.scan_buffer), 0);
        encoder.dispatch_threads(
            MTLSize::new(vertex_count as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        &self.output_buffer
    }
}
```

## Tests

```rust
//! Tests for Issue #158: GPU Compositor Integration
//!
//! Validates parallel z-sort, visibility culling, and vertex compaction.

use metal::Device;
use rust_experiment::gpu_os::{CompositorPipeline, CompositorVertex};

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// ============================================================================
// Bitonic Sort Tests
// ============================================================================

#[test]
fn test_bitonic_sort_small() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // Create unsorted vertices with different depths
    let mut vertices: Vec<CompositorVertex> = (0..16)
        .map(|i| CompositorVertex {
            position: [0.0, 0.0, (15 - i) as f32 / 16.0],  // Reverse order
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
            window_id: 0,
            flags: 1,
        })
        .collect();

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let output = compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify sorted by depth (ascending - furthest first for back-to-front)
    let result: &[CompositorVertex] = unsafe {
        std::slice::from_raw_parts(
            output.contents() as *const CompositorVertex,
            vertices.len(),
        )
    };

    for i in 1..result.len() {
        assert!(
            result[i].position[2] >= result[i-1].position[2],
            "Vertex {} (z={}) should be >= vertex {} (z={})",
            i, result[i].position[2], i-1, result[i-1].position[2]
        );
    }
}

#[test]
fn test_bitonic_sort_1024() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // Create 1024 vertices with random depths
    let vertices: Vec<CompositorVertex> = (0..1024)
        .map(|i| {
            let seed = i as u32 * 1103515245 + 12345;
            CompositorVertex {
                position: [0.0, 0.0, (seed % 1000) as f32 / 1000.0],
                color: [1.0, 1.0, 1.0, 1.0],
                uv: [0.0, 0.0],
                window_id: i as u32 % 10,
                flags: 1,
            }
        })
        .collect();

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let output = compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify sorted
    let result: &[CompositorVertex] = unsafe {
        std::slice::from_raw_parts(
            output.contents() as *const CompositorVertex,
            vertices.len(),
        )
    };

    for i in 1..result.len() {
        assert!(
            result[i].position[2] >= result[i-1].position[2],
            "Not sorted at index {}: {} < {}",
            i, result[i].position[2], result[i-1].position[2]
        );
    }
}

// ============================================================================
// Visibility Culling Tests
// ============================================================================

#[test]
fn test_visibility_all_visible() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // All vertices visible (no occlusion)
    let vertices: Vec<CompositorVertex> = (0..100)
        .map(|i| CompositorVertex {
            position: [i as f32 * 10.0, 0.0, i as f32 / 100.0],
            color: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            window_id: 0,
            flags: 1,
        })
        .collect();

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // All should be visible
    let count = unsafe { *(output_count.contents() as *const u32) };
    assert_eq!(count, 100, "Expected all 100 vertices visible, got {}", count);
}

#[test]
fn test_visibility_partial_cull() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // Alternate visible/invisible
    let vertices: Vec<CompositorVertex> = (0..100)
        .map(|i| CompositorVertex {
            position: [0.0, 0.0, i as f32 / 100.0],
            color: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            window_id: 0,
            flags: if i % 2 == 0 { 1 } else { 0 },  // Even = visible, Odd = culled
        })
        .collect();

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Half should be visible
    let count = unsafe { *(output_count.contents() as *const u32) };
    assert_eq!(count, 50, "Expected 50 visible vertices, got {}", count);
}

// ============================================================================
// Vertex Compaction Tests
// ============================================================================

#[test]
fn test_compaction_removes_gaps() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // Every 3rd vertex is visible
    let vertices: Vec<CompositorVertex> = (0..99)
        .map(|i| CompositorVertex {
            position: [i as f32, 0.0, i as f32 / 99.0],
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
            window_id: i as u32,
            flags: if i % 3 == 0 { 1 } else { 0 },
        })
        .collect();

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let output = compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let count = unsafe { *(output_count.contents() as *const u32) };
    assert_eq!(count, 33, "Expected 33 visible vertices (0,3,6,...,96)");

    // Verify output is contiguous (no gaps)
    let result: &[CompositorVertex] = unsafe {
        std::slice::from_raw_parts(
            output.contents() as *const CompositorVertex,
            count as usize,
        )
    };

    // Each visible vertex should have window_id divisible by 3
    for (i, v) in result.iter().enumerate() {
        assert!(
            v.window_id % 3 == 0,
            "Vertex {} has window_id {} which is not divisible by 3",
            i, v.window_id
        );
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_compositor_pipeline() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // Simulate 5 overlapping windows with 6 vertices each (quads)
    let mut vertices = Vec::new();
    for window_id in 0..5u32 {
        let depth = window_id as f32 / 5.0;
        for _ in 0..6 {
            vertices.push(CompositorVertex {
                position: [100.0 + window_id as f32 * 50.0, 100.0, depth],
                color: [1.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                window_id,
                flags: 1,
            });
        }
    }

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let output = compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let count = unsafe { *(output_count.contents() as *const u32) };
    assert_eq!(count, 30, "Expected all 30 vertices (5 windows * 6 vertices)");

    // Verify depth ordering - window 0 should come before window 4
    let result: &[CompositorVertex] = unsafe {
        std::slice::from_raw_parts(
            output.contents() as *const CompositorVertex,
            count as usize,
        )
    };

    // First 6 vertices should be from window 0 (depth 0.0)
    for i in 0..6 {
        assert_eq!(result[i].window_id, 0, "First 6 should be window 0");
    }

    // Last 6 vertices should be from window 4 (depth 0.8)
    for i in 24..30 {
        assert_eq!(result[i].window_id, 4, "Last 6 should be window 4");
    }
}

#[test]
fn test_compositor_empty_input() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    let empty_buffer = device.new_buffer(
        std::mem::size_of::<CompositorVertex>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    compositor.composite(&encoder, &empty_buffer, 0, &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let count = unsafe { *(output_count.contents() as *const u32) };
    assert_eq!(count, 0, "Empty input should produce 0 output");
}

// ============================================================================
// Performance Tests
// ============================================================================

#[test]
fn test_compositor_performance_1024() {
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    let vertices: Vec<CompositorVertex> = (0..1024)
        .map(|i| CompositorVertex {
            position: [0.0, 0.0, (i as f32).sin()],
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
            window_id: i as u32 % 10,
            flags: 1,
        })
        .collect();

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    // Warm up
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let duration = start.elapsed();

    let per_frame_us = duration.as_micros() / 1000;
    println!("Compositor (1024 vertices): {}us/frame", per_frame_us);

    // Should complete in < 100us (0.1ms) per frame
    assert!(per_frame_us < 100, "Compositor too slow: {}us", per_frame_us);
}

#[test]
fn test_single_draw_call_verification() {
    // This test verifies the compositor produces output suitable for a single draw call
    let device = get_device();
    let compositor = CompositorPipeline::new(&device, 1024).unwrap();
    let queue = device.new_command_queue();

    // Create vertices from 10 different windows
    let mut vertices = Vec::new();
    for window_id in 0..10u32 {
        let depth = window_id as f32 / 10.0;
        for v in 0..6 {
            vertices.push(CompositorVertex {
                position: [v as f32 * 10.0, window_id as f32 * 50.0, depth],
                color: [window_id as f32 / 10.0, 0.5, 0.5, 1.0],
                uv: [0.0, 0.0],
                window_id,
                flags: 1,
            });
        }
    }

    let input_buffer = device.new_buffer_with_data(
        vertices.as_ptr() as *const _,
        (vertices.len() * std::mem::size_of::<CompositorVertex>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_count = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();

    let output = compositor.composite(&encoder, &input_buffer, vertices.len(), &output_count);

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let count = unsafe { *(output_count.contents() as *const u32) };

    // Verify: output is contiguous, sorted, and ready for single draw call
    assert_eq!(count, 60, "All 60 vertices should be in output");

    // The output buffer can now be bound directly to a render encoder:
    // encoder.set_vertex_buffer(0, Some(&output), 0);
    // encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, count as u64);

    println!("Output ready for single draw call: {} vertices", count);
}
```

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Draw calls | 1 | Single draw for all windows |
| Sort time | < 50us | For 1024 vertices |
| Compaction | O(log N) depth | Verified by algorithm |
| GPU utilization | > 80% | Parallel sort + scatter |
| CPU work | 0 | Only command buffer submit |

## Future Enhancements

1. **Hierarchical Z-Buffer (HZB)**: Build depth pyramid for early-out occlusion
2. **Temporal coherence**: Reuse previous frame's sort order as starting point
3. **Radix sort**: For > 10K vertices, radix sort may be faster than bitonic
4. **Indirect draw**: Use GPU-computed vertex count for indirect draw command
5. **Multi-window occlusion**: Cull vertices fully behind opaque windows

## References

- Existing bitonic sort: `src/gpu_os/benchmark_visual.rs`
- Existing prefix sum: `src/gpu_os/parallel_compact.rs`
- Z-order handling: `src/gpu_os/widget.rs`, `src/gpu_os/memory.rs`
- CompositorState: `src/gpu_os/gpu_app_system.rs`
