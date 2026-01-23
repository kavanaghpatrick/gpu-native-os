# PRD: GPU-Native OS Architecture v2.0

**Version**: 2.0 (Revised based on Grok-4, Gemini, Codex reviews + 10 research agents)
**Date**: 2026-01-23
**Status**: Ready for Implementation

---

## Executive Summary

A GPU-native operating system where **all application logic runs on the GPU** within a single threadgroup (1024 threads), using the **Unified Worker Model** where all threads participate in all phases. The CPU serves only as a minimal dispatcher (~0.5ms/frame) triggered by display VSync. State persists in GPU-accessible unified memory.

**Key Innovations**:
1. Single threadgroup = guaranteed `threadgroup_barrier()` synchronization
2. Unified Worker Model = 100% thread utilization (not fixed SIMD roles)
3. Hybrid rendering = compute for logic, fragment shaders for pixels
4. Memory-as-state = all persistence in GPU-accessible buffers
5. VSync-as-heartbeat = display refresh drives execution

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HARDWARE LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────────────┐   │
│  │   IOKit     │  │   Display   │  │         GPU (M4)              │   │
│  │   HID       │  │   120Hz     │  │  ┌─────────────────────────┐  │   │
│  │   Driver    │  │   VSync     │  │  │   Single Threadgroup    │  │   │
│  └──────┬──────┘  └──────┬──────┘  │  │   (1024 threads)        │  │   │
│         │                │         │  │   Unified Worker Model  │  │   │
│         ▼                ▼         │  └────────────┬────────────┘  │   │
│  ┌──────────────────────────────────────────────────┴───────────────┐  │
│  │                      UNIFIED MEMORY (Shared)                      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │  │
│  │  │  Input   │ │  Widget  │ │  Render  │ │    Framebuffer     │   │  │
│  │  │  Queue   │ │  State   │ │Commands  │ │   (Displayable)    │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Unified Worker Model

**Critical Change from v1**: All 1024 threads participate in ALL phases, not fixed SIMD roles.

### Why This Matters

| Model | Thread Utilization | Performance |
|-------|-------------------|-------------|
| Fixed SIMD Roles (v1) | ~3% per phase | Poor - 97% idle |
| Unified Worker (v2) | 100% per phase | Optimal |

### Execution Flow

```metal
kernel void gpu_os_kernel(
    device Memory* mem [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: INPUT COLLECTION (ALL 1024 threads)
    // ═══════════════════════════════════════════════════════════════
    // Each thread reads one event slot (handles up to 1024 events/frame)
    if (tid < event_count) {
        threadgroup_events[tid] = mem->input_queue.events[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: HIT TESTING (ALL 1024 threads)
    // ═══════════════════════════════════════════════════════════════
    // Each thread tests cursor against one widget (handles up to 1024 widgets)
    bool hit = false;
    if (tid < widget_count) {
        hit = point_in_rect(cursor, mem->widgets[tid].bounds);
    }
    // SIMD ballot to find topmost hit
    simd_vote hits = simd_ballot(hit);
    // Reduction across SIMD groups to find global winner
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: LAYOUT (ALL 1024 threads)
    // ═══════════════════════════════════════════════════════════════
    // Level-order BFS traversal with prefix sums
    for (uint level = 0; level < max_depth; level++) {
        uint level_start = level_offsets[level];
        uint level_count = level_counts[level];

        if (tid < level_count) {
            uint node = level_start + tid;
            // Compute constraints from parent (simd_shuffle if in same group)
            // Compute size using prefix sum for siblings
            float size = compute_flex_size(mem, node);
            float position = simd_prefix_exclusive_sum(size);
            mem->widgets[node].computed_bounds = make_bounds(position, size);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: VISIBILITY & SORTING (ALL 1024 threads)
    // ═══════════════════════════════════════════════════════════════
    // Parallel bitonic sort for Z-order
    if (tid < widget_count) {
        sort_key[tid] = make_sort_key(mem->widgets[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort using simd_shuffle
    for (uint k = 2; k <= 1024; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            bitonic_step(tid, sort_key, j, k);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 5: GEOMETRY GENERATION (ALL 1024 threads)
    // ═══════════════════════════════════════════════════════════════
    // Each thread generates vertices for one widget
    if (tid < visible_count) {
        uint widget_id = sorted_visible[tid];
        generate_quad_vertices(mem, widget_id, tid * 6);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 6: STATE UPDATE (ALL 1024 threads)
    // ═══════════════════════════════════════════════════════════════
    // Parallel state machine updates
    if (tid < widget_count) {
        update_animation_state(mem, tid, delta_time);
    }

    // Thread 0 updates global frame state
    if (tid == 0) {
        mem->frame_state.frame_number++;
        mem->frame_state.time += delta_time;
    }
}
```

---

## Hybrid Rendering Pipeline

**Key Insight**: Compute shaders for logic, fragment shaders for pixels.

### Why Hybrid?

| Task | Best Approach | Reason |
|------|---------------|--------|
| Input processing | Compute | Parallel, no pixels |
| Layout | Compute | Prefix sums, barriers |
| Hit testing | Compute | SIMD ballot |
| Geometry generation | Compute | Parallel vertex output |
| **Pixel rasterization** | **Fragment** | Hardware optimized, TBDR |
| **Blending** | **Fragment** | Hardware ROPs |
| **MSAA** | **Fragment** | Hardware support |

### Render Pass Structure

```swift
func draw(in view: MTKView) {
    let commandBuffer = queue.makeCommandBuffer()!

    // ═══════════════════════════════════════════════════════════════
    // COMPUTE PASS: All OS Logic (Single Threadgroup)
    // ═══════════════════════════════════════════════════════════════
    let compute = commandBuffer.makeComputeCommandEncoder()!
    compute.setComputePipelineState(osKernelPipeline)
    compute.setBuffer(memoryBuffer, offset: 0, index: 0)
    compute.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),           // ONE threadgroup
        threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
    )
    compute.endEncoding()

    // ═══════════════════════════════════════════════════════════════
    // RENDER PASS: Pixel Output (Hardware Rasterization)
    // ═══════════════════════════════════════════════════════════════
    let renderDesc = MTLRenderPassDescriptor()
    renderDesc.colorAttachments[0].texture = drawable.texture
    renderDesc.colorAttachments[0].loadAction = .clear
    renderDesc.colorAttachments[0].storeAction = .store

    let render = commandBuffer.makeRenderCommandEncoder(descriptor: renderDesc)!
    render.setRenderPipelineState(widgetRenderPipeline)
    render.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    // Indirect draw - GPU determined the count
    render.drawPrimitives(
        type: .triangle,
        indirectBuffer: drawArgsBuffer,
        indirectBufferOffset: 0
    )
    render.endEncoding()

    commandBuffer.present(drawable)
    commandBuffer.commit()
}
```

---

## Memory Architecture

### Unified Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU-ACCESSIBLE MEMORY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT BUFFER (4KB) - Written by IOKit, read by GPU             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ atomic head/tail │ InputEvent[256] ring buffer             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  WIDGET STATE (64KB) - GPU read/write                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ WidgetCompact[1024] │ parent[1024] │ children[1024]        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  VERTEX BUFFER (256KB) - GPU write, GPU read                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Vertex[6 * 1024] - 6 vertices per widget quad              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  DRAW ARGS (64B) - GPU write (indirect draw arguments)          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ vertex_count │ instance_count │ base_vertex │ base_instance│ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  FRAME STATE (256B) - GPU read/write                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ frame_number │ time │ cursor │ focused_widget │ modifiers  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  FONT ATLAS (512KB) - GPU read only (MSDF texture)              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1024 glyphs @ 32x32 texels @ 3 bytes (RGB MSDF)            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Total: ~900KB GPU memory for full OS state
```

### Threadgroup Memory (32KB on-chip)

```metal
struct ThreadgroupMemory {
    // Scratch for current frame processing
    InputEvent pending_events[64];           // 1.75KB
    float4 layout_scratch[256];              // 4KB
    uint32_t sort_keys[1024];                // 4KB
    uint32_t sorted_indices[1024];           // 4KB

    // SIMD communication
    float simd_reduction[32];                // 128B

    // Synchronization
    atomic_uint phase_counter;               // 4B
    atomic_uint visible_count;               // 4B

    // Remaining: ~18KB for expansion
};
```

### Register Strategy (Per-Thread Private)

Each thread owns one widget in registers during processing:

```metal
// Thread-local state (lives in registers, not memory)
float4 my_bounds;           // 16 bytes
float4 my_style;            // 16 bytes
uint my_flags;              // 4 bytes
float my_animation_t;       // 4 bytes
// Total: 40 bytes/thread = 40KB across 1024 threads
```

---

## Input Pipeline

### IOKit to GPU Path

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   USB/BT HID    │────▶│   IOKit Driver  │────▶│  Shared Buffer  │
│   (Hardware)    │     │   (Kernel)      │     │  (Unified Mem)  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GPU READS DIRECTLY                           │
│   atomic_load(input_queue.head) → process events → update tail  │
└─────────────────────────────────────────────────────────────────┘
```

### Input Event Structure

```metal
struct InputEvent {
    uint16_t type;              // MOUSE_MOVE, MOUSE_DOWN, KEY_DOWN, etc.
    uint16_t keycode;           // HID keycode or mouse button
    float2 position;            // Cursor position (normalized)
    float2 delta;               // Movement delta
    uint32_t modifiers;         // Shift, Ctrl, Alt, Cmd
    uint32_t timestamp;         // Frame-relative timestamp
};  // 24 bytes

struct InputQueue {
    atomic_uint head;           // Written by IOKit
    atomic_uint tail;           // Written by GPU
    InputEvent events[256];     // Ring buffer
};  // 6KB total
```

### CPU Shim (Minimal)

```swift
// This runs on CPU but is trivial - just copies events to shared buffer
class InputHandler {
    var inputBuffer: MTLBuffer  // Shared with GPU

    func handleHIDEvent(_ event: IOHIDEvent) {
        let queue = inputBuffer.contents().assumingMemoryBound(to: InputQueue.self)
        let head = queue.pointee.head.load(ordering: .relaxed)
        let slot = head % 256
        queue.pointee.events[Int(slot)] = translateEvent(event)
        queue.pointee.head.store(head + 1, ordering: .release)
    }
}
```

---

## Text Rendering

### MSDF Font Atlas Approach

```
Pre-baked:
┌─────────────────────────────────────────────────────────────┐
│   MSDF Font Atlas (512KB)                                    │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┐                        │
│   │ A │ B │ C │ D │ E │ F │ G │...│  32x32 texels per glyph │
│   ├───┼───┼───┼───┼───┼───┼───┼───┤                        │
│   │ a │ b │ c │ d │ e │ f │ g │...│  RGB = distance fields  │
│   └───┴───┴───┴───┴───┴───┴───┴───┘                        │
└─────────────────────────────────────────────────────────────┘

Runtime (Fragment Shader):
┌─────────────────────────────────────────────────────────────┐
│  float3 msd = atlas.sample(uv).rgb;                         │
│  float sd = median(msd.r, msd.g, msd.b);                    │
│  float alpha = smoothstep(0.5 - edge, 0.5 + edge, sd);      │
│  return float4(text_color.rgb, alpha);                       │
└─────────────────────────────────────────────────────────────┘
```

### Text Layout in Compute

```metal
// Each thread handles one character
if (tid < text_length) {
    uint codepoint = text_buffer[tid];
    uint glyph_id = codepoint_to_glyph[codepoint];

    // Prefix sum for horizontal positioning
    float advance = glyph_advances[glyph_id];
    float x_position = simd_prefix_exclusive_sum(advance);

    // Generate quad vertices
    float4 glyph_bounds = float4(x_position, baseline, advance, line_height);
    generate_glyph_quad(glyph_id, glyph_bounds, tid * 6);
}
```

---

## Widget System

### Widget Structure (Compressed for Register Fit)

```metal
struct WidgetCompact {
    half4 bounds;               // 8 bytes (x, y, w, h as half-precision)
    uint32_t packed_colors;     // 4 bytes (bg[16] + border[16])
    uint16_t packed_style;      // 2 bytes (border_width[4] + corner_radius[4] + type[4] + flags[4])
    uint16_t parent_id;         // 2 bytes
    uint16_t first_child;       // 2 bytes
    uint16_t next_sibling;      // 2 bytes
    uint16_t z_order;           // 2 bytes
    uint16_t _padding;          // 2 bytes
};  // 24 bytes total - fits 1024 widgets in 24KB
```

### Widget Types (Branchless Rendering)

```metal
enum WidgetType : uint8_t {
    CONTAINER = 0,    // Just a box, children layout
    BUTTON = 1,       // Clickable, hover states
    TEXT = 2,         // Text content
    IMAGE = 3,        // Texture reference
    SCROLL = 4,       // Scrollable container
    SLIDER = 5,       // Value 0-1
    CHECKBOX = 6,     // Boolean toggle
    INPUT = 7,        // Text input field
};

// Branchless type dispatch using function tables
constant auto widget_renderers = {
    render_container,
    render_button,
    render_text,
    render_image,
    render_scroll,
    render_slider,
    render_checkbox,
    render_input
};

// Call via: widget_renderers[widget.type](widget, output);
```

---

## Performance Budget

### Frame Timeline (8.33ms at 120Hz)

```
0.0ms ─────────────────────────────────────────────────── 8.33ms
│                                                              │
├─ CPU Dispatch (0.5ms) ───────────────────────────────────────┤
│  └─ nextDrawable + commit + present                          │
│                                                              │
├─ GPU Compute (2.0ms) ────────────────────────────────────────┤
│  ├─ Phase 1: Input (0.1ms)                                   │
│  ├─ Phase 2: Hit Test (0.1ms)                                │
│  ├─ Phase 3: Layout (0.5ms)                                  │
│  ├─ Phase 4: Sort (0.3ms)                                    │
│  ├─ Phase 5: Geometry (0.5ms)                                │
│  └─ Phase 6: State (0.5ms)                                   │
│                                                              │
├─ GPU Render (3.0ms) ─────────────────────────────────────────┤
│  └─ Fragment shaders: MSDF text, widget boxes, images        │
│                                                              │
├─ Headroom (2.83ms) ──────────────────────────────────────────┤
│  └─ Available for: animations, complex layouts, spikes       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Scaling Targets

| Metric | Target | Maximum | Notes |
|--------|--------|---------|-------|
| Widgets | 256 | 1024 | Thread count is limit |
| Text characters | 2000 | 8000 | Multi-pass if needed |
| Input events/frame | 64 | 256 | Ring buffer size |
| Tree depth | 6 | 12 | Layout passes |
| Resolution | 4K | 4K | Fragment shader handles |
| Frame rate | 120Hz | 120Hz | VSync locked |

---

## Power Efficiency

### Power Profile (from research)

| State | Power Draw | Duration/Frame |
|-------|-----------|----------------|
| GPU Active (compute) | ~0.7W | 2.0ms |
| GPU Active (render) | ~1.5W | 3.0ms |
| GPU Idle (power-gated) | ~0W | 2.8ms |
| CPU Dispatch | ~0.3W | 0.5ms |
| Display | 2.0W | Always |
| **Average** | **~4.8W** | - |

### Battery Life Estimate

```
MacBook Air (53.8 Wh) / 4.8W = ~11.2 hours
```

Competitive with traditional CPU-based UI approaches.

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Basic Metal setup with single threadgroup dispatch
- [ ] Shared memory buffer structure
- [ ] VSync-triggered frame loop
- [ ] Verify 1024 thread occupancy on target hardware

### Phase 2: Core Systems (Week 3-4)
- [ ] Input queue (ring buffer with atomics)
- [ ] Widget state structure
- [ ] Basic layout (flat, no hierarchy)
- [ ] Hit testing with SIMD ballot

### Phase 3: Layout Engine (Week 5-6)
- [ ] Hierarchical widget tree
- [ ] Flexbox-style constraints
- [ ] Prefix sum positioning
- [ ] Level-order BFS traversal

### Phase 4: Rendering (Week 7-8)
- [ ] Fragment shader pipeline
- [ ] MSDF text rendering
- [ ] Widget box rendering (corners, borders)
- [ ] Indirect draw from compute output

### Phase 5: Polish (Week 9-10)
- [ ] Animations (state interpolation)
- [ ] Scroll physics
- [ ] Focus management
- [ ] Keyboard navigation

---

## Success Criteria

1. **120fps sustained** on M4 MacBook Pro
2. **<16ms input-to-display latency** (ideally <10ms)
3. **<1ms CPU usage** per frame (dispatch only)
4. **256+ widgets** with smooth interaction
5. **Text rendering** at 2000+ characters
6. **11+ hours** battery life estimate
7. **Zero GPU timeouts** in 24-hour stress test

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Register pressure reduces threads | Medium | High | Query `maxTotalThreadsPerThreadgroup`, adapt dynamically |
| 4K rendering exceeds budget | Medium | Medium | Start at 1080p, optimize incrementally |
| Layout complexity exceeds time | Low | Medium | Cache layouts, dirty-rect updates |
| IOKit latency spikes | Low | Low | Ring buffer absorbs bursts |
| GPU driver bugs | Low | High | Extensive testing, fallback path |

---

## Open Questions (Resolved by Research)

| Question | Answer | Source |
|----------|--------|--------|
| Is 1024 threads enough? | Yes, for UI logic | Agent 1 |
| Is 32KB sufficient? | Yes, with compression | Agent 2 |
| Can VSync drive execution? | Yes, via MTKView | Agent 3 |
| Can IOKit write to GPU buffer? | Yes, unified memory | Agent 4 |
| Is barrier safe? | Yes, within threadgroup | Agent 5 |
| Compute vs fragment for pixels? | Fragment wins | Agent 6 |
| Power efficiency? | Competitive (~11hr) | Agent 11 |
| Watchdog timeout risk? | None at 6-8ms | Agent 10 |

---

## Appendix: Key Code Patterns

### A. SIMD Prefix Sum for Layout

```metal
float position = simd_prefix_exclusive_sum(size);
// Now each thread knows its position based on sum of preceding sizes
```

### B. SIMD Ballot for Hit Testing

```metal
bool hit = point_in_rect(cursor, bounds);
simd_vote hits = simd_ballot(hit);
uint first_hit = ctz(hits);  // Count trailing zeros
```

### C. Atomic Ring Buffer

```metal
uint head = atomic_load_explicit(&queue->head, memory_order_acquire);
uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
uint count = head - tail;  // Events available
// Process events...
atomic_store_explicit(&queue->tail, tail + count, memory_order_release);
```

### D. Branchless Widget Dispatch

```metal
// Instead of switch/case (causes divergence)
float4 color = select(bg_color, hover_color, is_hovered);
float corner = select(0.0, corner_radius, has_corners);
```

---

*This PRD represents a GPU-first approach to OS architecture, trading traditional CPU-centric design for guaranteed synchronization, parallel execution, and power efficiency.*
