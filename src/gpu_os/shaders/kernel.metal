// Issue #11: Unified Worker Model - Core GPU Kernel
//
// All 1024 threads participate in ALL phases (no fixed SIMD roles).
// This achieves 100% thread utilization per phase.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Data Structures (must match Rust definitions)
// ============================================================================

struct WidgetCompact {
    ushort4 bounds;             // 8 bytes (x, y, width, height as f16 bits)
    uint packed_colors;         // 4 bytes (bg[16] | border[16])
    ushort packed_style;        // 2 bytes
    ushort parent_id;           // 2 bytes
    ushort first_child;         // 2 bytes
    ushort next_sibling;        // 2 bytes
    ushort z_order;             // 2 bytes
    ushort _padding;            // 2 bytes
};  // 24 bytes total

struct InputEvent {
    ushort event_type;          // 2 bytes
    ushort keycode;             // 2 bytes
    float2 position;            // 8 bytes
    float2 delta;               // 8 bytes
    uint modifiers;             // 4 bytes
    uint timestamp;             // 4 bytes
};  // 28 bytes total

struct InputQueue {
    atomic_uint head;           // 4 bytes
    atomic_uint tail;           // 4 bytes
    uint _padding[2];           // 8 bytes
    InputEvent events[256];     // Ring buffer
};

struct DrawArguments {
    uint vertex_count;
    uint instance_count;
    uint vertex_start;
    uint base_instance;
};

struct FrameState {
    uint frame_number;
    float time;
    float cursor_x;
    float cursor_y;
    uint focused_widget;
    uint hovered_widget;
    uint modifiers;
    uint _padding;
};

// ============================================================================
// Kernel Parameters
// ============================================================================

struct KernelParams {
    uint widget_count;
    uint max_widgets;
    float delta_time;
    float time;
    uint frame_number;
};

// ============================================================================
// Threadgroup Shared Memory
// ============================================================================

// Note: Keep under 32KB to ensure occupancy
struct SharedData {
    InputEvent pending_events[64];      // 64 * 28 = 1792 bytes
    uint event_count;                    // 4 bytes
    float cursor_x;                      // 4 bytes
    float cursor_y;                      // 4 bytes
    atomic_uint hit_count;               // 4 bytes
    atomic_uint topmost_hit;             // 4 bytes
    atomic_uint topmost_z;               // 4 bytes
    atomic_uint visible_count;           // 4 bytes
    uint sort_keys[1024];                // 4KB
    // Total: ~6KB threadgroup memory
};

// ============================================================================
// Helper Functions
// ============================================================================

// Convert f16 bits to float
inline float f16_to_float(ushort bits) {
    // Use Metal's built-in half type for proper conversion
    return float(as_type<half>(bits));
}

// Check if point is inside widget bounds
inline bool point_in_rect(float2 point, ushort4 bounds) {
    float x = f16_to_float(bounds.x);
    float y = f16_to_float(bounds.y);
    float w = f16_to_float(bounds.z);
    float h = f16_to_float(bounds.w);
    return point.x >= x && point.x <= x + w &&
           point.y >= y && point.y <= y + h;
}

// Extract widget type from packed_style (bits 4-7)
inline uint get_widget_type(ushort packed_style) {
    return (packed_style >> 4) & 0xF;
}

// Check if widget is visible (flag bit 0)
inline bool is_visible(ushort packed_style) {
    return (packed_style & 0x1) != 0;
}

// ============================================================================
// Main Unified Worker Kernel
// ============================================================================

kernel void gpu_os_kernel(
    device WidgetCompact* widgets [[buffer(0)]],
    device InputQueue* input_queue [[buffer(1)]],
    device DrawArguments* draw_args [[buffer(2)]],
    device FrameState* frame_state [[buffer(3)]],
    constant KernelParams& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Shared threadgroup memory
    threadgroup SharedData shared;

    // =========================================================================
    // PHASE 1: INPUT COLLECTION (ALL 1024 threads participate)
    // =========================================================================

    if (tid == 0) {
        // Thread 0 initializes shared state
        // Note: Metal only supports memory_order_relaxed for device memory
        uint head = atomic_load_explicit(&input_queue->head, memory_order_relaxed);
        uint tail = atomic_load_explicit(&input_queue->tail, memory_order_relaxed);
        shared.event_count = min(head - tail, 64u);
        shared.cursor_x = frame_state->cursor_x;
        shared.cursor_y = frame_state->cursor_y;
        atomic_store_explicit(&shared.hit_count, 0, memory_order_relaxed);
        atomic_store_explicit(&shared.topmost_hit, 0xFFFFFFFF, memory_order_relaxed);
        atomic_store_explicit(&shared.topmost_z, 0, memory_order_relaxed);
        atomic_store_explicit(&shared.visible_count, 0, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread reads one event (up to 64 events per frame)
    if (tid < shared.event_count) {
        uint tail = atomic_load_explicit(&input_queue->tail, memory_order_relaxed);
        uint slot = (tail + tid) % 256;
        shared.pending_events[tid] = input_queue->events[slot];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 processes events and updates state
    if (tid == 0 && shared.event_count > 0) {
        uint old_tail = atomic_load_explicit(&input_queue->tail, memory_order_relaxed);
        atomic_store_explicit(&input_queue->tail, old_tail + shared.event_count, memory_order_relaxed);

        // Update cursor from mouse move events
        for (uint i = 0; i < shared.event_count; i++) {
            if (shared.pending_events[i].event_type == 1) { // MOUSE_MOVE
                shared.cursor_x = shared.pending_events[i].position.x;
                shared.cursor_y = shared.pending_events[i].position.y;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 2: HIT TESTING (ALL 1024 threads participate)
    // =========================================================================

    bool hit = false;
    ushort my_z = 0;

    if (tid < params.widget_count) {
        WidgetCompact w = widgets[tid];
        if (is_visible(w.packed_style)) {
            float2 cursor = float2(shared.cursor_x, shared.cursor_y);
            hit = point_in_rect(cursor, w.bounds);
            my_z = w.z_order;
        }
    }

    // Use SIMD operations for efficient hit detection
    bool any_hit_in_simd = simd_any(hit);

    if (hit) {
        atomic_fetch_add_explicit(&shared.hit_count, 1, memory_order_relaxed);

        // Atomic max to find topmost (highest z-order) hit
        uint current_z = atomic_load_explicit(&shared.topmost_z, memory_order_relaxed);
        while (my_z > current_z) {
            if (atomic_compare_exchange_weak_explicit(
                &shared.topmost_z, &current_z, uint(my_z),
                memory_order_relaxed, memory_order_relaxed)) {
                atomic_store_explicit(&shared.topmost_hit, tid, memory_order_relaxed);
                break;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 3: VISIBILITY COUNTING (ALL 1024 threads participate)
    // =========================================================================

    bool visible = false;
    uint my_sort_key = 0xFFFFFFFF;  // Max = not visible

    if (tid < params.widget_count) {
        WidgetCompact w = widgets[tid];
        visible = is_visible(w.packed_style);
        if (visible) {
            my_sort_key = (uint(w.z_order) << 16) | tid;
            atomic_fetch_add_explicit(&shared.visible_count, 1, memory_order_relaxed);
        }
    }

    shared.sort_keys[tid] = my_sort_key;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 4: BITONIC SORT BY Z-ORDER (ALL 1024 threads participate)
    // =========================================================================

    // Bitonic sort for 1024 elements
    for (uint k = 2; k <= 1024; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            uint ixj = tid ^ j;
            if (ixj > tid) {
                uint key_i = shared.sort_keys[tid];
                uint key_j = shared.sort_keys[ixj];

                bool ascending = (tid & k) == 0;
                bool should_swap = ascending ? (key_i > key_j) : (key_i < key_j);

                if (should_swap) {
                    shared.sort_keys[tid] = key_j;
                    shared.sort_keys[ixj] = key_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // =========================================================================
    // PHASE 5: STATE UPDATE (ALL 1024 threads participate)
    // =========================================================================

    // Each thread could update animation state for its widget
    if (tid < params.widget_count) {
        // Animation updates would go here
        // widgets[tid].some_animation_state += params.delta_time;
    }

    // Thread 0 updates global frame state and draw arguments
    if (tid == 0) {
        frame_state->frame_number = params.frame_number + 1;
        frame_state->time = params.time + params.delta_time;
        frame_state->cursor_x = shared.cursor_x;
        frame_state->cursor_y = shared.cursor_y;

        uint hit_widget = atomic_load_explicit(&shared.topmost_hit, memory_order_relaxed);
        if (hit_widget != 0xFFFFFFFF) {
            frame_state->hovered_widget = hit_widget;
        }

        uint vis_count = atomic_load_explicit(&shared.visible_count, memory_order_relaxed);
        draw_args->vertex_count = vis_count * 6;
        draw_args->instance_count = 1;
        draw_args->vertex_start = 0;
        draw_args->base_instance = 0;
    }
}

// ============================================================================
// Test Kernel (for basic verification)
// ============================================================================

kernel void test_kernel(
    device uint* output [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Each thread writes its ID to verify execution
    if (tgid == 0) {
        output[tid] = tid + 1;  // +1 to distinguish from zero-init
    }
}
