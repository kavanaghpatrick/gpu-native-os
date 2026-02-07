//! GPU Boot Sequence - Metal Kernels
//!
//! Issue #170 - GPU Boot Sequence
//!
//! THE GPU IS THE COMPUTER.
//! GPU initializes system, discovers apps, loads and launches.
//! CPU just provides buffers and I/O coprocessor.

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

constant uint MAX_BOOT_APPS = 32;
constant uint MAX_SLOTS = 64;

// Boot phases
constant uint PHASE_UNINIT = 0;
constant uint PHASE_SYSTEM_INIT = 1;
constant uint PHASE_SYSTEM_APPS = 2;
constant uint PHASE_DISCOVER = 3;
constant uint PHASE_LOADING = 4;
constant uint PHASE_COMPLETE = 5;

// System app types
constant uint APP_TYPE_COMPOSITOR = 200;
constant uint APP_TYPE_DOCK = 201;
constant uint APP_TYPE_MENUBAR = 202;
constant uint APP_TYPE_WINDOW_CHROME = 203;

// App flags
constant uint APP_FLAG_ACTIVE = 1;
constant uint APP_FLAG_VISIBLE = 2;
constant uint APP_FLAG_DIRTY = 4;

// Priority
constant uint PRIORITY_REALTIME = 0;

constant uint INVALID_SLOT = 0xFFFFFFFF;
constant uint INVALID_HANDLE = 0xFFFFFFFF;

// Memory layout
constant uint HEADER_SIZE = 32;
constant uint BITMAP_SIZE = 8;
constant uint DESCRIPTOR_SIZE = 128;
constant uint VERTEX_SIZE = 48;
constant uint VERTS_PER_SLOT = 1024;
constant uint STATE_CHUNK_SIZE = 64 * 1024;

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// App table header (32 bytes)
struct AppTableHeader {
    uint magic;
    uint version;
    uint max_slots;
    atomic_uint active_count;  // Issue #259 fix: must be atomic for safe concurrent access
    atomic_uint next_slot_hint;
    uint _pad[3];
};

// Boot state
struct BootState {
    // Phase tracking
    uint current_phase;
    uint phase_complete;

    // App discovery
    uint discovered_apps[MAX_BOOT_APPS];
    uint discovered_count;

    // Pending loads
    uint pending_handles[MAX_BOOT_APPS];
    uint pending_count;
    uint loaded_count;

    // Errors
    uint error_count;
    uint last_error;

    // Stats
    uint boot_start_frame;
    uint boot_end_frame;

    // Padding
    uint _pad[2];
};

// App descriptor (128 bytes) - simplified for boot
struct GpuAppDescriptor {
    // Identity & Lifecycle (16 bytes)
    uint flags;
    uint app_type;
    uint slot_id;
    uint window_id;

    // Memory (16 bytes)
    uint state_offset;
    uint state_size;
    uint vertex_offset;
    uint vertex_size;

    // Render state (16 bytes)
    uint vertex_count;
    uint vertex_budget;
    uint depth;
    uint render_order;

    // Timing (16 bytes)
    uint last_run_frame;
    uint frames_alive;
    uint priority;
    uint cpu_hint;

    // Layout (16 bytes)
    float x, y, width, height;

    // Visual (16 bytes)
    float bg_r, bg_g, bg_b, bg_a;

    // Animation (16 bytes)
    float anim_progress;
    float anim_speed;
    uint anim_state;
    uint _pad1;

    // Debug (16 bytes)
    uint debug_flags;
    uint error_code;
    uint _pad2[2];
};

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 1: SYSTEM INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

kernel void boot_phase1_system_init(
    device uchar* app_table_data [[buffer(0)]],
    device BootState* boot [[buffer(1)]],
    constant uint& max_slots [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Thread 0 initializes header
    if (tid == 0) {
        device AppTableHeader* header = (device AppTableHeader*)app_table_data;
        header->magic = 0x54505041;  // "APPT"
        header->version = 1;
        header->max_slots = max_slots;
        // Issue #259 fix: Use atomic store for atomic fields
        atomic_store_explicit(&header->active_count, 0, memory_order_relaxed);
        atomic_store_explicit(&header->next_slot_hint, 0, memory_order_relaxed);

        // Initialize free bitmap: all free (all 1s)
        // Issue #259 fix: Use atomic types for bitmap
        device atomic_uint* bitmap = (device atomic_uint*)(app_table_data + HEADER_SIZE);
        atomic_store_explicit(&bitmap[0], 0xFFFFFFFF, memory_order_relaxed);
        atomic_store_explicit(&bitmap[1], 0xFFFFFFFF, memory_order_relaxed);

        // Initialize boot state
        boot->current_phase = PHASE_SYSTEM_INIT;
        boot->phase_complete = 0;
        boot->discovered_count = 0;
        boot->pending_count = 0;
        boot->loaded_count = 0;
        boot->error_count = 0;
        boot->last_error = 0;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // All threads initialize their slot's descriptor (parallel)
    if (tid < max_slots) {
        device GpuAppDescriptor* apps = (device GpuAppDescriptor*)(app_table_data + HEADER_SIZE + BITMAP_SIZE);
        device GpuAppDescriptor* app = &apps[tid];

        app->flags = 0;  // INACTIVE
        app->app_type = 0;
        app->slot_id = tid;
        app->window_id = 0;
        app->state_offset = 0;
        app->state_size = 0;
        app->vertex_offset = 0;
        app->vertex_size = 0;
        app->vertex_count = 0;
        app->vertex_budget = 0;
        app->depth = 0;
        app->render_order = 0;
        app->last_run_frame = 0;
        app->frames_alive = 0;
        app->priority = 5;  // Default priority
        app->cpu_hint = 0;
        app->x = 0;
        app->y = 0;
        app->width = 0;
        app->height = 0;
        app->bg_r = 0.2;
        app->bg_g = 0.2;
        app->bg_b = 0.3;
        app->bg_a = 1.0;
        app->anim_progress = 0;
        app->anim_speed = 0;
        app->anim_state = 0;
        app->debug_flags = 0;
        app->error_code = 0;
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        boot->phase_complete = PHASE_SYSTEM_INIT;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHASE 2: SYSTEM APPS
// ═══════════════════════════════════════════════════════════════════════════════

// Allocate a specific slot (used for system apps with fixed slots)
// Issue #259 fix: Use atomic bitmap operations to prevent race conditions
inline void allocate_slot(
    device AppTableHeader* header,
    device atomic_uint* bitmap,  // Changed to atomic_uint*
    uint slot
) {
    // Clear bit in bitmap atomically
    uint word = slot / 32;
    uint bit = slot % 32;
    uint mask = ~(1u << bit);
    atomic_fetch_and_explicit(&bitmap[word], mask, memory_order_relaxed);
    atomic_fetch_add_explicit(&header->active_count, 1, memory_order_relaxed);
}

kernel void boot_phase2_system_apps(
    device uchar* app_table_data [[buffer(0)]],
    device uchar* unified_state [[buffer(1)]],
    device BootState* boot [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Wait for phase 1
    if (boot->phase_complete < PHASE_SYSTEM_INIT) return;

    device AppTableHeader* header = (device AppTableHeader*)app_table_data;
    // Issue #259 fix: Cast bitmap to atomic_uint* for atomic operations
    device atomic_uint* bitmap = (device atomic_uint*)(app_table_data + HEADER_SIZE);
    device GpuAppDescriptor* apps = (device GpuAppDescriptor*)(app_table_data + HEADER_SIZE + BITMAP_SIZE);

    // Each thread initializes one system app (slots 0-3)
    if (tid == 0) {
        // Compositor (slot 0)
        allocate_slot(header, bitmap, 0);
        device GpuAppDescriptor* app = &apps[0];
        app->flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE;
        app->app_type = APP_TYPE_COMPOSITOR;
        app->slot_id = 0;
        app->window_id = 0;
        app->priority = PRIORITY_REALTIME;
        app->state_offset = 0 * STATE_CHUNK_SIZE;
        app->state_size = 1024;
        app->vertex_offset = 0 * VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_size = VERTS_PER_SLOT * VERTEX_SIZE;
        app->depth = 100;  // Compositor renders last
    }
    else if (tid == 1) {
        // Dock (slot 1)
        allocate_slot(header, bitmap, 1);
        device GpuAppDescriptor* app = &apps[1];
        app->flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE;
        app->app_type = APP_TYPE_DOCK;
        app->slot_id = 1;
        app->window_id = 1;
        app->priority = PRIORITY_REALTIME;
        app->state_offset = 1 * STATE_CHUNK_SIZE;
        app->state_size = 4096;
        app->vertex_offset = 1 * VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_size = VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_budget = VERTS_PER_SLOT;
        app->depth = 90;
        app->y = 0;  // Bottom of screen
        app->height = 70;
    }
    else if (tid == 2) {
        // MenuBar (slot 2)
        allocate_slot(header, bitmap, 2);
        device GpuAppDescriptor* app = &apps[2];
        app->flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE;
        app->app_type = APP_TYPE_MENUBAR;
        app->slot_id = 2;
        app->window_id = 2;
        app->priority = PRIORITY_REALTIME;
        app->state_offset = 2 * STATE_CHUNK_SIZE;
        app->state_size = 8192;
        app->vertex_offset = 2 * VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_size = VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_budget = VERTS_PER_SLOT;
        app->depth = 90;
        app->y = 730;  // Top of screen (assuming 800 height, 70px menu bar)
        app->height = 30;
    }
    else if (tid == 3) {
        // Window Chrome (slot 3)
        allocate_slot(header, bitmap, 3);
        device GpuAppDescriptor* app = &apps[3];
        app->flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE;
        app->app_type = APP_TYPE_WINDOW_CHROME;
        app->slot_id = 3;
        app->window_id = 3;
        app->priority = PRIORITY_REALTIME;
        app->state_offset = 3 * STATE_CHUNK_SIZE;
        app->state_size = 256;
        app->vertex_offset = 3 * VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_size = VERTS_PER_SLOT * VERTEX_SIZE;
        app->vertex_budget = VERTS_PER_SLOT;
        app->depth = 80;
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        boot->current_phase = PHASE_SYSTEM_APPS;
        boot->phase_complete = PHASE_SYSTEM_APPS;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BOOT COMPLETION CHECK
// ═══════════════════════════════════════════════════════════════════════════════

kernel void boot_check_complete(
    device BootState* boot [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // For now, skip discover/load phases and mark complete
    // In full implementation, this would poll pending_count
    boot->current_phase = PHASE_COMPLETE;
    boot->phase_complete = PHASE_COMPLETE;
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUERY FUNCTIONS (for testing)
// ═══════════════════════════════════════════════════════════════════════════════

kernel void boot_get_active_count(
    device const uchar* app_table_data [[buffer(0)]],
    device uint* result [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    device const AppTableHeader* header = (device const AppTableHeader*)app_table_data;
    *result = header->active_count;
}

kernel void boot_get_app_type(
    device const uchar* app_table_data [[buffer(0)]],
    constant uint& slot [[buffer(1)]],
    device uint* result [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    device const GpuAppDescriptor* apps = (device const GpuAppDescriptor*)(app_table_data + HEADER_SIZE + BITMAP_SIZE);
    *result = apps[slot].app_type;
}
