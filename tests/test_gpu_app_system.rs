// GPU-Centric App System Tests (Issue #154)
//
// THE GPU IS THE COMPUTER. These tests verify that app lifecycle
// is managed BY the GPU, not by the CPU.

use metal::*;

// ============================================================================
// Test Structures (match Metal shader definitions)
// ============================================================================

/// App flags (must match Metal constants)
pub mod flags {
    pub const ACTIVE: u32 = 1 << 0;
    pub const VISIBLE: u32 = 1 << 1;
    pub const DIRTY: u32 = 1 << 2;
    pub const SUSPENDED: u32 = 1 << 3;
    pub const FOCUS: u32 = 1 << 4;
    pub const NEEDS_INIT: u32 = 1 << 5;
    pub const CLOSING: u32 = 1 << 6;
}

/// Priority levels
pub mod priority {
    pub const BACKGROUND: u32 = 0;
    pub const NORMAL: u32 = 1;
    pub const HIGH: u32 = 2;
    pub const REALTIME: u32 = 3;
}

/// App types (must match megakernel switch cases)
pub mod app_type {
    pub const NONE: u32 = 0;
    pub const GAME_OF_LIFE: u32 = 1;
    pub const PARTICLES: u32 = 2;
    pub const TEXT_EDITOR: u32 = 3;
    pub const FILESYSTEM: u32 = 4;
    pub const TERMINAL: u32 = 5;
    pub const DOCUMENT: u32 = 6;
    pub const CUSTOM: u32 = 100;
}

/// Invalid slot sentinel
pub const INVALID_SLOT: u32 = 0xFFFFFFFF;

/// GPU App Descriptor - 128 bytes, GPU-resident
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuAppDescriptor {
    // Identity & Lifecycle (16 bytes)
    pub flags: u32,
    pub app_type: u32,
    pub slot_id: u32,
    pub window_id: u32,

    // Memory Pointers (32 bytes)
    pub state_offset: u32,
    pub state_size: u32,
    pub vertex_offset: u32,
    pub vertex_size: u32,
    pub param_offset: u32,
    pub param_size: u32,
    pub _mem_pad: [u32; 2],

    // Execution State (16 bytes)
    pub frame_number: u32,
    pub input_head: u32,
    pub input_tail: u32,
    pub thread_count: u32,

    // Rendering (16 bytes)
    pub vertex_count: u32,
    pub clear_color: u32,
    pub preferred_width: f32,
    pub preferred_height: f32,

    // GPU Scheduling (16 bytes)
    pub priority: u32,
    pub last_run_frame: u32,
    pub accumulated_time: u32,
    pub _sched_pad: u32,

    // Input Queue (32 bytes) - inline ring buffer
    pub input_events: [u32; 8],
}

// Compile-time size check
const _: () = assert!(std::mem::size_of::<GpuAppDescriptor>() == 128);

/// App table header - lives at start of app table buffer
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AppTableHeader {
    pub max_slots: u32,
    pub active_count: u32,  // atomic in Metal
    pub free_bitmap: [u32; 2],  // 64 slots worth of bitmap
    pub _pad: [u32; 4],
}

/// Scheduler state
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SchedulerState {
    pub active_app_count: u32,
    pub total_compute_budget: u32,
    pub used_compute_budget: u32,
    pub _pad0: u32,
    pub priority_thresholds: [u32; 4],
    pub frame_quantum: u32,
    pub current_frame: u32,
    pub _pad1: [u32; 2],
}

// ============================================================================
// Metal Shader for Testing
// ============================================================================

const GPU_APP_SYSTEM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants (must match Rust)
// ============================================================================

constant uint APP_FLAG_ACTIVE = 1 << 0;
constant uint APP_FLAG_VISIBLE = 1 << 1;
constant uint APP_FLAG_DIRTY = 1 << 2;
constant uint APP_FLAG_SUSPENDED = 1 << 3;
constant uint APP_FLAG_FOCUS = 1 << 4;
constant uint APP_FLAG_NEEDS_INIT = 1 << 5;
constant uint APP_FLAG_CLOSING = 1 << 6;

constant uint PRIORITY_BACKGROUND = 0;
constant uint PRIORITY_NORMAL = 1;
constant uint PRIORITY_HIGH = 2;
constant uint PRIORITY_REALTIME = 3;

constant uint APP_TYPE_NONE = 0;
constant uint APP_TYPE_GAME_OF_LIFE = 1;
constant uint APP_TYPE_PARTICLES = 2;
constant uint APP_TYPE_TEXT_EDITOR = 3;
constant uint APP_TYPE_FILESYSTEM = 4;
constant uint APP_TYPE_TERMINAL = 5;
constant uint APP_TYPE_DOCUMENT = 6;
constant uint APP_TYPE_CUSTOM = 100;

constant uint INVALID_SLOT = 0xFFFFFFFF;
constant uint MAX_SLOTS = 64;

// ============================================================================
// Structures (must match Rust #[repr(C)])
// ============================================================================

struct GpuAppDescriptor {
    // Identity & Lifecycle (16 bytes)
    uint flags;
    uint app_type;
    uint slot_id;
    uint window_id;

    // Memory Pointers (32 bytes)
    uint state_offset;
    uint state_size;
    uint vertex_offset;
    uint vertex_size;
    uint param_offset;
    uint param_size;
    uint _mem_pad[2];

    // Execution State (16 bytes)
    uint frame_number;
    uint input_head;
    uint input_tail;
    uint thread_count;

    // Rendering (16 bytes)
    uint vertex_count;
    uint clear_color;
    float preferred_width;
    float preferred_height;

    // GPU Scheduling (16 bytes)
    uint priority;
    uint last_run_frame;
    uint accumulated_time;
    uint _sched_pad;

    // Input Queue (32 bytes)
    uint input_events[8];
};

struct AppTableHeader {
    uint max_slots;
    atomic_uint active_count;
    atomic_uint free_bitmap[2];  // 64 slots
    uint _pad[4];
};

struct SchedulerState {
    atomic_uint active_app_count;
    uint total_compute_budget;
    atomic_uint used_compute_budget;
    uint _pad0;
    uint priority_thresholds[4];
    uint frame_quantum;
    uint current_frame;
    uint _pad1[2];
};

struct AllocatorState {
    atomic_uint bump_pointer;
    uint pool_size;
    atomic_uint allocation_count;
    uint peak_usage;
};

// ============================================================================
// GPU SLOT ALLOCATOR
// Uses atomic bitmap operations for O(1) allocation
// ============================================================================

// Allocate a slot from the bitmap (returns INVALID_SLOT if none available)
inline uint allocate_slot(device AppTableHeader* header) {
    // Try each bitmap word
    for (uint word = 0; word < 2; word++) {
        uint bitmap = atomic_load_explicit(&header->free_bitmap[word], memory_order_relaxed);

        while (bitmap != 0) {
            // Find first set bit (free slot)
            uint bit = ctz(bitmap);  // Count trailing zeros
            uint slot = word * 32 + bit;

            if (slot >= header->max_slots) break;

            // Try to claim it
            uint mask = 1u << bit;
            uint old = atomic_fetch_and_explicit(
                &header->free_bitmap[word],
                ~mask,
                memory_order_relaxed
            );

            if (old & mask) {
                // Successfully claimed
                atomic_fetch_add_explicit(&header->active_count, 1, memory_order_relaxed);
                return slot;
            }

            // Someone else got it, try again
            bitmap = atomic_load_explicit(&header->free_bitmap[word], memory_order_relaxed);
        }
    }

    return INVALID_SLOT;
}

// Free a slot back to the bitmap
inline void free_slot(device AppTableHeader* header, uint slot) {
    if (slot >= header->max_slots) return;

    uint word = slot / 32;
    uint bit = slot % 32;
    uint mask = 1u << bit;

    atomic_fetch_or_explicit(&header->free_bitmap[word], mask, memory_order_relaxed);
    atomic_fetch_sub_explicit(&header->active_count, 1, memory_order_relaxed);
}

// ============================================================================
// GPU MEMORY ALLOCATOR (bump pointer with prefix sum batching)
// ============================================================================

// Simple bump allocation for testing
inline uint gpu_alloc(device AllocatorState* alloc, uint size, uint alignment) {
    // Align size
    size = (size + alignment - 1) & ~(alignment - 1);

    // Atomically reserve space
    uint offset = atomic_fetch_add_explicit(&alloc->bump_pointer, size, memory_order_relaxed);

    // Check bounds
    if (offset + size > alloc->pool_size) {
        // Out of memory - roll back
        atomic_fetch_sub_explicit(&alloc->bump_pointer, size, memory_order_relaxed);
        return INVALID_SLOT;
    }

    atomic_fetch_add_explicit(&alloc->allocation_count, 1, memory_order_relaxed);
    return offset;
}

// ============================================================================
// GPU APP LIFECYCLE
// ============================================================================

// Launch an app (called from terminal or other apps)
kernel void gpu_launch_app(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device AllocatorState* state_alloc [[buffer(2)]],
    device AllocatorState* vertex_alloc [[buffer(3)]],
    constant uint& app_type [[buffer(4)]],
    constant uint& state_size [[buffer(5)]],
    constant uint& vertex_size [[buffer(6)]],
    device uint* result_slot [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    // Only thread 0 does the allocation
    if (tid != 0) return;

    // 1. Allocate slot
    uint slot = allocate_slot(header);
    if (slot == INVALID_SLOT) {
        *result_slot = INVALID_SLOT;
        return;
    }

    // 2. Allocate memory
    uint state_off = gpu_alloc(state_alloc, state_size, 16);
    uint vertex_off = gpu_alloc(vertex_alloc, vertex_size, 16);

    if (state_off == INVALID_SLOT || vertex_off == INVALID_SLOT) {
        // Rollback
        free_slot(header, slot);
        *result_slot = INVALID_SLOT;
        return;
    }

    // 3. Initialize descriptor
    apps[slot].flags = APP_FLAG_ACTIVE | APP_FLAG_VISIBLE | APP_FLAG_DIRTY | APP_FLAG_NEEDS_INIT;
    apps[slot].app_type = app_type;
    apps[slot].slot_id = slot;
    apps[slot].window_id = 0;
    apps[slot].state_offset = state_off;
    apps[slot].state_size = state_size;
    apps[slot].vertex_offset = vertex_off;
    apps[slot].vertex_size = vertex_size;
    apps[slot].thread_count = 1024;
    apps[slot].priority = PRIORITY_NORMAL;

    *result_slot = slot;
}

// Close an app
kernel void gpu_close_app(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    constant uint& slot_id [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    if (slot_id >= header->max_slots) return;

    // Clear flags (marks for cleanup)
    apps[slot_id].flags = 0;

    // Free slot
    free_slot(header, slot_id);
}

// ============================================================================
// MEGAKERNEL - All app logic in one dispatch
// ============================================================================

// Simple test app: increments a counter in state
inline void test_app_update(
    device GpuAppDescriptor* app,
    device uchar* unified_state
) {
    device uint* counter = (device uint*)(unified_state + app->state_offset);
    *counter = *counter + 1;
    app->vertex_count = 6;  // One quad
}

// Megakernel that processes all apps
kernel void gpu_app_megakernel(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device uchar* unified_state [[buffer(2)]],
    constant uint& frame_number [[buffer(3)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;

    GpuAppDescriptor app = apps[slot_id];

    // Skip inactive
    if (!(app.flags & APP_FLAG_ACTIVE)) return;

    // Skip if not dirty
    if (!(app.flags & APP_FLAG_DIRTY)) return;

    // Dispatch based on app type
    switch (app.app_type) {
        case APP_TYPE_CUSTOM:
            test_app_update(&apps[slot_id], unified_state);
            break;
        // Other app types would go here
        default:
            break;
    }

    // Update frame tracking
    apps[slot_id].last_run_frame = frame_number;
    apps[slot_id].flags &= ~APP_FLAG_DIRTY;
}

// ============================================================================
// SCHEDULER - Determines which apps run
// ============================================================================

kernel void gpu_scheduler(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device SchedulerState* sched [[buffer(2)]],
    constant uint& frame_number [[buffer(3)]],
    device uint* runnable_count [[buffer(4)]],
    uint slot_id [[thread_position_in_grid]]
) {
    if (slot_id >= header->max_slots) return;

    GpuAppDescriptor app = apps[slot_id];

    // Skip inactive
    if (!(app.flags & APP_FLAG_ACTIVE)) return;

    // Mark as dirty if needed (for testing, always dirty)
    if (app.last_run_frame < frame_number) {
        apps[slot_id].flags |= APP_FLAG_DIRTY;
    }

    // Count runnable apps
    if (apps[slot_id].flags & APP_FLAG_DIRTY) {
        atomic_fetch_add_explicit((device atomic_uint*)runnable_count, 1, memory_order_relaxed);
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

kernel void get_app_stats(
    device const AppTableHeader* header [[buffer(0)]],
    device const GpuAppDescriptor* apps [[buffer(1)]],
    device uint* stats [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    stats[0] = atomic_load_explicit(&header->active_count, memory_order_relaxed);

    // Count by type
    uint counts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint i = 0; i < header->max_slots; i++) {
        if (apps[i].flags & APP_FLAG_ACTIVE) {
            uint type = min(apps[i].app_type, 7u);
            counts[type]++;
        }
    }

    stats[1] = counts[0];  // NONE
    stats[2] = counts[1];  // GAME_OF_LIFE
    stats[3] = counts[2];  // PARTICLES
    stats[4] = counts[3];  // TEXT_EDITOR
}
"#;

// ============================================================================
// Test Fixture
// ============================================================================

struct GpuAppSystemTest {
    device: Device,
    command_queue: CommandQueue,

    // Pipelines
    launch_pipeline: ComputePipelineState,
    close_pipeline: ComputePipelineState,
    megakernel_pipeline: ComputePipelineState,
    scheduler_pipeline: ComputePipelineState,
    stats_pipeline: ComputePipelineState,

    // Buffers
    app_table_buffer: Buffer,   // AppTableHeader + GpuAppDescriptor[]
    state_alloc_buffer: Buffer, // AllocatorState for app state
    vertex_alloc_buffer: Buffer, // AllocatorState for vertices
    unified_state_buffer: Buffer,
    scheduler_buffer: Buffer,
}

impl GpuAppSystemTest {
    fn new(device: &Device, max_slots: u32) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shader
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(GPU_APP_SYSTEM_SHADER, &options)
            .map_err(|e| format!("Shader compile failed: {}", e))?;

        let launch_pipeline = Self::create_pipeline(device, &library, "gpu_launch_app")?;
        let close_pipeline = Self::create_pipeline(device, &library, "gpu_close_app")?;
        let megakernel_pipeline = Self::create_pipeline(device, &library, "gpu_app_megakernel")?;
        let scheduler_pipeline = Self::create_pipeline(device, &library, "gpu_scheduler")?;
        let stats_pipeline = Self::create_pipeline(device, &library, "get_app_stats")?;

        // Create app table buffer (header + slots)
        let header_size = std::mem::size_of::<AppTableHeader>();
        let slots_size = (max_slots as usize) * std::mem::size_of::<GpuAppDescriptor>();
        let app_table_buffer = device.new_buffer(
            (header_size + slots_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize header
        unsafe {
            let header = app_table_buffer.contents() as *mut AppTableHeader;
            (*header).max_slots = max_slots;
            (*header).active_count = 0;
            // All slots free (1 = free, 0 = used)
            (*header).free_bitmap[0] = 0xFFFFFFFF;
            (*header).free_bitmap[1] = 0xFFFFFFFF;
        }

        // Allocator buffers
        let state_alloc_buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let vertex_alloc_buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);

        // Initialize allocators
        let state_pool_size: u32 = 64 * 1024 * 1024; // 64MB
        let vertex_pool_size: u32 = 16 * 1024 * 1024; // 16MB

        unsafe {
            let state_alloc = state_alloc_buffer.contents() as *mut u32;
            *state_alloc = 0; // bump_pointer
            *state_alloc.add(1) = state_pool_size; // pool_size

            let vertex_alloc = vertex_alloc_buffer.contents() as *mut u32;
            *vertex_alloc = 0;
            *vertex_alloc.add(1) = vertex_pool_size;
        }

        // Unified state buffer
        let unified_state_buffer = device.new_buffer(
            state_pool_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Scheduler buffer
        let scheduler_buffer = device.new_buffer(
            std::mem::size_of::<SchedulerState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device: device.clone(),
            command_queue,
            launch_pipeline,
            close_pipeline,
            megakernel_pipeline,
            scheduler_pipeline,
            stats_pipeline,
            app_table_buffer,
            state_alloc_buffer,
            vertex_alloc_buffer,
            unified_state_buffer,
            scheduler_buffer,
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let function = library
            .get_function(name, None)
            .map_err(|e| format!("Function {} not found: {}", name, e))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline {} failed: {}", name, e))
    }

    /// Get pointer to app table header
    fn header(&self) -> *const AppTableHeader {
        self.app_table_buffer.contents() as *const AppTableHeader
    }

    /// Get pointer to app slots (after header)
    fn apps(&self) -> *const GpuAppDescriptor {
        unsafe {
            (self.app_table_buffer.contents() as *const u8)
                .add(std::mem::size_of::<AppTableHeader>())
                as *const GpuAppDescriptor
        }
    }

    /// Launch an app on GPU
    fn launch_app(&self, app_type: u32, state_size: u32, vertex_size: u32) -> u32 {
        let app_type_buf = self.device.new_buffer_with_data(
            &app_type as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let state_size_buf = self.device.new_buffer_with_data(
            &state_size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_size_buf = self.device.new_buffer_with_data(
            &vertex_size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );
        let result_buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // Initialize result to INVALID_SLOT
        unsafe {
            *(result_buf.contents() as *mut u32) = INVALID_SLOT;
        }

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.launch_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            std::mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.state_alloc_buffer), 0);
        enc.set_buffer(3, Some(&self.vertex_alloc_buffer), 0);
        enc.set_buffer(4, Some(&app_type_buf), 0);
        enc.set_buffer(5, Some(&state_size_buf), 0);
        enc.set_buffer(6, Some(&vertex_size_buf), 0);
        enc.set_buffer(7, Some(&result_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        unsafe { *(result_buf.contents() as *const u32) }
    }

    /// Close an app on GPU
    fn close_app(&self, slot_id: u32) {
        let slot_buf = self.device.new_buffer_with_data(
            &slot_id as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.close_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            std::mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&slot_buf), 0);

        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Run megakernel for one frame
    fn run_frame(&self, frame_number: u32) {
        let frame_buf = self.device.new_buffer_with_data(
            &frame_number as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.command_queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.megakernel_pipeline);
        enc.set_buffer(0, Some(&self.app_table_buffer), 0);
        enc.set_buffer(
            1,
            Some(&self.app_table_buffer),
            std::mem::size_of::<AppTableHeader>() as u64,
        );
        enc.set_buffer(2, Some(&self.unified_state_buffer), 0);
        enc.set_buffer(3, Some(&frame_buf), 0);

        let max_slots = unsafe { (*(self.header())).max_slots };
        enc.dispatch_threads(
            MTLSize::new(max_slots as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Get active app count
    fn active_count(&self) -> u32 {
        unsafe { (*(self.header())).active_count }
    }

    /// Get app descriptor
    fn get_app(&self, slot: u32) -> GpuAppDescriptor {
        unsafe { *(self.apps().add(slot as usize)) }
    }

    /// Read counter from app state
    fn read_app_counter(&self, slot: u32) -> u32 {
        let app = self.get_app(slot);
        unsafe {
            let ptr = (self.unified_state_buffer.contents() as *const u8)
                .add(app.state_offset as usize) as *const u32;
            *ptr
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_struct_sizes() {
    assert_eq!(std::mem::size_of::<GpuAppDescriptor>(), 128);
    // AppTableHeader: actual size may include alignment padding
    let header_size = std::mem::size_of::<AppTableHeader>();
    assert!(header_size >= 32, "AppTableHeader too small: {} bytes", header_size);
    // SchedulerState: actual size may include alignment padding
    let scheduler_size = std::mem::size_of::<SchedulerState>();
    assert!(scheduler_size >= 32, "SchedulerState too small: {} bytes", scheduler_size);
}

#[test]
fn test_gpu_launch_single_app() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    assert_eq!(system.active_count(), 0);

    // Launch an app
    let slot = system.launch_app(app_type::CUSTOM, 4096, 1024);
    assert_ne!(slot, INVALID_SLOT, "Launch should succeed");
    assert!(slot < 64, "Slot should be valid");

    // Verify active count
    assert_eq!(system.active_count(), 1);

    // Verify descriptor
    let app = system.get_app(slot);
    assert_eq!(app.flags & flags::ACTIVE, flags::ACTIVE);
    assert_eq!(app.app_type, app_type::CUSTOM);
    assert_eq!(app.slot_id, slot);
    assert_eq!(app.state_size, 4096);
    assert_eq!(app.vertex_size, 1024);
}

#[test]
fn test_gpu_launch_multiple_apps() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    // Launch 10 apps
    let mut slots = Vec::new();
    for i in 0..10 {
        let slot = system.launch_app(app_type::CUSTOM, 1024, 512);
        assert_ne!(slot, INVALID_SLOT, "Launch {} should succeed", i);
        slots.push(slot);
    }

    assert_eq!(system.active_count(), 10);

    // All slots should be unique
    slots.sort();
    slots.dedup();
    assert_eq!(slots.len(), 10, "All slots should be unique");
}

#[test]
fn test_gpu_close_app() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    // Launch and close
    let slot = system.launch_app(app_type::CUSTOM, 4096, 1024);
    assert_eq!(system.active_count(), 1);

    system.close_app(slot);
    assert_eq!(system.active_count(), 0);

    // Descriptor should be cleared
    let app = system.get_app(slot);
    assert_eq!(app.flags, 0);
}

#[test]
fn test_slot_reuse_after_close() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    // Launch, close, launch again
    let slot1 = system.launch_app(app_type::CUSTOM, 1024, 512);
    system.close_app(slot1);
    let slot2 = system.launch_app(app_type::CUSTOM, 1024, 512);

    // Slot should be reused
    assert_eq!(slot1, slot2, "Closed slot should be reused");
}

#[test]
fn test_megakernel_runs_apps() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    // Launch app with state containing a counter
    let slot = system.launch_app(app_type::CUSTOM, 64, 64);

    // Counter should be 0 initially
    assert_eq!(system.read_app_counter(slot), 0);

    // Run frame
    system.run_frame(1);

    // Counter should be incremented by test_app_update
    assert_eq!(system.read_app_counter(slot), 1);

    // App should no longer be dirty
    let app = system.get_app(slot);
    assert_eq!(app.flags & flags::DIRTY, 0, "DIRTY flag should be cleared");
    assert_eq!(app.last_run_frame, 1);
}

#[test]
fn test_multiple_frames() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    let slot = system.launch_app(app_type::CUSTOM, 64, 64);

    // Run 100 frames
    for frame in 1..=100 {
        // Mark dirty before each frame
        unsafe {
            let app = (system.apps() as *mut GpuAppDescriptor).add(slot as usize);
            (*app).flags |= flags::DIRTY;
        }
        system.run_frame(frame);
    }

    // Counter should be 100
    assert_eq!(system.read_app_counter(slot), 100);

    let app = system.get_app(slot);
    assert_eq!(app.last_run_frame, 100);
}

#[test]
fn test_launch_to_capacity() {
    let device = Device::system_default().expect("No Metal device");
    let max_slots = 64;
    let system = GpuAppSystemTest::new(&device, max_slots).expect("Failed to create test system");

    // Fill all slots
    for i in 0..max_slots {
        let slot = system.launch_app(app_type::CUSTOM, 128, 64);
        assert_ne!(slot, INVALID_SLOT, "Launch {} should succeed", i);
    }

    assert_eq!(system.active_count(), max_slots);

    // Next launch should fail
    let slot = system.launch_app(app_type::CUSTOM, 128, 64);
    assert_eq!(slot, INVALID_SLOT, "Launch should fail when full");
}

#[test]
fn test_memory_isolation() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    // Launch two apps
    let slot1 = system.launch_app(app_type::CUSTOM, 256, 64);
    let slot2 = system.launch_app(app_type::CUSTOM, 256, 64);

    // Run frames - both should increment independently
    for frame in 1..=5 {
        unsafe {
            let apps = system.apps() as *mut GpuAppDescriptor;
            (*apps.add(slot1 as usize)).flags |= flags::DIRTY;
            (*apps.add(slot2 as usize)).flags |= flags::DIRTY;
        }
        system.run_frame(frame);
    }

    // Both counters should be 5
    assert_eq!(system.read_app_counter(slot1), 5);
    assert_eq!(system.read_app_counter(slot2), 5);

    // Memory offsets should be different
    let app1 = system.get_app(slot1);
    let app2 = system.get_app(slot2);
    assert_ne!(app1.state_offset, app2.state_offset, "State memory should not overlap");
}

#[test]
fn test_app_lifecycle_stress() {
    let device = Device::system_default().expect("No Metal device");
    let system = GpuAppSystemTest::new(&device, 64).expect("Failed to create test system");

    // Launch, run, close, repeat many times
    for iteration in 0..50 {
        // Launch 5 apps
        let mut slots = Vec::new();
        for _ in 0..5 {
            let slot = system.launch_app(app_type::CUSTOM, 512, 128);
            assert_ne!(slot, INVALID_SLOT, "Launch should succeed in iteration {}", iteration);
            slots.push(slot);
        }

        // Run a few frames
        for frame in 1..=3 {
            for &slot in &slots {
                unsafe {
                    let apps = system.apps() as *mut GpuAppDescriptor;
                    (*apps.add(slot as usize)).flags |= flags::DIRTY;
                }
            }
            system.run_frame(iteration * 10 + frame);
        }

        // Close all
        for slot in slots {
            system.close_app(slot);
        }

        assert_eq!(system.active_count(), 0, "All apps should be closed");
    }
}
