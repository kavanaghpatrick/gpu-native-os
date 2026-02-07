//! GPU-Native App Lifecycle (Issue #151)
//!
//! Manages the complete lifecycle of GPU-resident apps:
//! - Launch: Allocate slot, allocate memory, initialize descriptor
//! - Run: GPU dispatches work based on dirty flags
//! - Close: Free slot, free memory, cleanup pipelines
//!
//! The goal is to minimize CPU involvement - GPU handles most operations.

use crate::gpu_os::app_descriptor::{flags, GpuAppDescriptor, INVALID_SLOT, MAX_APP_SLOTS};
use crate::gpu_os::gpu_memory_pool::{GpuAppMemoryPool, ALLOC_FAILED, BLOCK_SIZE};
use crate::gpu_os::pipeline_table::{
    DeferredCleanup, DeferredCleanupQueue, GpuDispatchList, PipelineTable,
    resource_type, INVALID_HANDLE,
};
use crate::gpu_os::slot_allocator::GpuAppSlotPool;
use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, RenderPipelineState};

/// App launch request - everything needed to start an app
pub struct AppLaunchRequest {
    pub app_type: u32,
    pub window_id: u32,
    pub state_size: usize,
    pub vertex_size: usize,
    pub param_size: usize,
    pub thread_count: u32,
    pub vertex_count: u32,
    pub clear_color: [f32; 4],
    pub preferred_size: [f32; 2],
    pub compute_pipeline: ComputePipelineState,
    pub render_pipeline: RenderPipelineState,
}

/// Result of a successful app launch
pub struct AppLaunchResult {
    pub slot: u32,
    pub state_offset: u32,
    pub vertex_offset: u32,
    pub param_offset: u32,
    pub compute_handle: u32,
    pub render_handle: u32,
}

/// App close request
pub struct AppCloseRequest {
    pub slot: u32,
}

/// GPU App Lifecycle Manager
///
/// Coordinates slot allocation, memory management, and pipeline storage
/// for GPU-resident apps.
pub struct GpuAppLifecycle {
    slot_pool: GpuAppSlotPool,
    memory_pool: GpuAppMemoryPool,
    pipeline_table: PipelineTable,
    dispatch_list: GpuDispatchList,
    cleanup_queue: DeferredCleanupQueue,
}

impl GpuAppLifecycle {
    /// Create a new lifecycle manager
    pub fn new(device: &Device, max_slots: u32, memory_size: usize) -> Self {
        Self {
            slot_pool: GpuAppSlotPool::new(device, max_slots),
            memory_pool: GpuAppMemoryPool::new(device, memory_size),
            pipeline_table: PipelineTable::new(),
            dispatch_list: GpuDispatchList::new(device, 64),
            cleanup_queue: DeferredCleanupQueue::new(device, 64),
        }
    }

    /// Get slot pool buffer (for GPU access)
    pub fn slot_pool_buffer(&self) -> &Buffer {
        self.slot_pool.buffer()
    }

    /// Get memory pool buffer (for GPU access)
    pub fn memory_pool_buffer(&self) -> &Buffer {
        self.memory_pool.buffer()
    }

    /// Get dispatch list buffer (for GPU access)
    pub fn dispatch_list_buffer(&self) -> &Buffer {
        self.dispatch_list.buffer()
    }

    /// Get cleanup queue buffer (for GPU access)
    pub fn cleanup_queue_buffer(&self) -> &Buffer {
        self.cleanup_queue.buffer()
    }

    /// Get pipeline table (for CPU dispatch)
    pub fn pipeline_table(&self) -> &PipelineTable {
        &self.pipeline_table
    }

    /// Get mutable pipeline table
    pub fn pipeline_table_mut(&mut self) -> &mut PipelineTable {
        &mut self.pipeline_table
    }

    /// Launch an app
    ///
    /// 1. Allocate slot
    /// 2. Allocate memory for buffers
    /// 3. Store pipelines
    /// 4. Initialize descriptor
    pub fn launch(&mut self, request: AppLaunchRequest) -> Result<AppLaunchResult, String> {
        // 1. Allocate slot
        let slot = self.slot_pool.allocate_cpu();
        if slot == INVALID_SLOT {
            return Err("No available app slots".to_string());
        }

        // 2. Allocate memory
        let state_offset = if request.state_size > 0 {
            let offset = self.memory_pool.allocate_cpu(request.state_size);
            if offset == ALLOC_FAILED {
                self.slot_pool.free_cpu(slot);
                return Err("Failed to allocate state buffer".to_string());
            }
            offset
        } else {
            0
        };

        let vertex_offset = if request.vertex_size > 0 {
            let offset = self.memory_pool.allocate_cpu(request.vertex_size);
            if offset == ALLOC_FAILED {
                if state_offset != 0 {
                    self.memory_pool.free_cpu(state_offset, request.state_size);
                }
                self.slot_pool.free_cpu(slot);
                return Err("Failed to allocate vertex buffer".to_string());
            }
            offset
        } else {
            0
        };

        let param_offset = if request.param_size > 0 {
            let offset = self.memory_pool.allocate_cpu(request.param_size);
            if offset == ALLOC_FAILED {
                if vertex_offset != 0 {
                    self.memory_pool.free_cpu(vertex_offset, request.vertex_size);
                }
                if state_offset != 0 {
                    self.memory_pool.free_cpu(state_offset, request.state_size);
                }
                self.slot_pool.free_cpu(slot);
                return Err("Failed to allocate param buffer".to_string());
            }
            offset
        } else {
            0
        };

        // 3. Store pipelines
        let compute_handle = self.pipeline_table.add_compute(request.compute_pipeline);
        if compute_handle == INVALID_HANDLE {
            self.free_memory(state_offset, request.state_size);
            self.free_memory(vertex_offset, request.vertex_size);
            self.free_memory(param_offset, request.param_size);
            self.slot_pool.free_cpu(slot);
            return Err("Failed to store compute pipeline".to_string());
        }

        let render_handle = self.pipeline_table.add_render(request.render_pipeline);
        if render_handle == INVALID_HANDLE {
            self.pipeline_table.remove_compute(compute_handle);
            self.free_memory(state_offset, request.state_size);
            self.free_memory(vertex_offset, request.vertex_size);
            self.free_memory(param_offset, request.param_size);
            self.slot_pool.free_cpu(slot);
            return Err("Failed to store render pipeline".to_string());
        }

        // 4. Initialize descriptor
        let mut desc = GpuAppDescriptor::default();
        desc.activate(slot, request.window_id, request.app_type);
        desc.state_offset = state_offset;
        desc.state_size = request.state_size as u32;
        desc.vertex_offset = vertex_offset;
        desc.vertex_size = request.vertex_size as u32;
        desc.param_offset = param_offset;
        desc.param_size = request.param_size as u32;
        desc.thread_count = request.thread_count;
        desc.vertex_count = request.vertex_count;
        desc.clear_color = request.clear_color;
        desc.preferred_size = request.preferred_size;
        desc.compute_pipeline_handle = compute_handle;
        desc.render_pipeline_handle = render_handle;

        self.slot_pool.update_slot(slot, &desc);

        Ok(AppLaunchResult {
            slot,
            state_offset,
            vertex_offset,
            param_offset,
            compute_handle,
            render_handle,
        })
    }

    /// Close an app (CPU-side)
    ///
    /// 1. Read descriptor to get buffer offsets and pipeline handles
    /// 2. Free memory
    /// 3. Remove pipelines
    /// 4. Free slot
    pub fn close_cpu(&mut self, slot: u32) -> Result<(), String> {
        if slot >= self.slot_pool.max_slots() {
            return Err("Invalid slot".to_string());
        }

        // 1. Get descriptor
        let desc = self.slot_pool.get_slot(slot).ok_or("Failed to read slot")?;

        if desc.flags & flags::ACTIVE == 0 {
            return Err("Slot not active".to_string());
        }

        // 2. Free memory
        if desc.state_size > 0 {
            self.memory_pool.free_cpu(desc.state_offset, desc.state_size as usize);
        }
        if desc.vertex_size > 0 {
            self.memory_pool.free_cpu(desc.vertex_offset, desc.vertex_size as usize);
        }
        if desc.param_size > 0 {
            self.memory_pool.free_cpu(desc.param_offset, desc.param_size as usize);
        }

        // 3. Remove pipelines
        if desc.compute_pipeline_handle != INVALID_SLOT {
            self.pipeline_table.remove_compute(desc.compute_pipeline_handle);
        }
        if desc.render_pipeline_handle != INVALID_SLOT {
            self.pipeline_table.remove_render(desc.render_pipeline_handle);
        }

        // 4. Free slot
        self.slot_pool.free_cpu(slot);

        Ok(())
    }

    /// Process deferred cleanup queue
    ///
    /// Called periodically by CPU to clean up resources marked by GPU.
    pub fn process_cleanup(&mut self) -> usize {
        self.cleanup_queue.process(&mut self.pipeline_table)
    }

    /// Get dispatch requests from GPU
    pub fn get_dispatch_requests(&self) -> Vec<crate::gpu_os::pipeline_table::DispatchRequest> {
        self.dispatch_list.requests()
    }

    /// Reset dispatch list (call after processing)
    pub fn reset_dispatch_list(&self) {
        self.dispatch_list.reset();
    }

    /// Get active app count
    pub fn active_app_count(&self) -> u32 {
        self.slot_pool.active_count()
    }

    /// Get memory utilization
    pub fn memory_utilization(&self) -> f32 {
        self.memory_pool.utilization()
    }

    /// Get all active app descriptors
    pub fn active_apps(&self) -> Vec<(u32, GpuAppDescriptor)> {
        self.slot_pool.active_slots()
    }

    /// Get a slot descriptor
    pub fn get_slot(&self, slot: u32) -> Option<GpuAppDescriptor> {
        self.slot_pool.get_slot(slot)
    }

    /// Update a slot descriptor
    pub fn update_slot(&self, slot: u32, desc: &GpuAppDescriptor) {
        self.slot_pool.update_slot(slot, desc);
    }

    /// Mark an app as dirty (needs redraw)
    pub fn mark_dirty(&self, slot: u32) {
        if let Some(ptr) = self.slot_pool.get_slot_mut(slot) {
            unsafe {
                (*ptr).flags |= flags::DIRTY;
            }
        }
    }

    /// Clear dirty flag
    pub fn clear_dirty(&self, slot: u32) {
        if let Some(ptr) = self.slot_pool.get_slot_mut(slot) {
            unsafe {
                (*ptr).flags &= !flags::DIRTY;
            }
        }
    }

    /// Helper to free memory if allocated
    fn free_memory(&self, offset: u32, size: usize) {
        if offset != 0 && size > 0 {
            self.memory_pool.free_cpu(offset, size);
        }
    }
}

/// Metal shader code combining all lifecycle operations
pub const APP_LIFECYCLE_METAL_HEADER: &str = r#"
// Include app descriptor, slot allocator, memory pool, and dispatch list headers
// (These are included separately)

// GPU-side app close
// Called when user clicks close button
void gpu_close_app(
    device SlotPoolHeader* slot_header,
    device GpuAppDescriptor* slots,
    device MemoryPoolHeader* memory_pool,
    device atomic_uint* memory_bitmap,
    device CleanupQueueHeader* cleanup_header,
    device DeferredCleanup* cleanup_items,
    uint slot
) {
    if (slot >= slot_header->max_slots) return;

    GpuAppDescriptor* desc = &slots[slot];
    if (!(desc->flags & APP_FLAG_ACTIVE)) return;

    // 1. Free memory allocations
    if (desc->state_size > 0) {
        memory_pool_free(memory_pool, memory_bitmap, desc->state_offset, desc->state_size);
    }
    if (desc->vertex_size > 0) {
        memory_pool_free(memory_pool, memory_bitmap, desc->vertex_offset, desc->vertex_size);
    }
    if (desc->param_size > 0) {
        memory_pool_free(memory_pool, memory_bitmap, desc->param_offset, desc->param_size);
    }

    // 2. Queue pipeline cleanup (CPU must release Metal objects)
    if (desc->compute_pipeline_handle != INVALID_SLOT) {
        queue_cleanup(cleanup_header, cleanup_items,
                      RESOURCE_COMPUTE_PIPELINE, desc->compute_pipeline_handle);
    }
    if (desc->render_pipeline_handle != INVALID_SLOT) {
        queue_cleanup(cleanup_header, cleanup_items,
                      RESOURCE_RENDER_PIPELINE, desc->render_pipeline_handle);
    }

    // 3. Free the slot
    slot_free(slot_header, slots, slot);
}

// GPU-side check if app needs render
bool gpu_app_needs_render(device const GpuAppDescriptor* slots, uint slot) {
    const GpuAppDescriptor* desc = &slots[slot];
    return (desc->flags & APP_FLAG_ACTIVE) != 0 &&
           (desc->flags & APP_FLAG_VISIBLE) != 0 &&
           (desc->flags & APP_FLAG_DIRTY) != 0;
}

// GPU-side queue app for dispatch
void gpu_queue_app_dispatch(
    device DispatchListHeader* dispatch_header,
    device DispatchRequest* dispatch_requests,
    device const GpuAppDescriptor* slots,
    uint slot
) {
    const GpuAppDescriptor* desc = &slots[slot];
    if (!gpu_app_needs_render(slots, slot)) return;

    queue_dispatch(dispatch_header, dispatch_requests,
                   slot, desc->thread_count, DISPATCH_FLAG_COMPUTE | DISPATCH_FLAG_RENDER);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_lifecycle() -> Option<GpuAppLifecycle> {
        Device::system_default().map(|device| {
            GpuAppLifecycle::new(&device, 64, 1024 * 1024) // 64 slots, 1MB memory
        })
    }

    // Note: Full launch/close tests require actual Metal pipelines
    // These tests verify the data structure operations

    #[test]
    fn test_lifecycle_creation() {
        if let Some(lifecycle) = create_test_lifecycle() {
            assert_eq!(lifecycle.active_app_count(), 0);
            assert_eq!(lifecycle.memory_utilization(), 0.0);
        }
    }

    #[test]
    fn test_close_invalid_slot() {
        if let Some(mut lifecycle) = create_test_lifecycle() {
            // Should fail gracefully
            assert!(lifecycle.close_cpu(1000).is_err());
        }
    }

    #[test]
    fn test_close_inactive_slot() {
        if let Some(mut lifecycle) = create_test_lifecycle() {
            // Slot 0 is not active
            assert!(lifecycle.close_cpu(0).is_err());
        }
    }

    #[test]
    fn test_dispatch_list_access() {
        if let Some(lifecycle) = create_test_lifecycle() {
            let requests = lifecycle.get_dispatch_requests();
            assert!(requests.is_empty());
        }
    }

    #[test]
    fn test_cleanup_processing() {
        if let Some(mut lifecycle) = create_test_lifecycle() {
            let count = lifecycle.process_cleanup();
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn test_mark_dirty() {
        if let Some(lifecycle) = create_test_lifecycle() {
            // Allocate a slot
            let slot = lifecycle.slot_pool.allocate_cpu();
            assert_ne!(slot, INVALID_SLOT);

            // Set up as active
            let mut desc = GpuAppDescriptor::default();
            desc.flags = flags::ACTIVE;
            lifecycle.update_slot(slot, &desc);

            // Mark dirty
            lifecycle.mark_dirty(slot);

            let updated = lifecycle.get_slot(slot).unwrap();
            assert!(updated.flags & flags::DIRTY != 0);

            // Clear dirty
            lifecycle.clear_dirty(slot);
            let updated2 = lifecycle.get_slot(slot).unwrap();
            assert!(updated2.flags & flags::DIRTY == 0);
        }
    }

    #[test]
    fn test_active_apps() {
        if let Some(lifecycle) = create_test_lifecycle() {
            // Initially empty
            assert!(lifecycle.active_apps().is_empty());

            // Allocate and activate a slot
            let slot = lifecycle.slot_pool.allocate_cpu();
            let mut desc = GpuAppDescriptor::default();
            desc.flags = flags::ACTIVE;
            desc.window_id = 100;
            lifecycle.update_slot(slot, &desc);

            let active = lifecycle.active_apps();
            assert_eq!(active.len(), 1);
            assert_eq!(active[0].1.window_id, 100);
        }
    }
}
