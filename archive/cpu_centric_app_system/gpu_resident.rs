//! GPU-Resident App System Integration (Issue #153)
//!
//! Integrates the GPU-resident app lifecycle management with the existing
//! desktop environment. Provides a gradual migration path from CPU-centric
//! to GPU-resident app management.
//!
//! # Migration Strategy
//!
//! Phase 1: Parallel Systems (Current)
//! - Both legacy AppRegistry and GPU-resident system active
//! - New dynamic apps use GPU-resident system
//! - Built-in apps continue to use legacy system
//!
//! Phase 2: Hybrid Dispatch
//! - GPU queues dispatch requests for GPU-resident apps
//! - CPU processes dispatch list and runs legacy apps
//!
//! Phase 3: Full Migration
//! - All apps converted to GPU-resident format
//! - Legacy AppRegistry removed

use crate::gpu_os::app_descriptor::{flags as app_flags, GpuAppDescriptor, INVALID_SLOT};
use crate::gpu_os::app_lifecycle::{AppLaunchRequest, AppLaunchResult, GpuAppLifecycle};
use crate::gpu_os::pipeline_table::{dispatch_flags, DispatchRequest, INVALID_HANDLE};
use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, RenderPipelineState};

/// Default memory pool size (64MB)
pub const DEFAULT_MEMORY_POOL_SIZE: usize = 64 * 1024 * 1024;

/// Default max app slots
pub const DEFAULT_MAX_APP_SLOTS: u32 = 64;

/// Feature flag for GPU-resident apps
pub static USE_GPU_RESIDENT_APPS: bool = true;

/// GPU-Resident App System
///
/// Wraps GpuAppLifecycle and provides integration with the desktop.
pub struct GpuResidentAppSystem {
    lifecycle: GpuAppLifecycle,
    enabled: bool,
}

impl GpuResidentAppSystem {
    /// Create a new GPU-resident app system
    pub fn new(device: &Device) -> Self {
        Self {
            lifecycle: GpuAppLifecycle::new(device, DEFAULT_MAX_APP_SLOTS, DEFAULT_MEMORY_POOL_SIZE),
            enabled: USE_GPU_RESIDENT_APPS,
        }
    }

    /// Create with custom settings
    pub fn with_settings(device: &Device, max_slots: u32, memory_size: usize) -> Self {
        Self {
            lifecycle: GpuAppLifecycle::new(device, max_slots, memory_size),
            enabled: USE_GPU_RESIDENT_APPS,
        }
    }

    /// Check if GPU-resident apps are enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable the system
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get the lifecycle manager
    pub fn lifecycle(&self) -> &GpuAppLifecycle {
        &self.lifecycle
    }

    /// Get mutable lifecycle manager
    pub fn lifecycle_mut(&mut self) -> &mut GpuAppLifecycle {
        &mut self.lifecycle
    }

    /// Get slot pool buffer for GPU binding
    pub fn slot_pool_buffer(&self) -> &Buffer {
        self.lifecycle.slot_pool_buffer()
    }

    /// Get memory pool buffer for GPU binding
    pub fn memory_pool_buffer(&self) -> &Buffer {
        self.lifecycle.memory_pool_buffer()
    }

    /// Get dispatch list buffer for GPU binding
    pub fn dispatch_list_buffer(&self) -> &Buffer {
        self.lifecycle.dispatch_list_buffer()
    }

    /// Get cleanup queue buffer for GPU binding
    pub fn cleanup_queue_buffer(&self) -> &Buffer {
        self.lifecycle.cleanup_queue_buffer()
    }

    /// Launch a GPU-resident app
    pub fn launch_app(
        &mut self,
        app_type: u32,
        window_id: u32,
        state_size: usize,
        vertex_size: usize,
        param_size: usize,
        thread_count: u32,
        vertex_count: u32,
        clear_color: [f32; 4],
        preferred_size: [f32; 2],
        compute_pipeline: ComputePipelineState,
        render_pipeline: RenderPipelineState,
    ) -> Result<u32, String> {
        if !self.enabled {
            return Err("GPU-resident apps not enabled".to_string());
        }

        let request = AppLaunchRequest {
            app_type,
            window_id,
            state_size,
            vertex_size,
            param_size,
            thread_count,
            vertex_count,
            clear_color,
            preferred_size,
            compute_pipeline,
            render_pipeline,
        };

        let result = self.lifecycle.launch(request)?;
        Ok(result.slot)
    }

    /// Close a GPU-resident app
    pub fn close_app(&mut self, slot: u32) -> Result<(), String> {
        self.lifecycle.close_cpu(slot)
    }

    /// Process dispatch requests from GPU
    ///
    /// Returns dispatch requests that the CPU should process.
    /// After processing, call `reset_dispatch_list()`.
    pub fn get_pending_dispatches(&self) -> Vec<DispatchRequest> {
        self.lifecycle.get_dispatch_requests()
    }

    /// Reset the dispatch list after processing
    pub fn reset_dispatch_list(&self) {
        self.lifecycle.reset_dispatch_list();
    }

    /// Process deferred cleanup queue
    ///
    /// Should be called periodically (e.g., once per second or after batch closes)
    pub fn process_cleanup(&mut self) -> usize {
        self.lifecycle.process_cleanup()
    }

    /// Get number of active GPU-resident apps
    pub fn active_count(&self) -> u32 {
        self.lifecycle.active_app_count()
    }

    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f32 {
        self.lifecycle.memory_utilization()
    }

    /// Get all active app descriptors
    pub fn active_apps(&self) -> Vec<(u32, GpuAppDescriptor)> {
        self.lifecycle.active_apps()
    }

    /// Get app descriptor by slot
    pub fn get_app(&self, slot: u32) -> Option<GpuAppDescriptor> {
        self.lifecycle.get_slot(slot)
    }

    /// Update app descriptor
    pub fn update_app(&self, slot: u32, desc: &GpuAppDescriptor) {
        self.lifecycle.update_slot(slot, desc);
    }

    /// Mark app as needing redraw
    pub fn mark_dirty(&self, slot: u32) {
        self.lifecycle.mark_dirty(slot);
    }

    /// Clear dirty flag
    pub fn clear_dirty(&self, slot: u32) {
        self.lifecycle.clear_dirty(slot);
    }

    /// Find app by window ID
    pub fn find_by_window(&self, window_id: u32) -> Option<(u32, GpuAppDescriptor)> {
        self.active_apps()
            .into_iter()
            .find(|(_, desc)| desc.window_id == window_id)
    }

    /// Close app by window ID
    pub fn close_by_window(&mut self, window_id: u32) -> Result<(), String> {
        if let Some((slot, _)) = self.find_by_window(window_id) {
            self.close_app(slot)
        } else {
            Err(format!("No app found for window {}", window_id))
        }
    }
}

/// Statistics for monitoring the GPU-resident system
#[derive(Clone, Debug, Default)]
pub struct GpuResidentStats {
    pub active_apps: u32,
    pub memory_used_bytes: u32,
    pub memory_total_bytes: u32,
    pub memory_utilization: f32,
    pub dispatch_requests: u32,
    pub cleanup_pending: u32,
}

impl GpuResidentAppSystem {
    /// Get system statistics
    pub fn stats(&self) -> GpuResidentStats {
        let dispatch_requests = self.lifecycle.get_dispatch_requests().len() as u32;

        GpuResidentStats {
            active_apps: self.active_count(),
            memory_used_bytes: 0, // TODO: expose from lifecycle
            memory_total_bytes: DEFAULT_MEMORY_POOL_SIZE as u32,
            memory_utilization: self.memory_utilization(),
            dispatch_requests,
            cleanup_pending: 0, // TODO: expose from lifecycle
        }
    }
}

/// Metal shader header for GPU-resident system integration
pub const GPU_RESIDENT_METAL_HEADER: &str = r#"
// GPU-Resident App System Integration
// Combines all headers for event loop access

// Include order:
// 1. app_descriptor.metal (GpuAppDescriptor, flags)
// 2. slot_allocator.metal (slot_allocate, slot_free)
// 3. gpu_memory_pool.metal (memory_pool_allocate, memory_pool_free)
// 4. pipeline_table.metal (DispatchRequest, queue_dispatch, queue_cleanup)
// 5. app_lifecycle.metal (gpu_close_app, gpu_queue_app_dispatch)

// Event loop integration - handle window close
void handle_window_close_gpu_resident(
    device SlotPoolHeader* slot_header,
    device GpuAppDescriptor* slots,
    device MemoryPoolHeader* memory_pool,
    device atomic_uint* memory_bitmap,
    device CleanupQueueHeader* cleanup_header,
    device DeferredCleanup* cleanup_items,
    device GpuWindow* windows,
    uint window_index
) {
    // Get app slot from window
    uint app_slot = windows[window_index].app_slot;
    if (app_slot == INVALID_SLOT) return;

    // Close the app (frees slot, memory, queues pipeline cleanup)
    gpu_close_app(
        slot_header, slots,
        memory_pool, memory_bitmap,
        cleanup_header, cleanup_items,
        app_slot
    );

    // Clear window's app reference
    windows[window_index].app_slot = INVALID_SLOT;
    windows[window_index].flags &= ~WINDOW_FLAG_VISIBLE;
}

// Build dispatch list for all active apps
void build_dispatch_list(
    device const GpuAppDescriptor* slots,
    device DispatchListHeader* dispatch_header,
    device DispatchRequest* dispatch_requests,
    uint max_slots,
    uint tid
) {
    if (tid >= max_slots) return;

    const GpuAppDescriptor* desc = &slots[tid];

    // Only dispatch active, visible, dirty apps
    if ((desc->flags & APP_FLAG_ACTIVE) == 0) return;
    if ((desc->flags & APP_FLAG_VISIBLE) == 0) return;
    if ((desc->flags & APP_FLAG_DIRTY) == 0) return;

    // Queue for dispatch
    queue_dispatch(
        dispatch_header, dispatch_requests,
        tid, desc->thread_count,
        DISPATCH_FLAG_COMPUTE | DISPATCH_FLAG_RENDER
    );
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_system() -> Option<GpuResidentAppSystem> {
        Device::system_default().map(|device| GpuResidentAppSystem::new(&device))
    }

    #[test]
    fn test_system_creation() {
        if let Some(system) = create_test_system() {
            assert!(system.is_enabled());
            assert_eq!(system.active_count(), 0);
            assert_eq!(system.memory_utilization(), 0.0);
        }
    }

    #[test]
    fn test_system_enable_disable() {
        if let Some(mut system) = create_test_system() {
            assert!(system.is_enabled());

            system.set_enabled(false);
            assert!(!system.is_enabled());

            system.set_enabled(true);
            assert!(system.is_enabled());
        }
    }

    #[test]
    fn test_dispatch_list_empty() {
        if let Some(system) = create_test_system() {
            let dispatches = system.get_pending_dispatches();
            assert!(dispatches.is_empty());
        }
    }

    #[test]
    fn test_find_by_window_empty() {
        if let Some(system) = create_test_system() {
            assert!(system.find_by_window(100).is_none());
        }
    }

    #[test]
    fn test_close_nonexistent_window() {
        if let Some(mut system) = create_test_system() {
            assert!(system.close_by_window(999).is_err());
        }
    }

    #[test]
    fn test_stats() {
        if let Some(system) = create_test_system() {
            let stats = system.stats();
            assert_eq!(stats.active_apps, 0);
            assert_eq!(stats.memory_utilization, 0.0);
            assert_eq!(stats.dispatch_requests, 0);
        }
    }

    #[test]
    fn test_buffer_access() {
        if let Some(system) = create_test_system() {
            // Verify buffers are accessible
            let _ = system.slot_pool_buffer();
            let _ = system.memory_pool_buffer();
            let _ = system.dispatch_list_buffer();
            let _ = system.cleanup_queue_buffer();
        }
    }
}
