//! GPU Pipeline Handle Table (Issue #152)
//!
//! CPU-side storage for Metal pipelines, referenced by GPU via handle IDs.
//! GPU queues dispatch requests, CPU looks up pipelines and encodes commands.

use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, RenderPipelineState};
use std::sync::atomic::{AtomicU32, Ordering};

/// Invalid handle marker
pub const INVALID_HANDLE: u32 = 0xFFFFFFFF;

/// Maximum pipelines
pub const MAX_COMPUTE_PIPELINES: usize = 256;
pub const MAX_RENDER_PIPELINES: usize = 256;
pub const MAX_DISPATCH_REQUESTS: usize = 64;

/// Dispatch request from GPU
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchRequest {
    pub app_slot: u32,
    pub thread_count: u32,
    pub flags: u32,
    pub _padding: u32,
}

const _: () = assert!(std::mem::size_of::<DispatchRequest>() == 16);

/// Dispatch request flags
pub mod dispatch_flags {
    pub const COMPUTE: u32 = 1 << 0; // Run compute shader
    pub const RENDER: u32 = 1 << 1;  // Run render pass
}

/// Dispatch list header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchListHeader {
    pub count: u32, // Atomic on GPU
    pub _padding: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<DispatchListHeader>() == 16);

/// GPU Dispatch List - GPU writes dispatch requests, CPU processes them
pub struct GpuDispatchList {
    buffer: Buffer,
    max_requests: usize,
}

impl GpuDispatchList {
    /// Create a new dispatch list
    pub fn new(device: &Device, max_requests: usize) -> Self {
        let max_requests = max_requests.min(MAX_DISPATCH_REQUESTS);
        let header_size = std::mem::size_of::<DispatchListHeader>();
        let requests_size = max_requests * std::mem::size_of::<DispatchRequest>();
        let total_size = header_size + requests_size;

        let buffer = device.new_buffer(
            total_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize
        unsafe {
            let ptr = buffer.contents() as *mut u8;
            let header = ptr as *mut DispatchListHeader;
            (*header).count = 0;
            (*header)._padding = [0; 3];

            // Clear requests
            let requests = ptr.add(header_size) as *mut DispatchRequest;
            for i in 0..max_requests {
                *requests.add(i) = DispatchRequest::default();
            }
        }

        Self {
            buffer,
            max_requests,
        }
    }

    /// Get the underlying Metal buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get current request count
    pub fn count(&self) -> u32 {
        unsafe {
            let header = self.buffer.contents() as *const DispatchListHeader;
            (*header).count
        }
    }

    /// Reset the dispatch list (call after processing)
    pub fn reset(&self) {
        unsafe {
            let header = self.buffer.contents() as *mut DispatchListHeader;
            (*header).count = 0;
        }
    }

    /// Get all pending dispatch requests
    pub fn requests(&self) -> Vec<DispatchRequest> {
        let count = self.count().min(self.max_requests as u32);
        let mut result = Vec::with_capacity(count as usize);

        unsafe {
            let ptr = self.buffer.contents() as *const u8;
            let header_size = std::mem::size_of::<DispatchListHeader>();
            let requests = ptr.add(header_size) as *const DispatchRequest;

            for i in 0..count {
                result.push(*requests.add(i as usize));
            }
        }

        result
    }

    /// CPU-side: add a dispatch request (for testing)
    pub fn add_request(&self, request: DispatchRequest) -> bool {
        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let header = ptr as *mut DispatchListHeader;
            let count = (*header).count as usize;

            if count >= self.max_requests {
                return false;
            }

            let header_size = std::mem::size_of::<DispatchListHeader>();
            let requests = ptr.add(header_size) as *mut DispatchRequest;
            *requests.add(count) = request;
            (*header).count += 1;

            true
        }
    }
}

/// Pipeline Table - stores compiled Metal pipelines, referenced by handle
pub struct PipelineTable {
    compute: Vec<Option<ComputePipelineState>>,
    render: Vec<Option<RenderPipelineState>>,
    free_compute: Vec<u32>,
    free_render: Vec<u32>,
}

impl PipelineTable {
    /// Create a new pipeline table
    pub fn new() -> Self {
        Self {
            compute: Vec::with_capacity(MAX_COMPUTE_PIPELINES),
            render: Vec::with_capacity(MAX_RENDER_PIPELINES),
            free_compute: Vec::new(),
            free_render: Vec::new(),
        }
    }

    /// Add a compute pipeline, returns handle
    pub fn add_compute(&mut self, pipeline: ComputePipelineState) -> u32 {
        if let Some(handle) = self.free_compute.pop() {
            self.compute[handle as usize] = Some(pipeline);
            handle
        } else {
            if self.compute.len() >= MAX_COMPUTE_PIPELINES {
                return INVALID_HANDLE;
            }
            let handle = self.compute.len() as u32;
            self.compute.push(Some(pipeline));
            handle
        }
    }

    /// Add a render pipeline, returns handle
    pub fn add_render(&mut self, pipeline: RenderPipelineState) -> u32 {
        if let Some(handle) = self.free_render.pop() {
            self.render[handle as usize] = Some(pipeline);
            handle
        } else {
            if self.render.len() >= MAX_RENDER_PIPELINES {
                return INVALID_HANDLE;
            }
            let handle = self.render.len() as u32;
            self.render.push(Some(pipeline));
            handle
        }
    }

    /// Remove a compute pipeline
    pub fn remove_compute(&mut self, handle: u32) {
        if (handle as usize) < self.compute.len() {
            self.compute[handle as usize] = None;
            self.free_compute.push(handle);
        }
    }

    /// Remove a render pipeline
    pub fn remove_render(&mut self, handle: u32) {
        if (handle as usize) < self.render.len() {
            self.render[handle as usize] = None;
            self.free_render.push(handle);
        }
    }

    /// Get a compute pipeline by handle
    pub fn get_compute(&self, handle: u32) -> Option<&ComputePipelineState> {
        self.compute.get(handle as usize).and_then(|o| o.as_ref())
    }

    /// Get a render pipeline by handle
    pub fn get_render(&self, handle: u32) -> Option<&RenderPipelineState> {
        self.render.get(handle as usize).and_then(|o| o.as_ref())
    }

    /// Get number of active compute pipelines
    pub fn compute_count(&self) -> usize {
        self.compute.iter().filter(|o| o.is_some()).count()
    }

    /// Get number of active render pipelines
    pub fn render_count(&self) -> usize {
        self.render.iter().filter(|o| o.is_some()).count()
    }

    /// Check if a compute handle is valid
    pub fn is_valid_compute(&self, handle: u32) -> bool {
        self.get_compute(handle).is_some()
    }

    /// Check if a render handle is valid
    pub fn is_valid_render(&self, handle: u32) -> bool {
        self.get_render(handle).is_some()
    }
}

impl Default for PipelineTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Deferred cleanup item - GPU marks resources for CPU cleanup
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeferredCleanup {
    pub resource_type: u32,
    pub resource_id: u32,
    pub flags: u32,
    pub _padding: u32,
}

/// Resource types for deferred cleanup
pub mod resource_type {
    pub const COMPUTE_PIPELINE: u32 = 1;
    pub const RENDER_PIPELINE: u32 = 2;
    pub const BUFFER: u32 = 3;
    pub const TEXTURE: u32 = 4;
}

/// Deferred cleanup queue header
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CleanupQueueHeader {
    pub count: u32, // Atomic on GPU
    pub _padding: [u32; 3],
}

/// Maximum cleanup items
pub const MAX_CLEANUP_ITEMS: usize = 64;

/// Deferred cleanup queue
pub struct DeferredCleanupQueue {
    buffer: Buffer,
    max_items: usize,
}

impl DeferredCleanupQueue {
    /// Create a new cleanup queue
    pub fn new(device: &Device, max_items: usize) -> Self {
        let max_items = max_items.min(MAX_CLEANUP_ITEMS);
        let header_size = std::mem::size_of::<CleanupQueueHeader>();
        let items_size = max_items * std::mem::size_of::<DeferredCleanup>();
        let total_size = header_size + items_size;

        let buffer = device.new_buffer(
            total_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize
        unsafe {
            let ptr = buffer.contents() as *mut u8;
            let header = ptr as *mut CleanupQueueHeader;
            (*header).count = 0;
            (*header)._padding = [0; 3];
        }

        Self { buffer, max_items }
    }

    /// Get the underlying Metal buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get current item count
    pub fn count(&self) -> u32 {
        unsafe {
            let header = self.buffer.contents() as *const CleanupQueueHeader;
            (*header).count
        }
    }

    /// Get all pending cleanup items
    pub fn items(&self) -> Vec<DeferredCleanup> {
        let count = self.count().min(self.max_items as u32);
        let mut result = Vec::with_capacity(count as usize);

        unsafe {
            let ptr = self.buffer.contents() as *const u8;
            let header_size = std::mem::size_of::<CleanupQueueHeader>();
            let items = ptr.add(header_size) as *const DeferredCleanup;

            for i in 0..count {
                result.push(*items.add(i as usize));
            }
        }

        result
    }

    /// Reset the cleanup queue (call after processing)
    pub fn reset(&self) {
        unsafe {
            let header = self.buffer.contents() as *mut CleanupQueueHeader;
            (*header).count = 0;
        }
    }

    /// Process cleanup items and clear queue
    pub fn process(&self, pipeline_table: &mut PipelineTable) -> usize {
        let items = self.items();
        let count = items.len();

        for item in items {
            match item.resource_type {
                resource_type::COMPUTE_PIPELINE => {
                    pipeline_table.remove_compute(item.resource_id);
                }
                resource_type::RENDER_PIPELINE => {
                    pipeline_table.remove_render(item.resource_id);
                }
                // Buffer and texture cleanup handled elsewhere
                _ => {}
            }
        }

        self.reset();
        count
    }

    /// CPU-side: add a cleanup item (for testing)
    pub fn add_item(&self, item: DeferredCleanup) -> bool {
        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let header = ptr as *mut CleanupQueueHeader;
            let count = (*header).count as usize;

            if count >= self.max_items {
                return false;
            }

            let header_size = std::mem::size_of::<CleanupQueueHeader>();
            let items = ptr.add(header_size) as *mut DeferredCleanup;
            *items.add(count) = item;
            (*header).count += 1;

            true
        }
    }
}

/// Metal shader code for dispatch list operations
pub const DISPATCH_LIST_METAL_HEADER: &str = r#"
// Dispatch request
struct DispatchRequest {
    uint app_slot;
    uint thread_count;
    uint flags;
    uint _padding;
};

// Dispatch flags
#define DISPATCH_FLAG_COMPUTE (1u << 0)
#define DISPATCH_FLAG_RENDER  (1u << 1)

// Dispatch list header
struct DispatchListHeader {
    atomic_uint count;
    uint _padding[3];
};

// Queue a dispatch request
void queue_dispatch(
    device DispatchListHeader* header,
    device DispatchRequest* requests,
    uint app_slot,
    uint thread_count,
    uint flags
) {
    uint slot = atomic_fetch_add_explicit(&header->count, 1, memory_order_relaxed);
    if (slot < 64) {  // MAX_DISPATCH_REQUESTS
        requests[slot] = (DispatchRequest){
            app_slot,
            thread_count,
            flags,
            0
        };
    }
}

// Deferred cleanup item
struct DeferredCleanup {
    uint resource_type;
    uint resource_id;
    uint flags;
    uint _padding;
};

// Resource types
#define RESOURCE_COMPUTE_PIPELINE 1
#define RESOURCE_RENDER_PIPELINE  2
#define RESOURCE_BUFFER           3
#define RESOURCE_TEXTURE          4

// Cleanup queue header
struct CleanupQueueHeader {
    atomic_uint count;
    uint _padding[3];
};

// Queue a cleanup request
void queue_cleanup(
    device CleanupQueueHeader* header,
    device DeferredCleanup* items,
    uint resource_type,
    uint resource_id
) {
    uint slot = atomic_fetch_add_explicit(&header->count, 1, memory_order_relaxed);
    if (slot < 64) {  // MAX_CLEANUP_ITEMS
        items[slot] = (DeferredCleanup){
            resource_type,
            resource_id,
            0,
            0
        };
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Device> {
        Device::system_default()
    }

    // DispatchList tests

    #[test]
    fn test_dispatch_list_creation() {
        if let Some(device) = create_test_device() {
            let list = GpuDispatchList::new(&device, 32);
            assert_eq!(list.count(), 0);
        }
    }

    #[test]
    fn test_dispatch_list_add_request() {
        if let Some(device) = create_test_device() {
            let list = GpuDispatchList::new(&device, 32);

            let req = DispatchRequest {
                app_slot: 5,
                thread_count: 1024,
                flags: dispatch_flags::COMPUTE,
                _padding: 0,
            };

            assert!(list.add_request(req));
            assert_eq!(list.count(), 1);

            let requests = list.requests();
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].app_slot, 5);
            assert_eq!(requests[0].thread_count, 1024);
        }
    }

    #[test]
    fn test_dispatch_list_reset() {
        if let Some(device) = create_test_device() {
            let list = GpuDispatchList::new(&device, 32);

            for i in 0..5 {
                list.add_request(DispatchRequest {
                    app_slot: i,
                    thread_count: 100,
                    flags: 0,
                    _padding: 0,
                });
            }

            assert_eq!(list.count(), 5);
            list.reset();
            assert_eq!(list.count(), 0);
        }
    }

    #[test]
    fn test_dispatch_list_overflow() {
        if let Some(device) = create_test_device() {
            let list = GpuDispatchList::new(&device, 4);

            for i in 0..4 {
                assert!(list.add_request(DispatchRequest {
                    app_slot: i,
                    thread_count: 100,
                    flags: 0,
                    _padding: 0,
                }));
            }

            // Should fail - list full
            assert!(!list.add_request(DispatchRequest::default()));
        }
    }

    // PipelineTable tests (without actual pipelines)

    #[test]
    fn test_pipeline_table_creation() {
        let table = PipelineTable::new();
        assert_eq!(table.compute_count(), 0);
        assert_eq!(table.render_count(), 0);
    }

    #[test]
    fn test_pipeline_table_invalid_handle() {
        let table = PipelineTable::new();
        assert!(!table.is_valid_compute(0));
        assert!(!table.is_valid_compute(INVALID_HANDLE));
        assert!(!table.is_valid_render(0));
    }

    // DeferredCleanupQueue tests

    #[test]
    fn test_cleanup_queue_creation() {
        if let Some(device) = create_test_device() {
            let queue = DeferredCleanupQueue::new(&device, 32);
            assert_eq!(queue.count(), 0);
        }
    }

    #[test]
    fn test_cleanup_queue_add_item() {
        if let Some(device) = create_test_device() {
            let queue = DeferredCleanupQueue::new(&device, 32);

            let item = DeferredCleanup {
                resource_type: resource_type::COMPUTE_PIPELINE,
                resource_id: 5,
                flags: 0,
                _padding: 0,
            };

            assert!(queue.add_item(item));
            assert_eq!(queue.count(), 1);

            let items = queue.items();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].resource_type, resource_type::COMPUTE_PIPELINE);
            assert_eq!(items[0].resource_id, 5);
        }
    }

    #[test]
    fn test_cleanup_queue_reset() {
        if let Some(device) = create_test_device() {
            let queue = DeferredCleanupQueue::new(&device, 32);

            for i in 0..5 {
                queue.add_item(DeferredCleanup {
                    resource_type: resource_type::BUFFER,
                    resource_id: i,
                    flags: 0,
                    _padding: 0,
                });
            }

            assert_eq!(queue.count(), 5);
            queue.reset();
            assert_eq!(queue.count(), 0);
        }
    }

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<DispatchRequest>(), 16);
        assert_eq!(std::mem::size_of::<DispatchListHeader>(), 16);
        assert_eq!(std::mem::size_of::<DeferredCleanup>(), 16);
        assert_eq!(std::mem::size_of::<CleanupQueueHeader>(), 16);
    }
}
