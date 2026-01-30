//! GPU Heap - Issue #167
//!
//! GPU-native memory allocator with slab allocation and lock-free free lists.
//!
//! THE GPU IS THE COMPUTER. The GPU owns all heap state.
//! Rust only allocates the buffer and binds it to compute encoders.
//!
//! # Design
//!
//! - Slab allocation with 8 size classes (64B to 64KB)
//! - Lock-free free lists via atomic CAS
//! - Bump allocator for new blocks
//! - Batch operations (N threads allocate N blocks)
//!
//! # Usage
//!
//! ```ignore
//! let heap = GpuHeap::new(&device, 64 * 1024 * 1024); // 64MB
//! heap.bind_to_encoder(&encoder, 0, 1);
//! // GPU kernels can now allocate/free via atomic operations
//! ```

use metal::*;
use std::mem;

// ═══════════════════════════════════════════════════════════════════════════
// Constants (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════

pub const SIZE_CLASS_COUNT: usize = 8;
pub const SIZE_CLASSES: [u32; 8] = [64, 128, 256, 512, 1024, 4096, 16384, 65536];

pub const INVALID_OFFSET: u32 = 0xFFFFFFFF;

// ═══════════════════════════════════════════════════════════════════════════
// Heap Structures (must match Metal shader)
// ═══════════════════════════════════════════════════════════════════════════

/// Heap header stored at start of heap buffer
/// Size: 64 bytes (cache-line aligned)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HeapHeader {
    /// Per-size-class free list heads
    pub free_list_heads: [u32; SIZE_CLASS_COUNT],
    /// Per-size-class counts (stats)
    pub free_list_counts: [u32; SIZE_CLASS_COUNT],
    /// Bump allocator pointer
    pub bump_ptr: u32,
    /// Total heap size
    pub heap_size: u32,
    /// Total bytes allocated
    pub total_allocated: u32,
    /// Number of allocations
    pub allocation_count: u32,
    /// Padding to 64 bytes
    pub _padding: [u32; 4],
}

/// Block header stored at start of each allocated block
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockHeader {
    pub size_class: u32,
    pub next_free: u32,
    pub flags: u32,
    pub _padding: u32,
}

/// Vector header for GpuVector
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuVectorHeader {
    pub len: u32,
    pub capacity: u32,
    pub element_size: u32,
    pub data_offset: u32,
    pub pending_pushes: u32,
    pub _padding: [u32; 3],
}

/// Cuckoo hash entry
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CuckooEntry {
    pub state: u32,
    pub key: u32,
    pub value: u32,
    pub _padding: u32,
}

/// HashMap header for GpuHashMap
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuHashMapHeader {
    pub count: u32,
    pub capacity: u32,
    pub table1_offset: u32,
    pub table2_offset: u32,
    pub insert_failures: u32,
    pub _padding: [u32; 3],
}

/// GPU String with SSO
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuString {
    pub data: [u8; 24],
    pub len: u8,
    pub flags: u8,
    pub _pad: [u8; 2],
    pub hash: u32,
}

impl Default for GpuString {
    fn default() -> Self {
        Self {
            data: [0; 24],
            len: 0,
            flags: 0x80, // SSO flag
            _pad: [0; 2],
            hash: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Heap
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-native heap allocator
///
/// Rust side only holds buffer references. GPU owns all state.
pub struct GpuHeap {
    /// Main heap buffer (StorageModePrivate for GPU-only access)
    buffer: Buffer,
    /// Heap size in bytes
    size: usize,
    /// Compute pipelines
    init_pipeline: ComputePipelineState,
    alloc_pipeline: ComputePipelineState,
    free_pipeline: ComputePipelineState,
    /// Command queue for operations
    command_queue: CommandQueue,
}

impl GpuHeap {
    /// Create a new GPU heap with the given size
    pub fn new(device: &Device, size: usize) -> Result<Self, String> {
        // Create heap buffer (GPU-only for best performance)
        let buffer = device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared, // Need Shared for init from CPU
        );
        buffer.set_label("GpuHeap");

        // Load shader library
        let library = device
            .new_library_with_source(include_str!("shaders/gpu_data_structures.metal"), &CompileOptions::new())
            .map_err(|e| format!("Failed to compile gpu_data_structures.metal: {}", e))?;

        // Create compute pipelines
        let init_fn = library
            .get_function("heap_init", None)
            .map_err(|e| format!("Failed to get heap_init: {}", e))?;
        let init_pipeline = device
            .new_compute_pipeline_state_with_function(&init_fn)
            .map_err(|e| format!("Failed to create init pipeline: {}", e))?;

        let alloc_fn = library
            .get_function("heap_alloc_batch", None)
            .map_err(|e| format!("Failed to get heap_alloc_batch: {}", e))?;
        let alloc_pipeline = device
            .new_compute_pipeline_state_with_function(&alloc_fn)
            .map_err(|e| format!("Failed to create alloc pipeline: {}", e))?;

        let free_fn = library
            .get_function("heap_free_batch", None)
            .map_err(|e| format!("Failed to get heap_free_batch: {}", e))?;
        let free_pipeline = device
            .new_compute_pipeline_state_with_function(&free_fn)
            .map_err(|e| format!("Failed to create free pipeline: {}", e))?;

        let command_queue = device.new_command_queue();

        let heap = Self {
            buffer,
            size,
            init_pipeline,
            alloc_pipeline,
            free_pipeline,
            command_queue,
        };

        // Initialize heap via GPU kernel
        heap.initialize(device)?;

        Ok(heap)
    }

    /// Initialize the heap (runs GPU kernel)
    fn initialize(&self, device: &Device) -> Result<(), String> {
        let size_buffer = device.new_buffer_with_data(
            &(self.size as u32) as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(&self.buffer), 0);
        encoder.set_buffer(1, Some(&size_buffer), 0);

        let threads = MTLSize::new(1, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(threadgroups, threads);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Get the heap buffer for binding to compute encoder
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get heap size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bind heap to compute encoder at given buffer indices
    ///
    /// - `header_index`: buffer index for HeapHeader
    /// - `data_index`: buffer index for heap data
    pub fn bind_to_encoder(&self, encoder: &ComputeCommandEncoderRef, header_index: u64, data_index: u64) {
        encoder.set_buffer(header_index, Some(&self.buffer), 0);
        encoder.set_buffer(data_index, Some(&self.buffer), 0);
    }

    /// Allocate blocks (batch operation)
    ///
    /// Returns offsets for each allocation (INVALID_OFFSET if failed)
    pub fn alloc_batch(&self, device: &Device, sizes: &[u32]) -> Vec<u32> {
        if sizes.is_empty() {
            return vec![];
        }

        let count = sizes.len();

        // Create input buffer
        let sizes_buffer = device.new_buffer_with_data(
            sizes.as_ptr() as *const _,
            (count * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer
        let results_buffer = device.new_buffer(
            (count * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create count buffer
        let count_buffer = device.new_buffer_with_data(
            &(count as u32) as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Execute kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.alloc_pipeline);
        encoder.set_buffer(0, Some(&self.buffer), 0);
        encoder.set_buffer(1, Some(&self.buffer), 0);
        encoder.set_buffer(2, Some(&sizes_buffer), 0);
        encoder.set_buffer(3, Some(&results_buffer), 0);
        encoder.set_buffer(4, Some(&count_buffer), 0);

        let threads_per_group = self.alloc_pipeline.thread_execution_width() as u64;
        let threadgroups = ((count as u64) + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let results_ptr = results_buffer.contents() as *const u32;
        let mut results = vec![0u32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(results_ptr, results.as_mut_ptr(), count);
        }

        results
    }

    /// Free blocks (batch operation)
    pub fn free_batch(&self, device: &Device, offsets: &[u32]) {
        if offsets.is_empty() {
            return;
        }

        let count = offsets.len();

        // Create input buffer
        let offsets_buffer = device.new_buffer_with_data(
            offsets.as_ptr() as *const _,
            (count * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create count buffer
        let count_buffer = device.new_buffer_with_data(
            &(count as u32) as *const u32 as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Execute kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.free_pipeline);
        encoder.set_buffer(0, Some(&self.buffer), 0);
        encoder.set_buffer(1, Some(&self.buffer), 0);
        encoder.set_buffer(2, Some(&offsets_buffer), 0);
        encoder.set_buffer(3, Some(&count_buffer), 0);

        let threads_per_group = self.free_pipeline.thread_execution_width() as u64;
        let threadgroups = ((count as u64) + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Read heap stats (for debugging only)
    pub fn read_stats(&self) -> HeapStats {
        let ptr = self.buffer.contents() as *const HeapHeader;
        let header = unsafe { *ptr };

        HeapStats {
            total_allocated: header.total_allocated,
            allocation_count: header.allocation_count,
            bump_ptr: header.bump_ptr,
            heap_size: header.heap_size,
            free_list_counts: header.free_list_counts,
        }
    }
}

/// Heap statistics (for debugging)
#[derive(Clone, Debug)]
pub struct HeapStats {
    pub total_allocated: u32,
    pub allocation_count: u32,
    pub bump_ptr: u32,
    pub heap_size: u32,
    pub free_list_counts: [u32; SIZE_CLASS_COUNT],
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU Vector
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-native resizable vector
///
/// Allocated within a GpuHeap. All operations run on GPU.
pub struct GpuVector {
    /// Offset within heap buffer
    offset: u32,
    /// Element size in bytes
    element_size: u32,
    /// Reference to heap
    heap: *const GpuHeap,
    /// Compute pipelines
    push_pipeline: ComputePipelineState,
    init_pipeline: ComputePipelineState,
    /// Reusable command queue (prevents resource leaks)
    command_queue: CommandQueue,
}

impl GpuVector {
    /// Create a new vector in the heap
    pub fn new(
        device: &Device,
        heap: &GpuHeap,
        element_size: u32,
        initial_capacity: u32,
    ) -> Result<Self, String> {
        // Calculate required size
        let header_size = mem::size_of::<GpuVectorHeader>() as u32;
        let data_size = element_size * initial_capacity;
        let total_size = header_size + data_size;

        // Allocate from heap
        let offsets = heap.alloc_batch(device, &[total_size]);
        let offset = offsets[0];
        if offset == INVALID_OFFSET {
            return Err("Failed to allocate vector".to_string());
        }

        // Load shader
        let library = device
            .new_library_with_source(include_str!("shaders/gpu_data_structures.metal"), &CompileOptions::new())
            .map_err(|e| format!("Failed to compile shader: {}", e))?;

        let init_fn = library.get_function("vector_init", None).map_err(|e| format!("{}", e))?;
        let init_pipeline = device.new_compute_pipeline_state_with_function(&init_fn).map_err(|e| format!("{}", e))?;

        let push_fn = library.get_function("vector_push_batch", None).map_err(|e| format!("{}", e))?;
        let push_pipeline = device.new_compute_pipeline_state_with_function(&push_fn).map_err(|e| format!("{}", e))?;

        // Create ONE command queue to reuse (prevents resource leaks)
        let command_queue = device.new_command_queue();

        let vec = Self {
            offset,
            element_size,
            heap: heap as *const _,
            push_pipeline,
            init_pipeline,
            command_queue,
        };

        // Initialize vector header
        vec.initialize(device, heap, initial_capacity)?;

        Ok(vec)
    }

    fn initialize(&self, device: &Device, heap: &GpuHeap, capacity: u32) -> Result<(), String> {
        // Reuse stored command queue (prevents resource leaks)
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let offsets = [self.offset];
        let capacities = [capacity];
        let elem_sizes = [self.element_size];
        let count = 1u32;

        let offsets_buf = device.new_buffer_with_data(offsets.as_ptr() as *const _, 4, MTLResourceOptions::StorageModeShared);
        let caps_buf = device.new_buffer_with_data(capacities.as_ptr() as *const _, 4, MTLResourceOptions::StorageModeShared);
        let sizes_buf = device.new_buffer_with_data(elem_sizes.as_ptr() as *const _, 4, MTLResourceOptions::StorageModeShared);
        let count_buf = device.new_buffer_with_data(&count as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(heap.buffer()), 0);
        encoder.set_buffer(1, Some(&offsets_buf), 0);
        encoder.set_buffer(2, Some(&caps_buf), 0);
        encoder.set_buffer(3, Some(&sizes_buf), 0);
        encoder.set_buffer(4, Some(&count_buf), 0);

        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Get the offset of this vector in the heap
    pub fn offset(&self) -> u32 {
        self.offset
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GPU HashMap
// ═══════════════════════════════════════════════════════════════════════════

/// GPU-native hash map with Cuckoo hashing
///
/// O(1) guaranteed lookups - exactly 2 memory accesses per key.
pub struct GpuHashMap {
    /// Offset within heap buffer
    offset: u32,
    /// Capacity (power of 2)
    capacity: u32,
    /// Reference to heap
    heap: *const GpuHeap,
    /// Compute pipelines
    init_pipeline: ComputePipelineState,
    insert_pipeline: ComputePipelineState,
    get_pipeline: ComputePipelineState,
    remove_pipeline: ComputePipelineState,
    /// Reusable command queue (prevents resource leaks)
    command_queue: CommandQueue,
}

impl GpuHashMap {
    /// Create a new hash map in the heap
    pub fn new(device: &Device, heap: &GpuHeap, capacity: u32) -> Result<Self, String> {
        // Capacity must be power of 2
        let capacity = capacity.next_power_of_two();

        // Calculate required size: header + 2 tables
        let header_size = mem::size_of::<GpuHashMapHeader>() as u32;
        let table_size = capacity * mem::size_of::<CuckooEntry>() as u32;
        let total_size = header_size + 2 * table_size;

        // Allocate from heap
        let offsets = heap.alloc_batch(device, &[total_size]);
        let offset = offsets[0];
        if offset == INVALID_OFFSET {
            return Err("Failed to allocate hashmap".to_string());
        }

        // Load shader
        let library = device
            .new_library_with_source(include_str!("shaders/gpu_data_structures.metal"), &CompileOptions::new())
            .map_err(|e| format!("Failed to compile shader: {}", e))?;

        let init_fn = library.get_function("hashmap_init", None).map_err(|e| format!("{}", e))?;
        let init_pipeline = device.new_compute_pipeline_state_with_function(&init_fn).map_err(|e| format!("{}", e))?;

        let insert_fn = library.get_function("hashmap_insert_batch", None).map_err(|e| format!("{}", e))?;
        let insert_pipeline = device.new_compute_pipeline_state_with_function(&insert_fn).map_err(|e| format!("{}", e))?;

        let get_fn = library.get_function("hashmap_get_batch", None).map_err(|e| format!("{}", e))?;
        let get_pipeline = device.new_compute_pipeline_state_with_function(&get_fn).map_err(|e| format!("{}", e))?;

        let remove_fn = library.get_function("hashmap_remove_batch", None).map_err(|e| format!("{}", e))?;
        let remove_pipeline = device.new_compute_pipeline_state_with_function(&remove_fn).map_err(|e| format!("{}", e))?;

        // Create ONE command queue to reuse (prevents resource leaks)
        let command_queue = device.new_command_queue();

        let map = Self {
            offset,
            capacity,
            heap: heap as *const _,
            init_pipeline,
            insert_pipeline,
            get_pipeline,
            remove_pipeline,
            command_queue,
        };

        // Initialize hashmap
        map.initialize(device, heap)?;

        Ok(map)
    }

    fn initialize(&self, device: &Device, heap: &GpuHeap) -> Result<(), String> {
        // Reuse stored command queue (prevents resource leaks)
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let offsets = [self.offset];
        let capacities = [self.capacity];
        let count = self.capacity; // Use capacity threads to clear tables
        let heap_size = heap.size();

        let offsets_buf = device.new_buffer_with_data(offsets.as_ptr() as *const _, 4, MTLResourceOptions::StorageModeShared);
        let caps_buf = device.new_buffer_with_data(capacities.as_ptr() as *const _, 4, MTLResourceOptions::StorageModeShared);
        let count_buf = device.new_buffer_with_data(&count as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);
        let heap_size_buf = device.new_buffer_with_data(&heap_size as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        encoder.set_compute_pipeline_state(&self.init_pipeline);
        encoder.set_buffer(0, Some(heap.buffer()), 0);
        encoder.set_buffer(1, Some(&offsets_buf), 0);
        encoder.set_buffer(2, Some(&caps_buf), 0);
        encoder.set_buffer(3, Some(&count_buf), 0);
        encoder.set_buffer(4, Some(&heap_size_buf), 0);

        let threads_per_group = 64u64;
        let threadgroups = ((self.capacity as u64) + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups.max(1), 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Insert key-value pairs (batch operation)
    ///
    /// Returns vec of 1s (success) or 0s (failed - needs rehash)
    pub fn insert_batch(&self, device: &Device, heap: &GpuHeap, keys: &[u32], values: &[u32]) -> Vec<u32> {
        assert_eq!(keys.len(), values.len());
        let count = keys.len();
        if count == 0 {
            return vec![];
        }

        // Reuse stored command queue (prevents resource leaks)
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let keys_buf = device.new_buffer_with_data(keys.as_ptr() as *const _, (count * 4) as u64, MTLResourceOptions::StorageModeShared);
        let vals_buf = device.new_buffer_with_data(values.as_ptr() as *const _, (count * 4) as u64, MTLResourceOptions::StorageModeShared);
        let results_buf = device.new_buffer((count * 4) as u64, MTLResourceOptions::StorageModeShared);
        let count_buf = device.new_buffer_with_data(&(count as u32) as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        encoder.set_compute_pipeline_state(&self.insert_pipeline);
        encoder.set_buffer(0, Some(heap.buffer()), 0);
        encoder.set_bytes(1, 4, &self.offset as *const _ as *const _);
        encoder.set_buffer(2, Some(&keys_buf), 0);
        encoder.set_buffer(3, Some(&vals_buf), 0);
        encoder.set_buffer(4, Some(&results_buf), 0);
        encoder.set_buffer(5, Some(&count_buf), 0);

        let threads_per_group = 64u64;
        let threadgroups = ((count as u64) + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups.max(1), 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let results_ptr = results_buf.contents() as *const u32;
        let mut results = vec![0u32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(results_ptr, results.as_mut_ptr(), count);
        }

        results
    }

    /// Look up values by keys (batch operation)
    ///
    /// Returns (values, found) where found[i] = 1 if key was found
    pub fn get_batch(&self, device: &Device, heap: &GpuHeap, keys: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let count = keys.len();
        if count == 0 {
            return (vec![], vec![]);
        }

        // Reuse stored command queue (prevents resource leaks)
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let keys_buf = device.new_buffer_with_data(keys.as_ptr() as *const _, (count * 4) as u64, MTLResourceOptions::StorageModeShared);
        let vals_buf = device.new_buffer((count * 4) as u64, MTLResourceOptions::StorageModeShared);
        let found_buf = device.new_buffer((count * 4) as u64, MTLResourceOptions::StorageModeShared);
        let count_buf = device.new_buffer_with_data(&(count as u32) as *const _ as *const _, 4, MTLResourceOptions::StorageModeShared);

        encoder.set_compute_pipeline_state(&self.get_pipeline);
        encoder.set_buffer(0, Some(heap.buffer()), 0);
        encoder.set_bytes(1, 4, &self.offset as *const _ as *const _);
        encoder.set_buffer(2, Some(&keys_buf), 0);
        encoder.set_buffer(3, Some(&vals_buf), 0);
        encoder.set_buffer(4, Some(&found_buf), 0);
        encoder.set_buffer(5, Some(&count_buf), 0);

        let threads_per_group = 64u64;
        let threadgroups = ((count as u64) + threads_per_group - 1) / threads_per_group;

        encoder.dispatch_thread_groups(
            MTLSize::new(threadgroups.max(1), 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let vals_ptr = vals_buf.contents() as *const u32;
        let found_ptr = found_buf.contents() as *const u32;
        let mut values = vec![0u32; count];
        let mut found = vec![0u32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(vals_ptr, values.as_mut_ptr(), count);
            std::ptr::copy_nonoverlapping(found_ptr, found.as_mut_ptr(), count);
        }

        (values, found)
    }

    /// Get the offset of this hashmap in the heap
    pub fn offset(&self) -> u32 {
        self.offset
    }

    /// Get capacity
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_creation() {
        let device = Device::system_default().expect("No Metal device");
        let heap = GpuHeap::new(&device, 1024 * 1024).expect("Failed to create heap");

        let stats = heap.read_stats();
        assert_eq!(stats.heap_size, 1024 * 1024);
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_heap_alloc_free() {
        let device = Device::system_default().expect("No Metal device");
        let heap = GpuHeap::new(&device, 1024 * 1024).expect("Failed to create heap");

        // Allocate some blocks
        let sizes = vec![64, 128, 256, 512];
        let offsets = heap.alloc_batch(&device, &sizes);

        assert_eq!(offsets.len(), 4);
        for offset in &offsets {
            assert_ne!(*offset, INVALID_OFFSET);
        }

        let stats = heap.read_stats();
        assert_eq!(stats.allocation_count, 4);

        // Free the blocks
        heap.free_batch(&device, &offsets);

        let stats = heap.read_stats();
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_hashmap_basic() {
        let device = Device::system_default().expect("No Metal device");
        let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");
        let map = GpuHashMap::new(&device, &heap, 256).expect("Failed to create hashmap");

        // Insert some key-value pairs
        let keys = vec![1, 2, 3, 4, 5];
        let values = vec![10, 20, 30, 40, 50];
        let results = map.insert_batch(&device, &heap, &keys, &values);

        // All should succeed
        for r in &results {
            assert_eq!(*r, 1);
        }

        // Look them up
        let (got_values, found) = map.get_batch(&device, &heap, &keys);

        for i in 0..5 {
            assert_eq!(found[i], 1);
            assert_eq!(got_values[i], values[i]);
        }

        // Look up non-existent key
        let (_, found) = map.get_batch(&device, &heap, &[999]);
        assert_eq!(found[0], 0);
    }
}
