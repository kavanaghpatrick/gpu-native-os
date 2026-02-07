//! GPU App Memory Pool (Issue #150)
//!
//! Block-based memory pool for GPU app buffers. Uses atomic bitmap for
//! lock-free allocation/deallocation entirely on GPU.

use metal::{Buffer, Device, MTLResourceOptions};
use std::sync::atomic::{AtomicU32, Ordering};

/// Block size - 4KB (page aligned)
pub const BLOCK_SIZE: usize = 4096;

/// Default pool size - 64MB
pub const DEFAULT_POOL_SIZE: usize = 64 * 1024 * 1024;

/// Header size in bytes
pub const HEADER_SIZE: usize = 64;

/// Allocation failure marker
pub const ALLOC_FAILED: u32 = 0xFFFFFFFF;

/// Memory pool header stored at start of GPU buffer
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemoryPoolHeader {
    pub pool_size: u32,
    pub allocated_bytes: u32,  // Atomic in GPU
    pub free_list_head: u32,   // Atomic in GPU (unused in bitmap approach)
    pub block_size: u32,
    pub block_count: u32,
    pub bitmap_offset: u32,
    pub data_offset: u32,
    pub _pad: u32,
    pub _reserved: [u32; 8],   // Reserve space for future use
}

const _: () = assert!(std::mem::size_of::<MemoryPoolHeader>() == HEADER_SIZE);

/// GPU App Memory Pool
///
/// Pre-allocated GPU buffer with block-based allocation.
/// Apps get offsets into this pool, not separate Metal buffers.
pub struct GpuAppMemoryPool {
    buffer: Buffer,
    block_count: u32,
    bitmap_words: usize,
    data_offset: usize,
}

impl GpuAppMemoryPool {
    /// Create a new memory pool
    ///
    /// # Arguments
    /// * `device` - Metal device
    /// * `size` - Total usable memory size in bytes
    pub fn new(device: &Device, size: usize) -> Self {
        let block_count = (size / BLOCK_SIZE) as u32;
        let bitmap_words = ((block_count as usize) + 31) / 32;
        let bitmap_size = bitmap_words * 4;

        // Align data offset to block size
        let data_offset = ((HEADER_SIZE + bitmap_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        let total_size = data_offset + size;

        let buffer = device.new_buffer(
            total_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize header and bitmap
        unsafe {
            let ptr = buffer.contents() as *mut u8;

            // Write header
            let header = ptr as *mut MemoryPoolHeader;
            (*header).pool_size = size as u32;
            (*header).allocated_bytes = 0;
            (*header).free_list_head = ALLOC_FAILED;
            (*header).block_size = BLOCK_SIZE as u32;
            (*header).block_count = block_count;
            (*header).bitmap_offset = HEADER_SIZE as u32;
            (*header).data_offset = data_offset as u32;
            (*header)._pad = 0;
            (*header)._reserved = [0; 8];

            // Clear bitmap (all blocks free)
            let bitmap = ptr.add(HEADER_SIZE) as *mut u32;
            for i in 0..bitmap_words {
                *bitmap.add(i) = 0;
            }
        }

        Self {
            buffer,
            block_count,
            bitmap_words,
            data_offset,
        }
    }

    /// Get the underlying Metal buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get total pool size in bytes
    pub fn pool_size(&self) -> usize {
        self.block_count as usize * BLOCK_SIZE
    }

    /// Get number of blocks
    pub fn block_count(&self) -> u32 {
        self.block_count
    }

    /// Get current allocated bytes (CPU read, may be stale)
    pub fn allocated_bytes(&self) -> u32 {
        unsafe {
            let header = self.buffer.contents() as *const MemoryPoolHeader;
            (*header).allocated_bytes
        }
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f32 {
        let allocated = self.allocated_bytes() as f32;
        let total = self.pool_size() as f32;
        if total > 0.0 {
            allocated / total
        } else {
            0.0
        }
    }

    /// Get free bytes
    pub fn free_bytes(&self) -> u32 {
        self.pool_size() as u32 - self.allocated_bytes()
    }

    /// CPU-side allocation (for initial setup)
    /// Returns offset into pool, or ALLOC_FAILED
    pub fn allocate_cpu(&self, size: usize) -> u32 {
        let num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if num_blocks == 0 {
            return ALLOC_FAILED;
        }

        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let header = ptr as *mut MemoryPoolHeader;
            let bitmap = ptr.add(HEADER_SIZE) as *mut u32;

            // Simple first-fit scan
            let total_blocks = self.block_count as usize;

            let mut start = 0;
            while start + num_blocks <= total_blocks {
                let mut found = true;

                // Check if num_blocks consecutive blocks are free
                for i in 0..num_blocks {
                    let block = start + i;
                    let word = block / 32;
                    let bit = block % 32;

                    if *bitmap.add(word) & (1u32 << bit) != 0 {
                        found = false;
                        start = block + 1; // Skip to after this allocated block
                        break;
                    }
                }

                if found {
                    // Mark blocks as allocated
                    for i in 0..num_blocks {
                        let block = start + i;
                        let word = block / 32;
                        let bit = block % 32;
                        *bitmap.add(word) |= 1u32 << bit;
                    }

                    // Update allocated bytes
                    (*header).allocated_bytes += (num_blocks * BLOCK_SIZE) as u32;

                    // Return offset
                    return self.data_offset as u32 + (start * BLOCK_SIZE) as u32;
                }
            }
        }

        ALLOC_FAILED
    }

    /// CPU-side free (for cleanup)
    pub fn free_cpu(&self, offset: u32, size: usize) {
        if offset == ALLOC_FAILED {
            return;
        }

        let num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if num_blocks == 0 {
            return;
        }

        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let header = ptr as *mut MemoryPoolHeader;
            let bitmap = ptr.add(HEADER_SIZE) as *mut u32;

            let start_block = (offset as usize - self.data_offset) / BLOCK_SIZE;

            // Clear blocks in bitmap
            for i in 0..num_blocks {
                let block = start_block + i;
                if block < self.block_count as usize {
                    let word = block / 32;
                    let bit = block % 32;
                    *bitmap.add(word) &= !(1u32 << bit);
                }
            }

            // Update allocated bytes
            let freed = (num_blocks * BLOCK_SIZE) as u32;
            if (*header).allocated_bytes >= freed {
                (*header).allocated_bytes -= freed;
            }
        }
    }

    /// Get pointer to memory at offset
    pub fn get_ptr(&self, offset: u32) -> *mut u8 {
        unsafe { (self.buffer.contents() as *mut u8).add(offset as usize) }
    }

    /// Check if offset is valid
    pub fn is_valid_offset(&self, offset: u32) -> bool {
        let offset = offset as usize;
        offset >= self.data_offset && offset < self.data_offset + self.pool_size()
    }
}

/// Metal shader code for memory pool operations
pub const MEMORY_POOL_METAL_HEADER: &str = r#"
// Memory Pool Header
struct MemoryPoolHeader {
    uint pool_size;
    atomic_uint allocated_bytes;
    atomic_uint free_list_head;
    uint block_size;
    uint block_count;
    uint bitmap_offset;
    uint data_offset;
    uint _pad;
    uint _reserved[8];
};

#define BLOCK_SIZE 4096
#define ALLOC_FAILED 0xFFFFFFFF

// Allocate contiguous blocks (single thread only)
uint memory_pool_allocate(
    device MemoryPoolHeader* pool,
    device atomic_uint* bitmap,
    uint size
) {
    uint num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0) return ALLOC_FAILED;

    uint total_blocks = pool->block_count;

    // First-fit scan
    for (uint start = 0; start + num_blocks <= total_blocks; ) {
        bool found = true;

        // Check if num_blocks consecutive blocks are free
        for (uint i = 0; i < num_blocks && found; i++) {
            uint block = start + i;
            uint word = block / 32;
            uint bit = block % 32;

            uint mask = atomic_load_explicit(&bitmap[word], memory_order_relaxed);
            if (mask & (1u << bit)) {
                found = false;
                start = block + 1;
            }
        }

        if (found) {
            // Try to atomically mark all blocks
            bool success = true;
            for (uint i = 0; i < num_blocks && success; i++) {
                uint block = start + i;
                uint word = block / 32;
                uint bit = block % 32;

                uint old_val = atomic_fetch_or_explicit(
                    &bitmap[word],
                    1u << bit,
                    memory_order_relaxed
                );

                // Check if bit was already set (contention)
                if (old_val & (1u << bit)) {
                    // Someone else got it, undo and retry
                    for (uint j = 0; j < i; j++) {
                        uint undo_block = start + j;
                        uint undo_word = undo_block / 32;
                        uint undo_bit = undo_block % 32;
                        atomic_fetch_and_explicit(
                            &bitmap[undo_word],
                            ~(1u << undo_bit),
                            memory_order_relaxed
                        );
                    }
                    success = false;
                    start = block + 1;
                }
            }

            if (success) {
                atomic_fetch_add_explicit(
                    &pool->allocated_bytes,
                    num_blocks * BLOCK_SIZE,
                    memory_order_relaxed
                );
                return pool->data_offset + start * BLOCK_SIZE;
            }
        }
    }

    return ALLOC_FAILED;
}

// Free blocks
void memory_pool_free(
    device MemoryPoolHeader* pool,
    device atomic_uint* bitmap,
    uint offset,
    uint size
) {
    if (offset == ALLOC_FAILED) return;

    uint num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint start_block = (offset - pool->data_offset) / BLOCK_SIZE;

    for (uint i = 0; i < num_blocks; i++) {
        uint block = start_block + i;
        uint word = block / 32;
        uint bit = block % 32;

        atomic_fetch_and_explicit(
            &bitmap[word],
            ~(1u << bit),
            memory_order_relaxed
        );
    }

    atomic_fetch_sub_explicit(
        &pool->allocated_bytes,
        num_blocks * BLOCK_SIZE,
        memory_order_relaxed
    );
}

// Get pointer to allocated memory
device uint8_t* memory_pool_get_ptr(
    device uint8_t* pool_buffer,
    uint offset
) {
    return pool_buffer + offset;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool() -> Option<GpuAppMemoryPool> {
        Device::system_default().map(|device| {
            GpuAppMemoryPool::new(&device, 1024 * 1024) // 1MB
        })
    }

    #[test]
    fn test_pool_initialization() {
        if let Some(pool) = create_test_pool() {
            assert_eq!(pool.allocated_bytes(), 0);
            assert_eq!(pool.block_count(), 256); // 1MB / 4KB = 256 blocks
            assert_eq!(pool.utilization(), 0.0);
        }
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<MemoryPoolHeader>(), HEADER_SIZE);
    }

    #[test]
    fn test_allocate_single_block() {
        if let Some(pool) = create_test_pool() {
            let offset = pool.allocate_cpu(1000); // Less than one block
            assert_ne!(offset, ALLOC_FAILED);
            assert!(pool.is_valid_offset(offset));
            assert_eq!(pool.allocated_bytes(), BLOCK_SIZE as u32);
        }
    }

    #[test]
    fn test_allocate_multiple_blocks() {
        if let Some(pool) = create_test_pool() {
            let offset = pool.allocate_cpu(10000); // ~3 blocks
            assert_ne!(offset, ALLOC_FAILED);
            assert!(pool.is_valid_offset(offset));
            assert_eq!(pool.allocated_bytes(), 3 * BLOCK_SIZE as u32);
        }
    }

    #[test]
    fn test_allocate_and_free() {
        if let Some(pool) = create_test_pool() {
            let offset = pool.allocate_cpu(BLOCK_SIZE);
            assert_ne!(offset, ALLOC_FAILED);
            assert_eq!(pool.allocated_bytes(), BLOCK_SIZE as u32);

            pool.free_cpu(offset, BLOCK_SIZE);
            assert_eq!(pool.allocated_bytes(), 0);
        }
    }

    #[test]
    fn test_multiple_allocations() {
        if let Some(pool) = create_test_pool() {
            let a1 = pool.allocate_cpu(BLOCK_SIZE);
            let a2 = pool.allocate_cpu(BLOCK_SIZE);
            let a3 = pool.allocate_cpu(BLOCK_SIZE);

            assert_ne!(a1, ALLOC_FAILED);
            assert_ne!(a2, ALLOC_FAILED);
            assert_ne!(a3, ALLOC_FAILED);

            // All should be different
            assert_ne!(a1, a2);
            assert_ne!(a2, a3);
            assert_ne!(a1, a3);

            assert_eq!(pool.allocated_bytes(), 3 * BLOCK_SIZE as u32);
        }
    }

    #[test]
    fn test_free_and_reuse() {
        if let Some(pool) = create_test_pool() {
            let a1 = pool.allocate_cpu(BLOCK_SIZE);
            pool.free_cpu(a1, BLOCK_SIZE);

            // Next allocation should reuse the freed block
            let a2 = pool.allocate_cpu(BLOCK_SIZE);
            assert_eq!(a1, a2);
        }
    }

    #[test]
    fn test_pool_exhaustion() {
        if let Some(pool) = Device::system_default().map(|device| {
            GpuAppMemoryPool::new(&device, BLOCK_SIZE * 4) // 4 blocks only
        }) {
            let a1 = pool.allocate_cpu(BLOCK_SIZE);
            let a2 = pool.allocate_cpu(BLOCK_SIZE);
            let a3 = pool.allocate_cpu(BLOCK_SIZE);
            let a4 = pool.allocate_cpu(BLOCK_SIZE);

            assert_ne!(a1, ALLOC_FAILED);
            assert_ne!(a2, ALLOC_FAILED);
            assert_ne!(a3, ALLOC_FAILED);
            assert_ne!(a4, ALLOC_FAILED);

            // Pool should be exhausted now
            let a5 = pool.allocate_cpu(BLOCK_SIZE);
            assert_eq!(a5, ALLOC_FAILED);
        }
    }

    #[test]
    fn test_fragmentation_and_coalescing() {
        if let Some(pool) = Device::system_default().map(|device| {
            GpuAppMemoryPool::new(&device, BLOCK_SIZE * 10)
        }) {
            // Allocate 5 single blocks
            let offsets: Vec<u32> = (0..5).map(|_| pool.allocate_cpu(BLOCK_SIZE)).collect();

            // Free every other block
            pool.free_cpu(offsets[1], BLOCK_SIZE);
            pool.free_cpu(offsets[3], BLOCK_SIZE);

            // Try to allocate 2 contiguous blocks - should fail due to fragmentation
            let big = pool.allocate_cpu(BLOCK_SIZE * 2);

            // Should find space after the initial 5 blocks
            assert_ne!(big, ALLOC_FAILED);
        }
    }

    #[test]
    fn test_utilization() {
        if let Some(pool) = create_test_pool() {
            assert_eq!(pool.utilization(), 0.0);

            pool.allocate_cpu(512 * 1024); // 512KB = half
            // Note: utilization is based on allocated blocks, not requested size
            let util = pool.utilization();
            assert!(util > 0.4 && util < 0.6);
        }
    }

    #[test]
    fn test_zero_size_allocation() {
        if let Some(pool) = create_test_pool() {
            let offset = pool.allocate_cpu(0);
            assert_eq!(offset, ALLOC_FAILED);
        }
    }

    #[test]
    fn test_invalid_free() {
        if let Some(pool) = create_test_pool() {
            // Freeing invalid offset should not crash
            pool.free_cpu(ALLOC_FAILED, BLOCK_SIZE);
            assert_eq!(pool.allocated_bytes(), 0);
        }
    }
}
