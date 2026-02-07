//! GPU App Slot Allocator (Issue #149)
//!
//! GPU-native allocator for app slots. Uses atomic free list for
//! lock-free allocation/deallocation entirely on GPU.

use crate::gpu_os::app_descriptor::{GpuAppDescriptor, INVALID_SLOT, MAX_APP_SLOTS};
use metal::{Buffer, Device, MTLResourceOptions};
use std::sync::atomic::{AtomicU32, Ordering};

/// Header for the slot pool - stored at beginning of GPU buffer
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SlotPoolHeader {
    pub free_head: u32,      // Atomic: index of first free slot
    pub active_count: u32,   // Atomic: number of active apps
    pub max_slots: u32,      // Maximum number of slots
    pub _pad: u32,
}

const HEADER_SIZE: usize = std::mem::size_of::<SlotPoolHeader>();
const _: () = assert!(HEADER_SIZE == 16);

/// GPU App Slot Pool
///
/// Manages a pool of app slots with lock-free allocation/deallocation.
/// Slots are stored in a GPU buffer and can be allocated/freed from
/// GPU compute kernels using atomic operations.
pub struct GpuAppSlotPool {
    buffer: Buffer,
    max_slots: u32,
}

impl GpuAppSlotPool {
    /// Create a new slot pool
    ///
    /// # Arguments
    /// * `device` - Metal device
    /// * `max_slots` - Maximum number of app slots
    pub fn new(device: &Device, max_slots: u32) -> Self {
        let max_slots = max_slots.min(MAX_APP_SLOTS as u32);
        let slot_size = std::mem::size_of::<GpuAppDescriptor>();
        let total_size = HEADER_SIZE + (max_slots as usize * slot_size);

        let buffer = device.new_buffer(
            total_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize header and free list
        unsafe {
            let ptr = buffer.contents() as *mut u8;

            // Write header
            let header = ptr as *mut SlotPoolHeader;
            (*header).free_head = 0; // Start with slot 0
            (*header).active_count = 0;
            (*header).max_slots = max_slots;
            (*header)._pad = 0;

            // Initialize slots - build free list
            // Each free slot stores the next free slot index in slot_id
            let slots = ptr.add(HEADER_SIZE) as *mut GpuAppDescriptor;
            for i in 0..max_slots {
                let slot = slots.add(i as usize);
                // Free slots use slot_id to store next pointer
                (*slot).slot_id = if i + 1 < max_slots {
                    i + 1
                } else {
                    INVALID_SLOT
                };
                (*slot).flags = 0;
                // Zero out rest of the slot
                (*slot).window_id = 0;
                (*slot).app_type = 0;
                (*slot).state_offset = 0;
                (*slot).state_size = 0;
                (*slot).vertex_offset = 0;
                (*slot).vertex_size = 0;
                (*slot).param_offset = 0;
                (*slot).param_size = 0;
                (*slot).extra_offset = 0;
                (*slot).extra_size = 0;
                (*slot).thread_count = 0;
                (*slot).vertex_count = 0;
                (*slot).clear_color = [0.0, 0.0, 0.0, 1.0];
                (*slot).preferred_size = [800.0, 600.0];
                (*slot).compute_pipeline_handle = INVALID_SLOT;
                (*slot).render_pipeline_handle = INVALID_SLOT;
                (*slot).total_time = 0.0;
                (*slot).frame_count = 0;
                (*slot).mouse_x = 0.0;
                (*slot).mouse_y = 0.0;
                (*slot).mouse_buttons = 0;
                (*slot).key_modifiers = 0;
                (*slot)._runtime_pad = [0; 4];
            }
        }

        Self { buffer, max_slots }
    }

    /// Get the underlying Metal buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get maximum number of slots
    pub fn max_slots(&self) -> u32 {
        self.max_slots
    }

    /// Get current number of active apps (may be stale if GPU is modifying)
    pub fn active_count(&self) -> u32 {
        unsafe {
            let header = self.buffer.contents() as *const SlotPoolHeader;
            (*header).active_count
        }
    }

    /// Get current free list head (may be stale if GPU is modifying)
    pub fn free_head(&self) -> u32 {
        unsafe {
            let header = self.buffer.contents() as *const SlotPoolHeader;
            (*header).free_head
        }
    }

    /// CPU-side slot allocation (for initial setup)
    /// Returns allocated slot index, or INVALID_SLOT if pool exhausted
    pub fn allocate_cpu(&self) -> u32 {
        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let header = ptr as *mut SlotPoolHeader;
            let slots = ptr.add(HEADER_SIZE) as *mut GpuAppDescriptor;

            let slot = (*header).free_head;
            if slot == INVALID_SLOT {
                return INVALID_SLOT; // Pool exhausted
            }

            // Pop from free list
            let next = (*slots.add(slot as usize)).slot_id;
            (*header).free_head = next;
            (*header).active_count += 1;

            // Initialize the slot
            (*slots.add(slot as usize)).slot_id = slot;
            (*slots.add(slot as usize)).flags = 0; // Caller will set flags

            slot
        }
    }

    /// CPU-side slot free (for cleanup)
    pub fn free_cpu(&self, slot: u32) {
        if slot >= self.max_slots {
            return;
        }

        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let header = ptr as *mut SlotPoolHeader;
            let slots = ptr.add(HEADER_SIZE) as *mut GpuAppDescriptor;

            // Clear the slot
            (*slots.add(slot as usize)).flags = 0;

            // Push to free list
            (*slots.add(slot as usize)).slot_id = (*header).free_head;
            (*header).free_head = slot;

            if (*header).active_count > 0 {
                (*header).active_count -= 1;
            }
        }
    }

    /// Get a slot descriptor (read-only)
    pub fn get_slot(&self, slot: u32) -> Option<GpuAppDescriptor> {
        if slot >= self.max_slots {
            return None;
        }

        unsafe {
            let ptr = self.buffer.contents() as *const u8;
            let slots = ptr.add(HEADER_SIZE) as *const GpuAppDescriptor;
            Some(*slots.add(slot as usize))
        }
    }

    /// Get a mutable pointer to a slot descriptor
    pub fn get_slot_mut(&self, slot: u32) -> Option<*mut GpuAppDescriptor> {
        if slot >= self.max_slots {
            return None;
        }

        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let slots = ptr.add(HEADER_SIZE) as *mut GpuAppDescriptor;
            Some(slots.add(slot as usize))
        }
    }

    /// Update a slot with a new descriptor
    pub fn update_slot(&self, slot: u32, desc: &GpuAppDescriptor) {
        if slot >= self.max_slots {
            return;
        }

        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            let slots = ptr.add(HEADER_SIZE) as *mut GpuAppDescriptor;
            *slots.add(slot as usize) = *desc;
        }
    }

    /// Iterate over all active slots
    pub fn active_slots(&self) -> Vec<(u32, GpuAppDescriptor)> {
        let mut result = Vec::new();
        unsafe {
            let ptr = self.buffer.contents() as *const u8;
            let slots = ptr.add(HEADER_SIZE) as *const GpuAppDescriptor;

            for i in 0..self.max_slots {
                let slot = &*slots.add(i as usize);
                if slot.flags & crate::gpu_os::app_descriptor::flags::ACTIVE != 0 {
                    result.push((i, *slot));
                }
            }
        }
        result
    }
}

/// Metal shader code for slot allocator operations
pub const SLOT_ALLOCATOR_METAL_HEADER: &str = r#"
// Slot Pool Header
struct SlotPoolHeader {
    atomic_uint free_head;
    atomic_uint active_count;
    uint max_slots;
    uint _pad;
};

// Allocate a slot from the pool (single thread)
uint slot_allocate(
    device SlotPoolHeader* header,
    device GpuAppDescriptor* slots
) {
    // Atomic pop from free list
    uint slot = atomic_load_explicit(&header->free_head, memory_order_relaxed);

    if (slot == INVALID_SLOT) {
        return INVALID_SLOT;  // Pool exhausted
    }

    // CAS loop to pop
    while (true) {
        uint next = slots[slot].slot_id;  // Free slots store next in slot_id

        if (atomic_compare_exchange_weak_explicit(
            &header->free_head,
            &slot,
            next,
            memory_order_relaxed,
            memory_order_relaxed
        )) {
            break;
        }

        if (slot == INVALID_SLOT) {
            return INVALID_SLOT;
        }
    }

    // Initialize the slot
    slots[slot].slot_id = slot;
    slots[slot].flags = 0;

    atomic_fetch_add_explicit(&header->active_count, 1, memory_order_relaxed);
    return slot;
}

// Free a slot back to the pool (single thread)
void slot_free(
    device SlotPoolHeader* header,
    device GpuAppDescriptor* slots,
    uint slot
) {
    if (slot >= header->max_slots) return;

    // Clear the slot
    slots[slot].flags = 0;

    // Atomic push to free list
    uint old_head = atomic_load_explicit(&header->free_head, memory_order_relaxed);

    while (true) {
        slots[slot].slot_id = old_head;  // Point to old head

        if (atomic_compare_exchange_weak_explicit(
            &header->free_head,
            &old_head,
            slot,
            memory_order_release,
            memory_order_relaxed
        )) {
            break;
        }
    }

    atomic_fetch_sub_explicit(&header->active_count, 1, memory_order_relaxed);
}

// Check if a slot is active
bool slot_is_active(device const GpuAppDescriptor* slots, uint slot) {
    return (slots[slot].flags & APP_FLAG_ACTIVE) != 0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool() -> Option<GpuAppSlotPool> {
        Device::system_default().map(|device| GpuAppSlotPool::new(&device, 64))
    }

    #[test]
    fn test_pool_initialization() {
        if let Some(pool) = create_test_pool() {
            assert_eq!(pool.active_count(), 0);
            assert_eq!(pool.max_slots(), 64);
            assert_eq!(pool.free_head(), 0);
        }
    }

    #[test]
    fn test_free_list_structure() {
        if let Some(pool) = create_test_pool() {
            // Verify free list: 0 -> 1 -> 2 -> ... -> 63 -> INVALID
            for i in 0..63u32 {
                let slot = pool.get_slot(i).unwrap();
                assert_eq!(slot.slot_id, i + 1);
                assert_eq!(slot.flags, 0);
            }
            // Last slot points to INVALID
            let last = pool.get_slot(63).unwrap();
            assert_eq!(last.slot_id, INVALID_SLOT);
        }
    }

    #[test]
    fn test_allocate_single_slot() {
        if let Some(pool) = create_test_pool() {
            let slot = pool.allocate_cpu();
            assert_eq!(slot, 0);
            assert_eq!(pool.active_count(), 1);
            assert_eq!(pool.free_head(), 1);
        }
    }

    #[test]
    fn test_allocate_multiple_slots() {
        if let Some(pool) = create_test_pool() {
            let s1 = pool.allocate_cpu();
            let s2 = pool.allocate_cpu();
            let s3 = pool.allocate_cpu();

            assert_eq!(s1, 0);
            assert_eq!(s2, 1);
            assert_eq!(s3, 2);
            assert_eq!(pool.active_count(), 3);
            assert_eq!(pool.free_head(), 3);
        }
    }

    #[test]
    fn test_free_slot() {
        if let Some(pool) = create_test_pool() {
            let slot = pool.allocate_cpu();
            assert_eq!(slot, 0);
            assert_eq!(pool.active_count(), 1);

            pool.free_cpu(slot);
            assert_eq!(pool.active_count(), 0);
            assert_eq!(pool.free_head(), 0); // Freed slot is now head
        }
    }

    #[test]
    fn test_allocate_free_reuse() {
        if let Some(pool) = create_test_pool() {
            let s1 = pool.allocate_cpu();
            assert_eq!(s1, 0);

            pool.free_cpu(s1);

            // Next allocation should reuse slot 0
            let s2 = pool.allocate_cpu();
            assert_eq!(s2, 0);
        }
    }

    #[test]
    fn test_pool_exhaustion() {
        if let Some(pool) = Device::system_default().map(|device| GpuAppSlotPool::new(&device, 4)) {
            // Allocate all 4 slots
            for i in 0..4 {
                let slot = pool.allocate_cpu();
                assert_eq!(slot, i);
            }

            // Next allocation should fail
            let slot = pool.allocate_cpu();
            assert_eq!(slot, INVALID_SLOT);
        }
    }

    #[test]
    fn test_update_slot() {
        if let Some(pool) = create_test_pool() {
            let slot = pool.allocate_cpu();

            let mut desc = GpuAppDescriptor::default();
            desc.slot_id = slot;
            desc.window_id = 100;
            desc.app_type = 5;
            desc.flags = crate::gpu_os::app_descriptor::flags::ACTIVE;
            desc.thread_count = 1024;

            pool.update_slot(slot, &desc);

            let retrieved = pool.get_slot(slot).unwrap();
            assert_eq!(retrieved.window_id, 100);
            assert_eq!(retrieved.app_type, 5);
            assert_eq!(retrieved.thread_count, 1024);
        }
    }

    #[test]
    fn test_active_slots() {
        if let Some(pool) = create_test_pool() {
            // Allocate 3 slots and mark them active
            for i in 0..3 {
                let slot = pool.allocate_cpu();
                let mut desc = GpuAppDescriptor::default();
                desc.slot_id = slot;
                desc.flags = crate::gpu_os::app_descriptor::flags::ACTIVE;
                desc.window_id = (i + 1) * 10;
                pool.update_slot(slot, &desc);
            }

            let active = pool.active_slots();
            assert_eq!(active.len(), 3);

            // Verify window IDs
            let window_ids: Vec<u32> = active.iter().map(|(_, d)| d.window_id).collect();
            assert!(window_ids.contains(&10));
            assert!(window_ids.contains(&20));
            assert!(window_ids.contains(&30));
        }
    }

    #[test]
    fn test_free_invalid_slot() {
        if let Some(pool) = create_test_pool() {
            // Should not crash
            pool.free_cpu(1000);
            pool.free_cpu(INVALID_SLOT);
            assert_eq!(pool.active_count(), 0);
        }
    }

    #[test]
    fn test_lifo_order() {
        if let Some(pool) = create_test_pool() {
            // Allocate 3 slots
            let s1 = pool.allocate_cpu();
            let s2 = pool.allocate_cpu();
            let s3 = pool.allocate_cpu();

            // Free in order: s1, s2, s3
            pool.free_cpu(s1);
            pool.free_cpu(s2);
            pool.free_cpu(s3);

            // Allocations should come back in LIFO order: s3, s2, s1
            assert_eq!(pool.allocate_cpu(), s3);
            assert_eq!(pool.allocate_cpu(), s2);
            assert_eq!(pool.allocate_cpu(), s1);
        }
    }
}
