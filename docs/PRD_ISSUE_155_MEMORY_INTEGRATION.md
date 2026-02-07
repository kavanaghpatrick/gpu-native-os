# PRD: GPU Memory Management Integration (Issue #155)

## Overview

Implement O(1) GPU-native memory allocation with atomic free list for memory reuse.

## GPU-Native Principles

| CPU Pattern (Wrong) | GPU Pattern (Right) |
|---------------------|---------------------|
| O(log n) prefix sum for single alloc | O(1) atomic free list pop |
| Sequential free list traversal | Atomic exchange in 1 operation |
| Defragmentation pass | Lazy coalescing on free |

## Design

### O(1) Free List (Lock-Free Stack)

```metal
struct FreeBlock {
    uint next;      // Index of next free block (forms linked list)
    uint size;      // Size of this block
    uint offset;    // Offset in memory pool
    uint _pad;
};

struct MemoryPool {
    atomic_uint freelist_head;    // Head of free list (LIFO stack)
    atomic_uint bump_pointer;     // Fallback bump allocator
    atomic_uint free_count;       // Number of free blocks
    uint pool_size;
    uint block_count;             // Total blocks in freelist array
    uint _pad[3];
};

// O(1) ALLOCATE - atomic pop from free list
inline uint gpu_alloc_o1(
    device MemoryPool* pool,
    device FreeBlock* blocks,
    uint size
) {
    // Try free list first - O(1) atomic pop
    uint head = atomic_load_explicit(&pool->freelist_head, memory_order_acquire);

    while (head != INVALID_SLOT) {
        FreeBlock block = blocks[head];

        // Check size fits
        if (block.size >= size) {
            // Try to pop this block
            if (atomic_compare_exchange_weak_explicit(
                &pool->freelist_head,
                &head,
                block.next,
                memory_order_release,
                memory_order_relaxed
            )) {
                atomic_fetch_sub_explicit(&pool->free_count, 1, memory_order_relaxed);
                return block.offset;
            }
            // CAS failed, head updated, retry
        } else {
            // Block too small, this is a limitation of LIFO
            // For MVP, fall through to bump allocator
            break;
        }
    }

    // Fallback: O(1) bump allocation
    uint aligned_size = (size + 15) & ~15;
    uint offset = atomic_fetch_add_explicit(&pool->bump_pointer, aligned_size, memory_order_relaxed);

    if (offset + aligned_size > pool->pool_size) {
        // OOM - rollback
        atomic_fetch_sub_explicit(&pool->bump_pointer, aligned_size, memory_order_relaxed);
        return INVALID_SLOT;
    }

    return offset;
}

// O(1) FREE - atomic push to free list
inline void gpu_free_o1(
    device MemoryPool* pool,
    device FreeBlock* blocks,
    uint offset,
    uint size
) {
    // Allocate a block descriptor
    uint block_idx = atomic_fetch_add_explicit(&pool->block_count, 1, memory_order_relaxed);

    blocks[block_idx].offset = offset;
    blocks[block_idx].size = size;

    // O(1) atomic push to head
    uint old_head = atomic_load_explicit(&pool->freelist_head, memory_order_relaxed);
    do {
        blocks[block_idx].next = old_head;
    } while (!atomic_compare_exchange_weak_explicit(
        &pool->freelist_head,
        &old_head,
        block_idx,
        memory_order_release,
        memory_order_relaxed
    ));

    atomic_fetch_add_explicit(&pool->free_count, 1, memory_order_relaxed);
}
```

### Why O(1) Beats O(log n) Here

```
App launch rate: ~1-10 per second
Parallel prefix sum: 10 barriers for 1024 batch
Free list atomic: 1 CAS operation

For single allocations:
  Prefix sum: ~1000 cycles (barriers + memory traffic)
  Free list:  ~50 cycles (one atomic CAS)

GPU utilization:
  Prefix sum: 1024 threads wait at each barrier
  Free list:  1 thread does CAS, others do useful work
```

### Memory Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ Memory Pool (64 MB)                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ [App 0 State][App 1 State][FREE][App 3 State][FREE][...]           │
│                              ↓                   ↓                   │
│                         freelist[0]         freelist[1]              │
│                         next→[1]            next→INVALID             │
└─────────────────────────────────────────────────────────────────────┘

Free list is LIFO stack:
  head → block[0] → block[1] → INVALID

Alloc: pop head in O(1)
Free:  push to head in O(1)
```

### Updated App Close with Memory Free

```metal
kernel void gpu_close_app(
    device AppTableHeader* header [[buffer(0)]],
    device GpuAppDescriptor* apps [[buffer(1)]],
    device MemoryPool* state_pool [[buffer(2)]],
    device FreeBlock* state_blocks [[buffer(3)]],
    device MemoryPool* vertex_pool [[buffer(4)]],
    device FreeBlock* vertex_blocks [[buffer(5)]],
    constant uint& slot_id [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    if (slot_id >= header->max_slots) return;

    GpuAppDescriptor app = apps[slot_id];
    if (!(app.flags & APP_FLAG_ACTIVE)) return;

    // Free memory - O(1) each
    if (app.state_size > 0) {
        gpu_free_o1(state_pool, state_blocks, app.state_offset, app.state_size);
    }
    if (app.vertex_size > 0) {
        gpu_free_o1(vertex_pool, vertex_blocks, app.vertex_offset, app.vertex_size);
    }

    // Clear descriptor
    apps[slot_id].flags = 0;

    // Free slot - O(1)
    free_slot(header, slot_id);
}
```

## Implementation

### Rust Structures

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeBlock {
    pub next: u32,
    pub size: u32,
    pub offset: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemoryPool {
    pub freelist_head: u32,  // atomic
    pub bump_pointer: u32,   // atomic
    pub free_count: u32,     // atomic
    pub pool_size: u32,
    pub block_count: u32,    // atomic
    pub _pad: [u32; 3],
}
```

## Tests

```rust
#[test]
fn test_alloc_free_reuse() {
    let mut system = GpuAppSystem::new(&device)?;

    // Launch app, note memory offset
    let slot1 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    let offset1 = system.get_app(slot1).unwrap().state_offset;

    // Close app - memory goes to free list
    system.close_app(slot1);

    // Launch new app - should reuse freed memory
    let slot2 = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    let offset2 = system.get_app(slot2).unwrap().state_offset;

    assert_eq!(offset1, offset2, "Should reuse freed memory via O(1) free list");
}

#[test]
fn test_free_list_lifo() {
    let mut system = GpuAppSystem::new(&device)?;

    // Launch 3 apps
    let a = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    let b = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    let c = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();

    let offset_c = system.get_app(c).unwrap().state_offset;

    // Close C, B, A (reverse order)
    system.close_app(c);
    system.close_app(b);
    system.close_app(a);

    // Launch new app - should get C's memory (LIFO)
    let d = system.launch_app(app_type::CUSTOM, 1024, 512).unwrap();
    assert_eq!(system.get_app(d).unwrap().state_offset, offset_c);
}

#[test]
fn test_memory_stats() {
    let mut system = GpuAppSystem::new(&device)?;

    // Launch apps
    for _ in 0..10 {
        system.launch_app(app_type::CUSTOM, 1024, 512);
    }

    let stats = system.memory_stats();
    assert_eq!(stats.free_block_count, 0);
    assert!(stats.bump_pointer > 0);

    // Close half
    for i in 0..5 {
        system.close_app(i);
    }

    let stats = system.memory_stats();
    assert_eq!(stats.free_block_count, 5);
}
```

## Success Metrics

1. **Allocation time**: O(1) - single atomic CAS (~50 cycles)
2. **Free time**: O(1) - single atomic CAS (~50 cycles)
3. **Memory reuse**: >90% of freed memory reused
4. **No barriers**: Zero synchronization barriers in alloc/free path
