// GPU Data Structures - Issue #167
//
// THE GPU IS THE COMPUTER. All data structure operations run on GPU.
//
// Design principles:
// - Batch everything (N threads for N operations)
// - Cuckoo hashing (O(1) guaranteed, no SIMD divergence)
// - Slab allocation (size classes, lock-free free lists)
// - GPU owns all state (CPU only allocates buffers)

#include <metal_stdlib>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

constant uint SIZE_CLASS_COUNT = 8;
constant uint SIZE_CLASSES[8] = {64, 128, 256, 512, 1024, 4096, 16384, 65536};

constant uint BLOCK_FREE = 0;
constant uint BLOCK_ALLOCATED = 1;

constant uint CUCKOO_EMPTY = 0;
constant uint CUCKOO_OCCUPIED = 1;
constant uint CUCKOO_INSERTING = 2;
constant uint CUCKOO_UPDATING = 3;  // Issue #252 fix: state for atomic value updates

constant uint INVALID_OFFSET = 0xFFFFFFFF;

// ═══════════════════════════════════════════════════════════════════════════
// HEAP STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

struct HeapHeader {
    // Per-size-class free lists (8 size classes)
    atomic_uint free_list_heads[8];  // Head offset of each free list
    atomic_uint free_list_counts[8]; // Count per list (stats)

    // Bump allocator for new blocks
    atomic_uint bump_ptr;
    uint heap_size;

    // Stats
    atomic_uint total_allocated;
    atomic_uint allocation_count;

    uint _padding[4];  // Align to 64 bytes
};

struct BlockHeader {
    uint size_class;      // Which size class (0-7)
    uint next_free;       // Next in free list (if free)
    uint flags;           // BLOCK_FREE | BLOCK_ALLOCATED
    uint _padding;
};

// ═══════════════════════════════════════════════════════════════════════════
// VECTOR STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

struct GpuVectorHeader {
    atomic_uint len;        // Current element count
    uint capacity;          // Max elements before resize
    uint element_size;      // sizeof(T)
    uint data_offset;       // Offset to data (after header)
    atomic_uint pending_pushes; // For resize signaling
    uint _padding[3];
};

// ═══════════════════════════════════════════════════════════════════════════
// HASHMAP STRUCTURES (CUCKOO HASHING)
// ═══════════════════════════════════════════════════════════════════════════

struct CuckooEntry {
    atomic_uint state;  // EMPTY=0, OCCUPIED=1, INSERTING=2
    uint key;
    uint value;
    uint _padding;
};

struct GpuHashMapHeader {
    atomic_uint count;
    uint capacity;          // Must be power of 2
    uint table1_offset;     // Offset to first hash table
    uint table2_offset;     // Offset to second hash table
    atomic_uint insert_failures; // Count of items needing rehash
    uint _padding[3];
};

// ═══════════════════════════════════════════════════════════════════════════
// STRING STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

constant uchar STRING_FLAG_SSO = 0x80;
constant uchar STRING_FLAG_HEAP = 0x40;
constant uint SSO_MAX_LEN = 23;

struct GpuString {
    uchar data[24];     // Small string data OR heap offset (first 4 bytes)
    uchar len;          // Length (0-23 for SSO, or high bits for heap)
    uchar flags;        // SSO_FLAG | HEAP_FLAG
    uchar _pad[2];
    uint hash;          // Pre-computed hash
};

// ═══════════════════════════════════════════════════════════════════════════
// HASH FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

// MurmurHash3 finalizer - hash function 1
inline uint hash1(uint key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

// Different constants - hash function 2
inline uint hash2(uint key) {
    key ^= key >> 16;
    key *= 0xcc9e2d51;
    key ^= key >> 13;
    key *= 0x1b873593;
    key ^= key >> 16;
    return key;
}

// FNV-1a hash for strings
inline uint fnv1a_hash(device const uchar* data, uint len) {
    uint hash = 2166136261u;
    for (uint i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619u;
    }
    return hash;
}

// ═══════════════════════════════════════════════════════════════════════════
// SIZE CLASS HELPERS
// ═══════════════════════════════════════════════════════════════════════════

// Get size class for a given allocation size
inline uint get_size_class(uint size) {
    // Include block header in size
    uint total = size + sizeof(BlockHeader);

    for (uint i = 0; i < SIZE_CLASS_COUNT; i++) {
        if (total <= SIZE_CLASSES[i]) {
            return i;
        }
    }
    return SIZE_CLASS_COUNT - 1; // Largest class
}

inline uint get_class_size(uint size_class) {
    return SIZE_CLASSES[min(size_class, SIZE_CLASS_COUNT - 1)];
}

// ═══════════════════════════════════════════════════════════════════════════
// HEAP INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

kernel void heap_init(
    device HeapHeader* heap [[buffer(0)]],
    constant uint& heap_size [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Initialize free lists to empty
    for (uint i = 0; i < SIZE_CLASS_COUNT; i++) {
        atomic_store_explicit(&heap->free_list_heads[i], INVALID_OFFSET, memory_order_relaxed);
        atomic_store_explicit(&heap->free_list_counts[i], 0, memory_order_relaxed);
    }

    // Bump pointer starts after header
    atomic_store_explicit(&heap->bump_ptr, sizeof(HeapHeader), memory_order_relaxed);
    heap->heap_size = heap_size;

    atomic_store_explicit(&heap->total_allocated, 0, memory_order_relaxed);
    atomic_store_explicit(&heap->allocation_count, 0, memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// HEAP ALLOCATION (BATCH)
// ═══════════════════════════════════════════════════════════════════════════

// Allocate a single block (called per-thread in batch)
inline uint alloc_block(
    device HeapHeader* heap,
    device uchar* heap_data,
    uint size_class
) {
    uint class_size = get_class_size(size_class);

    // Try free list first (bounded retry to prevent SIMD divergence/livelock)
    // Max 64 attempts prevents indefinite spinning under high contention
    constant uint MAX_CAS_RETRIES = 64;
    uint head = atomic_load_explicit(&heap->free_list_heads[size_class], memory_order_relaxed);
    for (uint retry = 0; retry < MAX_CAS_RETRIES && head != INVALID_OFFSET; retry++) {
        device BlockHeader* block = (device BlockHeader*)(heap_data + head);
        uint next = block->next_free;

        // Try to CAS the head
        if (atomic_compare_exchange_weak_explicit(
            &heap->free_list_heads[size_class],
            &head, next,
            memory_order_relaxed, memory_order_relaxed
        )) {
            // Got a block from free list
            block->flags = BLOCK_ALLOCATED;
            atomic_fetch_sub_explicit(&heap->free_list_counts[size_class], 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&heap->total_allocated, class_size, memory_order_relaxed);
            atomic_fetch_add_explicit(&heap->allocation_count, 1, memory_order_relaxed);
            return head + sizeof(BlockHeader); // Return data offset
        }
        // CAS failed, reload head and retry
        head = atomic_load_explicit(&heap->free_list_heads[size_class], memory_order_relaxed);
    }
    // Free list exhausted or max retries hit - fall through to bump allocator

    // Free list empty, bump allocate
    uint offset = atomic_fetch_add_explicit(&heap->bump_ptr, class_size, memory_order_relaxed);

    if (offset + class_size > heap->heap_size) {
        // Out of memory - revert bump pointer
        atomic_fetch_sub_explicit(&heap->bump_ptr, class_size, memory_order_relaxed);
        return INVALID_OFFSET;
    }

    // Initialize block header
    device BlockHeader* block = (device BlockHeader*)(heap_data + offset);
    block->size_class = size_class;
    block->next_free = INVALID_OFFSET;
    block->flags = BLOCK_ALLOCATED;

    atomic_fetch_add_explicit(&heap->total_allocated, class_size, memory_order_relaxed);
    atomic_fetch_add_explicit(&heap->allocation_count, 1, memory_order_relaxed);

    return offset + sizeof(BlockHeader);
}

// Batch allocation - each thread allocates one block
kernel void heap_alloc_batch(
    device HeapHeader* heap [[buffer(0)]],
    device uchar* heap_data [[buffer(1)]],
    device const uint* sizes [[buffer(2)]],
    device uint* results [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint size = sizes[tid];
    uint size_class = get_size_class(size);

    results[tid] = alloc_block(heap, heap_data, size_class);
}

// ═══════════════════════════════════════════════════════════════════════════
// HEAP FREE (BATCH)
// ═══════════════════════════════════════════════════════════════════════════

kernel void heap_free_batch(
    device HeapHeader* heap [[buffer(0)]],
    device uchar* heap_data [[buffer(1)]],
    device const uint* offsets [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint data_offset = offsets[tid];
    if (data_offset == INVALID_OFFSET) return;

    // Get block header (before data)
    uint block_offset = data_offset - sizeof(BlockHeader);
    device BlockHeader* block = (device BlockHeader*)(heap_data + block_offset);

    if (block->flags != BLOCK_ALLOCATED) return; // Already free

    uint size_class = block->size_class;
    uint class_size = get_class_size(size_class);

    block->flags = BLOCK_FREE;

    // Add to free list via CAS
    // Issue #265 fix: Limit retries to avoid potential infinite loop
    uint old_head;
    for (int retry = 0; retry < 100; retry++) {
        old_head = atomic_load_explicit(&heap->free_list_heads[size_class], memory_order_relaxed);
        block->next_free = old_head;
        if (atomic_compare_exchange_weak_explicit(
            &heap->free_list_heads[size_class],
            &old_head, block_offset,
            memory_order_relaxed, memory_order_relaxed
        )) {
            break;
        }
    }

    atomic_fetch_add_explicit(&heap->free_list_counts[size_class], 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(&heap->total_allocated, class_size, memory_order_relaxed);
    atomic_fetch_sub_explicit(&heap->allocation_count, 1, memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// VECTOR OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

// Initialize a vector at given heap offset
kernel void vector_init(
    device uchar* heap_data [[buffer(0)]],
    device const uint* vec_offsets [[buffer(1)]],
    device const uint* capacities [[buffer(2)]],
    device const uint* element_sizes [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint offset = vec_offsets[tid];
    if (offset == INVALID_OFFSET) return;

    device GpuVectorHeader* header = (device GpuVectorHeader*)(heap_data + offset);
    atomic_store_explicit(&header->len, 0, memory_order_relaxed);
    header->capacity = capacities[tid];
    header->element_size = element_sizes[tid];
    header->data_offset = sizeof(GpuVectorHeader);
    atomic_store_explicit(&header->pending_pushes, 0, memory_order_relaxed);
}

// Batch push - each thread pushes one element
kernel void vector_push_batch(
    device uchar* heap_data [[buffer(0)]],
    device const uint* vec_offsets [[buffer(1)]],
    device const uchar* values [[buffer(2)]],
    device const uint* value_offsets [[buffer(3)]],
    device uint* results [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint vec_offset = vec_offsets[tid];
    device GpuVectorHeader* header = (device GpuVectorHeader*)(heap_data + vec_offset);

    // Atomically claim a slot
    uint my_index = atomic_fetch_add_explicit(&header->len, 1, memory_order_relaxed);

    if (my_index >= header->capacity) {
        // Overflow - revert and signal resize needed
        atomic_fetch_sub_explicit(&header->len, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&header->pending_pushes, 1, memory_order_relaxed);
        results[tid] = INVALID_OFFSET;
        return;
    }

    // Copy value to slot
    uint elem_size = header->element_size;
    device uchar* dst = heap_data + vec_offset + header->data_offset + my_index * elem_size;
    device const uchar* src = values + value_offsets[tid];

    for (uint i = 0; i < elem_size; i++) {
        dst[i] = src[i];
    }

    results[tid] = my_index;
}

// Get vector length
inline uint vector_len(device const uchar* heap_data, uint vec_offset) {
    device const GpuVectorHeader* header = (device const GpuVectorHeader*)(heap_data + vec_offset);
    return atomic_load_explicit(&header->len, memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// HASHMAP OPERATIONS (CUCKOO HASHING)
// ═══════════════════════════════════════════════════════════════════════════

// Initialize a hashmap at given heap offset
// Issue #269 fix: Added heap_size parameter for bounds checking
kernel void hashmap_init(
    device uchar* heap_data [[buffer(0)]],
    device const uint* map_offsets [[buffer(1)]],
    device const uint* capacities [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& heap_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint offset = map_offsets[tid];
    if (offset == INVALID_OFFSET) return;

    uint capacity = capacities[tid];

    // Issue #269: Bounds check - ensure hash tables fit within heap
    uint table1_size = capacity * sizeof(CuckooEntry);
    uint table2_size = capacity * sizeof(CuckooEntry);
    uint required_size = sizeof(GpuHashMapHeader) + table1_size + table2_size;

    // Check for overflow and bounds
    if (required_size < sizeof(GpuHashMapHeader)) return;  // Overflow
    if (offset > heap_size) return;  // Invalid offset
    if (offset + required_size > heap_size) return;  // Doesn't fit

    device GpuHashMapHeader* header = (device GpuHashMapHeader*)(heap_data + offset);
    atomic_store_explicit(&header->count, 0, memory_order_relaxed);
    header->capacity = capacity;
    header->table1_offset = sizeof(GpuHashMapHeader);
    header->table2_offset = sizeof(GpuHashMapHeader) + capacity * sizeof(CuckooEntry);
    atomic_store_explicit(&header->insert_failures, 0, memory_order_relaxed);

    // Zero out both tables
    device CuckooEntry* table1 = (device CuckooEntry*)(heap_data + offset + header->table1_offset);
    device CuckooEntry* table2 = (device CuckooEntry*)(heap_data + offset + header->table2_offset);

    // Each thread clears one entry (launch with capacity threads)
    // For init, tid indexes into entries
    if (tid < capacity) {
        atomic_store_explicit(&table1[tid].state, CUCKOO_EMPTY, memory_order_relaxed);
        table1[tid].key = 0;
        table1[tid].value = 0;

        atomic_store_explicit(&table2[tid].state, CUCKOO_EMPTY, memory_order_relaxed);
        table2[tid].key = 0;
        table2[tid].value = 0;
    }
}

// Batch insert - each thread inserts one key-value pair
// Issue #293 fix: Write key/value BEFORE setting state to OCCUPIED to prevent
// race condition where concurrent lookups see OCCUPIED but read garbage data.
kernel void hashmap_insert_batch(
    device uchar* heap_data [[buffer(0)]],
    constant uint& map_offset [[buffer(1)]],
    device const uint* keys [[buffer(2)]],
    device const uint* values [[buffer(3)]],
    device uint* results [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    device GpuHashMapHeader* header = (device GpuHashMapHeader*)(heap_data + map_offset);
    device CuckooEntry* table1 = (device CuckooEntry*)(heap_data + map_offset + header->table1_offset);
    device CuckooEntry* table2 = (device CuckooEntry*)(heap_data + map_offset + header->table2_offset);

    uint key = keys[tid];
    uint value = values[tid];
    uint mask = header->capacity - 1;

    uint slot1 = hash1(key) & mask;
    uint slot2 = hash2(key) & mask;

    // Try table1 first
    // Issue #293 fix: Use INSERTING state to prevent race conditions.
    // CAS from EMPTY to INSERTING first, then write data, then set OCCUPIED.
    uint expected = CUCKOO_EMPTY;
    if (atomic_compare_exchange_weak_explicit(
        &table1[slot1].state, &expected, CUCKOO_INSERTING,
        memory_order_relaxed, memory_order_relaxed
    )) {
        // Success - we own the slot exclusively during INSERTING state
        table1[slot1].key = key;
        table1[slot1].value = value;
        threadgroup_barrier(mem_flags::mem_device);  // Ensure writes visible
        atomic_store_explicit(&table1[slot1].state, CUCKOO_OCCUPIED, memory_order_relaxed);
        atomic_fetch_add_explicit(&header->count, 1, memory_order_relaxed);
        results[tid] = 1;
        return;
    }
    // CAS failed - slot is not empty

    // Check if key already exists in table1
    // Issue #252 fix: Use CAS to acquire slot for atomic update
    expected = CUCKOO_OCCUPIED;
    if (atomic_compare_exchange_weak_explicit(
        &table1[slot1].state, &expected, CUCKOO_UPDATING,
        memory_order_relaxed, memory_order_relaxed
    )) {
        if (table1[slot1].key == key) {
            // Update existing - we own the slot
            table1[slot1].value = value;
            threadgroup_barrier(mem_flags::mem_device);  // Ensure value visible before state
            atomic_store_explicit(&table1[slot1].state, CUCKOO_OCCUPIED, memory_order_relaxed);
            results[tid] = 1;
            return;
        }
        // Wrong key - release slot back to OCCUPIED
        atomic_store_explicit(&table1[slot1].state, CUCKOO_OCCUPIED, memory_order_relaxed);
    }

    // Try table2
    // Issue #293 fix: Same pattern - CAS to INSERTING, write data, then set OCCUPIED
    expected = CUCKOO_EMPTY;
    if (atomic_compare_exchange_weak_explicit(
        &table2[slot2].state, &expected, CUCKOO_INSERTING,
        memory_order_relaxed, memory_order_relaxed
    )) {
        // Success - we own the slot exclusively during INSERTING state
        table2[slot2].key = key;
        table2[slot2].value = value;
        threadgroup_barrier(mem_flags::mem_device);  // Ensure writes visible
        atomic_store_explicit(&table2[slot2].state, CUCKOO_OCCUPIED, memory_order_relaxed);
        atomic_fetch_add_explicit(&header->count, 1, memory_order_relaxed);
        results[tid] = 1;
        return;
    }
    // CAS failed - slot is not empty

    // Check if key already exists in table2
    // Issue #252 fix: Use CAS to acquire slot for atomic update
    expected = CUCKOO_OCCUPIED;
    if (atomic_compare_exchange_weak_explicit(
        &table2[slot2].state, &expected, CUCKOO_UPDATING,
        memory_order_relaxed, memory_order_relaxed
    )) {
        if (table2[slot2].key == key) {
            // Update existing - we own the slot
            table2[slot2].value = value;
            threadgroup_barrier(mem_flags::mem_device);  // Ensure value visible before state
            atomic_store_explicit(&table2[slot2].state, CUCKOO_OCCUPIED, memory_order_relaxed);
            results[tid] = 1;
            return;
        }
        // Wrong key - release slot back to OCCUPIED
        atomic_store_explicit(&table2[slot2].state, CUCKOO_OCCUPIED, memory_order_relaxed);
    }

    // Both slots occupied by different keys - insertion failed
    atomic_fetch_add_explicit(&header->insert_failures, 1, memory_order_relaxed);
    results[tid] = 0; // Failed - needs rehash or eviction
}

// Batch lookup - each thread looks up one key
kernel void hashmap_get_batch(
    device const uchar* heap_data [[buffer(0)]],
    constant uint& map_offset [[buffer(1)]],
    device const uint* keys [[buffer(2)]],
    device uint* values [[buffer(3)]],
    device uint* found [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    device const GpuHashMapHeader* header = (device const GpuHashMapHeader*)(heap_data + map_offset);
    device const CuckooEntry* table1 = (device const CuckooEntry*)(heap_data + map_offset + header->table1_offset);
    device const CuckooEntry* table2 = (device const CuckooEntry*)(heap_data + map_offset + header->table2_offset);

    uint key = keys[tid];
    uint mask = header->capacity - 1;

    uint slot1 = hash1(key) & mask;
    uint slot2 = hash2(key) & mask;

    // Check table1
    // Issue #252 fix: Accept UPDATING state (entry exists, being modified)
    uint state1 = atomic_load_explicit(&table1[slot1].state, memory_order_relaxed);
    if ((state1 == CUCKOO_OCCUPIED || state1 == CUCKOO_UPDATING) &&
        table1[slot1].key == key) {
        values[tid] = table1[slot1].value;
        found[tid] = 1;
        return;
    }

    // Check table2
    // Issue #252 fix: Accept UPDATING state (entry exists, being modified)
    uint state2 = atomic_load_explicit(&table2[slot2].state, memory_order_relaxed);
    if ((state2 == CUCKOO_OCCUPIED || state2 == CUCKOO_UPDATING) &&
        table2[slot2].key == key) {
        values[tid] = table2[slot2].value;
        found[tid] = 1;
        return;
    }

    found[tid] = 0;
}

// Batch remove - each thread removes one key
kernel void hashmap_remove_batch(
    device uchar* heap_data [[buffer(0)]],
    constant uint& map_offset [[buffer(1)]],
    device const uint* keys [[buffer(2)]],
    device uint* removed [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    device GpuHashMapHeader* header = (device GpuHashMapHeader*)(heap_data + map_offset);
    device CuckooEntry* table1 = (device CuckooEntry*)(heap_data + map_offset + header->table1_offset);
    device CuckooEntry* table2 = (device CuckooEntry*)(heap_data + map_offset + header->table2_offset);

    uint key = keys[tid];
    uint mask = header->capacity - 1;

    uint slot1 = hash1(key) & mask;
    uint slot2 = hash2(key) & mask;

    // Check table1
    // Issue #252 fix: Use CAS to acquire slot for atomic remove
    uint expected = CUCKOO_OCCUPIED;
    if (atomic_compare_exchange_weak_explicit(
        &table1[slot1].state, &expected, CUCKOO_UPDATING,
        memory_order_relaxed, memory_order_relaxed
    )) {
        if (table1[slot1].key == key) {
            // Remove - we own the slot
            atomic_store_explicit(&table1[slot1].state, CUCKOO_EMPTY, memory_order_relaxed);
            atomic_fetch_sub_explicit(&header->count, 1, memory_order_relaxed);
            removed[tid] = 1;
            return;
        }
        // Wrong key - release slot back to OCCUPIED
        atomic_store_explicit(&table1[slot1].state, CUCKOO_OCCUPIED, memory_order_relaxed);
    }

    // Check table2
    // Issue #252 fix: Use CAS to acquire slot for atomic remove
    expected = CUCKOO_OCCUPIED;
    if (atomic_compare_exchange_weak_explicit(
        &table2[slot2].state, &expected, CUCKOO_UPDATING,
        memory_order_relaxed, memory_order_relaxed
    )) {
        if (table2[slot2].key == key) {
            // Remove - we own the slot
            atomic_store_explicit(&table2[slot2].state, CUCKOO_EMPTY, memory_order_relaxed);
            atomic_fetch_sub_explicit(&header->count, 1, memory_order_relaxed);
            removed[tid] = 1;
            return;
        }
        // Wrong key - release slot back to OCCUPIED
        atomic_store_explicit(&table2[slot2].state, CUCKOO_OCCUPIED, memory_order_relaxed);
    }

    removed[tid] = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// STRING OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

// Create strings from raw bytes (batch)
kernel void string_create_batch(
    device GpuString* strings [[buffer(0)]],
    device const uchar* src_data [[buffer(1)]],
    device const uint* src_offsets [[buffer(2)]],
    device const uint* src_lengths [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint offset = src_offsets[tid];
    uint len = src_lengths[tid];
    device const uchar* src = src_data + offset;

    device GpuString* s = &strings[tid];

    if (len <= SSO_MAX_LEN) {
        // Small string optimization
        for (uint i = 0; i < len; i++) {
            s->data[i] = src[i];
        }
        for (uint i = len; i < 24; i++) {
            s->data[i] = 0;
        }
        s->len = (uchar)len;
        s->flags = STRING_FLAG_SSO;
    } else {
        // Too large for SSO - would need heap allocation
        // For now, truncate to SSO_MAX_LEN
        for (uint i = 0; i < SSO_MAX_LEN; i++) {
            s->data[i] = src[i];
        }
        s->data[SSO_MAX_LEN] = 0;
        s->len = SSO_MAX_LEN;
        s->flags = STRING_FLAG_SSO;
        len = SSO_MAX_LEN;
    }

    // Compute hash
    s->hash = fnv1a_hash(s->data, len);
}

// Compare strings (batch) - returns -1, 0, or 1
kernel void string_compare_batch(
    device const GpuString* strings_a [[buffer(0)]],
    device const GpuString* strings_b [[buffer(1)]],
    device int* results [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    GpuString a = strings_a[tid];
    GpuString b = strings_b[tid];

    // Fast path: different hashes means different strings
    if (a.hash != b.hash) {
        results[tid] = (a.hash < b.hash) ? -1 : 1;
        return;
    }

    // Hashes match - compare lengths
    uint len_a = a.len;
    uint len_b = b.len;

    if (len_a != len_b) {
        results[tid] = (len_a < len_b) ? -1 : 1;
        return;
    }

    // Same length, same hash - compare bytes
    for (uint i = 0; i < len_a; i++) {
        if (a.data[i] != b.data[i]) {
            results[tid] = (a.data[i] < b.data[i]) ? -1 : 1;
            return;
        }
    }

    results[tid] = 0; // Equal
}

// Hash strings (batch) - for when hash needs recalculation
kernel void string_hash_batch(
    device GpuString* strings [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    device GpuString* s = &strings[tid];
    uint len = s->len;

    s->hash = fnv1a_hash(s->data, len);
}
