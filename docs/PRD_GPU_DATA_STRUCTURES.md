# PRD: GPU Data Structures - Vector, HashMap, String, Tree

**Issue**: #167 - GPU Data Structures
**Priority**: Critical (Foundation for GPU-native applications)
**Status**: Rewritten for GPU-First Architecture

---

## THE GPU IS THE COMPUTER

**This is NOT "data structures that run on GPU." This is "data structures designed BY GPU engineers FOR GPU execution."**

### CPU-Centric Thinking (WRONG)
```
CPU: "I need to insert an element"
CPU: dispatch kernel with one element
GPU: one thread does insert
GPU: returns to CPU
CPU: "I need to insert another element"
... repeat 1000 times = 1000 kernel dispatches
```

### GPU-Native Thinking (RIGHT)
```
GPU: persistent kernel running
GPU: sees 1000 insert requests in queue
GPU: 1000 threads each grab one request (parallel)
GPU: parallel prefix sum calculates all offsets in O(log n)
GPU: 1000 threads write simultaneously
GPU: ONE atomic to update length
GPU: continues running, never returns to CPU
```

---

## Core Design Principles

### 1. BATCH EVERYTHING
Never design for single-element operations. Every operation should handle N elements with N threads.

```metal
// WRONG: Single insert
void insert(T value);

// RIGHT: Batch insert with parallel prefix
void insert_batch(T* values, uint count);  // 1000 inserts = 1 dispatch
```

### 2. PARALLEL BY DEFAULT
If an operation can't be parallelized, redesign the data structure.

```metal
// WRONG: Linear probe (sequential, SIMD divergence)
while (table[slot] != EMPTY) { slot = (slot + 1) % capacity; }

// RIGHT: Cuckoo hashing (each thread checks exactly 2 slots, no loop)
uint slot1 = hash1(key) % capacity;
uint slot2 = hash2(key) % capacity;
// Parallel insert via atomic CAS
```

### 3. O(1) LOOKUPS ALWAYS
On GPU, O(log n) is almost as bad as O(n) due to SIMD divergence. Pre-compute everything.

### 4. GPU OWNS ALL STATE
CPU allocates buffer once. GPU manages everything inside it. CPU never reads/writes except for debugging.

### 5. PERSISTENT KERNEL MODEL
Data structures are accessed by a persistent kernel that never returns to CPU. No "dispatch per operation."

---

## Architecture

### Memory Layout

```
GPU Heap Buffer (64MB default):
┌──────────────────────────────────────────────────────────────┐
│ Heap Header (64 bytes)                                        │
│   - free_list_heads[8]: atomic_uint  (size class free lists) │
│   - bump_ptr: atomic_uint                                     │
│   - stats: allocation counts, peak usage                      │
├──────────────────────────────────────────────────────────────┤
│ Request Queue (4KB)                                           │
│   - Allocation requests from GPU threads                      │
│   - Processed by allocator threadgroup                        │
├──────────────────────────────────────────────────────────────┤
│ Free Lists (per size class)                                   │
│   - 64B, 128B, 256B, 512B, 1KB, 4KB, 16KB, 64KB              │
├──────────────────────────────────────────────────────────────┤
│ Data Blocks (bulk of memory)                                  │
│   - Actual storage for vectors, hashmaps, strings, trees     │
│   - 64-byte aligned                                           │
└──────────────────────────────────────────────────────────────┘
```

### Size Classes (Slab Allocation)

Instead of arbitrary-sized allocation (causes fragmentation), use fixed size classes:

| Class | Size | Use Case |
|-------|------|----------|
| 0 | 64B | Small strings, small vectors |
| 1 | 128B | Medium strings |
| 2 | 256B | HashMap entries |
| 3 | 512B | Small arrays |
| 4 | 1KB | Medium arrays |
| 5 | 4KB | Large arrays |
| 6 | 16KB | Very large arrays |
| 7 | 64KB | Huge allocations |

**Why**: Each free list is lock-free (atomic CAS on head). No coalescing needed. O(1) alloc/free.

---

## Component 1: GPU Allocator

### Design: Lock-Free Slab Allocator

```metal
struct HeapHeader {
    // Per-size-class free lists (8 size classes)
    atomic_uint free_list_heads[8];  // Head of each free list
    atomic_uint free_list_counts[8]; // Count per list (stats)

    // Bump allocator for new blocks
    atomic_uint bump_ptr;
    uint heap_size;

    // Stats
    atomic_uint total_allocated;
    atomic_uint allocation_count;

    uint _padding[4];  // Align to 64 bytes
};

// Block header - stored at start of each block
struct BlockHeader {
    uint size_class;      // Which size class (0-7)
    uint next_free;       // Next in free list (if free)
    uint flags;           // ALLOCATED | FREE
    uint _padding;
};
```

### Operations

```metal
// BATCH allocation - allocate N blocks simultaneously
// Uses parallel prefix sum: O(log n) time, 1 atomic per size class
kernel void gpu_alloc_batch(
    device HeapHeader* heap,
    device uchar* heap_data,
    device const AllocationRequest* requests,  // [size_class, count] pairs
    device AllocationResult* results,          // Output: offsets
    constant uint& request_count,
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Group requests by size class
    threadgroup uint class_counts[8];
    threadgroup uint class_prefix[8];

    // Phase 1: Count requests per size class (parallel reduce)
    // Phase 2: Prefix sum to get offsets
    // Phase 3: Single atomic per size class to reserve blocks
    // Phase 4: Each thread gets its block from appropriate list

    // This allocates 1000 blocks with ~8 atomics, not 1000 atomics
}

// BATCH free - free N blocks simultaneously
kernel void gpu_free_batch(
    device HeapHeader* heap,
    device uchar* heap_data,
    device const uint* blocks_to_free,
    constant uint& count,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint block_offset = blocks_to_free[tid];
    device BlockHeader* header = (device BlockHeader*)(heap_data + block_offset);
    uint size_class = header->size_class;

    // Add to free list via CAS
    uint old_head;
    do {
        old_head = atomic_load_explicit(&heap->free_list_heads[size_class], memory_order_relaxed);
        header->next_free = old_head;
    } while (!atomic_compare_exchange_weak_explicit(
        &heap->free_list_heads[size_class], &old_head, block_offset,
        memory_order_release, memory_order_relaxed
    ));
}
```

---

## Component 2: GpuVector<T>

### Design: Parallel-Friendly Resizable Array

```metal
struct GpuVectorHeader {
    atomic_uint len;        // Current element count
    uint capacity;          // Max elements before resize
    uint element_size;      // sizeof(T)
    uint data_offset;       // Offset to data (always sizeof(header))
    atomic_uint pending_pushes; // For batch coordination
    uint _padding[3];
};
```

### Operations

```metal
// BATCH push - push N elements simultaneously
// Uses parallel prefix sum for offset calculation
kernel void gpu_vector_push_batch(
    device uchar* heap_data,
    device uint* vector_ptrs,          // Which vectors to push to
    device const uchar* values,        // Values to push (packed)
    device const uint* value_sizes,    // Size of each value
    constant uint& push_count,
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tid >= push_count) return;

    uint vec_ptr = vector_ptrs[tid];
    device GpuVectorHeader* header = (device GpuVectorHeader*)(heap_data + vec_ptr);

    // Atomically claim a slot
    uint my_index = atomic_fetch_add_explicit(&header->len, 1, memory_order_relaxed);

    if (my_index >= header->capacity) {
        // Need resize - revert and signal
        atomic_fetch_sub_explicit(&header->len, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&header->pending_pushes, 1, memory_order_relaxed);
        return;  // Retry after resize
    }

    // Write value
    uint elem_size = header->element_size;
    device uchar* dst = heap_data + vec_ptr + header->data_offset + my_index * elem_size;
    device const uchar* src = values + tid * elem_size;

    for (uint i = 0; i < elem_size; i++) {
        dst[i] = src[i];
    }
}

// PARALLEL get - each thread reads one element
// No kernel needed - direct memory access:
//   T value = *(T*)(heap_data + vec_ptr + header->data_offset + index * sizeof(T))

// PARALLEL map - apply function to all elements
kernel void gpu_vector_map(
    device uchar* heap_data,
    uint vec_ptr,
    uint tid [[thread_position_in_grid]]
) {
    device GpuVectorHeader* header = (device GpuVectorHeader*)(heap_data + vec_ptr);
    uint len = atomic_load_explicit(&header->len, memory_order_relaxed);

    if (tid >= len) return;

    // Each thread processes one element - perfect parallelism
    device T* elem = (device T*)(heap_data + vec_ptr + header->data_offset + tid * sizeof(T));
    *elem = transform(*elem);
}
```

---

## Component 3: GpuHashMap<K,V>

### Design: Cuckoo Hashing (Parallel Insert)

**Why Cuckoo Hashing?**
- Linear/quadratic probing: variable iterations = SIMD divergence disaster
- Cuckoo hashing: exactly 2 lookups per key, always. Perfect for GPU.

```metal
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

// Two hash functions
inline uint hash1(uint key) {
    // MurmurHash3 finalizer
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

inline uint hash2(uint key) {
    // Different constants
    key ^= key >> 16;
    key *= 0xcc9e2d51;
    key ^= key >> 13;
    key *= 0x1b873593;
    key ^= key >> 16;
    return key;
}
```

### Operations

```metal
// BATCH insert - insert N key-value pairs simultaneously
kernel void gpu_hashmap_insert_batch(
    device uchar* heap_data,
    uint map_ptr,
    device const uint* keys,
    device const uint* values,
    device uint* results,  // 1=success, 0=failed (needs rehash)
    constant uint& count,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    device GpuHashMapHeader* header = (device GpuHashMapHeader*)(heap_data + map_ptr);
    device CuckooEntry* table1 = (device CuckooEntry*)(heap_data + map_ptr + header->table1_offset);
    device CuckooEntry* table2 = (device CuckooEntry*)(heap_data + map_ptr + header->table2_offset);

    uint key = keys[tid];
    uint value = values[tid];
    uint mask = header->capacity - 1;

    uint slot1 = hash1(key) & mask;
    uint slot2 = hash2(key) & mask;

    // Try table1 first
    uint expected = 0;  // EMPTY
    if (atomic_compare_exchange_strong_explicit(
        &table1[slot1].state, &expected, 1,
        memory_order_acq_rel, memory_order_relaxed
    )) {
        table1[slot1].key = key;
        table1[slot1].value = value;
        atomic_fetch_add_explicit(&header->count, 1, memory_order_relaxed);
        results[tid] = 1;
        return;
    }

    // Try table2
    expected = 0;
    if (atomic_compare_exchange_strong_explicit(
        &table2[slot2].state, &expected, 1,
        memory_order_acq_rel, memory_order_relaxed
    )) {
        table2[slot2].key = key;
        table2[slot2].value = value;
        atomic_fetch_add_explicit(&header->count, 1, memory_order_relaxed);
        results[tid] = 1;
        return;
    }

    // Both slots occupied - needs eviction chain (rare)
    // Signal for CPU-assisted rehash or retry
    atomic_fetch_add_explicit(&header->insert_failures, 1, memory_order_relaxed);
    results[tid] = 0;
}

// BATCH lookup - O(1) guaranteed, exactly 2 memory accesses per key
kernel void gpu_hashmap_get_batch(
    device const uchar* heap_data,
    uint map_ptr,
    device const uint* keys,
    device uint* values,
    device uint* found,  // 1=found, 0=not found
    constant uint& count,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    device const GpuHashMapHeader* header = (device const GpuHashMapHeader*)(heap_data + map_ptr);
    device const CuckooEntry* table1 = (device const CuckooEntry*)(heap_data + map_ptr + header->table1_offset);
    device const CuckooEntry* table2 = (device const CuckooEntry*)(heap_data + map_ptr + header->table2_offset);

    uint key = keys[tid];
    uint mask = header->capacity - 1;

    uint slot1 = hash1(key) & mask;
    uint slot2 = hash2(key) & mask;

    // Check table1
    if (atomic_load_explicit(&table1[slot1].state, memory_order_relaxed) == 1 &&
        table1[slot1].key == key) {
        values[tid] = table1[slot1].value;
        found[tid] = 1;
        return;
    }

    // Check table2
    if (atomic_load_explicit(&table2[slot2].state, memory_order_relaxed) == 1 &&
        table2[slot2].key == key) {
        values[tid] = table2[slot2].value;
        found[tid] = 1;
        return;
    }

    found[tid] = 0;
}
```

---

## Component 4: GpuString

### Design: Copy-on-Write with Interning

For GPU, string operations should be:
1. **Comparison**: Parallel character-by-character (N threads for N chars)
2. **Hashing**: Parallel reduction
3. **Storage**: Interned strings (same string = same pointer)

```metal
struct GpuStringHeader {
    uint hash;              // Pre-computed hash for O(1) comparison
    uint len;               // Length in bytes
    uint ref_count;         // For interning (atomic)
    uint flags;             // INTERNED | HEAP | SSO
};

// Small String Optimization: strings <= 24 bytes stored inline
struct GpuString {
    union {
        struct {  // Small string (inline)
            uchar data[24];
            uchar len;      // 0-24
            uchar flags;    // SSO_FLAG set
            ushort _pad;
        } small;
        struct {  // Large string (pointer)
            uint header_offset;  // Offset to GpuStringHeader in heap
            uint hash;           // Cached hash
            uint _reserved[4];
            uchar len_high;      // High bits of length
            uchar flags;         // SSO_FLAG clear
            ushort _pad;
        } large;
    };
};

constant uchar STRING_FLAG_SSO = 0x80;
constant uchar STRING_FLAG_INTERNED = 0x40;
```

### Operations

```metal
// PARALLEL string compare - N threads for N characters
kernel void gpu_string_compare_batch(
    device const uchar* heap_data,
    device const GpuString* strings_a,
    device const GpuString* strings_b,
    device int* results,  // -1, 0, +1
    constant uint& count,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    GpuString a = strings_a[tid];
    GpuString b = strings_b[tid];

    // Fast path: same hash = probably equal
    uint hash_a = (a.small.flags & STRING_FLAG_SSO) ? compute_hash(a.small.data, a.small.len) : a.large.hash;
    uint hash_b = (b.small.flags & STRING_FLAG_SSO) ? compute_hash(b.small.data, b.small.len) : b.large.hash;

    if (hash_a != hash_b) {
        results[tid] = (hash_a < hash_b) ? -1 : 1;
        return;
    }

    // Hashes match - need full comparison
    // For interned strings, same hash + same length = equal (guaranteed unique)
    // Otherwise, compare bytes...
}

// PARALLEL hash computation - reduction across characters
inline uint compute_hash_parallel(
    device const uchar* data,
    uint len,
    uint tid,
    uint threads,
    threadgroup uint* partial_hashes
) {
    // Each thread hashes a chunk
    uint chunk_size = (len + threads - 1) / threads;
    uint start = tid * chunk_size;
    uint end = min(start + chunk_size, len);

    uint local_hash = 2166136261u;  // FNV offset basis
    for (uint i = start; i < end; i++) {
        local_hash ^= data[i];
        local_hash *= 16777619u;
    }

    partial_hashes[tid] = local_hash;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce (combine hashes)
    for (uint stride = threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_hashes[tid] ^= partial_hashes[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return partial_hashes[0];
}
```

---

## Component 5: GpuTree (B+ Tree)

### Design: Bulk-Synchronized Parallel B+ Tree

Traditional B-tree operations are sequential. For GPU, we use **bulk operations**:
- Insert 1000 keys at once, sort them, then bulk-load
- Queries: each thread traverses independently (good parallelism if many queries)

```metal
struct BTreeNode {
    uint key_count;
    uint is_leaf;
    uint parent;
    uint next_leaf;  // For range scans
    uint keys[15];   // B+ tree order 16
    union {
        uint children[16];  // Internal: child pointers
        uint values[15];    // Leaf: values
    };
};

struct GpuTreeHeader {
    uint root;
    atomic_uint node_count;
    atomic_uint entry_count;
    uint height;
    uint free_node_list;  // Free list of nodes
    uint _padding[3];
};
```

### Operations

```metal
// BATCH range query - N threads query N ranges simultaneously
kernel void gpu_tree_range_batch(
    device const uchar* heap_data,
    uint tree_ptr,
    device const uint* low_keys,
    device const uint* high_keys,
    device uint* result_counts,
    device uint* result_values,  // Flattened output
    constant uint& query_count,
    constant uint& max_results_per_query,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= query_count) return;

    // Each thread does its own traversal
    // This is O(log n + k) per thread, but N threads run in parallel
    // Total: O(log n + k) wall-clock time for N queries

    uint low = low_keys[tid];
    uint high = high_keys[tid];
    uint result_offset = tid * max_results_per_query;
    uint count = 0;

    // Find leaf containing low key
    // Scan leaves until > high key
    // ...
}
```

---

## Rust-Side API

**CRITICAL: Rust does NOT manage data structures. Rust only:**
1. Allocates the heap buffer once
2. Binds it to compute encoders
3. Reads stats for debugging

```rust
/// GPU heap - just a buffer wrapper, GPU owns all state
pub struct GpuHeap {
    buffer: Buffer,
    size: usize,
}

impl GpuHeap {
    /// Create heap buffer. GPU will initialize via kernel.
    pub fn new(device: &Device, size: usize) -> Self {
        // Main heap is GPU-only (Private storage)
        let buffer = device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModePrivate,  // GPU-ONLY - no CPU access
        );
        Self { buffer, size }
    }

    /// Get buffer for binding to compute encoder
    pub fn buffer(&self) -> &Buffer { &self.buffer }

    // NOTE: debug_stats() is NOT possible with StorageModePrivate
    // If debug stats are needed, use a separate small Shared buffer
    // that the GPU periodically copies stats into, or use Metal's
    // GPU profiler/debugger tools.
    //
    // The GPU owns all heap state. CPU does not read it during operation.
}

/// For debugging: separate stats buffer (Shared storage, GPU writes periodically)
pub struct GpuHeapStats {
    stats_buffer: Buffer,  // 64 bytes, StorageModeShared
}

impl GpuHeapStats {
    pub fn new(device: &Device) -> Self {
        let stats_buffer = device.new_buffer(
            64,
            MTLResourceOptions::StorageModeShared,  // GPU writes, CPU reads
        );
        Self { stats_buffer }
    }

    /// Read stats (only valid after GPU has written them)
    pub fn read(&self) -> HeapStats {
        unsafe {
            let ptr = self.stats_buffer.contents() as *const HeapStats;
            ptr.read()
        }
    }
}

// NO methods for alloc/free/push/get - those are GPU operations
// Apps use Metal kernels directly or via persistent megakernel
```

---

## Integration with Megakernel

These data structures are used by the persistent megakernel (Issue #154):

```metal
// In megakernel - data structures are just heap offsets
kernel void megakernel(
    device uchar* heap [[buffer(0)]],
    device AppState* apps [[buffer(1)]],
    // ...
) {
    // App's widget list is a GpuVector
    uint widgets_vec = apps[app_id].widgets_offset;

    // Push new widget - direct call, no dispatch needed
    uint index = atomic_fetch_add_explicit(
        &((device GpuVectorHeader*)(heap + widgets_vec))->len,
        1, memory_order_relaxed
    );

    // Write widget data
    device Widget* w = (device Widget*)(heap + widgets_vec + 16 + index * sizeof(Widget));
    *w = new_widget;
}
```

---

## Test Plan

### Correctness Tests

```rust
#[test]
fn test_parallel_vector_push_1m() {
    // Push 1M elements from 1M threads
    // Verify all elements present, no duplicates, no gaps
}

#[test]
fn test_cuckoo_hashmap_collision() {
    // Insert keys that collide in hash1
    // Verify they land in hash2
    // Verify lookup finds them
}

#[test]
fn test_string_interning() {
    // Create same string twice
    // Verify same heap offset (interned)
}
```

### Performance Tests

```rust
#[test]
fn bench_vector_push_throughput() {
    // Target: >100M ops/sec for batch push
}

#[test]
fn bench_hashmap_lookup_throughput() {
    // Target: >500M ops/sec (2 memory accesses per lookup)
}
```

---

## Success Metrics

| Metric | Target | Why |
|--------|--------|-----|
| Batch vector push | >100M ops/sec | Parallel prefix sum |
| HashMap lookup | >500M ops/sec | Exactly 2 memory accesses |
| HashMap insert | >50M ops/sec | Atomic CAS on 2 slots |
| String compare (same hash) | O(1) | Hash pre-computed |
| Allocator batch alloc | 1 atomic per size class | Slab allocation |
| CPU involvement | 0 | GPU owns all state |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu_os/gpu_heap.rs` | Buffer wrapper (Rust) |
| `src/gpu_os/shaders/gpu_data_structures.metal` | All kernels |
| `tests/test_issue_167_data_structures.rs` | Tests |

---

## Anti-Patterns (DO NOT DO)

| Bad Pattern | Why Bad | Good Pattern |
|-------------|---------|--------------|
| Single-element insert | 1 dispatch per insert | Batch insert |
| Linear probing | SIMD divergence | Cuckoo hashing |
| Rust manages state | CPU in loop | GPU owns state |
| CPU reads completion | CPU polling | GPU continues working |
| Arbitrary size alloc | Fragmentation | Size classes |
