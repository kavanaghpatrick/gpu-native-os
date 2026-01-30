# PRD: GPU-Native HashMap Implementation

## Overview

Implement a GPU-native HashMap for `gpu_std` using Cuckoo hashing with O(1) guaranteed lookups. This is critical for enabling Rust programs that use `HashMap` to run on GPU.

## Why Cuckoo Hashing?

Traditional hash table approaches don't work well on GPU:

| Approach | Issue on GPU |
|----------|--------------|
| Linear probing | Variable iterations = SIMD divergence (32 threads must wait for slowest) |
| Chaining | Pointer chasing = high latency, cache misses |
| Robin Hood | Variable work per insert = SIMD divergence |

**Cuckoo hashing is ideal because:**
- Exactly 2 lookups per key, always (no SIMD divergence)
- No pointers (two contiguous arrays)
- Lock-free concurrent reads
- Predictable performance

## Architecture

### Bucketed Cuckoo Hashing

```
Table 1 (hash1)                  Table 2 (hash2)
┌─────────────────────────┐     ┌─────────────────────────┐
│ Bucket 0: 16 entries    │     │ Bucket 0: 16 entries    │
├─────────────────────────┤     ├─────────────────────────┤
│ Bucket 1: 16 entries    │     │ Bucket 1: 16 entries    │
├─────────────────────────┤     ├─────────────────────────┤
│ ...                     │     │ ...                     │
└─────────────────────────┘     └─────────────────────────┘
```

### Why 16 Entries Per Bucket?

- Apple Silicon cache line = 128 bytes
- 16 entries × 8 bytes (key + value) = 128 bytes = 1 cache line
- Reading a bucket = 1 memory transaction
- Better load factor (0.95+) compared to single-entry buckets

### Memory Layout

```metal
struct CuckooEntry {
    uint32_t key;    // 4 bytes
    uint32_t value;  // 4 bytes
};  // 8 bytes total

struct alignas(128) CuckooBucket {
    CuckooEntry entries[16];
};  // 128 bytes = 1 cache line
```

### Hash Functions

**For integer keys:** PCG hash (fast, good distribution)
```metal
uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
```

**Two hash functions for Cuckoo:**
```metal
uint hash1(uint key) { return pcg_hash(key); }
uint hash2(uint key) { return pcg_hash(key ^ 0xDEADBEEF); }
```

### Insert Algorithm

```
insert(key, value):
    bucket1 = hash1(key) % num_buckets
    bucket2 = hash2(key) % num_buckets

    // Try table 1
    if table1[bucket1] has empty slot:
        insert into table1[bucket1]
        return success

    // Try table 2
    if table2[bucket2] has empty slot:
        insert into table2[bucket2]
        return success

    // Eviction chain (bounded to 4*log2(capacity))
    for i in 0..max_evictions:
        // Pick random entry from bucket1
        victim = random_entry(table1[bucket1])
        swap(key, value, victim.key, victim.value)

        // Try to place victim in its alternate location
        alt_bucket = hash2(victim.key) % num_buckets
        if table2[alt_bucket] has empty slot:
            insert into table2[alt_bucket]
            return success

        // Continue evicting from table2
        swap bucket/table references

    // Eviction limit reached - use stash or resize
    return add_to_stash(key, value)
```

### Lookup Algorithm (O(1) Guaranteed)

```metal
bool lookup(uint key, uint* value_out) {
    uint bucket1 = hash1(key) % num_buckets;
    uint bucket2 = hash2(key) % num_buckets;

    // Check table 1 bucket (16 entries, parallel scan)
    for (uint i = 0; i < 16; i++) {
        if (table1[bucket1].entries[i].key == key) {
            *value_out = table1[bucket1].entries[i].value;
            return true;
        }
    }

    // Check table 2 bucket
    for (uint i = 0; i < 16; i++) {
        if (table2[bucket2].entries[i].key == key) {
            *value_out = table2[bucket2].entries[i].value;
            return true;
        }
    }

    // Check stash (small overflow buffer)
    return check_stash(key, value_out);
}
```

## Rust API

```rust
// In gpu_std/src/collections/hashmap.rs

pub struct HashMap<K, V, S = FxHasher> {
    table1: GpuBuffer<CuckooBucket>,
    table2: GpuBuffer<CuckooBucket>,
    stash: GpuBuffer<StashEntry>,
    len: u32,
    hasher: S,
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    /// Create new empty HashMap
    pub fn new() -> Self;

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self;

    /// Insert key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Get reference to value
    pub fn get(&self, key: &K) -> Option<&V>;

    /// Get mutable reference to value
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V>;

    /// Remove key
    pub fn remove(&mut self, key: &K) -> Option<V>;

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool;

    /// Number of entries
    pub fn len(&self) -> usize;

    /// Is empty?
    pub fn is_empty(&self) -> bool;

    /// Clear all entries
    pub fn clear(&mut self);

    /// Iterate over all entries (returns Vec, not lazy iterator)
    pub fn iter(&self) -> Vec<(&K, &V)>;

    /// Iterate over keys
    pub fn keys(&self) -> Vec<&K>;

    /// Iterate over values
    pub fn values(&self) -> Vec<&V>;

    // Batch operations (GPU-optimized)

    /// Insert multiple entries in parallel
    pub fn insert_batch(&mut self, entries: &[(K, V)]);

    /// Lookup multiple keys in parallel
    pub fn get_batch(&self, keys: &[K]) -> Vec<Option<&V>>;
}

// Entry API
impl<K: Hash + Eq, V> HashMap<K, V> {
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V>;
}

pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}
```

## Implementation Phases

### Phase 1: Core Data Structures
- [ ] Define CuckooEntry, CuckooBucket structs
- [ ] Implement PCG hash function
- [ ] Create GpuHashMap struct with two tables

### Phase 2: Basic Operations
- [ ] Implement lookup (O(1) guaranteed)
- [ ] Implement insert with eviction chain
- [ ] Implement remove (using tombstones)
- [ ] Implement resize

### Phase 3: Rust API
- [ ] Implement HashMap trait interface
- [ ] Add FxHasher as default
- [ ] Implement Entry API

### Phase 4: Batch Operations
- [ ] Implement insert_batch (parallel insert)
- [ ] Implement get_batch (parallel lookup)

### Phase 5: Generic Keys
- [ ] Support string keys (xxHash32)
- [ ] Support arbitrary hashable types

## Constants

```rust
const BUCKET_SIZE: usize = 16;        // Entries per bucket
const BUCKET_BYTES: usize = 128;      // Apple Silicon cache line
const DEFAULT_CAPACITY: usize = 1024; // Default number of entries
const MAX_EVICTIONS: usize = 64;      // 4 * log2(1024)
const STASH_SIZE: usize = 16;         // Overflow buffer
const TOMBSTONE: u32 = 0xDEADDEAD;    // Deleted entry marker
const EMPTY: u32 = 0;                 // Empty entry marker
```

## Trade-offs

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Hash algorithm | PCG | Branchless, fast, good distribution |
| Bucket size | 16 | Matches cache line, high load factor |
| Eviction limit | 4*log2(n) | Balances insert time vs resize frequency |
| Deletion | Tombstones | Simpler than rebuilding |
| Iteration | Returns Vec | GPU can't do lazy iteration |

## Success Criteria

1. O(1) lookup guaranteed (no SIMD divergence)
2. Load factor > 0.9 before resize
3. Insert amortized O(1)
4. Passes all std::collections::HashMap API tests
5. Works with #[no_std] crate (for WASM compilation)

## References

- [cuCollections](https://github.com/NVIDIA/cuCollections) - NVIDIA's GPU hash tables
- [BGHT](https://github.com/owensgroup/BGHT) - Bucketed GPU Hash Tables
- Cuckoo Hashing paper (Pagh & Rodler, 2004)
- Hive Hash Table paper (GPU warp-cooperative)
