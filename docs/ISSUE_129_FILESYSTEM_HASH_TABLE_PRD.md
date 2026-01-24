# Issue #129: O(1) GPU Hash Table for Filesystem Directory Lookup

## Problem Statement

The current `path_lookup_kernel` performs O(n) linear scan through ALL directory entries to find a single file by name. This is catastrophic for large filesystems.

**Current Code (O(n)):**
```metal
// filesystem.rs path_lookup_kernel lines 346-375
for (uint32_t i = tid; i < params.total_entries; i += 1024) {
    if (inodes[entry_inode].parent_id == current_inode) {
        if (entries[i].name_hash == component.hash) {
            // Found it - but we scanned potentially millions of entries!
        }
    }
}
```

**Impact:** Looking up `/Users/foo/bar/file.txt` with 1M files scans 1M entries × 4 components = 4M comparisons!

## Solution: GPU-Native Hash Table

Use direct-mapped hash table with open addressing for O(1) average-case lookup.

### Data Structure

```rust
/// Hash table entry for directory lookups
/// Key: (parent_inode, name_hash)
/// Value: entry_index into entries array
#[repr(C)]
pub struct DirHashEntry {
    pub parent_inode: u32,  // Part of key
    pub name_hash: u32,     // Part of key
    pub entry_index: u32,   // Value: index into entries array
    pub _padding: u32,      // Alignment
}

/// GPU-resident hash table
pub struct GpuDirHashTable {
    table: Buffer,           // [DirHashEntry; capacity]
    capacity: u32,           // Power of 2 for fast modulo
    mask: u32,               // capacity - 1
}
```

### Hash Function

```metal
// Combine parent_inode and name_hash into single lookup key
inline uint64_t dir_hash_key(uint32_t parent_inode, uint32_t name_hash) {
    // Use fibhash for good distribution
    uint64_t key = ((uint64_t)parent_inode << 32) | name_hash;
    return key * 0x9E3779B97F4A7C15ULL;  // Golden ratio hash
}

inline uint32_t hash_slot(uint64_t key, uint32_t mask) {
    return (uint32_t)(key >> 32) & mask;  // Use high bits for slot
}
```

### Lookup Algorithm (O(1) average, O(n) worst case with linear probing)

```metal
// New kernel: O(1) directory entry lookup
kernel void dir_lookup_hash(
    device const DirHashEntry* hash_table [[buffer(0)]],
    constant uint32_t& table_mask [[buffer(1)]],
    device const DirEntryCompact* entries [[buffer(2)]],
    constant uint32_t& parent_inode [[buffer(3)]],
    constant uint32_t& name_hash [[buffer(4)]],
    device uint32_t* result_entry_idx [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single-threaded lookup

    uint64_t key = dir_hash_key(parent_inode, name_hash);
    uint32_t slot = hash_slot(key, table_mask);

    // Linear probing with max 32 attempts
    for (uint32_t probe = 0; probe < 32; probe++) {
        uint32_t idx = (slot + probe) & table_mask;
        DirHashEntry entry = hash_table[idx];

        if (entry.parent_inode == parent_inode && entry.name_hash == name_hash) {
            *result_entry_idx = entry.entry_index;
            return;
        }

        if (entry.entry_index == 0xFFFFFFFF) {
            // Empty slot - not found
            *result_entry_idx = 0xFFFFFFFF;
            return;
        }
    }

    // Max probes exceeded - not found (should never happen with good load factor)
    *result_entry_idx = 0xFFFFFFFF;
}
```

### Parallel Batch Lookup (Optimized)

```metal
// Batch lookup: one thread per path component
kernel void batch_dir_lookup_hash(
    device const DirHashEntry* hash_table [[buffer(0)]],
    constant uint32_t& table_mask [[buffer(1)]],
    device const PathComponent* components [[buffer(2)]],
    device const uint32_t* parent_inodes [[buffer(3)]],  // Pre-resolved parents
    device uint32_t* result_inodes [[buffer(4)]],
    constant uint32_t& component_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= component_count) return;

    uint32_t parent = parent_inodes[gid];
    uint32_t name_hash = components[gid].hash;

    uint64_t key = dir_hash_key(parent, name_hash);
    uint32_t slot = hash_slot(key, table_mask);

    // Each thread probes independently - no divergence!
    for (uint32_t probe = 0; probe < 32; probe++) {
        uint32_t idx = (slot + probe) & table_mask;
        DirHashEntry entry = hash_table[idx];

        if (entry.parent_inode == parent && entry.name_hash == name_hash) {
            result_inodes[gid] = entry.entry_index;
            return;
        }
        if (entry.entry_index == 0xFFFFFFFF) break;
    }

    result_inodes[gid] = 0xFFFFFFFF;  // Not found
}
```

### Table Building (CPU-side, one-time)

```rust
impl GpuDirHashTable {
    pub fn build(device: &Device, entries: &[DirEntryCompact], inodes: &[InodeCompact]) -> Self {
        // Use 2x capacity for ~50% load factor (good for linear probing)
        let capacity = (entries.len() * 2).next_power_of_two() as u32;
        let mask = capacity - 1;

        let mut table = vec![DirHashEntry::empty(); capacity as usize];

        for (i, entry) in entries.iter().enumerate() {
            let parent = inodes[entry.inode_id as usize].parent_id;
            let key = dir_hash_key(parent, entry.name_hash);
            let mut slot = hash_slot(key, mask);

            // Linear probing to find empty slot
            while table[slot as usize].entry_index != 0xFFFFFFFF {
                slot = (slot + 1) & mask;
            }

            table[slot as usize] = DirHashEntry {
                parent_inode: parent,
                name_hash: entry.name_hash,
                entry_index: i as u32,
                _padding: 0,
            };
        }

        let buffer = device.new_buffer_with_data(
            table.as_ptr() as *const _,
            (capacity as usize * std::mem::size_of::<DirHashEntry>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self { table: buffer, capacity, mask }
    }
}
```

## Integration with Path Lookup

```metal
// Modified path_lookup_kernel using hash table
kernel void path_lookup_hash(
    device const DirHashEntry* hash_table [[buffer(0)]],
    constant uint32_t& table_mask [[buffer(1)]],
    device const InodeCompact* inodes [[buffer(2)]],
    device const PathComponent* components [[buffer(3)]],
    constant uint32_t& component_count [[buffer(4)]],
    constant uint32_t& start_inode [[buffer(5)]],
    device uint32_t* result_inode [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint32_t current = start_inode;

    for (uint32_t i = 0; i < component_count; i++) {
        uint64_t key = dir_hash_key(current, components[i].hash);
        uint32_t slot = hash_slot(key, table_mask);

        bool found = false;
        for (uint32_t probe = 0; probe < 32; probe++) {
            uint32_t idx = (slot + probe) & table_mask;
            DirHashEntry entry = hash_table[idx];

            if (entry.parent_inode == current && entry.name_hash == components[i].hash) {
                current = entry.entry_index;  // Move to this entry's inode
                found = true;
                break;
            }
            if (entry.entry_index == 0xFFFFFFFF) break;
        }

        if (!found) {
            *result_inode = 0xFFFFFFFF;  // Path not found
            return;
        }
    }

    *result_inode = current;
}
```

## Benchmarks

### Test Cases

1. **Large filesystem:** 1M files, 10K directories
2. **Deep paths:** /a/b/c/d/e/f/g/h/i/j/file.txt (10 components)
3. **Batch lookup:** 10K paths resolved simultaneously

### Expected Performance

| Operation | Current O(n) | New O(1) | Speedup |
|-----------|-------------|----------|---------|
| Single lookup (1M files) | ~10ms | ~1μs | 10,000x |
| Batch 10K paths | ~100s | ~10ms | 10,000x |
| Path resolution (10 components) | ~100ms | ~10μs | 10,000x |

### Benchmark Code

```rust
#[test]
fn benchmark_dir_lookup() {
    let device = Device::system_default().unwrap();

    // Create filesystem with 1M entries
    let (entries, inodes) = create_large_filesystem(1_000_000);

    // Build hash table
    let hash_table = GpuDirHashTable::build(&device, &entries, &inodes);

    // Benchmark: lookup random paths
    let paths = generate_random_paths(&entries, &inodes, 10_000);

    let old_time = benchmark_linear_scan(&device, &paths);
    let new_time = benchmark_hash_lookup(&device, &hash_table, &paths);

    println!("Linear scan: {:.2}ms, Hash lookup: {:.2}μs, Speedup: {:.0}x",
        old_time * 1e3, new_time * 1e6, old_time / new_time);

    assert!(new_time < old_time / 100.0, "Hash should be 100x+ faster");
}
```

## Memory Overhead

| Files | Hash Table Size | Overhead |
|-------|-----------------|----------|
| 100K | 3.2 MB | 32 bytes/entry × 2x capacity |
| 1M | 32 MB | Acceptable |
| 10M | 320 MB | May need optimization |

## Success Criteria

1. **Correctness:** All path lookups return same results as linear scan
2. **Performance:** ≥1000x speedup for filesystems with ≥100K entries
3. **Memory:** Hash table size ≤ 64 bytes per entry
4. **Load factor:** Maintain <70% to ensure O(1) average case

## Implementation Steps

1. Add `DirHashEntry` struct to `filesystem.rs`
2. Implement `GpuDirHashTable::build()` on CPU
3. Create hash lookup Metal kernels
4. Integrate with existing path lookup API
5. Add cache invalidation when filesystem changes
6. Add tests and benchmarks
