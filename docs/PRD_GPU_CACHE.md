# PRD: GPU-Native Path Cache (Priority 2)

**Issue**: #27 - GPU-Side Hash Table Cache for Hot Paths
**Priority**: ⭐⭐⭐ Critical (10x speedup for hot paths)
**Status**: Not Started
**Effort**: 1 day

---

## Problem Statement

Even with batching (97x speedup from Issue #26), we still do full directory searches for repeated paths:
- Accessing `/src/main.rs` 1000 times = 1000 full GPU searches
- **90%** of filesystem accesses hit the **same 1000 paths**
- Each batch lookup: **4.4µs per path** (good, but can be better)
- No memory of previous lookups

**Critical Constraint**: Must remain **100% GPU-native** (no CPU caching)

---

## Solution Overview

Implement **GPU-side hash table** stored in GPU global memory. Lookup kernel checks cache first with parallel hash probe before full search.

### Key Benefits

- **GPU cache hit**: **~50ns** (global memory read on M4 Pro)
- **GPU cache miss**: 4.4µs (fall back to batched lookup)
- **With 90% hit rate**: avg **~0.5µs per lookup** (10x improvement)
- **100% GPU-native**: Cache lives in GPU buffers, accessed by GPU threads
- **Parallel access**: All 1024 threads check cache simultaneously

---

## Technical Design

### Architecture

```
GPU Memory Layout:
  ┌─────────────────────────────────┐
  │ Inode Buffer (existing)         │
  │ DirEntry Buffer (existing)      │
  ├─────────────────────────────────┤
  │ Path Cache Buffer (NEW)         │  ← Hash table in GPU memory
  │   - 1024 cache entries          │
  │   - Hash → (path_hash, inode)   │
  │   - Direct-mapped cache         │
  └─────────────────────────────────┘

Lookup Flow:
  1. GPU threads compute hash(path)
  2. Parallel check: cache[hash % 1024]
  3. If match → return inode (50ns)
  4. If miss → full search + store result in cache
```

### Data Structures

#### Metal Side (GPU)

```metal
// GPU cache entry - 64 bytes (cache line aligned)
struct PathCacheEntry {
    uint64_t path_hash;      // xxHash3 of full path string
    uint32_t inode_id;       // Cached inode result
    uint32_t access_count;   // For statistics/debugging
    uint64_t timestamp;      // Last access time (for eviction)
    uint32_t path_len;       // Length of cached path
    uint32_t _padding;
    char path[40];           // Inline path storage (for verification)
};
// Total: 64 bytes

// Cache parameters
constant uint32_t CACHE_SIZE = 1024;  // Power of 2 for fast modulo
constant uint32_t CACHE_MASK = 1023;  // CACHE_SIZE - 1
```

#### Rust Side

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PathCacheEntry {
    pub path_hash: u64,      // xxHash3 of full path
    pub inode_id: u32,       // Cached result
    pub access_count: u32,   // Statistics
    pub timestamp: u64,      // Last access (frame number)
    pub path_len: u32,       // Path length
    pub _padding: u32,
    pub path: [u8; 40],      // Inline path for verification
}

impl PathCacheEntry {
    pub fn is_valid(&self) -> bool {
        self.path_hash != 0
    }

    pub fn is_match(&self, path: &str, hash: u64) -> bool {
        if self.path_hash != hash {
            return false;
        }

        // Verify path matches (handle hash collisions)
        if self.path_len as usize != path.len() {
            return false;
        }

        let path_bytes = path.as_bytes();
        for i in 0..path.len().min(40) {
            if self.path[i] != path_bytes[i] {
                return false;
            }
        }

        true
    }
}

pub struct PathCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub total_entries: usize,
}
```

---

## Algorithm Pseudocode

### GPU Kernel (Modified Batch Lookup with Cache)

```metal
kernel void batch_lookup_with_cache(
    device InodeCompact* inodes [[buffer(0)]],
    device DirEntryCompact* entries [[buffer(1)]],
    constant BatchParams& params [[buffer(2)]],
    device PathComponent* all_components [[buffer(4)]],
    device PathMetadata* path_metadata [[buffer(5)]],
    device BatchResult* results [[buffer(6)]],
    device PathCacheEntry* cache [[buffer(8)]],        // NEW: GPU cache
    device atomic_ulong* cache_hits [[buffer(9)]],     // NEW: Hit counter
    device atomic_ulong* cache_misses [[buffer(10)]],  // NEW: Miss counter
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
)
{
    uint path_idx = gid / 1024;
    if (path_idx >= params.batch_size) return;

    PathMetadata meta = path_metadata[path_idx];

    // === CACHE CHECK (NEW) ===
    // Only thread 0 checks cache (avoid duplicate checks)
    threadgroup bool cache_hit;
    threadgroup uint32_t cached_inode;

    if (tid == 0) {
        cache_hit = false;

        // Compute full path hash (using components)
        uint64_t full_path_hash = compute_full_path_hash(
            all_components, meta.start_idx, meta.component_count
        );

        // Probe cache (direct-mapped)
        uint32_t cache_slot = (uint32_t)(full_path_hash & CACHE_MASK);
        PathCacheEntry entry = cache[cache_slot];

        // Check if entry matches our path
        if (entry.path_hash == full_path_hash) {
            // Verify path components match (handle collisions)
            bool match = verify_path_components(
                entry, all_components, meta.start_idx, meta.component_count
            );

            if (match) {
                cache_hit = true;
                cached_inode = entry.inode_id;

                // Update statistics
                atomic_fetch_add_explicit(cache_hits, 1, memory_order_relaxed);
                cache[cache_slot].access_count++;
            }
        }

        if (!cache_hit) {
            atomic_fetch_add_explicit(cache_misses, 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === CACHE HIT - Fast path ===
    if (cache_hit) {
        if (tid == 0) {
            results[path_idx].inode_id = cached_inode;
            results[path_idx].status = STATUS_SUCCESS;
        }
        return;  // Done! ~50ns
    }

    // === CACHE MISS - Full lookup ===
    // ... existing full directory search code ...
    uint32_t current_inode = meta.start_inode;

    for (uint32_t comp_idx = 0; comp_idx < meta.component_count; comp_idx++) {
        // ... parallel search through directories ...
    }

    // === UPDATE CACHE ===
    if (tid == 0 && current_inode != INVALID_INODE) {
        // Compute full path hash
        uint64_t full_path_hash = compute_full_path_hash(
            all_components, meta.start_idx, meta.component_count
        );

        uint32_t cache_slot = (uint32_t)(full_path_hash & CACHE_MASK);

        // Store in cache (overwrites existing entry)
        cache[cache_slot].path_hash = full_path_hash;
        cache[cache_slot].inode_id = current_inode;
        cache[cache_slot].access_count = 1;
        cache[cache_slot].timestamp = params.frame_number;

        // Store path inline (up to 40 chars)
        cache[cache_slot].path_len = min(meta.total_path_len, 40);
        copy_path_to_cache(&cache[cache_slot], all_components, meta);
    }

    if (tid == 0) {
        results[path_idx].inode_id = current_inode;
        results[path_idx].status = (current_inode != INVALID_INODE)
            ? STATUS_SUCCESS
            : STATUS_NOT_FOUND;
    }
}

// Helper: Hash full path from components
uint64_t compute_full_path_hash(
    device PathComponent* components,
    uint32_t start_idx,
    uint32_t count
) {
    uint64_t hash = 0x9E3779B185EBCA87UL;  // xxHash3 seed

    for (uint32_t i = 0; i < count; i++) {
        PathComponent comp = components[start_idx + i];

        // Mix in component hash
        hash ^= comp.hash;
        hash *= 0x9E3779B185EBCA87UL;
    }

    return hash;
}
```

### Rust API

```rust
impl GpuFilesystem {
    /// Create filesystem with GPU cache
    pub fn new(device: &Device, max_inodes: usize) -> Result<Self, String> {
        // ... existing initialization ...

        // Allocate GPU cache buffer
        let cache_size = 1024;
        let cache_buffer = builder.create_buffer(
            cache_size * mem::size_of::<PathCacheEntry>()
        );

        // Initialize cache to zeros
        unsafe {
            let ptr = cache_buffer.contents() as *mut PathCacheEntry;
            for i in 0..cache_size {
                *ptr.add(i) = PathCacheEntry {
                    path_hash: 0,
                    inode_id: 0,
                    access_count: 0,
                    timestamp: 0,
                    path_len: 0,
                    _padding: 0,
                    path: [0; 40],
                };
            }
        }

        // Cache statistics buffers
        let cache_hits_buffer = builder.create_buffer(mem::size_of::<u64>());
        let cache_misses_buffer = builder.create_buffer(mem::size_of::<u64>());

        Ok(Self {
            // ... existing fields ...
            cache_buffer,
            cache_hits_buffer,
            cache_misses_buffer,
            cache_size,
        })
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> PathCacheStats {
        unsafe {
            let hits = *(self.cache_hits_buffer.contents() as *const u64);
            let misses = *(self.cache_misses_buffer.contents() as *const u64);
            let total = hits + misses;

            PathCacheStats {
                hits,
                misses,
                hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
                total_entries: self.cache_size,
            }
        }
    }

    /// Clear GPU cache (invalidate all entries)
    pub fn clear_cache(&mut self) {
        unsafe {
            let ptr = self.cache_buffer.contents() as *mut PathCacheEntry;
            for i in 0..self.cache_size {
                (*ptr.add(i)).path_hash = 0;
            }

            // Reset statistics
            *(self.cache_hits_buffer.contents() as *mut u64) = 0;
            *(self.cache_misses_buffer.contents() as *mut u64) = 0;
        }
    }
}
```

---

## Test Specification

### Unit Tests

```rust
#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn test_cache_entry_size() {
        assert_eq!(mem::size_of::<PathCacheEntry>(), 64);
    }

    #[test]
    #[ignore]
    fn test_cache_hit_simple() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        // First lookup - cache miss
        let result1 = fs.lookup_path("/src").unwrap();
        assert_eq!(result1, src_id);

        let stats1 = fs.cache_stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second lookup - cache hit!
        let result2 = fs.lookup_path("/src").unwrap();
        assert_eq!(result2, src_id);

        let stats2 = fs.cache_stats();
        assert_eq!(stats2.hits, 1);
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hit_rate, 0.5);
    }

    #[test]
    #[ignore]
    fn test_cache_hit_batch() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();
        let docs_id = fs.add_file(0, "docs", FileType::Directory).unwrap();

        let paths = vec!["/src", "/docs"];

        // First batch - all misses
        let results1 = fs.lookup_batch(&paths).unwrap();
        assert_eq!(results1[0].as_ref().unwrap(), &src_id);
        assert_eq!(results1[1].as_ref().unwrap(), &docs_id);

        let stats1 = fs.cache_stats();
        assert_eq!(stats1.misses, 2);

        // Second batch - all hits!
        let results2 = fs.lookup_batch(&paths).unwrap();
        assert_eq!(results2[0].as_ref().unwrap(), &src_id);
        assert_eq!(results2[1].as_ref().unwrap(), &docs_id);

        let stats2 = fs.cache_stats();
        assert_eq!(stats2.hits, 2);
        assert_eq!(stats2.hit_rate, 0.5);
    }

    #[test]
    #[ignore]
    fn test_cache_hot_paths() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        // Access same path 100 times
        for _ in 0..100 {
            let result = fs.lookup_path("/src").unwrap();
            assert_eq!(result, src_id);
        }

        let stats = fs.cache_stats();
        assert_eq!(stats.hits, 99);  // First is miss, rest are hits
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.99);
    }

    #[test]
    #[ignore]
    fn test_cache_collision_handling() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        // Create many files to potentially cause hash collisions
        for i in 0..100 {
            fs.add_file(0, &format!("file{}", i), FileType::Regular).unwrap();
        }

        // Lookup all files twice
        for _ in 0..2 {
            for i in 0..100 {
                let _ = fs.lookup_path(&format!("/file{}", i));
            }
        }

        let stats = fs.cache_stats();
        // Should have high hit rate even with potential collisions
        assert!(stats.hit_rate > 0.9, "Hit rate: {}", stats.hit_rate);
    }

    #[test]
    #[ignore]
    fn test_cache_clear() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        // Warm cache
        let _ = fs.lookup_path("/src");

        let stats1 = fs.cache_stats();
        assert_eq!(stats1.misses, 1);

        // Clear cache
        fs.clear_cache();

        let stats2 = fs.cache_stats();
        assert_eq!(stats2.hits, 0);
        assert_eq!(stats2.misses, 0);

        // Next lookup should miss
        let _ = fs.lookup_path("/src");

        let stats3 = fs.cache_stats();
        assert_eq!(stats3.misses, 1);
        assert_eq!(stats3.hits, 0);
    }
}
```

### Performance Tests

```rust
#[test]
#[ignore]
fn bench_cache_effectiveness() {
    use std::time::Instant;

    let device = Device::system_default().unwrap();
    let mut fs = GpuFilesystem::new(&device, 2048).unwrap();

    // Create 1000 files
    for i in 0..1000 {
        fs.add_file(0, &format!("file{:04}", i), FileType::Regular).unwrap();
    }

    // Simulate realistic workload (90% access top 10% of files)
    let mut paths = Vec::new();
    for _ in 0..900 {
        let idx = rand::random::<usize>() % 100; // Hot 100 files
        paths.push(format!("/file{:04}", idx));
    }
    for _ in 0..100 {
        let idx = 100 + rand::random::<usize>() % 900; // Cold files
        paths.push(format!("/file{:04}", idx));
    }

    // Benchmark with cache
    let start = Instant::now();
    for path in &paths {
        let _ = fs.lookup_path(path);
    }
    let with_cache = start.elapsed();

    let stats = fs.cache_stats();
    println!("Cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("Time: {:.2}ms", with_cache.as_secs_f64() * 1000.0);

    // Should achieve >80% hit rate
    assert!(stats.hit_rate > 0.8);
}
```

---

## Success Criteria

### Functional

- ✅ GPU cache correctly stores path→inode mappings in GPU memory
- ✅ Cache hits return correct inode without full search
- ✅ Cache misses fall back to full lookup and populate cache
- ✅ Hash collision handling (path verification)
- ✅ Statistics tracking (hits, misses, hit rate)
- ✅ Cache invalidation (clear all entries)
- ✅ All unit tests passing

### Performance

- ✅ Cache hit latency: **<100ns** (GPU global memory access)
- ✅ Cache hit rate: **>80%** for realistic workloads
- ✅ **>10x speedup** for hot paths vs full lookup
- ✅ Zero CPU involvement (100% GPU-native)
- ✅ No performance degradation on cache miss

---

## Implementation Plan

1. **Add PathCacheEntry struct** (1 hour)
   - Rust and Metal definitions
   - Size validation (64 bytes)

2. **Allocate GPU cache buffer** (1 hour)
   - 1024-entry cache in GPU memory
   - Statistics buffers (hits/misses)
   - Initialize to zeros

3. **Implement cache lookup in Metal kernel** (3 hours)
   - Hash computation for full path
   - Cache probe logic
   - Collision handling
   - Fast return on hit

4. **Implement cache update on miss** (2 hours)
   - Store result after full lookup
   - Update statistics
   - Handle eviction (simple overwrite)

5. **Add Rust API** (1 hour)
   - cache_stats() method
   - clear_cache() method

6. **Add tests** (2 hours)
   - Unit tests
   - Performance benchmarks
   - Hot path simulation

**Total Effort**: 10 hours (~1 day)

---

## Cache Design Decisions

### Direct-Mapped vs Set-Associative

**Choice**: Direct-mapped (hash % 1024)

**Rationale**:
- Simpler GPU implementation
- No need for LRU tracking on GPU
- Single memory access (no searching)
- Good enough for 90% hit rate

### Cache Size: 1024 entries

**Memory**: 1024 × 64 bytes = **64 KB**

**Rationale**:
- Fits in L2 cache on M4 Pro GPU
- Covers top 1000 paths (power-law distribution)
- Power of 2 for fast modulo (bitwise AND)

### Eviction Policy: Overwrite oldest

**Policy**: Simple replacement (no LRU)

**Rationale**:
- GPU has no efficient LRU mechanism
- Power-law access means hot entries rarely evicted
- Timestamp field available for future optimization

---

## Future Enhancements (Out of Scope)

- **2-way set associative**: Better collision handling
- **Larger cache**: 4096 entries (256 KB)
- **Cache warming**: Preload common paths at startup
- **Per-directory caching**: Cache directory contents separately
- **Hierarchical cache**: L1 (small, fast) + L2 (large, slower)

---

## References

- GPU cache architecture: CUDA Programming Guide
- Hash table on GPU: "GPU Hash Tables" paper
- Filesystem locality: USENIX "File Access Patterns" study
- Issue #26: Batch Path Lookup (provides 4.4µs baseline)
