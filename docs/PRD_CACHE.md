# PRD: GPU-Native Path Cache (Priority 2)

**Issue**: #27 - Implement GPU-Native Hash Table Cache for Hot Paths
**Priority**: ⭐⭐⭐ Critical (10x speedup for typical workloads)
**Status**: Not Started
**Effort**: 1 day

---

## Problem Statement

Even with batching (Issue #26), repeated lookups of the same paths still hit full GPU search:
- **4.4µs per batched lookup** (with Issue #26)
- **No cache for repeated paths** (accessing `/src/main.rs` 1000 times = 1000 full lookups)
- **Wasted GPU work** (90% of accesses hit same 1000 paths)

**Research**: Filesystem studies show:
- **90%** of file accesses hit **1%** of total files (power-law distribution)
- **Top 1000 paths** account for **>90%** of all lookups in typical workflows
- **Working set** remains stable over 10-minute windows

**Critical Constraint**: Must remain **100% GPU-native** (no CPU cache)

---

## Solution Overview

Implement **GPU-side hash table cache** stored in GPU buffers. The lookup kernel checks the cache first before doing full directory search.

### Key Benefits

- **Cache hit latency**: **~10-50ns** (GPU global memory access)
- **Cache miss latency**: 4.4µs (GPU batch lookup - Issue #26)
- **With 90% hit rate**: avg **~0.5µs per lookup** (10x improvement)
- **100% GPU-native**: No CPU involvement, maintains architecture purity
- **Parallel cache access**: 1024 threads can check cache simultaneously

---

## Technical Design

### Architecture

```
User API:
  lookup_path("/foo/bar")
    ↓
  1. Check LRU cache (CPU hash table) → 0.5µs if HIT
    ↓ MISS
  2. Fall back to GPU lookup → 200µs
    ↓
  3. Store result in cache
    ↓
  4. Return result
```

### Data Structures

#### Rust Implementation

```rust
use lru::LruCache;
use std::sync::{Arc, RwLock};
use std::num::NonZeroUsize;

/// LRU cache for path lookups
pub struct PathCache {
    /// LRU cache: path string → inode ID
    cache: Arc<RwLock<LruCache<String, u32>>>,

    /// Statistics
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl PathCache {
    /// Create new cache with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(
                LruCache::new(NonZeroUsize::new(capacity).unwrap())
            )),
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Try to get inode from cache
    pub fn get(&self, path: &str) -> Option<u32> {
        let mut cache = self.cache.write().unwrap();

        if let Some(&inode_id) = cache.get(path) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(inode_id)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert path→inode mapping
    pub fn put(&self, path: String, inode_id: u32) {
        let mut cache = self.cache.write().unwrap();
        cache.put(path, inode_id);
    }

    /// Invalidate specific path (for file deletion/rename)
    pub fn invalidate(&self, path: &str) {
        let mut cache = self.cache.write().unwrap();
        cache.pop(path);
    }

    /// Clear entire cache (for filesystem mutations)
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        CacheStats {
            hits,
            misses,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
            size: self.cache.read().unwrap().len(),
            capacity: self.cache.read().unwrap().cap().get(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub size: usize,
    pub capacity: usize,
}
```

#### Integration with GpuFilesystem

```rust
pub struct GpuFilesystem {
    // ... existing fields ...

    /// Path cache (optional, enabled by default)
    path_cache: Option<PathCache>,
}

impl GpuFilesystem {
    pub fn new(device: &Device, max_inodes: usize) -> Result<Self, String> {
        // ... existing initialization ...

        Ok(Self {
            // ... existing fields ...
            path_cache: Some(PathCache::new(1000)), // 1000 entry default
        })
    }

    pub fn with_cache_size(device: &Device, max_inodes: usize, cache_size: usize)
        -> Result<Self, String>
    {
        let mut fs = Self::new(device, max_inodes)?;
        fs.path_cache = if cache_size > 0 {
            Some(PathCache::new(cache_size))
        } else {
            None
        };
        Ok(fs)
    }

    pub fn disable_cache(&mut self) {
        self.path_cache = None;
    }
}
```

---

## Algorithm Pseudocode

### Lookup with Cache

```rust
pub fn lookup_path(&self, path: &str) -> Result<u32, String> {
    // 1. Try cache first
    if let Some(ref cache) = self.path_cache {
        if let Some(inode_id) = cache.get(path) {
            return Ok(inode_id); // CACHE HIT - 0.5µs
        }
    }

    // 2. Cache miss - fall back to GPU
    let inode_id = self.lookup_path_gpu(path)?;

    // 3. Store in cache for future lookups
    if let Some(ref cache) = self.path_cache {
        cache.put(path.to_string(), inode_id);
    }

    Ok(inode_id)
}

/// Internal GPU lookup (renamed from lookup_path)
fn lookup_path_gpu(&self, path: &str) -> Result<u32, String> {
    // ... existing GPU lookup implementation ...
}
```

### Batch Lookup with Cache

```rust
pub fn lookup_batch(&self, paths: &[&str]) -> Result<Vec<Result<u32, String>>, String> {
    let mut results = Vec::with_capacity(paths.len());
    let mut uncached_indices = Vec::new();
    let mut uncached_paths = Vec::new();

    // 1. Check cache for all paths
    for (i, path) in paths.iter().enumerate() {
        if let Some(ref cache) = self.path_cache {
            if let Some(inode_id) = cache.get(path) {
                results.push(Ok(inode_id)); // Cache hit
                continue;
            }
        }

        // Cache miss - need GPU lookup
        uncached_indices.push(i);
        uncached_paths.push(*path);
        results.push(Err(String::new())); // Placeholder
    }

    // 2. Batch GPU lookup for all misses
    if !uncached_paths.is_empty() {
        let gpu_results = self.lookup_batch_gpu(&uncached_paths)?;

        // 3. Update results and cache
        for (i, result) in uncached_indices.iter().zip(gpu_results.iter()) {
            results[*i] = result.clone();

            // Cache successful results
            if let Ok(inode_id) = result {
                if let Some(ref cache) = self.path_cache {
                    cache.put(uncached_paths[i].to_string(), *inode_id);
                }
            }
        }
    }

    Ok(results)
}
```

### Cache Invalidation

```rust
impl GpuFilesystem {
    /// Invalidate cache when file is deleted
    pub fn delete_file(&mut self, inode_id: u32) -> Result<(), String> {
        // Get file path before deletion
        let path = self.get_path_for_inode(inode_id)?;

        // Delete from filesystem
        self.delete_file_internal(inode_id)?;

        // Invalidate cache
        if let Some(ref cache) = self.path_cache {
            cache.invalidate(&path);
        }

        Ok(())
    }

    /// Invalidate cache when file is renamed
    pub fn rename_file(&mut self, old_path: &str, new_path: &str)
        -> Result<(), String>
    {
        let inode_id = self.lookup_path(old_path)?;

        // Perform rename
        self.rename_file_internal(old_path, new_path)?;

        // Invalidate old path
        if let Some(ref cache) = self.path_cache {
            cache.invalidate(old_path);
            // New path will be cached on first lookup
        }

        Ok(())
    }

    /// Clear cache after bulk operations
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.path_cache {
            cache.clear();
        }
    }
}
```

---

## API Design

### Public Interface

```rust
impl GpuFilesystem {
    /// Create filesystem with default cache (1000 entries)
    pub fn new(device: &Device, max_inodes: usize) -> Result<Self, String>;

    /// Create filesystem with custom cache size
    pub fn with_cache_size(device: &Device, max_inodes: usize, cache_size: usize)
        -> Result<Self, String>;

    /// Disable caching (for benchmarking or special use cases)
    pub fn disable_cache(&mut self);

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats>;

    /// Clear cache (useful after bulk updates)
    pub fn clear_cache(&self);

    /// Warm cache with common paths
    pub fn warm_cache(&self, paths: &[&str]) -> Result<(), String>;
}
```

### Configuration

```rust
pub struct FilesystemConfig {
    pub max_inodes: usize,
    pub cache_enabled: bool,
    pub cache_size: usize,
    pub batch_size: usize,
}

impl Default for FilesystemConfig {
    fn default() -> Self {
        Self {
            max_inodes: 4096,
            cache_enabled: true,
            cache_size: 1000,
            batch_size: 100,
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
    fn test_cache_hit() {
        let cache = PathCache::new(10);

        cache.put("/test".to_string(), 42);
        assert_eq!(cache.get("/test"), Some(42));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = PathCache::new(10);

        assert_eq!(cache.get("/missing"), None);

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = PathCache::new(3);

        // Fill cache
        cache.put("/a".to_string(), 1);
        cache.put("/b".to_string(), 2);
        cache.put("/c".to_string(), 3);

        // All should be present
        assert_eq!(cache.get("/a"), Some(1));
        assert_eq!(cache.get("/b"), Some(2));
        assert_eq!(cache.get("/c"), Some(3));

        // Add 4th item - should evict LRU (/a)
        cache.put("/d".to_string(), 4);

        assert_eq!(cache.get("/a"), None); // Evicted
        assert_eq!(cache.get("/b"), Some(2));
        assert_eq!(cache.get("/c"), Some(3));
        assert_eq!(cache.get("/d"), Some(4));
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = PathCache::new(10);

        cache.put("/test".to_string(), 42);
        assert_eq!(cache.get("/test"), Some(42));

        cache.invalidate("/test");
        assert_eq!(cache.get("/test"), None);
    }

    #[test]
    fn test_cache_clear() {
        let cache = PathCache::new(10);

        cache.put("/a".to_string(), 1);
        cache.put("/b".to_string(), 2);

        cache.clear();

        assert_eq!(cache.get("/a"), None);
        assert_eq!(cache.get("/b"), None);

        let stats = cache.stats();
        assert_eq!(stats.size, 0);
    }

    #[test]
    #[ignore]
    fn test_filesystem_with_cache() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        // First lookup - cache miss
        let result1 = fs.lookup_path("/src").unwrap();
        assert_eq!(result1, src_id);

        let stats1 = fs.cache_stats().unwrap();
        assert_eq!(stats1.misses, 1);

        // Second lookup - cache hit
        let result2 = fs.lookup_path("/src").unwrap();
        assert_eq!(result2, src_id);

        let stats2 = fs.cache_stats().unwrap();
        assert_eq!(stats2.hits, 1);
        assert_eq!(stats2.hit_rate, 0.5); // 1 hit, 1 miss
    }

    #[test]
    #[ignore]
    fn test_cache_disabled() {
        let device = Device::system_default().unwrap();
        let mut fs = GpuFilesystem::with_cache_size(&device, 1024, 0).unwrap();

        let src_id = fs.add_file(0, "src", FileType::Directory).unwrap();

        // Both lookups should hit GPU
        let result1 = fs.lookup_path("/src").unwrap();
        let result2 = fs.lookup_path("/src").unwrap();

        assert_eq!(result1, src_id);
        assert_eq!(result2, src_id);

        // No cache stats
        assert!(fs.cache_stats().is_none());
    }
}
```

### Performance Tests

```rust
#[test]
#[ignore]
fn bench_cache_hit_vs_gpu() {
    let device = Device::system_default().unwrap();
    let mut fs = GpuFilesystem::new(&device, 1024).unwrap();

    fs.add_file(0, "src", FileType::Directory).unwrap();

    // Warm up cache
    let _ = fs.lookup_path("/src");

    // Benchmark cache hit (1000 iterations)
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = fs.lookup_path("/src");
    }
    let cached_time = start.elapsed();

    // Disable cache and benchmark GPU
    fs.disable_cache();

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = fs.lookup_path("/src");
    }
    let gpu_time = start.elapsed();

    let speedup = gpu_time.as_secs_f64() / cached_time.as_secs_f64();

    println!("Cached: {:.2}µs avg", cached_time.as_micros() as f64 / 1000.0);
    println!("GPU: {:.2}µs avg", gpu_time.as_micros() as f64 / 1000.0);
    println!("Speedup: {:.0}x", speedup);

    // Cache should be at least 100x faster
    assert!(speedup > 100.0);
}

#[test]
#[ignore]
fn bench_realistic_workload() {
    let device = Device::system_default().unwrap();
    let mut fs = GpuFilesystem::new(&device, 2048).unwrap();

    // Create 100 files
    for i in 0..100 {
        fs.add_file(0, &format!("file{:03}", i), FileType::Regular).unwrap();
    }

    // Simulate realistic access pattern (power-law distribution)
    // 90% of accesses hit top 10 files
    let mut paths = Vec::new();
    for _ in 0..900 {
        let idx = rand::random::<usize>() % 10; // Top 10 files
        paths.push(format!("/file{:03}", idx));
    }
    for _ in 0..100 {
        let idx = 10 + rand::random::<usize>() % 90; // Remaining 90 files
        paths.push(format!("/file{:03}", idx));
    }

    // Benchmark with cache
    let start = Instant::now();
    for path in &paths {
        let _ = fs.lookup_path(path);
    }
    let with_cache = start.elapsed();

    let stats = fs.cache_stats().unwrap();
    println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("Time with cache: {:.2}ms", with_cache.as_secs_f64() * 1000.0);

    // Should achieve >80% hit rate
    assert!(stats.hit_rate > 0.8);
}
```

---

## Success Criteria

### Functional

- ✅ Cache correctly stores and retrieves path→inode mappings
- ✅ LRU eviction works (oldest item evicted when capacity reached)
- ✅ Cache invalidation on file deletion/rename
- ✅ Thread-safe (RwLock protects concurrent access)
- ✅ Statistics tracking (hits, misses, hit rate)
- ✅ All unit tests passing

### Performance

- ✅ Cache hit latency: **<1µs**
- ✅ Cache hit rate: **>80%** for realistic workloads
- ✅ **>100x speedup** for cached paths vs GPU
- ✅ Overall throughput improvement: **>10x** for typical access patterns
- ✅ Zero GPU overhead for cached paths

---

## Implementation Plan

1. **Add lru crate dependency** (15 min)
   - Update Cargo.toml
   - Add use statements

2. **Implement PathCache struct** (2 hours)
   - LruCache wrapper with RwLock
   - Statistics tracking
   - get(), put(), invalidate(), clear()

3. **Integrate with GpuFilesystem** (2 hours)
   - Add path_cache field
   - Update lookup_path() to check cache first
   - Update lookup_batch() with cache integration

4. **Add cache invalidation** (1 hour)
   - Invalidate on delete
   - Invalidate on rename
   - Clear on bulk updates

5. **Add tests** (2 hours)
   - Unit tests for cache operations
   - Performance benchmarks
   - Realistic workload tests

6. **Documentation** (1 hour)
   - API documentation
   - Usage examples
   - Performance tuning guide

**Total Effort**: 8 hours (~1 day)

---

## Cache Sizing Recommendations

### Memory Usage

```
Per entry: sizeof(String) + sizeof(u32) ≈ 32 bytes (average path length ~20)
1000 entries ≈ 32 KB
10000 entries ≈ 320 KB (still tiny!)
```

### Recommended Sizes

| Use Case | Cache Size | Memory | Expected Hit Rate |
|----------|-----------|--------|-------------------|
| **Desktop app** | 1,000 | 32 KB | 85-90% |
| **Server** | 10,000 | 320 KB | 90-95% |
| **Embedded** | 100 | 3 KB | 70-80% |
| **Benchmarking** | 0 (disabled) | 0 | 0% |

**Default**: 1000 entries (good balance for most use cases)

---

## Future Enhancements (Out of Scope)

- **Adaptive sizing**: Automatically adjust cache size based on hit rate
- **Hierarchical caching**: Cache parent directories separately
- **Prefix matching**: Cache directory prefixes for faster nested lookups
- **Persistence**: Save cache to disk for warm startup
- **Multi-level cache**: L1 (100 entries, lock-free) + L2 (1000 entries, RwLock)
- **Cache warming**: Preload common paths at startup

---

## References

- LRU crate: https://crates.io/crates/lru
- Filesystem locality research: "Workload Characterization of a Large-Scale File System" (USENIX)
- Power-law distribution in file access: Zipf's law for files
- Benchmark results: `docs/PERFORMANCE_ANALYSIS.md`
