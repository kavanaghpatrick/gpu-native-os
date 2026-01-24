# PRD: GPU-Resident Persistent File Cache (Issue #127)

## THE GPU IS THE COMPUTER

**Problem**: Every search reloads files from disk, even if recently searched
**Solution**: GPU-resident file cache with GPU-managed LRU eviction and prefetch

## Core Concept: GPU Manages Its Own Memory

The GPU should manage file caching like a CPU manages L2/L3 cache:
- **Hot files stay resident** - No reload penalty
- **LRU eviction on GPU** - GPU decides what to evict
- **Prefetch prediction** - GPU predicts next files based on access patterns
- **Zero CPU involvement** - Cache is entirely GPU-autonomous

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU-RESIDENT FILE CACHE                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Slot 0    │  │   Slot 1    │  │   Slot N    │         │
│  │  file.rs    │  │  main.rs    │  │  lib.rs     │  ...    │
│  │  LRU: 0     │  │  LRU: 3     │  │  LRU: 1     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  METADATA (GPU-resident):                                   │
│  - File hash → slot mapping (GPU hash table)                │
│  - LRU counters (GPU atomics)                               │
│  - Access frequency (GPU histogram)                         │
│  - Prefetch queue (GPU ring buffer)                         │
├─────────────────────────────────────────────────────────────┤
│  PERSISTENT KERNEL:                                         │
│  - Monitors access patterns                                 │
│  - Triggers async prefetch                                  │
│  - Evicts cold files when cache full                        │
└─────────────────────────────────────────────────────────────┘
```

## GPU Data Structures

### 1. GPU Hash Table (Cuckoo Hashing)
```metal
struct CacheEntry {
    uint64_t file_hash;      // 64-bit hash of file path
    uint32_t slot_index;     // Index into data slots
    uint32_t file_size;      // Size in bytes
    atomic_uint lru_counter; // For LRU eviction
    atomic_uint access_count;// For frequency analysis
    uint32_t flags;          // VALID, LOADING, EVICTING
};

// Cuckoo hash table for O(1) lookup
struct GpuCacheHashTable {
    CacheEntry entries[HASH_TABLE_SIZE];
    uint32_t bucket_count;
    uint32_t seed1, seed2;  // Hash seeds for cuckoo hashing
};
```

### 2. LRU Counter Array
```metal
// GPU-managed LRU using atomic counters
struct LRUManager {
    atomic_uint global_clock;           // Increments on every access
    atomic_uint slot_timestamps[MAX_SLOTS];  // Last access time per slot
};

// Update LRU on access (called by search kernel)
inline void touch_lru(device LRUManager* lru, uint slot) {
    uint timestamp = atomic_fetch_add_explicit(
        &lru->global_clock, 1, memory_order_relaxed);
    atomic_store_explicit(
        &lru->slot_timestamps[slot], timestamp, memory_order_relaxed);
}

// Find eviction candidate (oldest timestamp)
inline uint find_eviction_candidate(device LRUManager* lru, uint slot_count) {
    uint min_time = UINT_MAX;
    uint candidate = 0;

    for (uint i = 0; i < slot_count; i++) {
        uint time = atomic_load_explicit(
            &lru->slot_timestamps[i], memory_order_relaxed);
        if (time < min_time) {
            min_time = time;
            candidate = i;
        }
    }
    return candidate;
}
```

### 3. Prefetch Prediction (Markov Chain)
```metal
// GPU-resident Markov chain for prefetch prediction
struct PrefetchPredictor {
    // transition_counts[from_file][to_file] = count of from→to accesses
    atomic_uint transition_counts[TRACKED_FILES][TRACKED_FILES];
    uint32_t file_to_index[TRACKED_FILES];  // file_hash → index mapping
};

// Update transition on file access
inline void record_transition(
    device PrefetchPredictor* pred,
    uint prev_file_idx,
    uint curr_file_idx
) {
    atomic_fetch_add_explicit(
        &pred->transition_counts[prev_file_idx][curr_file_idx],
        1, memory_order_relaxed);
}

// Predict next file (highest transition count)
inline uint predict_next_file(
    device PrefetchPredictor* pred,
    uint curr_file_idx
) {
    uint max_count = 0;
    uint predicted = INVALID_FILE;

    for (uint i = 0; i < TRACKED_FILES; i++) {
        uint count = atomic_load_explicit(
            &pred->transition_counts[curr_file_idx][i],
            memory_order_relaxed);
        if (count > max_count) {
            max_count = count;
            predicted = i;
        }
    }
    return predicted;
}
```

## GPU Kernels

### Kernel 1: Cache Lookup
```metal
kernel void cache_lookup(
    device GpuCacheHashTable* cache [[buffer(0)]],
    device uint64_t* query_hashes [[buffer(1)]],      // Files to find
    device CacheLookupResult* results [[buffer(2)]],  // Slot or MISS
    device LRUManager* lru [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint64_t hash = query_hashes[tid];

    // Cuckoo hash lookup (check both buckets)
    uint bucket1 = hash % cache->bucket_count;
    uint bucket2 = (hash ^ cache->seed1) % cache->bucket_count;

    CacheEntry* entry = nullptr;
    if (cache->entries[bucket1].file_hash == hash) {
        entry = &cache->entries[bucket1];
    } else if (cache->entries[bucket2].file_hash == hash) {
        entry = &cache->entries[bucket2];
    }

    if (entry && (entry->flags & FLAG_VALID)) {
        results[tid].hit = true;
        results[tid].slot = entry->slot_index;
        results[tid].size = entry->file_size;

        // Update LRU
        touch_lru(lru, entry->slot_index);
    } else {
        results[tid].hit = false;
    }
}
```

### Kernel 2: Eviction Decision
```metal
kernel void eviction_decision(
    device GpuCacheHashTable* cache [[buffer(0)]],
    device LRUManager* lru [[buffer(1)]],
    device EvictionRequest* requests [[buffer(2)]],  // Files to load
    device EvictionDecision* decisions [[buffer(3)]],
    device atomic_uint* eviction_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one eviction request
    EvictionRequest req = requests[tid];

    if (!req.valid) return;

    // Find coldest slot using parallel reduction
    uint candidate = find_eviction_candidate_parallel(lru, cache->slot_count);

    // Atomically claim the slot
    uint expected = FLAG_VALID;
    if (atomic_compare_exchange_strong(
        &cache->entries[candidate].flags,
        &expected,
        FLAG_EVICTING
    )) {
        decisions[tid].slot = candidate;
        decisions[tid].evict_hash = cache->entries[candidate].file_hash;
        atomic_fetch_add_explicit(eviction_count, 1, memory_order_relaxed);
    }
}
```

### Kernel 3: Prefetch Controller (Persistent)
```metal
// Persistent kernel - runs continuously, checking for prefetch opportunities
kernel void prefetch_controller(
    device GpuCacheHashTable* cache [[buffer(0)]],
    device PrefetchPredictor* predictor [[buffer(1)]],
    device PrefetchQueue* queue [[buffer(2)]],        // Ring buffer of requests
    device atomic_uint* should_exit [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Single-thread persistent kernel
    if (tid != 0) return;

    while (!atomic_load_explicit(should_exit, memory_order_relaxed)) {
        // Check recent access pattern
        uint last_file = get_last_accessed_file(cache);

        // Predict next file
        uint predicted = predict_next_file(predictor, last_file);

        if (predicted != INVALID_FILE) {
            // Check if already cached
            CacheLookupResult result = lookup_sync(cache, predicted);

            if (!result.hit) {
                // Queue prefetch request
                enqueue_prefetch(queue, predicted);
            }
        }

        // Yield to other work (spin less aggressively)
        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

## Rust Implementation

```rust
pub struct GpuFileCache {
    device: Device,
    queue: CommandQueue,

    // GPU-resident data structures
    hash_table: Buffer,       // CacheEntry[]
    data_slots: Buffer,       // Actual file contents
    lru_manager: Buffer,      // LRU timestamps
    predictor: Buffer,        // Markov chain

    // Persistent kernel state
    prefetch_queue: Buffer,
    prefetch_controller_running: Arc<AtomicBool>,

    // Configuration
    slot_count: usize,
    slot_size: usize,         // Max file size per slot
    total_cache_size: usize,
}

impl GpuFileCache {
    pub fn new(device: &Device, cache_size_mb: usize) -> Result<Self, CacheError> {
        let slot_size = 1024 * 1024;  // 1MB per slot
        let slot_count = (cache_size_mb * 1024 * 1024) / slot_size;

        // Allocate GPU buffers
        let hash_table = device.new_buffer(
            slot_count * 2 * size_of::<CacheEntry>(),  // 2x for cuckoo
            MTLResourceOptions::StorageModeShared,
        );

        let data_slots = device.new_buffer(
            slot_count * slot_size,
            MTLResourceOptions::StorageModeShared,
        );

        // ... allocate other buffers ...

        Ok(Self { /* ... */ })
    }

    /// Lookup files in cache, returns hits and misses
    pub fn lookup(&self, file_hashes: &[u64]) -> CacheLookupBatch {
        // Upload query hashes
        let query_buffer = self.device.new_buffer_with_data(file_hashes, ...);
        let results_buffer = self.device.new_buffer(
            file_hashes.len() * size_of::<CacheLookupResult>(),
            ...
        );

        // Dispatch lookup kernel
        let cmd = self.queue.command_buffer();
        let encoder = cmd.compute_encoder();
        encoder.set_pipeline(&self.lookup_pipeline);
        encoder.set_buffer(0, &self.hash_table);
        encoder.set_buffer(1, &query_buffer);
        encoder.set_buffer(2, &results_buffer);
        encoder.set_buffer(3, &self.lru_manager);
        encoder.dispatch_threads(file_hashes.len(), 1, 1);
        encoder.end();
        cmd.commit();
        cmd.wait_until_completed();

        // Read results
        CacheLookupBatch::from_buffer(&results_buffer, file_hashes.len())
    }

    /// Load files into cache (with automatic eviction)
    pub fn load(&mut self, files: &[PathBuf]) -> Result<(), CacheError> {
        let lookup = self.lookup(&hash_paths(files));

        // Load misses using GPU batch I/O
        let misses: Vec<_> = files.iter()
            .zip(lookup.results.iter())
            .filter(|(_, r)| !r.hit)
            .map(|(f, _)| f.clone())
            .collect();

        if misses.is_empty() {
            return Ok(());
        }

        // Decide evictions on GPU
        let eviction_decisions = self.decide_evictions(misses.len());

        // Load files into evicted slots
        self.load_into_slots(&misses, &eviction_decisions)?;

        Ok(())
    }

    /// Search cached files
    pub fn search(&self, pattern: &str) -> Vec<SearchMatch> {
        // Search directly in cache slots - no loading needed!
        let cmd = self.queue.command_buffer();
        let encoder = cmd.compute_encoder();
        encoder.set_pipeline(&self.search_pipeline);
        encoder.set_buffer(0, &self.data_slots);  // Search cache directly
        encoder.set_buffer(1, &self.hash_table);
        // ... pattern buffer, results buffer ...
        encoder.dispatch_threads(self.slot_count * THREADS_PER_SLOT, 1, 1);
        encoder.end();
        cmd.commit();
        cmd.wait_until_completed();

        self.read_search_results()
    }

    /// Start prefetch controller (persistent kernel)
    pub fn start_prefetch_controller(&self) {
        self.prefetch_controller_running.store(true, Ordering::Release);

        // Dispatch persistent kernel
        let cmd = self.queue.command_buffer();
        let encoder = cmd.compute_encoder();
        encoder.set_pipeline(&self.prefetch_controller_pipeline);
        encoder.set_buffer(0, &self.hash_table);
        encoder.set_buffer(1, &self.predictor);
        encoder.set_buffer(2, &self.prefetch_queue);
        encoder.set_buffer(3, &self.exit_flag_buffer);
        encoder.dispatch_threads(1, 1, 1);
        encoder.end();
        cmd.commit();
        // Don't wait - runs persistently
    }
}
```

## Performance Model

### Cache Hit Scenario (Common Case)
```
User searches for "TODO" in project:
1. Hash file paths → 10µs
2. GPU cache lookup → 20µs (10,000 files, cuckoo hash O(1))
3. 95% hit rate → load 500 files
4. Search cached data → 16ms (same as before)
5. Total: ~17ms (vs 163ms + 16ms = 179ms without cache)

Speedup: 10x for repeated searches
```

### Cache Miss Scenario
```
First search (cold cache):
1. Hash file paths → 10µs
2. GPU cache lookup → 20µs (all misses)
3. GPU eviction decision → 100µs
4. Load via MTLIOCommandQueue → 30ms (Issue #125)
5. Search → 16ms
6. Total: ~50ms (similar to current, but subsequent = 17ms)
```

### Memory Budget
```
1GB GPU cache:
- 1000 slots × 1MB each
- Hash table: 48KB (1000 × 2 × 24 bytes)
- LRU metadata: 4KB
- Predictor: 4MB (64 × 64 × 1024 bytes)
- Total overhead: <5MB (<0.5%)
```

## Test Plan

### Test 1: Hash Table Correctness
```rust
#[test]
fn test_cuckoo_hash_insert_lookup() {
    let cache = GpuFileCache::new(&device, 100)?;

    // Insert 1000 unique hashes
    let hashes: Vec<u64> = (0..1000).map(|i| hash64(i)).collect();
    cache.insert_all(&hashes);

    // Lookup all - should hit
    let results = cache.lookup(&hashes);
    assert!(results.iter().all(|r| r.hit));

    // Lookup unknowns - should miss
    let unknowns: Vec<u64> = (1000..2000).map(|i| hash64(i)).collect();
    let results = cache.lookup(&unknowns);
    assert!(results.iter().all(|r| !r.hit));
}
```

### Test 2: LRU Eviction
```rust
#[test]
fn test_lru_evicts_coldest() {
    let cache = GpuFileCache::new(&device, 10)?;  // 10 slots only

    // Fill cache
    let files: Vec<_> = (0..10).map(|i| format!("file{}.txt", i)).collect();
    cache.load(&files);

    // Access files 5-9 (make 0-4 cold)
    for i in 5..10 {
        cache.touch(hash64(&files[i]));
    }

    // Load 5 new files (should evict 0-4)
    let new_files: Vec<_> = (10..15).map(|i| format!("file{}.txt", i)).collect();
    cache.load(&new_files);

    // Files 0-4 should be evicted
    for i in 0..5 {
        assert!(!cache.lookup(&[hash64(&files[i])]).results[0].hit);
    }

    // Files 5-14 should be cached
    for i in 5..15 {
        let hash = if i < 10 { hash64(&files[i]) } else { hash64(&new_files[i-10]) };
        assert!(cache.lookup(&[hash]).results[0].hit);
    }
}
```

### Test 3: Prefetch Accuracy
```rust
#[test]
fn test_prefetch_prediction() {
    let cache = GpuFileCache::new(&device, 100)?;

    // Simulate access pattern: A → B → C → A → B → C ...
    let pattern = ["a.rs", "b.rs", "c.rs"];
    for _ in 0..100 {
        for file in &pattern {
            cache.access(hash64(file));
        }
    }

    // After accessing A, predictor should suggest B
    let prediction = cache.predict_next(hash64("a.rs"));
    assert_eq!(prediction, Some(hash64("b.rs")));
}
```

### Test 4: End-to-End Performance
```rust
#[test]
fn benchmark_cache_vs_no_cache() {
    let files = collect_files(".", 10_000);

    // Without cache (current behavior)
    let no_cache_times: Vec<f64> = (0..5).map(|_| {
        let searcher = GpuContentSearch::new(&device);
        time(|| {
            searcher.load_files(&files);
            searcher.search("TODO");
        })
    }).collect();

    // With cache
    let cache = GpuFileCache::new(&device, 500)?;  // 500MB cache
    let cache_times: Vec<f64> = (0..5).map(|_| {
        time(|| {
            cache.load(&files);  // 2nd+ iteration = cache hits
            cache.search("TODO");
        })
    }).collect();

    println!("No cache: {:?}", no_cache_times);
    println!("With cache: {:?}", cache_times);

    // First iteration similar, subsequent 5-10x faster
    assert!(cache_times[4] < no_cache_times[4] / 5.0);
}
```

## Success Metrics

- [ ] Cache lookup: <50µs for 10,000 files
- [ ] Cache hit rate: >90% for repeated searches in same directory
- [ ] LRU eviction: Correctly evicts coldest files
- [ ] Prefetch accuracy: >70% correct predictions for sequential access
- [ ] Repeated search speedup: >5x compared to uncached

## Dependencies

- Issue #125 (GPU Batch I/O) - For loading cache misses
- Issue #126 (Parallel Compact) - For efficient slot packing
- Work Queue (existing) - For persistent prefetch kernel
