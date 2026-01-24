// Issue #127: GPU-Resident Persistent File Cache
//
// THE GPU IS THE COMPUTER - GPU manages its own cache with LRU eviction

use metal::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

// These will be implemented in src/gpu_os/gpu_cache.rs
// use rust_experiment::gpu_os::gpu_cache::{GpuFileCache, CacheLookupResult};

/// Hash a string to u64 (for file path hashing)
fn hash64(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// CPU reference: simple LRU cache
struct CpuLruCache {
    capacity: usize,
    entries: Vec<(u64, Vec<u8>, u64)>, // (hash, data, timestamp)
    clock: u64,
}

impl CpuLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            clock: 0,
        }
    }

    fn lookup(&mut self, hash: u64) -> Option<&[u8]> {
        self.clock += 1;
        for entry in &mut self.entries {
            if entry.0 == hash {
                entry.2 = self.clock; // Update timestamp
                return Some(&entry.1);
            }
        }
        None
    }

    fn insert(&mut self, hash: u64, data: Vec<u8>) {
        self.clock += 1;

        // Check if already exists
        for entry in &mut self.entries {
            if entry.0 == hash {
                entry.1 = data;
                entry.2 = self.clock;
                return;
            }
        }

        // Evict if full
        if self.entries.len() >= self.capacity {
            // Find oldest
            let oldest_idx = self.entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.2)
                .map(|(i, _)| i)
                .unwrap();
            self.entries.remove(oldest_idx);
        }

        self.entries.push((hash, data, self.clock));
    }

    fn hit_rate(&self, lookups: &[u64]) -> f64 {
        let mut cache = Self::new(self.capacity);
        cache.entries = self.entries.clone();
        cache.clock = self.clock;

        let mut hits = 0;
        for &hash in lookups {
            if cache.lookup(hash).is_some() {
                hits += 1;
            }
        }
        hits as f64 / lookups.len() as f64
    }
}

#[test]
fn test_cpu_lru_reference() {
    let mut cache = CpuLruCache::new(3);

    // Insert 3 items
    cache.insert(hash64("a"), vec![1]);
    cache.insert(hash64("b"), vec![2]);
    cache.insert(hash64("c"), vec![3]);

    // All should hit
    assert!(cache.lookup(hash64("a")).is_some());
    assert!(cache.lookup(hash64("b")).is_some());
    assert!(cache.lookup(hash64("c")).is_some());

    // Insert 4th item - should evict oldest (which is now 'a' since we just accessed it last)
    // Actually 'a' was accessed most recently, so 'b' should be evicted
    // Wait - we accessed a, then b, then c, so the order is a < b < c
    // Then we lookup a (now most recent), b, c
    // So order is now: original c < b < a (since a was looked up last)
    // No wait, lookups in order: a, b, c - so c is most recent
    // Let's trace through:
    // After inserts: a(1), b(2), c(3)
    // lookup a -> a(4), b(2), c(3)
    // lookup b -> a(4), b(5), c(3)
    // lookup c -> a(4), b(5), c(6)
    // Insert d: evict oldest = a(4)
    cache.insert(hash64("d"), vec![4]);

    // 'a' should be evicted (oldest after lookups)
    assert!(cache.lookup(hash64("a")).is_none(), "a should be evicted");
    assert!(cache.lookup(hash64("b")).is_some());
    assert!(cache.lookup(hash64("c")).is_some());
    assert!(cache.lookup(hash64("d")).is_some());

    println!("CPU LRU reference: OK");
}

#[test]
fn test_cuckoo_hash_shader() {
    let device = Device::system_default().expect("No Metal device");

    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct CacheEntry {
            uint64_t file_hash;
            uint32_t slot_index;
            uint32_t file_size;
            uint32_t lru_timestamp;
            uint32_t flags;
        };

        constant uint FLAG_EMPTY = 0;
        constant uint FLAG_VALID = 1;

        // Simple hash functions for cuckoo hashing
        inline uint hash1(uint64_t key, uint table_size) {
            return uint(key % table_size);
        }

        inline uint hash2(uint64_t key, uint table_size) {
            return uint((key >> 32) % table_size);
        }

        // Lookup in cuckoo hash table
        kernel void cuckoo_lookup(
            device CacheEntry* table [[buffer(0)]],
            constant uint& table_size [[buffer(1)]],
            device uint64_t* query_hashes [[buffer(2)]],
            device int* results [[buffer(3)]],  // -1 = miss, else slot index
            uint tid [[thread_position_in_grid]]
        ) {
            uint64_t hash = query_hashes[tid];

            // Check first bucket
            uint idx1 = hash1(hash, table_size);
            if (table[idx1].flags == FLAG_VALID && table[idx1].file_hash == hash) {
                results[tid] = int(table[idx1].slot_index);
                return;
            }

            // Check second bucket
            uint idx2 = hash2(hash, table_size);
            if (table[idx2].flags == FLAG_VALID && table[idx2].file_hash == hash) {
                results[tid] = int(table[idx2].slot_index);
                return;
            }

            // Miss
            results[tid] = -1;
        }

        // Insert into cuckoo hash table (single-threaded for simplicity)
        kernel void cuckoo_insert(
            device CacheEntry* table [[buffer(0)]],
            constant uint& table_size [[buffer(1)]],
            device CacheEntry* to_insert [[buffer(2)]],
            device uint* success [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid != 0) return;

            CacheEntry entry = to_insert[0];
            uint64_t hash = entry.file_hash;

            // Try first bucket
            uint idx1 = hash1(hash, table_size);
            if (table[idx1].flags == FLAG_EMPTY) {
                table[idx1] = entry;
                table[idx1].flags = FLAG_VALID;
                *success = 1;
                return;
            }

            // Try second bucket
            uint idx2 = hash2(hash, table_size);
            if (table[idx2].flags == FLAG_EMPTY) {
                table[idx2] = entry;
                table[idx2].flags = FLAG_VALID;
                *success = 1;
                return;
            }

            // Both full - need to evict and rehash (simplified: just fail)
            *success = 0;
        }
    "#;

    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .expect("Failed to compile shader");

    let lookup_fn = library.get_function("cuckoo_lookup", None).unwrap();
    let insert_fn = library.get_function("cuckoo_insert", None).unwrap();

    let lookup_pipeline = device.new_compute_pipeline_state_with_function(&lookup_fn).unwrap();
    let insert_pipeline = device.new_compute_pipeline_state_with_function(&insert_fn).unwrap();

    // Create hash table (2x capacity for cuckoo)
    let table_size = 64u32;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CacheEntry {
        file_hash: u64,
        slot_index: u32,
        file_size: u32,
        lru_timestamp: u32,
        flags: u32,
    }

    let table: Vec<CacheEntry> = vec![CacheEntry::default(); table_size as usize];
    let table_buffer = device.new_buffer_with_data(
        table.as_ptr() as *const _,
        (table_size as usize * std::mem::size_of::<CacheEntry>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();

    // Insert some entries
    let test_hashes: Vec<u64> = (0..10).map(|i| hash64(&format!("file{}.rs", i))).collect();

    for (i, &hash) in test_hashes.iter().enumerate() {
        let entry = CacheEntry {
            file_hash: hash,
            slot_index: i as u32,
            file_size: 1000 + i as u32,
            lru_timestamp: 0,
            flags: 0, // Will be set to 1 by shader
        };

        let entry_buffer = device.new_buffer_with_data(
            &entry as *const _ as *const _,
            std::mem::size_of::<CacheEntry>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let success_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&insert_pipeline);
        encoder.set_buffer(0, Some(&table_buffer), 0);
        encoder.set_bytes(1, 4, &table_size as *const _ as *const _);
        encoder.set_buffer(2, Some(&entry_buffer), 0);
        encoder.set_buffer(3, Some(&success_buffer), 0);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let success = unsafe { *(success_buffer.contents() as *const u32) };
        assert_eq!(success, 1, "Insert {} failed", i);
    }

    println!("Inserted {} entries", test_hashes.len());

    // Lookup all (should all hit)
    let query_buffer = device.new_buffer_with_data(
        test_hashes.as_ptr() as *const _,
        (test_hashes.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let results_buffer = device.new_buffer(
        (test_hashes.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&lookup_pipeline);
    encoder.set_buffer(0, Some(&table_buffer), 0);
    encoder.set_bytes(1, 4, &table_size as *const _ as *const _);
    encoder.set_buffer(2, Some(&query_buffer), 0);
    encoder.set_buffer(3, Some(&results_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new(test_hashes.len() as u64, 1, 1),
        MTLSize::new(test_hashes.len() as u64, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Check results
    let results_ptr = results_buffer.contents() as *const i32;
    let results: Vec<i32> = unsafe {
        std::slice::from_raw_parts(results_ptr, test_hashes.len()).to_vec()
    };

    for (i, &result) in results.iter().enumerate() {
        assert_eq!(result, i as i32, "Lookup {} failed: got {}", i, result);
    }

    println!("GPU cuckoo hash lookup: OK (all {} hits)", test_hashes.len());

    // Lookup unknown hashes (should all miss)
    let unknown_hashes: Vec<u64> = (100..110).map(|i| hash64(&format!("unknown{}.rs", i))).collect();
    let unknown_buffer = device.new_buffer_with_data(
        unknown_hashes.as_ptr() as *const _,
        (unknown_hashes.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let unknown_results = device.new_buffer(
        (unknown_hashes.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&lookup_pipeline);
    encoder.set_buffer(0, Some(&table_buffer), 0);
    encoder.set_bytes(1, 4, &table_size as *const _ as *const _);
    encoder.set_buffer(2, Some(&unknown_buffer), 0);
    encoder.set_buffer(3, Some(&unknown_results), 0);
    encoder.dispatch_threads(
        MTLSize::new(unknown_hashes.len() as u64, 1, 1),
        MTLSize::new(unknown_hashes.len() as u64, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let unknown_ptr = unknown_results.contents() as *const i32;
    let unknowns: Vec<i32> = unsafe {
        std::slice::from_raw_parts(unknown_ptr, unknown_hashes.len()).to_vec()
    };

    for (i, &result) in unknowns.iter().enumerate() {
        assert_eq!(result, -1, "Unknown {} should miss: got {}", i, result);
    }

    println!("GPU cuckoo hash miss: OK (all {} misses)", unknown_hashes.len());
}

#[test]
fn test_lru_counter_shader() {
    let device = Device::system_default().expect("No Metal device");

    let shader_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        // Update LRU timestamp on access
        kernel void touch_lru(
            device atomic_uint* global_clock [[buffer(0)]],
            device atomic_uint* timestamps [[buffer(1)]],
            device uint* slots_to_touch [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            uint slot = slots_to_touch[tid];
            uint new_time = atomic_fetch_add_explicit(global_clock, 1, memory_order_relaxed);
            atomic_store_explicit(&timestamps[slot], new_time, memory_order_relaxed);
        }

        // Find slot with oldest timestamp (eviction candidate)
        kernel void find_oldest(
            device uint* timestamps [[buffer(0)]],
            constant uint& slot_count [[buffer(1)]],
            device atomic_uint* oldest_slot [[buffer(2)]],
            device atomic_uint* oldest_time [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            // Simple parallel reduction (not optimal, but demonstrates concept)
            uint my_time = timestamps[tid];

            // Use atomicMin pattern (simplified)
            uint old_time = atomic_load_explicit(oldest_time, memory_order_relaxed);
            while (my_time < old_time) {
                if (atomic_compare_exchange_weak_explicit(
                    oldest_time, &old_time, my_time,
                    memory_order_relaxed, memory_order_relaxed)) {
                    atomic_store_explicit(oldest_slot, tid, memory_order_relaxed);
                    break;
                }
            }
        }
    "#;

    let library = device
        .new_library_with_source(shader_source, &CompileOptions::new())
        .expect("Failed to compile shader");

    let touch_fn = library.get_function("touch_lru", None).unwrap();
    let touch_pipeline = device.new_compute_pipeline_state_with_function(&touch_fn).unwrap();

    // Test LRU updates
    let slot_count = 10u32;

    let global_clock = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let timestamps = device.new_buffer(
        (slot_count * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Initialize timestamps to sequential values (0, 1, 2, ...)
    {
        let ptr = timestamps.contents() as *mut u32;
        for i in 0..slot_count {
            unsafe { *ptr.add(i as usize) = i; }
        }
    }

    let queue = device.new_command_queue();

    // Touch slots 0, 2, 4 (should update their timestamps)
    let slots_to_touch: Vec<u32> = vec![0, 2, 4];
    let slots_buffer = device.new_buffer_with_data(
        slots_to_touch.as_ptr() as *const _,
        (slots_to_touch.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&touch_pipeline);
    encoder.set_buffer(0, Some(&global_clock), 0);
    encoder.set_buffer(1, Some(&timestamps), 0);
    encoder.set_buffer(2, Some(&slots_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new(slots_to_touch.len() as u64, 1, 1),
        MTLSize::new(slots_to_touch.len() as u64, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Check that touched slots have higher timestamps
    let ts_ptr = timestamps.contents() as *const u32;
    let ts: Vec<u32> = unsafe {
        std::slice::from_raw_parts(ts_ptr, slot_count as usize).to_vec()
    };

    println!("Timestamps after touch: {:?}", ts);

    // Slots 0, 2, 4 should have timestamps >= 10 (original max)
    assert!(ts[0] >= 10, "Slot 0 should be updated");
    assert!(ts[2] >= 10, "Slot 2 should be updated");
    assert!(ts[4] >= 10, "Slot 4 should be updated");

    // Slots 1, 3, 5, etc. should still have original values
    assert_eq!(ts[1], 1, "Slot 1 should be unchanged");
    assert_eq!(ts[3], 3, "Slot 3 should be unchanged");

    println!("GPU LRU timestamps: OK");
}

#[test]
fn benchmark_cache_lookup_scaling() {
    let device = Device::system_default().expect("No Metal device");

    println!("\n=== Cache Lookup Scaling Benchmark ===\n");

    // Simulate different cache sizes and lookup counts
    for &n in &[100, 1000, 10000] {
        // CPU hash map lookup
        let hashes: Vec<u64> = (0..n).map(|i| hash64(&format!("file{}.rs", i))).collect();

        let cpu_start = Instant::now();
        let mut cpu_hits = 0;
        let map: std::collections::HashMap<u64, usize> = hashes.iter()
            .enumerate()
            .map(|(i, &h)| (h, i))
            .collect();

        for &h in &hashes {
            if map.get(&h).is_some() {
                cpu_hits += 1;
            }
        }
        let cpu_time = cpu_start.elapsed();

        // Note: GPU lookup timing requires full implementation
        // Estimate based on memory bandwidth and parallelism
        let estimated_gpu_us = n as f64 * 0.001; // ~1ns per lookup with GPU parallelism

        println!("N = {:>5}: CPU HashMap {:.2}ms ({} hits), GPU ~{:.2}µs estimated",
            n,
            cpu_time.as_secs_f64() * 1000.0,
            cpu_hits,
            estimated_gpu_us);
    }
}

// ============================================================================
// Tests for full cache implementation (to be implemented)
// ============================================================================

#[test]
#[ignore = "Requires GpuFileCache implementation"]
fn test_cache_hit_rate() {
    // let device = Device::system_default().expect("No Metal device");
    // let cache = GpuFileCache::new(&device, 100)?;  // 100MB cache

    // // Load 500 files
    // let files: Vec<_> = (0..500).map(|i| format!("src/file{}.rs", i)).collect();
    // cache.load(&files);

    // // First lookup - should all hit (just loaded)
    // let result1 = cache.lookup(&files);
    // assert_eq!(result1.hit_rate(), 1.0, "Expected 100% hit rate");

    // // Access same files again - should still hit
    // let result2 = cache.lookup(&files);
    // assert_eq!(result2.hit_rate(), 1.0, "Expected 100% hit rate");

    // // Access different files - should miss
    // let new_files: Vec<_> = (500..1000).map(|i| format!("src/file{}.rs", i)).collect();
    // let result3 = cache.lookup(&new_files);
    // assert_eq!(result3.hit_rate(), 0.0, "Expected 0% hit rate for new files");
}

#[test]
#[ignore = "Requires GpuFileCache implementation"]
fn test_cache_eviction() {
    // let device = Device::system_default().expect("No Metal device");
    // let cache = GpuFileCache::new(&device, 10)?;  // Small cache: 10 slots

    // // Fill cache
    // let files1: Vec<_> = (0..10).map(|i| format!("file{}.rs", i)).collect();
    // cache.load(&files1);

    // // Touch files 5-9 (make 0-4 cold)
    // for i in 5..10 {
    //     cache.touch(&files1[i]);
    // }

    // // Load 5 new files
    // let files2: Vec<_> = (10..15).map(|i| format!("file{}.rs", i)).collect();
    // cache.load(&files2);

    // // Files 0-4 should be evicted
    // for i in 0..5 {
    //     assert!(!cache.lookup(&[files1[i].clone()]).results[0].hit);
    // }

    // // Files 5-9 and 10-14 should be cached
    // for i in 5..10 {
    //     assert!(cache.lookup(&[files1[i].clone()]).results[0].hit);
    // }
    // for f in &files2 {
    //     assert!(cache.lookup(&[f.clone()]).results[0].hit);
    // }
}

#[test]
#[ignore = "Requires GpuFileCache implementation"]
fn benchmark_cache_vs_no_cache() {
    // let device = Device::system_default().expect("No Metal device");

    // // Collect real test files
    // let files = collect_test_files(Path::new("."), 1000);
    // println!("Testing with {} files", files.len());

    // // Without cache - load every time
    // let no_cache_times: Vec<f64> = (0..5).map(|_| {
    //     let start = Instant::now();
    //     let searcher = GpuContentSearch::new(&device);
    //     searcher.load_files(&files);
    //     searcher.search("TODO");
    //     start.elapsed().as_secs_f64() * 1000.0
    // }).collect();

    // // With cache - first load, then cache hits
    // let cache = GpuFileCache::new(&device, 500)?;  // 500MB cache
    // let cache_times: Vec<f64> = (0..5).map(|i| {
    //     let start = Instant::now();
    //     if i == 0 {
    //         cache.load(&files);  // First time: cold load
    //     }
    //     cache.search("TODO");  // Search cached data
    //     start.elapsed().as_secs_f64() * 1000.0
    // }).collect();

    // println!("\nNo cache: {:?}", no_cache_times);
    // println!("With cache: {:?}", cache_times);

    // // Subsequent searches should be 5-10x faster
    // let no_cache_avg: f64 = no_cache_times.iter().sum::<f64>() / 5.0;
    // let cache_warm_avg: f64 = cache_times[1..].iter().sum::<f64>() / 4.0;

    // println!("\nNo cache avg: {:.1}ms", no_cache_avg);
    // println!("Cache warm avg: {:.1}ms", cache_warm_avg);
    // println!("Speedup: {:.1}x", no_cache_avg / cache_warm_avg);

    // assert!(cache_warm_avg < no_cache_avg / 3.0, "Expected 3x+ speedup with warm cache");
}
