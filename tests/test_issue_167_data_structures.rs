//! Issue #167: GPU Data Structures Tests
//!
//! Tests for GPU-native heap allocator, vector, hashmap, and string.

use metal::Device;
use rust_experiment::gpu_os::gpu_heap::{GpuHeap, GpuHashMap, GpuVector, INVALID_OFFSET};

// ═══════════════════════════════════════════════════════════════════════════
// HEAP TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_heap_creation() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 1024 * 1024).expect("Failed to create heap");

    let stats = heap.read_stats();
    assert_eq!(stats.heap_size, 1024 * 1024);
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.total_allocated, 0);
    println!("Heap created: {} bytes", stats.heap_size);
}

#[test]
fn test_heap_single_alloc() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 1024 * 1024).expect("Failed to create heap");

    let offsets = heap.alloc_batch(&device, &[64]);
    assert_eq!(offsets.len(), 1);
    assert_ne!(offsets[0], INVALID_OFFSET);

    let stats = heap.read_stats();
    assert_eq!(stats.allocation_count, 1);
    println!("Single alloc at offset: {}", offsets[0]);
}

#[test]
fn test_heap_batch_alloc() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");

    // Allocate 100 blocks of various sizes
    let sizes: Vec<u32> = (0..100).map(|i| 64 + (i % 8) * 64).collect();
    let offsets = heap.alloc_batch(&device, &sizes);

    assert_eq!(offsets.len(), 100);
    for (i, offset) in offsets.iter().enumerate() {
        assert_ne!(*offset, INVALID_OFFSET, "Allocation {} failed", i);
    }

    let stats = heap.read_stats();
    assert_eq!(stats.allocation_count, 100);
    println!("Batch alloc: {} allocations, {} bytes used", stats.allocation_count, stats.total_allocated);
}

#[test]
fn test_heap_alloc_free_cycle() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 1024 * 1024).expect("Failed to create heap");

    // Allocate
    let sizes = vec![64, 128, 256, 512, 1024];
    let offsets = heap.alloc_batch(&device, &sizes);

    let stats_after_alloc = heap.read_stats();
    assert_eq!(stats_after_alloc.allocation_count, 5);
    println!("After alloc: {} blocks, {} bytes", stats_after_alloc.allocation_count, stats_after_alloc.total_allocated);

    // Free all
    heap.free_batch(&device, &offsets);

    let stats_after_free = heap.read_stats();
    assert_eq!(stats_after_free.allocation_count, 0);
    println!("After free: {} blocks, {} bytes", stats_after_free.allocation_count, stats_after_free.total_allocated);

    // Verify free list counts increased
    let total_free: u32 = stats_after_free.free_list_counts.iter().sum();
    assert!(total_free > 0, "Free list should have entries");
    println!("Free list entries: {:?}", stats_after_free.free_list_counts);
}

#[test]
fn test_heap_reuse_freed_blocks() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 1024 * 1024).expect("Failed to create heap");

    // Allocate 64-byte blocks
    let sizes = vec![32; 10]; // Will use 64-byte size class
    let offsets1 = heap.alloc_batch(&device, &sizes);

    let bump_after_first = heap.read_stats().bump_ptr;

    // Free them
    heap.free_batch(&device, &offsets1);

    // Allocate again - should reuse freed blocks
    let offsets2 = heap.alloc_batch(&device, &sizes);

    let bump_after_second = heap.read_stats().bump_ptr;

    // Bump pointer should not have advanced (blocks were reused from free list)
    assert_eq!(bump_after_first, bump_after_second, "Should reuse freed blocks");
    println!("Freed blocks reused successfully");
}

// ═══════════════════════════════════════════════════════════════════════════
// HASHMAP TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_hashmap_creation() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");
    let map = GpuHashMap::new(&device, &heap, 256).expect("Failed to create hashmap");

    assert_eq!(map.capacity(), 256);
    assert_ne!(map.offset(), INVALID_OFFSET);
    println!("HashMap created: capacity={}, offset={}", map.capacity(), map.offset());
}

#[test]
fn test_hashmap_insert_get() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");
    let map = GpuHashMap::new(&device, &heap, 256).expect("Failed to create hashmap");

    // Insert key-value pairs
    let keys = vec![1, 2, 3, 4, 5, 100, 200, 300];
    let values = vec![10, 20, 30, 40, 50, 1000, 2000, 3000];
    let insert_results = map.insert_batch(&device, &heap, &keys, &values);

    // All should succeed
    for (i, r) in insert_results.iter().enumerate() {
        assert_eq!(*r, 1, "Insert {} failed", i);
    }
    println!("Inserted {} key-value pairs", keys.len());

    // Look them up
    let (got_values, found) = map.get_batch(&device, &heap, &keys);

    for i in 0..keys.len() {
        assert_eq!(found[i], 1, "Key {} not found", keys[i]);
        assert_eq!(got_values[i], values[i], "Value mismatch for key {}", keys[i]);
    }
    println!("All lookups successful");
}

#[test]
fn test_hashmap_not_found() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");
    let map = GpuHashMap::new(&device, &heap, 256).expect("Failed to create hashmap");

    // Insert some keys
    map.insert_batch(&device, &heap, &[1, 2, 3], &[10, 20, 30]);

    // Look up non-existent keys
    let (_, found) = map.get_batch(&device, &heap, &[999, 888, 777]);

    for (i, f) in found.iter().enumerate() {
        assert_eq!(*f, 0, "Non-existent key {} should not be found", i);
    }
    println!("Non-existent keys correctly not found");
}

#[test]
fn test_hashmap_update_existing() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");
    let map = GpuHashMap::new(&device, &heap, 256).expect("Failed to create hashmap");

    // Insert
    map.insert_batch(&device, &heap, &[42], &[100]);

    // Verify
    let (values, found) = map.get_batch(&device, &heap, &[42]);
    assert_eq!(found[0], 1);
    assert_eq!(values[0], 100);

    // Update with new value
    map.insert_batch(&device, &heap, &[42], &[999]);

    // Verify updated
    let (values, found) = map.get_batch(&device, &heap, &[42]);
    assert_eq!(found[0], 1);
    assert_eq!(values[0], 999);
    println!("Update existing key successful");
}

#[test]
fn test_hashmap_batch_performance() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 64 * 1024 * 1024).expect("Failed to create heap");
    let map = GpuHashMap::new(&device, &heap, 16384).expect("Failed to create hashmap");

    // Insert 1000 key-value pairs
    let keys: Vec<u32> = (0..1000).collect();
    let values: Vec<u32> = keys.iter().map(|k| k * 10).collect();

    let start = std::time::Instant::now();
    let results = map.insert_batch(&device, &heap, &keys, &values);
    let insert_time = start.elapsed();

    let success_count = results.iter().filter(|&&r| r == 1).count();
    println!("Inserted {}/{} in {:?}", success_count, keys.len(), insert_time);

    // Lookup all
    let start = std::time::Instant::now();
    let (got_values, found) = map.get_batch(&device, &heap, &keys);
    let lookup_time = start.elapsed();

    let found_count = found.iter().filter(|&&f| f == 1).count();
    println!("Found {}/{} in {:?}", found_count, keys.len(), lookup_time);

    // Verify correctness
    for i in 0..keys.len() {
        if found[i] == 1 {
            assert_eq!(got_values[i], values[i]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VECTOR TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_vector_creation() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 4 * 1024 * 1024).expect("Failed to create heap");
    let vec = GpuVector::new(&device, &heap, 4, 100).expect("Failed to create vector");

    assert_ne!(vec.offset(), INVALID_OFFSET);
    println!("Vector created at offset: {}", vec.offset());
}

// ═══════════════════════════════════════════════════════════════════════════
// STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Run with --ignored
fn test_heap_stress() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 256 * 1024 * 1024).expect("Failed to create heap");

    // Allocate 10000 blocks
    let sizes: Vec<u32> = (0..10000).map(|i| 64 + (i % 8) * 64).collect();

    let start = std::time::Instant::now();
    let offsets = heap.alloc_batch(&device, &sizes);
    let alloc_time = start.elapsed();

    let success = offsets.iter().filter(|&&o| o != INVALID_OFFSET).count();
    println!("Allocated {}/{} blocks in {:?}", success, sizes.len(), alloc_time);

    let stats = heap.read_stats();
    println!("Heap stats: {} blocks, {} bytes, bump_ptr={}",
             stats.allocation_count, stats.total_allocated, stats.bump_ptr);

    // Free all
    let start = std::time::Instant::now();
    heap.free_batch(&device, &offsets);
    let free_time = start.elapsed();

    println!("Freed all blocks in {:?}", free_time);

    let stats = heap.read_stats();
    assert_eq!(stats.allocation_count, 0);
}

#[test]
#[ignore] // Run with --ignored
fn test_hashmap_stress() {
    let device = Device::system_default().expect("No Metal device");
    let heap = GpuHeap::new(&device, 128 * 1024 * 1024).expect("Failed to create heap");
    let map = GpuHashMap::new(&device, &heap, 65536).expect("Failed to create hashmap");

    // Insert 10000 key-value pairs
    let keys: Vec<u32> = (0..10000).collect();
    let values: Vec<u32> = keys.iter().map(|k| k * 7).collect();

    let start = std::time::Instant::now();
    let results = map.insert_batch(&device, &heap, &keys, &values);
    let insert_time = start.elapsed();

    let success = results.iter().filter(|&&r| r == 1).count();
    println!("Inserted {}/{} in {:?}", success, keys.len(), insert_time);
    println!("Throughput: {:.2}M inserts/sec", (success as f64) / insert_time.as_secs_f64() / 1_000_000.0);

    // Lookup all
    let start = std::time::Instant::now();
    let (got_values, found) = map.get_batch(&device, &heap, &keys);
    let lookup_time = start.elapsed();

    let found_count = found.iter().filter(|&&f| f == 1).count();
    println!("Found {}/{} in {:?}", found_count, keys.len(), lookup_time);
    println!("Throughput: {:.2}M lookups/sec", (found_count as f64) / lookup_time.as_secs_f64() / 1_000_000.0);

    // Verify
    let mut correct = 0;
    for i in 0..keys.len() {
        if found[i] == 1 && got_values[i] == values[i] {
            correct += 1;
        }
    }
    println!("Correct: {}/{}", correct, found_count);
}
