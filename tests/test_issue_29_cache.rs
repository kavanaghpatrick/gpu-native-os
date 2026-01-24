// Tests for Issue #29: GPU-Native Hash Table Cache
//
// These tests verify the GPU-side caching implementation for hot paths.
// Run with: cargo test --test test_issue_29_cache

use metal::Device;
use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};

#[test]
fn test_cache_entry_size_is_64_bytes() {
    use rust_experiment::gpu_os::filesystem::PathCacheEntry;
    assert_eq!(
        std::mem::size_of::<PathCacheEntry>(), 64,
        "PathCacheEntry must be 64 bytes for cache-line alignment"
    );
}

#[test]
fn test_cache_miss_on_first_lookup() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create test file
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    fs.add_file(src, "main.rs", FileType::Regular).unwrap();

    // First lookup - should be a cache miss
    let paths = vec!["/src/main.rs"];
    let results = fs.lookup_batch(&paths).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_ok());

    // Check stats - should have 1 miss
    let stats = fs.cache_stats();
    assert_eq!(stats.misses, 1, "First lookup should be a cache miss");
    assert_eq!(stats.hits, 0, "No cache hits yet");
}

#[test]
fn test_cache_hit_on_repeated_lookup() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create test file
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    fs.add_file(src, "main.rs", FileType::Regular).unwrap();

    let paths = vec!["/src/main.rs"];

    // First lookup - cache miss
    let result1 = fs.lookup_batch(&paths).unwrap();
    assert!(result1[0].is_ok());

    // Second lookup - should be cache hit
    let result2 = fs.lookup_batch(&paths).unwrap();
    assert!(result2[0].is_ok());
    assert_eq!(result1[0], result2[0], "Same path should return same inode");

    // Check stats
    let stats = fs.cache_stats();
    assert_eq!(stats.hits, 1, "Second lookup should be a cache hit");
    assert_eq!(stats.misses, 1, "Should still have 1 miss from first lookup");
    assert_eq!(stats.hit_rate, 0.5, "50% hit rate (1 hit, 1 miss)");
}

#[test]
fn test_cache_with_multiple_paths() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create test files
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    fs.add_file(src, "main.rs", FileType::Regular).unwrap();
    fs.add_file(src, "lib.rs", FileType::Regular).unwrap();
    fs.add_file(src, "utils.rs", FileType::Regular).unwrap();

    // First batch - all misses
    let paths = vec!["/src/main.rs", "/src/lib.rs", "/src/utils.rs"];
    let results = fs.lookup_batch(&paths).unwrap();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.is_ok()));

    let stats1 = fs.cache_stats();
    assert_eq!(stats1.misses, 3, "All paths should miss on first lookup");
    assert_eq!(stats1.hits, 0);

    // Second batch - all hits
    let results2 = fs.lookup_batch(&paths).unwrap();
    assert_eq!(results2.len(), 3);
    assert_eq!(results, results2, "Results should be identical");

    let stats2 = fs.cache_stats();
    assert_eq!(stats2.hits, 3, "All paths should hit on second lookup");
    assert_eq!(stats2.misses, 3, "Miss count should not increase");
    assert_eq!(stats2.hit_rate, 0.5, "50% hit rate (3 hits, 3 misses)");
}

#[test]
fn test_cache_with_mixed_hits_and_misses() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create test files
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    fs.add_file(src, "main.rs", FileType::Regular).unwrap();
    fs.add_file(src, "lib.rs", FileType::Regular).unwrap();
    fs.add_file(src, "utils.rs", FileType::Regular).unwrap();

    // Prime cache with main.rs and lib.rs
    let paths1 = vec!["/src/main.rs", "/src/lib.rs"];
    fs.lookup_batch(&paths1).unwrap();

    // Mixed batch - 2 hits (main.rs, lib.rs) + 1 miss (utils.rs)
    let paths2 = vec!["/src/main.rs", "/src/lib.rs", "/src/utils.rs"];
    let results = fs.lookup_batch(&paths2).unwrap();
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(|r| r.is_ok()));

    let stats = fs.cache_stats();
    assert_eq!(stats.hits, 2, "main.rs and lib.rs should hit");
    assert_eq!(stats.misses, 3, "2 from first batch + 1 from utils.rs");
    assert!(stats.hit_rate > 0.3 && stats.hit_rate < 0.5, "Hit rate should be ~40%");
}

#[test]
fn test_cache_stats_accuracy() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create files
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    for i in 0..10 {
        fs.add_file(src, &format!("file{}.rs", i), FileType::Regular).unwrap();
    }

    // Initial stats
    let stats0 = fs.cache_stats();
    assert_eq!(stats0.hits, 0);
    assert_eq!(stats0.misses, 0);
    assert_eq!(stats0.hit_rate, 0.0);
    assert_eq!(stats0.total_entries, 0, "Cache should be empty");

    // Lookup 10 files - all misses
    let mut paths = Vec::new();
    for i in 0..10 {
        paths.push(format!("/src/file{}.rs", i));
    }
    let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    fs.lookup_batch(&path_refs).unwrap();

    let stats1 = fs.cache_stats();
    assert_eq!(stats1.misses, 10);
    assert_eq!(stats1.hits, 0);
    assert!(stats1.total_entries > 0, "Cache should have entries now");

    // Repeat - all hits
    fs.lookup_batch(&path_refs).unwrap();

    let stats2 = fs.cache_stats();
    assert_eq!(stats2.misses, 10);
    assert_eq!(stats2.hits, 10);
    assert_eq!(stats2.hit_rate, 0.5, "50% hit rate");
}

#[test]
fn test_clear_cache_resets_stats() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create test file
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    fs.add_file(src, "main.rs", FileType::Regular).unwrap();

    // Generate some stats
    let paths = vec!["/src/main.rs"];
    fs.lookup_batch(&paths).unwrap(); // Miss
    fs.lookup_batch(&paths).unwrap(); // Hit

    let stats_before = fs.cache_stats();
    assert_eq!(stats_before.hits, 1);
    assert_eq!(stats_before.misses, 1);

    // Clear cache
    fs.clear_cache();

    // Stats should be reset
    let stats_after = fs.cache_stats();
    assert_eq!(stats_after.hits, 0, "Hits should be reset");
    assert_eq!(stats_after.misses, 0, "Misses should be reset");
    assert_eq!(stats_after.hit_rate, 0.0, "Hit rate should be 0");
    assert_eq!(stats_after.total_entries, 0, "Cache should be empty");

    // Next lookup should be a miss again
    fs.lookup_batch(&paths).unwrap();
    let stats_final = fs.cache_stats();
    assert_eq!(stats_final.misses, 1);
    assert_eq!(stats_final.hits, 0);
}

#[test]
fn test_cache_handles_not_found_paths() {
    let device = Device::system_default().expect("No Metal device found");
    let fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Lookup non-existent path
    let paths = vec!["/nonexistent/path.rs"];
    let results = fs.lookup_batch(&paths).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_err(), "Non-existent path should return error");

    // Stats should still count it as a miss
    let stats = fs.cache_stats();
    assert_eq!(stats.misses, 1, "Failed lookups should still count as cache misses");
}

#[test]
fn test_cache_collision_overwrites_old_entry() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    // Create many files to force hash collisions (direct-mapped cache with 1024 slots)
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();

    // Create paths that will eventually collide in a 1024-entry cache
    // We can't easily predict collisions, but we can create enough entries
    // to verify the cache handles updates correctly
    for i in 0..20 {
        fs.add_file(src, &format!("file{:03}.rs", i), FileType::Regular).unwrap();
    }

    let mut paths = Vec::new();
    for i in 0..20 {
        paths.push(format!("/src/file{:03}.rs", i));
    }
    let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();

    // First lookup - all misses
    fs.lookup_batch(&path_refs).unwrap();

    let stats1 = fs.cache_stats();
    assert_eq!(stats1.misses, 20);

    // Second lookup - all hits (if no collisions) or some hits
    fs.lookup_batch(&path_refs).unwrap();

    let stats2 = fs.cache_stats();
    // Should have at least some hits
    assert!(stats2.hits > 0, "Should have cache hits on repeated lookups");
}

#[test]
fn test_cache_with_different_batch_sizes() {
    let device = Device::system_default().expect("No Metal device found");
    let mut fs = GpuFilesystem::new(&device, 256).expect("Failed to create filesystem");

    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    fs.add_file(src, "a.rs", FileType::Regular).unwrap();
    fs.add_file(src, "b.rs", FileType::Regular).unwrap();

    // Single path batch
    let single = vec!["/src/a.rs"];
    fs.lookup_batch(&single).unwrap(); // Miss
    fs.lookup_batch(&single).unwrap(); // Hit

    let stats1 = fs.cache_stats();
    assert_eq!(stats1.hits, 1);
    assert_eq!(stats1.misses, 1);

    // Multiple path batch
    let multiple = vec!["/src/a.rs", "/src/b.rs"];
    fs.lookup_batch(&multiple).unwrap(); // 1 hit (a.rs), 1 miss (b.rs)

    let stats2 = fs.cache_stats();
    assert_eq!(stats2.hits, 2, "a.rs should hit again");
    assert_eq!(stats2.misses, 2, "b.rs should miss");
}
