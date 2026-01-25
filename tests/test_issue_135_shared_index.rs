//! Issue #135: Shared GPU-Resident Filesystem Index
//!
//! Tests for the unified filesystem index that all GPU tools share.

use metal::Device;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::tempdir;

// Import the module (will be created)
use rust_experiment::gpu_os::shared_index::{
    GpuFilesystemIndex, IndexManifest, IndexInfo, DEFAULT_EXCLUDES,
};

#[test]
fn test_index_directory_creation() {
    let temp = tempdir().unwrap();
    let device = Device::system_default().expect("No Metal device");

    let fs_index = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to create index");

    // Check directory structure
    assert!(temp.path().join("index").exists(), "index/ should exist");
    assert!(temp.path().join("index/manifest.json").exists(), "manifest.json should exist");
}

#[test]
fn test_manifest_serialization() {
    let manifest = IndexManifest {
        version: 1,
        indexes: vec![
            IndexInfo {
                name: "home".to_string(),
                path: PathBuf::from("/tmp/home.idx"),
                root: PathBuf::from("/Users/test"),
                entry_count: 50000,
                size_bytes: 12800000,
                built_at: chrono::Utc::now(),
                exclude_patterns: vec![".git".to_string(), "node_modules".to_string()],
            }
        ],
        last_updated: chrono::Utc::now(),
    };

    // Serialize
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert!(json.contains("\"name\": \"home\""));
    assert!(json.contains("\"entry_count\": 50000"));

    // Deserialize
    let parsed: IndexManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.version, 1);
    assert_eq!(parsed.indexes.len(), 1);
    assert_eq!(parsed.indexes[0].name, "home");
}

#[test]
fn test_shared_index_load_time() {
    let temp = tempdir().unwrap();
    let device = Device::system_default().expect("No Metal device");

    // First load (builds index)
    let _ = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to create index");

    // Second load (should be instant via mmap)
    let start = Instant::now();
    let fs_index = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to load index");
    let load_time = start.elapsed();

    println!("Index load time: {:?}", load_time);
    println!("Total entries: {}", fs_index.total_entries());

    // Should load in under 50ms (mmap is nearly instant)
    assert!(load_time < Duration::from_millis(50),
        "Index load took {:?}, expected < 50ms", load_time);
}

#[test]
fn test_gpu_buffer_access() {
    let device = Device::system_default().expect("No Metal device");

    // Use real home directory for meaningful test
    let fs_index = GpuFilesystemIndex::load_or_create(&device)
        .expect("Failed to load index");

    if fs_index.total_entries() == 0 {
        println!("Skipping buffer test - no entries in index");
        return;
    }

    // Get GPU buffer for home index
    let buffer = fs_index.get_buffer("home");
    assert!(buffer.is_some(), "Should have home buffer");

    let buffer = buffer.unwrap();
    println!("Home buffer size: {} bytes", buffer.length());

    // Get all buffers for parallel search
    let all_buffers = fs_index.all_buffers();
    println!("Total indexes: {}", all_buffers.len());

    for (name, buf, count) in &all_buffers {
        println!("  {}: {} entries, {} bytes", name, count, buf.length());
    }
}

#[test]
fn test_stale_detection() {
    let temp = tempdir().unwrap();
    let device = Device::system_default().expect("No Metal device");

    let fs_index = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to create index");

    // Just built, shouldn't be stale with 1 hour threshold
    assert!(!fs_index.is_stale(Duration::from_secs(3600)),
        "Fresh index should not be stale");

    // With 0 duration, everything is stale
    assert!(fs_index.is_stale(Duration::from_secs(0)),
        "Index should be stale with 0 threshold");
}

#[test]
fn test_default_excludes() {
    // Verify default exclude patterns are reasonable
    assert!(DEFAULT_EXCLUDES.contains(&".git"));
    assert!(DEFAULT_EXCLUDES.contains(&"node_modules"));
    assert!(DEFAULT_EXCLUDES.contains(&"target"));
    assert!(DEFAULT_EXCLUDES.contains(&".cache"));

    println!("Default excludes: {:?}", DEFAULT_EXCLUDES);
}

#[test]
fn test_memory_usage_tracking() {
    let temp = tempdir().unwrap();
    let device = Device::system_default().expect("No Metal device");

    let fs_index = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to create index");

    let memory = fs_index.memory_usage();
    let entries = fs_index.total_entries();

    println!("Memory usage: {} bytes", memory);
    println!("Entry count: {}", entries);

    if entries > 0 {
        let bytes_per_entry = memory as f64 / entries as f64;
        println!("Bytes per entry: {:.1}", bytes_per_entry);

        // Should be reasonable (< 1KB per entry)
        assert!(bytes_per_entry < 1024.0,
            "Memory usage too high: {} bytes/entry", bytes_per_entry);
    }
}

#[test]
fn test_rebuild_index() {
    let temp = tempdir().unwrap();
    let device = Device::system_default().expect("No Metal device");

    // Create initial index
    let mut fs_index = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to create index");

    let initial_entries = fs_index.total_entries();

    // Rebuild
    fs_index.rebuild_all().expect("Failed to rebuild");

    let rebuilt_entries = fs_index.total_entries();

    println!("Initial entries: {}", initial_entries);
    println!("Rebuilt entries: {}", rebuilt_entries);

    // Should have same or similar count after rebuild
    // (exact match not guaranteed due to filesystem changes)
}

#[test]
fn test_get_home_index() {
    let device = Device::system_default().expect("No Metal device");

    let fs_index = GpuFilesystemIndex::load_or_create(&device)
        .expect("Failed to load index");

    // Should have a "home" index by default
    let home_index = fs_index.get("home");
    assert!(home_index.is_some(), "Should have 'home' index");

    // Convenience method
    let home_index2 = fs_index.home();
    assert!(home_index2.is_some(), "home() should return index");
}

#[test]
fn test_index_info_metadata() {
    let temp = tempdir().unwrap();
    let device = Device::system_default().expect("No Metal device");

    let fs_index = GpuFilesystemIndex::load_or_create_at(&device, temp.path())
        .expect("Failed to create index");

    let manifest = fs_index.manifest();

    assert_eq!(manifest.version, 1);
    assert!(!manifest.indexes.is_empty());

    for info in &manifest.indexes {
        println!("Index '{}': {} entries, {} bytes",
            info.name, info.entry_count, info.size_bytes);
        assert!(!info.name.is_empty());
        assert!(info.path.exists() || info.entry_count == 0);
    }
}

// =============================================================================
// BENCHMARKS
// =============================================================================

#[test]
fn benchmark_shared_vs_per_tool_index() {
    let device = Device::system_default().expect("No Metal device");
    let iterations = 3;

    println!("\n=== Shared Index vs Per-Tool Index ===\n");

    // Ensure shared index exists
    let _ = GpuFilesystemIndex::load_or_create(&device)
        .expect("Failed to create shared index");

    // Benchmark: Load shared index (warm cache)
    let mut shared_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = GpuFilesystemIndex::load_or_create(&device).unwrap();
        shared_times.push(start.elapsed());
    }

    // Benchmark: Build per-tool index (simulates old behavior)
    let mut build_times = Vec::new();
    for _ in 0..iterations {
        let temp = tempdir().unwrap();
        let start = Instant::now();
        let _ = GpuFilesystemIndex::load_or_create_at(&device, temp.path()).unwrap();
        build_times.push(start.elapsed());
    }

    let shared_avg = shared_times.iter().map(|d| d.as_secs_f64()).sum::<f64>()
                     / iterations as f64 * 1000.0;
    let build_avg = build_times.iter().map(|d| d.as_secs_f64()).sum::<f64>()
                    / iterations as f64 * 1000.0;

    println!("Shared index load: {:.1}ms (average of {})", shared_avg, iterations);
    println!("Fresh index build: {:.1}ms (average of {})", build_avg, iterations);

    if build_avg > shared_avg {
        println!("Speedup:           {:.1}x", build_avg / shared_avg);
    }
}

#[test]
fn benchmark_buffer_access_latency() {
    let device = Device::system_default().expect("No Metal device");
    let fs_index = GpuFilesystemIndex::load_or_create(&device)
        .expect("Failed to load index");

    if fs_index.total_entries() == 0 {
        println!("Skipping buffer benchmark - no entries");
        return;
    }

    let iterations = 100;

    println!("\n=== Buffer Access Latency ({} entries) ===\n", fs_index.total_entries());

    // Benchmark: Get buffer reference (should be instant)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fs_index.get_buffer("home");
    }
    let total = start.elapsed();
    let per_access = total.as_nanos() as f64 / iterations as f64;
    println!("  get_buffer(): {:.0}ns per call", per_access);

    // Benchmark: Get all buffers
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fs_index.all_buffers();
    }
    let total = start.elapsed();
    let per_access = total.as_nanos() as f64 / iterations as f64;
    println!("  all_buffers(): {:.0}ns per call", per_access);

    // Note: Actual GPU search would use GpuPathSearch with these buffers
    println!("\n  (Use GpuPathSearch for GPU-accelerated search)");
}

#[test]
fn benchmark_memory_efficiency() {
    let device = Device::system_default().expect("No Metal device");
    let fs_index = GpuFilesystemIndex::load_or_create(&device)
        .expect("Failed to load index");

    let memory = fs_index.memory_usage();
    let entries = fs_index.total_entries();

    println!("\n=== Memory Efficiency ===\n");
    println!("  Total entries: {}", entries);
    println!("  Memory usage:  {} bytes ({:.1} MB)",
        memory, memory as f64 / (1024.0 * 1024.0));

    if entries > 0 {
        let bytes_per_entry = memory as f64 / entries as f64;
        println!("  Per entry:     {:.1} bytes", bytes_per_entry);

        // Compare to naive approach (storing full paths)
        // Average path is ~50 chars, so ~50 bytes minimum
        println!("  Efficiency:    ~{:.1}x vs naive path storage",
            50.0 / bytes_per_entry.min(50.0));
    }
}
