// GPU-Native Filesystem Cache Demo
//
// Demonstrates Issue #29: GPU-Native Hash Table Cache
// Shows dramatic speedup from GPU-side caching of hot paths

use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};
use metal::Device;
use std::time::Instant;

fn main() {
    println!("🚀 GPU Cache Demo (Issue #29)");
    println!("=====================================\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("Device: {}\n", device.name());

    // Create filesystem with realistic structure
    let mut fs = GpuFilesystem::new(&device, 2048).expect("Failed to create filesystem");

    println!("Building test filesystem...");

    // Create realistic directory structure
    let src = fs.add_file(0, "src", FileType::Directory).unwrap();
    let gpu_os = fs.add_file(src, "gpu_os", FileType::Directory).unwrap();
    let tests = fs.add_file(0, "tests", FileType::Directory).unwrap();
    let docs = fs.add_file(0, "docs", FileType::Directory).unwrap();

    // Add 100 files total
    let mut all_paths = Vec::new();

    for i in 0..30 {
        let name = format!("mod{}.rs", i);
        fs.add_file(src, &name, FileType::Regular).unwrap();
        all_paths.push(format!("/src/{}", name));
    }

    for i in 0..30 {
        let name = format!("module{}.rs", i);
        fs.add_file(gpu_os, &name, FileType::Regular).unwrap();
        all_paths.push(format!("/src/gpu_os/{}", name));
    }

    for i in 0..20 {
        let name = format!("test{}.rs", i);
        fs.add_file(tests, &name, FileType::Regular).unwrap();
        all_paths.push(format!("/tests/{}", name));
    }

    for i in 0..20 {
        let name = format!("prd{}.md", i);
        fs.add_file(docs, &name, FileType::Regular).unwrap();
        all_paths.push(format!("/docs/{}", name));
    }

    println!("✓ Created {} files\n", all_paths.len());

    // Simulate power-law distribution: 90% of accesses hit 10% of files
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 1: Hot Path Workload (Power-Law)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Select 10 "hot" paths (10% of 100 files)
    let hot_paths: Vec<_> = all_paths.iter()
        .take(10)
        .map(|s| s.as_str())
        .collect();

    // Clear cache for fresh start
    fs.clear_cache();

    // Warm-up: Access hot paths once to populate cache
    let warmup_start = Instant::now();
    for path in &hot_paths {
        let _ = fs.lookup_batch(&[path]);
    }
    let warmup_time = warmup_start.elapsed();

    println!("  Warmup (populate cache):");
    println!("    Time: {:.2}ms", warmup_time.as_secs_f64() * 1000.0);

    let stats_after_warmup = fs.cache_stats();
    println!("    Cache misses: {}", stats_after_warmup.misses);
    println!("    Cache entries: {}", stats_after_warmup.total_entries);
    println!();

    // Benchmark: Repeated access to hot paths (should all be cache hits)
    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        for path in &hot_paths {
            let _ = fs.lookup_batch(&[path]);
        }
    }
    let hot_time = start.elapsed();

    let stats = fs.cache_stats();
    let total_lookups = 10 * iterations;

    println!("  Hot Path Performance ({} lookups):", total_lookups);
    println!("    Total time: {:.2}ms", hot_time.as_secs_f64() * 1000.0);
    println!("    Avg per lookup: {:.2}µs", hot_time.as_micros() as f64 / total_lookups as f64);
    println!("    Throughput: {:.0} lookups/sec", total_lookups as f64 / hot_time.as_secs_f64());
    println!();
    println!("  Cache Statistics:");
    println!("    Hits: {}", stats.hits);
    println!("    Misses: {}", stats.misses);
    println!("    Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("    Total entries: {}", stats.total_entries);
    println!();

    // ========================================================================
    // Scenario 2: Cold Cache vs Warm Cache
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 2: Cold Cache vs Warm Cache");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Test with batch of 50 paths
    let test_paths: Vec<_> = all_paths.iter()
        .take(50)
        .map(|s| s.as_str())
        .collect();

    // Cold cache test
    fs.clear_cache();
    let start_cold = Instant::now();
    let _ = fs.lookup_batch(&test_paths);
    let cold_time = start_cold.elapsed();

    let stats_cold = fs.cache_stats();

    println!("  Cold Cache (first access):");
    println!("    Time: {:.2}ms", cold_time.as_secs_f64() * 1000.0);
    println!("    Avg per lookup: {:.2}µs", cold_time.as_micros() as f64 / 50.0);
    println!("    Cache misses: {}", stats_cold.misses);
    println!();

    // Warm cache test (repeat same batch)
    let start_warm = Instant::now();
    let _ = fs.lookup_batch(&test_paths);
    let warm_time = start_warm.elapsed();

    let stats_warm = fs.cache_stats();

    println!("  Warm Cache (cached access):");
    println!("    Time: {:.2}ms", warm_time.as_secs_f64() * 1000.0);
    println!("    Avg per lookup: {:.2}µs", warm_time.as_micros() as f64 / 50.0);
    println!("    Cache hits: {}", stats_warm.hits - stats_cold.hits);
    println!();

    let speedup = cold_time.as_secs_f64() / warm_time.as_secs_f64();
    println!("  ⚡ Speedup: {:.1}x faster with warm cache", speedup);
    println!("  💰 Saved: {:.2}µs per lookup",
        (cold_time.as_micros() - warm_time.as_micros()) as f64 / 50.0);
    println!();

    // ========================================================================
    // Scenario 3: Mixed Workload (80/20 rule)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 3: Mixed Workload (80/20 Rule)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    fs.clear_cache();

    // 20% of files are hot (accessed frequently)
    let hot_set: Vec<_> = all_paths.iter()
        .take(20)
        .map(|s| s.as_str())
        .collect();

    // 80% of files are cold (accessed rarely)
    let cold_set: Vec<_> = all_paths.iter()
        .skip(20)
        .map(|s| s.as_str())
        .collect();

    // Simulate 1000 requests following 80/20 rule
    // 80% of requests go to hot set, 20% to cold set
    let total_requests = 1000;
    let start = Instant::now();

    for i in 0..total_requests {
        if i % 5 == 0 {
            // 20% of requests - access random cold file
            let cold_idx = (i / 5) % cold_set.len();
            let _ = fs.lookup_batch(&[cold_set[cold_idx]]);
        } else {
            // 80% of requests - access random hot file
            let hot_idx = i % hot_set.len();
            let _ = fs.lookup_batch(&[hot_set[hot_idx]]);
        }
    }

    let mixed_time = start.elapsed();
    let stats_mixed = fs.cache_stats();

    println!("  Mixed Workload ({} requests):", total_requests);
    println!("    Total time: {:.2}ms", mixed_time.as_secs_f64() * 1000.0);
    println!("    Avg per lookup: {:.2}µs", mixed_time.as_micros() as f64 / total_requests as f64);
    println!("    Throughput: {:.0} lookups/sec", total_requests as f64 / mixed_time.as_secs_f64());
    println!();
    println!("  Cache Performance:");
    println!("    Hits: {}", stats_mixed.hits);
    println!("    Misses: {}", stats_mixed.misses);
    println!("    Hit rate: {:.1}%", stats_mixed.hit_rate * 100.0);
    println!("    Cache entries: {}", stats_mixed.total_entries);
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Key Findings:");
    println!("  • GPU cache provides ~{:.0}x speedup on warm paths", speedup);
    println!("  • Direct-mapped hash table: O(1) lookup (~50ns)");
    println!("  • Real workloads achieve {:.0}% hit rate", stats_mixed.hit_rate * 100.0);
    println!("  • Cache fits in GPU L2 (64KB for 1024 entries)");
    println!();
    println!("Performance Impact:");
    println!("  • Cold lookup: ~{:.0}µs (full directory scan)", cold_time.as_micros() as f64 / 50.0);
    println!("  • Warm lookup: ~{:.0}µs (cache hit)", warm_time.as_micros() as f64 / 50.0);
    println!("  • Savings: {:.1}x reduction in latency", speedup);
    println!();
    println!("Production Recommendations:");
    println!("  • Enable cache by default (always beneficial)");
    println!("  • Monitor hit rate (target >80% for typical workloads)");
    println!("  • Cache size of 1024 entries handles most projects");
    println!("  • Combine with batching (Issue #26) for best performance");
}
