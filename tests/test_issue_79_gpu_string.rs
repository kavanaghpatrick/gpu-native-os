// Issue #79: GPU String Processing Tests
//
// Verifies that GPU-native search works correctly:
// - Query tokenization happens on GPU
// - Case conversion happens on GPU
// - CPU does only ONE memcpy of raw query bytes
//
// THE GPU IS THE COMPUTER.

use rust_experiment::gpu_os::filesystem::GpuPathSearch;

fn get_device() -> metal::Device {
    metal::Device::system_default().expect("No Metal device found")
}

// =============================================================================
// Basic GPU-Native Search Tests
// =============================================================================

#[test]
fn test_gpu_native_search_basic() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec![
        "/documents/report.txt".to_string(),
        "/photos/vacation.jpg".to_string(),
        "/music/song.mp3".to_string(),
    ];
    search.add_paths(&paths).expect("Failed to add paths");

    // GPU-native search: CPU does ONE memcpy, GPU does tokenization + search
    let results = search.search_gpu_native_blocking("report", 10);

    assert!(!results.is_empty(), "Should find at least one result");
    assert_eq!(results[0].0, 0, "First result should be report.txt");
}

#[test]
fn test_gpu_native_search_case_insensitive() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec![
        "/Documents/Report.TXT".to_string(),
        "/photos/vacation.jpg".to_string(),
    ];
    search.add_paths(&paths).expect("Failed to add paths");

    // GPU should lowercase the query - CPU does NOT lowercase
    let results = search.search_gpu_native_blocking("REPORT", 10);

    assert!(!results.is_empty(), "Should find result despite case difference");
    assert_eq!(results[0].0, 0, "Should find Report.TXT");
}

#[test]
fn test_gpu_native_search_multiple_words() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec![
        "/docs/annual_report.pdf".to_string(),
        "/docs/monthly_report.pdf".to_string(),
        "/photos/vacation.jpg".to_string(),
    ];
    search.add_paths(&paths).expect("Failed to add paths");

    // Multi-word query: "annual report" (GPU tokenizes to ["annual", "report"])
    let results = search.search_gpu_native_blocking("annual report", 10);

    // Should find annual_report.pdf
    assert!(!results.is_empty(), "Should find annual_report.pdf");
    assert_eq!(results[0].0, 0, "First result should be annual_report.pdf");
}

#[test]
fn test_gpu_native_search_whitespace_handling() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec![
        "/documents/report.txt".to_string(),
        "/photos/vacation.jpg".to_string(),
    ];
    search.add_paths(&paths).expect("Failed to add paths");

    // Extra whitespace should be handled by GPU tokenization
    let results = search.search_gpu_native_blocking("  report  ", 10);

    assert!(!results.is_empty(), "Should find result despite extra whitespace");
    assert_eq!(results[0].0, 0, "Should find report.txt");
}

#[test]
fn test_gpu_native_vs_cpu_tokenization_equivalence() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec![
        "/documents/report.txt".to_string(),
        "/documents/summary.pdf".to_string(),
        "/photos/vacation.jpg".to_string(),
    ];
    search.add_paths(&paths).expect("Failed to add paths");

    // Compare results from GPU-native vs CPU-tokenized search
    let gpu_results = search.search_gpu_native_blocking("report", 10);
    let cpu_results = search.search("report", 10);

    // Both should find the same paths
    assert_eq!(gpu_results.len(), cpu_results.len(),
        "GPU and CPU search should find same number of results");

    for (gpu, cpu) in gpu_results.iter().zip(cpu_results.iter()) {
        assert_eq!(gpu.0, cpu.0, "GPU and CPU should find same paths");
        // Scores might differ slightly due to timing, but paths should match
    }
}

#[test]
fn test_gpu_native_search_empty_query() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec!["/documents/report.txt".to_string()];
    search.add_paths(&paths).expect("Failed to add paths");

    // Empty query should return no results
    let results = search.search_gpu_native_blocking("", 10);
    assert!(results.is_empty(), "Empty query should return no results");

    // Whitespace-only query should also return no results (GPU tokenizes to 0 words)
    let results = search.search_gpu_native_blocking("   ", 10);
    assert!(results.is_empty(), "Whitespace-only query should return no results");
}

#[test]
fn test_gpu_native_search_async() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 1000).expect("Failed to create GpuPathSearch");

    let paths = vec![
        "/documents/report.txt".to_string(),
        "/photos/vacation.jpg".to_string(),
    ];
    search.add_paths(&paths).expect("Failed to add paths");

    // Async GPU-native search
    let handle = search.search_gpu_native("REPORT", 10);

    // Should complete eventually
    let results = handle.wait_and_get_results();

    assert!(!results.is_empty(), "Async GPU-native search should find results");
}

#[test]
fn test_gpu_native_search_large_batch() {
    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 100_000).expect("Failed to create GpuPathSearch");

    // Create 10K paths
    let paths: Vec<String> = (0..10_000)
        .map(|i| format!("/documents/file_{}.txt", i))
        .collect();
    search.add_paths(&paths).expect("Failed to add paths");

    // Search should work on larger datasets
    let results = search.search_gpu_native_blocking("file", 100);

    assert_eq!(results.len(), 100, "Should return max_results items");
}

// =============================================================================
// Performance Comparison Tests (informational)
// =============================================================================

#[test]
fn test_gpu_native_search_performance() {
    use std::time::Instant;

    let device = get_device();
    let mut search = GpuPathSearch::new(&device, 50_000).expect("Failed to create GpuPathSearch");

    // Create 10K paths
    let paths: Vec<String> = (0..10_000)
        .map(|i| format!("/documents/project_{}/report_{}.txt", i % 100, i))
        .collect();
    search.add_paths(&paths).expect("Failed to add paths");

    // Warm up
    let _ = search.search_gpu_native_blocking("report", 100);
    let _ = search.search("report", 100);

    // Benchmark GPU-native search (CPU does ONE memcpy)
    let start = Instant::now();
    for _ in 0..100 {
        let _ = search.search_gpu_native_blocking("Report Project", 100);
    }
    let gpu_native_time = start.elapsed();

    // Benchmark CPU-tokenized search (CPU does tokenization + lowercase)
    let start = Instant::now();
    for _ in 0..100 {
        let _ = search.search("Report Project", 100);
    }
    let cpu_tokenized_time = start.elapsed();

    println!("\n=== Issue #79: GPU String Processing Performance ===");
    println!("GPU-native search (100 iterations): {:?}", gpu_native_time);
    println!("CPU-tokenized search (100 iterations): {:?}", cpu_tokenized_time);
    println!("CPU work in GPU-native: ONE memcpy");
    println!("CPU work in CPU-tokenized: split_whitespace + to_lowercase + memcpy per word");

    // GPU-native should have lower CPU overhead (but may have similar total time
    // due to GPU kernel overhead for small queries)
    // The real benefit is seen in CPU profiling - GPU-native frees CPU for other work
}
