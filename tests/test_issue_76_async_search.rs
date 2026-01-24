// Test: Issue #76 - Async GPU Pipeline
//
// Validates the non-blocking search API and SharedEvent synchronization.

use metal::Device;
use rust_experiment::gpu_os::filesystem::GpuPathSearch;
use std::time::{Duration, Instant};

/// Test that search_async returns immediately without blocking
#[test]
fn test_search_async_returns_immediately() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 10000).expect("Failed to create search");

    // Add some test paths
    let paths: Vec<String> = (0..1000)
        .map(|i| format!("/usr/local/bin/test_program_{}", i))
        .collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    // Time how long search_async takes to return
    let start = Instant::now();
    let _handle = search.search_async("test", 100);
    let dispatch_time = start.elapsed();

    // Should return in < 5ms (not waiting for GPU)
    assert!(
        dispatch_time < Duration::from_millis(5),
        "search_async should return immediately, took {:?}",
        dispatch_time
    );
}

/// Test that SearchHandle correctly reports completion
#[test]
fn test_search_handle_completion() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 10000).expect("Failed to create search");

    let paths: Vec<String> = (0..500)
        .map(|i| format!("/home/user/documents/file_{}.txt", i))
        .collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    let handle = search.search_async("file", 50);

    // Eventually should complete
    let results = handle.wait_and_get_results();

    // Should find matches (all paths contain "file")
    assert!(!results.is_empty(), "Should find matches for 'file'");
    assert!(results.len() <= 50, "Should respect max_results");
}

/// Test non-blocking try_get_results
#[test]
fn test_search_handle_try_get_results() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 10000).expect("Failed to create search");

    let paths: Vec<String> = (0..100)
        .map(|i| format!("/tmp/test_{}", i))
        .collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    let handle = search.search_async("test", 20);

    // Poll until complete (with timeout)
    let start = Instant::now();
    let timeout = Duration::from_secs(5);
    let mut result = None;

    while start.elapsed() < timeout {
        if let Some(r) = handle.try_get_results() {
            result = Some(r);
            break;
        }
        std::thread::yield_now();
    }

    assert!(result.is_some(), "Search should complete within timeout");
    let results = result.unwrap();
    assert!(!results.is_empty(), "Should find matches");
}

/// Test empty query handling
#[test]
fn test_search_async_empty_query() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 1000).expect("Failed to create search");

    let paths: Vec<String> = (0..10).map(|i| format!("/path/{}", i)).collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    // Empty query should return immediately with no results
    let handle = search.search_async("", 10);

    // Should be immediately complete
    assert!(handle.is_complete(), "Empty query should complete immediately");

    let results = handle.wait_and_get_results();
    assert!(results.is_empty(), "Empty query should return no results");
}

/// Test signal value increments across searches
#[test]
fn test_signal_value_increments() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 10000).expect("Failed to create search");

    // Use simple paths with clear searchable terms
    let paths: Vec<String> = (0..100)
        .map(|i| format!("/documents/report_{}.txt", i))
        .collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    // Do first search and get signal
    let handle1 = search.search_async("report", 50);
    let signal1 = handle1.signal_value();
    let results1 = handle1.wait_and_get_results();

    // Do second search - signal should increment
    let handle2 = search.search_async("documents", 50);
    let signal2 = handle2.signal_value();
    let _results2 = handle2.wait_and_get_results();

    // Second signal should be higher (this is the main test)
    assert!(signal2 > signal1, "Signal values should increment: {} vs {}", signal1, signal2);

    // First search should find results (all paths contain "report")
    assert!(!results1.is_empty(), "Should find report matches, got {} results", results1.len());
}

/// Test that search results are correctly scored
#[test]
fn test_search_async_scoring() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 10000).expect("Failed to create search");

    let paths = vec![
        "/exact_match".to_string(),
        "/no_match_here".to_string(),
        "/another_exact_match".to_string(),
        "/something_else".to_string(),
    ];

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    let handle = search.search_async("exact", 10);
    let results = handle.wait_and_get_results();

    // Should find the paths with "exact" in them
    assert!(!results.is_empty(), "Should find matches for 'exact'");

    // All results should have positive scores
    for (idx, score) in &results {
        assert!(*score > 0, "Match at index {} should have positive score", idx);
    }
}

/// Test signal value tracking
#[test]
fn test_signal_value_tracking() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 1000).expect("Failed to create search");

    let paths: Vec<String> = (0..10).map(|i| format!("/test/{}", i)).collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    // Get initial signal value
    let initial = search.current_signal_value();

    // Do a search
    let handle = search.search_async("test", 5);
    let search_signal = handle.signal_value();

    // Signal value should have been allocated
    assert!(
        search_signal > initial,
        "Search should allocate new signal value"
    );

    // Wait for completion
    let _ = handle.wait_and_get_results();

    // Current signal should now be >= search signal
    assert!(
        search.current_signal_value() >= search_signal,
        "Signal should be updated after completion"
    );
}

/// Benchmark: Compare blocking vs async search dispatch time
#[test]
fn benchmark_async_vs_blocking_dispatch() {
    let device = Device::system_default().expect("No Metal device available");
    let search = GpuPathSearch::new(&device, 100000).expect("Failed to create search");

    // Create larger dataset
    let paths: Vec<String> = (0..50000)
        .map(|i| format!("/usr/local/lib/libsomething_{}.dylib", i))
        .collect();

    let mut search = search;
    search.add_paths(&paths).expect("Failed to add paths");

    // Time async dispatch (should be fast)
    let async_start = Instant::now();
    let handle = search.search_async("lib", 100);
    let async_dispatch = async_start.elapsed();

    // Time blocking search (will be slower because it waits)
    let blocking_start = Instant::now();
    let _ = search.search("lib", 100);
    let blocking_time = blocking_start.elapsed();

    println!("Async dispatch time: {:?}", async_dispatch);
    println!("Blocking search time: {:?}", blocking_time);

    // Async dispatch should be significantly faster than blocking
    // (blocking includes GPU execution time)
    assert!(
        async_dispatch < blocking_time,
        "Async dispatch ({:?}) should be faster than blocking ({:?})",
        async_dispatch,
        blocking_time
    );

    // Clean up - wait for async to complete
    let _ = handle.wait_and_get_results();
}
