// Benchmark: Async vs Blocking Search Performance
//
// Measures the real-world impact of Issue #76 async pipeline changes.
// Run with: cargo run --release --example benchmark_async_search

use metal::Device;
use rust_experiment::gpu_os::filesystem::GpuPathSearch;
use std::time::{Duration, Instant};

fn main() {
    println!("=== Async Search Pipeline Benchmark (Issue #76) ===\n");

    let device = Device::system_default().expect("No Metal device");
    println!("Device: {}", device.name());

    // Create search with varying sizes
    for &path_count in &[1_000, 10_000, 50_000, 100_000] {
        benchmark_size(&device, path_count);
    }

    println!("\n=== CPU Utilization Test ===\n");
    benchmark_cpu_utilization(&device);
}

fn benchmark_size(device: &Device, path_count: usize) {
    println!("--- {} paths ---", path_count);

    let mut search = GpuPathSearch::new(device, path_count + 1000)
        .expect("Failed to create search");

    // Generate test paths
    let paths: Vec<String> = (0..path_count)
        .map(|i| format!("/usr/local/lib/libsomething_{}.dylib", i))
        .collect();

    search.add_paths(&paths).expect("Failed to add paths");

    let queries = ["lib", "something", "local", "usr"];
    let iterations = 20;

    // Benchmark BLOCKING search
    let mut blocking_times = Vec::with_capacity(iterations * queries.len());
    for _ in 0..iterations {
        for query in &queries {
            let start = Instant::now();
            let _results = search.search(query, 100);
            blocking_times.push(start.elapsed());
        }
    }

    // Benchmark ASYNC search (dispatch only)
    let mut async_dispatch_times = Vec::with_capacity(iterations * queries.len());
    let mut async_total_times = Vec::with_capacity(iterations * queries.len());
    for _ in 0..iterations {
        for query in &queries {
            let start = Instant::now();
            let handle = search.search_async(query, 100);
            let dispatch_time = start.elapsed();
            async_dispatch_times.push(dispatch_time);

            // Now wait for results
            let _results = handle.wait_and_get_results();
            async_total_times.push(start.elapsed());
        }
    }

    // Calculate stats
    let blocking_avg = average(&blocking_times);
    let async_dispatch_avg = average(&async_dispatch_times);
    let async_total_avg = average(&async_total_times);

    println!("  Blocking search:     {:>8.2}µs avg", blocking_avg.as_micros());
    println!("  Async dispatch:      {:>8.2}µs avg  ({:.1}x faster)",
        async_dispatch_avg.as_micros(),
        blocking_avg.as_micros() as f64 / async_dispatch_avg.as_micros().max(1) as f64
    );
    println!("  Async total:         {:>8.2}µs avg", async_total_avg.as_micros());
    println!("  CPU freed:           {:>8.2}µs per search\n",
        (blocking_avg - async_dispatch_avg).as_micros()
    );
}

fn benchmark_cpu_utilization(device: &Device) {
    let mut search = GpuPathSearch::new(device, 60_000)
        .expect("Failed to create search");

    let paths: Vec<String> = (0..50_000)
        .map(|i| format!("/home/user/documents/project/src/module_{}/file_{}.rs", i % 100, i))
        .collect();

    search.add_paths(&paths).expect("Failed to add paths");

    println!("Simulating real-world usage: search while doing other work\n");

    // BLOCKING: Can't do other work
    let start = Instant::now();
    let mut blocking_work_done = 0u64;
    for _ in 0..10 {
        let _results = search.search("module", 50);
        // Try to do work - but we were blocked during search
    }
    let blocking_elapsed = start.elapsed();
    println!("Blocking mode:");
    println!("  10 searches took:    {:>6.2}ms", blocking_elapsed.as_secs_f64() * 1000.0);
    println!("  Work done during:    {} units", blocking_work_done);

    // ASYNC: Can do work while GPU processes
    let start = Instant::now();
    let mut async_work_done = 0u64;
    for _ in 0..10 {
        let handle = search.search_async("module", 50);

        // Do work while GPU processes
        while !handle.is_complete() {
            // Simulate CPU work (e.g., processing input, updating UI state)
            for _ in 0..100 {
                async_work_done += 1;
            }
            std::hint::spin_loop();
        }
        let _results = handle.wait_and_get_results();
    }
    let async_elapsed = start.elapsed();
    println!("\nAsync mode:");
    println!("  10 searches took:    {:>6.2}ms", async_elapsed.as_secs_f64() * 1000.0);
    println!("  Work done during:    {} units", async_work_done);

    println!("\nConclusion:");
    if async_work_done > 0 {
        println!("  ✓ Async mode allows CPU to do {} units of work", async_work_done);
        println!("    while GPU processes searches (vs 0 in blocking mode)");
    }

    // Measure dispatch latency specifically
    println!("\n--- Dispatch Latency Histogram ---\n");
    let mut latencies: Vec<Duration> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let start = Instant::now();
        let handle = search.search_async("file", 20);
        latencies.push(start.elapsed());
        let _ = handle.wait_and_get_results(); // Clean up
    }

    latencies.sort();
    let p50 = latencies[500];
    let p95 = latencies[950];
    let p99 = latencies[990];
    let min = latencies[0];
    let max = latencies[999];

    println!("  Min:     {:>6}µs", min.as_micros());
    println!("  P50:     {:>6}µs", p50.as_micros());
    println!("  P95:     {:>6}µs", p95.as_micros());
    println!("  P99:     {:>6}µs", p99.as_micros());
    println!("  Max:     {:>6}µs", max.as_micros());

    if p99.as_micros() < 1000 {
        println!("\n  ✓ 99% of dispatches complete in <1ms (target met)");
    } else {
        println!("\n  ✗ P99 dispatch latency exceeds 1ms target");
    }
}

fn average(durations: &[Duration]) -> Duration {
    let total: Duration = durations.iter().sum();
    total / durations.len() as u32
}
