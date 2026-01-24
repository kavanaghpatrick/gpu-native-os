// Issue #79: GPU String Processing Benchmark
//
// Compares CPU tokenization vs GPU tokenization for filesystem search.
//
// Goal: CPU does ONE memcpy, GPU does EVERYTHING ELSE.
//
// "THE GPU IS THE COMPUTER" - CPU should be idle during search.

use rust_experiment::gpu_os::filesystem::GpuPathSearch;
use std::time::Instant;

fn main() {
    let device = metal::Device::system_default().expect("No Metal device found");
    println!("Device: {}", device.name());
    println!();

    // Create search with 100K path capacity
    let mut search = GpuPathSearch::new(&device, 100_000).expect("Failed to create GpuPathSearch");

    // Generate test paths
    println!("Generating 50,000 test paths...");
    let paths: Vec<String> = (0..50_000)
        .map(|i| format!("/documents/project_{}/report_{}.txt", i % 100, i))
        .collect();
    search.add_paths(&paths).expect("Failed to add paths");
    println!("Paths indexed on GPU.");
    println!();

    // Test queries with various characteristics
    let test_queries = vec![
        ("Simple", "report"),
        ("Two words", "project report"),
        ("Three words", "document project report"),
        ("Case mixed", "REPORT Project"),
        ("Extra whitespace", "  report   project  "),
    ];

    println!("=================================================================");
    println!("Issue #79: GPU String Processing Benchmark");
    println!("=================================================================");
    println!();
    println!("CPU-tokenized search:");
    println!("  - CPU: split_whitespace(), to_lowercase(), memcpy per word");
    println!("  - GPU: fuzzy search, sort");
    println!();
    println!("GPU-native search:");
    println!("  - CPU: ONE memcpy of raw query bytes");
    println!("  - GPU: tokenization, lowercase, fuzzy search, sort");
    println!();
    println!("THE GPU IS THE COMPUTER. CPU should be idle.");
    println!("=================================================================");
    println!();

    const ITERATIONS: usize = 1000;

    for (name, query) in test_queries {
        println!("Query: \"{}\" ({})", query, name);
        println!("-----------------------------------------------------------------");

        // Warm up
        for _ in 0..10 {
            let _ = search.search_gpu_native_blocking(query, 100);
            let _ = search.search(query, 100);
        }

        // Benchmark GPU-native search (CPU does ONE memcpy)
        let start = Instant::now();
        let mut gpu_native_results = 0;
        for _ in 0..ITERATIONS {
            let results = search.search_gpu_native_blocking(query, 100);
            gpu_native_results = results.len();
        }
        let gpu_native_time = start.elapsed();

        // Benchmark CPU-tokenized search
        let start = Instant::now();
        let mut cpu_tokenized_results = 0;
        for _ in 0..ITERATIONS {
            let results = search.search(query, 100);
            cpu_tokenized_results = results.len();
        }
        let cpu_tokenized_time = start.elapsed();

        // Verify results match
        assert_eq!(gpu_native_results, cpu_tokenized_results,
            "Result count mismatch: GPU-native={}, CPU-tokenized={}",
            gpu_native_results, cpu_tokenized_results);

        println!("  GPU-native ({} iterations):    {:?}  ({} results)",
            ITERATIONS, gpu_native_time, gpu_native_results);
        println!("  CPU-tokenized ({} iterations): {:?}  ({} results)",
            ITERATIONS, cpu_tokenized_time, cpu_tokenized_results);

        let gpu_per_iter = gpu_native_time.as_nanos() / ITERATIONS as u128;
        let cpu_per_iter = cpu_tokenized_time.as_nanos() / ITERATIONS as u128;
        println!("  Per iteration: GPU-native={}ns, CPU-tokenized={}ns", gpu_per_iter, cpu_per_iter);

        if gpu_native_time < cpu_tokenized_time {
            let speedup = cpu_tokenized_time.as_nanos() as f64 / gpu_native_time.as_nanos() as f64;
            println!("  GPU-native is {:.2}x faster", speedup);
        } else {
            let overhead = gpu_native_time.as_nanos() as f64 / cpu_tokenized_time.as_nanos() as f64;
            println!("  GPU-native has {:.2}x overhead (tokenize kernel dispatch cost)", overhead);
            println!("  NOTE: Real benefit is CPU freedom - use async search!");
        }
        println!();
    }

    // Test async GPU-native search to show CPU freedom
    println!("=================================================================");
    println!("Async GPU-Native Search (CPU Freedom Test)");
    println!("=================================================================");
    println!();

    let start = Instant::now();
    let mut handles = Vec::new();

    // Launch 100 async searches
    for i in 0..100 {
        let query = format!("report_{}", i % 10);
        handles.push(search.search_gpu_native(&query, 10));
    }

    let dispatch_time = start.elapsed();
    println!("Dispatched 100 async GPU-native searches in {:?}", dispatch_time);
    println!("CPU is now FREE to do other work while GPU processes...");

    // Do some "other work" (simulate)
    let other_work_start = Instant::now();
    let mut sum: u64 = 0;
    for i in 0..1_000_000 {
        sum = sum.wrapping_add(i);
    }
    let other_work_time = other_work_start.elapsed();
    println!("CPU did {} iterations of 'other work' in {:?} (sum={})",
        1_000_000, other_work_time, sum);

    // Wait for results
    let wait_start = Instant::now();
    let mut total_results = 0;
    for handle in handles {
        total_results += handle.wait_and_get_results().len();
    }
    let wait_time = wait_start.elapsed();

    println!("Collected {} total results in {:?}", total_results, wait_time);
    println!();
    println!("Total async time: {:?} (dispatch + other work + wait)",
        dispatch_time + other_work_time + wait_time);
    println!();

    println!("=================================================================");
    println!("Summary: THE GPU IS THE COMPUTER");
    println!("=================================================================");
    println!();
    println!("GPU-native search eliminates CPU string processing:");
    println!("  - No split_whitespace() on CPU");
    println!("  - No to_lowercase() on CPU");
    println!("  - No per-word memcpy on CPU");
    println!("  - Just ONE memcpy of raw query bytes");
    println!();
    println!("GPU handles everything else:");
    println!("  - Tokenization (whitespace splitting)");
    println!("  - Case conversion (lowercase)");
    println!("  - Fuzzy matching (3M parallel threads)");
    println!("  - Result sorting");
    println!();
    println!("With async dispatch, CPU is truly FREE during search.");
}
