//! Issue #133: Persistent Search Kernel - Eliminate dispatch overhead
//!
//! Tests for persistent GPU kernel with work queue.

use metal::*;
use rust_experiment::gpu_os::persistent_search::PersistentSearchQueue;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Work item status constants
const STATUS_EMPTY: u32 = 0;
const STATUS_READY: u32 = 1;
const STATUS_PROCESSING: u32 = 2;
const STATUS_DONE: u32 = 3;

/// Simulated work item for testing queue logic
#[repr(C)]
struct TestWorkItem {
    pattern: [u8; 64],
    pattern_len: u32,
    status: AtomicU32,
    result_count: AtomicU32,
    _padding: [u32; 2],
}

impl Default for TestWorkItem {
    fn default() -> Self {
        Self {
            pattern: [0u8; 64],
            pattern_len: 0,
            status: AtomicU32::new(STATUS_EMPTY),
            result_count: AtomicU32::new(0),
            _padding: [0; 2],
        }
    }
}

/// Simulated control block for testing
#[repr(C)]
struct TestControlBlock {
    head: AtomicU32,
    tail: AtomicU32,
    shutdown: AtomicU32,
    heartbeat: AtomicU32,
}

impl Default for TestControlBlock {
    fn default() -> Self {
        Self {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            shutdown: AtomicU32::new(0),
            heartbeat: AtomicU32::new(0),
        }
    }
}

#[test]
fn test_work_queue_logic() {
    // Test the queue logic without GPU
    const QUEUE_SIZE: usize = 4;
    let mut work_items: Vec<TestWorkItem> = (0..QUEUE_SIZE).map(|_| TestWorkItem::default()).collect();
    let control = TestControlBlock::default();

    println!("Work queue logic test:");

    // Producer: submit 3 work items
    for i in 0..3 {
        let tail = control.tail.fetch_add(1, Ordering::AcqRel) as usize;
        let idx = tail % QUEUE_SIZE;

        let pattern = format!("pattern_{}", i);
        work_items[idx].pattern[..pattern.len()].copy_from_slice(pattern.as_bytes());
        work_items[idx].pattern_len = pattern.len() as u32;
        work_items[idx].status.store(STATUS_READY, Ordering::Release);

        println!("  Submitted work item {} at index {}", i, idx);
    }

    // Consumer: process work items
    for _ in 0..3 {
        let head = control.head.load(Ordering::Acquire) as usize;
        let tail = control.tail.load(Ordering::Acquire) as usize;

        assert!(head < tail, "Queue should have work");

        let idx = head % QUEUE_SIZE;
        let status = work_items[idx].status.load(Ordering::Acquire);
        assert_eq!(status, STATUS_READY, "Work item should be ready");

        // Process
        work_items[idx].status.store(STATUS_PROCESSING, Ordering::Release);
        work_items[idx].result_count.store(42, Ordering::Release);
        work_items[idx].status.store(STATUS_DONE, Ordering::Release);

        control.head.fetch_add(1, Ordering::Release);
        println!("  Processed work item at index {}", idx);
    }

    // Verify all items processed
    assert_eq!(control.head.load(Ordering::Relaxed), 3);
    assert_eq!(control.tail.load(Ordering::Relaxed), 3);
    println!("  All work items processed successfully");
}

#[test]
fn test_queue_wraparound() {
    // Test queue wraparound behavior
    const QUEUE_SIZE: usize = 4;
    let control = TestControlBlock::default();

    println!("Queue wraparound test:");

    // Submit and process 10 items (wraps around twice)
    for i in 0..10 {
        let tail = control.tail.fetch_add(1, Ordering::AcqRel);
        let idx = tail as usize % QUEUE_SIZE;
        println!("  Item {}: tail={}, idx={}", i, tail, idx);

        // Simulate processing
        control.head.fetch_add(1, Ordering::Release);
    }

    assert_eq!(control.head.load(Ordering::Relaxed), 10);
    assert_eq!(control.tail.load(Ordering::Relaxed), 10);
    println!("  Wraparound handled correctly");
}

#[test]
fn test_shutdown_signal() {
    let control = TestControlBlock::default();

    println!("Shutdown signal test:");

    // Initially not shutdown
    assert_eq!(control.shutdown.load(Ordering::Relaxed), 0);
    println!("  Initial state: running");

    // Signal shutdown
    control.shutdown.store(1, Ordering::Release);
    assert_eq!(control.shutdown.load(Ordering::Acquire), 1);
    println!("  After signal: shutdown");

    // Kernel would check this flag and exit loop
}

#[test]
fn test_heartbeat_monitoring() {
    let control = TestControlBlock::default();

    println!("Heartbeat monitoring test:");

    // Simulate GPU incrementing heartbeat
    for i in 0..5 {
        let prev = control.heartbeat.fetch_add(1, Ordering::Relaxed);
        println!("  Heartbeat {}: {} -> {}", i, prev, prev + 1);
    }

    assert_eq!(control.heartbeat.load(Ordering::Relaxed), 5);

    // In real implementation:
    // - CPU checks heartbeat periodically
    // - If heartbeat doesn't change for N seconds, kernel is stuck
    // - Can trigger recovery or error
}

#[test]
fn benchmark_dispatch_overhead() {
    // Measure baseline dispatch overhead to quantify potential savings
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n=== Dispatch Overhead Benchmark ===\n");

    // Measure empty command buffer overhead
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let cmd_buffer = queue.new_command_buffer();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    let elapsed = start.elapsed();
    let per_dispatch_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("Empty dispatch overhead:");
    println!("  {} iterations in {:.1}ms", iterations, elapsed.as_secs_f64() * 1000.0);
    println!("  {:.1}µs per dispatch", per_dispatch_us);

    // Typical overhead is 100-500µs per dispatch
    // Persistent kernel eliminates this for repeated searches

    let estimated_savings_100_searches = per_dispatch_us * 100.0 / 1000.0;
    println!("\nEstimated savings for 100 searches: {:.1}ms", estimated_savings_100_searches);
}

#[test]
fn test_atomic_operations_performance() {
    // Test atomic operation performance for queue management
    let counter = AtomicU32::new(0);

    let iterations = 1_000_000;
    let start = Instant::now();

    for _ in 0..iterations {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    println!("\nAtomic operations performance:");
    println!("  {} ops in {:.1}ms", iterations, elapsed.as_secs_f64() * 1000.0);
    println!("  {:.1}M ops/sec", ops_per_sec / 1_000_000.0);

    // GPU atomics are typically 10-100x slower than CPU
    // But still fast enough for queue management (one atomic per work item)
}

#[test]
fn test_spin_wait_vs_yield() {
    // Compare spin-wait strategies
    let flag = AtomicU32::new(0);

    println!("\nSpin-wait strategy comparison:");

    // Strategy 1: Tight spin
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = flag.load(Ordering::Relaxed);
    }
    let tight_time = start.elapsed();

    // Strategy 2: Spin with hint
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = flag.load(Ordering::Relaxed);
        std::hint::spin_loop();
    }
    let hint_time = start.elapsed();

    println!("  Tight spin:      {:.1}µs for 10K iterations", tight_time.as_micros());
    println!("  Spin with hint:  {:.1}µs for 10K iterations", hint_time.as_micros());

    // spin_loop() hint tells CPU to reduce power while waiting
    // Important for persistent kernels that may spin frequently
}

// ============================================================================
// Full Implementation Tests using PersistentSearchQueue
// ============================================================================

#[test]
fn test_persistent_kernel_correctness() {
    // Verify persistent kernel produces correct results
    let device = Device::system_default().expect("No Metal device");
    let mut queue = PersistentSearchQueue::new(&device, 1024 * 1024)
        .expect("Failed to create queue");

    println!("\n=== Persistent Kernel Correctness Test ===\n");

    // Load test data
    let data = b"Hello World Hello World Hello test pattern test pattern test";
    queue.load_data(0, data).expect("Failed to load data");

    // Start kernel with limited iterations
    queue.start_kernel(10000);

    // Test 1: Basic search
    let handle = queue.submit_search("Hello", false, 0).expect("Queue full");
    let result = queue.wait_result_timeout(handle, Duration::from_secs(5))
        .expect("Search timed out");
    println!("  'Hello': {} matches (expected 3)", result.match_count);
    assert_eq!(result.match_count, 3, "Expected 3 matches for 'Hello'");

    // Test 2: Another pattern
    // Data: "...test pattern test pattern test" = 3 instances of "test"
    let handle = queue.submit_search("test", false, 0).expect("Queue full");
    let result = queue.wait_result_timeout(handle, Duration::from_secs(5))
        .expect("Search timed out");
    println!("  'test': {} matches (expected 3)", result.match_count);
    assert_eq!(result.match_count, 3, "Expected 3 matches for 'test'");

    // Test 3: Not found
    let handle = queue.submit_search("NotFound", false, 0).expect("Queue full");
    let result = queue.wait_result_timeout(handle, Duration::from_secs(5))
        .expect("Search timed out");
    println!("  'NotFound': {} matches (expected 0)", result.match_count);
    assert_eq!(result.match_count, 0, "Expected 0 matches for 'NotFound'");

    // Shutdown
    queue.shutdown();
    println!("\n  All correctness tests passed!");
}

#[test]
fn benchmark_persistent_vs_traditional() {
    // Full benchmark comparing dispatch approaches
    let device = Device::system_default().expect("No Metal device");
    let mut queue = PersistentSearchQueue::new(&device, 1024 * 1024)
        .expect("Failed to create queue");

    println!("\n=== Persistent vs Traditional Benchmark ===\n");

    // Load test data - 100KB of text
    let mut data = Vec::new();
    for _ in 0..1000 {
        data.extend_from_slice(b"Hello World, this is a test string for searching patterns. TODO find me. ");
        data.extend_from_slice(b"More text here with various patterns to search for in the benchmark test. ");
    }
    queue.load_data(0, &data).expect("Failed to load data");

    let iterations = 50;

    // Traditional: new dispatch per search
    let trad_start = Instant::now();
    for _ in 0..iterations {
        let _count = queue.oneshot_search("TODO", false).expect("Search failed");
    }
    let trad_time = trad_start.elapsed();

    // Start persistent kernel
    queue.start_kernel(100000);

    // Warmup
    for _ in 0..5 {
        let handle = queue.submit_search("TODO", false, 0).expect("Queue full");
        queue.wait_result_timeout(handle, Duration::from_secs(5)).expect("Timeout");
    }

    // Persistent: reuse running kernel
    let pers_start = Instant::now();
    for _ in 0..iterations {
        let handle = queue.submit_search("TODO", false, 0).expect("Queue full");
        queue.wait_result_timeout(handle, Duration::from_secs(5)).expect("Search timed out");
    }
    let pers_time = pers_start.elapsed();

    queue.shutdown();

    let trad_per_search = trad_time.as_micros() as f64 / iterations as f64;
    let pers_per_search = pers_time.as_micros() as f64 / iterations as f64;
    let speedup = trad_time.as_secs_f64() / pers_time.as_secs_f64();

    println!("  Data size: {} bytes", data.len());
    println!("  Iterations: {}", iterations);
    println!();
    println!("  Traditional (new dispatch per search):");
    println!("    Total time: {:.1}ms", trad_time.as_secs_f64() * 1000.0);
    println!("    Per search: {:.1}us", trad_per_search);
    println!();
    println!("  Persistent (reuse running kernel):");
    println!("    Total time: {:.1}ms", pers_time.as_secs_f64() * 1000.0);
    println!("    Per search: {:.1}us", pers_per_search);
    println!();
    println!("  Speedup: {:.2}x", speedup);
    println!();

    // Note: speedup may be < 1x for small data because dispatch overhead is only ~19us
    // The benefit is more significant for rapid repeated searches (search-as-you-type)
    println!("  Note: Dispatch overhead is ~19us, so benefit is workload-dependent.");
}

#[test]
fn test_persistent_kernel_stability() {
    // Run persistent kernel for extended period
    // Verify no memory leaks, hangs, or crashes
    let device = Device::system_default().expect("No Metal device");
    let mut queue = PersistentSearchQueue::new(&device, 1024 * 1024)
        .expect("Failed to create queue");

    println!("\n=== Persistent Kernel Stability Test ===\n");

    // Load test data
    let data = b"Stability test data with patterns to search repeatedly over time.";
    queue.load_data(0, data).expect("Failed to load data");

    // Start kernel
    queue.start_kernel(1000000);  // High iteration limit

    let initial_heartbeat = queue.heartbeat();
    println!("  Initial heartbeat: {}", initial_heartbeat);

    // Run many searches
    let num_searches = 100;
    let start = Instant::now();

    for i in 0..num_searches {
        let handle = queue.submit_search("test", false, 0).expect("Queue full");
        let result = queue.wait_result_timeout(handle, Duration::from_secs(1));

        if result.is_none() {
            panic!("Search {} timed out - kernel may be stuck", i);
        }
    }

    let elapsed = start.elapsed();
    let final_heartbeat = queue.heartbeat();

    println!("  Completed {} searches in {:.1}ms", num_searches, elapsed.as_secs_f64() * 1000.0);
    println!("  Final heartbeat: {} (delta: {})", final_heartbeat, final_heartbeat - initial_heartbeat);

    // Check stats
    let stats = queue.stats();
    println!("  Queue stats: head={}, tail={}", stats.head, stats.tail);

    // Verify heartbeat is advancing (kernel is alive)
    assert!(final_heartbeat > initial_heartbeat, "Heartbeat should advance");

    // Clean shutdown
    queue.shutdown();
    assert!(!queue.is_running(), "Queue should be stopped after shutdown");

    println!("\n  Stability test passed!");
}

#[test]
fn test_persistent_queue_operations() {
    // Test queue operations: submit, is_complete, stats
    let device = Device::system_default().expect("No Metal device");
    let mut queue = PersistentSearchQueue::new(&device, 1024 * 1024)
        .expect("Failed to create queue");

    println!("\n=== Queue Operations Test ===\n");

    let data = b"Test data for queue operations testing.";
    queue.load_data(0, data).expect("Failed to load data");

    // Check initial stats
    let stats = queue.stats();
    assert_eq!(stats.head, 0);
    assert_eq!(stats.tail, 0);
    assert!(!stats.shutdown);
    println!("  Initial stats: head={}, tail={}", stats.head, stats.tail);

    // Start kernel
    queue.start_kernel(10000);
    assert!(queue.is_running());

    // Submit search
    let handle = queue.submit_search("Test", false, 0).expect("Queue full");
    println!("  Submitted search, handle idx={}", handle.idx);

    // Wait for completion
    let result = queue.wait_result(handle).expect("Search failed");
    println!("  Result: {} matches", result.match_count);

    // Check it's complete
    assert!(queue.is_complete(handle));

    // Shutdown
    queue.shutdown();
    assert!(!queue.is_running());

    println!("\n  Queue operations test passed!");
}
