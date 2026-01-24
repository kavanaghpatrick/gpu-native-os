// Issue #91: GPU-Native Parallel Prefix Allocator
// Integration tests and benchmarks

use rust_experiment::gpu_os::parallel_alloc::{
    GpuParallelAllocator, AllocationRequest, AllocatorStats, MAX_BATCH_SIZE,
};
use metal::Device;
use std::time::Instant;

fn create_allocator(pool_size: usize) -> GpuParallelAllocator {
    let device = Device::system_default().expect("No Metal device found");
    GpuParallelAllocator::new(&device, pool_size).expect("Failed to create allocator")
}

// ============================================================================
// Correctness Tests
// ============================================================================

#[test]
fn test_single_allocation_correctness() {
    let allocator = create_allocator(1024 * 1024);

    let requests = vec![AllocationRequest { size: 256, alignment: 16 }];
    let results = allocator.alloc_batch(&requests);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].valid, 1, "Allocation should be valid");
    assert_eq!(results[0].size, 256);
    assert_eq!(results[0].offset % 16, 0, "Offset should be 16-byte aligned");
}

#[test]
fn test_batch_no_overlap() {
    let allocator = create_allocator(1024 * 1024);

    // Allocate 100 chunks of 1KB each
    let requests: Vec<AllocationRequest> = (0..100)
        .map(|_| AllocationRequest { size: 1024, alignment: 16 })
        .collect();

    let results = allocator.alloc_batch(&requests);

    // Verify no overlaps
    for i in 0..results.len() {
        if results[i].valid != 1 {
            continue;
        }

        let start_i = results[i].offset;
        let end_i = start_i + results[i].size;

        for j in (i + 1)..results.len() {
            if results[j].valid != 1 {
                continue;
            }

            let start_j = results[j].offset;
            let end_j = start_j + results[j].size;

            // Check for overlap
            let overlaps = start_i < end_j && start_j < end_i;
            assert!(
                !overlaps,
                "Allocation {} [{}, {}) overlaps with {} [{}, {})",
                i, start_i, end_i, j, start_j, end_j
            );
        }
    }
}

#[test]
fn test_alignment_guarantees() {
    let allocator = create_allocator(1024 * 1024);

    // Test various alignment requirements
    let alignments = [4, 8, 16, 32, 64, 128, 256];

    for alignment in alignments {
        let requests: Vec<AllocationRequest> = (0..10)
            .map(|_| AllocationRequest { size: 100, alignment })
            .collect();

        let results = allocator.alloc_batch(&requests);
        allocator.reset();

        for (i, result) in results.iter().enumerate() {
            if result.valid == 1 {
                assert_eq!(
                    result.offset % alignment, 0,
                    "Allocation {} with alignment {} has unaligned offset {}",
                    i, alignment, result.offset
                );
            }
        }
    }
}

#[test]
fn test_mixed_sizes() {
    let allocator = create_allocator(1024 * 1024);

    // Mixed sizes like a real allocator would see
    let requests = vec![
        AllocationRequest { size: 16, alignment: 8 },      // Tiny
        AllocationRequest { size: 1024, alignment: 16 },   // Small
        AllocationRequest { size: 4096, alignment: 64 },   // Medium
        AllocationRequest { size: 64, alignment: 32 },     // Small with big alignment
        AllocationRequest { size: 256, alignment: 256 },   // Alignment == size
    ];

    let results = allocator.alloc_batch(&requests);

    assert_eq!(results.len(), 5);
    for result in &results {
        assert_eq!(result.valid, 1, "All allocations should succeed");
    }
}

#[test]
fn test_pool_exhaustion() {
    let allocator = create_allocator(1024); // Small pool

    // Try to allocate more than available
    let requests: Vec<AllocationRequest> = (0..10)
        .map(|_| AllocationRequest { size: 200, alignment: 16 })
        .collect();

    let results = allocator.alloc_batch(&requests);

    let valid_count = results.iter().filter(|r| r.valid == 1).count();
    let invalid_count = results.iter().filter(|r| r.valid == 0).count();

    // Some should succeed, some should fail
    assert!(valid_count > 0, "Some allocations should succeed");
    assert!(invalid_count > 0, "Some allocations should fail (out of memory)");
}

#[test]
fn test_reset_clears_state() {
    let allocator = create_allocator(1024 * 1024);

    // Allocate some memory
    let requests: Vec<AllocationRequest> = (0..50)
        .map(|_| AllocationRequest { size: 1024, alignment: 16 })
        .collect();
    allocator.alloc_batch(&requests);

    // Reset
    allocator.reset();

    // Should be able to allocate from the beginning again
    let results = allocator.alloc_batch(&requests);

    // First allocation should be at offset 0 (or near it, depending on alignment)
    let min_offset = results.iter().filter(|r| r.valid == 1).map(|r| r.offset).min().unwrap();
    assert!(min_offset < 256, "After reset, allocations should start near beginning");
}

#[test]
fn test_max_batch_size() {
    let allocator = create_allocator(16 * 1024 * 1024); // 16MB pool

    // Allocate exactly MAX_BATCH_SIZE allocations
    let requests: Vec<AllocationRequest> = (0..MAX_BATCH_SIZE)
        .map(|_| AllocationRequest { size: 64, alignment: 16 })
        .collect();

    let results = allocator.alloc_batch(&requests);

    assert_eq!(results.len(), MAX_BATCH_SIZE);

    let valid_count = results.iter().filter(|r| r.valid == 1).count();
    assert_eq!(valid_count, MAX_BATCH_SIZE, "All allocations should succeed with large pool");
}

// ============================================================================
// Benchmarks
// ============================================================================

fn benchmark_gpu_allocation(batch_size: usize, iterations: usize) -> std::time::Duration {
    let allocator = create_allocator(1024 * 1024 * 64); // 64MB pool

    let requests: Vec<AllocationRequest> = (0..batch_size)
        .map(|i| AllocationRequest {
            size: 64 + ((i as u32) % 192), // Varying sizes 64-255
            alignment: 16
        })
        .collect();

    let start = Instant::now();

    for _ in 0..iterations {
        let _ = allocator.alloc_batch(&requests);
        allocator.reset();
    }

    start.elapsed()
}

fn benchmark_cpu_allocation(batch_size: usize, iterations: usize) -> std::time::Duration {
    use std::sync::atomic::{AtomicU32, Ordering};

    let bump = AtomicU32::new(0);
    let pool_size = 64 * 1024 * 1024u32;

    let start = Instant::now();

    for _ in 0..iterations {
        bump.store(0, Ordering::SeqCst);

        for i in 0..batch_size {
            let size = 64 + ((i as u32) % 192);
            let alignment = 16u32;

            // CPU atomic bump allocation with alignment
            loop {
                let current = bump.load(Ordering::SeqCst);
                let aligned = (current + alignment - 1) & !(alignment - 1);
                let next = aligned + size;

                if next > pool_size {
                    break; // Out of memory
                }

                if bump.compare_exchange(current, next, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    break; // Allocation succeeded
                }
                // CAS failed, retry
            }
        }
    }

    start.elapsed()
}

#[test]
fn benchmark_allocation_throughput() {
    println!("\n=== GPU Parallel Prefix Allocator Benchmarks ===\n");

    let batch_sizes = [32, 64, 128, 256, 512, 1024];
    let iterations = 100;

    println!("Batch Size | GPU Time (ms) | CPU Time (ms) | Speedup");
    println!("-----------|---------------|---------------|--------");

    for &batch_size in &batch_sizes {
        let gpu_time = benchmark_gpu_allocation(batch_size, iterations);
        let cpu_time = benchmark_cpu_allocation(batch_size, iterations);

        let gpu_ms = gpu_time.as_secs_f64() * 1000.0;
        let cpu_ms = cpu_time.as_secs_f64() * 1000.0;
        let speedup = cpu_ms / gpu_ms;

        println!(
            "{:>10} | {:>13.3} | {:>13.3} | {:>6.1}x",
            batch_size, gpu_ms, cpu_ms, speedup
        );
    }

    println!("\n(Lower is better. Speedup > 1 means GPU is faster)\n");
}

#[test]
fn benchmark_warp_allocation() {
    println!("\n=== Warp-Level (32 threads) Allocation Benchmark ===\n");

    let allocator = create_allocator(64 * 1024 * 1024);
    let iterations = 1000;

    // 32-thread warp allocation
    let requests: Vec<AllocationRequest> = (0..32)
        .map(|i| AllocationRequest {
            size: 64 + (i as u32 * 8),
            alignment: 16
        })
        .collect();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = allocator.alloc_batch_warp(&requests);
        allocator.reset();
    }
    let warp_time = start.elapsed();

    // Compare with batch allocation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = allocator.alloc_batch(&requests);
        allocator.reset();
    }
    let batch_time = start.elapsed();

    println!("Warp allocation (SIMD shuffle):  {:.3} ms for {} iterations",
             warp_time.as_secs_f64() * 1000.0, iterations);
    println!("Batch allocation (prefix sum):   {:.3} ms for {} iterations",
             batch_time.as_secs_f64() * 1000.0, iterations);
    println!("Warp allocation speedup:         {:.1}x faster\n",
             batch_time.as_secs_f64() / warp_time.as_secs_f64());
}

#[test]
fn benchmark_latency() {
    println!("\n=== Allocation Latency (Single Batch) ===\n");

    let allocator = create_allocator(64 * 1024 * 1024);

    let batch_sizes = [1, 8, 32, 128, 512, 1024];

    for &batch_size in &batch_sizes {
        let requests: Vec<AllocationRequest> = (0..batch_size)
            .map(|_| AllocationRequest { size: 128, alignment: 16 })
            .collect();

        // Warmup
        for _ in 0..10 {
            let _ = allocator.alloc_batch(&requests);
            allocator.reset();
        }

        // Measure
        let mut times = Vec::with_capacity(100);
        for _ in 0..100 {
            let start = Instant::now();
            let _ = allocator.alloc_batch(&requests);
            times.push(start.elapsed());
            allocator.reset();
        }

        times.sort();
        let median = times[50];
        let p99 = times[99];

        println!(
            "Batch {:>4}: median {:>6.1}us, p99 {:>6.1}us",
            batch_size,
            median.as_secs_f64() * 1_000_000.0,
            p99.as_secs_f64() * 1_000_000.0
        );
    }
    println!();
}

#[test]
fn test_stats_collection() {
    let allocator = create_allocator(1024 * 1024);

    // Allocate some memory
    let requests: Vec<AllocationRequest> = (0..100)
        .map(|_| AllocationRequest { size: 512, alignment: 16 })
        .collect();

    allocator.alloc_batch(&requests);

    let stats = allocator.stats();

    println!("\nAllocator Stats:");
    println!("  Used: {} bytes", stats.used);
    println!("  Total: {} bytes", stats.total);
    println!("  Allocation count: {}", stats.allocation_count);
    println!("  Peak usage: {} bytes", stats.peak_usage);
    println!("  Usage: {:.1}%", stats.usage_percent());

    assert!(stats.used > 0, "Used should be > 0");
    assert!(stats.allocation_count > 0, "Should have some allocations");
}
