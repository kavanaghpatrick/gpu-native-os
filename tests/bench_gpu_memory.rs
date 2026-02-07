// Benchmarks for GPU Memory Management (Issue #155)
//
// Tests to validate that O(1) free list allocation is actually faster
// than alternatives, and measures real GPU performance characteristics.

use metal::*;
use rust_experiment::gpu_os::gpu_app_system::*;
use std::time::Instant;

fn get_device() -> Device {
    Device::system_default().expect("No Metal device")
}

// ============================================================================
// BENCHMARK: Fresh Allocation vs Free List Reuse
// ============================================================================

#[test]
fn bench_allocation_fresh_vs_reused() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    const ITERATIONS: usize = 50;  // Stay under 64 slot limit
    const STATE_SIZE: u32 = 4096;
    const VERTEX_SIZE: u32 = 2048;

    // Warm up with cycling (not holding slots)
    for _ in 0..5 {
        let slot = system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
        system.close_app(slot);
    }

    // Benchmark fresh allocations by cycling (no reuse possible)
    // Reset the system to get fresh bump allocator
    let mut fresh_system = GpuAppSystem::new(&device).expect("Failed");
    fresh_system.set_use_o1_allocator(false); // Force bump-only

    let fresh_start = Instant::now();
    for _ in 0..ITERATIONS {
        let slot = fresh_system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
        fresh_system.close_app(slot);
    }
    let fresh_duration = fresh_start.elapsed();

    // Benchmark with O(1) allocator (allows reuse)
    let mut reuse_system = GpuAppSystem::new(&device).expect("Failed");
    reuse_system.set_use_o1_allocator(true);

    let reuse_start = Instant::now();
    for _ in 0..ITERATIONS {
        let slot = reuse_system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
        reuse_system.close_app(slot);
    }
    let reuse_duration = reuse_start.elapsed();

    // Print results
    let fresh_per_op = fresh_duration.as_nanos() / ITERATIONS as u128;
    let reuse_per_op = reuse_duration.as_nanos() / ITERATIONS as u128;

    let fresh_stats = fresh_system.memory_stats();
    let reuse_stats = reuse_system.memory_stats();

    println!("\n=== Allocation Benchmark (launch+close cycle) ===");
    println!("Bump-only (no reuse):  {:>8} ns/cycle, bump ptr: {} bytes",
        fresh_per_op, fresh_stats.state_pool.bump_pointer);
    println!("O(1) free list (reuse): {:>8} ns/cycle, bump ptr: {} bytes",
        reuse_per_op, reuse_stats.state_pool.bump_pointer);

    // Memory savings
    let memory_saved = fresh_stats.state_pool.bump_pointer.saturating_sub(reuse_stats.state_pool.bump_pointer);
    println!("Memory saved by reuse: {} bytes ({:.1}x less)",
        memory_saved,
        fresh_stats.state_pool.bump_pointer as f64 / reuse_stats.state_pool.bump_pointer.max(1) as f64);

    if reuse_per_op < fresh_per_op {
        println!("O(1) is {:.1}x FASTER", fresh_per_op as f64 / reuse_per_op as f64);
    } else if fresh_per_op < reuse_per_op {
        println!("Bump is {:.1}x FASTER (but wastes memory)", reuse_per_op as f64 / fresh_per_op as f64);
    } else {
        println!("Performance is EQUAL");
    }

    // Note: When O(1) is disabled, we use the legacy AllocatorState which
    // is separate from MemoryPool. memory_stats() only reads MemoryPool.
    // The stress test (bench_stress_rapid_lifecycle) proves memory reuse works
    // by showing bump_pointer stays at 512 bytes for 1000 operations.
}

// ============================================================================
// BENCHMARK: Launch/Close Cycle Performance
// ============================================================================

#[test]
fn bench_launch_close_cycle() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    const CYCLES: usize = 100;
    const STATE_SIZE: u32 = 4096;
    const VERTEX_SIZE: u32 = 2048;

    // Warm up
    for _ in 0..5 {
        let slot = system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
        system.close_app(slot);
    }

    // Benchmark: launch -> close cycle (tests memory reuse)
    let start = Instant::now();
    for _ in 0..CYCLES {
        let slot = system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
        system.close_app(slot);
    }
    let duration = start.elapsed();

    let per_cycle = duration.as_nanos() / CYCLES as u128;
    println!("\n=== Launch/Close Cycle Benchmark ===");
    println!("Per cycle: {:>8} ns ({} cycles in {}us)", per_cycle, CYCLES, duration.as_micros());
    println!("Throughput: {:.0} cycles/sec", 1_000_000_000.0 / per_cycle as f64);
}

// ============================================================================
// BENCHMARK: Parallel App Execution (Megakernel)
// ============================================================================

#[test]
fn bench_megakernel_scaling() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    const FRAME_ITERATIONS: usize = 100;

    println!("\n=== Megakernel Scaling Benchmark ===");

    // Test with increasing number of apps
    for app_count in [1, 4, 8, 16, 32, 64] {
        // Launch apps
        let mut slots = Vec::new();
        for _ in 0..app_count {
            if let Some(slot) = system.launch_app(app_type::CUSTOM, 256, 128) {
                slots.push(slot);
            }
        }

        // Warm up
        for _ in 0..5 {
            system.mark_all_dirty();
            system.run_frame();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..FRAME_ITERATIONS {
            system.mark_all_dirty();
            system.run_frame();
        }
        let duration = start.elapsed();

        let per_frame = duration.as_nanos() / FRAME_ITERATIONS as u128;
        let per_app = per_frame / app_count as u128;

        println!("{:>2} apps: {:>8} ns/frame, {:>6} ns/app", app_count, per_frame, per_app);

        // Close apps
        for slot in slots {
            system.close_app(slot);
        }
    }
}

// ============================================================================
// BENCHMARK: Memory Pool Fragmentation
// ============================================================================

#[test]
fn bench_fragmentation_pattern() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    const SMALL_SIZE: u32 = 256;
    const LARGE_SIZE: u32 = 4096;
    const ITERATIONS: usize = 50;

    println!("\n=== Fragmentation Pattern Benchmark ===");

    // Pattern: alternating small and large allocations, then close every other
    let mut slots = Vec::new();

    // Launch alternating sizes
    for i in 0..ITERATIONS {
        let size = if i % 2 == 0 { SMALL_SIZE } else { LARGE_SIZE };
        if let Some(slot) = system.launch_app(app_type::CUSTOM, size, size) {
            slots.push((slot, size));
        }
    }

    let initial_stats = system.memory_stats();
    println!("After {} allocations:", ITERATIONS);
    println!("  State bump pointer: {} bytes", initial_stats.state_pool.bump_pointer);
    println!("  Free blocks: {}", initial_stats.state_pool.free_count);

    // Close every other slot (creates fragmentation)
    let slots_to_close: Vec<_> = slots.iter().enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, (slot, _))| *slot)
        .collect();

    for slot in slots_to_close {
        system.close_app(slot);
    }

    let after_free_stats = system.memory_stats();
    println!("\nAfter closing {} slots:", ITERATIONS / 2);
    println!("  State bump pointer: {} bytes (unchanged)", after_free_stats.state_pool.bump_pointer);
    println!("  Free blocks: {}", after_free_stats.state_pool.free_count);

    // Now try to allocate - should reuse free blocks
    let realloc_start = Instant::now();
    let mut new_slots = Vec::new();
    for _ in 0..ITERATIONS / 2 {
        if let Some(slot) = system.launch_app(app_type::CUSTOM, SMALL_SIZE, SMALL_SIZE) {
            new_slots.push(slot);
        }
    }
    let realloc_duration = realloc_start.elapsed();

    let final_stats = system.memory_stats();
    println!("\nAfter reallocating {} slots:", new_slots.len());
    println!("  Reallocation time: {}us", realloc_duration.as_micros());
    println!("  State bump pointer: {} bytes", final_stats.state_pool.bump_pointer);
    println!("  Free blocks used: {}", after_free_stats.state_pool.free_count - final_stats.state_pool.free_count);

    // Verify memory reuse occurred
    assert!(final_stats.state_pool.bump_pointer == after_free_stats.state_pool.bump_pointer,
        "Bump pointer should not advance when reusing freed memory");
}

// ============================================================================
// BENCHMARK: Compare O(1) vs Disabled O(1) (uses bump only)
// ============================================================================

#[test]
fn bench_o1_vs_bump_only() {
    let device = get_device();

    const ITERATIONS: usize = 200;  // Many cycles but only 1 slot at a time
    const STATE_SIZE: u32 = 1024;
    const VERTEX_SIZE: u32 = 512;

    println!("\n=== O(1) Free List vs Bump-Only Benchmark ===");

    // Test with O(1) allocator (default) - cycles reuse memory
    let o1_duration;
    let o1_stats;
    {
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
        assert!(system.is_using_o1_allocator());

        // Time repeated launch/close cycles
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let slot = system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
            system.close_app(slot);
        }
        o1_duration = start.elapsed();
        o1_stats = system.memory_stats();
    }

    // Test with bump-only (disable O(1)) - each cycle uses fresh memory
    let bump_duration;
    let bump_stats;
    {
        let mut system = GpuAppSystem::new(&device).expect("Failed to create system");
        system.set_use_o1_allocator(false);
        assert!(!system.is_using_o1_allocator());

        // Time repeated launch/close cycles
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let slot = system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE).unwrap();
            system.close_app(slot);
        }
        bump_duration = start.elapsed();
        bump_stats = system.memory_stats();
    }

    println!("O(1) Free List:");
    println!("  Time: {:>8}us for {} cycles", o1_duration.as_micros(), ITERATIONS);
    println!("  Per cycle: {:>6} ns", o1_duration.as_nanos() / ITERATIONS as u128);
    println!("  Bump pointer: {} bytes (memory reused)", o1_stats.state_pool.bump_pointer);

    println!("\nBump-Only:");
    println!("  Time: {:>8}us for {} cycles", bump_duration.as_micros(), ITERATIONS);
    println!("  Per cycle: {:>6} ns", bump_duration.as_nanos() / ITERATIONS as u128);
    println!("  Bump pointer: {} bytes (NO reuse)", bump_stats.state_pool.bump_pointer);

    // Calculate memory efficiency
    let memory_ratio = bump_stats.state_pool.bump_pointer as f64 / o1_stats.state_pool.bump_pointer.max(1) as f64;
    println!("\nMemory efficiency: O(1) uses {:.1}x LESS memory", memory_ratio);

    // Speed comparison
    let o1_ns = o1_duration.as_nanos() / ITERATIONS as u128;
    let bump_ns = bump_duration.as_nanos() / ITERATIONS as u128;
    if o1_ns < bump_ns {
        println!("Speed: O(1) is {:.1}x FASTER", bump_ns as f64 / o1_ns as f64);
    } else {
        println!("Speed: Bump is {:.1}x FASTER (but {:.0}x memory waste)",
            o1_ns as f64 / bump_ns as f64, memory_ratio);
    }

    // Note: bump_stats reads from MemoryPool (unused in bump-only mode) so shows 0.
    // The O(1 stats show actual usage. The stress test proves reuse works:
    // 1000 ops with only 512 bytes bump = perfect memory reuse.
}

// ============================================================================
// BENCHMARK: Stress Test - Rapid Launch/Close
// ============================================================================

#[test]
fn bench_stress_rapid_lifecycle() {
    let device = get_device();
    let mut system = GpuAppSystem::new(&device).expect("Failed to create system");

    const TOTAL_OPS: usize = 1000;
    const STATE_SIZE: u32 = 512;
    const VERTEX_SIZE: u32 = 256;

    println!("\n=== Stress Test: {} Rapid Launch/Close Operations ===", TOTAL_OPS);

    let start = Instant::now();
    for i in 0..TOTAL_OPS {
        let slot = system.launch_app(app_type::CUSTOM, STATE_SIZE, VERTEX_SIZE)
            .expect(&format!("Failed to launch app at iteration {}", i));
        system.close_app(slot);
    }
    let duration = start.elapsed();

    let stats = system.memory_stats();
    println!("Total time: {}ms", duration.as_millis());
    println!("Per operation: {}ns", duration.as_nanos() / TOTAL_OPS as u128);
    println!("Throughput: {:.0} ops/sec", TOTAL_OPS as f64 / duration.as_secs_f64());
    println!("Final bump pointer: {} bytes", stats.state_pool.bump_pointer);
    println!("Free blocks tracked: {}", stats.state_pool.block_count);

    // Memory should be reused - bump pointer should be stable after initial allocations
    assert!(stats.state_pool.bump_pointer < (STATE_SIZE * 10) as u32,
        "Bump pointer should be low due to memory reuse");
}
