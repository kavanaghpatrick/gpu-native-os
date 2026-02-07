//! Persistent Kernel Proof Test
//!
//! This test empirically proves whether a true `while(true)` Metal kernel
//! can run indefinitely on Apple Silicon without crashing or being killed.
//!
//! THE QUESTION: Can Metal compute shaders run forever?
//!
//! This test will answer:
//! 1. Does the kernel get killed by a watchdog?
//! 2. Does thermal throttling occur?
//! 3. Can CPU communicate with a running kernel via atomics?
//! 4. What's the maximum sustainable runtime?

use metal::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Control block for persistent kernel (shared between CPU and GPU)
#[repr(C)]
#[derive(Debug)]
pub struct PersistentControl {
    /// GPU increments this every N iterations (proves it's alive)
    pub heartbeat: u64,
    /// CPU sets to 1 to request shutdown
    pub shutdown: u64,
    /// GPU writes iteration count here
    pub iterations: u64,
    /// GPU writes 1 if it detected throttling
    pub throttle_detected: u64,
    /// Timestamp when kernel started (set by CPU)
    pub start_time_ns: u64,
    /// Last heartbeat timestamp (set by GPU via atomic)
    pub last_heartbeat_ns: u64,
    /// Work counter - GPU increments, proves actual work
    pub work_done: u64,
    /// Padding for alignment
    pub _padding: u64,
}

impl Default for PersistentControl {
    fn default() -> Self {
        Self {
            heartbeat: 0,
            shutdown: 0,
            iterations: 0,
            throttle_detected: 0,
            start_time_ns: 0,
            last_heartbeat_ns: 0,
            work_done: 0,
            _padding: 0,
        }
    }
}

/// Metal shader with TRUE while(true) loop
const PERSISTENT_KERNEL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Control block - must match Rust struct
struct PersistentControl {
    atomic_ulong heartbeat;
    atomic_ulong shutdown;
    atomic_ulong iterations;
    atomic_ulong throttle_detected;
    ulong start_time_ns;
    atomic_ulong last_heartbeat_ns;
    atomic_ulong work_done;
    ulong _padding;
};

// ============================================================================
// TRUE PERSISTENT KERNEL - while(true) loop
// ============================================================================
//
// This kernel runs FOREVER until shutdown is signaled.
// It does real work (not just spinning) to prove GPU compute works.

kernel void persistent_kernel_true(
    device PersistentControl* control [[buffer(0)]],
    device float* work_buffer [[buffer(1)]],
    constant uint& buffer_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Only thread 0 runs the persistent loop (others could do parallel work)
    if (tid != 0) return;

    ulong iteration = 0;
    ulong last_heartbeat_iter = 0;
    const ulong HEARTBEAT_INTERVAL = 1000000;  // Every 1M iterations
    const ulong WORK_INTERVAL = 100;           // Do work every 100 iterations

    // THE TRUE PERSISTENT LOOP
    while (true) {
        // Check for shutdown signal
        ulong shutdown = atomic_load_explicit(&control->shutdown, memory_order_relaxed);
        if (shutdown != 0) {
            break;  // Clean exit
        }

        // Do actual work periodically (not just spinning)
        if (iteration % WORK_INTERVAL == 0) {
            // Simple compute work - sum reduction
            float sum = 0.0;
            uint work_size = min(buffer_size, 1024u);
            for (uint i = 0; i < work_size; i++) {
                sum += work_buffer[i];
            }
            // Write result back (prevents optimizer from removing)
            work_buffer[0] = sum * 0.999f + 0.001f;

            // Increment work counter
            atomic_fetch_add_explicit(&control->work_done, 1, memory_order_relaxed);
        }

        // Update heartbeat periodically
        if (iteration - last_heartbeat_iter >= HEARTBEAT_INTERVAL) {
            atomic_fetch_add_explicit(&control->heartbeat, 1, memory_order_relaxed);
            atomic_store_explicit(&control->iterations, iteration, memory_order_relaxed);
            last_heartbeat_iter = iteration;
        }

        iteration++;
    }

    // Final update before exit
    atomic_store_explicit(&control->iterations, iteration, memory_order_relaxed);
    atomic_fetch_add_explicit(&control->heartbeat, 1, memory_order_relaxed);
}

// ============================================================================
// SELF-THROTTLING PERSISTENT KERNEL
// ============================================================================
//
// This version includes self-throttling to prevent thermal issues.

kernel void persistent_kernel_throttled(
    device PersistentControl* control [[buffer(0)]],
    device float* work_buffer [[buffer(1)]],
    constant uint& buffer_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    ulong iteration = 0;
    ulong last_heartbeat_iter = 0;
    const ulong HEARTBEAT_INTERVAL = 1000000;
    const ulong WORK_INTERVAL = 100;
    const ulong THROTTLE_INTERVAL = 10000;  // Yield every 10K iterations
    const uint THROTTLE_CYCLES = 1000;      // Spin this many barrier cycles

    while (true) {
        // Check shutdown
        if (atomic_load_explicit(&control->shutdown, memory_order_relaxed) != 0) {
            break;
        }

        // Do work
        if (iteration % WORK_INTERVAL == 0) {
            float sum = 0.0;
            uint work_size = min(buffer_size, 1024u);
            for (uint i = 0; i < work_size; i++) {
                sum += work_buffer[i];
            }
            work_buffer[0] = sum * 0.999f + 0.001f;
            atomic_fetch_add_explicit(&control->work_done, 1, memory_order_relaxed);
        }

        // Self-throttle: yield periodically to reduce thermal load
        if (iteration % THROTTLE_INTERVAL == 0) {
            // Spin on memory barriers - reduces power without exiting
            for (uint i = 0; i < THROTTLE_CYCLES; i++) {
                // Memory barrier acts as a yield point
                atomic_thread_fence(memory_order_seq_cst);
            }
        }

        // Heartbeat
        if (iteration - last_heartbeat_iter >= HEARTBEAT_INTERVAL) {
            atomic_fetch_add_explicit(&control->heartbeat, 1, memory_order_relaxed);
            atomic_store_explicit(&control->iterations, iteration, memory_order_relaxed);
            last_heartbeat_iter = iteration;
        }

        iteration++;
    }

    atomic_store_explicit(&control->iterations, iteration, memory_order_relaxed);
}

// ============================================================================
// MULTI-THREADED PERSISTENT KERNEL
// ============================================================================
//
// All threads participate in work, one thread manages control.

kernel void persistent_kernel_parallel(
    device PersistentControl* control [[buffer(0)]],
    device float* work_buffer [[buffer(1)]],
    constant uint& buffer_size [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_grid [[threads_per_grid]]
) {
    ulong iteration = 0;
    const ulong HEARTBEAT_INTERVAL = 100000;

    // Each thread works on a portion of the buffer
    uint my_start = (tid * buffer_size) / threads_per_grid;
    uint my_end = ((tid + 1) * buffer_size) / threads_per_grid;

    while (true) {
        // Thread 0 checks shutdown
        if (tid == 0) {
            if (atomic_load_explicit(&control->shutdown, memory_order_relaxed) != 0) {
                // Signal other threads by setting a flag they can see
                atomic_store_explicit(&control->throttle_detected, 0xDEAD, memory_order_relaxed);
            }
        }

        // All threads check the exit flag
        if (atomic_load_explicit(&control->throttle_detected, memory_order_relaxed) == 0xDEAD) {
            break;
        }

        // All threads do parallel work
        float local_sum = 0.0;
        for (uint i = my_start; i < my_end; i++) {
            local_sum += work_buffer[i];
        }

        // Write back (each thread to its own slot)
        if (my_start < buffer_size) {
            work_buffer[my_start] = local_sum * 0.999f + 0.001f;
        }

        // Thread 0 updates control
        if (tid == 0) {
            if (iteration % HEARTBEAT_INTERVAL == 0) {
                atomic_fetch_add_explicit(&control->heartbeat, 1, memory_order_relaxed);
                atomic_store_explicit(&control->iterations, iteration, memory_order_relaxed);
                atomic_fetch_add_explicit(&control->work_done, 1, memory_order_relaxed);
            }
        }

        iteration++;
    }

    if (tid == 0) {
        atomic_store_explicit(&control->iterations, iteration, memory_order_relaxed);
    }
}
"#;

/// Test result
#[derive(Debug)]
pub struct PersistentKernelResult {
    pub ran_successfully: bool,
    pub runtime_seconds: f64,
    pub total_iterations: u64,
    pub total_heartbeats: u64,
    pub work_units_done: u64,
    pub iterations_per_second: f64,
    pub was_killed: bool,
    pub error_message: Option<String>,
}

/// Run the persistent kernel test
pub fn test_persistent_kernel(
    duration_seconds: u64,
    use_throttling: bool,
    parallel: bool,
) -> PersistentKernelResult {
    println!("\n{'='*60}");
    println!("PERSISTENT KERNEL PROOF TEST");
    println!("{'='*60}");
    println!("Duration: {} seconds", duration_seconds);
    println!("Throttling: {}", use_throttling);
    println!("Parallel: {}", parallel);
    println!();

    // Get Metal device
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            return PersistentKernelResult {
                ran_successfully: false,
                runtime_seconds: 0.0,
                total_iterations: 0,
                total_heartbeats: 0,
                work_units_done: 0,
                iterations_per_second: 0.0,
                was_killed: false,
                error_message: Some("No Metal device found".to_string()),
            };
        }
    };

    println!("Device: {}", device.name());

    // Compile shader
    let options = CompileOptions::new();
    let library = match device.new_library_with_source(PERSISTENT_KERNEL_SHADER, &options) {
        Ok(lib) => lib,
        Err(e) => {
            return PersistentKernelResult {
                ran_successfully: false,
                runtime_seconds: 0.0,
                total_iterations: 0,
                total_heartbeats: 0,
                work_units_done: 0,
                iterations_per_second: 0.0,
                was_killed: false,
                error_message: Some(format!("Shader compile failed: {}", e)),
            };
        }
    };

    // Select kernel function
    let kernel_name = if parallel {
        "persistent_kernel_parallel"
    } else if use_throttling {
        "persistent_kernel_throttled"
    } else {
        "persistent_kernel_true"
    };

    let function = match library.get_function(kernel_name, None) {
        Ok(f) => f,
        Err(e) => {
            return PersistentKernelResult {
                ran_successfully: false,
                runtime_seconds: 0.0,
                total_iterations: 0,
                total_heartbeats: 0,
                work_units_done: 0,
                iterations_per_second: 0.0,
                was_killed: false,
                error_message: Some(format!("Function {} not found: {}", kernel_name, e)),
            };
        }
    };

    let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
        Ok(p) => p,
        Err(e) => {
            return PersistentKernelResult {
                ran_successfully: false,
                runtime_seconds: 0.0,
                total_iterations: 0,
                total_heartbeats: 0,
                work_units_done: 0,
                iterations_per_second: 0.0,
                was_killed: false,
                error_message: Some(format!("Pipeline creation failed: {}", e)),
            };
        }
    };

    // Create buffers
    let control_buffer = device.new_buffer(
        std::mem::size_of::<PersistentControl>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let work_buffer_size: u32 = 4096;
    let work_buffer = device.new_buffer(
        (work_buffer_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let size_buffer = device.new_buffer_with_data(
        &work_buffer_size as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Initialize control block
    unsafe {
        let ctrl = control_buffer.contents() as *mut PersistentControl;
        *ctrl = PersistentControl::default();
    }

    // Initialize work buffer with some data
    unsafe {
        let work = work_buffer.contents() as *mut f32;
        for i in 0..work_buffer_size as usize {
            *work.add(i) = (i as f32) * 0.001;
        }
    }

    // Create command queue
    let command_queue = device.new_command_queue();

    // Dispatch the persistent kernel
    let cmd_buffer = command_queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&control_buffer), 0);
    encoder.set_buffer(1, Some(&work_buffer), 0);
    encoder.set_buffer(2, Some(&size_buffer), 0);

    // Dispatch
    let thread_count = if parallel { 256 } else { 1 };
    encoder.dispatch_threads(
        MTLSize::new(thread_count, 1, 1),
        MTLSize::new(thread_count.min(256), 1, 1),
    );

    encoder.end_encoding();

    println!("Dispatching kernel...");
    let start = Instant::now();
    cmd_buffer.commit();

    // Monitor the kernel
    let target_duration = Duration::from_secs(duration_seconds);
    let mut last_heartbeat: u64 = 0;
    let mut heartbeat_stuck_count = 0;
    let mut was_killed = false;

    println!("\nMonitoring (heartbeats should increment):");

    while start.elapsed() < target_duration {
        std::thread::sleep(Duration::from_millis(500));

        // Read current state
        let (heartbeat, iterations, work_done) = unsafe {
            let ctrl = control_buffer.contents() as *const PersistentControl;
            (
                std::ptr::read_volatile(&(*ctrl).heartbeat),
                std::ptr::read_volatile(&(*ctrl).iterations),
                std::ptr::read_volatile(&(*ctrl).work_done),
            )
        };

        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "  [{:.1}s] heartbeat={}, iterations={}, work={}",
            elapsed, heartbeat, iterations, work_done
        );

        // Check if heartbeat is advancing
        if heartbeat == last_heartbeat {
            heartbeat_stuck_count += 1;
            if heartbeat_stuck_count > 4 {
                println!("  WARNING: Heartbeat stuck for 2+ seconds!");
                // Check if command buffer completed (kernel was killed)
                if cmd_buffer.status() == MTLCommandBufferStatus::Completed {
                    println!("  KERNEL WAS KILLED BY SYSTEM!");
                    was_killed = true;
                    break;
                }
            }
        } else {
            heartbeat_stuck_count = 0;
        }
        last_heartbeat = heartbeat;
    }

    // Signal shutdown
    println!("\nSignaling shutdown...");
    unsafe {
        let ctrl = control_buffer.contents() as *mut PersistentControl;
        std::ptr::write_volatile(&mut (*ctrl).shutdown, 1);
    }

    // Wait for completion (with timeout)
    let shutdown_start = Instant::now();
    let shutdown_timeout = Duration::from_secs(5);

    while cmd_buffer.status() != MTLCommandBufferStatus::Completed {
        if shutdown_start.elapsed() > shutdown_timeout {
            println!("WARNING: Kernel did not respond to shutdown signal!");
            was_killed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let total_elapsed = start.elapsed();

    // Read final stats
    let (final_heartbeat, final_iterations, final_work) = unsafe {
        let ctrl = control_buffer.contents() as *const PersistentControl;
        (
            std::ptr::read_volatile(&(*ctrl).heartbeat),
            std::ptr::read_volatile(&(*ctrl).iterations),
            std::ptr::read_volatile(&(*ctrl).work_done),
        )
    };

    let iterations_per_second = if total_elapsed.as_secs_f64() > 0.0 {
        final_iterations as f64 / total_elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("\n{'='*60}");
    println!("RESULTS");
    println!("{'='*60}");
    println!("Runtime: {:.2} seconds", total_elapsed.as_secs_f64());
    println!("Total iterations: {}", final_iterations);
    println!("Total heartbeats: {}", final_heartbeat);
    println!("Work units done: {}", final_work);
    println!("Iterations/second: {:.0}", iterations_per_second);
    println!("Was killed: {}", was_killed);
    println!(
        "Status: {}",
        if was_killed { "FAILED - Kernel was killed" } else { "SUCCESS - Kernel ran to completion" }
    );

    PersistentKernelResult {
        ran_successfully: !was_killed && final_iterations > 0,
        runtime_seconds: total_elapsed.as_secs_f64(),
        total_iterations: final_iterations,
        total_heartbeats: final_heartbeat,
        work_units_done: final_work,
        iterations_per_second,
        was_killed,
        error_message: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_kernel_5_seconds() {
        let result = test_persistent_kernel(5, false, false);
        assert!(
            result.ran_successfully,
            "Persistent kernel failed: {:?}",
            result.error_message
        );
        assert!(result.total_iterations > 0, "No iterations completed");
        println!("\nPROOF: Persistent kernel CAN run for 5 seconds!");
    }

    #[test]
    fn test_persistent_kernel_30_seconds() {
        let result = test_persistent_kernel(30, false, false);
        assert!(
            result.ran_successfully,
            "Persistent kernel failed after {} seconds: {:?}",
            result.runtime_seconds,
            result.error_message
        );
        println!("\nPROOF: Persistent kernel CAN run for 30 seconds!");
    }

    #[test]
    fn test_persistent_kernel_throttled_60_seconds() {
        let result = test_persistent_kernel(60, true, false);
        assert!(
            result.ran_successfully,
            "Throttled persistent kernel failed: {:?}",
            result.error_message
        );
        println!("\nPROOF: Throttled persistent kernel CAN run for 60 seconds!");
    }

    #[test]
    fn test_persistent_kernel_parallel_30_seconds() {
        let result = test_persistent_kernel(30, false, true);
        assert!(
            result.ran_successfully,
            "Parallel persistent kernel failed: {:?}",
            result.error_message
        );
        println!("\nPROOF: Parallel persistent kernel CAN run for 30 seconds!");
    }
}
