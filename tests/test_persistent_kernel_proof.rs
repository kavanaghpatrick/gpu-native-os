//! PERSISTENT KERNEL PROOF TEST
//!
//! Run with: cargo test --test test_persistent_kernel_proof -- --nocapture

use metal::*;
use std::time::{Duration, Instant};

#[repr(C)]
struct Control {
    heartbeat: u32,
    shutdown: u32,
    iterations: u32,
    work_done: u32,
}

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Control {
    atomic_uint heartbeat;
    atomic_uint shutdown;
    atomic_uint iterations;
    atomic_uint work_done;
};

kernel void persistent_kernel(
    device Control* ctrl [[buffer(0)]],
    device float* data [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        atomic_fetch_add_explicit(&ctrl->heartbeat, 1, memory_order_relaxed);
    }

    uint iter = 0;

    while (true) {
        // Check shutdown
        uint shutdown = atomic_load_explicit(&ctrl->shutdown, memory_order_relaxed);
        if (shutdown != 0) break;

        // All threads do parallel work
        uint my_idx = tid % 1024;
        data[my_idx] = data[my_idx] * 0.9999f + 0.0001f;

        // Thread 0 updates stats
        if (tid == 0 && (iter % 50000 == 0)) {
            atomic_fetch_add_explicit(&ctrl->heartbeat, 1, memory_order_relaxed);
            atomic_store_explicit(&ctrl->iterations, iter, memory_order_relaxed);
            atomic_fetch_add_explicit(&ctrl->work_done, 1, memory_order_relaxed);
        }

        iter++;
    }

    if (tid == 0) {
        atomic_store_explicit(&ctrl->iterations, iter, memory_order_relaxed);
        atomic_fetch_add_explicit(&ctrl->heartbeat, 1, memory_order_relaxed);
    }
}
"#;

fn read_control(ctrl_buffer: &Buffer) -> (u32, u32, u32) {
    unsafe {
        let ctrl = ctrl_buffer.contents() as *const Control;
        (
            std::ptr::read_volatile(&(*ctrl).heartbeat),
            std::ptr::read_volatile(&(*ctrl).iterations),
            std::ptr::read_volatile(&(*ctrl).work_done),
        )
    }
}

#[test]
fn test_persistent_kernel() {
    println!("\n========================================");
    println!("PERSISTENT KERNEL PROOF TEST");
    println!("========================================\n");

    let device = Device::system_default().expect("No Metal device");
    println!("Device: {}", device.name());

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(SHADER, &options)
        .expect("Shader compile failed");

    let function = library
        .get_function("persistent_kernel", None)
        .expect("Function not found");

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed");

    let ctrl_buffer = device.new_buffer(
        std::mem::size_of::<Control>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let data_buffer = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);

    unsafe {
        let ctrl = ctrl_buffer.contents() as *mut Control;
        (*ctrl).heartbeat = 0;
        (*ctrl).shutdown = 0;
        (*ctrl).iterations = 0;
        (*ctrl).work_done = 0;

        let data = data_buffer.contents() as *mut f32;
        for i in 0..1024 {
            *data.add(i) = 1.0;
        }
    }

    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&ctrl_buffer), 0);
    enc.set_buffer(1, Some(&data_buffer), 0);

    // 32 threads (one SIMD group)
    enc.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();

    println!("Dispatching (SINGLE dispatch)...\n");
    let start = Instant::now();
    cmd.commit();

    let test_duration = Duration::from_secs(15);
    let mut last_hb = 0u32;
    let mut advancing_count = 0;
    let mut stuck_since: Option<Instant> = None;

    println!("  Time   Heartbeat  Iterations     Work  Status");

    while start.elapsed() < test_duration {
        std::thread::sleep(Duration::from_millis(500));

        let (hb, iter, work) = read_control(&ctrl_buffer);
        let elapsed = start.elapsed().as_secs_f64();

        let status = if hb > last_hb {
            advancing_count += 1;
            stuck_since = None;
            "RUNNING"
        } else {
            if stuck_since.is_none() {
                stuck_since = Some(Instant::now());
            }
            let stuck_time = stuck_since.unwrap().elapsed().as_secs_f64();
            if stuck_time > 3.0 { "STALLED" } else { "..." }
        };

        println!(
            "  {:>4.1}s  {:>9}  {:>10}  {:>6}  {}",
            elapsed, hb, iter, work, status
        );

        if cmd.status() == MTLCommandBufferStatus::Completed {
            println!("\n  *** KERNEL KILLED BY SYSTEM ***");
            break;
        }

        last_hb = hb;
    }

    // Shutdown
    println!("\nSending shutdown...");
    unsafe {
        let ctrl = ctrl_buffer.contents() as *mut Control;
        std::ptr::write_volatile(&mut (*ctrl).shutdown, 1);
    }

    // Wait for exit
    let mut shutdown_ok = false;
    for _ in 0..50 {
        if cmd.status() == MTLCommandBufferStatus::Completed {
            shutdown_ok = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let (final_hb, final_iter, final_work) = read_control(&ctrl_buffer);
    let total_time = start.elapsed().as_secs_f64();

    println!("\n========================================");
    println!("RESULTS");
    println!("========================================");
    println!("Runtime: {:.1}s", total_time);
    println!("Heartbeats: {}", final_hb);
    println!("Iterations: {}", final_iter);
    println!("Work units: {}", final_work);
    println!("Shutdown OK: {}", shutdown_ok);
    println!("Advancing count: {}", advancing_count);

    // Analysis
    println!("\n========================================");
    println!("ANALYSIS");
    println!("========================================");

    if cmd.status() == MTLCommandBufferStatus::Completed && shutdown_ok && advancing_count > 0 {
        println!("SUCCESS: Persistent kernel works!");
        println!("- Kernel ran continuously");
        println!("- CPU could communicate via atomics");
        println!("- Clean shutdown worked");
    } else if advancing_count > 0 {
        println!("PARTIAL: Kernel ran but stalled or didn't shutdown");
        if !shutdown_ok {
            println!("- Shutdown signal not received (cache coherency issue?)");
        }
    } else {
        println!("FAILED: Kernel never advanced heartbeat");
    }
    println!("========================================\n");
}
