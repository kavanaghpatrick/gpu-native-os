// Tests for Issue #11: Unified Worker Model - Core GPU Kernel
//
// These tests verify the core GPU kernel implementation.
// Run with: cargo test --test test_issue_11_kernel

use metal::Device;
use rust_experiment::gpu_os::kernel::{GpuKernel, KernelParams};
use rust_experiment::gpu_os::memory::GpuMemory;

fn setup() -> (Device, GpuMemory) {
    let device = Device::system_default().expect("No Metal device available");
    let memory = GpuMemory::new(&device, 1024);
    (device, memory)
}

fn default_params() -> KernelParams {
    KernelParams {
        widget_count: 256,
        max_widgets: 1024,
        delta_time: 1.0 / 120.0,
        time: 0.0,
        frame_number: 0,
    }
}

#[test]
fn test_kernel_compiles() {
    let device = Device::system_default().expect("No Metal device available");
    let kernel = GpuKernel::new(&device);
    assert!(kernel.is_ok(), "Kernel should compile without errors: {:?}", kernel.err());
}

#[test]
fn test_kernel_supports_1024_threads() {
    let device = Device::system_default().expect("No Metal device available");
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");

    assert!(
        kernel.supports_1024_threads(),
        "Kernel must support 1024 threads per threadgroup. Got: {}",
        kernel.max_threads_per_threadgroup()
    );
}

#[test]
fn test_kernel_executes_all_threads() {
    let (device, memory) = setup();
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");
    let queue = device.new_command_queue();

    let params = default_params();
    let stats = kernel.run_with_stats(&queue, &memory, &params);

    assert!(
        stats.threads_executed >= 1024,
        "All 1024 threads must execute. Got: {}",
        stats.threads_executed
    );
}

#[test]
fn test_kernel_completes_all_phases() {
    let (device, memory) = setup();
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");
    let queue = device.new_command_queue();

    let params = default_params();
    let stats = kernel.run_with_stats(&queue, &memory, &params);

    // Should complete at least 5 phases: input, hit test, visibility, sort, state update
    assert!(
        stats.phases_completed >= 5,
        "Kernel must complete all phases. Got: {}",
        stats.phases_completed
    );
}

#[test]
fn test_kernel_execution_time_under_2ms() {
    let (device, memory) = setup();
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");
    let queue = device.new_command_queue();

    let params = default_params();

    // Warm up
    for _ in 0..5 {
        kernel.run_sync(&queue, &memory, &params);
    }

    // Benchmark
    let mut times = Vec::new();
    for i in 0..100 {
        let params = KernelParams {
            frame_number: i as u32,
            ..params
        };
        let time_ms = kernel.run_sync(&queue, &memory, &params);
        times.push(time_ms);
    }

    let avg_ms = times.iter().sum::<f64>() / times.len() as f64;

    assert!(
        avg_ms < 2.0,
        "Kernel execution must be under 2ms. Got: {:.3}ms",
        avg_ms
    );
}

#[test]
fn test_kernel_no_fixed_simd_roles() {
    // This test verifies that all threads participate in all phases,
    // not just specific SIMD groups assigned to specific roles.

    let (device, memory) = setup();
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");
    let queue = device.new_command_queue();

    let params = default_params();
    let stats = kernel.run_with_stats(&queue, &memory, &params);

    // If using fixed SIMD roles, we'd see only 32 threads (1 SIMD group) per phase
    // With unified worker model, we should see all 1024 threads
    assert!(
        stats.threads_executed >= 1024,
        "Unified worker model requires ALL threads to participate"
    );
}

#[test]
fn test_test_kernel_runs_all_threads() {
    let device = Device::system_default().expect("No Metal device available");
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");
    let queue = device.new_command_queue();

    let results = kernel.run_test_kernel(&queue, &device);

    // Each thread should write its ID + 1
    let threads = results.len().min(1024);
    let mut all_executed = true;
    for i in 0..threads {
        if results[i] != (i as u32 + 1) {
            all_executed = false;
            break;
        }
    }

    assert!(
        all_executed,
        "Test kernel should execute all {} threads",
        threads
    );
}

#[test]
fn test_frame_state_updates() {
    let (device, memory) = setup();
    let kernel = GpuKernel::new(&device).expect("Kernel should compile");
    let queue = device.new_command_queue();

    // Run kernel with frame 0
    let params = KernelParams {
        widget_count: 10,
        max_widgets: 1024,
        delta_time: 0.0083,
        time: 1.0,
        frame_number: 0,
    };

    kernel.run_sync(&queue, &memory, &params);

    // Check frame state was updated
    let frame_state = memory.frame_state();
    assert_eq!(
        frame_state.frame_number, 1,
        "Frame number should increment from 0 to 1"
    );
}
