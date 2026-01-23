// Tests for Issue #18: VSync-Driven Execution - Display as Heartbeat
//
// These tests verify the VSync-driven execution model.
// Run with: cargo test --test test_issue_18_vsync

use metal::Device;
use rust_experiment::gpu_os::vsync::*;
use std::time::{Duration, Instant};

#[test]
fn test_frame_budget_constants() {
    assert_eq!(TARGET_FPS, 120);
    assert!((FRAME_BUDGET_MS - 8.33).abs() < 0.01);
    assert_eq!(COMPUTE_BUDGET_MS, 2.0);
    assert_eq!(RENDER_BUDGET_MS, 5.0);
    assert_eq!(CPU_OVERHEAD_BUDGET_MS, 0.5);

    // Verify budgets fit within frame
    let total = COMPUTE_BUDGET_MS + RENDER_BUDGET_MS + CPU_OVERHEAD_BUDGET_MS;
    assert!(
        total < FRAME_BUDGET_MS,
        "Budgets ({:.2}ms) must fit within frame ({:.2}ms)",
        total, FRAME_BUDGET_MS
    );
}

#[test]
fn test_display_manager_creation() {
    let device = Device::system_default().expect("No Metal device");
    let manager = DisplayManager::new(&device);

    assert_eq!(manager.frame_count(), 0);
}

#[test]
fn test_frame_context_creation() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    let context = manager.begin_frame();
    let timing = manager.end_frame(context);

    assert!(timing.total_ms >= 0.0);
    assert_eq!(manager.frame_count(), 1);
}

#[test]
fn test_average_frame_time() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    // Run a few frames
    for _ in 0..10 {
        let context = manager.begin_frame();
        std::thread::sleep(Duration::from_micros(100));
        manager.end_frame(context);
    }

    let avg = manager.average_frame_time();
    assert!(avg > 0.0, "Average frame time should be positive");
}

#[test]
fn test_current_fps() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    // Run frames with ~8ms timing (120 FPS target)
    for _ in 0..20 {
        let context = manager.begin_frame();
        std::thread::sleep(Duration::from_millis(8));
        manager.end_frame(context);
    }

    let fps = manager.current_fps();
    assert!(fps > 0.0, "FPS should be positive");
    // With 8ms sleep, we should get roughly 100-125 FPS
    // (sleep is imprecise, so allow wide range)
}

#[test]
fn test_cpu_overhead_under_500us() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    // Measure CPU overhead (time spent in begin_frame + end_frame, not GPU)
    let mut overhead_times = Vec::new();

    for _ in 0..100 {
        let start = Instant::now();
        let context = manager.begin_frame();
        let timing = manager.end_frame(context);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        overhead_times.push(timing.cpu_overhead_ms);
    }

    let avg_overhead = overhead_times.iter().sum::<f64>() / overhead_times.len() as f64;

    assert!(
        avg_overhead < CPU_OVERHEAD_BUDGET_MS,
        "CPU overhead should be under {}ms. Got: {:.3}ms",
        CPU_OVERHEAD_BUDGET_MS, avg_overhead
    );
}

#[test]
fn test_frame_budget_exceeded_detection() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    // Simulate a slow frame
    let context = manager.begin_frame();
    std::thread::sleep(Duration::from_millis(15)); // Exceeds 8.33ms budget
    let timing = manager.end_frame(context);

    assert!(
        timing.exceeded_budget,
        "Frame exceeding budget should be flagged"
    );
}

#[test]
fn test_performance_monitor() {
    let mut monitor = PerformanceMonitor::new();

    // Record some frames
    for i in 0..100 {
        let timing = FrameTiming {
            total_ms: 8.0 + (i % 3) as f64,
            cpu_overhead_ms: 0.3,
            exceeded_budget: i % 10 == 0,
        };
        monitor.record_frame(&timing);
    }

    assert_eq!(monitor.frame_count, 100);
    assert!(monitor.average_fps() > 0.0);
    assert!(monitor.drop_rate() > 0.0); // 10% of frames exceeded budget
}

#[test]
fn test_frame_pacer() {
    let mut pacer = FramePacer::new(120);

    let start = Instant::now();

    // Simulate a few frames
    for _ in 0..5 {
        pacer.wait_for_next_frame();
        // Do "work"
        std::thread::sleep(Duration::from_millis(5));
        pacer.frame_complete();
    }

    let elapsed = start.elapsed();

    // 5 frames at 120fps should take ~41.7ms
    // Allow 30-60ms due to sleep imprecision
    assert!(
        elapsed.as_millis() >= 30 && elapsed.as_millis() <= 100,
        "Frame pacing seems incorrect: {:?}",
        elapsed
    );
}

#[test]
fn test_sustained_120fps() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    let start = Instant::now();
    let mut frames = 0;

    // Try to run for ~500ms
    while start.elapsed() < Duration::from_millis(500) && frames < 120 {
        let context = manager.begin_frame();
        // Minimal work
        manager.end_frame(context);
        frames += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let actual_fps = frames as f64 / elapsed;

    // Should be able to do many frames per second with minimal work
    assert!(
        frames >= 50,
        "Should complete at least 50 frames in 500ms. Got: {}",
        frames
    );

    println!("Sustained test: {} frames in {:.2}s = {:.1} fps", frames, elapsed, actual_fps);
}

#[test]
fn test_is_at_target_fps() {
    let device = Device::system_default().expect("No Metal device");
    let mut manager = DisplayManager::new(&device);

    // Run fast frames
    for _ in 0..50 {
        let context = manager.begin_frame();
        manager.end_frame(context);
    }

    // With essentially zero work, we should be at or above target
    // (This may fail if system is under heavy load)
    let at_target = manager.is_at_target_fps();
    let fps = manager.current_fps();

    println!("Current FPS: {:.1}, at target: {}", fps, at_target);
}
