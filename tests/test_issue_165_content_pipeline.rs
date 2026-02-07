//! Issue #165: Content Pipeline Tests
//!
//! Tests for GPU-driven file I/O with CPU coprocessor.

use metal::Device;
use rust_experiment::gpu_os::content_pipeline::{
    ContentPipeline, IOCoprocessor,
    STATUS_LOADING, STATUS_READY, STATUS_CLOSED, STATUS_ERROR,
    INVALID_HANDLE, DEFAULT_POOL_SIZE,
};
use std::io::Write;
use std::sync::Arc;
use tempfile::NamedTempFile;

// ═══════════════════════════════════════════════════════════════════════════════
// BASIC TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_pipeline_creation() {
    let device = Device::system_default().expect("No Metal device");
    let pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    let stats = pipeline.read_stats();
    assert_eq!(stats.request_head, 0);
    assert_eq!(stats.request_tail, 0);
    assert_eq!(stats.handle_count, 0);
    println!("Pipeline created successfully");
}

#[test]
fn test_gpu_request_read() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Register a path
    let path_idx = pipeline.register_path("/tmp/test.txt".into());
    assert_eq!(path_idx, 0);

    // GPU requests read
    let handle_slots = pipeline.gpu_request_reads(&device, &[path_idx]);
    assert_eq!(handle_slots.len(), 1);
    assert_ne!(handle_slots[0], INVALID_HANDLE);

    // Check stats - request should be queued
    let stats = pipeline.read_stats();
    assert_eq!(stats.total_reads, 1);
    assert_eq!(stats.request_head, 1);

    // Handle should be in LOADING state
    let handle = pipeline.read_handle(handle_slots[0]);
    assert_eq!(handle.status, STATUS_LOADING);
    assert_eq!(handle.path_idx, path_idx);

    println!("GPU request read successful, handle slot: {}", handle_slots[0]);
}

#[test]
fn test_batch_request_reads() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 4 * 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Register multiple paths
    let mut path_indices = Vec::new();
    for i in 0..10 {
        let idx = pipeline.register_path(format!("/tmp/test{}.txt", i).into());
        path_indices.push(idx);
    }

    // GPU requests all reads
    let handle_slots = pipeline.gpu_request_reads(&device, &path_indices);
    assert_eq!(handle_slots.len(), 10);

    // All should have valid handles
    for slot in &handle_slots {
        assert_ne!(*slot, INVALID_HANDLE);
    }

    // Check stats
    let stats = pipeline.read_stats();
    assert_eq!(stats.total_reads, 10);
    assert_eq!(stats.request_head, 10);

    println!("Batch request for {} files successful", handle_slots.len());
}

// ═══════════════════════════════════════════════════════════════════════════════
// FILE I/O TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_read_small_file() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Create a test file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let test_content = b"Hello, GPU World!";
    temp_file.write_all(test_content).expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    // Register path
    let path_idx = pipeline.register_path(temp_file.path().to_path_buf());

    // GPU requests read
    let handle_slots = pipeline.gpu_request_reads(&device, &[path_idx]);
    assert_eq!(handle_slots.len(), 1);

    // CPU processes requests (simulating coprocessor)
    let processed = pipeline.process_requests();
    assert_eq!(processed, 1);

    // Handle should now be ready
    let handle = pipeline.read_handle(handle_slots[0]);
    assert_eq!(handle.status, STATUS_READY);
    assert_eq!(handle.file_size, test_content.len() as u32);

    // Read content from pool
    let content = pipeline.read_content(handle.content_offset, handle.file_size);
    assert_eq!(content, test_content);

    println!("Read small file successful: {:?}", String::from_utf8_lossy(&content));
}

#[test]
fn test_read_nonexistent_file() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Register path to nonexistent file
    let path_idx = pipeline.register_path("/this/file/does/not/exist.txt".into());

    // GPU requests read
    let handle_slots = pipeline.gpu_request_reads(&device, &[path_idx]);

    // CPU processes requests
    pipeline.process_requests();

    // Handle should be in ERROR state
    let handle = pipeline.read_handle(handle_slots[0]);
    assert_eq!(handle.status, STATUS_ERROR);
    assert_ne!(handle.error_code, 0);

    println!("Nonexistent file correctly returned error: {}", handle.error_code);
}

#[test]
fn test_concurrent_reads() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 4 * 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Create multiple test files
    let mut temp_files = Vec::new();
    let mut path_indices = Vec::new();

    for i in 0..20 {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        write!(temp_file, "File {} content: {}", i, "x".repeat(100)).expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let path_idx = pipeline.register_path(temp_file.path().to_path_buf());
        path_indices.push(path_idx);
        temp_files.push(temp_file);
    }

    // GPU requests all reads concurrently
    let handle_slots = pipeline.gpu_request_reads(&device, &path_indices);
    assert_eq!(handle_slots.len(), 20);

    // CPU processes all requests
    let processed = pipeline.process_requests();
    assert_eq!(processed, 20);

    // All should be ready
    let mut ready_count = 0;
    for slot in &handle_slots {
        let handle = pipeline.read_handle(*slot);
        if handle.status == STATUS_READY {
            ready_count += 1;
            // Verify content
            let content = pipeline.read_content(handle.content_offset, handle.file_size);
            assert!(content.len() > 0);
        }
    }

    assert_eq!(ready_count, 20);
    println!("All {} concurrent reads completed successfully", ready_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATUS POLLING TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_gpu_status_polling() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Create test file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(b"Test content").expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    let path_idx = pipeline.register_path(temp_file.path().to_path_buf());

    // GPU requests read
    let handle_slots = pipeline.gpu_request_reads(&device, &[path_idx]);

    // GPU polls status - should be LOADING
    let statuses = pipeline.gpu_check_status(&device, &handle_slots);
    assert_eq!(statuses[0], STATUS_LOADING);

    // CPU processes request
    pipeline.process_requests();

    // GPU polls status - should be READY
    let statuses = pipeline.gpu_check_status(&device, &handle_slots);
    assert_eq!(statuses[0], STATUS_READY);

    println!("GPU status polling works correctly");
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLOSE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_close_handle() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Create test file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(b"Test content").expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    let path_idx = pipeline.register_path(temp_file.path().to_path_buf());

    // GPU requests read
    let handle_slots = pipeline.gpu_request_reads(&device, &[path_idx]);

    // CPU processes request
    pipeline.process_requests();

    // GPU closes handle
    pipeline.gpu_close_handles(&device, &handle_slots);

    // CPU processes close request
    pipeline.process_requests();

    // Handle should be CLOSED
    let handle = pipeline.read_handle(handle_slots[0]);
    assert_eq!(handle.status, STATUS_CLOSED);

    println!("Handle close works correctly");
}

// ═══════════════════════════════════════════════════════════════════════════════
// COPROCESSOR TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_io_coprocessor() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Create test file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    temp_file.write_all(b"Coprocessor test content").expect("Failed to write");
    temp_file.flush().expect("Failed to flush");

    let path_idx = pipeline.register_path(temp_file.path().to_path_buf());

    // Wrap in Arc for sharing
    let pipeline = Arc::new(pipeline);

    // Start coprocessor
    let coprocessor = IOCoprocessor::new(pipeline.clone());
    let handle = coprocessor.start();

    // GPU requests read
    let handle_slots = pipeline.gpu_request_reads(&device, &[path_idx]);

    // Wait for coprocessor to process (poll status)
    for _ in 0..100 {
        let statuses = pipeline.gpu_check_status(&device, &handle_slots);
        if statuses[0] == STATUS_READY {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Check result
    let file_handle = pipeline.read_handle(handle_slots[0]);
    assert_eq!(file_handle.status, STATUS_READY);

    // Stop coprocessor
    // Note: In a real scenario, we'd have a proper shutdown mechanism
    println!("I/O coprocessor test successful");
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Run with --ignored
fn test_stress_many_files() {
    let device = Device::system_default().expect("No Metal device");
    let mut pipeline = ContentPipeline::new(&device, 64 * 1024 * 1024).expect("Failed to create pipeline");

    pipeline.initialize(&device);

    // Create 100 test files
    let mut temp_files = Vec::new();
    let mut path_indices = Vec::new();

    for i in 0..100 {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        write!(temp_file, "File {} with random content: {:?}",
               i, std::time::Instant::now()).expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let path_idx = pipeline.register_path(temp_file.path().to_path_buf());
        path_indices.push(path_idx);
        temp_files.push(temp_file);
    }

    let start = std::time::Instant::now();

    // GPU requests all reads
    let handle_slots = pipeline.gpu_request_reads(&device, &path_indices);

    let request_time = start.elapsed();
    println!("Requested {} files in {:?}", handle_slots.len(), request_time);

    // CPU processes all
    let start = std::time::Instant::now();
    let processed = pipeline.process_requests();
    let process_time = start.elapsed();

    println!("Processed {} requests in {:?}", processed, process_time);

    // Verify all ready
    let mut ready = 0;
    for slot in &handle_slots {
        let handle = pipeline.read_handle(*slot);
        if handle.status == STATUS_READY {
            ready += 1;
        }
    }

    println!("Ready: {}/{}", ready, handle_slots.len());
    assert_eq!(ready, 100);
}
