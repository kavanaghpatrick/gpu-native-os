// Issue #82: Zero-Copy mmap Buffer Tests
//
// Verifies that MmapBuffer provides zero-copy file-to-GPU access:
// - File → mmap → newBufferWithBytesNoCopy → GPU (same memory!)
//
// THE GPU IS THE COMPUTER. No CPU copies allowed.

use rust_experiment::gpu_os::mmap_buffer::{MmapBuffer, MmapError, PAGE_SIZE, align_to_page};
use std::fs::File;
use std::io::Write;

fn get_device() -> metal::Device {
    metal::Device::system_default().expect("No Metal device found")
}

// =============================================================================
// Basic Creation Tests
// =============================================================================

#[test]
fn test_mmap_buffer_from_file() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    // Create test file with known content
    let mut file = File::create(&path).unwrap();
    file.write_all(&[0u8; 8192]).unwrap(); // 2 pages
    drop(file);

    // Create mmap buffer
    let buffer = MmapBuffer::from_file(&device, &path).unwrap();

    // Verify buffer exists and has correct size
    assert_eq!(buffer.file_size(), 8192);
    assert_eq!(buffer.aligned_size(), 8192); // Already page-aligned
    assert_eq!(buffer.metal_buffer().length(), 8192);
}

#[test]
fn test_mmap_buffer_alignment() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    // Create file with non-page-aligned size
    let mut file = File::create(&path).unwrap();
    file.write_all(&[0u8; 5000]).unwrap(); // Not page-aligned
    drop(file);

    let buffer = MmapBuffer::from_file(&device, &path).unwrap();

    // File size should be original
    assert_eq!(buffer.file_size(), 5000);

    // Buffer should be rounded up to page boundary
    assert_eq!(buffer.aligned_size(), PAGE_SIZE * 2); // 8192
    assert_eq!(buffer.metal_buffer().length() as usize, PAGE_SIZE * 2);
    assert_eq!(buffer.metal_buffer().length() % PAGE_SIZE as u64, 0);
}

#[test]
fn test_mmap_buffer_empty_file() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.bin");

    File::create(&path).unwrap(); // Empty file

    let result = MmapBuffer::from_file(&device, &path);
    assert!(matches!(result, Err(MmapError::EmptyFile)));
}

#[test]
fn test_mmap_buffer_nonexistent_file() {
    let device = get_device();

    let result = MmapBuffer::from_file(&device, "/nonexistent/path/to/file.bin");
    assert!(matches!(result, Err(MmapError::IoError(_))));
}

#[test]
fn test_mmap_buffer_from_bytes() {
    let device = get_device();

    let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let buffer = MmapBuffer::from_bytes(&device, &data).unwrap();

    assert_eq!(buffer.file_size(), 1000);
    assert_eq!(buffer.aligned_size(), PAGE_SIZE);

    // Verify content
    unsafe {
        let ptr = buffer.as_ptr();
        for i in 0..1000 {
            assert_eq!(*ptr.add(i), (i % 256) as u8);
        }
    }
}

// =============================================================================
// Storage Mode Tests
// =============================================================================

#[test]
fn test_mmap_buffer_storage_mode() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    let mut file = File::create(&path).unwrap();
    file.write_all(&[0u8; 4096]).unwrap();
    drop(file);

    let buffer = MmapBuffer::from_file(&device, &path).unwrap();

    // Must be shared mode for unified memory
    assert_eq!(
        buffer.metal_buffer().storage_mode(),
        metal::MTLStorageMode::Shared
    );
}

// =============================================================================
// Data Integrity Tests
// =============================================================================

#[test]
fn test_mmap_buffer_data_integrity() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    // Create file with known pattern
    let pattern: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let mut file = File::create(&path).unwrap();
    file.write_all(&pattern).unwrap();
    drop(file);

    let buffer = MmapBuffer::from_file(&device, &path).unwrap();

    // Verify data through Metal buffer contents (CPU readback for test)
    let ptr = buffer.metal_buffer().contents() as *const u8;
    unsafe {
        for i in 0..4096 {
            assert_eq!(
                *ptr.add(i),
                (i % 256) as u8,
                "Mismatch at byte {}", i
            );
        }
    }
}

#[test]
fn test_mmap_buffer_large_file() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.bin");

    // Create 1MB file
    let size = 1024 * 1024;
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    std::fs::write(&path, &data).unwrap();

    let buffer = MmapBuffer::from_file(&device, &path).unwrap();

    assert_eq!(buffer.file_size(), size);
    assert_eq!(buffer.aligned_size(), size); // Already aligned

    // Spot check some values
    let ptr = buffer.metal_buffer().contents() as *const u8;
    unsafe {
        assert_eq!(*ptr.add(0), 0);
        assert_eq!(*ptr.add(255), 255);
        assert_eq!(*ptr.add(256), 0);
        assert_eq!(*ptr.add(1000), (1000 % 256) as u8);
        assert_eq!(*ptr.add(size - 1), ((size - 1) % 256) as u8);
    }
}

// =============================================================================
// GPU Access Test
// =============================================================================

#[test]
fn test_mmap_buffer_gpu_compute() {
    use metal::*;

    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    // Create file with u32 integers
    let numbers: Vec<u32> = (0..1024).collect();
    let bytes: Vec<u8> = numbers.iter()
        .flat_map(|n| n.to_le_bytes())
        .collect();
    std::fs::write(&path, &bytes).unwrap();

    let input = MmapBuffer::from_file(&device, &path).unwrap();

    // Create output buffer
    let output = device.new_buffer(
        4096,
        MTLResourceOptions::StorageModeShared,
    );

    // Simple shader that doubles each value
    let shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void double_values(
            device const uint* input [[buffer(0)]],
            device uint* output [[buffer(1)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid < 1024) {
                output[tid] = input[tid] * 2;
            }
        }
    "#;

    // Compile shader
    let options = CompileOptions::new();
    let library = device.new_library_with_source(shader, &options)
        .expect("Failed to compile shader");
    let function = library.get_function("double_values", None)
        .expect("Failed to get function");
    let pipeline = device.new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline");

    // Run compute
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), 0);
    enc.set_buffer(1, Some(&output), 0);
    enc.dispatch_threads(
        MTLSize::new(1024, 1, 1),
        MTLSize::new(64, 1, 1),
    );
    enc.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // Verify GPU computed correct results from mmap'd data
    let result_ptr = output.contents() as *const u32;
    unsafe {
        for i in 0..1024 {
            assert_eq!(
                *result_ptr.add(i),
                (i * 2) as u32,
                "GPU computed wrong value at index {}", i
            );
        }
    }
}

// =============================================================================
// madvise Tests
// =============================================================================

#[test]
fn test_mmap_buffer_advise_methods() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    let mut file = File::create(&path).unwrap();
    file.write_all(&[0u8; 8192]).unwrap();
    drop(file);

    let buffer = MmapBuffer::from_file(&device, &path).unwrap();

    // These should not panic
    buffer.advise_sequential();
    buffer.advise_willneed();
    buffer.advise_random();

    // Verify buffer still works after advise calls
    assert_eq!(buffer.file_size(), 8192);
}

// =============================================================================
// Helper Function Tests
// =============================================================================

#[test]
fn test_align_to_page() {
    assert_eq!(align_to_page(0), 0);
    assert_eq!(align_to_page(1), PAGE_SIZE);
    assert_eq!(align_to_page(PAGE_SIZE - 1), PAGE_SIZE);
    assert_eq!(align_to_page(PAGE_SIZE), PAGE_SIZE);
    assert_eq!(align_to_page(PAGE_SIZE + 1), PAGE_SIZE * 2);
    assert_eq!(align_to_page(PAGE_SIZE * 2), PAGE_SIZE * 2);
    assert_eq!(align_to_page(5000), PAGE_SIZE * 2); // 8192
}

// =============================================================================
// Performance Benchmark
// =============================================================================

#[test]
fn benchmark_mmap_vs_read() {
    use std::time::Instant;

    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("benchmark.bin");

    // Create 10MB test file
    let size = 10 * 1024 * 1024;
    let data = vec![0u8; size];
    std::fs::write(&path, &data).unwrap();

    // Benchmark traditional read + copy
    let start_read = Instant::now();
    let read_data = std::fs::read(&path).unwrap();
    let _buffer_read = device.new_buffer_with_data(
        read_data.as_ptr() as *const _,
        read_data.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let read_time = start_read.elapsed();

    // Benchmark zero-copy mmap
    let start_mmap = Instant::now();
    let buffer_mmap = MmapBuffer::from_file(&device, &path).unwrap();
    buffer_mmap.advise_willneed(); // Trigger prefetch
    let mmap_time = start_mmap.elapsed();

    println!("\n=== Issue #82: Zero-Copy mmap Benchmark (10MB file) ===");
    println!("Traditional (read + copy): {:?}", read_time);
    println!("Zero-copy (mmap):          {:?}", mmap_time);

    if mmap_time < read_time {
        let speedup = read_time.as_nanos() as f64 / mmap_time.as_nanos() as f64;
        println!("mmap is {:.1}x faster", speedup);
    }

    // mmap should generally be faster (no data copy)
    // But we don't assert this strictly since it depends on disk cache state
}

// =============================================================================
// Drop/Cleanup Test
// =============================================================================

#[test]
fn test_mmap_buffer_drop() {
    let device = get_device();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");

    let mut file = File::create(&path).unwrap();
    file.write_all(&[0u8; 4096]).unwrap();
    drop(file);

    // Create and immediately drop
    {
        let _buffer = MmapBuffer::from_file(&device, &path).unwrap();
        // Buffer dropped here - munmap should be called
    }

    // File should still exist and be readable
    assert!(path.exists());
    let contents = std::fs::read(&path).unwrap();
    assert_eq!(contents.len(), 4096);
}
