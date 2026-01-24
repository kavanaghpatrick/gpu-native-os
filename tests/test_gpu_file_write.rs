// Test GPU-Direct File Writing via mmap
//
// THE GPU IS THE COMPUTER - GPU edits should persist to disk automatically.

use metal::*;
use rust_experiment::gpu_os::mmap_buffer::WritableMmapBuffer;
use std::fs;
use tempfile::NamedTempFile;

const WRITE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// GPU writes data to mmap'd buffer
kernel void write_pattern(
    device uchar* data [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    // Write a recognizable pattern: tid % 256
    data[tid] = uchar(tid % 256);
}

// GPU modifies existing data
kernel void increment_all(
    device uchar* data [[buffer(0)]],
    constant uint& size [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    data[tid] = data[tid] + 1;
}

// GPU writes text
kernel void write_message(
    device uchar* data [[buffer(0)]],
    constant uchar* message [[buffer(1)]],
    constant uint& msg_len [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= msg_len) return;
    data[tid] = message[tid];
}
"#;

fn create_pipeline(device: &Device, name: &str) -> ComputePipelineState {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(WRITE_SHADER, &options)
        .expect("Shader compile failed");
    let function = library.get_function(name, None).expect("Function not found");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Pipeline failed")
}

#[test]
fn test_gpu_write_to_file() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          GPU-Direct File Writing Test                                          ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Create a new file for GPU to write to
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_path_buf();
    drop(temp_file); // Close so we can recreate with size

    let file_size: u32 = 4096;

    // Create writable mmap buffer
    let buffer = WritableMmapBuffer::create(&device, &path, file_size as usize)
        .expect("Failed to create writable mmap buffer");

    println!("\n  Created file: {} ({} bytes)", path.display(), file_size);

    // GPU writes a pattern
    let pipeline = create_pipeline(&device, "write_pattern");
    let size_buf = device.new_buffer_with_data(
        &file_size as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(buffer.metal_buffer()), 0);
    enc.set_buffer(1, Some(&size_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(file_size as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    println!("  GPU wrote pattern to buffer");

    // Sync to disk
    buffer.sync().expect("Sync failed");
    println!("  Synced to disk");

    // Drop buffer to ensure file is closed
    drop(buffer);

    // Read file back with standard IO (CPU) and verify
    let data = fs::read(&path).expect("Failed to read file");

    let mut errors = 0;
    for (i, &byte) in data.iter().enumerate().take(file_size as usize) {
        let expected = (i % 256) as u8;
        if byte != expected {
            errors += 1;
            if errors <= 3 {
                println!("  ERROR at {}: expected {}, got {}", i, expected, byte);
            }
        }
    }

    if errors == 0 {
        println!("  ✓ Verified: All {} bytes match expected pattern", file_size);
    } else {
        println!("  ✗ Found {} errors", errors);
    }

    // Clean up
    let _ = fs::remove_file(&path);

    assert_eq!(errors, 0, "GPU write verification failed");
}

#[test]
fn test_gpu_edit_existing_file() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          GPU Edit Existing File Test                                           ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    // Create file with initial data
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_path_buf();

    // Write initial data: all zeros
    let initial_data = vec![0u8; 4096];
    fs::write(&path, &initial_data).expect("Failed to write initial data");

    println!("\n  Created file with {} zeros", initial_data.len());

    // Open for GPU editing
    let buffer = WritableMmapBuffer::open(&device, &path)
        .expect("Failed to open writable mmap buffer");

    // GPU increments all bytes
    let pipeline = create_pipeline(&device, "increment_all");
    let size: u32 = 4096;
    let size_buf = device.new_buffer_with_data(
        &size as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(buffer.metal_buffer()), 0);
    enc.set_buffer(1, Some(&size_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(size as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    println!("  GPU incremented all bytes");

    buffer.sync().expect("Sync failed");
    drop(buffer);

    // Verify: all bytes should now be 1
    let data = fs::read(&path).expect("Failed to read file");
    let all_ones = data.iter().take(4096).all(|&b| b == 1);

    if all_ones {
        println!("  ✓ Verified: All bytes are now 1");
    } else {
        let wrong_count = data.iter().take(4096).filter(|&&b| b != 1).count();
        println!("  ✗ {} bytes are not 1", wrong_count);
    }

    assert!(all_ones, "GPU edit verification failed");
}

#[test]
fn test_gpu_write_text_message() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          GPU Write Text to File                                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_path_buf();
    drop(temp_file);

    // Create file
    let buffer = WritableMmapBuffer::create(&device, &path, 4096)
        .expect("Failed to create writable mmap buffer");

    // GPU writes a message
    let message = b"Hello from GPU! THE GPU IS THE COMPUTER.\n";
    let msg_len = message.len() as u32;

    let msg_buf = device.new_buffer_with_data(
        message.as_ptr() as *const _,
        message.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let len_buf = device.new_buffer_with_data(
        &msg_len as *const _ as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let pipeline = create_pipeline(&device, "write_message");

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(buffer.metal_buffer()), 0);
    enc.set_buffer(1, Some(&msg_buf), 0);
    enc.set_buffer(2, Some(&len_buf), 0);
    enc.dispatch_threads(
        MTLSize::new(msg_len as u64, 1, 1),
        MTLSize::new(64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    buffer.sync().expect("Sync failed");
    drop(buffer);

    // Read and verify
    let data = fs::read(&path).expect("Failed to read file");
    let written = &data[..message.len()];

    println!("\n  GPU wrote: {:?}", String::from_utf8_lossy(written));

    if written == message {
        println!("  ✓ Message verified!");
    } else {
        println!("  ✗ Message mismatch");
    }

    let _ = fs::remove_file(&path);

    assert_eq!(written, message, "Message verification failed");
}

#[test]
fn test_full_file_lifecycle() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════════╗");
    println!("║          Full File Lifecycle: Create → Edit → Save → Reopen                    ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════╝");

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let path = temp_dir.path().join("gpu_file.bin");

    // Step 1: CREATE new file with GPU
    println!("\n  Step 1: CREATE");
    {
        let buffer = WritableMmapBuffer::create(&device, &path, 1024)
            .expect("Failed to create file");

        let pipeline = create_pipeline(&device, "write_pattern");
        let size: u32 = 1024;
        let size_buf = device.new_buffer_with_data(
            &size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(buffer.metal_buffer()), 0);
        enc.set_buffer(1, Some(&size_buf), 0);
        enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        buffer.sync().expect("Sync failed");
        println!("    Created and wrote initial pattern");
    }

    // Step 2: REOPEN and EDIT
    println!("  Step 2: EDIT");
    {
        let buffer = WritableMmapBuffer::open(&device, &path)
            .expect("Failed to reopen file");

        let pipeline = create_pipeline(&device, "increment_all");
        let size: u32 = 1024;
        let size_buf = device.new_buffer_with_data(
            &size as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(buffer.metal_buffer()), 0);
        enc.set_buffer(1, Some(&size_buf), 0);
        enc.dispatch_threads(MTLSize::new(1024, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        buffer.sync().expect("Sync failed");
        println!("    Incremented all bytes");
    }

    // Step 3: VERIFY with CPU
    println!("  Step 3: VERIFY");
    let data = fs::read(&path).expect("Failed to read file");

    let mut errors = 0;
    for (i, &byte) in data.iter().enumerate().take(1024) {
        let expected = ((i % 256) + 1) as u8; // Original pattern + 1
        if byte != expected {
            errors += 1;
        }
    }

    if errors == 0 {
        println!("    ✓ All 1024 bytes verified (pattern + 1)");
    } else {
        println!("    ✗ {} errors found", errors);
    }

    println!("\n  Full lifecycle complete: CREATE → EDIT → SAVE → VERIFY");

    assert_eq!(errors, 0, "Lifecycle test failed");
}
