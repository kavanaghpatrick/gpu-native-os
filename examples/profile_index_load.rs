// Profile GPU-direct index loading to identify optimization opportunities
//
// Breaks down each step to find where overhead comes from

use metal::*;
use std::time::Instant;
use std::path::Path;

fn main() {
    let device = Device::system_default().expect("No Metal device");

    // Create test index if needed
    let index_path = "test_index.bin";
    if !Path::new(index_path).exists() {
        create_test_index(100_000);
    }

    let file_size = std::fs::metadata(index_path).unwrap().len() as usize;
    println!("=== GPU-Direct Index Load Profiling ===\n");
    println!("File: {} ({:.2} MB)\n", index_path, file_size as f64 / 1024.0 / 1024.0);

    // Warm up
    let _ = std::fs::read(index_path);

    // Profile mmap for comparison
    println!("--- mmap baseline ---");
    let mmap_time = profile_mmap(&device, index_path);
    println!("Total: {:?}\n", mmap_time);

    // Profile GPU-direct step by step
    println!("--- GPU-direct breakdown ---");
    profile_gpu_direct_detailed(&device, index_path, file_size);

    // Test optimizations
    println!("\n--- Optimization experiments ---");
    test_optimizations(&device, index_path, file_size);
}

fn profile_mmap(device: &Device, path: &str) -> std::time::Duration {
    let start = Instant::now();
    let _index = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(device, path).unwrap();
    start.elapsed()
}

fn profile_gpu_direct_detailed(device: &Device, path: &str, file_size: usize) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    let mut timings = Vec::new();

    // Step 1: Create IO Queue
    let t0 = Instant::now();
    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    timings.push(("Create IO Queue", t0.elapsed()));

    // Step 2: Open file handle
    let t1 = Instant::now();
    let file_handle = GpuIOFileHandle::open(device, path).unwrap();
    timings.push(("Open file handle", t1.elapsed()));

    // Step 3: Allocate GPU buffer
    let t2 = Instant::now();
    let aligned_size = (file_size + 4095) & !4095;
    let buffer = device.new_buffer(aligned_size as u64, MTLResourceOptions::StorageModeShared);
    timings.push(("Allocate GPU buffer", t2.elapsed()));

    // Step 4: Create command buffer
    let t3 = Instant::now();
    let cmd_buffer = io_queue.command_buffer().unwrap();
    timings.push(("Create cmd buffer", t3.elapsed()));

    // Step 5: Queue load command
    let t4 = Instant::now();
    cmd_buffer.load_buffer(&buffer, 0, file_size as u64, &file_handle, 0);
    timings.push(("Queue load cmd", t4.elapsed()));

    // Step 6: Commit
    let t5 = Instant::now();
    cmd_buffer.commit();
    timings.push(("Commit", t5.elapsed()));

    // Step 7: Wait for completion
    let t6 = Instant::now();
    cmd_buffer.wait_until_completed();
    timings.push(("Wait completion", t6.elapsed()));

    // Step 8: Validate header (read from GPU buffer)
    let t7 = Instant::now();
    let _header = unsafe { *(buffer.contents() as *const u32) };
    timings.push(("Read header", t7.elapsed()));

    // Print breakdown
    let total: std::time::Duration = timings.iter().map(|(_, d)| *d).sum();
    println!("{:<20} {:>12} {:>8}", "Step", "Time", "% Total");
    println!("{}", "-".repeat(42));
    for (name, duration) in &timings {
        let pct = duration.as_secs_f64() / total.as_secs_f64() * 100.0;
        println!("{:<20} {:>12?} {:>7.1}%", name, duration, pct);
    }
    println!("{}", "-".repeat(42));
    println!("{:<20} {:>12?}", "TOTAL", total);
}

fn test_optimizations(device: &Device, path: &str, file_size: usize) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    // Test 1: Reuse IO queue (amortize creation cost)
    println!("\n1. Reusing IO queue across multiple loads:");
    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();

    for i in 0..3 {
        let start = Instant::now();

        let file_handle = GpuIOFileHandle::open(device, path).unwrap();
        let aligned_size = (file_size + 4095) & !4095;
        let buffer = device.new_buffer(aligned_size as u64, MTLResourceOptions::StorageModeShared);
        let cmd_buffer = io_queue.command_buffer().unwrap();
        cmd_buffer.load_buffer(&buffer, 0, file_size as u64, &file_handle, 0);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        println!("   Load {}: {:?}", i + 1, start.elapsed());
    }

    // Test 2: Pre-allocated buffer pool
    println!("\n2. Pre-allocated buffer (skip allocation):");
    let pre_alloc_buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );

    for i in 0..3 {
        let start = Instant::now();

        let file_handle = GpuIOFileHandle::open(device, path).unwrap();
        let cmd_buffer = io_queue.command_buffer().unwrap();
        cmd_buffer.load_buffer(&pre_alloc_buffer, 0, file_size as u64, &file_handle, 0);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        println!("   Load {}: {:?}", i + 1, start.elapsed());
    }

    // Test 3: Different queue types
    println!("\n3. Queue type comparison:");
    for (name, queue_type) in [
        ("Concurrent", IOQueueType::Concurrent),
        ("Serial", IOQueueType::Serial),
    ] {
        let queue = GpuIOQueue::new(device, IOPriority::High, queue_type).unwrap();
        let file_handle = GpuIOFileHandle::open(device, path).unwrap();

        let start = Instant::now();
        let cmd_buffer = queue.command_buffer().unwrap();
        cmd_buffer.load_buffer(&pre_alloc_buffer, 0, file_size as u64, &file_handle, 0);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        println!("   {}: {:?}", name, start.elapsed());
    }

    // Test 4: Different priorities
    println!("\n4. Priority comparison:");
    for (name, priority) in [
        ("High", IOPriority::High),
        ("Normal", IOPriority::Normal),
        ("Low", IOPriority::Low),
    ] {
        let queue = GpuIOQueue::new(device, priority, IOQueueType::Concurrent).unwrap();
        let file_handle = GpuIOFileHandle::open(device, path).unwrap();

        let start = Instant::now();
        let cmd_buffer = queue.command_buffer().unwrap();
        cmd_buffer.load_buffer(&pre_alloc_buffer, 0, file_size as u64, &file_handle, 0);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        println!("   {}: {:?}", name, start.elapsed());
    }

    // Test 5: Async load (measure just the wait time)
    println!("\n5. Async load (commit vs wait breakdown):");
    let file_handle = GpuIOFileHandle::open(device, path).unwrap();
    let cmd_buffer = io_queue.command_buffer().unwrap();
    cmd_buffer.load_buffer(&pre_alloc_buffer, 0, file_size as u64, &file_handle, 0);

    let t_commit = Instant::now();
    cmd_buffer.commit();
    let commit_time = t_commit.elapsed();

    let t_wait = Instant::now();
    cmd_buffer.wait_until_completed();
    let wait_time = t_wait.elapsed();

    println!("   Commit: {:?}", commit_time);
    println!("   Wait:   {:?}", wait_time);
    println!("   (Wait is actual I/O time)");
}

fn create_test_index(num_entries: u32) {
    use std::io::Write;

    let header_size = 4096;
    let entry_size = 256;

    let mut file = std::fs::File::create("test_index.bin").unwrap();

    // Header
    file.write_all(&0x47505549u32.to_le_bytes()).unwrap(); // magic
    file.write_all(&1u32.to_le_bytes()).unwrap(); // version
    file.write_all(&num_entries.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap(); // flags
    file.write_all(&0u64.to_le_bytes()).unwrap(); // build_time
    file.write_all(&vec![0u8; header_size - 24]).unwrap();

    // Entries
    for i in 0..num_entries {
        let path = format!("/test/path/to/file_{:06}.txt", i);
        let mut entry = vec![0u8; entry_size];
        let path_bytes = path.as_bytes();
        let len = path_bytes.len().min(224);
        entry[..len].copy_from_slice(&path_bytes[..len]);
        entry[224] = len as u8;
        file.write_all(&entry).unwrap();
    }

    println!("Created test_index.bin with {} entries", num_entries);
}
