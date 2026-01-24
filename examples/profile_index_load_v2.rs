// Profile GPU-direct index loading - V2 with algorithmic optimizations
//
// Key insight: mmap is "lazy" - it returns instantly, actual I/O happens on access
// GPU-direct waits for ALL data. Can we make it lazy too?

use metal::*;
use std::time::Instant;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn main() {
    let device = Device::system_default().expect("No Metal device");

    // Create test index if needed
    let index_path = "test_index.bin";
    if !Path::new(index_path).exists() {
        create_test_index(100_000);
    }

    let file_size = std::fs::metadata(index_path).unwrap().len() as usize;
    println!("=== GPU-Direct Index Load - Algorithmic Optimizations ===\n");
    println!("File: {} ({:.2} MB)\n", index_path, file_size as f64 / 1024.0 / 1024.0);

    // Warm up
    let _ = std::fs::read(index_path);

    // Baseline comparison
    println!("--- Baselines ---");

    let t = Instant::now();
    let _mmap = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(&device, index_path).unwrap();
    println!("mmap:                {:?}", t.elapsed());

    let t = Instant::now();
    let _gpu = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load_gpu_direct(&device, index_path).unwrap();
    println!("GPU-direct (sync):   {:?}", t.elapsed());

    // Optimization 1: Async load - return handle immediately, wait later
    println!("\n--- Opt 1: Async Load (measure time-to-usable) ---");
    test_async_load(&device, index_path, file_size);

    // Optimization 2: Header-first load - load header, validate, then rest
    println!("\n--- Opt 2: Header-First Load ---");
    test_header_first(&device, index_path, file_size);

    // Optimization 3: Chunked streaming - first chunk available sooner
    println!("\n--- Opt 3: Chunked Streaming ---");
    test_chunked_streaming(&device, index_path, file_size);

    // Optimization 4: Parallel with compute - process while loading
    println!("\n--- Opt 4: Overlap Load + Compute ---");
    test_overlap_compute(&device, index_path, file_size);

    // Optimization 5: Hybrid approach - right tool for the job
    println!("\n--- Opt 5: Hybrid (mmap for hot, GPU-direct for cold) ---");
    test_hybrid(&device, index_path, file_size);
}

fn test_async_load(device: &Device, path: &str, file_size: usize) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    // Pre-create reusable resources
    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    let buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );

    // Measure time to "start" load (what caller experiences)
    let t_start = Instant::now();

    let file_handle = GpuIOFileHandle::open(device, path).unwrap();
    let cmd_buffer = io_queue.command_buffer().unwrap();
    cmd_buffer.load_buffer(&buffer, 0, file_size as u64, &file_handle, 0);
    cmd_buffer.commit();

    let time_to_start = t_start.elapsed();

    // Now wait for completion
    let t_wait = Instant::now();
    cmd_buffer.wait_until_completed();
    let time_to_complete = t_wait.elapsed();

    println!("  Time to start (async return): {:?}", time_to_start);
    println!("  Time waiting (I/O):           {:?}", time_to_complete);
    println!("  → CPU free during I/O: {:?}", time_to_complete);
}

fn test_header_first(device: &Device, path: &str, file_size: usize) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    let buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );

    let t_start = Instant::now();

    // Load just header first (4KB)
    let file_handle = GpuIOFileHandle::open(device, path).unwrap();
    let cmd_buffer = io_queue.command_buffer().unwrap();
    cmd_buffer.load_buffer(&buffer, 0, 4096, &file_handle, 0);
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    let header_time = t_start.elapsed();

    // Validate header
    let magic = unsafe { *(buffer.contents() as *const u32) };
    let entry_count = unsafe { *((buffer.contents() as *const u32).add(2)) };

    if magic != 0x47505549 {
        println!("  Invalid header!");
        return;
    }

    // Now load rest
    let t_rest = Instant::now();
    let file_handle2 = GpuIOFileHandle::open(device, path).unwrap();
    let cmd_buffer2 = io_queue.command_buffer().unwrap();
    cmd_buffer2.load_buffer(&buffer, 4096, (file_size - 4096) as u64, &file_handle2, 4096);
    cmd_buffer2.commit();
    cmd_buffer2.wait_until_completed();

    let rest_time = t_rest.elapsed();

    println!("  Header load + validate: {:?} ({} entries)", header_time, entry_count);
    println!("  Rest of data:           {:?}", rest_time);
    println!("  Total:                  {:?}", header_time + rest_time);
    println!("  → Can start using metadata after {:?}", header_time);
}

fn test_chunked_streaming(device: &Device, path: &str, file_size: usize) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    let buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );

    // Load in 1MB chunks
    let chunk_size = 1024 * 1024;
    let num_chunks = (file_size + chunk_size - 1) / chunk_size;

    let t_start = Instant::now();
    let mut first_chunk_time = None;

    for i in 0..num_chunks {
        let offset = i * chunk_size;
        let size = std::cmp::min(chunk_size, file_size - offset);

        let file_handle = GpuIOFileHandle::open(device, path).unwrap();
        let cmd_buffer = io_queue.command_buffer().unwrap();
        cmd_buffer.load_buffer(&buffer, offset as u64, size as u64, &file_handle, offset as u64);
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        if first_chunk_time.is_none() {
            first_chunk_time = Some(t_start.elapsed());
        }
    }

    let total_time = t_start.elapsed();

    println!("  Chunk size: 1MB, {} chunks", num_chunks);
    println!("  First chunk ready: {:?}", first_chunk_time.unwrap());
    println!("  All chunks done:   {:?}", total_time);
    println!("  → Can start processing after {:?}", first_chunk_time.unwrap());
}

fn test_overlap_compute(device: &Device, path: &str, file_size: usize) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    let buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );

    let t_start = Instant::now();

    // Start I/O (don't wait)
    let file_handle = GpuIOFileHandle::open(device, path).unwrap();
    let cmd_buffer = io_queue.command_buffer().unwrap();
    cmd_buffer.load_buffer(&buffer, 0, file_size as u64, &file_handle, 0);
    cmd_buffer.commit();

    let after_commit = t_start.elapsed();

    // Simulate CPU work while GPU loads (e.g., prepare query, UI, etc.)
    let cpu_work_start = Instant::now();
    let mut sum: u64 = 0;
    for i in 0..1_000_000 {
        sum = sum.wrapping_add(i);
    }
    std::hint::black_box(sum);
    let cpu_work_time = cpu_work_start.elapsed();

    // Now wait for I/O
    let wait_start = Instant::now();
    cmd_buffer.wait_until_completed();
    let actual_wait = wait_start.elapsed();

    let total = t_start.elapsed();

    println!("  I/O started at:     {:?}", after_commit);
    println!("  CPU work done:      {:?} (overlapped)", cpu_work_time);
    println!("  Additional wait:    {:?}", actual_wait);
    println!("  Total wall time:    {:?}", total);

    if actual_wait < std::time::Duration::from_micros(100) {
        println!("  → I/O completed DURING CPU work! Perfect overlap.");
    } else {
        println!("  → Saved ~{:?} by overlapping", cpu_work_time.saturating_sub(actual_wait));
    }
}

fn test_hybrid(device: &Device, path: &str, file_size: usize) {
    // Strategy: Use mmap for small/hot files, GPU-direct for large/cold

    let threshold = 10 * 1024 * 1024; // 10MB threshold

    println!("  Strategy: mmap if < 10MB or hot, GPU-direct if > 10MB and cold");
    println!("  File size: {:.2} MB", file_size as f64 / 1024.0 / 1024.0);

    if file_size < threshold {
        let t = Instant::now();
        let _index = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(device, path).unwrap();
        println!("  Used mmap (small file): {:?}", t.elapsed());
    } else {
        // For large files, check if it's likely in cache
        // Heuristic: if mmap is very fast, file is hot
        let t = Instant::now();
        let _probe = std::fs::File::open(path).unwrap();
        let probe_time = t.elapsed();

        if probe_time < std::time::Duration::from_micros(100) {
            // File is hot, use mmap
            let t = Instant::now();
            let _index = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(device, path).unwrap();
            println!("  Used mmap (hot cache detected): {:?}", t.elapsed());
        } else {
            // File is cold, use GPU-direct
            let t = Instant::now();
            let _index = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load_gpu_direct(device, path).unwrap();
            println!("  Used GPU-direct (cold cache): {:?}", t.elapsed());
        }
    }
}

fn create_test_index(num_entries: u32) {
    use std::io::Write;

    let header_size = 4096;
    let entry_size = 256;

    let mut file = std::fs::File::create("test_index.bin").unwrap();

    file.write_all(&0x47505549u32.to_le_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&num_entries.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    file.write_all(&vec![0u8; header_size - 24]).unwrap();

    for i in 0..num_entries {
        let path = format!("/test/path/to/file_{:06}.txt", i);
        let mut entry = vec![0u8; entry_size];
        let path_bytes = path.as_bytes();
        let len = path_bytes.len().min(224);
        entry[..len].copy_from_slice(&path_bytes[..len]);
        entry[224] = len as u8;
        file.write_all(&entry).unwrap();
    }

    println!("Created test_index.bin with {} entries\n", num_entries);
}
