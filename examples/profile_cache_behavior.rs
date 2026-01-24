// Investigation: Does GPU-direct read from page cache or disk?
//
// Key question: If file is in RAM, does MTLIOCommandQueue use it?

use metal::*;
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("No Metal device");

    println!("=== Does GPU-Direct Read from Page Cache? ===\n");

    // Create test file
    let test_path = "test_cache_10mb.bin";
    let file_size = 10 * 1024 * 1024;
    create_test_file(test_path, file_size);

    println!("--- Test 1: Measure actual disk vs cache speed ---\n");

    // NVMe SSD speed is typically 3-7 GB/s
    // RAM speed is typically 50-200 GB/s (unified memory even faster)
    // If GPU-direct reads from cache, we should see RAM-like speeds

    // Cold: Read file we haven't touched
    println!("Creating fresh 50MB file (definitely cold)...");
    create_test_file("cold_file.bin", 50 * 1024 * 1024);

    // Measure GPU-direct on cold file
    let cold_time = measure_gpu_direct(&device, "cold_file.bin", 50 * 1024 * 1024);
    let cold_throughput = (50.0 * 1024.0 * 1024.0) / cold_time.as_secs_f64() / 1e9;
    println!("Cold file GPU-direct: {:?} ({:.1} GB/s)\n", cold_time, cold_throughput);

    // Now warm it up
    println!("Warming cache (reading file into RAM)...");
    let _ = std::fs::read("cold_file.bin");

    // Measure again (should be in cache now)
    let hot_time = measure_gpu_direct(&device, "cold_file.bin", 50 * 1024 * 1024);
    let hot_throughput = (50.0 * 1024.0 * 1024.0) / hot_time.as_secs_f64() / 1e9;
    println!("Hot file GPU-direct:  {:?} ({:.1} GB/s)\n", hot_time, hot_throughput);

    println!("--- Analysis ---\n");
    println!("If both show ~10 GB/s: GPU-direct reads from page cache (memory copy)");
    println!("If cold is slower:     GPU-direct hits disk for cold, cache for hot");
    println!("Actual NVMe speed:     ~3-7 GB/s");
    println!("Memory copy speed:     ~10-50 GB/s\n");

    if cold_throughput > 8.0 && hot_throughput > 8.0 {
        println!("RESULT: Both fast! GPU-direct IS reading from page cache.");
        println!("        The 10 GB/s is memory-to-memory copy speed.");
        println!("        The bottleneck is the COPY, not disk I/O.\n");
    } else if hot_throughput > cold_throughput * 1.5 {
        println!("RESULT: Hot is faster than cold.");
        println!("        GPU-direct uses page cache when available.\n");
    }

    println!("--- Test 2: The real question - can we avoid the copy? ---\n");

    // mmap + newBufferWithBytesNoCopy avoids copy
    // GPU-direct always copies

    // Measure mmap (no copy)
    let mmap_time = measure_mmap(&device, "cold_file.bin");
    let mmap_throughput = (50.0 * 1024.0 * 1024.0) / mmap_time.as_secs_f64() / 1e9;
    println!("mmap (no copy):       {:?} ({:.1} GB/s)", mmap_time, mmap_throughput);
    println!("GPU-direct (copy):    {:?} ({:.1} GB/s)\n", hot_time, hot_throughput);

    println!("--- The Fundamental Difference ---\n");
    println!("mmap + newBufferWithBytesNoCopy:");
    println!("  Page cache page → GPU sees SAME physical page");
    println!("  Zero copy. Instant.\n");

    println!("MTLIOCommandQueue:");
    println!("  Page cache page → Copy to NEW buffer → GPU sees new buffer");
    println!("  Always copies. ~10 GB/s.\n");

    println!("--- Can GPU-direct avoid copying? ---\n");
    println!("NO. MTLIOCommandQueue is designed for disk-to-buffer streaming.");
    println!("It always allocates destination buffer and copies.\n");

    println!("To get mmap-like behavior, we'd need:");
    println!("  1. Check if file is in page cache");
    println!("  2. If yes: use mmap (zero copy)");
    println!("  3. If no: use GPU-direct (async, CPU-free)\n");

    println!("--- Test 3: Hybrid approach ---\n");
    test_hybrid(&device, "cold_file.bin", 50 * 1024 * 1024);

    // Cleanup
    let _ = std::fs::remove_file(test_path);
    let _ = std::fs::remove_file("cold_file.bin");
}

fn measure_gpu_direct(device: &Device, path: &str, file_size: usize) -> std::time::Duration {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    let buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );
    let file_handle = GpuIOFileHandle::open(device, path).unwrap();

    let start = Instant::now();
    let cmd = io_queue.command_buffer().unwrap();
    cmd.load_buffer(&buffer, 0, file_size as u64, &file_handle, 0);
    cmd.commit();
    cmd.wait_until_completed();
    start.elapsed()
}

fn measure_mmap(device: &Device, path: &str) -> std::time::Duration {
    let start = Instant::now();
    let _ = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(device, path);
    start.elapsed()
}

fn test_hybrid(device: &Device, path: &str, file_size: usize) {
    // Smart hybrid: detect cache status and pick optimal method

    // Heuristic: measure time to read first 4KB
    // If very fast (<100µs), file is likely in cache → use mmap
    // If slow (>1ms), file is cold → use GPU-direct for CPU freedom

    let probe_start = Instant::now();
    let mut probe_buf = [0u8; 4096];
    let file = std::fs::File::open(path).unwrap();
    use std::io::Read;
    let _ = (&file).take(4096).read(&mut probe_buf);
    let probe_time = probe_start.elapsed();

    println!("Cache probe (4KB read): {:?}", probe_time);

    let start = Instant::now();
    if probe_time < std::time::Duration::from_micros(500) {
        // Hot cache - use mmap (zero copy)
        let _ = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(device, path);
        println!("Chose: mmap (cache hit detected)");
    } else {
        // Cold cache - use GPU-direct (CPU free during I/O)
        let _ = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load_gpu_direct(device, path);
        println!("Chose: GPU-direct (cache miss detected)");
    }
    println!("Hybrid load time: {:?}", start.elapsed());
}

fn create_test_file(path: &str, size: usize) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).unwrap();

    // Write valid index header
    file.write_all(&0x47505549u32.to_le_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&((size / 256) as u32).to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();

    // Fill rest with data
    let chunk = vec![0xABu8; 1024 * 1024];
    let mut written = 24;
    while written < size {
        let to_write = std::cmp::min(chunk.len(), size - written);
        file.write_all(&chunk[..to_write]).unwrap();
        written += to_write;
    }
}
