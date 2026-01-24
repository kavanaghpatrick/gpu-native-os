// Deep dive: Why is mmap faster than GPU-direct for hot cache?
//
// Key insight: They're doing fundamentally different things!

use metal::*;
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("No Metal device");

    // Create test files of various sizes
    println!("=== mmap vs GPU-direct: Understanding the I/O Difference ===\n");

    create_test_file("test_1mb.bin", 1024 * 1024);
    create_test_file("test_10mb.bin", 10 * 1024 * 1024);
    create_test_file("test_50mb.bin", 50 * 1024 * 1024);

    println!("\n--- What each method actually does ---\n");

    println!("mmap (newBufferWithBytesNoCopy):");
    println!("  1. mmap() syscall - maps file into virtual address space");
    println!("  2. newBufferWithBytesNoCopy - tells GPU 'this memory is yours'");
    println!("  3. NO DATA MOVEMENT for hot cache - pages already in RAM");
    println!("  4. First GPU access uses existing cached pages\n");

    println!("GPU-direct (MTLIOCommandQueue):");
    println!("  1. Create IO queue, open file handle (setup overhead)");
    println!("  2. Allocate fresh GPU buffer");
    println!("  3. Issue load command - ALWAYS reads from storage");
    println!("  4. Even if file is in page cache, data is COPIED to new buffer\n");

    // Test 1: Measure what mmap actually does
    println!("--- Test 1: What does mmap() actually do? ---\n");
    test_mmap_breakdown("test_10mb.bin");

    // Test 2: Measure page fault cost
    println!("\n--- Test 2: Page fault cost (first access after mmap) ---\n");
    test_page_fault_cost("test_10mb.bin");

    // Test 3: GPU-direct always copies
    println!("\n--- Test 3: GPU-direct always copies (even from cache) ---\n");
    test_gpu_direct_cache_behavior(&device, "test_10mb.bin");

    // Test 4: Cold cache comparison (purge and reload)
    println!("\n--- Test 4: Simulated cold cache (read different file first) ---\n");
    test_cold_cache_simulation(&device);

    // Clean up
    let _ = std::fs::remove_file("test_1mb.bin");
    let _ = std::fs::remove_file("test_10mb.bin");
    let _ = std::fs::remove_file("test_50mb.bin");
}

fn test_mmap_breakdown(path: &str) {
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    let file = File::open(path).unwrap();
    let file_size = file.metadata().unwrap().len() as usize;

    // Step 1: Just the mmap syscall
    let t1 = Instant::now();
    let mmap_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            file_size,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    let mmap_time = t1.elapsed();

    // Step 2: First byte access (triggers page fault for first page)
    let t2 = Instant::now();
    let _first_byte = unsafe { *(mmap_ptr as *const u8) };
    let first_access_time = t2.elapsed();

    // Step 3: Access all pages (trigger all page faults)
    let t3 = Instant::now();
    let mut sum: u64 = 0;
    for i in (0..file_size).step_by(4096) {
        sum = sum.wrapping_add(unsafe { *(mmap_ptr as *const u8).add(i) } as u64);
    }
    std::hint::black_box(sum);
    let all_pages_time = t3.elapsed();

    // Step 4: Access all pages again (no page faults, just memory read)
    let t4 = Instant::now();
    let mut sum2: u64 = 0;
    for i in (0..file_size).step_by(4096) {
        sum2 = sum2.wrapping_add(unsafe { *(mmap_ptr as *const u8).add(i) } as u64);
    }
    std::hint::black_box(sum2);
    let cached_access_time = t4.elapsed();

    unsafe { libc::munmap(mmap_ptr, file_size) };

    println!("  mmap() syscall:           {:?}  (just sets up page tables)", mmap_time);
    println!("  First byte access:        {:?}  (1 page fault)", first_access_time);
    println!("  Touch all pages (cold):   {:?}  ({} page faults)", all_pages_time, file_size / 4096);
    println!("  Touch all pages (hot):    {:?}  (no page faults)", cached_access_time);
    println!("\n  KEY: mmap() returns BEFORE any I/O. Cost is deferred to access time.");
}

fn test_page_fault_cost(path: &str) {
    use std::fs::File;
    use std::os::unix::io::AsRawFd;

    // Read file to warm cache
    let _ = std::fs::read(path);

    let file = File::open(path).unwrap();
    let file_size = file.metadata().unwrap().len() as usize;

    let mmap_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            file_size,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };

    // Even with hot cache, first access has page fault overhead
    let t1 = Instant::now();
    let mut sum: u64 = 0;
    for i in (0..file_size).step_by(4096) {
        sum = sum.wrapping_add(unsafe { *(mmap_ptr as *const u8).add(i) } as u64);
    }
    std::hint::black_box(sum);
    let first_pass = t1.elapsed();

    // Second pass - no page faults
    let t2 = Instant::now();
    let mut sum2: u64 = 0;
    for i in (0..file_size).step_by(4096) {
        sum2 = sum2.wrapping_add(unsafe { *(mmap_ptr as *const u8).add(i) } as u64);
    }
    std::hint::black_box(sum2);
    let second_pass = t2.elapsed();

    unsafe { libc::munmap(mmap_ptr, file_size) };

    let page_fault_overhead = first_pass.saturating_sub(second_pass);
    let num_pages = file_size / 4096;

    println!("  First pass (with page faults):  {:?}", first_pass);
    println!("  Second pass (no page faults):   {:?}", second_pass);
    println!("  Page fault overhead:            {:?} ({} pages)", page_fault_overhead, num_pages);
    println!("  Per-page fault cost:            {:?}", page_fault_overhead / num_pages as u32);
}

fn test_gpu_direct_cache_behavior(device: &Device, path: &str) {
    use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOFileHandle, IOPriority, IOQueueType};

    let file_size = std::fs::metadata(path).unwrap().len() as usize;

    // Warm cache
    let _ = std::fs::read(path);

    let io_queue = GpuIOQueue::new(device, IOPriority::High, IOQueueType::Concurrent).unwrap();
    let buffer = device.new_buffer(
        ((file_size + 4095) & !4095) as u64,
        MTLResourceOptions::StorageModeShared
    );

    // First load (file in cache)
    let file_handle1 = GpuIOFileHandle::open(device, path).unwrap();
    let t1 = Instant::now();
    let cmd1 = io_queue.command_buffer().unwrap();
    cmd1.load_buffer(&buffer, 0, file_size as u64, &file_handle1, 0);
    cmd1.commit();
    cmd1.wait_until_completed();
    let first_load = t1.elapsed();

    // Second load (file still in cache, buffer already allocated)
    let file_handle2 = GpuIOFileHandle::open(device, path).unwrap();
    let t2 = Instant::now();
    let cmd2 = io_queue.command_buffer().unwrap();
    cmd2.load_buffer(&buffer, 0, file_size as u64, &file_handle2, 0);
    cmd2.commit();
    cmd2.wait_until_completed();
    let second_load = t2.elapsed();

    println!("  First GPU-direct load (hot cache):  {:?}", first_load);
    println!("  Second GPU-direct load (hot cache): {:?}", second_load);
    println!("  Throughput: {:.1} GB/s", (file_size as f64 / 1e9) / first_load.as_secs_f64());
    println!("\n  KEY: GPU-direct ALWAYS copies data, even from page cache.");
    println!("       It cannot reuse existing mapped pages like mmap does.");
}

fn test_cold_cache_simulation(device: &Device) {
    // Create a large file to push test file out of cache
    let evict_size = 200 * 1024 * 1024; // 200MB
    create_test_file("evict.bin", evict_size);

    // Our test file
    let test_path = "test_10mb.bin";
    let file_size = std::fs::metadata(test_path).unwrap().len() as usize;

    println!("  Simulating cold cache by reading 200MB of other data...\n");

    // Read eviction file to push test file out
    let _ = std::fs::read("evict.bin");

    // Now test both methods on "cold" file
    let t_mmap = Instant::now();
    let _mmap_idx = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load(device, test_path);
    let mmap_time = t_mmap.elapsed();

    // Evict again
    let _ = std::fs::read("evict.bin");

    let t_gpu = Instant::now();
    let _gpu_idx = rust_experiment::gpu_os::gpu_index::GpuResidentIndex::load_gpu_direct(device, test_path);
    let gpu_time = t_gpu.elapsed();

    println!("  Cold cache results (file likely evicted):");
    println!("    mmap:       {:?}", mmap_time);
    println!("    GPU-direct: {:?}", gpu_time);

    if mmap_time > gpu_time {
        println!("    → GPU-direct wins on cold cache!");
    } else {
        println!("    → mmap still wins (cache not fully evicted on macOS)");
    }

    let _ = std::fs::remove_file("evict.bin");
}

fn create_test_file(path: &str, size: usize) {
    use std::io::Write;
    let mut file = std::fs::File::create(path).unwrap();

    // Write header for index format
    file.write_all(&0x47505549u32.to_le_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&((size / 256) as u32).to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();

    // Pad to size
    let remaining = size - 24;
    let chunk = vec![0u8; 1024 * 1024];
    let mut written = 0;
    while written < remaining {
        let to_write = std::cmp::min(chunk.len(), remaining - written);
        file.write_all(&chunk[..to_write]).unwrap();
        written += to_write;
    }

    println!("Created {} ({:.1} MB)", path, size as f64 / 1024.0 / 1024.0);
}
