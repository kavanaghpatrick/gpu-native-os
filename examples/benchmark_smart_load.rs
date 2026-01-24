// Benchmark: Smart Index Loading
//
// Tests load_smart() which auto-detects cache status and picks optimal method

use metal::Device;
use rust_experiment::gpu_os::gpu_index::GpuResidentIndex;
use std::time::Instant;
use std::path::Path;

fn main() {
    let device = Device::system_default().expect("No Metal device");

    println!("=== Smart Index Loading Benchmark ===\n");

    // Create test index
    let index_path = "test_smart_load.bin";
    if !Path::new(index_path).exists() {
        create_test_index(index_path, 100_000);
    }

    let file_size = std::fs::metadata(index_path).unwrap().len();
    println!("Index: {} ({:.2} MB, 100K entries)\n", index_path, file_size as f64 / 1024.0 / 1024.0);

    // Test 1: Hot cache comparison
    println!("--- Test 1: Hot Cache (file in RAM) ---\n");

    // Warm the cache
    let _ = std::fs::read(index_path);

    // Compare all methods
    for _ in 0..3 {
        let t1 = Instant::now();
        let idx1 = GpuResidentIndex::load(&device, index_path).unwrap();
        let mmap_time = t1.elapsed();

        let t2 = Instant::now();
        let idx2 = GpuResidentIndex::load_gpu_direct(&device, index_path).unwrap();
        let gpu_time = t2.elapsed();

        let t3 = Instant::now();
        let idx3 = GpuResidentIndex::load_smart(&device, index_path).unwrap();
        let smart_time = t3.elapsed();

        println!("mmap:       {:>10?}  ({} entries)", mmap_time, idx1.entry_count());
        println!("GPU-direct: {:>10?}  ({} entries)", gpu_time, idx2.entry_count());
        println!("smart:      {:>10?}  ({} entries, chose: {})\n",
            smart_time, idx3.entry_count(),
            if idx3.is_gpu_direct() { "GPU-direct" } else { "mmap" });
    }

    // Test 2: Async loading
    println!("--- Test 2: Async Loading (CPU free during I/O) ---\n");

    let t_start = Instant::now();
    let async_handle = GpuResidentIndex::load_async(&device, index_path).unwrap();
    let launch_time = t_start.elapsed();

    println!("Async launched in:    {:?}", launch_time);
    println!("CPU is now FREE to do other work...\n");

    // Simulate CPU work
    let cpu_start = Instant::now();
    let mut sum: u64 = 0;
    for i in 0..10_000_000 {
        sum = sum.wrapping_add(i);
    }
    std::hint::black_box(sum);
    let cpu_work_time = cpu_start.elapsed();
    println!("Did CPU work:         {:?}", cpu_work_time);

    // Check if complete
    println!("Is loading done?      {}", async_handle.is_complete());

    // Wait for completion
    let wait_start = Instant::now();
    let idx = async_handle.wait().unwrap();
    let wait_time = wait_start.elapsed();

    println!("Additional wait:      {:?}", wait_time);
    println!("Total wall time:      {:?}", t_start.elapsed());
    println!("Index loaded:         {} entries\n", idx.entry_count());

    // Test 3: Multiple sequential loads (demonstrates queue reuse benefit)
    println!("--- Test 3: Sequential Loads ---\n");

    let mut total_mmap = std::time::Duration::ZERO;
    let mut total_smart = std::time::Duration::ZERO;

    for i in 0..5 {
        let t1 = Instant::now();
        let _ = GpuResidentIndex::load(&device, index_path).unwrap();
        total_mmap += t1.elapsed();

        let t2 = Instant::now();
        let _ = GpuResidentIndex::load_smart(&device, index_path).unwrap();
        total_smart += t2.elapsed();

        if i == 0 {
            println!("Load 1 - mmap: {:?}, smart: {:?}", t1.elapsed(), t2.elapsed());
        }
    }

    println!("5 loads - mmap total:  {:?}", total_mmap);
    println!("5 loads - smart total: {:?}", total_smart);
    println!("smart chose mmap:      {} (correct for hot cache)", !GpuResidentIndex::load_smart(&device, index_path).unwrap().is_gpu_direct());

    // Summary
    println!("\n=== Summary ===\n");
    println!("load()        - Always uses mmap (zero-copy, best for hot cache)");
    println!("load_gpu_direct() - Always uses GPU I/O (CPU free, best for cold cache)");
    println!("load_smart()  - Auto-detects and picks optimal method");
    println!("load_async()  - Non-blocking, CPU free during load\n");

    println!("Recommendation: Use load_smart() for general use,");
    println!("                load_async() when CPU responsiveness matters.");

    // Cleanup
    let _ = std::fs::remove_file(index_path);
}

fn create_test_index(path: &str, num_entries: u32) {
    use std::io::Write;

    let header_size = 4096;
    let entry_size = 256;

    let mut file = std::fs::File::create(path).unwrap();

    // Header
    file.write_all(&0x47505549u32.to_le_bytes()).unwrap(); // magic
    file.write_all(&1u32.to_le_bytes()).unwrap(); // version
    file.write_all(&num_entries.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap(); // flags
    file.write_all(&0u64.to_le_bytes()).unwrap(); // build_time
    file.write_all(&vec![0u8; header_size - 24]).unwrap();

    // Entries
    for i in 0..num_entries {
        let path_str = format!("/home/user/documents/project/src/module_{:04}/file_{:06}.rs", i / 100, i);
        let mut entry = vec![0u8; entry_size];
        let path_bytes = path_str.as_bytes();
        let len = path_bytes.len().min(224);
        entry[..len].copy_from_slice(&path_bytes[..len]);
        entry[224] = len as u8;
        file.write_all(&entry).unwrap();
    }

    println!("Created {} with {} entries\n", path, num_entries);
}
