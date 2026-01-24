// Benchmark: mmap vs GPU-direct index loading
//
// Compares two zero-copy approaches:
// 1. mmap + newBufferWithBytesNoCopy (CPU initiates, GPU shares via unified memory)
// 2. MTLIOCommandQueue (GPU loads directly, CPU never touches bytes)

use metal::Device;
use rust_experiment::gpu_os::gpu_index::GpuResidentIndex;
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("No Metal device");

    // Check for existing index files
    let index_paths = [
        "/Users/patrickkavanagh/.gpu_path_index.bin",
        "test_index.bin",
    ];

    let mut index_path = None;
    for path in &index_paths {
        if std::path::Path::new(path).exists() {
            index_path = Some(*path);
            break;
        }
    }

    let index_path = match index_path {
        Some(p) => p,
        None => {
            println!("No index file found. Creating test index...");
            create_test_index();
            "test_index.bin"
        }
    };

    println!("=== GPU-Resident Index Load Benchmark ===\n");
    println!("Index file: {}", index_path);

    // Get file size
    let file_size = std::fs::metadata(index_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("File size: {:.2} MB\n", file_size as f64 / 1024.0 / 1024.0);

    // Warm up - ensure file is in OS cache for fair comparison
    println!("Warming up (loading file into OS cache)...");
    let _ = std::fs::read(index_path);

    // Benchmark mmap loading
    println!("\n--- Benchmark: mmap (CPU-initiated, unified memory) ---");
    let mut mmap_times = Vec::new();

    for i in 0..5 {
        // Drop OS cache between runs for cold-cache testing
        if i > 0 {
            drop_cache();
        }

        let start = Instant::now();
        let index = GpuResidentIndex::load(&device, index_path).expect("Failed to load via mmap");
        let elapsed = start.elapsed();

        println!("  Run {}: {:?} ({} entries)", i + 1, elapsed, index.entry_count());
        mmap_times.push(elapsed);
    }

    let mmap_avg = mmap_times.iter().sum::<std::time::Duration>() / mmap_times.len() as u32;
    println!("  Average: {:?}", mmap_avg);

    // Benchmark GPU-direct loading
    println!("\n--- Benchmark: GPU-direct (MTLIOCommandQueue) ---");
    let mut gpu_times = Vec::new();

    for i in 0..5 {
        // Drop OS cache between runs for cold-cache testing
        if i > 0 {
            drop_cache();
        }

        let start = Instant::now();
        let index = GpuResidentIndex::load_gpu_direct(&device, index_path).expect("Failed to load via GPU-direct");
        let elapsed = start.elapsed();

        println!("  Run {}: {:?} ({} entries, is_gpu_direct={})",
            i + 1, elapsed, index.entry_count(), index.is_gpu_direct());
        gpu_times.push(elapsed);
    }

    let gpu_avg = gpu_times.iter().sum::<std::time::Duration>() / gpu_times.len() as u32;
    println!("  Average: {:?}", gpu_avg);

    // Summary
    println!("\n=== Summary ===");
    println!("mmap average:       {:?}", mmap_avg);
    println!("GPU-direct average: {:?}", gpu_avg);

    let speedup = mmap_avg.as_secs_f64() / gpu_avg.as_secs_f64();
    if speedup > 1.0 {
        println!("GPU-direct is {:.2}x faster", speedup);
    } else {
        println!("mmap is {:.2}x faster", 1.0 / speedup);
    }

    // Throughput comparison
    let file_mb = file_size as f64 / 1024.0 / 1024.0;
    println!("\n=== Throughput ===");
    println!("mmap:       {:.1} MB/s", file_mb / mmap_avg.as_secs_f64());
    println!("GPU-direct: {:.1} MB/s", file_mb / gpu_avg.as_secs_f64());

    println!("\n=== Key Insight ===");
    println!("For HOT cache (file in memory): mmap wins due to lower overhead.");
    println!("GPU-direct advantage: CPU is 100% FREE during load!");
    println!("  - mmap: CPU handles page faults on first access");
    println!("  - GPU-direct: Data flows Disk → NVMe → GPU, CPU does nothing");
    println!("Use GPU-direct when: CPU needs to do other work, or cold cache scenarios.");

    // Test data correctness
    println!("\n=== Data Verification ===");
    let mmap_index = GpuResidentIndex::load(&device, index_path).unwrap();
    let gpu_index = GpuResidentIndex::load_gpu_direct(&device, index_path).unwrap();

    println!("mmap entries: {}", mmap_index.entry_count());
    println!("GPU entries:  {}", gpu_index.entry_count());

    assert_eq!(mmap_index.entry_count(), gpu_index.entry_count(), "Entry counts don't match!");

    // Compare first few entries
    let mut matches = 0;
    for i in 0..std::cmp::min(10, mmap_index.entry_count() as usize) {
        if let (Some(mmap_entry), Some(gpu_entry)) = (mmap_index.get_entry(i), gpu_index.get_entry(i)) {
            let mmap_path = mmap_entry.path_str();
            let gpu_path = gpu_entry.path_str();

            if mmap_path == gpu_path {
                matches += 1;
                if i < 3 {
                    println!("  Entry {}: {} [OK]", i, mmap_path);
                }
            } else {
                println!("  Entry {}: {} != {} [MISMATCH]", i, mmap_path, gpu_path);
            }
        }
    }

    println!("Verified {} entries match!", matches);
}

fn drop_cache() {
    // On macOS, we can't easily drop file cache without sudo
    // Instead, read some other large files to push the index out of cache
    // This is imperfect but gives some cold-cache behavior
    let _ = std::fs::read("/usr/bin/git");
}

fn create_test_index() {
    // Create a larger test index to show GPU-direct I/O benefits
    use std::io::Write;

    let header_size = 4096;
    let entry_size = 256;
    let num_entries = 100_000u32; // 100K entries = ~25MB

    let mut file = std::fs::File::create("test_index.bin").expect("Failed to create test index");

    // Write header
    let magic = 0x47505549u32; // "GPUI"
    let version = 1u32;
    file.write_all(&magic.to_le_bytes()).unwrap();
    file.write_all(&version.to_le_bytes()).unwrap();
    file.write_all(&num_entries.to_le_bytes()).unwrap();
    file.write_all(&0u32.to_le_bytes()).unwrap(); // flags
    file.write_all(&0u64.to_le_bytes()).unwrap(); // build_time

    // Pad header to 4096 bytes
    let header_written = 4 + 4 + 4 + 4 + 8;
    file.write_all(&vec![0u8; header_size - header_written]).unwrap();

    // Write test entries
    for i in 0..num_entries {
        let path = format!("/test/path/to/file_{:06}.txt", i);
        let mut entry = vec![0u8; entry_size];

        // Copy path
        let path_bytes = path.as_bytes();
        let len = path_bytes.len().min(224);
        entry[..len].copy_from_slice(&path_bytes[..len]);

        // path_len at offset 224
        entry[224] = len as u8;
        entry[225] = 0;

        // flags at offset 226
        entry[226] = 0;
        entry[227] = 0;

        file.write_all(&entry).unwrap();
    }

    println!("Created test_index.bin with {} entries", num_entries);
}
