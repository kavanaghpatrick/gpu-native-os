// Issue #125: GPU-Initiated Batch I/O with MTLIOCommandQueue
//
// THE GPU IS THE COMPUTER - batch ALL file loads into single GPU command

use metal::*;
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Instant;

// Re-export when module exists
// use rust_experiment::gpu_os::batch_io::{GpuBatchLoader, BatchLoadHandle};
use rust_experiment::gpu_os::gpu_io::{GpuIOQueue, GpuIOBuffer, IOPriority, IOQueueType, supports_gpu_io};

/// Collect test files from a directory
fn collect_test_files(dir: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(dir, &mut files, max_files, 0);
    files
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>, max: usize, depth: usize) {
    if depth > 5 || files.len() >= max {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        if files.len() >= max {
            break;
        }

        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') && name != "target" {
                collect_recursive(&path, files, max, depth + 1);
            }
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ["rs", "txt", "md", "toml"].contains(&ext) {
                if let Ok(meta) = fs::metadata(&path) {
                    if meta.len() > 0 && meta.len() < 1_000_000 {
                        files.push(path);
                    }
                }
            }
        }
    }
}

#[test]
fn test_gpu_io_available() {
    let device = Device::system_default().expect("No Metal device");
    let supports = supports_gpu_io(&device);
    println!("MTLIOCommandQueue supported: {}", supports);
    // Should be true on Metal 3+ devices (Apple Silicon with macOS 13+)
}

#[test]
fn test_single_file_gpu_load() {
    let device = Device::system_default().expect("No Metal device");

    // Skip if GPU I/O not supported
    let queue = match GpuIOQueue::new(&device, IOPriority::Normal, IOQueueType::Concurrent) {
        Some(q) => q,
        None => {
            println!("Skipping: MTLIOCommandQueue not available");
            return;
        }
    };

    // Load a single file
    let test_file = Path::new("Cargo.toml");
    if !test_file.exists() {
        println!("Skipping: Cargo.toml not found");
        return;
    }

    let start = Instant::now();
    let buffer = GpuIOBuffer::load_file(&queue, test_file);
    let elapsed = start.elapsed();

    match buffer {
        Some(buf) => {
            println!("Loaded {} bytes in {:.2}ms via GPU I/O",
                buf.file_size(), elapsed.as_secs_f64() * 1000.0);

            // Verify content matches
            let expected = fs::read(test_file).expect("Failed to read file");
            let actual_ptr = buf.metal_buffer().contents() as *const u8;
            let actual = unsafe {
                std::slice::from_raw_parts(actual_ptr, buf.file_size() as usize)
            };
            assert_eq!(actual, expected.as_slice(), "Content mismatch!");
            println!("Content verified!");
        }
        None => {
            println!("GPU I/O load failed (may not be supported on this device)");
        }
    }
}

#[test]
fn test_batch_load_multiple_files() {
    let device = Device::system_default().expect("No Metal device");

    let queue = match GpuIOQueue::new(&device, IOPriority::High, IOQueueType::Concurrent) {
        Some(q) => q,
        None => {
            println!("Skipping: MTLIOCommandQueue not available");
            return;
        }
    };

    // Collect test files
    let files = collect_test_files(Path::new("."), 100);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    println!("Testing batch load of {} files", files.len());

    // Sequential GPU loads (current approach)
    let seq_start = Instant::now();
    let mut seq_buffers = Vec::new();
    for file in &files {
        if let Some(buf) = GpuIOBuffer::load_file(&queue, file) {
            seq_buffers.push(buf);
        }
    }
    let seq_time = seq_start.elapsed();

    // TODO: Implement true batch loading in GpuBatchLoader
    // let batch_start = Instant::now();
    // let batch_loader = GpuBatchLoader::new(&queue)?;
    // let batch_handle = batch_loader.load_batch(&files);
    // batch_handle.wait();
    // let batch_time = batch_start.elapsed();

    let total_bytes: u64 = seq_buffers.iter().map(|b| b.file_size()).sum();
    println!("Loaded {} files ({:.1} KB) in {:.1}ms via GPU I/O",
        seq_buffers.len(),
        total_bytes as f64 / 1024.0,
        seq_time.as_secs_f64() * 1000.0);

    let throughput_mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / seq_time.as_secs_f64();
    println!("Throughput: {:.1} MB/s", throughput_mbps);
}

#[test]
fn benchmark_gpu_io_vs_mmap() {
    let device = Device::system_default().expect("No Metal device");

    // Check GPU I/O availability
    let gpu_queue = GpuIOQueue::new(&device, IOPriority::High, IOQueueType::Concurrent);

    let files = collect_test_files(Path::new("."), 500);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    // Benchmark mmap approach (current)
    let mmap_start = Instant::now();
    let mut mmap_total = 0usize;
    for file in &files {
        if let Ok(data) = fs::read(file) {
            mmap_total += data.len();
        }
    }
    let mmap_time = mmap_start.elapsed();

    println!("\n=== File Loading Benchmark ({} files, {:.1} MB) ===\n",
        files.len(), mmap_total as f64 / (1024.0 * 1024.0));

    println!("std::fs::read (baseline):");
    println!("  Time: {:.1}ms", mmap_time.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} MB/s",
        (mmap_total as f64 / (1024.0 * 1024.0)) / mmap_time.as_secs_f64());

    // Benchmark GPU I/O if available
    if let Some(queue) = gpu_queue {
        let gpu_start = Instant::now();
        let mut gpu_total = 0u64;
        for file in &files {
            if let Some(buf) = GpuIOBuffer::load_file(&queue, file) {
                gpu_total += buf.file_size();
            }
        }
        let gpu_time = gpu_start.elapsed();

        println!("\nGPU I/O (MTLIOCommandQueue):");
        println!("  Time: {:.1}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Throughput: {:.1} MB/s",
            (gpu_total as f64 / (1024.0 * 1024.0)) / gpu_time.as_secs_f64());

        let speedup = mmap_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("\n  Speedup: {:.2}x", speedup);
    } else {
        println!("\nGPU I/O not available on this device");
    }
}

// ============================================================================
// Tests for batch loading (to be implemented)
// ============================================================================

#[test]
#[ignore = "Requires GpuBatchLoader implementation"]
fn test_batch_load_correctness() {
    // let files = collect_test_files(Path::new("."), 1000);

    // // Load sequential (reference)
    // let sequential: Vec<Vec<u8>> = files.iter()
    //     .filter_map(|f| fs::read(f).ok())
    //     .collect();

    // // Load batch
    // let batch_loader = GpuBatchLoader::new(&device)?;
    // let batch_handle = batch_loader.load_batch(&files);
    // let batch_result = batch_handle.wait();

    // // Verify each file matches
    // for (i, expected) in sequential.iter().enumerate() {
    //     let actual = batch_result.file_data(i);
    //     assert_eq!(actual, expected.as_slice(),
    //         "File {} content mismatch", files[i].display());
    // }
}

#[test]
#[ignore = "Requires GpuBatchLoader implementation"]
fn test_overlapped_search() {
    // Test that search can start before all files are loaded

    // let batch_loader = GpuBatchLoader::new(&device)?;
    // let files = collect_test_files(Path::new("."), 10000);

    // // Start loading
    // let handle = batch_loader.load_batch_async(&files);

    // // Immediately start searching (should work on completed files)
    // let mut total_matches = 0;
    // while !handle.is_complete() {
    //     let ready_count = handle.ready_count();
    //     if ready_count > 0 {
    //         let partial_matches = search_ready_files(&handle, "TODO");
    //         total_matches += partial_matches.len();
    //     }
    //     std::thread::sleep(std::time::Duration::from_micros(100));
    // }

    // // Final search on remaining
    // let final_matches = search_ready_files(&handle, "TODO");
    // total_matches += final_matches.len();

    // println!("Found {} matches with overlapped search", total_matches);
}

#[test]
#[ignore = "Requires GpuBatchLoader implementation"]
fn benchmark_batch_vs_sequential() {
    // let device = Device::system_default().expect("No Metal device");
    // let files = collect_test_files(Path::new("."), 10000);

    // println!("Benchmark: {} files", files.len());

    // // Sequential mmap
    // let seq_start = Instant::now();
    // let _seq_buffers: Vec<_> = files.iter()
    //     .filter_map(|f| MmapBuffer::from_file(&device, f).ok())
    //     .collect();
    // let seq_time = seq_start.elapsed();

    // // Batch GPU I/O
    // let batch_start = Instant::now();
    // let batch_loader = GpuBatchLoader::new(&device)?;
    // let handle = batch_loader.load_batch(&files);
    // handle.wait();
    // let batch_time = batch_start.elapsed();

    // println!("Sequential mmap: {:.1}ms", seq_time.as_secs_f64() * 1000.0);
    // println!("Batch GPU I/O:   {:.1}ms", batch_time.as_secs_f64() * 1000.0);
    // println!("Speedup:         {:.1}x", seq_time.as_secs_f64() / batch_time.as_secs_f64());

    // assert!(batch_time < seq_time / 3, "Expected at least 3x speedup");
}
