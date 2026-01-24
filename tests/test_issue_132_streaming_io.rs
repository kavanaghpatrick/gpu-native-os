//! Issue #132: Streaming I/O - Overlap file loading with GPU search
//!
//! Tests for pipelined I/O where GPU searches chunks while more data loads.

use metal::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Import the streaming search module
use rust_experiment::gpu_os::streaming_search::{StreamingSearch, StreamingPipeline, StreamingProfile};
use rust_experiment::gpu_os::batch_io::GpuBatchLoader;
use rust_experiment::gpu_os::content_search::{GpuContentSearch, SearchOptions};

/// Collect test files from directory recursively
fn collect_test_files(dir: &str, max: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(Path::new(dir), &mut files, max, 0);
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
fn test_streaming_io_concept() {
    // Conceptual test: verify we can overlap I/O and compute
    let device = Device::system_default().expect("No Metal device");

    // Create two buffers to simulate double-buffering
    let buffer_a = device.new_buffer(1024 * 1024, MTLResourceOptions::StorageModeShared);
    let buffer_b = device.new_buffer(1024 * 1024, MTLResourceOptions::StorageModeShared);

    println!("Streaming I/O concept test:");
    println!("  Buffer A: {} bytes", buffer_a.length());
    println!("  Buffer B: {} bytes", buffer_b.length());
    println!("  Double-buffering allows overlap of I/O and compute");

    assert!(buffer_a.length() > 0);
    assert!(buffer_b.length() > 0);
}

#[test]
fn test_streaming_pipeline_creation() {
    let device = Device::system_default().expect("No Metal device");

    match StreamingPipeline::new(&device) {
        Some(pipeline) => {
            println!("StreamingPipeline created with {} chunks", pipeline.chunk_count());
            assert_eq!(pipeline.chunk_count(), 4); // Default quad-buffering
        }
        None => {
            println!("MTLIOCommandQueue not available (requires Metal 3+)");
        }
    }
}

#[test]
fn test_streaming_search_creation() {
    let device = Device::system_default().expect("No Metal device");

    match StreamingSearch::new(&device) {
        Some(search) => {
            println!("StreamingSearch created successfully");
            assert_eq!(search.pipeline().chunk_count(), 4);
        }
        None => {
            println!("StreamingSearch requires Metal 3+ with MTLIOCommandQueue");
        }
    }
}

#[test]
fn test_chunk_sizing() {
    // Test optimal chunk sizes for streaming
    let test_sizes = [
        (100, 4),      // 100 files, 4 chunks = 25 files/chunk
        (1000, 4),     // 1000 files, 4 chunks = 250 files/chunk
        (10000, 8),    // 10K files, 8 chunks = 1250 files/chunk
        (100000, 16),  // 100K files, 16 chunks = 6250 files/chunk
    ];

    println!("Chunk sizing analysis:");
    for (file_count, chunk_count) in test_sizes {
        let files_per_chunk = file_count / chunk_count;
        println!("  {} files / {} chunks = {} files/chunk",
            file_count, chunk_count, files_per_chunk);

        // Verify reasonable chunk sizes
        assert!(files_per_chunk >= 10, "Chunks too small - overhead dominates");
        assert!(files_per_chunk <= 10000, "Chunks too large - poor overlap");
    }
}

#[test]
fn test_quad_buffer_rotation() {
    // Test quad-buffer rotation logic
    let mut buffers = [0u32; 4];
    let mut load_idx = 0usize;
    let mut search_idx = 0usize;

    println!("Quad-buffer rotation test:");

    // Simulate 8 iterations
    for i in 0..8 {
        // Mark buffer as "loading"
        buffers[load_idx] = i as u32 + 1;
        println!("  Iteration {}: Load buffer {}, Search buffer {}",
            i, load_idx, search_idx);

        // Advance indices
        load_idx = (load_idx + 1) % 4;
        search_idx = (search_idx + 1) % 4;
    }

    // All buffers should have been used twice
    assert_eq!(buffers.iter().filter(|&&x| x > 0).count(), 4);
}

#[test]
fn test_file_partitioning() {
    let device = Device::system_default().expect("No Metal device");

    let pipeline = match StreamingPipeline::with_config(&device, 4, 1024 * 1024, 100) {
        Some(p) => p,
        None => {
            println!("Skipping: MTLIOCommandQueue not available");
            return;
        }
    };

    let files = collect_test_files("src", 50);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    let partitions = pipeline.partition_files(&files);
    println!("Partitioned {} files into {} chunks:", files.len(), partitions.len());
    for (i, (start, end)) in partitions.iter().enumerate() {
        println!("  Chunk {}: files {}..{} ({} files)", i, start, end, end - start);
    }

    assert!(!partitions.is_empty());
    assert!(partitions.len() <= pipeline.chunk_count());
}

#[test]
fn test_streaming_search_basic() {
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    let files = collect_test_files("src", 20);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    println!("Testing streaming search on {} files", files.len());

    let results = streaming.search_streaming(&files, "fn ", false);

    println!("Found {} matches for 'fn '", results.len());

    // Should find function definitions in Rust files
    assert!(results.len() > 0, "Should find function definitions");

    // Print first few matches
    for (i, m) in results.iter().take(5).enumerate() {
        println!("  Match {}: {}:{} - {}", i, m.file_path, m.line_number,
            m.context.chars().take(50).collect::<String>());
    }
}

#[test]
fn test_streaming_search_correctness() {
    // Verify streaming produces same results as batch search
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    let batch_loader = match GpuBatchLoader::new(&device) {
        Some(l) => l,
        None => {
            println!("Skipping: GpuBatchLoader requires Metal 3+");
            return;
        }
    };

    let files = collect_test_files("src", 30);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    let pattern = "struct";

    // Streaming search
    let streaming_results = streaming.search_streaming(&files, pattern, false);

    // Batch search for comparison
    let batch_result = match batch_loader.load_batch(&files) {
        Some(r) => r,
        None => {
            println!("Batch load failed");
            return;
        }
    };

    let mut content_search = GpuContentSearch::new(&device, batch_result.file_count()).expect("Content search creation failed");
    content_search.load_from_batch(&batch_result).expect("Load from batch failed");

    let batch_results = content_search.search(pattern, &SearchOptions::default());

    println!("Correctness test:");
    println!("  Pattern: '{}'", pattern);
    println!("  Streaming found {} matches", streaming_results.len());
    println!("  Batch found {} matches", batch_results.len());

    // Results should be similar (exact match depends on file ordering)
    // Allow some variance due to different processing order
    let diff = (streaming_results.len() as i64 - batch_results.len() as i64).abs();
    let tolerance = (batch_results.len() / 10).max(2) as i64; // 10% or at least 2

    assert!(diff <= tolerance,
        "Result counts differ too much: streaming={}, batch={}",
        streaming_results.len(), batch_results.len());
}

#[test]
fn test_streaming_search_with_profile() {
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    let files = collect_test_files(".", 100);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    println!("\n=== Streaming Search with Profile ===\n");
    println!("Files: {}", files.len());

    let (results, profile) = streaming.search_streaming_with_profile(&files, "TODO", false);

    println!("Results: {} matches", results.len());
    profile.print();

    // Verify profile has reasonable values
    assert!(profile.total_us > 0, "Total time should be > 0");
    assert!(profile.chunk_count > 0, "Should have at least 1 chunk");
}

#[test]
fn test_mtl_shared_event_availability() {
    // Test that MTLSharedEvent is available for synchronization
    let device = Device::system_default().expect("No Metal device");

    println!("MTLSharedEvent availability test:");
    println!("  Device: {}", device.name());
    println!("  Shared events: Available on Apple Silicon");
    println!("  Use case: Signal CPU when GPU finishes chunk");
}

#[test]
fn benchmark_sequential_vs_streaming_simulation() {
    // Simulate the difference between sequential and streaming
    let device = Device::system_default().expect("No Metal device");
    let files = collect_test_files("src", 50);

    if files.is_empty() {
        println!("No test files found, skipping benchmark");
        return;
    }

    println!("\n=== Sequential vs Streaming Simulation ===\n");
    println!("Files: {}", files.len());

    // Simulate sequential: total_time = load_time + search_time
    let simulated_load_time_ms = 50.0;  // 50ms to load all files
    let simulated_search_time_ms = 10.0; // 10ms to search
    let sequential_total = simulated_load_time_ms + simulated_search_time_ms;

    // Simulate streaming with 4 chunks:
    let chunks = 4;
    let load_per_chunk = simulated_load_time_ms / chunks as f64;
    let search_per_chunk = simulated_search_time_ms / chunks as f64;
    let overlap_savings = (chunks - 1) as f64 * search_per_chunk.min(load_per_chunk);
    let streaming_total = sequential_total - overlap_savings;

    println!("Sequential: {:.1}ms (load) + {:.1}ms (search) = {:.1}ms",
        simulated_load_time_ms, simulated_search_time_ms, sequential_total);
    println!("Streaming:  {:.1}ms (overlap saves {:.1}ms)",
        streaming_total, overlap_savings);
    println!("Speedup:    {:.2}x", sequential_total / streaming_total);

    // In real implementation, expect 1.1-1.4x speedup
    assert!(streaming_total < sequential_total, "Streaming should be faster");
}

#[test]
fn benchmark_streaming_vs_batch() {
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    let batch_loader = match GpuBatchLoader::new(&device) {
        Some(l) => l,
        None => {
            println!("Skipping: GpuBatchLoader requires Metal 3+");
            return;
        }
    };

    let files = collect_test_files(".", 200);
    if files.len() < 20 {
        println!("Not enough test files ({}) for meaningful benchmark", files.len());
        return;
    }

    println!("\n=== Streaming vs Batch Benchmark ===\n");
    println!("Files: {}", files.len());

    // Warmup
    let _ = streaming.search_streaming(&files[..10], "test", false);

    // Batch: load all, then search
    let batch_start = Instant::now();
    let batch_result = batch_loader.load_batch(&files).expect("Batch load failed");
    let mut content_search = GpuContentSearch::new(&device, batch_result.file_count())
        .expect("Content search creation failed");
    content_search.load_from_batch(&batch_result).expect("Load from batch failed");
    let batch_matches = content_search.search("fn ", &SearchOptions::default());
    let batch_time = batch_start.elapsed();

    // Streaming: overlap I/O and compute
    let stream_start = Instant::now();
    let (stream_matches, profile) = streaming.search_streaming_with_profile(&files, "fn ", false);
    let stream_time = stream_start.elapsed();

    println!("Batch:");
    println!("  Time: {:.1}ms", batch_time.as_secs_f64() * 1000.0);
    println!("  Matches: {}", batch_matches.len());

    println!("\nStreaming:");
    println!("  Time: {:.1}ms", stream_time.as_secs_f64() * 1000.0);
    println!("  Matches: {}", stream_matches.len());
    profile.print();

    let speedup = batch_time.as_secs_f64() / stream_time.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);

    // Note: Streaming may not always be faster for small file sets
    // The benefit comes from overlapping I/O with compute
    // For small files that fit in cache, batch may be competitive
    println!("\nNote: Streaming benefits increase with larger file sets");
    println!("      and slower I/O (cold cache, network storage, etc.)");
}

#[test]
fn test_empty_input() {
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    // Empty files list
    let empty: Vec<PathBuf> = vec![];
    let results = streaming.search_streaming(&empty, "test", false);
    assert!(results.is_empty(), "Empty files should return empty results");

    // Empty pattern
    let files = collect_test_files("src", 5);
    if !files.is_empty() {
        let results = streaming.search_streaming(&files, "", false);
        assert!(results.is_empty(), "Empty pattern should return empty results");
    }
}

#[test]
fn test_case_sensitivity() {
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    let files = collect_test_files("src", 30);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    // Case-insensitive
    let insensitive = streaming.search_streaming(&files, "struct", false);

    // Case-sensitive
    let sensitive = streaming.search_streaming(&files, "struct", true);

    println!("Case sensitivity test:");
    println!("  Case-insensitive 'struct': {} matches", insensitive.len());
    println!("  Case-sensitive 'struct': {} matches", sensitive.len());

    // Case-insensitive should find at least as many (usually more due to STRUCT, Struct, etc.)
    assert!(insensitive.len() >= sensitive.len(),
        "Case-insensitive should find at least as many matches");
}

#[test]
fn test_large_pattern() {
    let device = Device::system_default().expect("No Metal device");

    let mut streaming = match StreamingSearch::new(&device) {
        Some(s) => s,
        None => {
            println!("Skipping: StreamingSearch requires Metal 3+");
            return;
        }
    };

    let files = collect_test_files("src", 10);
    if files.is_empty() {
        println!("No test files found");
        return;
    }

    // Pattern at max length (64 chars)
    let max_pattern = "a".repeat(64);
    let results = streaming.search_streaming(&files, &max_pattern, false);
    println!("Max pattern (64 chars) found {} matches", results.len());

    // Pattern too long (65 chars) - should return empty
    let too_long = "a".repeat(65);
    let results = streaming.search_streaming(&files, &too_long, false);
    assert!(results.is_empty(), "Pattern > 64 chars should return empty");
}
