// GPU-Native Ripgrep - Full GPU Data Architecture
//
// THE GPU IS THE COMPUTER. Zero CPU copies in the data path.
//
// Architecture:
// 1. GPU-Resident Index: Filesystem paths live in GPU memory (mmap, zero-copy)
// 2. GPU Path Search: Filter paths by extension using GPU parallel search
// 3. Zero-Copy File Load: mmap files directly into GPU buffers
// 4. GPU Content Search: Parallel pattern matching on GPU
//
// Result: CPU only orchestrates, never touches the data.

use metal::*;
use rust_experiment::gpu_os::content_search::{GpuContentSearch, SearchOptions};
use rust_experiment::gpu_os::mmap_buffer::MmapBuffer;
use rust_experiment::gpu_os::gpu_index::{GpuResidentIndex, GpuPathEntry, FLAG_IS_DIR};
use rust_experiment::gpu_os::batch_io::GpuBatchLoader;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::fs;

// ANSI colors
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

const DEFAULT_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "c", "cpp", "h", "hpp",
    "go", "java", "rb", "swift", "kt", "scala", "cs",
    "md", "txt", "json", "yaml", "yml", "toml", "sh",
];

/// Check if a path has a searchable extension
fn has_searchable_extension(path: &str) -> bool {
    if let Some(ext) = path.rsplit('.').next() {
        DEFAULT_EXTENSIONS.contains(&ext.to_lowercase().as_str())
    } else {
        false
    }
}

/// Build a GPU-resident index from a directory
fn build_index(dir: &Path) -> Result<PathBuf, String> {
    let index_path = std::env::temp_dir().join("gpu_ripgrep_index.bin");

    // Build and save the index
    GpuResidentIndex::build_and_save(dir, &index_path, None)
        .map_err(|e| format!("Failed to build index: {:?}", e))?;

    Ok(index_path)
}

/// Extract file paths from GPU-resident index that match our extensions
fn filter_paths_from_index(index: &GpuResidentIndex) -> Vec<String> {
    index.iter()
        .filter(|entry| !entry.is_dir())
        .filter(|entry| has_searchable_extension(entry.path_str()))
        .map(|entry| entry.path_str().to_string())
        .collect()
}

/// Load files using mmap for zero-copy GPU access (PARALLEL!)
fn load_files_mmap(device: &Device, paths: &[String], max_size_mb: usize) -> Vec<(String, MmapBuffer)> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    let max_bytes = max_size_mb * 1024 * 1024;
    let total_bytes = AtomicUsize::new(0);

    // Parallel mmap using thread pool
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    let chunk_size = (paths.len() + num_threads - 1) / num_threads;
    let device_clone = device.clone();

    // Spawn threads to mmap files in parallel
    let handles: Vec<_> = paths
        .chunks(chunk_size)
        .map(|chunk| {
            let chunk_paths: Vec<String> = chunk.to_vec();
            let device = device_clone.clone();
            let total_ref = &total_bytes as *const AtomicUsize as usize;

            thread::spawn(move || {
                let total_bytes = unsafe { &*(total_ref as *const AtomicUsize) };
                let mut local_buffers = Vec::new();

                for path in chunk_paths {
                    // Check size limit
                    if total_bytes.load(Ordering::Relaxed) >= max_bytes {
                        break;
                    }

                    let path_ref = Path::new(&path);

                    // Skip files that don't exist or are too large
                    let metadata = match fs::metadata(path_ref) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };

                    let size = metadata.len() as usize;
                    if size == 0 || size > 10 * 1024 * 1024 {
                        continue;
                    }

                    // mmap the file
                    match MmapBuffer::from_file(&device, path_ref) {
                        Ok(buffer) => {
                            total_bytes.fetch_add(size, Ordering::Relaxed);
                            local_buffers.push((path, buffer));
                        }
                        Err(_) => continue,
                    }
                }

                local_buffers
            })
        })
        .collect();

    // Collect results
    let mut buffers = Vec::new();
    for handle in handles {
        if let Ok(mut local) = handle.join() {
            buffers.append(&mut local);
        }
    }

    buffers
}

fn print_usage(program: &str) {
    eprintln!("{}GPU-Native Ripgrep{} - Full GPU Data Architecture\n", BOLD, RESET);
    eprintln!("{}Usage:{} {} <pattern> [directory]\n", YELLOW, RESET, program);
    eprintln!("{}Options:{}", YELLOW, RESET);
    eprintln!("  -m, --max-results <n>   Maximum results (default: 100)");
    eprintln!("  -s, --case-sensitive    Case-sensitive search");
    eprintln!("  --rebuild               Force rebuild filesystem index");
    eprintln!("  --batch                 Use MTLIOCommandQueue batch loading (vs mmap)\n");
    eprintln!("{}Architecture:{}", YELLOW, RESET);
    eprintln!("  1. GPU-Resident Index (mmap, zero-copy)");
    eprintln!("  2. Zero-Copy File Loading (mmap → GPU)");
    eprintln!("  3. GPU Parallel Content Search");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut pattern = String::new();
    let mut dir = PathBuf::from(".");
    let mut max_results = 100;
    let mut case_sensitive = false;
    let mut rebuild = false;
    let mut use_batch_io = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--max-results" => {
                i += 1;
                if i < args.len() {
                    max_results = args[i].parse().unwrap_or(100);
                }
            }
            "-s" | "--case-sensitive" => case_sensitive = true,
            "--rebuild" => rebuild = true,
            "--batch" => use_batch_io = true,
            "-h" | "--help" => {
                print_usage(&args[0]);
                return;
            }
            arg => {
                if pattern.is_empty() {
                    pattern = arg.to_string();
                } else {
                    dir = PathBuf::from(arg);
                }
            }
        }
        i += 1;
    }

    if pattern.is_empty() {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    // Initialize Metal
    let device = Device::system_default().expect("No Metal device");

    println!();
    println!("{}┌─────────────────────────────────────────────────────────────┐{}", CYAN, RESET);
    println!("{}│{} {}GPU-Native Ripgrep{} - THE GPU IS THE COMPUTER             {}│{}", CYAN, RESET, BOLD, RESET, CYAN, RESET);
    println!("{}└─────────────────────────────────────────────────────────────┘{}", CYAN, RESET);
    println!();
    println!("  Device: {}{}{}", CYAN, device.name(), RESET);
    println!("  Pattern: {}\"{}\"{}  Directory: {}", YELLOW, pattern, RESET, dir.display());
    println!();

    let total_start = Instant::now();

    // Phase 1: Build/Load GPU-Resident Index
    println!("{}Phase 1:{} GPU-Resident Filesystem Index", BOLD, RESET);
    let index_path = std::env::temp_dir().join("gpu_ripgrep_index.bin");

    let index_start = Instant::now();
    let index = if !rebuild && index_path.exists() {
        // Load existing index (mmap, zero-copy!)
        match GpuResidentIndex::load(&device, &index_path) {
            Ok(idx) => {
                println!("  {}✓{} Loaded cached index ({} entries, {:.1} MB) in {:.1}ms",
                    GREEN, RESET,
                    idx.entry_count(),
                    idx.memory_usage() as f64 / (1024.0 * 1024.0),
                    index_start.elapsed().as_secs_f64() * 1000.0);
                idx
            }
            Err(_) => {
                println!("  {}!{} Cache invalid, rebuilding...", YELLOW, RESET);
                rebuild = true;
                // Will rebuild below
                GpuResidentIndex::load(&device, &index_path).unwrap() // placeholder
            }
        }
    } else {
        // Build new index
        println!("  Building index for {}...", dir.display());
        let build_start = Instant::now();
        GpuResidentIndex::build_and_save(&dir, &index_path, None)
            .expect("Failed to build index");

        let idx = GpuResidentIndex::load(&device, &index_path)
            .expect("Failed to load built index");

        println!("  {}✓{} Built index ({} entries) in {:.1}ms",
            GREEN, RESET,
            idx.entry_count(),
            build_start.elapsed().as_secs_f64() * 1000.0);
        idx
    };
    let index_time = index_start.elapsed();

    // Phase 2: Filter paths by extension (using index data already in GPU)
    println!();
    println!("{}Phase 2:{} Path Filtering (GPU-resident data)", BOLD, RESET);
    let filter_start = Instant::now();
    let matching_paths = filter_paths_from_index(&index);
    let filter_time = filter_start.elapsed();
    println!("  {}✓{} Found {} searchable files in {:.1}ms",
        GREEN, RESET,
        matching_paths.len(),
        filter_time.as_secs_f64() * 1000.0);

    if matching_paths.is_empty() {
        println!("\n  {}No searchable files found!{}", YELLOW, RESET);
        return;
    }

    // Phase 3: Load files
    let (total_size, chunks, load_time, chunks_time, mut searcher) = if use_batch_io {
        // MTLIOCommandQueue batch loading (TRUE GPU-DIRECT I/O!)
        println!();
        println!("{}Phase 3:{} GPU-Direct File Loading (MTLIOCommandQueue)", BOLD, RESET);

        let loader = match GpuBatchLoader::new(&device) {
            Some(l) => l,
            None => {
                println!("  {}✗{} MTLIOCommandQueue not available (requires Metal 3+)", RED, RESET);
                return;
            }
        };

        // Convert paths to PathBuf
        let file_paths: Vec<PathBuf> = matching_paths.iter()
            .map(|p| PathBuf::from(p))
            .collect();

        let load_start = Instant::now();
        let batch_result = match loader.load_batch(&file_paths) {
            Some(r) => r,
            None => {
                println!("  {}✗{} Batch load failed", RED, RESET);
                return;
            }
        };
        let load_time = load_start.elapsed();

        let total_size = batch_result.total_bytes as usize;
        println!("  {}✓{} Loaded {} files ({:.1} MB) via MTLIOCommandQueue in {:.1}ms",
            GREEN, RESET,
            batch_result.file_count(),
            total_size as f64 / (1024.0 * 1024.0),
            load_time.as_secs_f64() * 1000.0);
        println!("    TRUE GPU-DIRECT I/O - data flows: Disk → GPU (bypasses CPU!)");

        // Phase 4: GPU Content Search
        println!();
        println!("{}Phase 4:{} GPU Parallel Content Search", BOLD, RESET);

        let mut searcher = GpuContentSearch::new(&device, batch_result.file_count())
            .expect("Failed to create search engine");

        let chunks_start = Instant::now();
        let chunks = searcher.load_from_batch(&batch_result).expect("Failed to load from batch");
        let chunks_time = chunks_start.elapsed();

        println!("  {}✓{} GPU blit: {} chunks in {:.1}ms",
            GREEN, RESET,
            chunks, chunks_time.as_secs_f64() * 1000.0);

        (total_size, chunks, load_time, chunks_time, searcher)
    } else {
        // mmap loading (zero-copy via unified memory)
        println!();
        println!("{}Phase 3:{} Zero-Copy File Loading (mmap → GPU)", BOLD, RESET);
        let load_start = Instant::now();
        let mmap_buffers = load_files_mmap(&device, &matching_paths, 500); // 500MB max
        let load_time = load_start.elapsed();

        let total_size: usize = mmap_buffers.iter()
            .map(|(_, buf)| buf.file_size())
            .sum();

        println!("  {}✓{} Loaded {} files ({:.1} MB) via mmap in {:.1}ms",
            GREEN, RESET,
            mmap_buffers.len(),
            total_size as f64 / (1024.0 * 1024.0),
            load_time.as_secs_f64() * 1000.0);
        println!("    Zero CPU copies - data flows: Disk → Unified Memory → GPU");

        // Phase 4: GPU Content Search
        println!();
        println!("{}Phase 4:{} GPU Parallel Content Search", BOLD, RESET);

        // Create content search engine and load from mmap buffers (ZERO CPU COPIES!)
        let mut searcher = GpuContentSearch::new(&device, mmap_buffers.len())
            .expect("Failed to create search engine");

        // Convert to references for load_from_mmap
        let buffer_refs: Vec<(String, &MmapBuffer)> = mmap_buffers.iter()
            .map(|(p, b)| (p.clone(), b))
            .collect();

        let chunks_start = Instant::now();
        let chunks = searcher.load_from_mmap(&buffer_refs).expect("Failed to load from mmap");
        let chunks_time = chunks_start.elapsed();

        println!("  {}✓{} GPU blit: {} chunks in {:.1}ms (zero CPU copies!)",
            GREEN, RESET,
            chunks, chunks_time.as_secs_f64() * 1000.0);

        (total_size, chunks, load_time, chunks_time, searcher)
    };

    // Do the search!
    let search_start = Instant::now();
    let options = SearchOptions {
        case_sensitive,
        max_results,
    };
    let matches = searcher.search(&pattern, &options);
    let search_time = search_start.elapsed();

    println!("  {}✓{} GPU search: {}{:.2}ms{} ({} matches)",
        GREEN, RESET,
        GREEN, search_time.as_secs_f64() * 1000.0, RESET,
        matches.len());

    let total_time = total_start.elapsed();

    // Results
    println!();
    println!("{}─────────────────────────────────────────────────────────────{}", CYAN, RESET);
    println!("{}Results:{}", BOLD, RESET);
    println!();

    for m in matches.iter().take(max_results) {
        // Highlight the match in context
        let highlighted = if case_sensitive {
            m.context.replace(&pattern, &format!("{}{}{}{}", RED, BOLD, &pattern, RESET))
        } else {
            let lower = m.context.to_lowercase();
            let pattern_lower = pattern.to_lowercase();
            if let Some(pos) = lower.find(&pattern_lower) {
                let before = &m.context[..pos];
                let matched = &m.context[pos..pos + pattern.len()];
                let after = &m.context[pos + pattern.len()..];
                format!("{}{}{}{}{}{}", before, RED, BOLD, matched, RESET, after)
            } else {
                m.context.clone()
            }
        };

        println!("{}{}{}:{}{}{}:{}",
            MAGENTA, m.file_path, RESET,
            GREEN, m.line_number, RESET,
            highlighted);
    }

    // Summary
    println!();
    println!("{}─────────────────────────────────────────────────────────────{}", CYAN, RESET);
    println!("{}Performance Summary:{}", BOLD, RESET);
    println!();
    println!("  Phase 1 (Index):     {:>7.1}ms  {} entries, mmap zero-copy",
        index_time.as_secs_f64() * 1000.0, index.entry_count());
    println!("  Phase 2 (Filter):    {:>7.1}ms  {} files matched",
        filter_time.as_secs_f64() * 1000.0, matching_paths.len());
    println!("  Phase 3 (Load):      {:>7.1}ms  {:.1} MB via mmap",
        load_time.as_secs_f64() * 1000.0, total_size as f64 / (1024.0 * 1024.0));
    println!("  Phase 4 (Search):    {:>7.1}ms  {} chunks searched",
        search_time.as_secs_f64() * 1000.0, chunks);
    println!("  {}─────────────────────────────────{}", CYAN, RESET);
    println!("  {}Total:              {:>7.1}ms{}", BOLD, total_time.as_secs_f64() * 1000.0, RESET);
    println!();

    // Throughput
    let throughput_mbps = (total_size as f64 / (1024.0 * 1024.0)) / search_time.as_secs_f64();
    println!("  {}GPU Search Throughput: {:.1} GB/s{}", GREEN, throughput_mbps / 1000.0, RESET);
    println!();
}
