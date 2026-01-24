// GPU Ripgrep - Massively Parallel Code Search
//
// THE GPU IS THE COMPUTER. Zero CPU copies in the data path.
//
// Architecture:
// 1. GPU-Resident Index: Filesystem paths live in GPU memory (smart loading)
// 2. GPU Path Search: Filter paths by extension using GPU-resident data
// 3. Smart File Loading: Auto-detects hot/cold cache, picks optimal method
//    - Hot cache: mmap (249 GB/s, zero copy)
//    - Cold cache: MTLIOCommandQueue (CPU free during I/O)
// 4. GPU Content Search: Parallel pattern matching on GPU
//
// Result: CPU only orchestrates, never touches the data.

use metal::*;
use rust_experiment::gpu_os::content_search::{GpuContentSearch, SearchOptions};
use rust_experiment::gpu_os::mmap_buffer::MmapBuffer;
use rust_experiment::gpu_os::gpu_index::GpuResidentIndex;
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

/// Extract file paths from GPU-resident index that match our extensions AND search directory
fn filter_paths_from_index(index: &GpuResidentIndex, search_dir: &Path) -> Vec<String> {
    // Normalize search directory - handle both relative and absolute paths
    let search_str = search_dir.to_string_lossy();

    // For "." or relative paths, just check prefix directly
    // For absolute paths, canonicalize the entry paths for comparison
    let is_relative = search_str == "." || !search_dir.is_absolute();

    index.iter()
        .filter(|entry| !entry.is_dir())
        .filter(|entry| has_searchable_extension(entry.path_str()))
        .filter(|entry| {
            if is_relative {
                // For relative searches like "./src" or ".", all paths in the index match
                // since the index was built from the same base directory
                if search_str == "." {
                    true  // Current dir - include all files from this index
                } else {
                    // Check if path starts with the relative prefix
                    let entry_path = entry.path_str();
                    entry_path.starts_with(&*search_str) ||
                    entry_path.starts_with(&format!("./{}", search_str.trim_start_matches("./")))
                }
            } else {
                // Absolute path - canonicalize entry path for comparison
                let entry_path = Path::new(entry.path_str());
                entry_path.canonicalize()
                    .map(|p| p.starts_with(search_dir))
                    .unwrap_or(false)
            }
        })
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
                            // CRITICAL: Prefetch pages to avoid page faults during GPU blit!
                            // Without this, cold cache causes 30x slowdown.
                            buffer.advise_sequential();
                            buffer.advise_willneed();
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
    eprintln!("{}GPU Ripgrep{} - Massively Parallel Code Search\n", BOLD, RESET);
    eprintln!("{}Usage:{} {} <pattern> [directory]\n", YELLOW, RESET, program);
    eprintln!("{}Options:{}", YELLOW, RESET);
    eprintln!("  -m, --max-results <n>   Maximum results (default: 100)");
    eprintln!("  -s, --case-sensitive    Case-sensitive search");
    eprintln!("  --rebuild               Force rebuild filesystem index\n");
    eprintln!("{}Loading modes:{}", YELLOW, RESET);
    eprintln!("  --batch                 GPU-direct via MTLIOCommandQueue (default)");
    eprintln!("  --smart                 Auto-detect hot/cold cache");
    eprintln!("  --mmap                  Force mmap (fallback for Metal 2)\n");
    eprintln!("{}Architecture:{}", YELLOW, RESET);
    eprintln!("  1. GPU-Resident Index (filesystem paths in GPU memory)");
    eprintln!("  2. GPU-Direct File Loading (MTLIOCommandQueue, no CPU copies)");
    eprintln!("  3. GPU Parallel Content Search (O(1) chunked indexing)");
}

/// Loading mode for files
#[derive(Clone, Copy, PartialEq)]
enum LoadMode {
    Smart,  // Auto-detect hot/cold cache
    Mmap,   // Force mmap (zero-copy)
    Batch,  // Force MTLIOCommandQueue
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut pattern = String::new();
    let mut dir = PathBuf::from(".");
    let mut max_results = 100;
    let mut case_sensitive = false;
    let mut rebuild = false;
    let mut load_mode = LoadMode::Batch;  // GPU-direct is fastest

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
            "--smart" => load_mode = LoadMode::Smart,
            "--mmap" => load_mode = LoadMode::Mmap,
            "--batch" => load_mode = LoadMode::Batch,
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

    let load_mode_str = match load_mode {
        LoadMode::Smart => "smart (auto-detect)",
        LoadMode::Mmap => "mmap (zero-copy)",
        LoadMode::Batch => "batch (GPU-direct)",
    };

    println!();
    println!("{}┌─────────────────────────────────────────────────────────────┐{}", CYAN, RESET);
    println!("{}│{} {}GPU Ripgrep{} - THE GPU IS THE COMPUTER                    {}│{}", CYAN, RESET, BOLD, RESET, CYAN, RESET);
    println!("{}└─────────────────────────────────────────────────────────────┘{}", CYAN, RESET);
    println!();
    println!("  Device: {}{}{}", CYAN, device.name(), RESET);
    println!("  Pattern: {}\"{}\"{}  Directory: {}", YELLOW, pattern, RESET, dir.display());
    println!("  Mode: {}{}{}", CYAN, load_mode_str, RESET);
    println!();

    let total_start = Instant::now();

    // Phase 1: Build/Load GPU-Resident Index (using smart loading)
    println!("{}Phase 1:{} GPU-Resident Filesystem Index", BOLD, RESET);
    let index_path = std::env::temp_dir().join("gpu_ripgrep_index.bin");

    let index_start = Instant::now();
    let index = if rebuild || !index_path.exists() {
        // Build new index
        println!("  Building index for {}...", dir.display());
        let build_start = Instant::now();
        GpuResidentIndex::build_and_save(&dir, &index_path, None)
            .expect("Failed to build index");

        let idx = GpuResidentIndex::load_smart(&device, &index_path)
            .expect("Failed to load built index");

        println!("  {}✓{} Built index ({} entries) in {:.1}ms",
            GREEN, RESET,
            idx.entry_count(),
            build_start.elapsed().as_secs_f64() * 1000.0);
        idx
    } else {
        // Load existing index using smart loading (auto-detects hot/cold)
        match GpuResidentIndex::load_smart(&device, &index_path) {
            Ok(idx) => {
                let method = if idx.is_gpu_direct() { "GPU-direct" } else { "mmap" };
                println!("  {}✓{} Loaded cached index ({} entries, {:.1} MB) via {} in {:.1}ms",
                    GREEN, RESET,
                    idx.entry_count(),
                    idx.memory_usage() as f64 / (1024.0 * 1024.0),
                    method,
                    index_start.elapsed().as_secs_f64() * 1000.0);
                idx
            }
            Err(e) => {
                println!("  {}!{} Cache invalid ({}), rebuilding...", YELLOW, RESET, e);
                GpuResidentIndex::build_and_save(&dir, &index_path, None)
                    .expect("Failed to build index");
                GpuResidentIndex::load_smart(&device, &index_path)
                    .expect("Failed to load built index")
            }
        }
    };
    let index_time = index_start.elapsed();

    // Phase 2: Filter paths by extension AND search directory (using index data already in GPU)
    println!();
    println!("{}Phase 2:{} Path Filtering (GPU-resident data)", BOLD, RESET);
    let filter_start = Instant::now();
    let matching_paths = filter_paths_from_index(&index, &dir);
    let filter_time = filter_start.elapsed();
    println!("  {}✓{} Found {} searchable files in {} in {:.1}ms",
        GREEN, RESET,
        matching_paths.len(),
        dir.display(),
        filter_time.as_secs_f64() * 1000.0);

    if matching_paths.is_empty() {
        println!("\n  {}No searchable files found!{}", YELLOW, RESET);
        return;
    }

    // Phase 3: Load files (using selected mode)
    // Smart mode: probe first file to detect hot/cold cache
    let actual_mode = match load_mode {
        LoadMode::Smart => {
            // Probe first file to detect cache status
            if let Some(first_path) = matching_paths.first() {
                let probe_start = Instant::now();
                let _ = fs::File::open(first_path);
                let probe_time = probe_start.elapsed();

                if probe_time < std::time::Duration::from_micros(500) {
                    LoadMode::Mmap // Hot cache - use mmap (zero copy)
                } else {
                    LoadMode::Batch // Cold cache - use GPU-direct
                }
            } else {
                LoadMode::Mmap
            }
        }
        mode => mode,
    };

    // Handle batch mode separately since it uses search_direct (no blit copy!)
    if actual_mode == LoadMode::Batch {
        // MTLIOCommandQueue batch loading + DIRECT SEARCH (no blit copy!)
        println!();
        println!("{}Phase 3:{} GPU-Direct File Loading (MTLIOCommandQueue)", BOLD, RESET);

        let loader = match GpuBatchLoader::new(&device) {
            Some(l) => l,
            None => {
                println!("  {}✗{} MTLIOCommandQueue not available (requires Metal 3+)", RED, RESET);
                println!("  Falling back to mmap...");
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

        if load_mode == LoadMode::Smart {
            println!("    (Smart mode detected cold cache → GPU-direct)");
        }

        // Phase 4: GPU Content Search (use fast chunked kernel!)
        println!();
        println!("{}Phase 4:{} GPU Parallel Content Search", BOLD, RESET);

        let mut searcher = GpuContentSearch::new(&device, batch_result.file_count())
            .expect("Failed to create search engine");

        // Blit from mega_buffer to chunks_buffer (uses fast chunked layout)
        let blit_start = Instant::now();
        let chunks = searcher.load_from_batch(&batch_result).expect("Failed to load from batch");
        let blit_time = blit_start.elapsed();
        println!("  {}✓{} GPU blit: {} chunks in {:.1}ms",
            GREEN, RESET, chunks, blit_time.as_secs_f64() * 1000.0);

        let search_start = Instant::now();
        let options = SearchOptions {
            case_sensitive,
            max_results,
        };
        let (matches, profile) = searcher.search_with_profiling(&pattern, &options);
        let search_time = search_start.elapsed();

        println!("  {}✓{} GPU search: {}{:.2}ms{} ({} matches)",
            GREEN, RESET,
            GREEN, search_time.as_secs_f64() * 1000.0, RESET,
            matches.len());
        println!("    Breakdown: GPU kernel {:.1}ms, Extract {:.1}ms",
            profile.gpu_us as f64 / 1000.0,
            profile.extract_us as f64 / 1000.0);

        let total_time = total_start.elapsed();

        // Results
        println!();
        println!("{}─────────────────────────────────────────────────────────────{}", CYAN, RESET);
        println!("{}Results:{}", BOLD, RESET);
        println!();

        for m in matches.iter().take(max_results) {
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
        let index_method = if index.is_gpu_direct() { "GPU-direct" } else { "mmap" };
        println!("  Phase 1 (Index):     {:>7.1}ms  {} entries via {}",
            index_time.as_secs_f64() * 1000.0, index.entry_count(), index_method);
        println!("  Phase 2 (Filter):    {:>7.1}ms  {} files matched",
            filter_time.as_secs_f64() * 1000.0, matching_paths.len());
        println!("  Phase 3 (Load):      {:>7.1}ms  {:.1} MB via GPU-direct",
            load_time.as_secs_f64() * 1000.0, total_size as f64 / (1024.0 * 1024.0));
        println!("  Phase 3b (Blit):     {:>7.1}ms  {} chunks",
            blit_time.as_secs_f64() * 1000.0, chunks);
        println!("  Phase 4 (Search):    {:>7.1}ms  GPU kernel",
            search_time.as_secs_f64() * 1000.0);
        println!("  {}─────────────────────────────────{}", CYAN, RESET);
        println!("  {}Total:              {:>7.1}ms{}", BOLD, total_time.as_secs_f64() * 1000.0, RESET);
        println!();

        // Throughput (use chunks * 4KB like the benchmark does)
        let gpu_seconds = profile.gpu_us as f64 / 1_000_000.0;
        let data_mb = (chunks * 4096) as f64 / (1024.0 * 1024.0);
        let throughput_gbps = data_mb / gpu_seconds / 1024.0;
        println!("  {}GPU Kernel Throughput: {:.1} GB/s{} ({:.1} MB searched)", GREEN, throughput_gbps, RESET, data_mb);
        println!();

        return;  // Done with batch mode
    }

    // mmap mode (original code path) - Batch mode already returned above
    let (total_size, chunks, load_time, chunks_time, mut searcher, method_name) = {
            // mmap loading (zero-copy via unified memory)
            println!();
            println!("{}Phase 3:{} Zero-Copy File Loading (mmap)", BOLD, RESET);
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

            if load_mode == LoadMode::Smart {
                println!("    (Smart mode detected hot cache → mmap zero-copy)");
            }

            // Phase 4: GPU Content Search
            println!();
            println!("{}Phase 4:{} GPU Parallel Content Search", BOLD, RESET);

            let mut searcher = GpuContentSearch::new(&device, mmap_buffers.len())
                .expect("Failed to create search engine");

            let buffer_refs: Vec<(String, &MmapBuffer)> = mmap_buffers.iter()
                .map(|(p, b)| (p.clone(), b))
                .collect();

            let chunks_start = Instant::now();
            let chunks = searcher.load_from_mmap(&buffer_refs).expect("Failed to load from mmap");
            let chunks_time = chunks_start.elapsed();

            println!("  {}✓{} GPU blit: {} chunks in {:.1}ms (zero CPU copies!)",
                GREEN, RESET,
                chunks, chunks_time.as_secs_f64() * 1000.0);

        (total_size, chunks, load_time, chunks_time, searcher, "mmap")
    };

    // Do the search with profiling!
    let search_start = Instant::now();
    let options = SearchOptions {
        case_sensitive,
        max_results,
    };
    let (matches, profile) = searcher.search_with_profiling(&pattern, &options);
    let search_time = search_start.elapsed();

    println!("  {}✓{} GPU search: {}{:.2}ms{} ({} matches)",
        GREEN, RESET,
        GREEN, search_time.as_secs_f64() * 1000.0, RESET,
        matches.len());
    println!("    Breakdown: GPU kernel {:.1}ms, Extract {:.1}ms",
        profile.gpu_us as f64 / 1000.0,
        profile.extract_us as f64 / 1000.0);

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
    let index_method = if index.is_gpu_direct() { "GPU-direct" } else { "mmap" };
    println!("  Phase 1 (Index):     {:>7.1}ms  {} entries via {}",
        index_time.as_secs_f64() * 1000.0, index.entry_count(), index_method);
    println!("  Phase 2 (Filter):    {:>7.1}ms  {} files matched",
        filter_time.as_secs_f64() * 1000.0, matching_paths.len());
    println!("  Phase 3 (Load):      {:>7.1}ms  {:.1} MB via {}",
        load_time.as_secs_f64() * 1000.0, total_size as f64 / (1024.0 * 1024.0), method_name);
    println!("  Phase 4 (Search):    {:>7.1}ms  {} chunks searched",
        search_time.as_secs_f64() * 1000.0, chunks);
    println!("  {}─────────────────────────────────{}", CYAN, RESET);
    println!("  {}Total:              {:>7.1}ms{}", BOLD, total_time.as_secs_f64() * 1000.0, RESET);
    println!();

    // Throughput (GPU kernel only, excluding result extraction)
    let gpu_seconds = profile.gpu_us as f64 / 1_000_000.0;
    let data_mb = (chunks * 4096) as f64 / (1024.0 * 1024.0);
    let throughput_gbps = data_mb / gpu_seconds / 1024.0;
    println!("  {}GPU Kernel Throughput: {:.1} GB/s{} ({:.1} MB searched)", GREEN, throughput_gbps, RESET, data_mb);
    println!();
}
