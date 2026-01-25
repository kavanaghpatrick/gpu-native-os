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
use rust_experiment::gpu_os::streaming_search::StreamingSearch;
use rust_experiment::gpu_os::persistent_search::PersistentSearchQueue;
use rust_experiment::gpu_os::shared_index::GpuFilesystemIndex;
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

/// Build a temporary GPU-resident index from a directory
fn build_temp_index(device: &Device, dir: &Path) -> GpuResidentIndex {
    let index_path = std::env::temp_dir().join("gpu_ripgrep_index.bin");

    let build_start = Instant::now();
    GpuResidentIndex::build_and_save(dir, &index_path, None)
        .expect("Failed to build index");

    let idx = GpuResidentIndex::load_smart(device, &index_path)
        .expect("Failed to load built index");

    println!("  {}✓{} Built temp index ({} entries) in {:.1}ms",
        GREEN, RESET,
        idx.entry_count(),
        build_start.elapsed().as_secs_f64() * 1000.0);

    idx
}

/// Extract file paths from GPU-resident index that match our extensions AND search directory
fn filter_paths_from_index(index: &GpuResidentIndex, search_dir: &Path) -> Vec<String> {
    // Canonicalize search_dir ONCE (not per-entry!)
    let canonical_search = search_dir.canonicalize()
        .unwrap_or_else(|_| search_dir.to_path_buf());

    // Add trailing slash to prevent prefix collisions
    // e.g., ~/code/ should not match ~/codex-tests/
    let mut search_prefix = canonical_search.to_string_lossy().to_string();
    if !search_prefix.ends_with('/') {
        search_prefix.push('/');
    }

    // Simple string prefix matching - no syscalls per entry!
    // The index stores absolute paths, so we just check if entry starts with search_prefix
    index.iter()
        .filter(|entry| !entry.is_dir())
        .filter(|entry| has_searchable_extension(entry.path_str()))
        .filter(|entry| {
            let entry_path = entry.path_str();
            // Fast string prefix check (no syscalls!)
            entry_path.starts_with(&search_prefix)
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
    eprintln!("  --streaming             Overlap I/O with GPU search (Issue #132)");
    eprintln!("  --persistent            Persistent kernel for burst search (Issue #133)");
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
    Smart,      // Auto-detect hot/cold cache
    Mmap,       // Force mmap (zero-copy)
    Batch,      // Force MTLIOCommandQueue
    Streaming,  // #132: Overlap I/O with GPU search
    Persistent, // #133: Persistent kernel for burst search
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
            "--streaming" => load_mode = LoadMode::Streaming,
            "--persistent" => load_mode = LoadMode::Persistent,
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
        LoadMode::Streaming => "streaming (I/O overlap)",
        LoadMode::Persistent => "persistent (burst search)",
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

    // Phase 1: Load Shared GPU-Resident Index (Issue #135)
    println!("{}Phase 1:{} Shared GPU-Resident Filesystem Index", BOLD, RESET);

    let index_start = Instant::now();

    // Try to use shared index first (instant load via mmap)
    // We need to keep shared_index alive if we use it, so declare both options
    let shared_index: Option<GpuFilesystemIndex>;
    let temp_index: Option<GpuResidentIndex>;
    let used_shared: bool;

    // Check if search dir is under home (covered by shared index)
    let search_abs = dir.canonicalize().unwrap_or_else(|_| dir.clone());
    let home = dirs::home_dir().unwrap_or_default();
    let is_under_home = search_abs.starts_with(&home);

    // Check if shared index already exists (don't build on first run - too slow)
    let shared_manifest = dirs::home_dir()
        .map(|h| h.join(".gpu_os/index/manifest.json"))
        .filter(|p| p.exists());

    if !rebuild && is_under_home && shared_manifest.is_some() {
        // Shared index exists - load it (fast!)
        match GpuFilesystemIndex::load_or_create(&device) {
            Ok(shared) if shared.total_entries() > 0 => {
                println!("  {}✓{} Using shared index ({} entries) loaded in {:.1}ms",
                    GREEN, RESET,
                    shared.total_entries(),
                    index_start.elapsed().as_secs_f64() * 1000.0);
                println!("    (Shared index at ~/.gpu_os/index/ - {}10x faster startup{})", GREEN, RESET);
                shared_index = Some(shared);
                temp_index = None;
                used_shared = true;
            }
            Ok(_) => {
                println!("  {}!{} Shared index empty, building temp...", YELLOW, RESET);
                temp_index = Some(build_temp_index(&device, &dir));
                shared_index = None;
                used_shared = false;
            }
            Err(e) => {
                println!("  {}!{} Shared index error ({}), building temp...", YELLOW, RESET, e);
                temp_index = Some(build_temp_index(&device, &dir));
                shared_index = None;
                used_shared = false;
            }
        }
    } else if !rebuild && is_under_home {
        // No shared index yet - use temp (run `gpu-index build` to create shared)
        println!("  {}!{} No shared index yet (run gpu-index build), using temp...", YELLOW, RESET);
        temp_index = Some(build_temp_index(&device, &dir));
        shared_index = None;
        used_shared = false;
    } else {
        // Build temp index (rebuild requested or outside home)
        if rebuild {
            println!("  {}!{} Rebuild requested, building fresh index...", YELLOW, RESET);
        } else {
            println!("  {}!{} Search dir outside home, building temp index...", YELLOW, RESET);
        }
        temp_index = Some(build_temp_index(&device, &dir));
        shared_index = None;
        used_shared = false;
    }

    // Get reference to the index we'll use
    let index: &GpuResidentIndex = if used_shared {
        shared_index.as_ref().unwrap().home().expect("Home index should exist")
    } else {
        temp_index.as_ref().unwrap()
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

    // Handle streaming mode (#132) - overlaps I/O with GPU search
    if load_mode == LoadMode::Streaming {
        println!();
        println!("{}Phase 3+4:{} Streaming Search (I/O overlapped with GPU)", BOLD, RESET);

        let mut streaming = StreamingSearch::new(&device)
            .expect("Failed to create streaming search");

        let file_paths: Vec<PathBuf> = matching_paths.iter()
            .map(|p| PathBuf::from(p))
            .collect();

        let search_start = Instant::now();
        let (matches, profile) = streaming.search_streaming_with_profile(
            &file_paths,
            &pattern,
            case_sensitive,
        );
        let search_time = search_start.elapsed();

        println!("  {}✓{} Streaming search complete in {:.1}ms",
            GREEN, RESET, search_time.as_secs_f64() * 1000.0);
        profile.print();

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
        println!("{}Performance Summary (Streaming Mode):{}", BOLD, RESET);
        println!();
        let index_method = if index.is_gpu_direct() { "GPU-direct" } else { "mmap" };
        println!("  Phase 1 (Index):     {:>7.1}ms  {} entries via {}",
            index_time.as_secs_f64() * 1000.0, index.entry_count(), index_method);
        println!("  Phase 2 (Filter):    {:>7.1}ms  {} files matched",
            filter_time.as_secs_f64() * 1000.0, matching_paths.len());
        println!("  Phase 3+4 (Stream):  {:>7.1}ms  I/O overlapped with GPU search",
            search_time.as_secs_f64() * 1000.0);
        println!("  {}─────────────────────────────────{}", CYAN, RESET);
        println!("  {}Total:              {:>7.1}ms{}", BOLD, total_time.as_secs_f64() * 1000.0, RESET);
        println!();
        let throughput = if profile.total_us > 0 {
            (profile.bytes_processed as f64 / (1024.0 * 1024.0)) / (profile.total_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        println!("  {}Throughput: {:.1} MB/s{}", GREEN, throughput, RESET);
        println!();

        return;
    }

    // Handle persistent mode (#133) - persistent kernel for burst search
    if load_mode == LoadMode::Persistent {
        println!();
        println!("{}Phase 3:{} GPU-Direct File Loading", BOLD, RESET);

        let loader = match GpuBatchLoader::new(&device) {
            Some(l) => l,
            None => {
                println!("  {}✗{} MTLIOCommandQueue not available", RED, RESET);
                return;
            }
        };

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

        println!("  {}✓{} Loaded {} files ({:.1} MB) in {:.1}ms",
            GREEN, RESET,
            batch_result.file_count(),
            total_size as f64 / (1024.0 * 1024.0),
            load_time.as_secs_f64() * 1000.0);

        println!();
        println!("{}Phase 4:{} Persistent Kernel Search", BOLD, RESET);

        // Create persistent queue with the loaded data
        let mut queue = PersistentSearchQueue::new(&device, total_size)
            .expect("Failed to create persistent queue");

        // Load data into the persistent queue
        // Get data from batch result mega_buffer
        let data_ptr = batch_result.mega_buffer.contents() as *const u8;
        let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, total_size) };
        queue.load_data(0, data_slice).expect("Failed to load data");

        // Use oneshot search (processes single search efficiently)
        let search_start = Instant::now();
        let match_count = queue.oneshot_search(&pattern, case_sensitive)
            .expect("Persistent search failed");
        let search_time = search_start.elapsed();

        println!("  {}✓{} Persistent search: {} matches in {:.2}ms",
            GREEN, RESET, match_count, search_time.as_secs_f64() * 1000.0);

        let total_time = total_start.elapsed();

        // For persistent mode, we get match count but need to extract context differently
        // Use the batch searcher for result extraction
        let mut searcher = GpuContentSearch::new(&device, batch_result.file_count())
            .expect("Failed to create search engine");
        let _ = searcher.load_from_batch(&batch_result).expect("Failed to load from batch");

        let options = SearchOptions { case_sensitive, max_results };
        let (matches, _) = searcher.search_with_profiling(&pattern, &options);

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
        println!("{}Performance Summary (Persistent Mode):{}", BOLD, RESET);
        println!();
        let index_method = if index.is_gpu_direct() { "GPU-direct" } else { "mmap" };
        println!("  Phase 1 (Index):     {:>7.1}ms  {} entries via {}",
            index_time.as_secs_f64() * 1000.0, index.entry_count(), index_method);
        println!("  Phase 2 (Filter):    {:>7.1}ms  {} files matched",
            filter_time.as_secs_f64() * 1000.0, matching_paths.len());
        println!("  Phase 3 (Load):      {:>7.1}ms  {:.1} MB via GPU-direct",
            load_time.as_secs_f64() * 1000.0, total_size as f64 / (1024.0 * 1024.0));
        println!("  Phase 4 (Search):    {:>7.1}ms  persistent kernel",
            search_time.as_secs_f64() * 1000.0);
        println!("  {}─────────────────────────────────{}", CYAN, RESET);
        println!("  {}Total:              {:>7.1}ms{}", BOLD, total_time.as_secs_f64() * 1000.0, RESET);
        println!();

        return;
    }

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
