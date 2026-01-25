// Filesystem Content Search Integration Demo
//
// Demonstrates the integrated GPU content search in GpuPathSearch.
// Uses MTLIOCommandQueue for GPU-direct I/O (3.5x faster than ripgrep).

use metal::Device;
use rust_experiment::gpu_os::filesystem::GpuPathSearch;
use rust_experiment::gpu_os::content_search::SearchOptions;
use rust_experiment::gpu_os::shared_index::GpuFilesystemIndex;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ANSI colors
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// Recursively collect file paths from a directory
fn collect_paths(dir: &Path, paths: &mut Vec<String>, max_depth: usize, current_depth: usize) {
    if current_depth > max_depth {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Skip hidden files and common noise directories
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }

        if path.is_dir() {
            paths.push(path.to_string_lossy().to_string());
            collect_paths(&path, paths, max_depth, current_depth + 1);
        } else if path.is_file() {
            paths.push(path.to_string_lossy().to_string());
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let pattern = args.get(1).map(|s| s.as_str()).unwrap_or("fn");
    let dir = args.get(2).map(|s| Path::new(s.as_str())).unwrap_or(Path::new("."));
    let extension = args.get(3).map(|s| s.as_str());

    println!();
    println!("{}┌─────────────────────────────────────────────────────────────┐{}", CYAN, RESET);
    println!("{}│{} {}Filesystem Content Search{} - GPU-Native Integration        {}│{}",
        CYAN, RESET, BOLD, RESET, CYAN, RESET);
    println!("{}└─────────────────────────────────────────────────────────────┘{}", CYAN, RESET);
    println!();

    let device = Device::system_default().expect("No Metal device");
    println!("  Device: {}{}{}", CYAN, device.name(), RESET);
    println!("  Pattern: {}\"{}\"{}  Directory: {}", YELLOW, pattern, RESET, dir.display());
    if let Some(ext) = extension {
        println!("  Extension filter: {}", ext);
    }
    println!();

    // Collect paths - try shared index first (100x faster for home directory)
    println!("{}Phase 1:{} Loading paths...", BOLD, RESET);
    let scan_start = Instant::now();
    let mut paths = Vec::new();

    // Canonicalize search directory for comparison
    let canonical_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let search_prefix = format!("{}/", canonical_dir.display());

    // Try shared GPU-resident index first (Issue #135)
    let mut used_shared_index = false;
    if let Ok(shared_idx) = GpuFilesystemIndex::load_or_create(&device) {
        if let Some(home_idx) = shared_idx.home() {
            // Filter paths under search directory
            for entry in home_idx.iter() {
                if !entry.is_dir() {
                    let entry_path = entry.path_str();
                    if entry_path.starts_with(&search_prefix) {
                        paths.push(entry_path.to_string());
                    }
                }
            }
            if !paths.is_empty() {
                used_shared_index = true;
            }
        }
    }

    // Fall back to filesystem scan if shared index didn't work
    if !used_shared_index {
        paths.clear();
        collect_paths(dir, &mut paths, 10, 0);
    }

    let scan_time = scan_start.elapsed();
    let source = if used_shared_index { "shared index" } else { "filesystem scan" };
    println!("  {}✓{} Found {} paths via {} in {:.1}ms",
        GREEN, RESET, paths.len(), source, scan_time.as_secs_f64() * 1000.0);

    if paths.is_empty() {
        println!("  {}No files found!{}", YELLOW, RESET);
        return;
    }

    // Create GPU path search and load paths
    println!();
    println!("{}Phase 2:{} Loading paths into GPU...", BOLD, RESET);
    let mut gpu_search = GpuPathSearch::new(&device, paths.len() + 1000)
        .expect("Failed to create GPU path search");

    let load_start = Instant::now();
    gpu_search.add_paths(&paths).expect("Failed to add paths");
    let load_time = load_start.elapsed();
    println!("  {}✓{} Loaded {} paths into GPU in {:.1}ms",
        GREEN, RESET, gpu_search.path_count(), load_time.as_secs_f64() * 1000.0);

    // Search content
    println!();
    println!("{}Phase 3:{} GPU Content Search (MTLIOCommandQueue)...", BOLD, RESET);

    let search_start = Instant::now();
    let options = SearchOptions {
        case_sensitive: false,
        max_results: 100,
    };
    let matches = gpu_search.search_content(pattern, extension, &options);
    let search_time = search_start.elapsed();

    println!("  {}✓{} Found {} matches in {:.1}ms",
        GREEN, RESET, matches.len(), search_time.as_secs_f64() * 1000.0);

    // Display results
    println!();
    println!("{}─────────────────────────────────────────────────────────────{}", CYAN, RESET);
    println!("{}Results:{}", BOLD, RESET);
    println!();

    for m in matches.iter().take(50) {
        // Highlight the match
        let context = m.context.trim();
        let highlighted = if let Some(pos) = context.to_lowercase().find(&pattern.to_lowercase()) {
            let before = &context[..pos];
            let matched = &context[pos..pos + pattern.len()];
            let after = &context[pos + pattern.len()..];
            format!("{}\x1b[31m\x1b[1m{}\x1b[0m{}", before, matched, after)
        } else {
            context.to_string()
        };

        println!("{}{}{}:{}{}{}:{}",
            MAGENTA, m.file_path, RESET,
            GREEN, m.line_number, RESET,
            highlighted);
    }

    // Summary
    println!();
    println!("{}─────────────────────────────────────────────────────────────{}", CYAN, RESET);
    println!("{}Summary:{}", BOLD, RESET);
    println!("  Filesystem scan: {:>7.1}ms  ({} paths)",
        scan_time.as_secs_f64() * 1000.0, paths.len());
    println!("  GPU path load:   {:>7.1}ms",
        load_time.as_secs_f64() * 1000.0);
    println!("  Content search:  {:>7.1}ms  ({} matches)",
        search_time.as_secs_f64() * 1000.0, matches.len());
    println!("  {}Total:           {:>7.1}ms{}",
        BOLD, (scan_time + load_time + search_time).as_secs_f64() * 1000.0, RESET);
    println!();
}
