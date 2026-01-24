// GPU Ripgrep Benchmark - Measure repeated search performance
//
// This demonstrates where GPU shines: once files are loaded,
// searches are extremely fast.

use metal::*;
use rust_experiment::gpu_os::content_search::{GpuContentSearch, SearchOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::fs;

const DEFAULT_EXTENSIONS: &[&str] = &["rs", "py", "js", "ts", "c", "cpp", "go", "java", "md", "txt", "json"];
const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "build", ".cargo"];

fn collect_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_recursive(dir, &mut files, 0);
    files
}

fn collect_recursive(dir: &Path, files: &mut Vec<PathBuf>, depth: usize) {
    if depth > 10 { return; } // Limit depth

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') && !SKIP_DIRS.contains(&name) {
                collect_recursive(&path, files, depth + 1);
            }
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if DEFAULT_EXTENSIONS.contains(&ext) {
                files.push(path);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(|s| s.as_str()).unwrap_or(".");

    let device = Device::system_default().expect("No Metal device");
    println!("\x1b[1mGPU Ripgrep Benchmark\x1b[0m using \x1b[36m{}\x1b[0m\n", device.name());

    // Phase 1: Collect files
    let collect_start = Instant::now();
    let files = collect_files(Path::new(dir));
    let collect_time = collect_start.elapsed();
    println!("  Collected {} files in {:.1}ms", files.len(), collect_time.as_secs_f64() * 1000.0);

    if files.is_empty() {
        println!("No files found!");
        return;
    }

    // Phase 2: Load to GPU
    let mut searcher = GpuContentSearch::new(&device, files.len().min(50000)).expect("Failed to create searcher");

    let load_start = Instant::now();
    let paths: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    let chunks = searcher.load_files(&paths).expect("Failed to load files");
    let load_time = load_start.elapsed();

    let mb = (chunks * 4096) as f64 / (1024.0 * 1024.0);
    println!("  Loaded {} chunks ({:.1} MB) in {:.1}ms", chunks, mb, load_time.as_secs_f64() * 1000.0);
    println!();

    // Phase 3: Multiple searches (this is where GPU shines!)
    let patterns = ["fn ", "impl ", "pub ", "use ", "let ", "const ", "struct ", "enum "];
    let options = SearchOptions { case_sensitive: false, max_results: 1000 };

    println!("\x1b[1m  Repeated searches (files already loaded):\x1b[0m");
    println!("  {:-<50}", "");

    let mut total_search_time = 0.0;

    for pattern in patterns {
        let search_start = Instant::now();
        let matches = searcher.search(pattern, &options);
        let search_time = search_start.elapsed().as_secs_f64() * 1000.0;
        total_search_time += search_time;

        println!("  {:12} {:>6} matches in \x1b[32m{:>6.2}ms\x1b[0m",
            format!("\"{}\"", pattern), matches.len(), search_time);
    }

    println!("  {:-<50}", "");
    println!("  \x1b[1mTotal:\x1b[0m {} searches in \x1b[32m{:.2}ms\x1b[0m ({:.2}ms avg)",
        patterns.len(), total_search_time, total_search_time / patterns.len() as f64);
    println!();

    // Summary
    let total_cold = collect_time.as_secs_f64() * 1000.0 + load_time.as_secs_f64() * 1000.0;
    println!("  \x1b[33mCold start:\x1b[0m {:.1}ms (collect + load)", total_cold);
    println!("  \x1b[32mWarm search:\x1b[0m {:.2}ms average per pattern", total_search_time / patterns.len() as f64);
    println!();
    println!("  After cold start, GPU searches {:.1} MB in ~{:.0}ms", mb, total_search_time / patterns.len() as f64);
}
