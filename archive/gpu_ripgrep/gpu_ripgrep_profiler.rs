// GPU Ripgrep Profiler - Comprehensive Benchmark Suite
//
// Compares GPU-accelerated search against native ripgrep across multiple scenarios.
// Tests ALL THREE I/O methods: CPU, mmap, and GPU-Direct (MTLIOCommandQueue).
//
// Usage: cargo run --release --example gpu_ripgrep_profiler [path]

use metal::*;
use rust_experiment::gpu_os::content_search::{GpuContentSearch, SearchOptions};
use rust_experiment::gpu_os::mmap_buffer::MmapBuffer;
use rust_experiment::gpu_os::batch_io::GpuBatchLoader;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// ANSI Colors
// ============================================================================

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

// ============================================================================
// Configuration
// ============================================================================

const WARMUP_RUNS: usize = 1;
const BENCHMARK_RUNS: usize = 3;

const TEST_PATTERNS: &[(&str, &str)] = &[
    ("fn ", "Common keyword"),
    ("impl ", "Rust impl blocks"),
    ("TODO", "Common marker"),
    ("use std::", "Import pattern"),
    ("pub struct", "Type definition"),
    ("Result<", "Error handling"),
    ("async fn", "Async functions"),
    ("unsafe", "Unsafe code"),
];

const EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "c", "cpp", "h", "go", "java",
    "md", "txt", "json", "yaml", "toml", "html", "css",
];

const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "build", "dist", ".next",
    "__pycache__", ".venv", "venv", ".cargo", ".rustup", "vendor",
];

// ============================================================================
// I/O Method Enum
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum IoMethod {
    CpuRead,      // fs::read → CPU → GPU copy (slowest)
    Mmap,         // mmap → unified memory (zero CPU copies)
    GpuDirect,    // MTLIOCommandQueue (TRUE GPU-DIRECT I/O)
}

impl IoMethod {
    fn name(&self) -> &'static str {
        match self {
            IoMethod::CpuRead => "CPU (fs::read)",
            IoMethod::Mmap => "mmap (zero-copy)",
            IoMethod::GpuDirect => "GPU-Direct (MTLIOCommandQueue)",
        }
    }

    fn short_name(&self) -> &'static str {
        match self {
            IoMethod::CpuRead => "cpu",
            IoMethod::Mmap => "mmap",
            IoMethod::GpuDirect => "gpu_direct",
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    pattern: String,
    io_method: String,
    file_count: usize,
    total_bytes: u64,
    chunk_count: usize,
    gpu_load_time: f64,
    gpu_search_time: f64,
    gpu_total_time: f64,
    gpu_matches: usize,
    rg_time: f64,
    rg_matches: usize,
    speedup: f64,
    search_speedup: f64,
    throughput_mbs: f64,
}

#[derive(Debug)]
struct ProfileStats {
    min: f64,
    max: f64,
    mean: f64,
    stddev: f64,
}

impl ProfileStats {
    fn from_samples(samples: &[f64]) -> Self {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / n as f64;

        let variance: f64 = sorted.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;

        Self {
            min: sorted[0],
            max: sorted[n - 1],
            mean,
            stddev: variance.sqrt(),
        }
    }
}

// ============================================================================
// Profiler
// ============================================================================

struct Profiler {
    device: Device,
    search_path: PathBuf,
    files: Vec<PathBuf>,
    total_bytes: u64,
    results: Vec<BenchmarkResult>,
    has_ripgrep: bool,
    has_gpu_direct: bool,
}

impl Profiler {
    fn new(path: &Path) -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        let has_ripgrep = Command::new("rg")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        // Check if GPU-Direct I/O is available
        let has_gpu_direct = GpuBatchLoader::new(&device).is_some();

        Ok(Self {
            device,
            search_path: path.to_path_buf(),
            files: Vec::new(),
            total_bytes: 0,
            results: Vec::new(),
            has_ripgrep,
            has_gpu_direct,
        })
    }

    fn collect_files(&mut self) {
        self.files.clear();
        self.total_bytes = 0;
        self.collect_recursive(&self.search_path.clone());

        for file in &self.files {
            if let Ok(meta) = fs::metadata(file) {
                self.total_bytes += meta.len();
            }
        }
    }

    fn collect_recursive(&mut self, dir: &Path) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                let name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                if !name.starts_with('.') && !SKIP_DIRS.contains(&name) {
                    self.collect_recursive(&path);
                }
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if EXTENSIONS.contains(&ext) {
                        self.files.push(path);
                    }
                }
            }
        }
    }

    fn run_ripgrep(&self, pattern: &str) -> (Duration, usize) {
        if !self.has_ripgrep {
            return (Duration::ZERO, 0);
        }

        let start = Instant::now();

        let output = Command::new("rg")
            .arg("--count-matches")
            .arg("--no-heading")
            .arg("-i")
            .arg(pattern)
            .arg(&self.search_path)
            .output();

        let elapsed = start.elapsed();

        let match_count = output
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .filter_map(|line| {
                        line.rsplit(':').next()
                            .and_then(|n| n.trim().parse::<usize>().ok())
                    })
                    .sum()
            })
            .unwrap_or(0);

        (elapsed, match_count)
    }

    // ========================================================================
    // I/O Method 1: CPU (fs::read)
    // ========================================================================

    fn run_gpu_search_cpu(&self, pattern: &str) -> (Duration, Duration, usize, usize) {
        let paths: Vec<&Path> = self.files.iter().map(|p| p.as_path()).collect();

        let mut searcher = GpuContentSearch::new(&self.device, self.files.len())
            .expect("Failed to create search engine");

        let load_start = Instant::now();
        let chunks = searcher.load_files(&paths).unwrap_or(0);
        let load_time = load_start.elapsed();

        let search_start = Instant::now();
        let options = SearchOptions {
            case_sensitive: false,
            max_results: 100000,
        };
        let matches = searcher.search(pattern, &options);
        let search_time = search_start.elapsed();

        (load_time, search_time, matches.len(), chunks)
    }

    // ========================================================================
    // I/O Method 2: mmap (zero-copy)
    // ========================================================================

    fn load_files_mmap(&self) -> Vec<(String, MmapBuffer)> {
        let max_bytes = 500 * 1024 * 1024; // 500MB max
        let total_bytes = AtomicUsize::new(0);

        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        let chunk_size = (self.files.len() + num_threads - 1) / num_threads;
        let device_clone = self.device.clone();

        let handles: Vec<_> = self.files
            .chunks(chunk_size)
            .map(|chunk| {
                let chunk_paths: Vec<PathBuf> = chunk.to_vec();
                let device = device_clone.clone();
                let total_ref = &total_bytes as *const AtomicUsize as usize;

                thread::spawn(move || {
                    let total_bytes = unsafe { &*(total_ref as *const AtomicUsize) };
                    let mut local_buffers = Vec::new();

                    for path in chunk_paths {
                        if total_bytes.load(Ordering::Relaxed) >= max_bytes {
                            break;
                        }

                        let metadata = match fs::metadata(&path) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };

                        let size = metadata.len() as usize;
                        if size == 0 || size > 10 * 1024 * 1024 {
                            continue;
                        }

                        match MmapBuffer::from_file(&device, &path) {
                            Ok(buffer) => {
                                total_bytes.fetch_add(size, Ordering::Relaxed);
                                local_buffers.push((path.to_string_lossy().to_string(), buffer));
                            }
                            Err(_) => continue,
                        }
                    }

                    local_buffers
                })
            })
            .collect();

        let mut buffers = Vec::new();
        for handle in handles {
            if let Ok(mut local) = handle.join() {
                buffers.append(&mut local);
            }
        }

        buffers
    }

    fn run_gpu_search_mmap(&self, pattern: &str) -> (Duration, Duration, usize, usize) {
        let load_start = Instant::now();
        let mmap_buffers = self.load_files_mmap();
        let load_time = load_start.elapsed();

        let mut searcher = GpuContentSearch::new(&self.device, mmap_buffers.len())
            .expect("Failed to create search engine");

        let buffer_refs: Vec<(String, &MmapBuffer)> = mmap_buffers.iter()
            .map(|(p, b)| (p.clone(), b))
            .collect();

        let blit_start = Instant::now();
        let chunks = searcher.load_from_mmap(&buffer_refs).unwrap_or(0);
        let blit_time = blit_start.elapsed();

        let search_start = Instant::now();
        let options = SearchOptions {
            case_sensitive: false,
            max_results: 100000,
        };
        let matches = searcher.search(pattern, &options);
        let search_time = search_start.elapsed();

        (load_time + blit_time, search_time, matches.len(), chunks)
    }

    // ========================================================================
    // I/O Method 3: GPU-Direct (MTLIOCommandQueue)
    // ========================================================================

    fn run_gpu_search_direct(&self, pattern: &str) -> Option<(Duration, Duration, usize, usize)> {
        let loader = GpuBatchLoader::new(&self.device)?;

        let load_start = Instant::now();
        let batch_result = loader.load_batch(&self.files)?;
        let load_time = load_start.elapsed();

        let mut searcher = GpuContentSearch::new(&self.device, batch_result.file_count())
            .expect("Failed to create search engine");

        let blit_start = Instant::now();
        let chunks = searcher.load_from_batch(&batch_result).ok()?;
        let blit_time = blit_start.elapsed();

        let search_start = Instant::now();
        let options = SearchOptions {
            case_sensitive: false,
            max_results: 100000,
        };
        let matches = searcher.search(pattern, &options);
        let search_time = search_start.elapsed();

        Some((load_time + blit_time, search_time, matches.len(), chunks))
    }

    // ========================================================================
    // Benchmark Runner
    // ========================================================================

    fn benchmark_pattern_with_method(&mut self, pattern: &str, description: &str, method: IoMethod) {
        println!("\n  {}[{}]{} Testing \"{}{}{}\"",
            DIM, method.name(), RESET, CYAN, pattern, RESET);

        let mut load_times = Vec::new();
        let mut search_times = Vec::new();
        let mut match_counts = Vec::new();
        let mut chunk_counts = Vec::new();

        // Warmup
        print!("    Warmup: ");
        for _ in 0..WARMUP_RUNS {
            print!(".");
            std::io::stdout().flush().ok();
            match method {
                IoMethod::CpuRead => { let _ = self.run_gpu_search_cpu(pattern); }
                IoMethod::Mmap => { let _ = self.run_gpu_search_mmap(pattern); }
                IoMethod::GpuDirect => { let _ = self.run_gpu_search_direct(pattern); }
            }
        }
        println!(" done");

        // Benchmark runs
        print!("    Benchmark: ");
        for _ in 0..BENCHMARK_RUNS {
            print!(".");
            std::io::stdout().flush().ok();

            let result = match method {
                IoMethod::CpuRead => Some(self.run_gpu_search_cpu(pattern)),
                IoMethod::Mmap => Some(self.run_gpu_search_mmap(pattern)),
                IoMethod::GpuDirect => self.run_gpu_search_direct(pattern),
            };

            if let Some((load, search, matches, chunks)) = result {
                load_times.push(load.as_secs_f64() * 1000.0);
                search_times.push(search.as_secs_f64() * 1000.0);
                match_counts.push(matches);
                chunk_counts.push(chunks);
            }
        }
        println!(" done");

        if load_times.is_empty() {
            println!("    {}Method not available{}", RED, RESET);
            return;
        }

        let load_stats = ProfileStats::from_samples(&load_times);
        let search_stats = ProfileStats::from_samples(&search_times);
        let total_times: Vec<f64> = (0..load_times.len())
            .map(|i| load_times[i] + search_times[i])
            .collect();
        let total_stats = ProfileStats::from_samples(&total_times);

        let avg_matches = match_counts.iter().sum::<usize>() / match_counts.len();
        let avg_chunks = chunk_counts.iter().sum::<usize>() / chunk_counts.len();

        // Get ripgrep baseline (only once per pattern)
        let (rg_time, rg_matches) = if self.has_ripgrep {
            let (time, matches) = self.run_ripgrep(pattern);
            (time.as_secs_f64() * 1000.0, matches)
        } else {
            (0.0, 0)
        };

        let speedup = if rg_time > 0.0 { rg_time / total_stats.mean } else { 0.0 };
        let search_speedup = if rg_time > 0.0 { rg_time / search_stats.mean } else { 0.0 };

        let total_mb = (avg_chunks * 4096) as f64 / (1024.0 * 1024.0);
        let throughput_mbs = if search_stats.mean > 0.0 {
            total_mb / (search_stats.mean / 1000.0)
        } else {
            0.0
        };

        println!("    Load:   {:>7.2}ms (±{:.2})", load_stats.mean, load_stats.stddev);
        println!("    Search: {:>7.2}ms (±{:.2})", search_stats.mean, search_stats.stddev);
        println!("    {}Total:  {:>7.2}ms{}", BOLD, total_stats.mean, RESET);
        println!("    Matches: {}, Throughput: {}{:.0} MB/s{}", avg_matches, GREEN, throughput_mbs, RESET);

        if self.has_ripgrep {
            if speedup >= 1.0 {
                println!("    vs rg: {}{:.2}x faster{} (search: {:.2}x)", GREEN, speedup, RESET, search_speedup);
            } else {
                println!("    vs rg: {}{:.2}x slower{} (search: {:.2}x)", RED, 1.0/speedup, RESET, search_speedup);
            }
        }

        self.results.push(BenchmarkResult {
            name: description.to_string(),
            pattern: pattern.to_string(),
            io_method: method.short_name().to_string(),
            file_count: self.files.len(),
            total_bytes: self.total_bytes,
            chunk_count: avg_chunks,
            gpu_load_time: load_stats.mean,
            gpu_search_time: search_stats.mean,
            gpu_total_time: total_stats.mean,
            gpu_matches: avg_matches,
            rg_time,
            rg_matches,
            speedup,
            search_speedup,
            throughput_mbs,
        });
    }

    fn benchmark_all_methods(&mut self) {
        println!("\n{}═══════════════════════════════════════════════════════════════{}", CYAN, RESET);
        println!("{}  I/O METHOD COMPARISON (all methods, same patterns){}", BOLD, RESET);
        println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, RESET);

        let methods = if self.has_gpu_direct {
            vec![IoMethod::CpuRead, IoMethod::Mmap, IoMethod::GpuDirect]
        } else {
            println!("\n{}Note:{} GPU-Direct I/O (MTLIOCommandQueue) not available on this device", YELLOW, RESET);
            vec![IoMethod::CpuRead, IoMethod::Mmap]
        };

        // Test with key patterns - one common, one rare
        let test_patterns = &[
            ("async fn", "Rare (few matches)"),
            ("fn ", "Common (many matches)"),
        ];

        for (pattern, description) in test_patterns.iter() {
            println!("\n{}Pattern: \"{}\" - {}{}", BOLD, pattern, description, RESET);

            for &method in &methods {
                self.benchmark_pattern_with_method(pattern, description, method);
            }
        }
    }

    fn generate_report(&self) {
        println!("\n{}═══════════════════════════════════════════════════════════════{}", CYAN, RESET);
        println!("{}  FINAL REPORT{}", BOLD, RESET);
        println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, RESET);

        if self.results.is_empty() {
            println!("No benchmark results to report.");
            return;
        }

        println!("\n{}Device:{} {}", BOLD, RESET, self.device.name());
        println!("{}Files:{} {} ({:.1} MB)", BOLD, RESET, self.files.len(), self.total_bytes as f64 / (1024.0 * 1024.0));
        println!("{}GPU-Direct:{} {}", BOLD, RESET, if self.has_gpu_direct { "Available" } else { "Not available" });

        // Group by I/O method
        println!("\n{}Performance by I/O Method:{}", BOLD, RESET);

        for method in &["cpu", "mmap", "gpu_direct"] {
            let method_results: Vec<_> = self.results.iter()
                .filter(|r| r.io_method == *method)
                .collect();

            if method_results.is_empty() {
                continue;
            }

            let avg_speedup: f64 = method_results.iter()
                .filter(|r| r.speedup > 0.0)
                .map(|r| r.speedup)
                .sum::<f64>() / method_results.iter().filter(|r| r.speedup > 0.0).count().max(1) as f64;

            let avg_load: f64 = method_results.iter().map(|r| r.gpu_load_time).sum::<f64>() / method_results.len() as f64;
            let avg_search: f64 = method_results.iter().map(|r| r.gpu_search_time).sum::<f64>() / method_results.len() as f64;
            let avg_throughput: f64 = method_results.iter().map(|r| r.throughput_mbs).sum::<f64>() / method_results.len() as f64;

            let method_name = match *method {
                "cpu" => "CPU (fs::read)",
                "mmap" => "mmap (zero-copy)",
                "gpu_direct" => "GPU-Direct (MTLIOCommandQueue)",
                _ => method,
            };

            println!("\n  {}{}:{}", YELLOW, method_name, RESET);
            println!("    Avg Load:   {:>7.2}ms", avg_load);
            println!("    Avg Search: {:>7.2}ms", avg_search);
            println!("    Avg Total:  {:>7.2}ms", avg_load + avg_search);
            println!("    Throughput: {:.0} MB/s", avg_throughput);

            if self.has_ripgrep {
                if avg_speedup >= 1.0 {
                    println!("    vs ripgrep: {}{:.2}x faster{}", GREEN, avg_speedup, RESET);
                } else {
                    println!("    vs ripgrep: {}{:.2}x slower{}", RED, 1.0/avg_speedup, RESET);
                }
            }
        }

        // Ship decision
        println!("\n{}═══════════════════════════════════════════════════════════════{}", CYAN, RESET);
        println!("{}  SHIP DECISION{}", BOLD, RESET);
        println!("{}═══════════════════════════════════════════════════════════════{}", CYAN, RESET);

        // Find best method
        let best_method = self.results.iter()
            .filter(|r| r.speedup > 0.0)
            .max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap());

        if let Some(best) = best_method {
            if best.speedup >= 1.5 {
                println!("\n{}✓ SHIP IT!{} Best method ({}) achieves {:.2}x speedup vs ripgrep",
                    GREEN, RESET, best.io_method, best.speedup);
            } else if best.speedup >= 1.0 {
                println!("\n{}◐ COMPETITIVE{} Best method ({}) achieves {:.2}x speedup",
                    YELLOW, RESET, best.io_method, best.speedup);
            } else {
                println!("\n{}✗ NEEDS WORK{} Even best method ({}) is {:.2}x slower",
                    RED, RESET, best.io_method, 1.0/best.speedup);
            }
        }
    }

    fn export_csv(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "pattern,description,io_method,files,bytes,chunks,load_ms,search_ms,total_ms,matches,rg_ms,rg_matches,speedup,search_speedup,throughput_mbs")?;

        for r in &self.results {
            writeln!(writer, "\"{}\",\"{}\",{},{},{},{},{:.2},{:.2},{:.2},{},{:.2},{},{:.2},{:.2},{:.0}",
                r.pattern, r.name, r.io_method, r.file_count, r.total_bytes, r.chunk_count,
                r.gpu_load_time, r.gpu_search_time, r.gpu_total_time,
                r.gpu_matches, r.rg_time, r.rg_matches, r.speedup, r.search_speedup,
                r.throughput_mbs)?;
        }

        Ok(())
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("{}╔═══════════════════════════════════════════════════════════════╗{}", CYAN, RESET);
    println!("{}║     GPU RIPGREP PROFILER - All I/O Methods Benchmark          ║{}", CYAN, RESET);
    println!("{}╚═══════════════════════════════════════════════════════════════╝{}", CYAN, RESET);

    let args: Vec<String> = env::args().collect();
    let search_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(".")
    };

    println!("\n{}Configuration:{}", BOLD, RESET);
    println!("  Search path: {}", search_path.display());
    println!("  Warmup runs: {}", WARMUP_RUNS);
    println!("  Benchmark runs: {}", BENCHMARK_RUNS);

    let mut profiler = match Profiler::new(&search_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}Error:{} {}", RED, RESET, e);
            std::process::exit(1);
        }
    };

    println!("  GPU device: {}", profiler.device.name());
    println!("  Ripgrep available: {}", if profiler.has_ripgrep { "yes" } else { "no" });
    println!("  GPU-Direct I/O: {}", if profiler.has_gpu_direct { "yes" } else { "no" });

    // Collect files
    println!("\n{}Collecting files...{}", DIM, RESET);
    let start = Instant::now();
    profiler.collect_files();
    let collect_time = start.elapsed();

    println!("  Found {} files ({:.1} MB) in {:.1}ms",
        profiler.files.len(),
        profiler.total_bytes as f64 / (1024.0 * 1024.0),
        collect_time.as_secs_f64() * 1000.0);

    if profiler.files.is_empty() {
        eprintln!("{}Error:{} No source files found in {}", RED, RESET, search_path.display());
        std::process::exit(1);
    }

    // Run benchmarks comparing all I/O methods
    profiler.benchmark_all_methods();

    // Generate report
    profiler.generate_report();

    // Export CSV
    let csv_path = "gpu_ripgrep_benchmark.csv";
    match profiler.export_csv(csv_path) {
        Ok(_) => println!("\n{}Benchmark data exported to:{} {}", DIM, RESET, csv_path),
        Err(e) => eprintln!("\n{}Warning:{} Failed to export CSV: {}", YELLOW, RESET, e),
    }

    println!();
}
