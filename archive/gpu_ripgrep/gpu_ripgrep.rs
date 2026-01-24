// GPU Ripgrep - Massively Parallel Code Search
//
// THE GPU IS THE COMPUTER. Search entire codebases in milliseconds.
//
// Usage: cargo run --release --example gpu_ripgrep -- <pattern> [path]
//
// Features:
// - GPU-parallel search across all files simultaneously
// - Boyer-Moore-Horspool algorithm on GPU
// - Case-insensitive by default (-s for case-sensitive)
// - File type filtering (--type rs, --type py, etc.)
// - Context lines (-A, -B, -C)

use metal::*;
use rust_experiment::gpu_os::content_search::{GpuContentSearch, SearchOptions, ContentMatch};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ANSI color codes
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// File extensions to search by default (source code)
const DEFAULT_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "c", "cpp", "h", "hpp",
    "go", "java", "rb", "php", "swift", "kt", "scala", "cs",
    "html", "css", "scss", "less", "json", "yaml", "yml", "toml",
    "md", "txt", "sh", "bash", "zsh", "fish", "sql", "graphql",
    "xml", "vue", "svelte", "astro", "metal", "glsl", "wgsl",
];

/// Directories to skip
const SKIP_DIRS: &[&str] = &[
    ".git", "node_modules", "target", "build", "dist", ".next",
    "__pycache__", ".venv", "venv", ".cargo", ".rustup",
    "vendor", "deps", "_build", ".build", "Pods", ".gradle",
];

struct SearchConfig {
    pattern: String,
    path: PathBuf,
    case_sensitive: bool,
    max_results: usize,
    file_types: Vec<String>,
    context_before: usize,
    context_after: usize,
    count_only: bool,
    files_with_matches: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            pattern: String::new(),
            path: PathBuf::from("."),
            case_sensitive: false,
            max_results: 10000,
            file_types: Vec::new(),
            context_before: 0,
            context_after: 0,
            count_only: false,
            files_with_matches: false,
        }
    }
}

fn parse_args() -> Result<SearchConfig, String> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return Err(format!(
            "{}GPU Ripgrep{} - Massively Parallel Code Search\n\n\
            {}Usage:{} {} <pattern> [path] [options]\n\n\
            {}Options:{}\n  \
            -s, --case-sensitive    Case-sensitive search\n  \
            -t, --type <ext>        Only search files with extension\n  \
            -c, --count             Only show match count\n  \
            -l, --files-with-matches Only show filenames\n  \
            -m, --max-count <n>     Maximum results (default: 10000)\n  \
            -A <n>                  Show n lines after match\n  \
            -B <n>                  Show n lines before match\n  \
            -C <n>                  Show n lines before and after\n\n\
            {}Examples:{}\n  \
            {} \"fn main\" src/\n  \
            {} -t rs \"impl.*Display\"\n  \
            {} -s \"TODO\" .",
            BOLD, RESET,
            YELLOW, RESET, args[0],
            YELLOW, RESET,
            YELLOW, RESET,
            args[0], args[0], args[0]
        ));
    }

    let mut config = SearchConfig::default();
    let mut i = 1;

    while i < args.len() {
        let arg = &args[i];

        match arg.as_str() {
            "-s" | "--case-sensitive" => config.case_sensitive = true,
            "-c" | "--count" => config.count_only = true,
            "-l" | "--files-with-matches" => config.files_with_matches = true,
            "-t" | "--type" => {
                i += 1;
                if i < args.len() {
                    config.file_types.push(args[i].clone());
                }
            }
            "-m" | "--max-count" => {
                i += 1;
                if i < args.len() {
                    config.max_results = args[i].parse().unwrap_or(10000);
                }
            }
            "-A" => {
                i += 1;
                if i < args.len() {
                    config.context_after = args[i].parse().unwrap_or(0);
                }
            }
            "-B" => {
                i += 1;
                if i < args.len() {
                    config.context_before = args[i].parse().unwrap_or(0);
                }
            }
            "-C" => {
                i += 1;
                if i < args.len() {
                    let ctx = args[i].parse().unwrap_or(0);
                    config.context_before = ctx;
                    config.context_after = ctx;
                }
            }
            _ => {
                if config.pattern.is_empty() {
                    config.pattern = arg.clone();
                } else if config.path == PathBuf::from(".") {
                    config.path = PathBuf::from(arg);
                }
            }
        }
        i += 1;
    }

    if config.pattern.is_empty() {
        return Err("No pattern specified".to_string());
    }

    Ok(config)
}

fn should_search_file(path: &Path, config: &SearchConfig) -> bool {
    // Check extension
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if !config.file_types.is_empty() {
        // User specified types - only search those
        if !config.file_types.iter().any(|t| t == ext) {
            return false;
        }
    } else {
        // Default: search known source code extensions
        if !DEFAULT_EXTENSIONS.contains(&ext) {
            return false;
        }
    }

    true
}

fn collect_files(dir: &Path, config: &SearchConfig) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files_recursive(dir, config, &mut files);
    files
}

fn collect_files_recursive(dir: &Path, config: &SearchConfig, files: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Skip hidden and known large directories
            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if name.starts_with('.') || SKIP_DIRS.contains(&name) {
                continue;
            }

            collect_files_recursive(&path, config, files);
        } else if path.is_file() && should_search_file(&path, config) {
            files.push(path);
        }
    }
}

fn highlight_match(line: &str, pattern: &str, case_sensitive: bool) -> String {
    if case_sensitive {
        line.replace(pattern, &format!("{}{}{}{}", RED, BOLD, pattern, RESET))
    } else {
        // Case-insensitive highlighting
        let lower_line = line.to_lowercase();
        let lower_pattern = pattern.to_lowercase();

        let mut result = String::new();
        let mut last_end = 0;

        for (start, _) in lower_line.match_indices(&lower_pattern) {
            result.push_str(&line[last_end..start]);
            result.push_str(RED);
            result.push_str(BOLD);
            result.push_str(&line[start..start + pattern.len()]);
            result.push_str(RESET);
            last_end = start + pattern.len();
        }
        result.push_str(&line[last_end..]);
        result
    }
}

fn print_match(m: &ContentMatch, config: &SearchConfig) {
    let highlighted = highlight_match(&m.context, &config.pattern, config.case_sensitive);

    println!(
        "{}{}{}:{}{}{}:{}",
        MAGENTA, m.file_path, RESET,
        GREEN, m.line_number, RESET,
        highlighted
    );
}

fn main() {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // Initialize Metal
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("{}Error:{} No Metal device found", RED, RESET);
            std::process::exit(1);
        }
    };

    eprintln!(
        "{}GPU Ripgrep{} using {}{}{}",
        BOLD, RESET, CYAN, device.name(), RESET
    );

    // Collect files
    let collect_start = Instant::now();
    let files = collect_files(&config.path, &config);
    let collect_time = collect_start.elapsed();

    if files.is_empty() {
        eprintln!("{}No files to search{}", YELLOW, RESET);
        std::process::exit(0);
    }

    eprintln!(
        "  Found {} files in {:.1}ms",
        files.len(),
        collect_time.as_secs_f64() * 1000.0
    );

    // Create GPU search engine
    let mut searcher = match GpuContentSearch::new(&device, files.len()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}Error:{} Failed to create search engine: {}", RED, RESET, e);
            std::process::exit(1);
        }
    };

    // Load files to GPU
    let load_start = Instant::now();
    let paths: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();
    let chunks = match searcher.load_files(&paths) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}Error:{} Failed to load files: {}", RED, RESET, e);
            std::process::exit(1);
        }
    };
    let load_time = load_start.elapsed();

    eprintln!(
        "  Loaded {} chunks ({:.1} MB) in {:.1}ms",
        chunks,
        (chunks * 4096) as f64 / (1024.0 * 1024.0),
        load_time.as_secs_f64() * 1000.0
    );

    // Search!
    let search_start = Instant::now();
    let options = SearchOptions {
        case_sensitive: config.case_sensitive,
        max_results: config.max_results,
    };
    let matches = searcher.search(&config.pattern, &options);
    let search_time = search_start.elapsed();

    eprintln!(
        "  {}GPU search: {:.2}ms{} ({} matches)\n",
        GREEN,
        search_time.as_secs_f64() * 1000.0,
        RESET,
        matches.len()
    );

    // Output results
    if config.count_only {
        println!("{}", matches.len());
    } else if config.files_with_matches {
        let mut seen_files: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for m in &matches {
            if seen_files.insert(&m.file_path) {
                println!("{}{}{}", MAGENTA, m.file_path, RESET);
            }
        }
    } else {
        for m in matches.iter().take(config.max_results) {
            print_match(m, &config);
        }
    }

    // Summary
    let total_time = collect_time + load_time + search_time;
    eprintln!(
        "\n{}Total:{} {:.2}ms (collect: {:.1}ms, load: {:.1}ms, search: {:.2}ms)",
        BOLD, RESET,
        total_time.as_secs_f64() * 1000.0,
        collect_time.as_secs_f64() * 1000.0,
        load_time.as_secs_f64() * 1000.0,
        search_time.as_secs_f64() * 1000.0
    );
}
