// GPU Filesystem Search - Interactive Terminal UI
//
// Clean terminal interface for searching your filesystem on the GPU

use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};
use metal::Device;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    clear_screen();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         🚀 GPU FILESYSTEM SEARCH - Interactive               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("🖥️  GPU: {}\n", device.name());

    // Ask user which directory to scan
    println!("Select directory to load into GPU:");
    println!("  [1] Current project ({:?})", std::env::current_dir().unwrap());
    println!("  [2] Home directory (~)");
    println!("  [3] Custom path\n");

    print!("Choice (1-3): ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();

    let scan_root = match choice.trim() {
        "2" => std::env::var("HOME").unwrap_or_else(|_| "/Users".to_string()),
        "3" => {
            print!("Enter full path: ");
            io::stdout().flush().unwrap();
            let mut path = String::new();
            io::stdin().read_line(&mut path).unwrap();
            path.trim().to_string()
        }
        _ => std::env::current_dir().unwrap().to_str().unwrap().to_string(),
    };

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📂 Loading filesystem into GPU: {}", scan_root);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    print!("Scanning... ");
    io::stdout().flush().unwrap();

    let start = Instant::now();
    let mut fs = GpuFilesystem::new(&device, 100_000).expect("Failed to create GPU filesystem");

    let mut stats = ScanStats {
        files: 0,
        dirs: 0,
        skipped: 0,
    };

    scan_directory(Path::new(&scan_root), 0, &mut fs, &mut stats, 0);

    println!("Done in {:.2}s\n", start.elapsed().as_secs_f64());

    println!("📊 Loaded into GPU:");
    println!("   • Files: {}", stats.files);
    println!("   • Directories: {}", stats.dirs);
    println!("   • Total: {}", stats.files + stats.dirs);
    println!("   • Skipped: {} (long names or permissions)", stats.skipped);
    println!("   • GPU Memory: ~{}KB", (stats.files + stats.dirs) * 64 / 1024);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔍 GPU PATH SEARCH");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Commands:");
    println!("  • Type a path to search (e.g., '/src', '/Cargo.toml')");
    println!("  • 'batch' - test batch lookup performance");
    println!("  • 'stats' - show GPU cache statistics");
    println!("  • 'help' - show this help");
    println!("  • 'quit' - exit\n");

    loop {
        print!("\n🔎 > ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "quit" | "exit" | "q" => {
                println!("\n👋 Goodbye!\n");
                break;
            }

            "stats" => {
                let cache_stats = fs.cache_stats();
                println!("\n╔══════════════════════════════════════════════════════════╗");
                println!("║                  GPU CACHE STATISTICS                    ║");
                println!("╠══════════════════════════════════════════════════════════╣");
                println!("║  Total Lookups:  {:>8}                                 ║", cache_stats.hits + cache_stats.misses);
                println!("║  Cache Hits:     {:>8}  (GPU cache ~50ns)              ║", cache_stats.hits);
                println!("║  Cache Misses:   {:>8}  (Full GPU scan)                ║", cache_stats.misses);
                println!("║  Hit Rate:       {:>7.1}%                                 ║", cache_stats.hit_rate * 100.0);
                println!("║  Cached Paths:   {:>8} / 1024                           ║", cache_stats.total_entries);
                println!("╚══════════════════════════════════════════════════════════╝");
            }

            "batch" => {
                println!("\n🚀 Testing batch GPU lookup performance...\n");

                let test_paths = vec![
                    "/src", "/docs", "/tests", "/examples",
                    "/Cargo.toml", "/README.md", "/LICENSE",
                ];

                let paths: Vec<&str> = test_paths.iter()
                    .filter(|p| p.len() <= 20)
                    .copied()
                    .collect();

                if paths.is_empty() {
                    println!("No test paths available (all too long)");
                    continue;
                }

                let start = Instant::now();
                match fs.lookup_batch(&paths) {
                    Ok(results) => {
                        let elapsed = start.elapsed();

                        println!("Results:");
                        for (path, result) in paths.iter().zip(results.iter()) {
                            match result {
                                Ok(inode) => println!("  ✅ {} → inode {}", path, inode),
                                Err(_) => println!("  ❌ {} → not found", path),
                            }
                        }

                        println!("\n⚡ Performance:");
                        println!("   • Total time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
                        println!("   • Per path: {:.1}µs", elapsed.as_micros() as f64 / paths.len() as f64);
                        println!("   • Throughput: {:.0} lookups/sec",
                                 paths.len() as f64 / elapsed.as_secs_f64());
                        println!("   • Single dispatch for {} paths (100x speedup!)", paths.len());
                    }
                    Err(e) => println!("Batch failed: {}", e),
                }
            }

            "help" => {
                println!("\n╔══════════════════════════════════════════════════════════╗");
                println!("║                         HELP                             ║");
                println!("╠══════════════════════════════════════════════════════════╣");
                println!("║  GPU Path Search commands:                               ║");
                println!("║                                                          ║");
                println!("║  <path>  - Search for a path (e.g., /src/lib.rs)         ║");
                println!("║  batch   - Benchmark batch lookup performance           ║");
                println!("║  stats   - Show GPU cache statistics                    ║");
                println!("║  help    - Show this help                               ║");
                println!("║  quit    - Exit                                          ║");
                println!("║                                                          ║");
                println!("║  Examples:                                               ║");
                println!("║    /src                                                  ║");
                println!("║    /Cargo.toml                                           ║");
                println!("║    /docs/README.md                                       ║");
                println!("╚══════════════════════════════════════════════════════════╝");
            }

            _ => {
                // Treat as path search
                let search_path = if input.starts_with('/') {
                    input.to_string()
                } else {
                    format!("/{}", input)
                };

                let start = Instant::now();
                match fs.lookup_path(&search_path) {
                    Ok(inode) => {
                        let elapsed = start.elapsed();
                        let cache_stats = fs.cache_stats();
                        let was_cached = cache_stats.hits > 0 && elapsed.as_micros() < 100;

                        println!("\n┌────────────────────────────────────────────────────┐");
                        println!("│ ✅ FOUND                                           │");
                        println!("├────────────────────────────────────────────────────┤");
                        println!("│  Path:       {}                            ", search_path);
                        println!("│  Inode:      {}                                  ", inode);
                        println!("│  Time:       {:.1}µs                             ", elapsed.as_micros());
                        if was_cached {
                            println!("│  Source:     ⚡ GPU Cache Hit (~50ns)             │");
                        } else {
                            println!("│  Source:     🔍 GPU Full Scan (1024 threads)      │");
                        }
                        println!("└────────────────────────────────────────────────────┘");
                    }
                    Err(e) => {
                        let elapsed = start.elapsed();
                        println!("\n┌────────────────────────────────────────────────────┐");
                        println!("│ ❌ NOT FOUND                                       │");
                        println!("├────────────────────────────────────────────────────┤");
                        println!("│  Path:       {}                            ", search_path);
                        println!("│  Time:       {:.1}µs                             ", elapsed.as_micros());
                        println!("│  Error:      {}                          ", e);
                        println!("└────────────────────────────────────────────────────┘");
                    }
                }
            }
        }
    }

    // Final summary
    let final_stats = fs.cache_stats();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    SESSION SUMMARY                       ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Files in GPU:    {:>8}                               ║", stats.files + stats.dirs);
    println!("║  Total searches:  {:>8}                               ║", final_stats.hits + final_stats.misses);
    println!("║  Cache hit rate:  {:>7.1}%                              ║", final_stats.hit_rate * 100.0);
    println!("║  GPU Device:      {}                    ║", device.name());
    println!("╚══════════════════════════════════════════════════════════╝\n");
}

struct ScanStats {
    files: usize,
    dirs: usize,
    skipped: usize,
}

fn scan_directory(
    path: &Path,
    parent_inode: u32,
    fs: &mut GpuFilesystem,
    stats: &mut ScanStats,
    depth: usize,
) {
    if depth > 20 {
        return;
    }

    // Show progress
    if (stats.dirs + stats.files) % 100 == 0 && stats.dirs + stats.files > 0 {
        print!("\rScanning... {} files, {} dirs", stats.files, stats.dirs);
        io::stdout().flush().unwrap();
    }

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => {
            stats.skipped += 1;
            return;
        }
    };

    for entry in entries {
        let Ok(entry) = entry else {
            stats.skipped += 1;
            continue;
        };

        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            stats.skipped += 1;
            continue;
        };

        // Skip if name too long (20 char limit)
        if name.len() > 20 {
            stats.skipped += 1;
            continue;
        }

        let Ok(metadata) = entry.metadata() else {
            stats.skipped += 1;
            continue;
        };

        let file_type = if metadata.is_dir() {
            FileType::Directory
        } else {
            FileType::Regular
        };

        match fs.add_file(parent_inode, name, file_type) {
            Ok(inode_id) => {
                if metadata.is_dir() {
                    stats.dirs += 1;
                    scan_directory(&entry.path(), inode_id, fs, stats, depth + 1);
                } else {
                    stats.files += 1;
                }
            }
            Err(_) => {
                stats.skipped += 1;
            }
        }
    }
}

fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
    io::stdout().flush().unwrap();
}
