// Load Real Filesystem into GPU
//
// Recursively scans root directory and loads into GPU-native filesystem
// Then allows searching for any file using GPU path lookup + cache

use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};
use metal::Device;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║     GPU-Native Filesystem: Real System Import            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("🖥️  GPU Device: {}\n", device.name());

    // Create GPU filesystem with large capacity
    let max_inodes = 100_000; // Support up to 100k files
    println!("📁 Initializing GPU filesystem (capacity: {} inodes)...", max_inodes);
    let mut fs = GpuFilesystem::new(&device, max_inodes).expect("Failed to create filesystem");

    // Track inode mapping: real path -> GPU inode
    let mut path_to_inode: HashMap<String, u32> = HashMap::new();
    path_to_inode.insert("/".to_string(), 0); // Root

    println!("✅ GPU filesystem initialized\n");

    // Ask user which directory to scan
    println!("Which directory do you want to scan?");
    println!("  1. Root (/) - entire system");
    println!("  2. Home directory (~)");
    println!("  3. Current project ({:?})", std::env::current_dir().unwrap());
    println!("  4. Custom path");
    print!("\nChoice (1-4): ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();

    let scan_root = match choice.trim() {
        "1" => "/".to_string(),
        "2" => std::env::var("HOME").unwrap_or_else(|_| "/Users".to_string()),
        "3" => std::env::current_dir().unwrap().to_str().unwrap().to_string(),
        "4" => {
            print!("Enter path: ");
            io::stdout().flush().unwrap();
            let mut custom = String::new();
            io::stdin().read_line(&mut custom).unwrap();
            custom.trim().to_string()
        }
        _ => {
            println!("Invalid choice, using current directory");
            std::env::current_dir().unwrap().to_str().unwrap().to_string()
        }
    };

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("📂 Scanning filesystem starting from: {}\n", scan_root);

    let start = Instant::now();
    let mut stats = ScanStats {
        files: 0,
        dirs: 0,
        skipped: 0,
        name_too_long: 0,
    };

    // Recursively scan and load
    scan_directory(scan_root.as_str(), 0, &mut fs, &mut path_to_inode, &mut stats, 0);

    let elapsed = start.elapsed();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("✅ Filesystem scan complete!\n");
    println!("📊 Statistics:");
    println!("   • Directories: {}", stats.dirs);
    println!("   • Files: {}", stats.files);
    println!("   • Total entries: {}", stats.files + stats.dirs);
    println!("   • Skipped (permissions): {}", stats.skipped);
    println!("   • Skipped (name too long): {}", stats.name_too_long);
    println!("   • Scan time: {:.2}s", elapsed.as_secs_f64());
    println!("   • GPU capacity used: {}%",
             ((stats.files + stats.dirs) as f64 / max_inodes as f64 * 100.0) as u32);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Interactive search
    println!("🔍 GPU Path Search - Try finding files!\n");
    println!("Examples:");
    println!("  - /usr/bin/python3");
    println!("  - /etc/hosts");
    println!("  - /Library/Fonts");
    println!("  - {} (your scan root)", scan_root);
    println!("\nCommands:");
    println!("  'search <path>' - search for a path");
    println!("  'batch <path1> <path2> ...' - batch search");
    println!("  'stats' - show cache stats");
    println!("  'quit' - exit");
    println!();

    loop {
        print!("gpu-fs> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "stats" => {
                let cache_stats = fs.cache_stats();
                println!("\n📊 GPU Cache Statistics:");
                println!("   • Total lookups: {}", cache_stats.hits + cache_stats.misses);
                println!("   • Cache hits: {}", cache_stats.hits);
                println!("   • Cache misses: {}", cache_stats.misses);
                println!("   • Hit rate: {:.1}%", cache_stats.hit_rate * 100.0);
                println!("   • Cached paths: {}/1024", cache_stats.total_entries);
                println!();
            }
            "search" => {
                if parts.len() < 2 {
                    println!("Usage: search <path>");
                    continue;
                }
                let path = parts[1..].join(" ");

                let start = Instant::now();
                match fs.lookup_path(&path) {
                    Ok(inode) => {
                        let elapsed = start.elapsed();
                        println!("✅ Found: {} → inode {} ({:.2}µs)",
                                 path, inode, elapsed.as_micros());
                    }
                    Err(e) => {
                        let elapsed = start.elapsed();
                        println!("❌ Not found: {} ({:.2}µs)", path, elapsed.as_micros());
                        println!("   Error: {}", e);
                    }
                }
            }
            "batch" => {
                if parts.len() < 2 {
                    println!("Usage: batch <path1> <path2> ...");
                    continue;
                }
                let paths: Vec<&str> = parts[1..].to_vec();

                let start = Instant::now();
                match fs.lookup_batch(&paths) {
                    Ok(results) => {
                        let elapsed = start.elapsed();
                        println!("\n⚡ Batch lookup results ({} paths in {:.2}µs):",
                                 paths.len(), elapsed.as_micros());
                        for (path, result) in paths.iter().zip(results.iter()) {
                            match result {
                                Ok(inode) => println!("  ✅ {} → inode {}", path, inode),
                                Err(e) => println!("  ❌ {} → {}", path, e),
                            }
                        }
                        println!("  Avg: {:.2}µs per path\n",
                                 elapsed.as_micros() as f64 / paths.len() as f64);
                    }
                    Err(e) => println!("❌ Batch failed: {}", e),
                }
            }
            _ => {
                // Treat unknown commands as search paths
                let start = Instant::now();
                match fs.lookup_path(input) {
                    Ok(inode) => {
                        let elapsed = start.elapsed();
                        println!("✅ Found: {} → inode {} ({:.2}µs)",
                                 input, inode, elapsed.as_micros());
                    }
                    Err(e) => {
                        let elapsed = start.elapsed();
                        println!("❌ Not found: {} ({:.2}µs)", input, elapsed.as_micros());
                        println!("   Error: {}", e);
                        println!("   (Try: search <path>, batch <path1> <path2>, stats, quit)");
                    }
                }
            }
        }
    }

    // Final stats
    let final_stats = fs.cache_stats();
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Final Session Statistics:");
    println!("   • Files loaded into GPU: {}", stats.files + stats.dirs);
    println!("   • Total GPU lookups: {}", final_stats.hits + final_stats.misses);
    println!("   • Cache hit rate: {:.1}%", final_stats.hit_rate * 100.0);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

struct ScanStats {
    files: usize,
    dirs: usize,
    skipped: usize,
    name_too_long: usize,
}

fn scan_directory(
    path: &str,
    parent_inode: u32,
    fs: &mut GpuFilesystem,
    path_map: &mut HashMap<String, u32>,
    stats: &mut ScanStats,
    depth: usize,
) {
    // Limit recursion depth to avoid stack overflow
    if depth > 20 {
        return;
    }

    // Show progress every 100 directories
    if stats.dirs % 100 == 0 && stats.dirs > 0 {
        print!("\r   Scanning... {} dirs, {} files", stats.dirs, stats.files);
        io::stdout().flush().unwrap();
    }

    let path_obj = Path::new(path);

    let entries = match fs::read_dir(path_obj) {
        Ok(e) => e,
        Err(_) => {
            stats.skipped += 1;
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => {
                stats.skipped += 1;
                continue;
            }
        };

        let file_name = entry.file_name();
        let name_str = match file_name.to_str() {
            Some(s) => s,
            None => {
                stats.skipped += 1;
                continue;
            }
        };

        // Skip if name is too long (current limit is 20 chars)
        if name_str.len() > 20 {
            stats.name_too_long += 1;
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => {
                stats.skipped += 1;
                continue;
            }
        };

        let file_type = if metadata.is_dir() {
            FileType::Directory
        } else {
            FileType::Regular
        };

        // Add to GPU filesystem
        match fs.add_file(parent_inode, name_str, file_type) {
            Ok(inode_id) => {
                let full_path = entry.path();
                let full_path_str = full_path.to_str().unwrap_or("").to_string();
                path_map.insert(full_path_str.clone(), inode_id);

                if metadata.is_dir() {
                    stats.dirs += 1;
                    // Recursively scan subdirectory
                    scan_directory(&full_path_str, inode_id, fs, path_map, stats, depth + 1);
                } else {
                    stats.files += 1;
                }
            }
            Err(_) => {
                // Probably hit max inodes limit
                stats.skipped += 1;
            }
        }
    }
}
