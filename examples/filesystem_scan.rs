// Scan Real Filesystem and Search with GPU
//
// Quick demo that loads your filesystem and lets you search

use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};
use metal::Device;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("\n🚀 GPU Filesystem Scanner\n");

    let device = Device::system_default().expect("No Metal device found");
    println!("GPU: {}", device.name());

    let max_inodes = 50_000;
    let mut fs = GpuFilesystem::new(&device, max_inodes).expect("Failed to create filesystem");

    // Scan current directory
    let scan_root = std::env::current_dir().unwrap();
    let scan_root_str = scan_root.to_str().unwrap();

    println!("Scanning: {}\n", scan_root_str);

    let start = Instant::now();
    let mut path_map: HashMap<String, u32> = HashMap::new();
    path_map.insert("/".to_string(), 0);

    let mut stats = ScanStats {
        files: 0,
        dirs: 0,
        skipped_perm: 0,
        skipped_long: 0,
    };

    scan_dir(&scan_root, 0, &mut fs, &mut stats, 0);

    println!("\n✅ Scan complete in {:.2}s", start.elapsed().as_secs_f64());
    println!("   Files: {}, Dirs: {}, Total: {}",
             stats.files, stats.dirs, stats.files + stats.dirs);
    println!("   Skipped: {} (long names), {} (permissions)\n",
             stats.skipped_long, stats.skipped_perm);

    // Build path index for faster searching
    println!("📁 Loaded files (showing first 20):");
    let mut all_paths = Vec::new();
    collect_paths(&scan_root, "/", &mut all_paths);

    for (i, path) in all_paths.iter().take(20).enumerate() {
        println!("   {}: {}", i + 1, path);
    }
    if all_paths.len() > 20 {
        println!("   ... and {} more", all_paths.len() - 20);
    }
    println!();

    // Interactive search
    loop {
        print!("Search (relative path, e.g., 'src', '/src/main.rs', or 'quit'): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "q" || input == "exit" {
            break;
        }

        // Ensure path starts with /
        let search_path = if input.starts_with('/') {
            input.to_string()
        } else {
            format!("/{}", input)
        };

        let start = Instant::now();
        match fs.lookup_path(&search_path) {
            Ok(inode) => {
                println!("✅ Found inode {} in {:.1}µs\n",
                         inode, start.elapsed().as_micros());
            }
            Err(e) => {
                println!("❌ Not found: {} ({:.1}µs)\n",
                         e, start.elapsed().as_micros());
            }
        }
    }

    let cache_stats = fs.cache_stats();
    println!("\n📊 Cache: {}/{} hits ({:.0}% hit rate)",
             cache_stats.hits, cache_stats.hits + cache_stats.misses,
             cache_stats.hit_rate * 100.0);
}

struct ScanStats {
    files: usize,
    dirs: usize,
    skipped_perm: usize,
    skipped_long: usize,
}

fn scan_dir(
    path: &Path,
    parent_inode: u32,
    fs: &mut GpuFilesystem,
    stats: &mut ScanStats,
    depth: usize,
) {
    if depth > 15 {
        return;
    }

    if stats.dirs % 50 == 0 && stats.dirs > 0 {
        print!("\rScanning... {} dirs, {} files", stats.dirs, stats.files);
        io::stdout().flush().unwrap();
    }

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => {
            stats.skipped_perm += 1;
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_name = entry.file_name();
        let name = match file_name.to_str() {
            Some(s) => s,
            None => continue,
        };

        // Skip long names (20 char limit in current implementation)
        if name.len() > 20 {
            stats.skipped_long += 1;
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
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
                    scan_dir(&entry.path(), inode_id, fs, stats, depth + 1);
                } else {
                    stats.files += 1;
                }
            }
            Err(_) => {
                stats.skipped_perm += 1;
            }
        }
    }
}

fn collect_paths(root: &Path, current_path: &str, paths: &mut Vec<String>) {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let file_name = entry.file_name();
        let name = match file_name.to_str() {
            Some(s) => s,
            None => continue,
        };

        if name.len() > 20 {
            continue;
        }

        let full_path = format!("{}/{}", current_path.trim_end_matches('/'), name);
        paths.push(full_path.clone());

        if let Ok(metadata) = entry.metadata() {
            if metadata.is_dir() && paths.len() < 500 {
                collect_paths(&entry.path(), &full_path, paths);
            }
        }
    }
}
