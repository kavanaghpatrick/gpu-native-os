//! Build the shared GPU-resident filesystem index for home directory
//!
//! Usage: cargo run --release --example build_shared_index

use std::path::Path;
use rust_experiment::gpu_os::shared_index::{build_index_fast, IndexManifest, IndexInfo};

fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    let output_dir = Path::new(&home).join(".gpu_os/index");
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    let index_path = output_dir.join("home.idx");
    let excludes: [&str; 0] = [];

    println!("Building home index to {}...", index_path.display());
    match build_index_fast(Path::new(&home), &index_path, &excludes) {
        Ok(()) => println!("Index built successfully!"),
        Err(e) => {
            eprintln!("Error: {:?}", e);
            return;
        }
    }

    // Get index file size and entry count from the file
    let index_size = std::fs::metadata(&index_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Entry count = (file_size - 4096 header) / 256 bytes per entry
    let entry_count = if index_size > 4096 {
        ((index_size - 4096) / 256) as u32
    } else {
        0
    };

    // Create proper manifest with IndexInfo
    let home_info = IndexInfo {
        name: "home".to_string(),
        path: index_path.clone(),
        root: Path::new(&home).to_path_buf(),
        entry_count,
        size_bytes: index_size,
        built_at: chrono::Utc::now(),
        exclude_patterns: vec![],
    };

    let manifest = IndexManifest::new(vec![home_info]);
    let manifest_json = serde_json::to_string_pretty(&manifest).expect("Failed to serialize manifest");

    let manifest_path = output_dir.join("manifest.json");
    std::fs::write(&manifest_path, manifest_json).expect("Failed to write manifest");
    println!("Wrote manifest to {}", manifest_path.display());
    println!("Index: {} entries, {} MB", entry_count, index_size / (1024 * 1024));
}
