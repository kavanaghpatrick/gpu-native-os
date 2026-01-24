// GPU-Native Filesystem Path Lookup Demo
//
// Demonstrates Issue #21: Path Lookup GPU Kernel
// Shows parallel hash-based path resolution running entirely on GPU

use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};
use metal::Device;

fn main() {
    // Create Metal device
    let device = Device::system_default().expect("No Metal device found");
    println!("Using device: {}", device.name());
    println!();

    // Create GPU filesystem
    let mut filesystem = GpuFilesystem::new(&device, 1024)
        .expect("Failed to create filesystem");

    // Build a realistic directory tree
    println!("📁 Building directory tree...");

    // /src
    let src_id = filesystem.add_file(0, "src", FileType::Directory).unwrap();
    filesystem.add_file(src_id, "main.rs", FileType::Regular).unwrap();
    filesystem.add_file(src_id, "lib.rs", FileType::Regular).unwrap();

    // /src/gpu_os
    let gpu_os_id = filesystem.add_file(src_id, "gpu_os", FileType::Directory).unwrap();
    filesystem.add_file(gpu_os_id, "mod.rs", FileType::Regular).unwrap();
    filesystem.add_file(gpu_os_id, "kernel.rs", FileType::Regular).unwrap();
    filesystem.add_file(gpu_os_id, "filesystem.rs", FileType::Regular).unwrap();

    // /docs
    let docs_id = filesystem.add_file(0, "docs", FileType::Directory).unwrap();
    filesystem.add_file(docs_id, "README.md", FileType::Regular).unwrap();
    filesystem.add_file(docs_id, "ARCHITECTURE.md", FileType::Regular).unwrap();

    // /tests
    let tests_id = filesystem.add_file(0, "tests", FileType::Directory).unwrap();
    filesystem.add_file(tests_id, "integration.rs", FileType::Regular).unwrap();

    // Root files
    filesystem.add_file(0, "Cargo.toml", FileType::Regular).unwrap();
    filesystem.add_file(0, "README.md", FileType::Regular).unwrap();

    println!("✅ Created 13 files in 4 directories");
    println!();

    // Demonstrate GPU path lookup
    println!("🔍 GPU Path Lookup Tests");
    println!("   Uses parallel hash-based search running on 1024 GPU threads");
    println!();

    let test_paths = vec![
        "/",
        "/src",
        "/src/main.rs",
        "/src/gpu_os",
        "/src/gpu_os/filesystem.rs",
        "/docs/README.md",
        "/tests/integration.rs",
        "/Cargo.toml",
        "/nonexistent",
        "/src/missing/file.rs",
    ];

    for path in test_paths {
        match filesystem.lookup_path(path) {
            Ok(inode_id) => {
                println!("  ✓ Found: {} → inode {}", path, inode_id);
            }
            Err(err) => {
                println!("  ✗ Not found: {} ({})", path, err);
            }
        }
    }

    println!();
    println!("📊 Performance Characteristics:");
    println!("   - Hash computation: O(path length)");
    println!("   - Directory search: O(entries / 1024) per level (GPU parallel)");
    println!("   - Worst case: O(depth * entries / 1024)");
    println!("   - Compare to traditional: O(depth * entries) sequential");
    println!();
    println!("💡 With 10,000 entries and depth 5:");
    println!("   - Traditional: ~50,000 comparisons (sequential)");
    println!("   - GPU-Native: ~50 cycles (1024 threads parallel)");
    println!("   - Speedup: ~1000x theoretical");
}
