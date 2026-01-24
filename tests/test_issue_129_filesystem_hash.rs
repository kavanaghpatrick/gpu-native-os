//! Issue #129: O(1) GPU Hash Table for Filesystem Directory Lookup
//!
//! Tests for hash table that replaces O(n) linear scan.

use metal::*;
use std::time::Instant;

// Import from library for GPU tests
use rust_experiment::gpu_os::filesystem::{
    GpuFilesystem, FileType, ROOT_INODE_ID, DirHashEntry,
};

// ============================================================================
// Local test struct for pure-Rust algorithm tests
// ============================================================================

/// Local hash table entry for pure-Rust tests (matches library struct layout)
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LocalHashEntry {
    parent_inode: u32,
    name_hash: u32,
    entry_index: u32,
    _padding: u32,
}

impl LocalHashEntry {
    fn empty() -> Self {
        Self {
            parent_inode: 0,
            name_hash: 0,
            entry_index: 0xFFFFFFFF,  // Sentinel for empty
            _padding: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.entry_index == 0xFFFFFFFF
    }
}

/// Simple hash function for testing
fn hash_name(name: &str) -> u32 {
    let mut hash = 0u32;
    for byte in name.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
    }
    hash
}

/// Combined key hash
fn dir_hash_key(parent_inode: u32, name_hash: u32) -> u64 {
    let key = ((parent_inode as u64) << 32) | (name_hash as u64);
    key.wrapping_mul(0x9E3779B97F4A7C15)  // Golden ratio hash
}

/// Hash slot from key
fn hash_slot(key: u64, mask: u32) -> u32 {
    ((key >> 32) as u32) & mask
}

/// Build hash table from entries (pure Rust implementation)
fn build_hash_table(entries: &[(u32, &str, u32)]) -> (Vec<LocalHashEntry>, u32) {
    let capacity = (entries.len() * 2).next_power_of_two() as u32;
    let mask = capacity - 1;
    let mut table = vec![LocalHashEntry::empty(); capacity as usize];

    for (parent_inode, name, entry_index) in entries {
        let name_hash = hash_name(name);
        let key = dir_hash_key(*parent_inode, name_hash);
        let mut slot = hash_slot(key, mask);

        // Linear probing
        while !table[slot as usize].is_empty() {
            slot = (slot + 1) & mask;
        }

        table[slot as usize] = LocalHashEntry {
            parent_inode: *parent_inode,
            name_hash,
            entry_index: *entry_index,
            _padding: 0,
        };
    }

    (table, mask)
}

/// O(n) linear scan lookup
fn lookup_linear(entries: &[(u32, &str, u32)], parent_inode: u32, name: &str) -> Option<u32> {
    let name_hash = hash_name(name);
    for (p, n, idx) in entries {
        if *p == parent_inode && hash_name(n) == name_hash {
            return Some(*idx);
        }
    }
    None
}

/// O(1) hash table lookup (pure Rust)
fn lookup_hash(table: &[LocalHashEntry], mask: u32, parent_inode: u32, name: &str) -> Option<u32> {
    let name_hash = hash_name(name);
    let key = dir_hash_key(parent_inode, name_hash);
    let mut slot = hash_slot(key, mask);

    for _ in 0..32 {
        let entry = &table[slot as usize];

        if entry.is_empty() {
            return None;
        }

        if entry.parent_inode == parent_inode && entry.name_hash == name_hash {
            return Some(entry.entry_index);
        }

        slot = (slot + 1) & mask;
    }

    None
}

// ============================================================================
// Pure Rust Algorithm Tests
// ============================================================================

#[test]
fn test_hash_table_correctness() {
    let entries: Vec<(u32, &str, u32)> = vec![
        (1, "file1.rs", 100),
        (1, "file2.rs", 101),
        (1, "subdir", 102),
        (2, "main.rs", 200),
        (2, "lib.rs", 201),
    ];

    let (table, mask) = build_hash_table(&entries);

    println!("Hash table correctness test:");
    println!("  Entries: {}", entries.len());
    println!("  Table capacity: {}", table.len());

    // Verify all entries can be found
    for (parent, name, expected_idx) in &entries {
        let found = lookup_hash(&table, mask, *parent, name);
        assert_eq!(found, Some(*expected_idx),
            "Failed to find {} in parent {}", name, parent);
    }

    // Verify non-existent entries return None
    assert_eq!(lookup_hash(&table, mask, 1, "nonexistent"), None);
    assert_eq!(lookup_hash(&table, mask, 999, "file1.rs"), None);

    println!("  All lookups correct!");
}

#[test]
fn test_hash_table_large() {
    let entry_count = 10000;
    static NAMES: [&str; 10] = [
        "file0.rs", "file1.rs", "file2.rs", "file3.rs", "file4.rs",
        "file5.rs", "file6.rs", "file7.rs", "file8.rs", "file9.rs",
    ];

    let entries: Vec<(u32, &str, u32)> = (0..entry_count)
        .map(|i| {
            let parent = (i / 100) as u32;
            let name = NAMES[i % 10];
            (parent, name, i as u32)
        })
        .collect();

    let (table, mask) = build_hash_table(&entries);

    println!("\nLarge hash table test:");
    println!("  Entries: {}", entry_count);
    println!("  Table capacity: {}", table.len());
    println!("  Load factor: {:.1}%", 100.0 * entry_count as f64 / table.len() as f64);

    // Verify random lookups
    for i in (0..entry_count).step_by(100) {
        let (parent, name, expected) = entries[i];
        let found = lookup_hash(&table, mask, parent, name);
        assert_eq!(found, Some(expected));
    }

    println!("  Random lookups verified!");
}

#[test]
fn benchmark_linear_vs_hash() {
    let entry_count = 10000;

    let entries_owned: Vec<(u32, String, u32)> = (0..entry_count)
        .map(|i| {
            let parent = (i / 100) as u32;
            let name = format!("file{}.rs", i % 100);
            (parent, name, i as u32)
        })
        .collect();

    let entries_ref: Vec<(u32, &str, u32)> = entries_owned
        .iter()
        .map(|(p, n, i)| (*p, n.as_str(), *i))
        .collect();

    let (table, mask) = build_hash_table(&entries_ref);

    println!("\n=== O(n) Linear vs O(1) Hash Benchmark ===\n");
    println!("Entries: {}", entry_count);

    let lookups = 1000;
    let lookup_indices: Vec<usize> = (0..lookups).map(|i| i * 10 % entry_count).collect();

    // Benchmark linear scan
    let linear_start = Instant::now();
    let mut linear_found = 0u32;
    for &idx in &lookup_indices {
        let (parent, ref name, _) = entries_owned[idx];
        if lookup_linear(&entries_ref, parent, name).is_some() {
            linear_found += 1;
        }
    }
    let linear_time = linear_start.elapsed();

    // Benchmark hash lookup
    let hash_start = Instant::now();
    let mut hash_found = 0u32;
    for &idx in &lookup_indices {
        let (parent, ref name, _) = entries_owned[idx];
        if lookup_hash(&table, mask, parent, name).is_some() {
            hash_found += 1;
        }
    }
    let hash_time = hash_start.elapsed();

    assert_eq!(linear_found, hash_found);

    let speedup = linear_time.as_secs_f64() / hash_time.as_secs_f64();

    println!("Linear O(n): {:.2}ms ({} lookups)", linear_time.as_secs_f64() * 1000.0, lookups);
    println!("Hash O(1):   {:.2}ms ({} lookups)", hash_time.as_secs_f64() * 1000.0, lookups);
    println!("Speedup:     {:.0}x", speedup);

    assert!(speedup > 10.0, "Expected >10x speedup, got {:.1}x", speedup);
}

#[test]
fn test_memory_overhead() {
    let entry_count = 100000;
    let table_capacity = ((entry_count * 2) as usize).next_power_of_two();
    let entry_size = std::mem::size_of::<LocalHashEntry>();
    let table_size = table_capacity * entry_size;

    println!("\nMemory overhead test:");
    println!("  {} entries", entry_count);
    println!("  Table capacity: {} (2x for 50% load factor)", table_capacity);
    println!("  Entry size: {} bytes", entry_size);
    println!("  Total table: {:.1} MB", table_size as f64 / (1024.0 * 1024.0));
    println!("  Per entry: {} bytes", entry_size * 2);

    assert!(table_size < 64 * 1024 * 1024, "Table too large for 100K entries");
}

// ============================================================================
// GPU Implementation Tests
// ============================================================================

#[test]
fn test_gpu_buffer_layout() {
    let device = Device::system_default().expect("No Metal device");

    // Verify DirHashEntry is GPU-friendly (uses library type)
    assert_eq!(std::mem::size_of::<DirHashEntry>(), 16, "Entry should be 16 bytes");
    assert_eq!(std::mem::align_of::<DirHashEntry>(), 4, "Entry should be 4-byte aligned");

    let capacity = 1024u64;
    let buffer = device.new_buffer(
        capacity * 16,
        MTLResourceOptions::StorageModeShared,
    );

    println!("\nGPU buffer layout test:");
    println!("  DirHashEntry size: 16 bytes");
    println!("  Buffer: {} entries, {} bytes", capacity, buffer.length());

    assert_eq!(buffer.length(), capacity * 16);
}

#[test]
fn test_gpu_hash_lookup_kernel() {
    let device = Device::system_default().expect("No Metal device");

    // Create filesystem and add files
    let mut fs = GpuFilesystem::new(&device, 1024).expect("Failed to create filesystem");

    // Create directory structure: /src/main.rs, /src/lib.rs, /tests/test.rs
    let src_id = fs.add_file(ROOT_INODE_ID, "src", FileType::Directory).expect("add src");
    let tests_id = fs.add_file(ROOT_INODE_ID, "tests", FileType::Directory).expect("add tests");
    let _main_id = fs.add_file(src_id, "main.rs", FileType::Regular).expect("add main.rs");
    let lib_id = fs.add_file(src_id, "lib.rs", FileType::Regular).expect("add lib.rs");
    let _test_id = fs.add_file(tests_id, "test.rs", FileType::Regular).expect("add test.rs");

    // Build hash table
    fs.build_hash_table();
    assert!(fs.has_hash_table(), "Hash table should be built");

    // Test O(1) hash lookup
    let result = fs.lookup_path_hash("/src/lib.rs");
    assert!(result.is_ok(), "Should find /src/lib.rs");
    assert_eq!(result.unwrap(), lib_id, "Should return correct inode");

    // Test not found
    let not_found = fs.lookup_path_hash("/src/nonexistent.rs");
    assert!(not_found.is_err(), "Should not find nonexistent file");

    println!("GPU hash lookup kernel test passed!");
}

#[test]
fn test_gpu_hash_table_build() {
    let device = Device::system_default().expect("No Metal device");

    // Create filesystem with many entries
    let mut fs = GpuFilesystem::new(&device, 10000).expect("Failed to create filesystem");

    // Add 1000 files in nested directories
    let mut dir_id = ROOT_INODE_ID;
    for i in 0..10 {
        let new_dir = fs.add_file(dir_id, &format!("dir{}", i), FileType::Directory)
            .expect("add dir");
        dir_id = new_dir;

        for j in 0..100 {
            fs.add_file(dir_id, &format!("file{}.rs", j), FileType::Regular)
                .expect("add file");
        }
    }

    // Build hash table
    fs.build_hash_table();

    // Check stats
    let stats = fs.hash_table_stats();
    assert!(stats.is_some(), "Should have hash table stats");
    let (load_factor, size_bytes) = stats.unwrap();

    println!("\nHash table build test:");
    println!("  Load factor: {:.1}%", load_factor * 100.0);
    println!("  Size: {} bytes ({:.1} KB)", size_bytes, size_bytes as f64 / 1024.0);

    assert!(load_factor < 0.6, "Load factor should be < 60%");
    assert!(load_factor > 0.2, "Load factor should be > 20%");
}

#[test]
fn benchmark_gpu_path_lookup() {
    let device = Device::system_default().expect("No Metal device");

    // Create filesystem with nested structure
    let mut fs = GpuFilesystem::new(&device, 10000).expect("Failed to create filesystem");

    // Build /a/b/c/d/e/f/g/h/i/file.rs (10 components deep)
    let mut dir_id = ROOT_INODE_ID;
    let dir_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
    for name in &dir_names {
        dir_id = fs.add_file(dir_id, name, FileType::Directory).expect("add dir");
    }
    let file_id = fs.add_file(dir_id, "file.rs", FileType::Regular).expect("add file");

    // Build hash table
    fs.build_hash_table();

    let path = "/a/b/c/d/e/f/g/h/i/file.rs";

    // Time linear scan (original lookup_path)
    let linear_start = Instant::now();
    for _ in 0..100 {
        let result = fs.lookup_path(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_id);
    }
    let linear_time = linear_start.elapsed();

    // Time hash lookup (new lookup_path_hash)
    let hash_start = Instant::now();
    for _ in 0..100 {
        let result = fs.lookup_path_hash(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), file_id);
    }
    let hash_time = hash_start.elapsed();

    let speedup = linear_time.as_secs_f64() / hash_time.as_secs_f64();

    println!("\n=== GPU Path Lookup Benchmark ===\n");
    println!("Path: {} (10 components)", path);
    println!("Linear scan (100 lookups): {:.2}ms", linear_time.as_secs_f64() * 1000.0);
    println!("Hash lookup (100 lookups): {:.2}ms", hash_time.as_secs_f64() * 1000.0);
    println!("Speedup: {:.1}x", speedup);
}
