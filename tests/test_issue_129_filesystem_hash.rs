//! Issue #129: O(1) GPU Hash Table for Filesystem Directory Lookup
//!
//! Tests for hash table that replaces O(n) linear scan.

use metal::*;
use std::collections::HashMap;
use std::time::Instant;

/// Hash table entry
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DirHashEntry {
    parent_inode: u32,
    name_hash: u32,
    entry_index: u32,
    _padding: u32,
}

impl DirHashEntry {
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

/// Build hash table from entries
fn build_hash_table(entries: &[(u32, &str, u32)]) -> (Vec<DirHashEntry>, u32) {
    // entries: [(parent_inode, name, entry_index), ...]

    let capacity = (entries.len() * 2).next_power_of_two() as u32;
    let mask = capacity - 1;
    let mut table = vec![DirHashEntry::empty(); capacity as usize];

    for (parent_inode, name, entry_index) in entries {
        let name_hash = hash_name(name);
        let key = dir_hash_key(*parent_inode, name_hash);
        let mut slot = hash_slot(key, mask);

        // Linear probing
        while !table[slot as usize].is_empty() {
            slot = (slot + 1) & mask;
        }

        table[slot as usize] = DirHashEntry {
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

/// O(1) hash table lookup
fn lookup_hash(table: &[DirHashEntry], mask: u32, parent_inode: u32, name: &str) -> Option<u32> {
    let name_hash = hash_name(name);
    let key = dir_hash_key(parent_inode, name_hash);
    let mut slot = hash_slot(key, mask);

    // Linear probing with max 32 attempts
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
    // Test with many entries
    let entry_count = 10000;
    let mut entries: Vec<(u32, &'static str, u32)> = Vec::new();

    // Generate test data using static strings
    static NAMES: [&str; 10] = [
        "file0.rs", "file1.rs", "file2.rs", "file3.rs", "file4.rs",
        "file5.rs", "file6.rs", "file7.rs", "file8.rs", "file9.rs",
    ];

    for i in 0..entry_count {
        let parent = (i / 100) as u32;
        let name = NAMES[i % 10];
        entries.push((parent, name, i as u32));
    }

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

    // Generate entries with owned strings for linear scan
    let entries_owned: Vec<(u32, String, u32)> = (0..entry_count)
        .map(|i| {
            let parent = (i / 100) as u32;
            let name = format!("file{}.rs", i % 100);
            (parent, name, i as u32)
        })
        .collect();

    // Convert to references for hash table
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

    println!("Linear O(n): {:.2}ms ({} lookups)",
        linear_time.as_secs_f64() * 1000.0, lookups);
    println!("Hash O(1):   {:.2}ms ({} lookups)",
        hash_time.as_secs_f64() * 1000.0, lookups);
    println!("Speedup:     {:.0}x", speedup);

    // With 10K entries, hash should be much faster
    assert!(speedup > 10.0, "Expected >10x speedup, got {:.1}x", speedup);
}

#[test]
fn test_memory_overhead() {
    let entry_count = 100000;
    let table_capacity = ((entry_count * 2) as usize).next_power_of_two();
    let entry_size = std::mem::size_of::<DirHashEntry>();
    let table_size = table_capacity * entry_size;

    println!("\nMemory overhead test:");
    println!("  {} entries", entry_count);
    println!("  Table capacity: {} (2x for 50% load factor)", table_capacity);
    println!("  Entry size: {} bytes", entry_size);
    println!("  Total table: {:.1} MB", table_size as f64 / (1024.0 * 1024.0));
    println!("  Per entry: {} bytes", entry_size * 2);  // 2x capacity

    // 16 bytes per entry * 2x capacity = 32 bytes per file
    // 1M files = 32 MB - acceptable
    assert!(table_size < 64 * 1024 * 1024, "Table too large for 100K entries");
}

#[test]
fn test_gpu_buffer_layout() {
    let device = Device::system_default().expect("No Metal device");

    // Verify DirHashEntry is GPU-friendly
    assert_eq!(std::mem::size_of::<DirHashEntry>(), 16, "Entry should be 16 bytes");
    assert_eq!(std::mem::align_of::<DirHashEntry>(), 4, "Entry should be 4-byte aligned");

    let capacity = 1024u64;
    let buffer = device.new_buffer(
        capacity * 16,  // 16 bytes per entry
        MTLResourceOptions::StorageModeShared,
    );

    println!("\nGPU buffer layout test:");
    println!("  DirHashEntry size: 16 bytes");
    println!("  Buffer: {} entries, {} bytes", capacity, buffer.length());

    assert_eq!(buffer.length(), capacity * 16);
}

// Placeholder for GPU implementation tests
#[test]
#[ignore = "Requires GPU implementation"]
fn test_gpu_hash_lookup_kernel() {
    // TODO: Test actual Metal kernel implementation
}

#[test]
#[ignore = "Requires GPU implementation"]
fn benchmark_gpu_path_lookup() {
    // TODO: Benchmark full path resolution with hash table
}
