// Issue #77: GPU-Resident Filesystem Index Tests
//
// Verifies that the filesystem index lives permanently in GPU memory:
// - CPU scans ONCE, GPU owns data forever
// - Zero-copy mmap loading
// - Instant load times regardless of index size
//
// THE GPU IS THE COMPUTER.

use rust_experiment::gpu_os::gpu_index::{
    GpuPathEntry, GpuResidentIndex, IndexError,
    GPU_ENTRY_SIZE, GPU_PATH_MAX_LEN, FLAG_IS_DIR, FLAG_IS_HIDDEN,
};
use std::fs::File;
use std::io::Write;

fn get_device() -> metal::Device {
    metal::Device::system_default().expect("No Metal device found")
}

// =============================================================================
// GpuPathEntry Tests
// =============================================================================

#[test]
fn test_gpu_path_entry_size() {
    // Must be exactly 256 bytes for cache alignment
    assert_eq!(std::mem::size_of::<GpuPathEntry>(), GPU_ENTRY_SIZE);
    assert_eq!(GPU_ENTRY_SIZE, 256);
}

#[test]
fn test_gpu_path_entry_creation() {
    let entry = GpuPathEntry::new("/home/user/test.txt", false, 1024, 1234567890);

    assert_eq!(entry.path_str(), "/home/user/test.txt");
    assert_eq!(entry.path_len, 19);
    assert!(!entry.is_dir());
    assert!(!entry.is_hidden());
    assert_eq!(entry.size, 1024);
    assert_eq!(entry.mtime, 1234567890);
}

#[test]
fn test_gpu_path_entry_directory() {
    let entry = GpuPathEntry::new("/home/user/docs", true, 0, 1234567890);

    assert!(entry.is_dir());
    assert_eq!(entry.flags & FLAG_IS_DIR, FLAG_IS_DIR);
}

#[test]
fn test_gpu_path_entry_hidden() {
    let entry = GpuPathEntry::new("/home/user/.hidden", false, 100, 0);
    assert!(entry.is_hidden());

    let entry2 = GpuPathEntry::new(".bashrc", false, 100, 0);
    assert!(entry2.is_hidden());

    let entry3 = GpuPathEntry::new("/home/user/visible.txt", false, 100, 0);
    assert!(!entry3.is_hidden());
}

#[test]
fn test_gpu_path_entry_long_path() {
    // Path longer than GPU_PATH_MAX_LEN should be truncated
    let long_path = "a".repeat(300);
    let entry = GpuPathEntry::new(&long_path, false, 0, 0);

    assert_eq!(entry.path_len as usize, GPU_PATH_MAX_LEN);
    assert_eq!(entry.path_str().len(), GPU_PATH_MAX_LEN);
}

#[test]
fn test_gpu_path_entry_as_bytes() {
    let entry = GpuPathEntry::new("/test", false, 0, 0);
    let bytes = entry.as_bytes();

    assert_eq!(bytes.len(), GPU_ENTRY_SIZE);
    // First bytes should be the path
    assert_eq!(bytes[0], b'/');
    assert_eq!(bytes[1], b't');
}

// =============================================================================
// Index Build and Load Tests
// =============================================================================

#[test]
fn test_index_build_and_load() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create some test files
    let test_dir = temp_dir.path().join("test_files");
    std::fs::create_dir_all(&test_dir).unwrap();
    File::create(test_dir.join("file1.txt")).unwrap();
    File::create(test_dir.join("file2.txt")).unwrap();
    std::fs::create_dir(test_dir.join("subdir")).unwrap();
    File::create(test_dir.join("subdir/file3.txt")).unwrap();

    // Build index
    let index_path = temp_dir.path().join("test.idx");
    let count = GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();

    assert!(count >= 4, "Should have at least 4 entries");

    // Load index
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();

    assert_eq!(index.entry_count(), count);
    assert!(index.memory_usage() > 0);
}

#[test]
fn test_index_zero_copy_buffer() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create test files
    let test_dir = temp_dir.path().join("test");
    std::fs::create_dir_all(&test_dir).unwrap();
    File::create(test_dir.join("a.txt")).unwrap();

    // Build and load index
    let index_path = temp_dir.path().join("test.idx");
    GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();

    // Buffer should be StorageModeShared (zero-copy)
    assert_eq!(
        index.entries_buffer().storage_mode(),
        metal::MTLStorageMode::Shared
    );
}

#[test]
fn test_index_entry_access() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create test files
    let test_dir = temp_dir.path().join("test");
    std::fs::create_dir_all(&test_dir).unwrap();
    let mut f = File::create(test_dir.join("hello.txt")).unwrap();
    f.write_all(b"hello world").unwrap();

    // Build and load index
    let index_path = temp_dir.path().join("test.idx");
    GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();

    // Access entries
    let entry = index.get_entry(0).expect("Should have at least one entry");
    assert!(entry.path_str().contains("hello.txt") || entry.path_str().contains("test"));
}

#[test]
fn test_index_iteration() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create test files
    let test_dir = temp_dir.path().join("test");
    std::fs::create_dir_all(&test_dir).unwrap();
    File::create(test_dir.join("a.txt")).unwrap();
    File::create(test_dir.join("b.txt")).unwrap();
    File::create(test_dir.join("c.txt")).unwrap();

    // Build and load index
    let index_path = temp_dir.path().join("test.idx");
    GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();

    // Iterate and count
    let count = index.iter().count();
    assert_eq!(count, index.entry_count() as usize);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_index_invalid_file() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();
    let bad_path = temp_dir.path().join("nonexistent.idx");

    let result = GpuResidentIndex::load(&device, &bad_path);
    assert!(matches!(result, Err(IndexError::MmapError(_))));
}

#[test]
fn test_index_invalid_magic() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();
    let bad_path = temp_dir.path().join("bad.idx");

    // Write garbage
    let mut f = File::create(&bad_path).unwrap();
    f.write_all(&[0u8; 8192]).unwrap();

    let result = GpuResidentIndex::load(&device, &bad_path);
    assert!(matches!(result, Err(IndexError::InvalidFormat(_))));
}

#[test]
fn test_index_empty_directory() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    let empty_dir = temp_dir.path().join("empty");
    std::fs::create_dir_all(&empty_dir).unwrap();

    let index_path = temp_dir.path().join("empty.idx");
    let count = GpuResidentIndex::build_and_save(&empty_dir, &index_path, None).unwrap();

    // Should have 0 entries (empty directory)
    assert_eq!(count, 0);

    // Loading should work (but entry_count = 0)
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();
    assert_eq!(index.entry_count(), 0);
}

// =============================================================================
// Performance Tests
// =============================================================================

#[test]
fn test_index_load_time() {
    use std::time::Instant;

    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create many test files
    let test_dir = temp_dir.path().join("test");
    std::fs::create_dir_all(&test_dir).unwrap();
    for i in 0..1000 {
        File::create(test_dir.join(format!("file_{}.txt", i))).unwrap();
    }

    // Build index
    let index_path = temp_dir.path().join("test.idx");
    GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();

    // Measure load time (should be instant via mmap)
    let start = Instant::now();
    let _index = GpuResidentIndex::load(&device, &index_path).unwrap();
    let load_time = start.elapsed();

    println!("\n=== Issue #77: GPU-Resident Index Load Time ===");
    println!("Entries: 1000");
    println!("Load time: {:?}", load_time);

    // Should be very fast (< 100ms) regardless of size
    assert!(
        load_time.as_millis() < 100,
        "Load time should be < 100ms, was {:?}",
        load_time
    );
}

#[test]
fn test_index_memory_efficiency() {
    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create test files
    let test_dir = temp_dir.path().join("test");
    std::fs::create_dir_all(&test_dir).unwrap();
    for i in 0..100 {
        File::create(test_dir.join(format!("file_{}.txt", i))).unwrap();
    }

    // Build and load index
    let index_path = temp_dir.path().join("test.idx");
    let count = GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();

    // Memory should be approximately entry_count * 256 + header (4096)
    let expected_min = 4096 + (count as usize * 256);
    let actual = index.memory_usage();

    assert!(
        actual >= expected_min,
        "Memory {} should be >= {} (header + {} entries * 256)",
        actual, expected_min, count
    );

    println!("\n=== Issue #77: Memory Efficiency ===");
    println!("Entries: {}", count);
    println!("Memory: {} bytes ({} KB)", actual, actual / 1024);
    println!("Per entry: {} bytes", actual / count.max(1) as usize);
}

// =============================================================================
// GPU Access Test
// =============================================================================

#[test]
fn test_index_gpu_accessible() {
    use metal::*;

    let device = get_device();
    let temp_dir = tempfile::tempdir().unwrap();

    // Create test files
    let test_dir = temp_dir.path().join("test");
    std::fs::create_dir_all(&test_dir).unwrap();
    File::create(test_dir.join("test.txt")).unwrap();

    // Build and load index
    let index_path = temp_dir.path().join("test.idx");
    GpuResidentIndex::build_and_save(&test_dir, &index_path, None).unwrap();
    let index = GpuResidentIndex::load(&device, &index_path).unwrap();

    // Simple shader that reads from index buffer
    let shader = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct GpuPathEntry {
            char path[224];
            uint16_t path_len;
            uint16_t flags;
            uint32_t parent_idx;
            uint64_t size;
            uint64_t mtime;
            uint8_t _reserved[8];
        };

        kernel void read_index(
            device const char* buffer [[buffer(0)]],
            device uint* output [[buffer(1)]],
            constant uint& header_size [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid == 0) {
                // Read first entry's path_len
                device const GpuPathEntry* entries =
                    (device const GpuPathEntry*)(buffer + header_size);
                output[0] = entries[0].path_len;
            }
        }
    "#;

    // Compile
    let options = CompileOptions::new();
    let library = device.new_library_with_source(shader, &options)
        .expect("Failed to compile shader");
    let function = library.get_function("read_index", None)
        .expect("Failed to get function");
    let pipeline = device.new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline");

    // Output buffer
    let output = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    // Run compute
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(index.entries_buffer()), 0);
    enc.set_buffer(1, Some(&output), 0);
    let header_size = index.entries_offset() as u32;
    enc.set_bytes(2, 4, &header_size as *const _ as *const _);
    enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    enc.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // Verify GPU read the path_len correctly
    let result = unsafe { *(output.contents() as *const u32) };
    let expected = index.get_entry(0).map(|e| e.path_len as u32).unwrap_or(0);

    assert_eq!(result, expected, "GPU should read correct path_len from index");
}
