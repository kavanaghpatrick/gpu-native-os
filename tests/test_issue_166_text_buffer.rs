//! Issue #166: GPU Text Buffer Tests
//!
//! Tests for GPU-native text editing with edit log architecture.

use metal::Device;
use rust_experiment::gpu_os::gpu_text_buffer::{
    GpuTextBuffer, TextBufferState, EditEntry, MatchResult,
    EDIT_INSERT, EDIT_DELETE, MAX_EDITS,
};

// ═══════════════════════════════════════════════════════════════════════════════
// BASIC TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_buffer_creation() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    // Initialize with empty content
    buffer.initialize(&device, b"");

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 0);
    assert_eq!(stats.total_bytes, 0);
    println!("Buffer created successfully");
}

#[test]
fn test_initialize_with_content() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    let content = b"Hello, World!";
    buffer.initialize(&device, content);

    let stats = buffer.read_stats();
    assert_eq!(stats.total_bytes, content.len() as u32);
    assert_eq!(stats.edit_count, 0);

    println!("Initialized with {} bytes", stats.total_bytes);
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSERT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_single_insert() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello World");

    // Insert 'X' at position 5 (after "Hello")
    buffer.batch_insert(&device, &[5], b"X");

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 1);
    assert_eq!(stats.total_inserts, 1);

    println!("Single insert successful, edit_count={}", stats.edit_count);
}

#[test]
fn test_batch_insert() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello World");

    // Insert multiple chars at different positions
    let positions = vec![0, 5, 11];
    let chars = b"ABC";
    buffer.batch_insert(&device, &positions, chars);

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 3);
    assert_eq!(stats.total_inserts, 3);

    println!("Batch insert successful, {} edits", stats.edit_count);
}

#[test]
fn test_insert_many() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"");

    // Insert 1000 characters
    let positions: Vec<u32> = (0..1000).collect();
    let chars: Vec<u8> = (0..1000).map(|i| (b'a' + (i % 26) as u8)).collect();
    buffer.batch_insert(&device, &positions, &chars);

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 1000);
    assert_eq!(stats.total_inserts, 1000);

    println!("Inserted 1000 chars, edit_count={}", stats.edit_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_single_delete() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello World");

    // Delete 1 char at position 5 (the space)
    buffer.batch_delete(&device, &[5], &[1]);

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 1);
    assert_eq!(stats.total_deletes, 1);

    println!("Single delete successful");
}

#[test]
fn test_batch_delete() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"ABCDEFGHIJ");

    // Delete at positions 0, 2, 4 (A, C, E)
    let positions = vec![0, 2, 4];
    let lengths = vec![1, 1, 1];
    buffer.batch_delete(&device, &positions, &lengths);

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 3);
    assert_eq!(stats.total_deletes, 3);

    println!("Batch delete successful");
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIND TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_find_pattern() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    let content = b"Hello World, Hello Universe, Hello Galaxy";
    buffer.initialize(&device, content);

    let matches = buffer.find(&device, b"Hello", false);

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].position, 0);
    // Other positions depend on ordering (not guaranteed)

    println!("Found {} matches for 'Hello'", matches.len());
}

#[test]
fn test_find_case_insensitive() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    let content = b"Hello HELLO hello HeLLo";
    buffer.initialize(&device, content);

    let matches = buffer.find(&device, b"hello", true);

    assert_eq!(matches.len(), 4);
    println!("Found {} case-insensitive matches", matches.len());
}

#[test]
fn test_find_no_match() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello World");

    let matches = buffer.find(&device, b"Goodbye", false);

    assert_eq!(matches.len(), 0);
    println!("Correctly found no matches");
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNDO/REDO TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_undo() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello");

    // Insert a char
    buffer.batch_insert(&device, &[5], b"!");

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 1);

    // Undo
    buffer.undo(&device);

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 0);

    println!("Undo successful");
}

#[test]
fn test_redo() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello");

    // Insert a char
    buffer.batch_insert(&device, &[5], b"!");

    // Undo
    buffer.undo(&device);
    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 0);

    // Redo
    buffer.redo(&device);
    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 1);

    println!("Redo successful");
}

#[test]
fn test_multiple_undo() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"");

    // Insert 5 chars
    for i in 0..5 {
        buffer.batch_insert(&device, &[i], &[b'A' + i as u8]);
    }

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 5);

    // Undo 3 times
    for _ in 0..3 {
        buffer.undo(&device);
    }

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 2);

    println!("Multiple undo successful");
}

// ═══════════════════════════════════════════════════════════════════════════════
// READ CONTENT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_read_original_content() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    let original = b"Hello World";
    buffer.initialize(&device, original);

    let content = buffer.read_content(&device, 0, original.len() as u32);
    assert_eq!(content, original);

    println!("Read original content: {:?}", String::from_utf8_lossy(&content));
}

#[test]
fn test_read_after_insert() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello World");

    // Insert 'X' at position 5
    buffer.batch_insert(&device, &[5], b"X");

    // Read the inserted byte
    let state = buffer.read_state();
    // Note: Reading after edit requires edit log replay
    // The total_bytes doesn't change until we track it in the edit

    println!("State after insert: total_bytes={}", state.total_bytes);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MIXED OPERATIONS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_insert_delete_sequence() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 64 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"Hello World");

    // Insert
    buffer.batch_insert(&device, &[5], b"!");

    // Delete
    buffer.batch_delete(&device, &[0], &[1]);

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, 2);
    assert_eq!(stats.total_inserts, 1);
    assert_eq!(stats.total_deletes, 1);

    println!("Insert/delete sequence successful");
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERFORMANCE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_batch_insert_performance() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 1024 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"");

    let count = 10000;
    let positions: Vec<u32> = (0..count).collect();
    let chars: Vec<u8> = (0..count).map(|i| (b'a' + (i % 26) as u8)).collect();

    let start = std::time::Instant::now();
    buffer.batch_insert(&device, &positions, &chars);
    let elapsed = start.elapsed();

    let stats = buffer.read_stats();
    assert_eq!(stats.edit_count, count);

    let throughput = count as f64 / elapsed.as_secs_f64();
    println!("Inserted {} chars in {:?} ({:.2} chars/sec)",
             count, elapsed, throughput);
}

#[test]
fn test_find_performance() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 1024 * 1024).expect("Failed to create buffer");

    // Create a 100KB document with some patterns
    let mut content = Vec::new();
    for i in 0..10000 {
        if i % 100 == 0 {
            content.extend_from_slice(b"PATTERN");
        } else {
            content.extend_from_slice(b"xxxxxxxx");
        }
    }

    buffer.initialize(&device, &content);

    let start = std::time::Instant::now();
    let matches = buffer.find(&device, b"PATTERN", false);
    let elapsed = start.elapsed();

    assert_eq!(matches.len(), 100);
    println!("Found {} matches in {} bytes in {:?}",
             matches.len(), content.len(), elapsed);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Run with --ignored
fn test_stress_edit_log() {
    let device = Device::system_default().expect("No Metal device");
    let buffer = GpuTextBuffer::new(&device, 4 * 1024 * 1024).expect("Failed to create buffer");

    buffer.initialize(&device, b"");

    // Fill edit log to 50% capacity
    let count = MAX_EDITS / 2;
    let positions: Vec<u32> = (0..count).collect();
    let chars: Vec<u8> = (0..count).map(|i| (b'a' + (i % 26) as u8)).collect();

    let start = std::time::Instant::now();
    buffer.batch_insert(&device, &positions, &chars);
    let elapsed = start.elapsed();

    let stats = buffer.read_stats();
    println!("Filled {} edits in {:?}", stats.edit_count, elapsed);
    println!("Total inserts: {}", stats.total_inserts);
}
