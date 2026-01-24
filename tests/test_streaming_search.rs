//! Integration tests for streaming search and text rendering

use metal::Device;
use rust_experiment::gpu_os::filesystem::{GpuStreamingSearch, parse_query_words};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer, colors};
use rust_experiment::gpu_os::app::GpuRuntime;

#[test]
fn test_streaming_search_creation() {
    let device = Device::system_default().expect("No Metal device");
    let streaming = GpuStreamingSearch::new(&device);
    assert!(streaming.is_ok(), "GpuStreamingSearch should be created successfully");
}

#[test]
fn test_streaming_search_empty_query() {
    let device = Device::system_default().expect("No Metal device");
    let mut streaming = GpuStreamingSearch::new(&device).unwrap();

    // Empty query should return no results
    let query_words = parse_query_words("");
    assert!(query_words.is_empty(), "Empty query should produce no words");

    // Process a chunk with empty query - should not crash
    let paths = vec!["test/path".to_string()];
    streaming.process_chunk(&paths, 0, &query_words);

    let results = streaming.get_results();
    assert!(results.is_empty(), "Empty query should return no results");
}

#[test]
fn test_streaming_search_single_chunk() {
    let device = Device::system_default().expect("No Metal device");
    let mut streaming = GpuStreamingSearch::new(&device).unwrap();

    // Create test paths
    let paths: Vec<String> = vec![
        "/Users/test/documents/report.pdf".to_string(),
        "/Users/test/photos/vacation.jpg".to_string(),
        "/Users/test/code/project/main.rs".to_string(),
        "/Users/test/code/project/lib.rs".to_string(),
        "/tmp/random.txt".to_string(),
    ];

    // Search for "test"
    let query_words = parse_query_words("test");
    assert_eq!(query_words.len(), 1, "Should have one query word");

    streaming.reset();
    streaming.process_chunk(&paths, 0, &query_words);

    let results = streaming.get_results();
    // All paths except /tmp should match "test"
    assert!(results.len() >= 4, "Should find at least 4 matches for 'test'");
}

#[test]
fn test_parse_query_words() {
    // Single word
    let words = parse_query_words("hello");
    assert_eq!(words.len(), 1);
    assert_eq!(words[0].0, "hello");

    // Multiple words
    let words = parse_query_words("hello world");
    assert_eq!(words.len(), 2);
    assert_eq!(words[0].0, "hello");
    assert_eq!(words[1].0, "world");

    // Case conversion
    let words = parse_query_words("HELLO");
    assert_eq!(words[0].0, "hello");

    // Empty
    let words = parse_query_words("");
    assert!(words.is_empty());

    // Whitespace only
    let words = parse_query_words("   ");
    assert!(words.is_empty());
}

#[test]
fn test_bitmap_font_creation() {
    let device = Device::system_default().expect("No Metal device");
    let font = BitmapFont::new(&device);

    assert_eq!(font.char_width, 8.0, "Font should be 8px wide");
    assert_eq!(font.char_height, 8.0, "Font should be 8px tall");
    assert_eq!(font.chars_per_row, 16, "Font atlas should have 16 chars per row");
}

#[test]
fn test_text_renderer_creation() {
    let device = Device::system_default().expect("No Metal device");
    let renderer = TextRenderer::new(&device, 1000);

    assert!(renderer.is_ok(), "TextRenderer should be created successfully");

    let mut renderer = renderer.unwrap();
    assert_eq!(renderer.char_count(), 0, "Fresh renderer should have 0 chars");

    // Add some text
    renderer.add_text("Hello", 10.0, 10.0, colors::WHITE);
    assert_eq!(renderer.char_count(), 5, "Should have 5 chars after adding 'Hello'");

    // Clear
    renderer.clear();
    assert_eq!(renderer.char_count(), 0, "Should have 0 chars after clear");
}

#[test]
fn test_gpu_runtime_has_text_rendering() {
    let device = Device::system_default().expect("No Metal device");
    let mut runtime = GpuRuntime::new(device);

    // Check font is accessible
    let font = runtime.font();
    assert_eq!(font.char_width, 8.0);

    // Check text renderer is accessible and works
    let text = runtime.text_renderer_mut();
    text.add_text("Test", 0.0, 0.0, colors::WHITE);
    assert_eq!(text.char_count(), 4);

    text.clear();
    assert_eq!(runtime.text_renderer().char_count(), 0);
}

#[test]
fn test_streaming_search_stats() {
    let device = Device::system_default().expect("No Metal device");
    let mut streaming = GpuStreamingSearch::new(&device).unwrap();

    let paths: Vec<String> = (0..100)
        .map(|i| format!("/test/path{}.txt", i))
        .collect();

    let query_words = parse_query_words("test");

    streaming.reset();
    streaming.process_chunk(&paths, 0, &query_words);

    let (total_searched, chunks_processed) = streaming.stats();
    assert_eq!(total_searched, 100, "Should have searched 100 paths");
    assert_eq!(chunks_processed, 1, "Should have processed 1 chunk");
}
