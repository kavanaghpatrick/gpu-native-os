//! GPU-Native HTML Tokenizer
//!
//! Converts raw HTML bytes into a stream of tokens using parallel GPU processing.
//! The tokenizer runs in two passes:
//! 1. Boundary detection: Mark positions where tokens start
//! 2. Token extraction: Extract token type, start, end from boundaries

use metal::*;

// Token types (must match Metal shader)
pub const TOKEN_NONE: u32 = 0;
pub const TOKEN_TAG_OPEN: u32 = 1;
pub const TOKEN_TAG_CLOSE: u32 = 2;
pub const TOKEN_TAG_SELF: u32 = 3;
pub const TOKEN_TEXT: u32 = 4;
pub const TOKEN_COMMENT: u32 = 5;
pub const TOKEN_DOCTYPE: u32 = 6;

// Buffer sizes
pub const MAX_DOCUMENT_SIZE: usize = 1024 * 1024; // 1MB
pub const MAX_TOKENS: usize = 65536; // 64K tokens
const THREAD_COUNT: u64 = 1024;

/// A token produced by the tokenizer
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Token {
    pub token_type: u32,
    pub start: u32,
    pub end: u32,
    pub _padding: u32,
}

impl Token {
    /// Get the token type as a string for debugging
    pub fn type_name(&self) -> &'static str {
        match self.token_type {
            TOKEN_TAG_OPEN => "TAG_OPEN",
            TOKEN_TAG_CLOSE => "TAG_CLOSE",
            TOKEN_TAG_SELF => "TAG_SELF",
            TOKEN_TEXT => "TEXT",
            TOKEN_COMMENT => "COMMENT",
            TOKEN_DOCTYPE => "DOCTYPE",
            _ => "UNKNOWN",
        }
    }

    /// Extract the token's text from the source HTML
    pub fn text<'a>(&self, html: &'a [u8]) -> &'a [u8] {
        &html[self.start as usize..self.end as usize]
    }
}

/// Metal shader source for the tokenizer
const TOKENIZER_SHADER: &str = include_str!("tokenizer.metal");

/// GPU-Native HTML Tokenizer
pub struct GpuTokenizer {
    device: Device,
    command_queue: CommandQueue,
    boundary_pipeline: ComputePipelineState,
    extract_pipeline: ComputePipelineState,

    // Buffers
    html_buffer: Buffer,
    boundary_buffer: Buffer,
    token_buffer: Buffer,
    token_count_buffer: Buffer,
    length_buffer: Buffer,
}

impl GpuTokenizer {
    /// Create a new GPU tokenizer
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(TOKENIZER_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile tokenizer shader: {}", e))?;

        // Create pipelines
        let boundary_fn = library
            .get_function("tokenize_boundaries", None)
            .map_err(|e| format!("Failed to find tokenize_boundaries function: {}", e))?;
        let extract_fn = library
            .get_function("tokenize_extract", None)
            .map_err(|e| format!("Failed to find tokenize_extract function: {}", e))?;

        let boundary_pipeline = device
            .new_compute_pipeline_state_with_function(&boundary_fn)
            .map_err(|e| format!("Failed to create boundary pipeline: {}", e))?;
        let extract_pipeline = device
            .new_compute_pipeline_state_with_function(&extract_fn)
            .map_err(|e| format!("Failed to create extract pipeline: {}", e))?;

        // Allocate buffers
        let html_buffer = device.new_buffer(
            MAX_DOCUMENT_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let boundary_buffer = device.new_buffer(
            (MAX_DOCUMENT_SIZE * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_buffer = device.new_buffer(
            (MAX_TOKENS * std::mem::size_of::<Token>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_count_buffer =
            device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let length_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            device: device.clone(),
            command_queue,
            boundary_pipeline,
            extract_pipeline,
            html_buffer,
            boundary_buffer,
            token_buffer,
            token_count_buffer,
            length_buffer,
        })
    }

    /// Tokenize HTML bytes into a stream of tokens
    pub fn tokenize(&mut self, html: &[u8]) -> Vec<Token> {
        if html.is_empty() {
            return Vec::new();
        }

        let length = html.len().min(MAX_DOCUMENT_SIZE);

        // Copy input HTML to GPU buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                html.as_ptr(),
                self.html_buffer.contents() as *mut u8,
                length,
            );
        }

        // Set length
        unsafe {
            let length_ptr = self.length_buffer.contents() as *mut u32;
            *length_ptr = length as u32;
        }

        // Reset token count
        unsafe {
            let count_ptr = self.token_count_buffer.contents() as *mut u32;
            *count_ptr = 0;
        }

        // Clear boundary buffer
        unsafe {
            std::ptr::write_bytes(
                self.boundary_buffer.contents() as *mut u8,
                0,
                length * 4,
            );
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1A: Boundary detection
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.boundary_pipeline);
            encoder.set_buffer(0, Some(&self.html_buffer), 0);
            encoder.set_buffer(1, Some(&self.boundary_buffer), 0);
            encoder.set_buffer(2, Some(&self.length_buffer), 0);
            encoder.dispatch_threads(
                MTLSize::new(THREAD_COUNT, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 1B: Token extraction
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.extract_pipeline);
            encoder.set_buffer(0, Some(&self.html_buffer), 0);
            encoder.set_buffer(1, Some(&self.boundary_buffer), 0);
            encoder.set_buffer(2, Some(&self.token_buffer), 0);
            encoder.set_buffer(3, Some(&self.token_count_buffer), 0);
            encoder.set_buffer(4, Some(&self.length_buffer), 0);
            encoder.dispatch_threads(
                MTLSize::new(THREAD_COUNT, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let count = unsafe {
            let count_ptr = self.token_count_buffer.contents() as *const u32;
            (*count_ptr as usize).min(MAX_TOKENS)
        };

        let tokens_ptr = self.token_buffer.contents() as *const Token;
        let mut tokens: Vec<Token> = (0..count)
            .map(|i| unsafe { *tokens_ptr.add(i) })
            .collect();

        // Sort tokens by start position (GPU extraction may produce out-of-order)
        tokens.sort_by_key(|t| t.start);

        tokens
    }

    /// Get the underlying Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> GpuTokenizer {
        let device = Device::system_default().expect("No Metal device");
        GpuTokenizer::new(&device).expect("Failed to create tokenizer")
    }

    #[test]
    fn test_simple_tag() {
        let mut tokenizer = setup();
        let html = b"<div>hello</div>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 3, "Expected 3 tokens, got {:?}", tokens);

        assert_eq!(tokens[0].token_type, TOKEN_TAG_OPEN);
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[0].text(html), b"<div>");

        assert_eq!(tokens[1].token_type, TOKEN_TEXT);
        assert_eq!(tokens[1].text(html), b"hello");

        assert_eq!(tokens[2].token_type, TOKEN_TAG_CLOSE);
        assert_eq!(tokens[2].text(html), b"</div>");
    }

    #[test]
    fn test_self_closing_tag() {
        let mut tokenizer = setup();
        let html = b"<br/><img src=\"x\"/>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 2, "Expected 2 tokens, got {:?}", tokens);
        assert_eq!(tokens[0].token_type, TOKEN_TAG_SELF);
        assert_eq!(tokens[0].text(html), b"<br/>");
        assert_eq!(tokens[1].token_type, TOKEN_TAG_SELF);
    }

    #[test]
    fn test_nested_tags() {
        let mut tokenizer = setup();
        let html = b"<div><p>text</p></div>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 5, "Expected 5 tokens, got {:?}", tokens);
        assert_eq!(tokens[0].token_type, TOKEN_TAG_OPEN); // <div>
        assert_eq!(tokens[1].token_type, TOKEN_TAG_OPEN); // <p>
        assert_eq!(tokens[2].token_type, TOKEN_TEXT); // text
        assert_eq!(tokens[3].token_type, TOKEN_TAG_CLOSE); // </p>
        assert_eq!(tokens[4].token_type, TOKEN_TAG_CLOSE); // </div>
    }

    #[test]
    fn test_attributes_with_quotes() {
        let mut tokenizer = setup();
        let html = b"<div class=\"foo\" id='bar'>content</div>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 3, "Expected 3 tokens, got {:?}", tokens);
        assert_eq!(tokens[0].token_type, TOKEN_TAG_OPEN);
        assert_eq!(tokens[0].text(html), b"<div class=\"foo\" id='bar'>");
        assert_eq!(tokens[1].token_type, TOKEN_TEXT);
        assert_eq!(tokens[1].text(html), b"content");
    }

    #[test]
    fn test_attributes_with_angle_bracket() {
        let mut tokenizer = setup();
        // < inside quotes should not start a new tag
        let html = b"<div data-value=\"a<b\">text</div>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 3, "Expected 3 tokens, got {:?}", tokens);
        assert_eq!(tokens[0].token_type, TOKEN_TAG_OPEN);
        assert_eq!(tokens[0].text(html), b"<div data-value=\"a<b\">");
    }

    #[test]
    fn test_comment() {
        let mut tokenizer = setup();
        let html = b"<div><!-- comment --></div>";
        let tokens = tokenizer.tokenize(html);

        let comment_token = tokens.iter().find(|t| t.token_type == TOKEN_COMMENT);
        assert!(
            comment_token.is_some(),
            "Expected comment token, got {:?}",
            tokens
        );
        assert_eq!(comment_token.unwrap().text(html), b"<!-- comment -->");
    }

    #[test]
    fn test_doctype() {
        let mut tokenizer = setup();
        let html = b"<!DOCTYPE html><html></html>";
        let tokens = tokenizer.tokenize(html);

        assert!(!tokens.is_empty());
        assert_eq!(
            tokens[0].token_type, TOKEN_DOCTYPE,
            "Expected DOCTYPE, got {:?}",
            tokens[0]
        );
    }

    #[test]
    fn test_whitespace_preservation() {
        let mut tokenizer = setup();
        let html = b"<pre>  spaces  \n  newlines  </pre>";
        let tokens = tokenizer.tokenize(html);

        let text_token = tokens.iter().find(|t| t.token_type == TOKEN_TEXT);
        assert!(text_token.is_some());
        let text = text_token.unwrap().text(html);
        assert!(text.contains(&b' '), "Should preserve spaces");
        assert!(text.contains(&b'\n'), "Should preserve newlines");
    }

    #[test]
    fn test_empty_document() {
        let mut tokenizer = setup();
        let html = b"";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_text_only() {
        let mut tokenizer = setup();
        let html = b"just plain text";
        let tokens = tokenizer.tokenize(html);

        // Text without tags should produce a text token
        assert!(
            tokens.len() <= 1,
            "Expected 0 or 1 token, got {:?}",
            tokens
        );
        if !tokens.is_empty() {
            assert_eq!(tokens[0].token_type, TOKEN_TEXT);
        }
    }

    #[test]
    fn test_many_small_tags() {
        let mut tokenizer = setup();
        let html = b"<a><b><c><d><e><f><g><h><i><j></j></i></h></g></f></e></d></c></b></a>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 20, "Expected 20 tokens, got {:?}", tokens);

        // Count open and close tags
        let open_count = tokens.iter().filter(|t| t.token_type == TOKEN_TAG_OPEN).count();
        let close_count = tokens.iter().filter(|t| t.token_type == TOKEN_TAG_CLOSE).count();

        assert_eq!(open_count, 10);
        assert_eq!(close_count, 10);
    }

    #[test]
    fn test_adjacent_tags() {
        let mut tokenizer = setup();
        let html = b"<div></div><span></span>";
        let tokens = tokenizer.tokenize(html);

        assert_eq!(tokens.len(), 4, "Expected 4 tokens, got {:?}", tokens);
    }

    #[test]
    fn test_multiline_html() {
        let mut tokenizer = setup();
        let html = b"<div>\n  <p>\n    Hello\n  </p>\n</div>";
        let tokens = tokenizer.tokenize(html);

        // Should have div open, p open, text, p close, div close
        // (whitespace-only text nodes may or may not be included)
        assert!(tokens.len() >= 4, "Expected at least 4 tokens, got {:?}", tokens);

        let tag_opens = tokens.iter().filter(|t| t.token_type == TOKEN_TAG_OPEN).count();
        let tag_closes = tokens.iter().filter(|t| t.token_type == TOKEN_TAG_CLOSE).count();

        assert_eq!(tag_opens, 2);
        assert_eq!(tag_closes, 2);
    }

    #[test]
    fn test_chunk_boundary_tag() {
        let mut tokenizer = setup();
        // Create HTML where a tag might span chunk boundary
        let mut html = vec![b'x'; 1020];
        html.extend_from_slice(b"<div>test</div>");

        let tokens = tokenizer.tokenize(&html);

        // Should find text (the x's), then the div tags
        let tag_open = tokens.iter().find(|t| t.token_type == TOKEN_TAG_OPEN);
        assert!(tag_open.is_some(), "Should find tag even at chunk boundary");
    }

    // Helper to generate HTML of approximately target size
    fn generate_html(target_size: usize) -> Vec<u8> {
        let mut html = Vec::with_capacity(target_size);
        html.extend_from_slice(b"<!DOCTYPE html><html><body>");

        while html.len() < target_size.saturating_sub(100) {
            html.extend_from_slice(b"<div class=\"item\">");
            html.extend_from_slice(b"<p>Some text content here.</p>");
            html.extend_from_slice(b"</div>");
        }

        html.extend_from_slice(b"</body></html>");
        html
    }

    #[test]
    fn test_performance_1kb() {
        let mut tokenizer = setup();
        let html = generate_html(1024);

        // Warmup run (GPU initialization overhead)
        let _ = tokenizer.tokenize(&html);

        let start = std::time::Instant::now();
        let tokens = tokenizer.tokenize(&html);
        let elapsed = start.elapsed();

        println!(
            "1KB: {} tokens in {:?} ({:.2} tokens/ms)",
            tokens.len(),
            elapsed,
            tokens.len() as f64 / elapsed.as_secs_f64() / 1000.0
        );
        assert!(
            elapsed.as_micros() < 5000,
            "Should tokenize 1KB in <5ms (after warmup), took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_performance_100kb() {
        let mut tokenizer = setup();
        let html = generate_html(100 * 1024);

        // Warmup
        let _ = tokenizer.tokenize(&html);

        let start = std::time::Instant::now();
        let tokens = tokenizer.tokenize(&html);
        let elapsed = start.elapsed();

        println!(
            "100KB: {} tokens in {:?} ({:.2} MB/s)",
            tokens.len(),
            elapsed,
            100.0 / 1024.0 / elapsed.as_secs_f64()
        );
        assert!(
            elapsed.as_millis() < 50,
            "Should tokenize 100KB in <50ms, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_performance_1mb() {
        let mut tokenizer = setup();
        let html = generate_html(1024 * 1024);

        // Warmup
        let _ = tokenizer.tokenize(&html);

        let start = std::time::Instant::now();
        let tokens = tokenizer.tokenize(&html);
        let elapsed = start.elapsed();

        println!(
            "1MB: {} tokens in {:?} ({:.2} MB/s)",
            tokens.len(),
            elapsed,
            1.0 / elapsed.as_secs_f64()
        );
        assert!(
            elapsed.as_millis() < 100,
            "Should tokenize 1MB in <100ms, took {:?}",
            elapsed
        );
    }
}
