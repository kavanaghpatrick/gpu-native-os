//! GPU-Native HTML Parser
//!
//! Converts token stream into a flat element array representing the DOM tree.
//! Uses three GPU passes:
//! 1. Element allocation (parallel)
//! 2. Tree construction (sequential - stack-based)
//! 3. Text extraction (parallel)

use metal::*;
use super::tokenizer::Token;

// Element types (must match Metal shader)
pub const ELEM_UNKNOWN: u32 = 0;
pub const ELEM_DIV: u32 = 1;
pub const ELEM_SPAN: u32 = 2;
pub const ELEM_P: u32 = 3;
pub const ELEM_A: u32 = 4;
pub const ELEM_H1: u32 = 5;
pub const ELEM_H2: u32 = 6;
pub const ELEM_H3: u32 = 7;
pub const ELEM_H4: u32 = 8;
pub const ELEM_H5: u32 = 9;
pub const ELEM_H6: u32 = 10;
pub const ELEM_UL: u32 = 11;
pub const ELEM_OL: u32 = 12;
pub const ELEM_LI: u32 = 13;
pub const ELEM_IMG: u32 = 14;
pub const ELEM_BR: u32 = 15;
pub const ELEM_HR: u32 = 16;
pub const ELEM_TABLE: u32 = 17;
pub const ELEM_TR: u32 = 18;
pub const ELEM_TD: u32 = 19;
pub const ELEM_TH: u32 = 20;
pub const ELEM_THEAD: u32 = 21;
pub const ELEM_TBODY: u32 = 22;
pub const ELEM_FORM: u32 = 23;
pub const ELEM_INPUT: u32 = 24;
pub const ELEM_BUTTON: u32 = 25;
pub const ELEM_TEXTAREA: u32 = 26;
pub const ELEM_SELECT: u32 = 27;
pub const ELEM_OPTION: u32 = 28;
pub const ELEM_LABEL: u32 = 29;
pub const ELEM_NAV: u32 = 30;
pub const ELEM_HEADER: u32 = 31;
pub const ELEM_FOOTER: u32 = 32;
pub const ELEM_MAIN: u32 = 33;
pub const ELEM_SECTION: u32 = 34;
pub const ELEM_ARTICLE: u32 = 35;
pub const ELEM_ASIDE: u32 = 36;
pub const ELEM_PRE: u32 = 37;
pub const ELEM_CODE: u32 = 38;
pub const ELEM_BLOCKQUOTE: u32 = 39;
pub const ELEM_STRONG: u32 = 40;
pub const ELEM_EM: u32 = 41;
pub const ELEM_B: u32 = 42;
pub const ELEM_I: u32 = 43;
pub const ELEM_U: u32 = 44;
pub const ELEM_SMALL: u32 = 45;
pub const ELEM_SUB: u32 = 46;
pub const ELEM_SUP: u32 = 47;
pub const ELEM_TEXT: u32 = 100;
pub const ELEM_HTML: u32 = 101;
pub const ELEM_HEAD: u32 = 102;
pub const ELEM_BODY: u32 = 103;
pub const ELEM_TITLE: u32 = 104;
pub const ELEM_META: u32 = 105;
pub const ELEM_LINK: u32 = 106;
pub const ELEM_STYLE: u32 = 107;
pub const ELEM_SCRIPT: u32 = 108;

// Buffer sizes
pub const MAX_ELEMENTS: usize = 16384;
pub const MAX_TEXT_SIZE: usize = 512 * 1024;
pub const MAX_STACK_DEPTH: usize = 256;
const THREAD_COUNT: u64 = 1024;

/// An element in the parsed DOM tree
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Element {
    pub element_type: u32,
    pub parent: i32,
    pub first_child: i32,
    pub next_sibling: i32,
    pub prev_sibling: i32,  // Issue #128: Enable O(1) cumulative height lookup
    pub text_start: u32,
    pub text_length: u32,
    pub token_index: u32,
}

impl Element {
    /// Get the element type as a string for debugging
    pub fn type_name(&self) -> &'static str {
        match self.element_type {
            ELEM_DIV => "div",
            ELEM_SPAN => "span",
            ELEM_P => "p",
            ELEM_A => "a",
            ELEM_H1 => "h1",
            ELEM_H2 => "h2",
            ELEM_H3 => "h3",
            ELEM_UL => "ul",
            ELEM_OL => "ol",
            ELEM_LI => "li",
            ELEM_IMG => "img",
            ELEM_BR => "br",
            ELEM_HR => "hr",
            ELEM_TABLE => "table",
            ELEM_TR => "tr",
            ELEM_TD => "td",
            ELEM_TH => "th",
            ELEM_FORM => "form",
            ELEM_INPUT => "input",
            ELEM_BUTTON => "button",
            ELEM_TEXT => "#text",
            ELEM_HTML => "html",
            ELEM_HEAD => "head",
            ELEM_BODY => "body",
            ELEM_NAV => "nav",
            ELEM_HEADER => "header",
            ELEM_FOOTER => "footer",
            ELEM_MAIN => "main",
            ELEM_SECTION => "section",
            ELEM_ARTICLE => "article",
            ELEM_STRONG => "strong",
            ELEM_EM => "em",
            ELEM_B => "b",
            ELEM_I => "i",
            ELEM_CODE => "code",
            ELEM_PRE => "pre",
            _ => "unknown",
        }
    }

    /// Get text content from the text buffer
    pub fn text<'a>(&self, text_buffer: &'a [u8]) -> &'a [u8] {
        if self.text_length == 0 {
            return &[];
        }
        let start = self.text_start as usize;
        let end = start + self.text_length as usize;
        if end <= text_buffer.len() {
            &text_buffer[start..end]
        } else {
            &[]
        }
    }

    /// Check if this element has children
    pub fn has_children(&self) -> bool {
        self.first_child >= 0
    }

    /// Check if this element has a next sibling
    pub fn has_next_sibling(&self) -> bool {
        self.next_sibling >= 0
    }

    /// Check if this element has a previous sibling
    pub fn has_prev_sibling(&self) -> bool {
        self.prev_sibling >= 0
    }

    /// Check if this is a text node
    pub fn is_text(&self) -> bool {
        self.element_type == ELEM_TEXT
    }
}

/// Metal shader source for the parser
const PARSER_SHADER: &str = include_str!("parser.metal");

/// GPU-Native HTML Parser
pub struct GpuParser {
    device: Device,
    command_queue: CommandQueue,
    allocate_pipeline: ComputePipelineState,
    tree_pipeline: ComputePipelineState,
    text_pipeline: ComputePipelineState,

    // Input buffers (reused from tokenizer or freshly allocated)
    token_buffer: Buffer,
    html_buffer: Buffer,

    // Working buffers
    token_to_element_buffer: Buffer,
    element_count_buffer: Buffer,
    token_count_buffer: Buffer,

    // Output buffers
    element_buffer: Buffer,
    text_buffer: Buffer,
    text_offset_buffer: Buffer,
}

impl GpuParser {
    /// Create a new GPU parser
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(PARSER_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile parser shader: {}", e))?;

        // Create pipelines
        let allocate_fn = library
            .get_function("parse_allocate_elements", None)
            .map_err(|e| format!("Failed to find parse_allocate_elements: {}", e))?;
        let tree_fn = library
            .get_function("parse_build_tree", None)
            .map_err(|e| format!("Failed to find parse_build_tree: {}", e))?;
        let text_fn = library
            .get_function("parse_extract_text", None)
            .map_err(|e| format!("Failed to find parse_extract_text: {}", e))?;

        let allocate_pipeline = device
            .new_compute_pipeline_state_with_function(&allocate_fn)
            .map_err(|e| format!("Failed to create allocate pipeline: {}", e))?;
        let tree_pipeline = device
            .new_compute_pipeline_state_with_function(&tree_fn)
            .map_err(|e| format!("Failed to create tree pipeline: {}", e))?;
        let text_pipeline = device
            .new_compute_pipeline_state_with_function(&text_fn)
            .map_err(|e| format!("Failed to create text pipeline: {}", e))?;

        // Allocate buffers
        let max_tokens = 65536usize;
        let token_buffer = device.new_buffer(
            (max_tokens * std::mem::size_of::<Token>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let html_buffer = device.new_buffer(
            (1024 * 1024) as u64,  // 1MB
            MTLResourceOptions::StorageModeShared,
        );
        let token_to_element_buffer = device.new_buffer(
            (max_tokens * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let element_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<Element>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let text_buffer = device.new_buffer(
            MAX_TEXT_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let element_count_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let token_count_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let text_offset_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        Ok(Self {
            device: device.clone(),
            command_queue,
            allocate_pipeline,
            tree_pipeline,
            text_pipeline,
            token_buffer,
            html_buffer,
            token_to_element_buffer,
            element_buffer,
            text_buffer,
            element_count_buffer,
            token_count_buffer,
            text_offset_buffer,
        })
    }

    /// Parse tokens into an element tree
    pub fn parse(&mut self, tokens: &[Token], html: &[u8]) -> (Vec<Element>, Vec<u8>) {
        if tokens.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let token_count = tokens.len().min(65536);
        let html_len = html.len().min(1024 * 1024);

        // Copy tokens to GPU
        unsafe {
            std::ptr::copy_nonoverlapping(
                tokens.as_ptr(),
                self.token_buffer.contents() as *mut Token,
                token_count,
            );
        }

        // Copy HTML to GPU
        unsafe {
            std::ptr::copy_nonoverlapping(
                html.as_ptr(),
                self.html_buffer.contents() as *mut u8,
                html_len,
            );
        }

        // Set token count
        unsafe {
            let ptr = self.token_count_buffer.contents() as *mut u32;
            *ptr = token_count as u32;
        }

        // Reset counters
        unsafe {
            let ptr = self.element_count_buffer.contents() as *mut u32;
            *ptr = 0;
            let ptr = self.text_offset_buffer.contents() as *mut u32;
            *ptr = 0;
        }

        // Clear token_to_element buffer
        unsafe {
            std::ptr::write_bytes(
                self.token_to_element_buffer.contents() as *mut i32,
                0xFF,  // -1 in two's complement
                token_count,
            );
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 2A: Allocate elements
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.allocate_pipeline);
            encoder.set_buffer(0, Some(&self.token_buffer), 0);
            encoder.set_buffer(1, Some(&self.token_to_element_buffer), 0);
            encoder.set_buffer(2, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(3, Some(&self.token_count_buffer), 0);
            encoder.dispatch_threads(
                MTLSize::new(THREAD_COUNT, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 2B: Build tree (sequential but on GPU)
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.tree_pipeline);
            encoder.set_buffer(0, Some(&self.token_buffer), 0);
            encoder.set_buffer(1, Some(&self.html_buffer), 0);
            encoder.set_buffer(2, Some(&self.token_to_element_buffer), 0);
            encoder.set_buffer(3, Some(&self.element_buffer), 0);
            encoder.set_buffer(4, Some(&self.token_count_buffer), 0);
            // Threadgroup memory for parent stack and last_child tracking
            encoder.set_threadgroup_memory_length(0, (MAX_STACK_DEPTH * 4) as u64);  // parent_stack
            encoder.set_threadgroup_memory_length(1, (MAX_STACK_DEPTH * 4) as u64);  // last_child
            encoder.dispatch_threads(
                MTLSize::new(THREAD_COUNT, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 2C: Extract text
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.text_pipeline);
            encoder.set_buffer(0, Some(&self.token_buffer), 0);
            encoder.set_buffer(1, Some(&self.html_buffer), 0);
            encoder.set_buffer(2, Some(&self.element_buffer), 0);
            encoder.set_buffer(3, Some(&self.text_buffer), 0);
            encoder.set_buffer(4, Some(&self.text_offset_buffer), 0);
            encoder.set_buffer(5, Some(&self.element_count_buffer), 0);
            encoder.dispatch_threads(
                MTLSize::new(THREAD_COUNT, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let element_count = unsafe {
            let ptr = self.element_count_buffer.contents() as *const u32;
            (*ptr as usize).min(MAX_ELEMENTS)
        };

        let text_size = unsafe {
            let ptr = self.text_offset_buffer.contents() as *const u32;
            (*ptr as usize).min(MAX_TEXT_SIZE)
        };

        let elements: Vec<Element> = unsafe {
            let ptr = self.element_buffer.contents() as *const Element;
            (0..element_count).map(|i| *ptr.add(i)).collect()
        };

        let text: Vec<u8> = unsafe {
            let ptr = self.text_buffer.contents() as *const u8;
            (0..text_size).map(|i| *ptr.add(i)).collect()
        };

        (elements, text)
    }

    /// Get the underlying Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tokenizer::{GpuTokenizer, TOKEN_TAG_OPEN, TOKEN_TAG_CLOSE, TOKEN_TEXT};

    fn setup() -> (GpuTokenizer, GpuParser) {
        let device = Device::system_default().expect("No Metal device");
        let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
        let parser = GpuParser::new(&device).expect("Failed to create parser");
        (tokenizer, parser)
    }

    #[test]
    fn test_simple_tree() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<div><p>hello</p></div>";
        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);

        assert_eq!(elements.len(), 3, "Expected 3 elements, got {:?}", elements);

        // div is root
        assert_eq!(elements[0].element_type, ELEM_DIV);
        assert_eq!(elements[0].parent, -1);
        assert_eq!(elements[0].first_child, 1);

        // p is child of div
        assert_eq!(elements[1].element_type, ELEM_P);
        assert_eq!(elements[1].parent, 0);
        assert_eq!(elements[1].first_child, 2);

        // text is child of p
        assert_eq!(elements[2].element_type, ELEM_TEXT);
        assert_eq!(elements[2].parent, 1);
        assert_eq!(elements[2].text(&text), b"hello");
    }

    #[test]
    fn test_siblings() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<ul><li>a</li><li>b</li><li>c</li></ul>";
        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);

        // Find ul
        let ul_idx = elements.iter()
            .position(|e| e.element_type == ELEM_UL)
            .expect("Should find ul");

        // Count li children via sibling chain
        let mut li_count = 0;
        let mut current = elements[ul_idx].first_child;
        while current >= 0 {
            let elem = &elements[current as usize];
            if elem.element_type == ELEM_LI {
                li_count += 1;
            }
            current = elem.next_sibling;
        }

        assert_eq!(li_count, 3, "Expected 3 li elements");
    }

    #[test]
    fn test_deep_nesting() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<div><div><div><div><p>deep</p></div></div></div></div>";
        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);

        // Find text element and count depth
        let text_idx = elements.iter()
            .position(|e| e.element_type == ELEM_TEXT)
            .expect("Should find text");

        let mut depth = 0;
        let mut current = elements[text_idx].parent;
        while current >= 0 {
            depth += 1;
            current = elements[current as usize].parent;
        }

        assert_eq!(depth, 5, "Text should be at depth 5");
    }

    #[test]
    fn test_self_closing() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<div><br/><img/></div>";
        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);

        // br and img should have no children
        let br = elements.iter().find(|e| e.element_type == ELEM_BR);
        let img = elements.iter().find(|e| e.element_type == ELEM_IMG);

        assert!(br.is_some(), "Should find br");
        assert!(img.is_some(), "Should find img");
        assert!(br.unwrap().first_child < 0, "br should have no children");
        assert!(img.unwrap().first_child < 0, "img should have no children");
    }

    #[test]
    fn test_mixed_content() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<p>Hello <b>bold</b> world</p>";
        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);

        // Find p
        let p_idx = elements.iter()
            .position(|e| e.element_type == ELEM_P)
            .expect("Should find p");

        // Count children
        let mut child_count = 0;
        let mut current = elements[p_idx].first_child;
        while current >= 0 {
            child_count += 1;
            current = elements[current as usize].next_sibling;
        }

        // p should have: text("Hello "), b, text(" world")
        assert!(child_count >= 2, "p should have multiple children, got {}", child_count);
    }

    #[test]
    fn test_empty_document() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"";
        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);

        assert_eq!(elements.len(), 0);
        assert_eq!(text.len(), 0);
    }

    #[test]
    fn test_text_only() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<p>just text</p>";
        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);

        let text_elem = elements.iter()
            .find(|e| e.element_type == ELEM_TEXT)
            .expect("Should find text element");

        assert_eq!(text_elem.text(&text), b"just text");
    }

    #[test]
    fn test_various_tags() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<html><head><title>Test</title></head><body><h1>Header</h1><nav><a>Link</a></nav></body></html>";
        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);

        // Check various tag types are recognized
        assert!(elements.iter().any(|e| e.element_type == ELEM_HTML));
        assert!(elements.iter().any(|e| e.element_type == ELEM_HEAD));
        assert!(elements.iter().any(|e| e.element_type == ELEM_TITLE));
        assert!(elements.iter().any(|e| e.element_type == ELEM_BODY));
        assert!(elements.iter().any(|e| e.element_type == ELEM_H1));
        assert!(elements.iter().any(|e| e.element_type == ELEM_NAV));
        assert!(elements.iter().any(|e| e.element_type == ELEM_A));
    }

    #[test]
    fn test_table_structure() {
        let (mut tokenizer, mut parser) = setup();
        let html = b"<table><tr><td>cell</td></tr></table>";
        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);

        assert!(elements.iter().any(|e| e.element_type == ELEM_TABLE));
        assert!(elements.iter().any(|e| e.element_type == ELEM_TR));
        assert!(elements.iter().any(|e| e.element_type == ELEM_TD));
    }

    fn generate_html(element_count: usize) -> Vec<u8> {
        let mut html = Vec::new();
        html.extend_from_slice(b"<div>");
        for i in 0..element_count {
            html.extend_from_slice(format!("<p>Item {}</p>", i).as_bytes());
        }
        html.extend_from_slice(b"</div>");
        html
    }

    #[test]
    fn test_performance_1k_elements() {
        let (mut tokenizer, mut parser) = setup();
        let html = generate_html(500);  // 500 p elements = ~1000 total with text nodes
        let tokens = tokenizer.tokenize(&html);

        // Warmup
        let _ = parser.parse(&tokens, &html);

        let start = std::time::Instant::now();
        let (elements, _) = parser.parse(&tokens, &html);
        let elapsed = start.elapsed();

        println!(
            "~1K elements: {} elements in {:?} ({:.2} elements/ms)",
            elements.len(),
            elapsed,
            elements.len() as f64 / elapsed.as_secs_f64() / 1000.0
        );

        assert!(
            elapsed.as_millis() < 10,
            "Should parse ~1K elements in <10ms, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_performance_5k_elements() {
        let (mut tokenizer, mut parser) = setup();
        let html = generate_html(2500);  // 2500 p elements = ~5000 with text
        let tokens = tokenizer.tokenize(&html);

        // Warmup
        let _ = parser.parse(&tokens, &html);

        let start = std::time::Instant::now();
        let (elements, _) = parser.parse(&tokens, &html);
        let elapsed = start.elapsed();

        println!(
            "~5K elements: {} elements in {:?} ({:.2} elements/ms)",
            elements.len(),
            elapsed,
            elements.len() as f64 / elapsed.as_secs_f64() / 1000.0
        );

        // With O(1) child linking optimization, should be much faster
        assert!(
            elapsed.as_millis() < 50,
            "Should parse ~5K elements in <50ms, took {:?}",
            elapsed
        );
    }
}
