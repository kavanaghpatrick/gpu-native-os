# Issue #102: HTML5 Parser Integration

## Summary
Integrate html5ever for spec-compliant HTML parsing, replacing our simple regex-based parser with a production-ready solution that handles real-world HTML.

## Motivation
Current parser (`document/parser.rs`) uses simple string matching that fails on:
- Malformed HTML (unclosed tags, incorrect nesting)
- HTML entities (`&amp;`, `&nbsp;`, etc.)
- Script/style content escaping
- DOCTYPE handling
- Comments and CDATA sections

html5ever is the same parser used by Servo and Firefox, battle-tested on billions of web pages.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         html5ever Parser            │
                    │  (Streaming, spec-compliant)        │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │      TreeSink Implementation        │
                    │   (Builds GPU-friendly DOM)         │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    │  GPUNode    │        │ Attributes  │        │   Strings   │
    │  Buffer     │        │   Buffer    │        │    Pool     │
    └─────────────┘        └─────────────┘        └─────────────┘
```

## Data Structures

### Rust Side (CPU)

```rust
// GPU-optimized node representation
#[repr(C)]
pub struct GPUNode {
    pub parent_idx: u32,
    pub first_child_idx: u32,
    pub last_child_idx: u32,
    pub next_sibling_idx: u32,
    pub prev_sibling_idx: u32,
    pub element_type: u16,      // Encoded element type
    pub flags: u16,             // Node flags (16 bits)
    pub attribute_offset: u32,  // Offset into attribute buffer
    pub attribute_count: u16,
    pub text_offset: u32,       // Offset into string pool (for text nodes)
    pub text_length: u16,
    pub _padding: [u8; 2],
}

// Attribute storage
#[repr(C)]
pub struct GPUAttribute {
    pub name_offset: u32,       // Offset into string pool
    pub name_length: u16,
    pub value_offset: u32,
    pub value_length: u16,
    pub name_hash: u32,         // For fast matching (class, id, etc.)
}

// Element type encoding (matches CSS type selectors)
pub const ELEM_UNKNOWN: u16 = 0;
pub const ELEM_HTML: u16 = 1;
pub const ELEM_HEAD: u16 = 2;
pub const ELEM_BODY: u16 = 3;
pub const ELEM_DIV: u16 = 4;
pub const ELEM_SPAN: u16 = 5;
pub const ELEM_P: u16 = 6;
pub const ELEM_A: u16 = 7;
pub const ELEM_IMG: u16 = 8;
pub const ELEM_H1: u16 = 9;
pub const ELEM_H2: u16 = 10;
// ... 256 common elements
pub const ELEM_TEXT: u16 = 255;

// TreeSink implementation
pub struct GPUTreeSink {
    nodes: Vec<GPUNode>,
    attributes: Vec<GPUAttribute>,
    string_pool: Vec<u8>,
    element_stack: Vec<u32>,  // Stack of open element indices
    current_parent: u32,
}

impl TreeSink for GPUTreeSink {
    type Handle = u32;  // Node index
    type Output = GPUDocument;

    fn create_element(&mut self, name: QualName, attrs: Vec<Attribute>) -> u32 {
        let idx = self.nodes.len() as u32;
        let element_type = encode_element_type(&name.local);

        // Store attributes
        let attr_offset = self.attributes.len() as u32;
        for attr in &attrs {
            self.attributes.push(GPUAttribute {
                name_offset: self.intern_string(&attr.name.local),
                name_length: attr.name.local.len() as u16,
                value_offset: self.intern_string(&attr.value),
                value_length: attr.value.len() as u16,
                name_hash: hash_attribute_name(&attr.name.local),
            });
        }

        self.nodes.push(GPUNode {
            parent_idx: self.current_parent,
            first_child_idx: INVALID_IDX,
            last_child_idx: INVALID_IDX,
            next_sibling_idx: INVALID_IDX,
            prev_sibling_idx: INVALID_IDX,
            element_type,
            flags: 0,
            attribute_offset: attr_offset,
            attribute_count: attrs.len() as u16,
            text_offset: 0,
            text_length: 0,
            _padding: [0; 2],
        });

        idx
    }

    fn append(&mut self, parent: &u32, child: NodeOrText<u32>) {
        match child {
            NodeOrText::AppendNode(child_idx) => {
                self.append_child(*parent, child_idx);
            }
            NodeOrText::AppendText(text) => {
                let text_node = self.create_text_node(&text);
                self.append_child(*parent, text_node);
            }
        }
    }

    // ... other TreeSink methods
}
```

### Metal Side (GPU)

```metal
// GPU node structure (must match Rust repr(C))
struct GPUNode {
    uint parent_idx;
    uint first_child_idx;
    uint last_child_idx;
    uint next_sibling_idx;
    uint prev_sibling_idx;
    ushort element_type;
    ushort flags;
    uint attribute_offset;
    ushort attribute_count;
    uint text_offset;
    ushort text_length;
    uchar2 _padding;
};

struct GPUAttribute {
    uint name_offset;
    ushort name_length;
    uint value_offset;
    ushort value_length;
    uint name_hash;
};

// Fast element type check
constant ushort ELEM_DIV = 4;
constant ushort ELEM_SPAN = 5;
constant ushort ELEM_P = 6;
// ...

// Helper to traverse DOM tree on GPU
uint get_ancestor_at_depth(
    device GPUNode* nodes,
    uint node_idx,
    uint target_depth
) {
    uint current = node_idx;
    uint depth = 0;

    // First, find current depth
    uint temp = current;
    while (temp != INVALID_IDX) {
        depth++;
        temp = nodes[temp].parent_idx;
    }

    // Walk up to target depth
    while (depth > target_depth && current != INVALID_IDX) {
        current = nodes[current].parent_idx;
        depth--;
    }

    return current;
}
```

## Implementation

### Step 1: Add Dependencies

```toml
# Cargo.toml
[dependencies]
html5ever = "0.38"
markup5ever = "0.38"
tendril = "0.4"  # Efficient string handling
```

### Step 2: TreeSink Implementation

```rust
// src/gpu_os/document/html_parser.rs

use html5ever::{
    parse_document, parse_fragment,
    tendril::TendrilSink,
    tree_builder::{NodeOrText, TreeSink, QuirksMode, ElementFlags},
    QualName, Attribute, ExpandedName,
};
use std::borrow::Cow;

impl GPUTreeSink {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1024),
            attributes: Vec::with_capacity(4096),
            string_pool: Vec::with_capacity(64 * 1024),
            element_stack: Vec::with_capacity(64),
            current_parent: INVALID_IDX,
        }
    }

    pub fn parse(html: &str) -> GPUDocument {
        let sink = GPUTreeSink::new();
        let parser = parse_document(sink, Default::default());
        let document = parser.one(html);
        document
    }

    fn intern_string(&mut self, s: &str) -> u32 {
        let offset = self.string_pool.len() as u32;
        self.string_pool.extend_from_slice(s.as_bytes());
        offset
    }

    fn append_child(&mut self, parent: u32, child: u32) {
        // Update child's parent
        self.nodes[child as usize].parent_idx = parent;

        // Update parent's child pointers
        let parent_node = &mut self.nodes[parent as usize];
        if parent_node.first_child_idx == INVALID_IDX {
            parent_node.first_child_idx = child;
            parent_node.last_child_idx = child;
        } else {
            // Link to previous last child
            let prev_last = parent_node.last_child_idx;
            self.nodes[prev_last as usize].next_sibling_idx = child;
            self.nodes[child as usize].prev_sibling_idx = prev_last;
            parent_node.last_child_idx = child;
        }
    }
}

fn encode_element_type(local_name: &str) -> u16 {
    match local_name {
        "html" => ELEM_HTML,
        "head" => ELEM_HEAD,
        "body" => ELEM_BODY,
        "div" => ELEM_DIV,
        "span" => ELEM_SPAN,
        "p" => ELEM_P,
        "a" => ELEM_A,
        "img" => ELEM_IMG,
        "h1" => ELEM_H1,
        "h2" => ELEM_H2,
        "h3" => ELEM_H3,
        "h4" => ELEM_H4,
        "h5" => ELEM_H5,
        "h6" => ELEM_H6,
        "ul" => ELEM_UL,
        "ol" => ELEM_OL,
        "li" => ELEM_LI,
        "table" => ELEM_TABLE,
        "tr" => ELEM_TR,
        "td" => ELEM_TD,
        "th" => ELEM_TH,
        "form" => ELEM_FORM,
        "input" => ELEM_INPUT,
        "button" => ELEM_BUTTON,
        "nav" => ELEM_NAV,
        "header" => ELEM_HEADER,
        "footer" => ELEM_FOOTER,
        "article" => ELEM_ARTICLE,
        "section" => ELEM_SECTION,
        "aside" => ELEM_ASIDE,
        "main" => ELEM_MAIN,
        _ => ELEM_UNKNOWN,
    }
}

fn hash_attribute_name(name: &str) -> u32 {
    // FNV-1a hash for fast comparison
    let mut hash: u32 = 2166136261;
    for byte in name.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}
```

### Step 3: GPU Buffer Upload

```rust
// src/gpu_os/document/gpu_dom.rs

pub struct GPUDom {
    node_buffer: metal::Buffer,
    attribute_buffer: metal::Buffer,
    string_pool_buffer: metal::Buffer,
    node_count: u32,
}

impl GPUDom {
    pub fn from_document(device: &metal::DeviceRef, doc: &GPUDocument) -> Self {
        let node_buffer = device.new_buffer_with_data(
            doc.nodes.as_ptr() as *const _,
            (doc.nodes.len() * std::mem::size_of::<GPUNode>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let attribute_buffer = device.new_buffer_with_data(
            doc.attributes.as_ptr() as *const _,
            (doc.attributes.len() * std::mem::size_of::<GPUAttribute>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let string_pool_buffer = device.new_buffer_with_data(
            doc.string_pool.as_ptr() as *const _,
            doc.string_pool.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Self {
            node_buffer,
            attribute_buffer,
            string_pool_buffer,
            node_count: doc.nodes.len() as u32,
        }
    }
}
```

## Benchmarks

### Benchmark 1: Parse Time Comparison

```rust
// benches/html_parser_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_html_parsing(c: &mut Criterion) {
    let test_cases = vec![
        ("small", include_str!("../test_pages/small.html")),      // ~1KB
        ("medium", include_str!("../test_pages/medium.html")),    // ~50KB
        ("wikipedia", include_str!("../test_pages/wikipedia.html")), // ~500KB
        ("complex", include_str!("../test_pages/complex.html")),  // ~2MB
    ];

    let mut group = c.benchmark_group("html_parsing");

    for (name, html) in test_cases {
        // Current simple parser
        group.bench_with_input(
            BenchmarkId::new("simple_parser", name),
            &html,
            |b, html| b.iter(|| simple_parse(html))
        );

        // html5ever parser
        group.bench_with_input(
            BenchmarkId::new("html5ever", name),
            &html,
            |b, html| b.iter(|| GPUTreeSink::parse(html))
        );
    }

    group.finish();
}

criterion_group!(benches, bench_html_parsing);
criterion_main!(benches);
```

### Benchmark 2: GPU Buffer Upload Time

```rust
fn bench_gpu_upload(c: &mut Criterion) {
    let device = metal::Device::system_default().unwrap();
    let wikipedia = include_str!("../test_pages/wikipedia.html");
    let doc = GPUTreeSink::parse(wikipedia);

    c.bench_function("gpu_dom_upload", |b| {
        b.iter(|| GPUDom::from_document(&device, &doc))
    });
}
```

### Expected Results

| Document Size | Simple Parser | html5ever | GPU Upload |
|---------------|---------------|-----------|------------|
| 1KB (10 nodes) | 0.05ms | 0.1ms | 0.01ms |
| 50KB (500 nodes) | 2ms | 1ms | 0.05ms |
| 500KB (5K nodes) | 20ms | 5ms | 0.2ms |
| 2MB (20K nodes) | 100ms | 15ms | 0.5ms |

## Tests

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_document() {
        let html = "<html><body><div>Hello</div></body></html>";
        let doc = GPUTreeSink::parse(html);

        assert_eq!(doc.nodes.len(), 5); // html, head, body, div, text

        // Check tree structure
        let body = doc.find_element(ELEM_BODY).unwrap();
        let div = doc.nodes[body as usize].first_child_idx;
        assert_eq!(doc.nodes[div as usize].element_type, ELEM_DIV);
    }

    #[test]
    fn test_malformed_html() {
        // html5ever should handle this gracefully
        let html = "<div><p>Unclosed paragraph<div>Nested div</div>";
        let doc = GPUTreeSink::parse(html);

        // Should still produce valid tree
        assert!(doc.nodes.len() > 0);
        assert!(doc.validate_tree());
    }

    #[test]
    fn test_html_entities() {
        let html = "<p>&amp; &lt; &gt; &nbsp;</p>";
        let doc = GPUTreeSink::parse(html);

        let text = doc.get_text_content(doc.find_element(ELEM_P).unwrap());
        assert_eq!(text, "& < > \u{00A0}");
    }

    #[test]
    fn test_attributes() {
        let html = r#"<div id="main" class="container wide" data-value="42"></div>"#;
        let doc = GPUTreeSink::parse(html);

        let div = doc.find_element(ELEM_DIV).unwrap();
        let attrs = doc.get_attributes(div);

        assert_eq!(attrs.len(), 3);
        assert_eq!(doc.get_attribute_value(div, "id"), Some("main"));
        assert_eq!(doc.get_attribute_value(div, "class"), Some("container wide"));
        assert_eq!(doc.get_attribute_value(div, "data-value"), Some("42"));
    }

    #[test]
    fn test_deep_nesting() {
        let html = "<div>".repeat(100) + "deep" + &"</div>".repeat(100);
        let doc = GPUTreeSink::parse(&html);

        assert!(doc.nodes.len() >= 101); // 100 divs + text
        assert!(doc.validate_tree());
    }

    #[test]
    fn test_gpu_upload() {
        let device = metal::Device::system_default().unwrap();
        let html = "<html><body><div>Test</div></body></html>";
        let doc = GPUTreeSink::parse(html);

        let gpu_dom = GPUDom::from_document(&device, &doc);

        assert_eq!(gpu_dom.node_count, doc.nodes.len() as u32);
    }

    #[test]
    fn test_sibling_traversal() {
        let html = "<ul><li>1</li><li>2</li><li>3</li></ul>";
        let doc = GPUTreeSink::parse(html);

        let ul = doc.find_element(ELEM_UL).unwrap();
        let first_li = doc.nodes[ul as usize].first_child_idx;

        // Traverse siblings
        let mut count = 0;
        let mut current = first_li;
        while current != INVALID_IDX {
            count += 1;
            current = doc.nodes[current as usize].next_sibling_idx;
        }

        assert_eq!(count, 3);
    }
}
```

### Integration Tests

```rust
// tests/test_issue_102_html5_parser.rs

#[test]
fn test_wikipedia_parse() {
    let html = std::fs::read_to_string("test_pages/wikipedia.html").unwrap();
    let doc = GPUTreeSink::parse(&html);

    // Verify basic structure
    assert!(doc.find_element(ELEM_HTML).is_some());
    assert!(doc.find_element(ELEM_BODY).is_some());

    // Count elements
    let div_count = doc.count_elements(ELEM_DIV);
    assert!(div_count > 100, "Expected many divs in Wikipedia page");

    // Verify no orphan nodes
    assert!(doc.validate_tree());
}

#[test]
fn test_gpu_dom_consistency() {
    let device = metal::Device::system_default().unwrap();
    let html = std::fs::read_to_string("test_pages/medium.html").unwrap();
    let doc = GPUTreeSink::parse(&html);
    let gpu_dom = GPUDom::from_document(&device, &doc);

    // Read back from GPU and verify
    let gpu_nodes = gpu_dom.read_nodes();

    for (i, (cpu, gpu)) in doc.nodes.iter().zip(gpu_nodes.iter()).enumerate() {
        assert_eq!(cpu.parent_idx, gpu.parent_idx, "Parent mismatch at {}", i);
        assert_eq!(cpu.first_child_idx, gpu.first_child_idx, "First child mismatch at {}", i);
        assert_eq!(cpu.element_type, gpu.element_type, "Element type mismatch at {}", i);
    }
}
```

## Acceptance Criteria

- [ ] html5ever parses all test pages without panics
- [ ] GPU buffer upload completes in <1ms for typical pages
- [ ] Tree structure matches html5ever's DOM
- [ ] All 256 common element types are encoded
- [ ] Attributes are correctly extracted and hashed
- [ ] Text nodes preserve content accurately
- [ ] Sibling/parent/child traversal works correctly
- [ ] Memory usage is reasonable (<100MB for 1MB HTML)

## Dependencies

- None (first step in pipeline)

## Blocked By

- None

## Blocks

- Issue #103: CSS Parser Integration
- Issue #104: GPU Selector Matching
