# Issue #104: GPU Selector Matching

## Summary
Implement a Metal compute kernel for CSS selector matching that runs entirely on the GPU, achieving 50-100x speedup over CPU-based matching.

## Motivation
Selector matching is O(rules × elements), currently the biggest CPU bottleneck in browser engines. For a page with 5000 elements and 5000 rules, that's 25 million comparisons. GPU parallelism can reduce this to milliseconds.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Selector Matching                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ GPUNode  │  │GPUSelector│  │ GPURule  │  │Attributes│    │
│  │ Buffer   │  │  Buffer   │  │  Buffer  │  │  Buffer  │    │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │             │             │           │
│       └──────────────┼─────────────┼─────────────┘           │
│                      │             │                         │
│                      ▼             ▼                         │
│           ┌──────────────────────────────────┐              │
│           │   match_selectors Kernel          │              │
│           │   (1 thread per node)             │              │
│           │                                   │              │
│           │   For each rule:                  │              │
│           │     Check selector chain          │              │
│           │     If match: record rule index   │              │
│           └──────────────┬───────────────────┘              │
│                          │                                   │
│                          ▼                                   │
│  Output:                                                     │
│  ┌─────────────────────────────────────────┐                │
│  │ matched_rules[node][rule_idx]           │                │
│  │ match_counts[node]                      │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Structures

### GPU Buffers

```metal
// Input buffers (from Issues #102, #103)
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
    ushort id_hash;       // Pre-computed hash of id attribute
    ushort class_bitmap;  // Bit flags for common classes
};

struct GPUSelector {
    uint type_hash;
    uint id_hash;
    uint class_hash;
    uint attribute_hash;
    uchar attribute_op;
    uchar combinator;
    uchar pseudo_class;
    uchar pseudo_element;
    uint next_selector;
    uint specificity;
};

struct GPURule {
    uint selector_offset;
    ushort selector_count;
    uint property_offset;
    ushort property_count;
    uint specificity;
    uint source_order;
    uchar origin;
    uchar3 _padding;
};

// Output buffer
struct MatchResult {
    uint rule_index;
    uint specificity;
};

// Pre-computed class hashes for each node (variable length)
struct NodeClasses {
    uint class_hashes[MAX_CLASSES_PER_NODE];  // Usually 8
    uint count;
};
```

## Metal Kernel Implementation

```metal
// src/gpu_os/document/selector_match.metal

#include <metal_stdlib>
using namespace metal;

constant uint INVALID_IDX = 0xFFFFFFFF;
constant uint MAX_MATCHES_PER_NODE = 256;
constant uint MAX_CLASSES_PER_NODE = 8;

// Combinator types
constant uchar COMB_NONE = 0;
constant uchar COMB_DESCENDANT = 1;
constant uchar COMB_CHILD = 2;
constant uchar COMB_ADJACENT_SIBLING = 3;
constant uchar COMB_GENERAL_SIBLING = 4;

// Check if a single selector matches a node
bool selector_matches_node(
    GPUSelector selector,
    GPUNode node,
    device NodeClasses* node_classes,
    device GPUAttribute* attributes,
    device uchar* string_pool
) {
    // Type selector check
    if (selector.type_hash != 0) {
        if (selector.type_hash != node.element_type) {
            return false;
        }
    }

    // ID selector check
    if (selector.id_hash != 0) {
        if (selector.id_hash != node.id_hash) {
            return false;
        }
    }

    // Class selector check
    if (selector.class_hash != 0) {
        NodeClasses classes = node_classes[node_idx];
        bool found = false;
        for (uint i = 0; i < classes.count && !found; i++) {
            if (classes.class_hashes[i] == selector.class_hash) {
                found = true;
            }
        }
        if (!found) return false;
    }

    // Attribute selector check
    if (selector.attribute_hash != 0) {
        bool found = false;
        for (uint i = 0; i < node.attribute_count && !found; i++) {
            GPUAttribute attr = attributes[node.attribute_offset + i];
            if (attr.name_hash == selector.attribute_hash) {
                // Check attribute operation
                switch (selector.attribute_op) {
                    case 0: // [attr] - exists
                        found = true;
                        break;
                    case 1: // [attr=value] - equals
                        // Compare values (simplified)
                        found = true; // Would need value comparison
                        break;
                    case 2: // [attr~=value] - contains word
                        found = true; // Would need word matching
                        break;
                    // ... other operations
                }
            }
        }
        if (!found) return false;
    }

    // Pseudo-class check (simplified)
    if (selector.pseudo_class != 0) {
        // :first-child
        if (selector.pseudo_class == 1) {
            if (node.prev_sibling_idx != INVALID_IDX) {
                return false;
            }
        }
        // :last-child
        else if (selector.pseudo_class == 2) {
            if (node.next_sibling_idx != INVALID_IDX) {
                return false;
            }
        }
        // :nth-child would need sibling counting
        // :hover, :active, :focus need state from CPU
    }

    return true;
}

// Check if selector chain matches (handles combinators)
bool selector_chain_matches(
    device GPUNode* nodes,
    device GPUSelector* selectors,
    device NodeClasses* node_classes,
    device GPUAttribute* attributes,
    device uchar* string_pool,
    uint node_idx,
    uint selector_idx,
    uint max_depth
) {
    GPUNode node = nodes[node_idx];
    GPUSelector selector = selectors[selector_idx];

    // First, check if the rightmost selector matches the node
    if (!selector_matches_node(selector, node, node_classes, attributes, string_pool)) {
        return false;
    }

    // If no combinator, we're done
    if (selector.combinator == COMB_NONE || selector.next_selector == INVALID_IDX) {
        return true;
    }

    GPUSelector next_sel = selectors[selector.next_selector];

    switch (selector.combinator) {
        case COMB_DESCENDANT: {
            // Check all ancestors
            uint ancestor = node.parent_idx;
            uint depth = 0;
            while (ancestor != INVALID_IDX && depth < max_depth) {
                if (selector_chain_matches(nodes, selectors, node_classes, attributes,
                                           string_pool, ancestor, selector.next_selector, max_depth - 1)) {
                    return true;
                }
                ancestor = nodes[ancestor].parent_idx;
                depth++;
            }
            return false;
        }

        case COMB_CHILD: {
            // Check only direct parent
            if (node.parent_idx == INVALID_IDX) return false;
            return selector_chain_matches(nodes, selectors, node_classes, attributes,
                                          string_pool, node.parent_idx, selector.next_selector, max_depth - 1);
        }

        case COMB_ADJACENT_SIBLING: {
            // Check previous sibling only
            if (node.prev_sibling_idx == INVALID_IDX) return false;
            return selector_chain_matches(nodes, selectors, node_classes, attributes,
                                          string_pool, node.prev_sibling_idx, selector.next_selector, max_depth - 1);
        }

        case COMB_GENERAL_SIBLING: {
            // Check all previous siblings
            uint sibling = node.prev_sibling_idx;
            while (sibling != INVALID_IDX) {
                if (selector_chain_matches(nodes, selectors, node_classes, attributes,
                                           string_pool, sibling, selector.next_selector, max_depth - 1)) {
                    return true;
                }
                sibling = nodes[sibling].prev_sibling_idx;
            }
            return false;
        }
    }

    return false;
}

// Main kernel: match all rules against all nodes
kernel void match_selectors(
    device GPUNode* nodes [[buffer(0)]],
    device GPUSelector* selectors [[buffer(1)]],
    device GPURule* rules [[buffer(2)]],
    device NodeClasses* node_classes [[buffer(3)]],
    device GPUAttribute* attributes [[buffer(4)]],
    device uchar* string_pool [[buffer(5)]],
    device MatchResult* matched_rules [[buffer(6)]],
    device atomic_uint* match_counts [[buffer(7)]],
    constant uint& node_count [[buffer(8)]],
    constant uint& rule_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    GPUNode node = nodes[gid];

    // Skip non-element nodes (text, comments)
    if (node.element_type == 0) return;

    uint match_offset = gid * MAX_MATCHES_PER_NODE;
    uint count = 0;

    // Check each rule
    for (uint r = 0; r < rule_count && count < MAX_MATCHES_PER_NODE; r++) {
        GPURule rule = rules[r];
        bool matched = false;

        // Check each selector in the selector list (comma-separated)
        for (uint s = 0; s < rule.selector_count && !matched; s++) {
            uint sel_idx = rule.selector_offset + s;

            if (selector_chain_matches(nodes, selectors, node_classes, attributes,
                                        string_pool, gid, sel_idx, 32)) {
                matched = true;
            }
        }

        if (matched) {
            matched_rules[match_offset + count] = MatchResult {
                .rule_index = r,
                .specificity = rule.specificity
            };
            count++;
        }
    }

    atomic_store_explicit(&match_counts[gid], count, memory_order_relaxed);
}

// Phase 2: Sort matches by specificity (per-node)
kernel void sort_matches(
    device MatchResult* matched_rules [[buffer(0)]],
    device uint* match_counts [[buffer(1)]],
    constant uint& node_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= node_count) return;

    uint count = match_counts[gid];
    if (count <= 1) return;

    uint offset = gid * MAX_MATCHES_PER_NODE;

    // Insertion sort (stable, good for small N)
    for (uint i = 1; i < count; i++) {
        MatchResult key = matched_rules[offset + i];
        int j = i - 1;

        while (j >= 0 && matched_rules[offset + j].specificity < key.specificity) {
            matched_rules[offset + j + 1] = matched_rules[offset + j];
            j--;
        }
        matched_rules[offset + j + 1] = key;
    }
}
```

## Rust Integration

```rust
// src/gpu_os/document/selector_match.rs

use metal::*;

pub struct GPUSelectorMatcher {
    device: Device,
    pipeline: ComputePipelineState,
    sort_pipeline: ComputePipelineState,
    matched_rules_buffer: Buffer,
    match_counts_buffer: Buffer,
}

impl GPUSelectorMatcher {
    pub fn new(device: &Device, library: &Library, max_nodes: usize) -> Self {
        let match_fn = library.get_function("match_selectors", None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&match_fn).unwrap();

        let sort_fn = library.get_function("sort_matches", None).unwrap();
        let sort_pipeline = device.new_compute_pipeline_state_with_function(&sort_fn).unwrap();

        let matched_rules_buffer = device.new_buffer(
            (max_nodes * MAX_MATCHES_PER_NODE * std::mem::size_of::<MatchResult>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let match_counts_buffer = device.new_buffer(
            (max_nodes * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            device: device.clone(),
            pipeline,
            sort_pipeline,
            matched_rules_buffer,
            match_counts_buffer,
        }
    }

    pub fn match_all(
        &self,
        command_buffer: &CommandBufferRef,
        gpu_dom: &GPUDom,
        gpu_css: &GPUCSS,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        // Phase 1: Match selectors
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&gpu_dom.node_buffer), 0);
        encoder.set_buffer(1, Some(&gpu_css.selector_buffer), 0);
        encoder.set_buffer(2, Some(&gpu_css.rule_buffer), 0);
        encoder.set_buffer(3, Some(&gpu_dom.node_classes_buffer), 0);
        encoder.set_buffer(4, Some(&gpu_dom.attribute_buffer), 0);
        encoder.set_buffer(5, Some(&gpu_dom.string_pool_buffer), 0);
        encoder.set_buffer(6, Some(&self.matched_rules_buffer), 0);
        encoder.set_buffer(7, Some(&self.match_counts_buffer), 0);
        encoder.set_bytes(8, std::mem::size_of::<u32>() as u64,
                          &gpu_dom.node_count as *const u32 as *const _);
        encoder.set_bytes(9, std::mem::size_of::<u32>() as u64,
                          &gpu_css.rule_count as *const u32 as *const _);

        let threadgroup_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(
            ((gpu_dom.node_count as u64 + 255) / 256) * 256,
            1, 1
        );
        encoder.dispatch_threads(grid_size, threadgroup_size);

        // Phase 2: Sort matches
        encoder.set_compute_pipeline_state(&self.sort_pipeline);
        encoder.set_buffer(0, Some(&self.matched_rules_buffer), 0);
        encoder.set_buffer(1, Some(&self.match_counts_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64,
                          &gpu_dom.node_count as *const u32 as *const _);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
    }

    pub fn get_matches(&self, node_idx: u32) -> Vec<MatchResult> {
        let counts = unsafe {
            std::slice::from_raw_parts(
                self.match_counts_buffer.contents() as *const u32,
                self.match_counts_buffer.length() as usize / 4,
            )
        };

        let count = counts[node_idx as usize] as usize;
        if count == 0 {
            return Vec::new();
        }

        let results = unsafe {
            std::slice::from_raw_parts(
                self.matched_rules_buffer.contents() as *const MatchResult,
                self.matched_rules_buffer.length() as usize / std::mem::size_of::<MatchResult>(),
            )
        };

        let offset = node_idx as usize * MAX_MATCHES_PER_NODE;
        results[offset..offset + count].to_vec()
    }
}
```

## Benchmarks

```rust
// benches/selector_matching_bench.rs

fn bench_selector_matching(c: &mut Criterion) {
    let device = Device::system_default().unwrap();

    let test_cases = vec![
        ("small", "test_pages/small.html", "test_css/simple.css"),
        ("medium", "test_pages/medium.html", "test_css/bootstrap.min.css"),
        ("wikipedia", "test_pages/wikipedia.html", "test_css/wikipedia.css"),
    ];

    let mut group = c.benchmark_group("selector_matching");

    for (name, html_path, css_path) in test_cases {
        let html = std::fs::read_to_string(html_path).unwrap();
        let css = std::fs::read_to_string(css_path).unwrap();

        let doc = GPUTreeSink::parse(&html);
        let gpu_dom = GPUDom::from_document(&device, &doc);

        let mut parser = CSSParser::new();
        parser.parse_stylesheet(&css).unwrap();
        let gpu_css = GPUCSS::from_parser(&device, &parser);

        // CPU baseline (sequential matching)
        group.bench_with_input(
            BenchmarkId::new("cpu_sequential", name),
            &(&doc, &parser),
            |b, (doc, parser)| {
                b.iter(|| cpu_match_selectors(doc, parser))
            }
        );

        // GPU matching
        let matcher = GPUSelectorMatcher::new(&device, &library, doc.nodes.len());

        group.bench_with_input(
            BenchmarkId::new("gpu_parallel", name),
            &(),
            |b, _| {
                b.iter(|| {
                    let command_buffer = command_queue.new_command_buffer();
                    matcher.match_all(&command_buffer, &gpu_dom, &gpu_css);
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                })
            }
        );
    }

    group.finish();
}
```

### Expected Results

| Page | Nodes | Rules | CPU Time | GPU Time | Speedup |
|------|-------|-------|----------|----------|---------|
| Small | 100 | 50 | 0.5ms | 0.05ms | 10x |
| Medium | 1000 | 1000 | 50ms | 0.5ms | 100x |
| Wikipedia | 5000 | 5000 | 500ms | 2ms | 250x |

## Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_selector() {
        let html = "<div></div><span></span><div></div>";
        let css = "div { color: red; }";

        let matches = run_matching(&html, &css);

        // Should match both divs
        assert_eq!(matches.get_count(0), 1); // First div
        assert_eq!(matches.get_count(1), 0); // span
        assert_eq!(matches.get_count(2), 1); // Second div
    }

    #[test]
    fn test_class_selector() {
        let html = r#"<div class="foo"></div><div class="bar"></div><div class="foo bar"></div>"#;
        let css = ".foo { color: red; }";

        let matches = run_matching(&html, &css);

        assert_eq!(matches.get_count(0), 1); // has .foo
        assert_eq!(matches.get_count(1), 0); // only .bar
        assert_eq!(matches.get_count(2), 1); // has .foo
    }

    #[test]
    fn test_id_selector() {
        let html = r#"<div id="main"></div><div id="other"></div>"#;
        let css = "#main { color: red; }";

        let matches = run_matching(&html, &css);

        assert_eq!(matches.get_count(0), 1);
        assert_eq!(matches.get_count(1), 0);
    }

    #[test]
    fn test_descendant_combinator() {
        let html = r#"<div><p></p></div><p></p>"#;
        let css = "div p { color: red; }";

        let matches = run_matching(&html, &css);

        // Only p inside div should match
        let p_inside = find_element(&html, "div > p");
        let p_outside = find_element(&html, "body > p");

        assert_eq!(matches.get_count(p_inside), 1);
        assert_eq!(matches.get_count(p_outside), 0);
    }

    #[test]
    fn test_child_combinator() {
        let html = r#"<div><p></p><span><p></p></span></div>"#;
        let css = "div > p { color: red; }";

        let matches = run_matching(&html, &css);

        // Only direct child p should match
        let direct_p = find_element(&html, "div > p:first-child");
        let nested_p = find_element(&html, "span > p");

        assert_eq!(matches.get_count(direct_p), 1);
        assert_eq!(matches.get_count(nested_p), 0);
    }

    #[test]
    fn test_specificity_sorting() {
        let html = r#"<div id="main" class="container"></div>"#;
        let css = r#"
            div { color: red; }
            .container { color: blue; }
            #main { color: green; }
        "#;

        let matches = run_matching(&html, &css);
        let div_matches = matches.get_matches(0);

        // Should be sorted by specificity: #main > .container > div
        assert_eq!(div_matches.len(), 3);
        assert!(div_matches[0].specificity > div_matches[1].specificity);
        assert!(div_matches[1].specificity > div_matches[2].specificity);
    }

    #[test]
    fn test_multiple_selectors() {
        let html = r#"<h1></h1><h2></h2><p></p>"#;
        let css = "h1, h2 { font-weight: bold; }";

        let matches = run_matching(&html, &css);

        assert_eq!(matches.get_count(0), 1); // h1
        assert_eq!(matches.get_count(1), 1); // h2
        assert_eq!(matches.get_count(2), 0); // p
    }

    #[test]
    fn test_gpu_cpu_parity() {
        let html = std::fs::read_to_string("test_pages/medium.html").unwrap();
        let css = std::fs::read_to_string("test_css/bootstrap.min.css").unwrap();

        let cpu_matches = cpu_match_selectors(&html, &css);
        let gpu_matches = gpu_match_selectors(&html, &css);

        // Verify identical results
        for i in 0..cpu_matches.node_count() {
            let cpu = cpu_matches.get_matches(i);
            let gpu = gpu_matches.get_matches(i);

            assert_eq!(cpu.len(), gpu.len(), "Mismatch at node {}", i);

            for (c, g) in cpu.iter().zip(gpu.iter()) {
                assert_eq!(c.rule_index, g.rule_index);
                assert_eq!(c.specificity, g.specificity);
            }
        }
    }
}
```

## Performance Optimization Notes

### Bloom Filter Pre-check
```metal
// Optional: Use bloom filter to skip definitely-non-matching rules
// Precompute bloom filter from selectors
uint bloom_hash(GPUSelector sel) {
    return sel.type_hash ^ sel.id_hash ^ sel.class_hash;
}

// In main kernel, skip rules that can't possibly match
if ((node_bloom & rule_bloom) != rule_bloom) continue;
```

### Threadgroup Optimization
```metal
// Use threadgroup memory for rule caching
threadgroup GPURule cached_rules[256];

// Load rules cooperatively
if (local_id < 256 && local_id < rule_count) {
    cached_rules[local_id] = rules[local_id];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

## Acceptance Criteria

- [ ] All selector types work correctly
- [ ] All combinators work correctly
- [ ] Specificity sorting is correct
- [ ] GPU and CPU produce identical results
- [ ] GPU is at least 50x faster for large pages
- [ ] Memory usage is bounded (MAX_MATCHES_PER_NODE)
- [ ] No infinite loops on circular DOM (max_depth protection)

## Dependencies

- Issue #102: HTML5 Parser Integration
- Issue #103: CSS Parser Integration

## Blocks

- Issue #105: GPU Cascade Resolution
