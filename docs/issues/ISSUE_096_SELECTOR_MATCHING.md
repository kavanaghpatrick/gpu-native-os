# Issue #96: GPU Selector Matching

## Summary
Match CSS selectors to DOM elements on GPU. For each element, find all matching rules and build a matched-rules list for cascade processing.

## Problem
After parsing CSS rules, we need to determine which rules apply to which elements. This requires testing each element against potentially thousands of selectors efficiently.

## Solution

### Algorithm Overview

```
For each element E (parallel):
    For each CSS rule R:
        If selector(R) matches E:
            Add R to E.matched_rules

Sort E.matched_rules by (specificity, source_order)
```

### Optimization: Hash-based filtering

Most selectors won't match most elements. Use hash filtering to quickly reject non-matching selectors:

```
Element has:
    - tag_hash: hash of element name (e.g., "div")
    - class_hashes[]: hashes of all class names
    - id_hash: hash of id attribute

Selector's rightmost part has:
    - name_hash

Quick reject: if selector.name_hash not in element's hashes, skip
```

### Data Structures

```rust
#[repr(C)]
pub struct ElementHashes {
    pub tag_hash: u32,
    pub id_hash: u32,
    pub class_hash_count: u32,
    pub class_hash_start: u32,  // Index into class_hashes buffer
}

#[repr(C)]
pub struct MatchedRule {
    pub rule_index: u32,
    pub specificity: u16,
    pub source_order: u16,
}

#[repr(C)]
pub struct ElementMatches {
    pub match_start: u32,     // Index into matched_rules buffer
    pub match_count: u32,
}
```

### GPU Kernels

```metal
// Pass 1: Compute element hashes (parallel per-element)
kernel void compute_element_hashes(
    device const Element* elements [[buffer(0)]],
    device const uint8_t* html [[buffer(1)]],
    device const Token* tokens [[buffer(2)]],
    device ElementHashes* hashes [[buffer(3)]],
    device uint* class_hashes [[buffer(4)]],
    device atomic_uint* class_hash_count [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    Element elem = elements[gid];
    ElementHashes h;

    // Hash tag name
    h.tag_hash = fnv1a_hash(html, elem.tag_start, elem.tag_length);

    // Hash id attribute (if present)
    if (elem.id_start > 0) {
        h.id_hash = fnv1a_hash(html, elem.id_start, elem.id_length);
    } else {
        h.id_hash = 0;
    }

    // Hash class names (space-separated)
    h.class_hash_start = atomic_fetch_add_explicit(class_hash_count, 0, memory_order_relaxed);
    h.class_hash_count = 0;

    if (elem.class_start > 0) {
        uint pos = elem.class_start;
        uint end = elem.class_start + elem.class_length;

        while (pos < end) {
            // Skip whitespace
            while (pos < end && (html[pos] == ' ' || html[pos] == '\t')) pos++;
            if (pos >= end) break;

            // Find class name end
            uint name_start = pos;
            while (pos < end && html[pos] != ' ' && html[pos] != '\t') pos++;
            uint name_len = pos - name_start;

            if (name_len > 0) {
                uint slot = atomic_fetch_add_explicit(class_hash_count, 1, memory_order_relaxed);
                class_hashes[slot] = fnv1a_hash(html, name_start, name_len);
                h.class_hash_count++;
            }
        }
    }

    hashes[gid] = h;
}

// Pass 2: Count matches per element (parallel per element-rule pair)
kernel void count_selector_matches(
    device const Element* elements [[buffer(0)]],
    device const ElementHashes* hashes [[buffer(1)]],
    device const uint* class_hashes [[buffer(2)]],
    device const CSSRule* rules [[buffer(3)]],
    device const SelectorPart* selectors [[buffer(4)]],
    device atomic_uint* match_counts [[buffer(5)]],  // Per element
    constant uint& element_count [[buffer(6)]],
    constant uint& rule_count [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]  // (element, rule)
) {
    uint elem_idx = gid.x;
    uint rule_idx = gid.y;

    if (elem_idx >= element_count) return;
    if (rule_idx >= rule_count) return;

    CSSRule rule = rules[rule_idx];
    if (rule.selector_count == 0) return;

    // Get rightmost selector part (subject of selector)
    SelectorPart subject = selectors[rule.selector_start + rule.selector_count - 1];

    // Quick hash check
    ElementHashes h = hashes[elem_idx];

    bool possible_match = false;
    if (subject.part_type == SEL_ELEMENT) {
        possible_match = (subject.name_hash == h.tag_hash);
    } else if (subject.part_type == SEL_ID) {
        possible_match = (subject.name_hash == h.id_hash);
    } else if (subject.part_type == SEL_CLASS) {
        for (uint i = 0; i < h.class_hash_count; i++) {
            if (class_hashes[h.class_hash_start + i] == subject.name_hash) {
                possible_match = true;
                break;
            }
        }
    } else if (subject.part_type == SEL_UNIVERSAL) {
        possible_match = true;
    }

    if (!possible_match) return;

    // Full selector match (walk up DOM tree for combinators)
    if (selector_matches(elements, hashes, class_hashes, selectors,
                         rule.selector_start, rule.selector_count, elem_idx)) {
        atomic_fetch_add_explicit(&match_counts[elem_idx], 1, memory_order_relaxed);
    }
}

// Helper: Full selector matching with combinators
bool selector_matches(
    device const Element* elements,
    device const ElementHashes* hashes,
    device const uint* class_hashes,
    device const SelectorPart* selectors,
    uint sel_start,
    uint sel_count,
    uint elem_idx
) {
    // Walk selector right-to-left, element tree bottom-to-top
    int sel_idx = sel_start + sel_count - 1;
    int curr_elem = elem_idx;

    while (sel_idx >= int(sel_start)) {
        SelectorPart part = selectors[sel_idx];

        if (!part_matches_element(hashes, class_hashes, part, curr_elem)) {
            // Combinator handling
            if (sel_idx < int(sel_start + sel_count - 1)) {
                SelectorPart prev = selectors[sel_idx + 1];
                if (prev.combinator == COMB_DESCENDANT) {
                    // Try ancestor
                    curr_elem = elements[curr_elem].parent;
                    if (curr_elem < 0) return false;
                    continue;  // Retry same selector part with ancestor
                }
            }
            return false;
        }

        // Move to next selector part
        sel_idx--;

        // Handle combinator
        if (sel_idx >= int(sel_start)) {
            SelectorPart prev = selectors[sel_idx + 1];
            switch (part.combinator) {
                case COMB_DESCENDANT:
                    curr_elem = elements[curr_elem].parent;
                    break;
                case COMB_CHILD:
                    curr_elem = elements[curr_elem].parent;
                    break;
                case COMB_ADJACENT:
                    curr_elem = find_previous_sibling(elements, curr_elem);
                    break;
                case COMB_SIBLING:
                    curr_elem = find_previous_sibling(elements, curr_elem);
                    break;
            }
            if (curr_elem < 0) return false;
        }
    }

    return true;
}

// Pass 3: Prefix sum for match buffer allocation
// (Use GPU prefix sum)

// Pass 4: Write matched rules (parallel per element-rule pair)
kernel void write_matched_rules(
    device const Element* elements [[buffer(0)]],
    device const ElementHashes* hashes [[buffer(1)]],
    device const uint* class_hashes [[buffer(2)]],
    device const CSSRule* rules [[buffer(3)]],
    device const SelectorPart* selectors [[buffer(4)]],
    device const uint* match_offsets [[buffer(5)]],  // From prefix sum
    device MatchedRule* matched_rules [[buffer(6)]],
    device atomic_uint* write_positions [[buffer(7)]],  // Per element
    constant uint& element_count [[buffer(8)]],
    constant uint& rule_count [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem_idx = gid.x;
    uint rule_idx = gid.y;

    if (elem_idx >= element_count) return;
    if (rule_idx >= rule_count) return;

    CSSRule rule = rules[rule_idx];
    if (rule.selector_count == 0) return;

    // Same matching logic as count pass
    if (!selector_matches(elements, hashes, class_hashes, selectors,
                         rule.selector_start, rule.selector_count, elem_idx)) {
        return;
    }

    // Write match
    uint base = match_offsets[elem_idx];
    uint slot = atomic_fetch_add_explicit(&write_positions[elem_idx], 1, memory_order_relaxed);

    MatchedRule match;
    match.rule_index = rule_idx;
    match.specificity = rule.specificity;
    match.source_order = rule.source_order;

    matched_rules[base + slot] = match;
}

// Pass 5: Sort matches per element by specificity (parallel per-element)
kernel void sort_matched_rules(
    device MatchedRule* matched_rules [[buffer(0)]],
    device const uint* match_offsets [[buffer(1)]],
    device const uint* match_counts [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    uint start = match_offsets[gid];
    uint count = match_counts[gid];
    if (count <= 1) return;

    // Simple bubble sort (count is usually small)
    for (uint i = 0; i < count - 1; i++) {
        for (uint j = 0; j < count - i - 1; j++) {
            MatchedRule a = matched_rules[start + j];
            MatchedRule b = matched_rules[start + j + 1];

            // Sort by specificity (desc), then source_order (asc)
            bool should_swap = false;
            if (b.specificity > a.specificity) {
                should_swap = true;
            } else if (b.specificity == a.specificity && b.source_order > a.source_order) {
                should_swap = true;
            }

            if (should_swap) {
                matched_rules[start + j] = b;
                matched_rules[start + j + 1] = a;
            }
        }
    }
}
```

## Pseudocode

```
FUNCTION match_selectors(elements, css_rules):
    // Pass 1: Compute element hashes
    element_hashes = gpu_dispatch(compute_element_hashes, elements)

    // Pass 2: Count matches per element
    match_counts = zeros(element_count)
    gpu_dispatch_2d(count_selector_matches, element_count, rule_count,
                    elements, element_hashes, css_rules, match_counts)

    // Pass 3: Prefix sum for allocation
    match_offsets = gpu_prefix_sum(match_counts)
    total_matches = match_offsets[element_count - 1] + match_counts[element_count - 1]

    // Pass 4: Write matches
    matched_rules = allocate(total_matches)
    write_positions = zeros(element_count)
    gpu_dispatch_2d(write_matched_rules, element_count, rule_count,
                    elements, element_hashes, css_rules, match_offsets, matched_rules)

    // Pass 5: Sort by specificity
    gpu_dispatch(sort_matched_rules, matched_rules, match_offsets, match_counts)

    RETURN SelectorMatchResult {
        matched_rules,
        match_offsets,
        match_counts
    }
```

## Tests

### Test 1: Element hash computation
```rust
#[test]
fn test_element_hashes() {
    let html = r#"<div id="main" class="container active"></div>"#;
    let elements = parse_html(html);
    let hashes = compute_hashes(&elements);

    assert_eq!(hashes[0].tag_hash, fnv1a(b"div"));
    assert_eq!(hashes[0].id_hash, fnv1a(b"main"));
    assert_eq!(hashes[0].class_hash_count, 2);
}
```

### Test 2: Simple selector matching
```rust
#[test]
fn test_simple_selector_matching() {
    let html = r#"<div class="foo"></div><span class="foo"></span>"#;
    let css = ".foo { color: red; }";

    let elements = parse_html(html);
    let rules = parse_css(css);
    let matches = match_selectors(&elements, &rules);

    // Both elements should match
    assert_eq!(matches.counts[0], 1);
    assert_eq!(matches.counts[1], 1);
}
```

### Test 3: ID selector
```rust
#[test]
fn test_id_selector() {
    let html = r#"<div id="main"></div><div id="other"></div>"#;
    let css = "#main { color: red; }";

    let elements = parse_html(html);
    let rules = parse_css(css);
    let matches = match_selectors(&elements, &rules);

    assert_eq!(matches.counts[0], 1);  // #main matches
    assert_eq!(matches.counts[1], 0);  // #other doesn't
}
```

### Test 4: Descendant combinator
```rust
#[test]
fn test_descendant_selector() {
    let html = r#"
        <div class="outer">
            <span>
                <a class="link"></a>
            </span>
        </div>
        <a class="link"></a>
    "#;
    let css = ".outer .link { color: red; }";

    let elements = parse_html(html);
    let rules = parse_css(css);
    let matches = match_selectors(&elements, &rules);

    // Only the nested .link should match
    let link1_idx = find_element_by_class(&elements, "link", 0);
    let link2_idx = find_element_by_class(&elements, "link", link1_idx + 1);

    assert_eq!(matches.counts[link1_idx], 1);  // Inside .outer
    assert_eq!(matches.counts[link2_idx], 0);  // Outside .outer
}
```

### Test 5: Child combinator
```rust
#[test]
fn test_child_selector() {
    let html = r#"
        <div class="parent">
            <span class="child"></span>
            <div><span class="grandchild"></span></div>
        </div>
    "#;
    let css = ".parent > span { color: red; }";

    let elements = parse_html(html);
    let rules = parse_css(css);
    let matches = match_selectors(&elements, &rules);

    let child_idx = find_element_by_class(&elements, "child", 0);
    let grandchild_idx = find_element_by_class(&elements, "grandchild", 0);

    assert_eq!(matches.counts[child_idx], 1);      // Direct child
    assert_eq!(matches.counts[grandchild_idx], 0); // Not direct child
}
```

### Test 6: Specificity ordering
```rust
#[test]
fn test_specificity_ordering() {
    let html = r#"<div id="foo" class="bar"></div>"#;
    let css = r#"
        div { color: red; }
        .bar { color: green; }
        #foo { color: blue; }
    "#;

    let elements = parse_html(html);
    let rules = parse_css(css);
    let matches = match_selectors(&elements, &rules);

    // All three rules match
    assert_eq!(matches.counts[0], 3);

    // Should be sorted by specificity: #foo, .bar, div
    let matched = get_matched_rules(&matches, 0);
    assert_eq!(matched[0].specificity, 100); // #foo
    assert_eq!(matched[1].specificity, 10);  // .bar
    assert_eq!(matched[2].specificity, 1);   // div
}
```

### Test 7: Wikipedia selectors
```rust
#[test]
fn test_wikipedia_selectors() {
    let html = include_str!("../testdata/wikipedia.html");
    let css = include_str!("../testdata/wikipedia.css");

    let elements = parse_html(html);
    let rules = parse_css(css);
    let matches = match_selectors(&elements, &rules);

    // Elements with class "mw-hidden" should match that rule
    let hidden_elements = find_elements_by_class(&elements, "mw-hidden");
    let mw_hidden_rule = find_rule_by_selector(&rules, ".mw-hidden");

    for elem_idx in hidden_elements {
        let elem_matches = get_matched_rules(&matches, elem_idx);
        assert!(elem_matches.iter().any(|m| m.rule_index == mw_hidden_rule));
    }
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/selector_match.rs` | Create | Matching implementation |
| `src/gpu_os/document/selector_match.metal` | Create | GPU kernels |
| `tests/test_issue_96_selector_matching.rs` | Create | Tests |

## Acceptance Criteria

1. [ ] Element hash computation (tag, id, classes)
2. [ ] Hash-based quick rejection
3. [ ] Simple selectors (element, class, id)
4. [ ] Descendant combinator (space)
5. [ ] Child combinator (>)
6. [ ] Adjacent sibling combinator (+)
7. [ ] General sibling combinator (~)
8. [ ] Specificity sorting
9. [ ] Source order tie-breaking
10. [ ] Wikipedia selectors match correctly
11. [ ] All tests pass
