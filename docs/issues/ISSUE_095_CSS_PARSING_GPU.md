# Issue #95: GPU CSS Parsing

## Summary
Parse CSS rules on GPU using parallel tokenization and rule extraction. Output structured rule data for selector matching.

## Problem
CSS files can be large (Wikipedia's is 100KB+). Parsing must be fast and produce GPU-friendly data structures for selector matching.

## Solution

### Data Structures

```rust
// CSS Token types
pub const TOK_IDENT: u8 = 1;       // foo, bar-baz
pub const TOK_HASH: u8 = 2;        // #id
pub const TOK_DOT: u8 = 3;         // .
pub const TOK_COLON: u8 = 4;       // :
pub const TOK_LBRACE: u8 = 5;      // {
pub const TOK_RBRACE: u8 = 6;      // }
pub const TOK_SEMICOLON: u8 = 7;   // ;
pub const TOK_COMMA: u8 = 8;       // ,
pub const TOK_GT: u8 = 9;          // >
pub const TOK_PLUS: u8 = 10;       // +
pub const TOK_TILDE: u8 = 11;      // ~
pub const TOK_STAR: u8 = 12;       // *
pub const TOK_LBRACKET: u8 = 13;   // [
pub const TOK_RBRACKET: u8 = 14;   // ]
pub const TOK_STRING: u8 = 15;     // "..." or '...'
pub const TOK_NUMBER: u8 = 16;     // 123, 1.5
pub const TOK_DIMENSION: u8 = 17;  // 10px, 1em
pub const TOK_PERCENTAGE: u8 = 18; // 50%
pub const TOK_AT: u8 = 19;         // @
pub const TOK_WHITESPACE: u8 = 20; // space/tab/newline
pub const TOK_COMMENT: u8 = 21;    // /* ... */

#[repr(C)]
pub struct CSSToken {
    pub token_type: u8,
    pub _pad1: u8,
    pub start: u16,      // Offset into CSS buffer (max 64KB per chunk)
    pub length: u16,
    pub _pad2: u16,
}

#[repr(C)]
pub struct CSSRule {
    pub selector_start: u32,    // Index into selector parts buffer
    pub selector_count: u16,    // Number of selector parts
    pub specificity: u16,       // Packed: (id*100 + class*10 + element)
    pub property_start: u32,    // Index into property buffer
    pub property_count: u16,    // Number of properties
    pub source_order: u16,      // For cascade ordering
}

#[repr(C)]
pub struct SelectorPart {
    pub part_type: u8,          // ELEMENT=1, CLASS=2, ID=3, PSEUDO=4, ATTR=5
    pub combinator: u8,         // NONE=0, DESC=1, CHILD=2, ADJ=3, SIB=4
    pub name_hash: u32,         // FNV-1a hash for fast comparison
    pub name_start: u16,        // Offset for full name
    pub name_length: u16,
    pub _padding: u32,
}

#[repr(C)]
pub struct CSSProperty {
    pub property_id: u16,       // Enum: DISPLAY, COLOR, MARGIN_TOP, etc.
    pub value_type: u8,         // KEYWORD, LENGTH, COLOR, etc.
    pub _pad: u8,
    pub value_start: u16,       // Offset into value buffer
    pub value_length: u16,
    pub numeric_value: f32,     // Pre-parsed for lengths/numbers
}

// Property IDs
pub const PROP_DISPLAY: u16 = 1;
pub const PROP_POSITION: u16 = 2;
pub const PROP_WIDTH: u16 = 3;
pub const PROP_HEIGHT: u16 = 4;
pub const PROP_MARGIN_TOP: u16 = 5;
pub const PROP_MARGIN_RIGHT: u16 = 6;
pub const PROP_MARGIN_BOTTOM: u16 = 7;
pub const PROP_MARGIN_LEFT: u16 = 8;
pub const PROP_PADDING_TOP: u16 = 9;
// ... etc
```

### GPU Kernels

```metal
// Pass 1: Classify each character (parallel per-character)
kernel void css_classify_chars(
    device const uint8_t* css [[buffer(0)]],
    device uint8_t* char_classes [[buffer(1)]],
    constant uint& css_length [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= css_length) return;

    uint8_t c = css[gid];
    uint8_t cls;

    if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '_' || c == '-') {
        cls = CHAR_IDENT;
    } else if (c >= '0' && c <= '9') {
        cls = CHAR_DIGIT;
    } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        cls = CHAR_WHITESPACE;
    } else if (c == '{') cls = CHAR_LBRACE;
    else if (c == '}') cls = CHAR_RBRACE;
    else if (c == ':') cls = CHAR_COLON;
    else if (c == ';') cls = CHAR_SEMICOLON;
    else if (c == '.') cls = CHAR_DOT;
    else if (c == '#') cls = CHAR_HASH;
    else if (c == ',') cls = CHAR_COMMA;
    else if (c == '>') cls = CHAR_GT;
    else if (c == '+') cls = CHAR_PLUS;
    else if (c == '~') cls = CHAR_TILDE;
    else if (c == '*') cls = CHAR_STAR;
    else if (c == '[') cls = CHAR_LBRACKET;
    else if (c == ']') cls = CHAR_RBRACKET;
    else if (c == '"' || c == '\'') cls = CHAR_QUOTE;
    else if (c == '/') cls = CHAR_SLASH;
    else cls = CHAR_OTHER;

    char_classes[gid] = cls;
}

// Pass 2: Mark token boundaries (parallel)
kernel void css_mark_token_boundaries(
    device const uint8_t* css [[buffer(0)]],
    device const uint8_t* char_classes [[buffer(1)]],
    device uint8_t* is_token_start [[buffer(2)]],
    constant uint& css_length [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= css_length) return;

    uint8_t curr_class = char_classes[gid];
    uint8_t prev_class = gid > 0 ? char_classes[gid - 1] : CHAR_WHITESPACE;

    // Token starts when class changes (with some exceptions)
    bool is_start = false;

    if (curr_class != prev_class) {
        is_start = true;
    }

    // Special: digits after ident are same token (e.g., "h1")
    if (curr_class == CHAR_DIGIT && prev_class == CHAR_IDENT) {
        is_start = false;
    }

    // Special: hyphen in ident continues token
    if (gid > 0 && css[gid] == '-' && prev_class == CHAR_IDENT) {
        is_start = false;
    }

    is_token_start[gid] = is_start ? 1 : 0;
}

// Pass 3: Parallel prefix sum to get token indices
// (Use built-in parallel scan)

// Pass 4: Build token array (parallel per-token)
kernel void css_build_tokens(
    device const uint8_t* css [[buffer(0)]],
    device const uint8_t* char_classes [[buffer(1)]],
    device const uint8_t* is_token_start [[buffer(2)]],
    device const uint* token_indices [[buffer(3)]],  // From prefix sum
    device CSSToken* tokens [[buffer(4)]],
    constant uint& css_length [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= css_length) return;
    if (is_token_start[gid] == 0) return;

    uint token_idx = token_indices[gid];

    // Find token end
    uint end = gid + 1;
    while (end < css_length && is_token_start[end] == 0) {
        end++;
    }

    CSSToken tok;
    tok.start = gid;
    tok.length = end - gid;

    // Determine token type from first character class
    uint8_t cls = char_classes[gid];
    if (cls == CHAR_IDENT) tok.token_type = TOK_IDENT;
    else if (cls == CHAR_DOT) tok.token_type = TOK_DOT;
    else if (cls == CHAR_HASH) tok.token_type = TOK_HASH;
    // ... etc

    tokens[token_idx] = tok;
}

// Pass 5: Extract rules (parallel per-rule)
// First mark rule boundaries at '{', then parallel process
kernel void css_extract_rules(
    device const CSSToken* tokens [[buffer(0)]],
    device const uint* rule_starts [[buffer(1)]],  // Indices of '{' tokens
    device CSSRule* rules [[buffer(2)]],
    device SelectorPart* selectors [[buffer(3)]],
    device CSSProperty* properties [[buffer(4)]],
    device atomic_uint* selector_count [[buffer(5)]],
    device atomic_uint* property_count [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& rule_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rule_count) return;

    uint brace_idx = rule_starts[gid];

    // Walk backwards to find selector start
    uint sel_start = brace_idx;
    while (sel_start > 0) {
        uint prev = sel_start - 1;
        if (tokens[prev].token_type == TOK_RBRACE ||
            tokens[prev].token_type == TOK_SEMICOLON) {
            break;
        }
        sel_start = prev;
    }

    // Parse selectors (before '{')
    uint sel_output_start = atomic_fetch_add_explicit(selector_count, 0, memory_order_relaxed);
    uint specificity = 0;
    uint sel_count = 0;

    for (uint i = sel_start; i < brace_idx; i++) {
        CSSToken tok = tokens[i];
        if (tok.token_type == TOK_WHITESPACE) continue;

        SelectorPart part;
        part.combinator = COMB_NONE;

        if (tok.token_type == TOK_HASH) {
            // #id
            i++;
            if (i < brace_idx && tokens[i].token_type == TOK_IDENT) {
                part.part_type = SEL_ID;
                part.name_start = tokens[i].start;
                part.name_length = tokens[i].length;
                part.name_hash = compute_hash(css, tokens[i].start, tokens[i].length);
                specificity += 100;
            }
        } else if (tok.token_type == TOK_DOT) {
            // .class
            i++;
            if (i < brace_idx && tokens[i].token_type == TOK_IDENT) {
                part.part_type = SEL_CLASS;
                part.name_start = tokens[i].start;
                part.name_length = tokens[i].length;
                part.name_hash = compute_hash(css, tokens[i].start, tokens[i].length);
                specificity += 10;
            }
        } else if (tok.token_type == TOK_IDENT) {
            // element
            part.part_type = SEL_ELEMENT;
            part.name_start = tok.start;
            part.name_length = tok.length;
            part.name_hash = compute_hash(css, tok.start, tok.length);
            specificity += 1;
        } else if (tok.token_type == TOK_GT) {
            // Child combinator
            if (sel_count > 0) {
                selectors[sel_output_start + sel_count - 1].combinator = COMB_CHILD;
            }
            continue;
        }
        // ... handle other selector types

        uint slot = atomic_fetch_add_explicit(selector_count, 1, memory_order_relaxed);
        selectors[slot] = part;
        sel_count++;
    }

    // Parse properties (between '{' and '}')
    uint prop_output_start = atomic_fetch_add_explicit(property_count, 0, memory_order_relaxed);
    uint prop_count = 0;

    // Find closing brace
    uint close_brace = brace_idx + 1;
    while (close_brace < token_count && tokens[close_brace].token_type != TOK_RBRACE) {
        close_brace++;
    }

    // Parse property:value pairs
    uint i = brace_idx + 1;
    while (i < close_brace) {
        // Skip whitespace
        while (i < close_brace && tokens[i].token_type == TOK_WHITESPACE) i++;
        if (i >= close_brace) break;

        // Property name
        if (tokens[i].token_type != TOK_IDENT) { i++; continue; }
        uint prop_name_start = tokens[i].start;
        uint prop_name_len = tokens[i].length;
        i++;

        // Colon
        while (i < close_brace && tokens[i].token_type == TOK_WHITESPACE) i++;
        if (i >= close_brace || tokens[i].token_type != TOK_COLON) continue;
        i++;

        // Value (until semicolon or end)
        while (i < close_brace && tokens[i].token_type == TOK_WHITESPACE) i++;
        uint value_start = i;
        while (i < close_brace &&
               tokens[i].token_type != TOK_SEMICOLON &&
               tokens[i].token_type != TOK_RBRACE) {
            i++;
        }
        uint value_end = i;

        // Create property
        CSSProperty prop;
        prop.property_id = lookup_property_id(css, prop_name_start, prop_name_len);
        prop.value_start = tokens[value_start].start;
        prop.value_length = tokens[value_end - 1].start + tokens[value_end - 1].length - tokens[value_start].start;
        prop.numeric_value = parse_numeric_value(css, tokens, value_start, value_end);

        uint slot = atomic_fetch_add_explicit(property_count, 1, memory_order_relaxed);
        properties[slot] = prop;
        prop_count++;

        // Skip semicolon
        if (i < close_brace && tokens[i].token_type == TOK_SEMICOLON) i++;
    }

    // Build rule
    CSSRule rule;
    rule.selector_start = sel_output_start;
    rule.selector_count = sel_count;
    rule.specificity = specificity;
    rule.property_start = prop_output_start;
    rule.property_count = prop_count;
    rule.source_order = gid;

    rules[gid] = rule;
}
```

## Pseudocode

```
FUNCTION parse_css_gpu(css_buffer):
    // Pass 1: Classify characters
    char_classes = gpu_dispatch(css_classify_chars, css_buffer)

    // Pass 2: Mark token boundaries
    token_starts = gpu_dispatch(css_mark_token_boundaries, css_buffer, char_classes)

    // Pass 3: Prefix sum to get token indices
    token_indices = gpu_prefix_sum(token_starts)
    token_count = token_indices[len(css_buffer) - 1]

    // Pass 4: Build token array
    tokens = gpu_dispatch(css_build_tokens, css_buffer, token_indices, token_count)

    // Pass 5: Find rule boundaries ('{' tokens)
    rule_starts = filter(tokens, t => t.type == TOK_LBRACE)
    rule_count = len(rule_starts)

    // Pass 6: Extract rules in parallel
    (rules, selectors, properties) = gpu_dispatch(
        css_extract_rules, tokens, rule_starts, rule_count
    )

    RETURN CSSParseResult {
        rules,
        selectors,
        properties,
        token_count,
        rule_count
    }
```

## API

```rust
pub struct GpuCSSParser {
    device: Device,
    classify_pipeline: ComputePipelineState,
    mark_boundaries_pipeline: ComputePipelineState,
    build_tokens_pipeline: ComputePipelineState,
    extract_rules_pipeline: ComputePipelineState,
    prefix_sum_pipeline: ComputePipelineState,
}

pub struct CSSParseResult {
    pub rules: Buffer,           // CSSRule[]
    pub selectors: Buffer,       // SelectorPart[]
    pub properties: Buffer,      // CSSProperty[]
    pub rule_count: u32,
    pub selector_count: u32,
    pub property_count: u32,
}

impl GpuCSSParser {
    pub fn new(device: &Device) -> Self;
    pub fn parse(&self, css_buffer: &Buffer, css_length: u32) -> CSSParseResult;
}
```

## Tests

### Test 1: Character classification
```rust
#[test]
fn test_char_classification() {
    let css = b".foo { color: red; }";
    let classes = classify_chars(css);

    assert_eq!(classes[0], CHAR_DOT);
    assert_eq!(classes[1], CHAR_IDENT); // 'f'
    assert_eq!(classes[5], CHAR_WHITESPACE);
    assert_eq!(classes[6], CHAR_LBRACE);
}
```

### Test 2: Tokenization
```rust
#[test]
fn test_tokenization() {
    let css = b"div.foo #bar { display: none; }";
    let tokens = tokenize(css);

    assert_eq!(tokens[0].token_type, TOK_IDENT);  // div
    assert_eq!(tokens[1].token_type, TOK_DOT);    // .
    assert_eq!(tokens[2].token_type, TOK_IDENT);  // foo
    assert_eq!(tokens[4].token_type, TOK_HASH);   // #
    assert_eq!(tokens[5].token_type, TOK_IDENT);  // bar
}
```

### Test 3: Rule extraction
```rust
#[test]
fn test_rule_extraction() {
    let css = b".hidden { display: none; }";
    let result = parse_css(css);

    assert_eq!(result.rule_count, 1);

    let rule = &result.rules[0];
    assert_eq!(rule.selector_count, 1);
    assert_eq!(rule.property_count, 1);
    assert_eq!(rule.specificity, 10); // One class selector
}
```

### Test 4: Specificity calculation
```rust
#[test]
fn test_specificity() {
    // #id = 100, .class = 10, element = 1
    let cases = vec![
        ("div", 1),
        (".foo", 10),
        ("#bar", 100),
        ("div.foo", 11),
        ("#bar.foo", 110),
        ("div#bar.foo.baz", 121),
    ];

    for (selector, expected_spec) in cases {
        let css = format!("{} {{ }}", selector);
        let result = parse_css(css.as_bytes());
        assert_eq!(result.rules[0].specificity, expected_spec);
    }
}
```

### Test 5: Property parsing
```rust
#[test]
fn test_property_parsing() {
    let css = b".test { margin-top: 10px; display: block; color: #ff0000; }";
    let result = parse_css(css);

    assert_eq!(result.rules[0].property_count, 3);

    let props = &result.properties;
    assert_eq!(props[0].property_id, PROP_MARGIN_TOP);
    assert_eq!(props[0].numeric_value, 10.0);

    assert_eq!(props[1].property_id, PROP_DISPLAY);
    assert_eq!(props[2].property_id, PROP_COLOR);
}
```

### Test 6: Complex selectors
```rust
#[test]
fn test_complex_selectors() {
    let css = b"div > p.foo + span { }";
    let result = parse_css(css);

    let selectors = &result.selectors;
    assert_eq!(selectors[0].part_type, SEL_ELEMENT); // div
    assert_eq!(selectors[0].combinator, COMB_CHILD); // >
    assert_eq!(selectors[1].part_type, SEL_ELEMENT); // p
    assert_eq!(selectors[2].part_type, SEL_CLASS);   // .foo
    assert_eq!(selectors[2].combinator, COMB_ADJACENT); // +
    assert_eq!(selectors[3].part_type, SEL_ELEMENT); // span
}
```

### Test 7: Wikipedia CSS parsing
```rust
#[test]
fn test_wikipedia_css() {
    let css = include_bytes!("../testdata/wikipedia.css");
    let result = parse_css(css);

    // Wikipedia CSS has many rules
    assert!(result.rule_count > 500);

    // Should have mw-hidden rule
    let has_mw_hidden = result.rules.iter().any(|r| {
        let sel = &result.selectors[r.selector_start as usize];
        sel.part_type == SEL_CLASS &&
        string_matches(css, sel.name_start, sel.name_length, b"mw-hidden")
    });
    assert!(has_mw_hidden);
}
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/gpu_os/document/css_parser.rs` | Create | Parser implementation |
| `src/gpu_os/document/css_parser.metal` | Create | GPU kernels |
| `src/gpu_os/document/css_types.rs` | Create | Data structures |
| `tests/test_issue_95_css_parsing.rs` | Create | Tests |
| `testdata/wikipedia.css` | Create | Test fixture |

## Acceptance Criteria

1. [ ] Parallel character classification
2. [ ] Parallel tokenization
3. [ ] Parallel rule extraction
4. [ ] Correct specificity calculation
5. [ ] Property ID lookup for common properties
6. [ ] Numeric value parsing for lengths
7. [ ] Complex selector parsing (descendant, child, sibling)
8. [ ] Wikipedia CSS parses without errors (>500 rules)
9. [ ] All tests pass
