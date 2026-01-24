# Issue #103: CSS Parser Integration

## Summary
Integrate cssparser for spec-compliant CSS parsing, converting stylesheets into GPU-friendly rule representations for fast selector matching.

## Motivation
Current style handling is limited to inline styles and hardcoded defaults. Real web pages use:
- External stylesheets (`<link rel="stylesheet">`)
- Embedded styles (`<style>` blocks)
- Imported styles (`@import`)
- Media queries (`@media`)
- Complex selectors (combinators, pseudo-classes, attribute selectors)

## Architecture

```
              ┌─────────────────────────────────────────┐
              │           CSS Sources                   │
              │  (external, embedded, inline, @import)  │
              └──────────────┬──────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────┐
              │         cssparser Tokenizer             │
              │    (Streaming CSS token stream)         │
              └──────────────┬──────────────────────────┘
                             │
              ┌──────────────▼──────────────────────────┐
              │       Rule Parser (Custom)              │
              │  (Selectors + Declaration blocks)       │
              └──────────────┬──────────────────────────┘
                             │
        ┌────────────────────┼────────────────────────┐
        │                    │                        │
        ▼                    ▼                        ▼
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│ GPUSelector  │    │ GPUProperty  │    │ GPURule         │
│   Buffer     │    │   Buffer     │    │   Buffer        │
└──────────────┘    └──────────────┘    └─────────────────┘
```

## Data Structures

### Rust Side (CPU)

```rust
// Selector representation for GPU
#[repr(C)]
pub struct GPUSelector {
    pub type_hash: u32,         // Element type hash (0 = any)
    pub id_hash: u32,           // ID selector hash (0 = none)
    pub class_hash: u32,        // Class selector hash (0 = none)
    pub attribute_hash: u32,    // Attribute name hash (0 = none)
    pub attribute_op: u8,       // 0=exists, 1=equals, 2=contains, etc.
    pub combinator: u8,         // 0=none, 1=descendant, 2=child, 3=sibling
    pub pseudo_class: u8,       // 0=none, 1=hover, 2=active, 3=focus, etc.
    pub pseudo_element: u8,     // 0=none, 1=before, 2=after, etc.
    pub next_selector: u32,     // Index of next selector in chain (for combinators)
    pub specificity: u32,       // Packed (a << 20) | (b << 10) | c
}

// CSS property for GPU
#[repr(C)]
pub struct GPUProperty {
    pub property_id: u16,       // Encoded property name
    pub value_type: u8,         // 0=keyword, 1=length, 2=color, 3=percentage
    pub flags: u8,              // important, inherited, etc.
    pub value: GPUPropertyValue,
}

#[repr(C)]
pub union GPUPropertyValue {
    pub keyword: u32,           // For keywords (display: block)
    pub length: f32,            // For lengths (width: 100px)
    pub color: [u8; 4],         // For colors (color: #ff0000)
    pub percentage: f32,        // For percentages (width: 50%)
}

// CSS rule (selector + properties)
#[repr(C)]
pub struct GPURule {
    pub selector_offset: u32,   // Index into selector buffer
    pub selector_count: u16,    // Number of selectors (comma-separated)
    pub property_offset: u32,   // Index into property buffer
    pub property_count: u16,
    pub specificity: u32,       // Highest specificity among selectors
    pub source_order: u32,      // For cascade ordering
    pub origin: u8,             // 0=user-agent, 1=user, 2=author
    pub _padding: [u8; 3],
}

// Property ID encoding
pub const PROP_DISPLAY: u16 = 1;
pub const PROP_POSITION: u16 = 2;
pub const PROP_WIDTH: u16 = 3;
pub const PROP_HEIGHT: u16 = 4;
pub const PROP_MARGIN_TOP: u16 = 5;
pub const PROP_MARGIN_RIGHT: u16 = 6;
pub const PROP_MARGIN_BOTTOM: u16 = 7;
pub const PROP_MARGIN_LEFT: u16 = 8;
pub const PROP_PADDING_TOP: u16 = 9;
// ... 200+ properties

// Keyword encoding
pub const KW_BLOCK: u32 = 1;
pub const KW_INLINE: u32 = 2;
pub const KW_FLEX: u32 = 3;
pub const KW_GRID: u32 = 4;
pub const KW_NONE: u32 = 5;
pub const KW_AUTO: u32 = 6;
pub const KW_INHERIT: u32 = 7;
pub const KW_INITIAL: u32 = 8;
// ...

// Parser state
pub struct CSSParser {
    selectors: Vec<GPUSelector>,
    properties: Vec<GPUProperty>,
    rules: Vec<GPURule>,
    source_order: u32,
}
```

### Metal Side (GPU)

```metal
// Must match Rust repr(C)
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

struct GPUProperty {
    ushort property_id;
    uchar value_type;
    uchar flags;
    uint value;  // Union interpreted based on value_type
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

// Property constants
constant ushort PROP_DISPLAY = 1;
constant ushort PROP_POSITION = 2;
constant ushort PROP_WIDTH = 3;
constant ushort PROP_HEIGHT = 4;
// ...

// Value type constants
constant uchar VALUE_KEYWORD = 0;
constant uchar VALUE_LENGTH = 1;
constant uchar VALUE_COLOR = 2;
constant uchar VALUE_PERCENTAGE = 3;

// Extract typed value from property
float get_length(GPUProperty prop) {
    return as_type<float>(prop.value);
}

float4 get_color(GPUProperty prop) {
    uchar4 bytes = as_type<uchar4>(prop.value);
    return float4(bytes) / 255.0;
}

float get_percentage(GPUProperty prop) {
    return as_type<float>(prop.value);
}
```

## Implementation

### Step 1: Add Dependencies

```toml
# Cargo.toml
[dependencies]
cssparser = "0.36"
selectors = "0.29"  # Optional: for complex selector parsing
```

### Step 2: CSS Rule Parser

```rust
// src/gpu_os/document/css_parser.rs

use cssparser::{Parser, ParserInput, Token, ParseError, CowRcStr};

impl CSSParser {
    pub fn new() -> Self {
        Self {
            selectors: Vec::with_capacity(1024),
            properties: Vec::with_capacity(4096),
            rules: Vec::with_capacity(256),
            source_order: 0,
        }
    }

    pub fn parse_stylesheet(&mut self, css: &str) -> Result<(), ParseError<'_, ()>> {
        let mut input = ParserInput::new(css);
        let mut parser = Parser::new(&mut input);

        while !parser.is_exhausted() {
            self.parse_rule(&mut parser)?;
        }

        Ok(())
    }

    fn parse_rule<'i>(&mut self, parser: &mut Parser<'i, '_>) -> Result<(), ParseError<'i, ()>> {
        // Parse selector(s)
        let selector_start = self.selectors.len() as u32;
        let mut selector_count = 0;
        let mut max_specificity = 0u32;

        loop {
            let selector = self.parse_selector(parser)?;
            max_specificity = max_specificity.max(selector.specificity);
            self.selectors.push(selector);
            selector_count += 1;

            // Check for comma (selector list) or opening brace
            match parser.next()? {
                Token::Comma => continue,
                Token::CurlyBracketBlock => break,
                _ => return Err(parser.new_unexpected_token_error(Token::CurlyBracketBlock)),
            }
        }

        // Parse declaration block
        let property_start = self.properties.len() as u32;
        let property_count = parser.parse_nested_block(|parser| {
            self.parse_declarations(parser)
        })?;

        // Create rule
        self.rules.push(GPURule {
            selector_offset: selector_start,
            selector_count,
            property_offset: property_start,
            property_count,
            specificity: max_specificity,
            source_order: self.source_order,
            origin: 2, // Author stylesheet
            _padding: [0; 3],
        });

        self.source_order += 1;
        Ok(())
    }

    fn parse_selector<'i>(&mut self, parser: &mut Parser<'i, '_>) -> Result<GPUSelector, ParseError<'i, ()>> {
        let mut selector = GPUSelector {
            type_hash: 0,
            id_hash: 0,
            class_hash: 0,
            attribute_hash: 0,
            attribute_op: 0,
            combinator: 0,
            pseudo_class: 0,
            pseudo_element: 0,
            next_selector: INVALID_IDX,
            specificity: 0,
        };

        let mut specificity_a = 0u32; // ID selectors
        let mut specificity_b = 0u32; // Class, attribute, pseudo-class
        let mut specificity_c = 0u32; // Type selectors, pseudo-elements

        loop {
            match parser.next_including_whitespace()? {
                Token::Ident(name) => {
                    // Type selector
                    selector.type_hash = hash_element_type(&name);
                    specificity_c += 1;
                }
                Token::IDHash(id) => {
                    selector.id_hash = hash_string(&id);
                    specificity_a += 1;
                }
                Token::Delim('.') => {
                    // Class selector
                    if let Token::Ident(class) = parser.next()? {
                        selector.class_hash = hash_string(&class);
                        specificity_b += 1;
                    }
                }
                Token::SquareBracketBlock => {
                    // Attribute selector [attr] or [attr=value]
                    parser.parse_nested_block(|parser| {
                        self.parse_attribute_selector(parser, &mut selector)
                    })?;
                    specificity_b += 1;
                }
                Token::Colon => {
                    // Pseudo-class or pseudo-element
                    match parser.next()? {
                        Token::Colon => {
                            // ::pseudo-element
                            if let Token::Ident(name) = parser.next()? {
                                selector.pseudo_element = encode_pseudo_element(&name);
                                specificity_c += 1;
                            }
                        }
                        Token::Ident(name) => {
                            // :pseudo-class
                            selector.pseudo_class = encode_pseudo_class(&name);
                            specificity_b += 1;
                        }
                        _ => {}
                    }
                }
                Token::WhiteSpace(_) => {
                    // Descendant combinator (space)
                    // Check if followed by another selector or end
                    let state = parser.state();
                    match parser.next_including_whitespace() {
                        Ok(Token::CurlyBracketBlock) | Ok(Token::Comma) => {
                            parser.reset(&state);
                            break;
                        }
                        Ok(_) => {
                            parser.reset(&state);
                            selector.combinator = 1; // Descendant
                            break;
                        }
                        Err(_) => break,
                    }
                }
                Token::Delim('>') => {
                    selector.combinator = 2; // Child
                    break;
                }
                Token::Delim('+') => {
                    selector.combinator = 3; // Adjacent sibling
                    break;
                }
                Token::Delim('~') => {
                    selector.combinator = 4; // General sibling
                    break;
                }
                Token::Delim('*') => {
                    // Universal selector
                    specificity_c += 0; // No specificity added
                }
                _ => break,
            }
        }

        selector.specificity = (specificity_a << 20) | (specificity_b << 10) | specificity_c;

        // If this selector has a combinator, parse the next part
        if selector.combinator != 0 {
            let next_selector = self.parse_selector(parser)?;
            selector.next_selector = self.selectors.len() as u32;
            self.selectors.push(next_selector);
        }

        Ok(selector)
    }

    fn parse_declarations<'i>(&mut self, parser: &mut Parser<'i, '_>) -> Result<u16, ParseError<'i, ()>> {
        let mut count = 0u16;

        while !parser.is_exhausted() {
            // Property name
            let name = match parser.next() {
                Ok(Token::Ident(name)) => name,
                Ok(Token::Semicolon) => continue,
                Err(_) => break,
                _ => continue,
            };

            // Colon
            parser.expect_colon()?;

            // Property value
            let property_id = encode_property_name(&name);
            if property_id != 0 {
                if let Some(prop) = self.parse_property_value(parser, property_id)? {
                    self.properties.push(prop);
                    count += 1;
                }
            } else {
                // Skip unknown property
                self.skip_until_semicolon(parser);
            }
        }

        Ok(count)
    }

    fn parse_property_value<'i>(
        &mut self,
        parser: &mut Parser<'i, '_>,
        property_id: u16
    ) -> Result<Option<GPUProperty>, ParseError<'i, ()>> {
        let mut prop = GPUProperty {
            property_id,
            value_type: 0,
            flags: 0,
            value: GPUPropertyValue { keyword: 0 },
        };

        let token = parser.next()?;

        match token {
            Token::Ident(value) => {
                // Keyword value
                prop.value_type = VALUE_KEYWORD;
                prop.value = GPUPropertyValue { keyword: encode_keyword(&value) };
            }
            Token::Dimension { value, unit, .. } => {
                // Length value
                prop.value_type = VALUE_LENGTH;
                let px_value = convert_to_px(value, &unit);
                prop.value = GPUPropertyValue { length: px_value };
            }
            Token::Percentage { unit_value, .. } => {
                prop.value_type = VALUE_PERCENTAGE;
                prop.value = GPUPropertyValue { percentage: unit_value };
            }
            Token::Hash(color) | Token::IDHash(color) => {
                // Color value
                prop.value_type = VALUE_COLOR;
                let rgba = parse_hex_color(&color);
                prop.value = GPUPropertyValue { color: rgba };
            }
            Token::Function(name) if name.eq_ignore_ascii_case("rgb") ||
                                      name.eq_ignore_ascii_case("rgba") => {
                prop.value_type = VALUE_COLOR;
                let rgba = parser.parse_nested_block(|p| parse_rgb_function(p))?;
                prop.value = GPUPropertyValue { color: rgba };
            }
            _ => return Ok(None),
        }

        // Check for !important
        loop {
            match parser.next() {
                Ok(Token::Delim('!')) => {
                    if let Ok(Token::Ident(imp)) = parser.next() {
                        if imp.eq_ignore_ascii_case("important") {
                            prop.flags |= 1; // Important flag
                        }
                    }
                }
                Ok(Token::Semicolon) | Err(_) => break,
                _ => {}
            }
        }

        Ok(Some(prop))
    }
}

fn convert_to_px(value: f32, unit: &str) -> f32 {
    match unit {
        "px" => value,
        "em" => value * 16.0,  // Base font size
        "rem" => value * 16.0,
        "pt" => value * 1.333,
        "pc" => value * 16.0,
        "in" => value * 96.0,
        "cm" => value * 37.795,
        "mm" => value * 3.7795,
        "vw" => value * 10.0,  // Placeholder, needs viewport
        "vh" => value * 10.0,
        _ => value,
    }
}

fn encode_property_name(name: &str) -> u16 {
    match name.to_lowercase().as_str() {
        "display" => PROP_DISPLAY,
        "position" => PROP_POSITION,
        "width" => PROP_WIDTH,
        "height" => PROP_HEIGHT,
        "margin" => PROP_MARGIN, // Shorthand
        "margin-top" => PROP_MARGIN_TOP,
        "margin-right" => PROP_MARGIN_RIGHT,
        "margin-bottom" => PROP_MARGIN_BOTTOM,
        "margin-left" => PROP_MARGIN_LEFT,
        "padding" => PROP_PADDING,
        "padding-top" => PROP_PADDING_TOP,
        "padding-right" => PROP_PADDING_RIGHT,
        "padding-bottom" => PROP_PADDING_BOTTOM,
        "padding-left" => PROP_PADDING_LEFT,
        "color" => PROP_COLOR,
        "background-color" => PROP_BACKGROUND_COLOR,
        "background" => PROP_BACKGROUND,
        "font-size" => PROP_FONT_SIZE,
        "font-family" => PROP_FONT_FAMILY,
        "font-weight" => PROP_FONT_WEIGHT,
        "line-height" => PROP_LINE_HEIGHT,
        "text-align" => PROP_TEXT_ALIGN,
        "border" => PROP_BORDER,
        "border-width" => PROP_BORDER_WIDTH,
        "border-color" => PROP_BORDER_COLOR,
        "border-style" => PROP_BORDER_STYLE,
        "border-radius" => PROP_BORDER_RADIUS,
        "overflow" => PROP_OVERFLOW,
        "visibility" => PROP_VISIBILITY,
        "opacity" => PROP_OPACITY,
        "z-index" => PROP_Z_INDEX,
        "top" => PROP_TOP,
        "right" => PROP_RIGHT,
        "bottom" => PROP_BOTTOM,
        "left" => PROP_LEFT,
        "flex" => PROP_FLEX,
        "flex-direction" => PROP_FLEX_DIRECTION,
        "justify-content" => PROP_JUSTIFY_CONTENT,
        "align-items" => PROP_ALIGN_ITEMS,
        "gap" => PROP_GAP,
        _ => 0, // Unknown property
    }
}

fn encode_keyword(value: &str) -> u32 {
    match value.to_lowercase().as_str() {
        "block" => KW_BLOCK,
        "inline" => KW_INLINE,
        "inline-block" => KW_INLINE_BLOCK,
        "flex" => KW_FLEX,
        "grid" => KW_GRID,
        "none" => KW_NONE,
        "auto" => KW_AUTO,
        "inherit" => KW_INHERIT,
        "initial" => KW_INITIAL,
        "unset" => KW_UNSET,
        "static" => KW_STATIC,
        "relative" => KW_RELATIVE,
        "absolute" => KW_ABSOLUTE,
        "fixed" => KW_FIXED,
        "sticky" => KW_STICKY,
        "visible" => KW_VISIBLE,
        "hidden" => KW_HIDDEN,
        "scroll" => KW_SCROLL,
        "left" => KW_LEFT,
        "right" => KW_RIGHT,
        "center" => KW_CENTER,
        "justify" => KW_JUSTIFY,
        "bold" => KW_BOLD,
        "normal" => KW_NORMAL,
        "row" => KW_ROW,
        "column" => KW_COLUMN,
        "wrap" => KW_WRAP,
        "nowrap" => KW_NOWRAP,
        "start" => KW_START,
        "end" => KW_END,
        "stretch" => KW_STRETCH,
        "space-between" => KW_SPACE_BETWEEN,
        "space-around" => KW_SPACE_AROUND,
        _ => 0,
    }
}
```

### Step 3: External CSS Loading

```rust
// src/gpu_os/document/css_loader.rs

use std::collections::HashMap;

pub struct CSSLoader {
    cache: HashMap<String, String>,
}

impl CSSLoader {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    pub fn load_stylesheet(&mut self, url: &str) -> Result<String, CSSLoadError> {
        // Check cache first
        if let Some(css) = self.cache.get(url) {
            return Ok(css.clone());
        }

        // Fetch via HTTP
        let response = ureq::get(url)
            .set("Accept", "text/css")
            .call()?;

        let css = response.into_string()?;
        self.cache.insert(url.to_string(), css.clone());
        Ok(css)
    }

    pub fn extract_stylesheets(&mut self, doc: &GPUDocument, base_url: &str) -> Vec<String> {
        let mut stylesheets = Vec::new();

        // Find <link rel="stylesheet"> elements
        for node_idx in 0..doc.nodes.len() {
            let node = &doc.nodes[node_idx];
            if node.element_type == ELEM_LINK {
                let rel = doc.get_attribute_value(node_idx as u32, "rel");
                if rel == Some("stylesheet") {
                    if let Some(href) = doc.get_attribute_value(node_idx as u32, "href") {
                        let url = resolve_url(base_url, href);
                        if let Ok(css) = self.load_stylesheet(&url) {
                            stylesheets.push(css);
                        }
                    }
                }
            }
        }

        // Find <style> elements
        for node_idx in 0..doc.nodes.len() {
            let node = &doc.nodes[node_idx];
            if node.element_type == ELEM_STYLE {
                let text_content = doc.get_text_content(node_idx as u32);
                stylesheets.push(text_content);
            }
        }

        stylesheets
    }
}
```

## Benchmarks

### Benchmark 1: CSS Parse Time

```rust
fn bench_css_parsing(c: &mut Criterion) {
    let test_cases = vec![
        ("minimal", "div { color: red; }"),
        ("medium", include_str!("../test_css/bootstrap.min.css")),
        ("large", include_str!("../test_css/wikipedia.css")),
    ];

    let mut group = c.benchmark_group("css_parsing");

    for (name, css) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("cssparser", name),
            &css,
            |b, css| {
                b.iter(|| {
                    let mut parser = CSSParser::new();
                    parser.parse_stylesheet(css)
                })
            }
        );
    }

    group.finish();
}
```

### Expected Results

| CSS Size | Rules | Parse Time | GPU Upload |
|----------|-------|------------|------------|
| 1KB | 20 | 0.1ms | 0.01ms |
| 100KB (Bootstrap) | 1000 | 5ms | 0.1ms |
| 500KB (Wikipedia) | 5000 | 20ms | 0.3ms |

## Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_selector() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("div { color: red; }").unwrap();

        assert_eq!(parser.rules.len(), 1);
        assert_eq!(parser.selectors.len(), 1);
        assert!(parser.selectors[0].type_hash != 0);
    }

    #[test]
    fn test_class_selector() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet(".container { width: 100px; }").unwrap();

        assert!(parser.selectors[0].class_hash != 0);
        assert_eq!(parser.selectors[0].type_hash, 0);
    }

    #[test]
    fn test_id_selector() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("#main { height: 50%; }").unwrap();

        assert!(parser.selectors[0].id_hash != 0);
    }

    #[test]
    fn test_compound_selector() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("div.container#main { color: blue; }").unwrap();

        let sel = &parser.selectors[0];
        assert!(sel.type_hash != 0);
        assert!(sel.class_hash != 0);
        assert!(sel.id_hash != 0);
    }

    #[test]
    fn test_descendant_combinator() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("div p { color: green; }").unwrap();

        assert_eq!(parser.selectors[0].combinator, 1); // Descendant
        assert!(parser.selectors[0].next_selector != INVALID_IDX);
    }

    #[test]
    fn test_child_combinator() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("ul > li { margin: 0; }").unwrap();

        assert_eq!(parser.selectors[0].combinator, 2); // Child
    }

    #[test]
    fn test_specificity() {
        let mut parser = CSSParser::new();

        // ID = 1,0,0
        parser.parse_stylesheet("#id { color: red; }").unwrap();
        assert_eq!(parser.selectors[0].specificity >> 20, 1);

        // Class = 0,1,0
        parser.selectors.clear();
        parser.parse_stylesheet(".class { color: red; }").unwrap();
        assert_eq!((parser.selectors[0].specificity >> 10) & 0x3FF, 1);

        // Type = 0,0,1
        parser.selectors.clear();
        parser.parse_stylesheet("div { color: red; }").unwrap();
        assert_eq!(parser.selectors[0].specificity & 0x3FF, 1);
    }

    #[test]
    fn test_property_parsing() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("div { width: 100px; height: 50%; color: #ff0000; }").unwrap();

        assert_eq!(parser.properties.len(), 3);

        // Width: 100px
        assert_eq!(parser.properties[0].property_id, PROP_WIDTH);
        assert_eq!(parser.properties[0].value_type, VALUE_LENGTH);

        // Height: 50%
        assert_eq!(parser.properties[1].property_id, PROP_HEIGHT);
        assert_eq!(parser.properties[1].value_type, VALUE_PERCENTAGE);

        // Color
        assert_eq!(parser.properties[2].property_id, PROP_COLOR);
        assert_eq!(parser.properties[2].value_type, VALUE_COLOR);
    }

    #[test]
    fn test_important() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("div { color: red !important; }").unwrap();

        assert_eq!(parser.properties[0].flags & 1, 1);
    }

    #[test]
    fn test_multiple_rules() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet(r#"
            div { color: red; }
            .class { color: blue; }
            #id { color: green; }
        "#).unwrap();

        assert_eq!(parser.rules.len(), 3);
        assert_eq!(parser.rules[0].source_order, 0);
        assert_eq!(parser.rules[1].source_order, 1);
        assert_eq!(parser.rules[2].source_order, 2);
    }

    #[test]
    fn test_selector_list() {
        let mut parser = CSSParser::new();
        parser.parse_stylesheet("h1, h2, h3 { font-weight: bold; }").unwrap();

        assert_eq!(parser.rules[0].selector_count, 3);
    }
}
```

## Acceptance Criteria

- [ ] Parse valid CSS without errors
- [ ] Support all common selectors (type, class, ID, attribute)
- [ ] Support combinators (descendant, child, sibling)
- [ ] Support pseudo-classes (:hover, :active, :first-child, etc.)
- [ ] Parse all common properties (50+ properties)
- [ ] Handle shorthand properties (margin, padding, border)
- [ ] Calculate specificity correctly
- [ ] Track source order for cascade
- [ ] Handle !important declarations
- [ ] Load external stylesheets via HTTP
- [ ] Extract <style> blocks from HTML
- [ ] GPU buffer upload in <1ms for typical stylesheets

## Dependencies

- Issue #102: HTML5 Parser Integration (for stylesheet extraction)

## Blocks

- Issue #104: GPU Selector Matching
- Issue #105: GPU Cascade Resolution
