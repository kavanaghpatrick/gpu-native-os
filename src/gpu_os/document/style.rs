//! GPU-Native CSS Style Resolution
//!
//! Matches CSS selectors to elements and computes final styles.
//! Fully parallel - each element can be styled independently.

use metal::*;
use super::parser::Element;
use super::tokenizer::Token;

// Selector types (must match Metal shader)
pub const SEL_UNIVERSAL: u32 = 0;
pub const SEL_TAG: u32 = 1;
pub const SEL_CLASS: u32 = 2;
pub const SEL_ID: u32 = 3;
pub const SEL_ATTRIBUTE: u32 = 4;
pub const SEL_PSEUDO: u32 = 5;

// Combinator types
pub const COMB_NONE: u32 = 0;
pub const COMB_DESCENDANT: u32 = 1;  // E F (space)
pub const COMB_CHILD: u32 = 2;       // E > F
pub const COMB_ADJACENT: u32 = 3;    // E + F
pub const COMB_SIBLING: u32 = 4;     // E ~ F

// Attribute match operators
pub const ATTR_EXISTS: u32 = 0;      // [attr]
pub const ATTR_EQUALS: u32 = 1;      // [attr=val]
pub const ATTR_CONTAINS: u32 = 2;    // [attr*=val]
pub const ATTR_STARTS: u32 = 3;      // [attr^=val]
pub const ATTR_ENDS: u32 = 4;        // [attr$=val]
pub const ATTR_WORD: u32 = 5;        // [attr~=val]
pub const ATTR_LANG: u32 = 6;        // [attr|=val]

// Pseudo-class types
pub const PSEUDO_FIRST_CHILD: u32 = 1;
pub const PSEUDO_LAST_CHILD: u32 = 2;
pub const PSEUDO_NTH_CHILD: u32 = 3;
pub const PSEUDO_FIRST_OF_TYPE: u32 = 4;
pub const PSEUDO_LAST_OF_TYPE: u32 = 5;
pub const PSEUDO_ONLY_CHILD: u32 = 6;
pub const PSEUDO_EMPTY: u32 = 7;
pub const PSEUDO_ROOT: u32 = 8;

// Display values
pub const DISPLAY_NONE: u32 = 0;
pub const DISPLAY_BLOCK: u32 = 1;
pub const DISPLAY_INLINE: u32 = 2;
pub const DISPLAY_FLEX: u32 = 3;
pub const DISPLAY_INLINE_BLOCK: u32 = 4;
pub const DISPLAY_TABLE: u32 = 5;
pub const DISPLAY_TABLE_ROW: u32 = 6;
pub const DISPLAY_TABLE_CELL: u32 = 7;

// Flex direction
pub const FLEX_ROW: u32 = 0;
pub const FLEX_COLUMN: u32 = 1;

// Justify content
pub const JUSTIFY_START: u32 = 0;
pub const JUSTIFY_CENTER: u32 = 1;
pub const JUSTIFY_END: u32 = 2;
pub const JUSTIFY_SPACE_BETWEEN: u32 = 3;
pub const JUSTIFY_SPACE_AROUND: u32 = 4;

// Align items
pub const ALIGN_START: u32 = 0;
pub const ALIGN_CENTER: u32 = 1;
pub const ALIGN_END: u32 = 2;
pub const ALIGN_STRETCH: u32 = 3;

// Text align
pub const TEXT_LEFT: u32 = 0;
pub const TEXT_CENTER: u32 = 1;
pub const TEXT_RIGHT: u32 = 2;

// CSS Position values
pub const POSITION_STATIC: u32 = 0;
pub const POSITION_RELATIVE: u32 = 1;
pub const POSITION_ABSOLUTE: u32 = 2;
pub const POSITION_FIXED: u32 = 3;

/// Special value indicating "auto" for offset properties
pub const OFFSET_AUTO: f32 = f32::MAX;

// CSS Overflow values
pub const OVERFLOW_VISIBLE: u32 = 0;
pub const OVERFLOW_HIDDEN: u32 = 1;
pub const OVERFLOW_SCROLL: u32 = 2;
pub const OVERFLOW_AUTO: u32 = 3;

// CSS Gradient types
pub const GRADIENT_NONE: u32 = 0;
pub const GRADIENT_LINEAR: u32 = 1;
pub const GRADIENT_RADIAL: u32 = 2;

// Property IDs
pub const PROP_DISPLAY: u32 = 0;
pub const PROP_WIDTH: u32 = 1;
pub const PROP_HEIGHT: u32 = 2;
pub const PROP_MARGIN: u32 = 3;
pub const PROP_PADDING: u32 = 4;
pub const PROP_COLOR: u32 = 5;
pub const PROP_BACKGROUND: u32 = 6;
pub const PROP_FONT_SIZE: u32 = 7;
pub const PROP_LINE_HEIGHT: u32 = 8;
pub const PROP_FONT_WEIGHT: u32 = 9;
pub const PROP_TEXT_ALIGN: u32 = 10;
pub const PROP_FLEX_DIRECTION: u32 = 11;
pub const PROP_JUSTIFY_CONTENT: u32 = 12;
pub const PROP_ALIGN_ITEMS: u32 = 13;
pub const PROP_FLEX_GROW: u32 = 14;
pub const PROP_FLEX_SHRINK: u32 = 15;
pub const PROP_BORDER_WIDTH: u32 = 16;
pub const PROP_BORDER_COLOR: u32 = 17;
pub const PROP_BORDER_RADIUS: u32 = 18;
pub const PROP_OPACITY: u32 = 19;
pub const PROP_POSITION: u32 = 20;
pub const PROP_TOP: u32 = 21;
pub const PROP_RIGHT: u32 = 22;
pub const PROP_BOTTOM: u32 = 23;
pub const PROP_LEFT: u32 = 24;
pub const PROP_Z_INDEX: u32 = 25;
pub const PROP_OVERFLOW: u32 = 26;
pub const PROP_OVERFLOW_X: u32 = 27;
pub const PROP_OVERFLOW_Y: u32 = 28;
pub const PROP_BOX_SHADOW: u32 = 29;        // offset_x, offset_y, blur, spread
pub const PROP_BOX_SHADOW_COLOR: u32 = 30;  // RGBA color
pub const PROP_BOX_SHADOW_INSET: u32 = 31;  // inset flag
pub const PROP_GRADIENT_TYPE: u32 = 32;     // type (NONE, LINEAR, RADIAL)
pub const PROP_GRADIENT_ANGLE: u32 = 33;    // angle in degrees
pub const PROP_GRADIENT_STOP: u32 = 34;     // color stop (idx, pos, r, g, b, a encoded)
pub const PROP_BORDER_COLLAPSE: u32 = 35;   // 0 = separate, 1 = collapse
pub const PROP_BORDER_SPACING: u32 = 36;    // spacing in pixels

// Buffer sizes
pub const MAX_SELECTORS: usize = 1024;
pub const MAX_STYLE_DEFS: usize = 4096;
const THREAD_COUNT: u64 = 1024;

/// A CSS selector (supports combinators and complex selectors)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Selector {
    pub selector_type: u32,      // TAG, CLASS, ID, ATTRIBUTE, PSEUDO
    pub hash: u32,               // Hash of tag/class/id name
    pub specificity: u32,
    pub style_start: u32,
    pub style_count: u32,
    pub combinator: u32,         // NONE, DESCENDANT, CHILD, ADJACENT, SIBLING
    pub next_part: i32,          // Index of next selector part (for complex selectors like "div p")
    pub pseudo_type: u32,        // For pseudo-classes: FIRST_CHILD, NTH_CHILD, etc.
    pub attr_name_hash: u32,     // For attribute selectors
    pub attr_op: u32,            // ATTR_EXISTS, ATTR_EQUALS, etc.
    pub attr_value_hash: u32,    // For attribute value matching
    pub nth_a: i32,              // For :nth-child(an+b)
    pub nth_b: i32,
    pub _padding: [u32; 3],
}

/// A single CSS property definition
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StyleDef {
    pub property_id: u32,
    pub values: [f32; 4],
}

// Property set bitmask flags (for tracking which properties were explicitly set)
pub const PROP_SET_COLOR: u32 = 1 << 0;
pub const PROP_SET_FONT_SIZE: u32 = 1 << 1;
pub const PROP_SET_LINE_HEIGHT: u32 = 1 << 2;
pub const PROP_SET_FONT_WEIGHT: u32 = 1 << 3;
pub const PROP_SET_TEXT_ALIGN: u32 = 1 << 4;

/// Computed style for an element
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ComputedStyle {
    pub display: u32,
    pub width: f32,
    pub height: f32,
    pub margin: [f32; 4],
    pub padding: [f32; 4],
    pub flex_direction: u32,
    pub justify_content: u32,
    pub align_items: u32,
    pub flex_grow: f32,
    pub flex_shrink: f32,
    pub color: [f32; 4],
    pub font_size: f32,
    pub line_height: f32,
    pub font_weight: u32,
    pub text_align: u32,
    pub background_color: [f32; 4],
    pub border_width: [f32; 4],
    pub border_color: [f32; 4],
    pub border_radius: f32,
    pub opacity: f32,
    // CSS Positioning
    pub position: u32,       // POSITION_STATIC, RELATIVE, ABSOLUTE, FIXED
    pub top: f32,            // Offset from top (OFFSET_AUTO = auto)
    pub right: f32,          // Offset from right
    pub bottom: f32,         // Offset from bottom
    pub left: f32,           // Offset from left
    pub z_index: i32,        // Stacking order (0 = auto/default)
    /// Bitmask of explicitly set properties (for inheritance)
    pub properties_set: u32,
    // CSS Overflow
    pub overflow_x: u32,     // OVERFLOW_VISIBLE, HIDDEN, SCROLL, AUTO
    pub overflow_y: u32,     // OVERFLOW_VISIBLE, HIDDEN, SCROLL, AUTO
    // Box Shadow (up to 4 shadows)
    pub shadow_count: u32,           // Number of shadows (0-4)
    pub shadow_offset_x: [f32; 4],   // Horizontal offset per shadow
    pub shadow_offset_y: [f32; 4],   // Vertical offset per shadow
    pub shadow_blur: [f32; 4],       // Blur radius per shadow
    pub shadow_spread: [f32; 4],     // Spread radius per shadow
    pub shadow_color: [f32; 16],     // RGBA color per shadow (4 floats * 4 shadows)
    pub shadow_inset: [u32; 4],      // 1 = inset, 0 = outset
    // Gradients (up to 8 color stops)
    pub gradient_type: u32,          // GRADIENT_NONE, LINEAR, RADIAL
    pub gradient_angle: f32,         // Angle in degrees (for linear gradient)
    pub gradient_stop_count: u32,    // Number of color stops (0-8)
    pub gradient_stop_colors: [f32; 32], // RGBA color per stop (4 floats * 8 stops)
    pub gradient_stop_positions: [f32; 8], // Position 0.0-1.0 per stop
    // Table layout
    pub border_collapse: u32,        // 0 = separate, 1 = collapse
    pub border_spacing: f32,         // Spacing between cells
    pub _padding: [f32; 2],          // Pad to maintain 16-byte alignment
}

impl Default for ComputedStyle {
    fn default() -> Self {
        Self {
            display: DISPLAY_BLOCK,
            width: 0.0,
            height: 0.0,
            margin: [0.0; 4],
            padding: [0.0; 4],
            flex_direction: FLEX_ROW,
            justify_content: JUSTIFY_START,
            align_items: ALIGN_STRETCH,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            color: [0.0, 0.0, 0.0, 1.0],
            font_size: 16.0,
            line_height: 1.2,
            font_weight: 400,
            text_align: TEXT_LEFT,
            background_color: [0.0, 0.0, 0.0, 0.0],
            border_width: [0.0; 4],
            border_color: [0.0, 0.0, 0.0, 1.0],
            border_radius: 0.0,
            opacity: 1.0,
            // CSS Positioning defaults
            position: POSITION_STATIC,
            top: OFFSET_AUTO,
            right: OFFSET_AUTO,
            bottom: OFFSET_AUTO,
            left: OFFSET_AUTO,
            z_index: 0,
            properties_set: 0,  // No properties explicitly set
            // CSS Overflow defaults
            overflow_x: OVERFLOW_VISIBLE,
            overflow_y: OVERFLOW_VISIBLE,
            // Box Shadow defaults
            shadow_count: 0,
            shadow_offset_x: [0.0; 4],
            shadow_offset_y: [0.0; 4],
            shadow_blur: [0.0; 4],
            shadow_spread: [0.0; 4],
            shadow_color: [0.0; 16],
            shadow_inset: [0; 4],
            // Gradient defaults
            gradient_type: GRADIENT_NONE,
            gradient_angle: 180.0,  // Default: to bottom
            gradient_stop_count: 0,
            gradient_stop_colors: [0.0; 32],
            gradient_stop_positions: [0.0; 8],
            // Table layout defaults
            border_collapse: 0,  // separate
            border_spacing: 0.0,
            _padding: [0.0; 2],
        }
    }
}

/// A parsed stylesheet
#[derive(Clone, Debug, Default)]
pub struct Stylesheet {
    pub selectors: Vec<Selector>,
    pub style_defs: Vec<StyleDef>,
}

impl Stylesheet {
    /// Parse a CSS string into a stylesheet
    pub fn parse(css: &str) -> Self {
        let mut selectors = Vec::new();
        let mut style_defs = Vec::new();

        // Simple CSS parser: selector { property: value; }
        let mut chars = css.chars().peekable();

        while chars.peek().is_some() {
            // Skip whitespace
            while chars.peek().map_or(false, |c| c.is_whitespace()) {
                chars.next();
            }

            if chars.peek().is_none() {
                break;
            }

            // Parse selector
            let mut selector_str = String::new();
            while chars.peek().map_or(false, |&c| c != '{') {
                selector_str.push(chars.next().unwrap());
            }
            let selector_str = selector_str.trim();

            if selector_str.is_empty() {
                break;
            }

            // Skip '{'
            chars.next();

            // Parse properties
            let mut properties_str = String::new();
            let mut brace_depth = 1;
            while brace_depth > 0 {
                match chars.next() {
                    Some('{') => {
                        brace_depth += 1;
                        properties_str.push('{');
                    }
                    Some('}') => {
                        brace_depth -= 1;
                        if brace_depth > 0 {
                            properties_str.push('}');
                        }
                    }
                    Some(c) => properties_str.push(c),
                    None => break,
                }
            }

            // Parse style definitions first
            let style_start = style_defs.len() as u32;
            let mut style_count = 0u32;

            for prop in properties_str.split(';') {
                let prop = prop.trim();
                if prop.is_empty() {
                    continue;
                }

                if let Some((name, value)) = prop.split_once(':') {
                    let name = name.trim();
                    let value = value.trim();

                    // Handle box-shadow specially (generates multiple StyleDefs)
                    if name == "box-shadow" {
                        let shadow_defs = parse_box_shadow(value);
                        for def in shadow_defs {
                            style_defs.push(def);
                            style_count += 1;
                        }
                    // Handle gradients in background property
                    } else if (name == "background" || name == "background-image")
                        && (value.contains("linear-gradient") || value.contains("radial-gradient"))
                    {
                        let gradient_defs = parse_gradient(value);
                        for def in gradient_defs {
                            style_defs.push(def);
                            style_count += 1;
                        }
                    } else if let Some(def) = parse_property(name, value) {
                        style_defs.push(def);
                        style_count += 1;
                    }
                }
            }

            if style_count > 0 {
                // Parse selector (may be complex with combinators)
                let base_idx = parse_full_selector(selector_str, &mut selectors);

                // Update first selector with style info
                if let Some(first) = selectors.get_mut(base_idx) {
                    first.style_start = style_start;
                    first.style_count = style_count;
                }
            }
        }

        Self { selectors, style_defs }
    }
}

/// Hash function matching Metal shader (djb2)
pub fn hash_string(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    for c in s.chars() {
        let c = c.to_ascii_lowercase() as u32;
        hash = hash.wrapping_shl(5).wrapping_add(hash).wrapping_add(c);
    }
    hash
}

/// Selector type for combined selectors (tag.class, tag#id)
pub const SEL_TAG_CLASS: u32 = 6;
pub const SEL_TAG_ID: u32 = 7;

/// Parse a simple selector part (tag, class, id, or pseudo-class)
fn parse_simple_selector(s: &str) -> Selector {
    let s = s.trim();

    // Check for pseudo-class
    if let Some(pseudo_pos) = s.find(':') {
        let (base, pseudo) = s.split_at(pseudo_pos);
        let pseudo = &pseudo[1..];  // Remove leading ':'

        let mut sel = if base.is_empty() {
            Selector {
                selector_type: SEL_UNIVERSAL,
                specificity: 0,
                ..Default::default()
            }
        } else {
            parse_simple_selector(base)
        };

        // Add pseudo-class specificity
        sel.specificity += 10;
        sel.pseudo_type = match pseudo {
            "first-child" => PSEUDO_FIRST_CHILD,
            "last-child" => PSEUDO_LAST_CHILD,
            "first-of-type" => PSEUDO_FIRST_OF_TYPE,
            "last-of-type" => PSEUDO_LAST_OF_TYPE,
            "only-child" => PSEUDO_ONLY_CHILD,
            "empty" => PSEUDO_EMPTY,
            "root" => PSEUDO_ROOT,
            p if p.starts_with("nth-child") => {
                // Parse nth-child(an+b)
                if let Some(start) = p.find('(') {
                    if let Some(end) = p.find(')') {
                        let formula = &p[start + 1..end];
                        let (a, b) = parse_nth_formula(formula);
                        sel.nth_a = a;
                        sel.nth_b = b;
                    }
                }
                PSEUDO_NTH_CHILD
            }
            _ => 0,
        };

        return sel;
    }

    // Check for attribute selector
    if s.contains('[') {
        if let Some(start) = s.find('[') {
            if let Some(end) = s.find(']') {
                let base = &s[..start];
                let attr_part = &s[start + 1..end];

                let mut sel = if base.is_empty() {
                    Selector {
                        selector_type: SEL_UNIVERSAL,
                        specificity: 0,
                        ..Default::default()
                    }
                } else {
                    parse_simple_selector(base)
                };

                sel.selector_type = SEL_ATTRIBUTE;
                sel.specificity += 10;

                // Parse attribute selector
                if let Some(eq_pos) = attr_part.find('=') {
                    let attr_name = attr_part[..eq_pos].trim_end_matches(|c| c == '^' || c == '$' || c == '*' || c == '~' || c == '|');
                    let attr_value = attr_part[eq_pos + 1..].trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace());

                    sel.attr_name_hash = hash_string(attr_name);
                    sel.attr_value_hash = hash_string(attr_value);

                    // Determine operator
                    let op_char = if eq_pos > 0 { attr_part.chars().nth(eq_pos - 1) } else { None };
                    sel.attr_op = match op_char {
                        Some('^') => ATTR_STARTS,
                        Some('$') => ATTR_ENDS,
                        Some('*') => ATTR_CONTAINS,
                        Some('~') => ATTR_WORD,
                        Some('|') => ATTR_LANG,
                        _ => ATTR_EQUALS,
                    };
                } else {
                    // Just [attr] - exists check
                    sel.attr_name_hash = hash_string(attr_part.trim());
                    sel.attr_op = ATTR_EXISTS;
                }

                return sel;
            }
        }
    }

    // Check for combined tag.class selector (e.g., "p.highlight")
    if let Some(dot_pos) = s.find('.') {
        if dot_pos > 0 {
            // Has tag before the dot
            let tag = &s[..dot_pos];
            let class = &s[dot_pos + 1..];
            return Selector {
                selector_type: SEL_TAG_CLASS,
                hash: hash_string(tag),          // tag hash in main hash
                attr_name_hash: hash_string(class),  // class hash in attr_name_hash
                specificity: 11,  // 1 (tag) + 10 (class)
                ..Default::default()
            };
        }
    }

    // Check for combined tag#id selector (e.g., "div#main")
    if let Some(hash_pos) = s.find('#') {
        if hash_pos > 0 {
            // Has tag before the hash
            let tag = &s[..hash_pos];
            let id = &s[hash_pos + 1..];
            return Selector {
                selector_type: SEL_TAG_ID,
                hash: hash_string(tag),          // tag hash in main hash
                attr_name_hash: hash_string(id), // id hash in attr_name_hash
                specificity: 101,  // 1 (tag) + 100 (id)
                ..Default::default()
            };
        }
    }

    // Simple selector
    if s == "*" {
        Selector {
            selector_type: SEL_UNIVERSAL,
            specificity: 0,
            ..Default::default()
        }
    } else if s.starts_with('.') {
        let class_name = &s[1..];
        Selector {
            selector_type: SEL_CLASS,
            hash: hash_string(class_name),
            specificity: 10,
            ..Default::default()
        }
    } else if s.starts_with('#') {
        let id_name = &s[1..];
        Selector {
            selector_type: SEL_ID,
            hash: hash_string(id_name),
            specificity: 100,
            ..Default::default()
        }
    } else {
        // Tag selector
        Selector {
            selector_type: SEL_TAG,
            hash: hash_string(s),
            specificity: 1,
            ..Default::default()
        }
    }
}

/// Parse nth-child formula like "2n+1", "odd", "even", "3"
fn parse_nth_formula(formula: &str) -> (i32, i32) {
    let formula = formula.trim().to_lowercase();

    if formula == "odd" {
        return (2, 1);
    }
    if formula == "even" {
        return (2, 0);
    }

    // Parse "an+b" format
    if let Some(n_pos) = formula.find('n') {
        let a_str = &formula[..n_pos];
        let a = if a_str.is_empty() || a_str == "+" {
            1
        } else if a_str == "-" {
            -1
        } else {
            a_str.parse().unwrap_or(1)
        };

        let b_part = &formula[n_pos + 1..];
        let b = if b_part.is_empty() {
            0
        } else {
            b_part.trim_start_matches('+').parse().unwrap_or(0)
        };

        (a, b)
    } else {
        // Just a number
        (0, formula.parse().unwrap_or(0))
    }
}

/// Parse a full selector (with combinators like "div > p" or "div p")
///
/// CSS selectors are written left-to-right (ancestor to descendant), but
/// matching starts from the target element. We reverse the order so:
/// - First selector = rightmost (target element)
/// - Combinator on each selector indicates how to find the previous element
/// - Last selector = leftmost (root of the selector chain)
fn parse_full_selector(s: &str, selectors: &mut Vec<Selector>) -> usize {
    let s = s.trim();

    // Tokenize by combinators, preserving combinator info
    // parts[i].1 is the combinator BEFORE parts[i] (how parts[i-1] relates to parts[i])
    let mut parts: Vec<(String, u32)> = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut pending_combinator = COMB_NONE;

    while let Some(c) = chars.next() {
        match c {
            '>' => {
                if !current.trim().is_empty() {
                    parts.push((current.trim().to_string(), pending_combinator));
                    current = String::new();
                }
                // Skip whitespace
                while chars.peek() == Some(&' ') { chars.next(); }
                pending_combinator = COMB_CHILD;
            }
            '+' => {
                if !current.trim().is_empty() {
                    parts.push((current.trim().to_string(), pending_combinator));
                    current = String::new();
                }
                while chars.peek() == Some(&' ') { chars.next(); }
                pending_combinator = COMB_ADJACENT;
            }
            '~' => {
                if !current.trim().is_empty() {
                    parts.push((current.trim().to_string(), pending_combinator));
                    current = String::new();
                }
                while chars.peek() == Some(&' ') { chars.next(); }
                pending_combinator = COMB_SIBLING;
            }
            ' ' => {
                // Could be descendant combinator or just whitespace before another combinator
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push((trimmed, pending_combinator));
                    current = String::new();

                    // Peek to see if next non-space is a combinator
                    let mut temp_chars = chars.clone();
                    while temp_chars.peek() == Some(&' ') { temp_chars.next(); }

                    match temp_chars.peek() {
                        Some('>') | Some('+') | Some('~') => {
                            // Next is an explicit combinator, will be set when we hit it
                            pending_combinator = COMB_NONE;
                        }
                        Some(_) => {
                            // Next is another selector part - this is descendant combinator
                            pending_combinator = COMB_DESCENDANT;
                        }
                        None => {
                            pending_combinator = COMB_NONE;
                        }
                    }
                }
                // Skip whitespace
                while chars.peek() == Some(&' ') { chars.next(); }
            }
            _ => current.push(c),
        }
    }

    // Add final part
    if !current.trim().is_empty() {
        parts.push((current.trim().to_string(), pending_combinator));
    }

    if parts.is_empty() {
        return 0;
    }

    // Reverse parts so rightmost (target) is first
    // For "div p" we have [("div", NONE), ("p", DESCENDANT)]
    // After reverse: [("p", DESCENDANT), ("div", NONE)]
    // The combinator on "p" tells us how to find "div" (its ancestor)
    parts.reverse();

    // Create selector chain
    let base_idx = selectors.len();
    let mut total_specificity = 0u32;

    for (i, (part, comb)) in parts.iter().enumerate() {
        let mut sel = parse_simple_selector(part);
        sel.combinator = *comb;  // Combinator indicates how to find next element
        total_specificity += sel.specificity;

        // Link to next part
        if i + 1 < parts.len() {
            sel.next_part = (base_idx + i + 1) as i32;
        } else {
            sel.next_part = -1;
        }

        selectors.push(sel);
    }

    // Update first selector with total specificity
    if let Some(first) = selectors.get_mut(base_idx) {
        first.specificity = total_specificity;
    }

    base_idx
}

fn parse_property(name: &str, value: &str) -> Option<StyleDef> {
    let property_id = match name {
        "display" => PROP_DISPLAY,
        "width" => PROP_WIDTH,
        "height" => PROP_HEIGHT,
        "margin" => PROP_MARGIN,
        "margin-top" | "margin-right" | "margin-bottom" | "margin-left" => PROP_MARGIN,
        "padding" => PROP_PADDING,
        "padding-top" | "padding-right" | "padding-bottom" | "padding-left" => PROP_PADDING,
        "color" => PROP_COLOR,
        "background" | "background-color" => PROP_BACKGROUND,
        "font-size" => PROP_FONT_SIZE,
        "line-height" => PROP_LINE_HEIGHT,
        "font-weight" => PROP_FONT_WEIGHT,
        "text-align" => PROP_TEXT_ALIGN,
        "flex-direction" => PROP_FLEX_DIRECTION,
        "justify-content" => PROP_JUSTIFY_CONTENT,
        "align-items" => PROP_ALIGN_ITEMS,
        "flex-grow" => PROP_FLEX_GROW,
        "flex-shrink" => PROP_FLEX_SHRINK,
        "border-width" => PROP_BORDER_WIDTH,
        "border-color" => PROP_BORDER_COLOR,
        "border-radius" => PROP_BORDER_RADIUS,
        "opacity" => PROP_OPACITY,
        "position" => PROP_POSITION,
        "top" => PROP_TOP,
        "right" => PROP_RIGHT,
        "bottom" => PROP_BOTTOM,
        "left" => PROP_LEFT,
        "z-index" => PROP_Z_INDEX,
        "overflow" => PROP_OVERFLOW,
        "overflow-x" => PROP_OVERFLOW_X,
        "overflow-y" => PROP_OVERFLOW_Y,
        "border-collapse" => PROP_BORDER_COLLAPSE,
        "border-spacing" => PROP_BORDER_SPACING,
        _ => return None,
    };

    let values = parse_value(property_id, value)?;

    Some(StyleDef { property_id, values })
}

fn parse_value(property_id: u32, value: &str) -> Option<[f32; 4]> {
    match property_id {
        PROP_DISPLAY => {
            let v = match value {
                "none" => DISPLAY_NONE as f32,
                "block" => DISPLAY_BLOCK as f32,
                "inline" => DISPLAY_INLINE as f32,
                "flex" => DISPLAY_FLEX as f32,
                "inline-block" => DISPLAY_INLINE_BLOCK as f32,
                "table" => DISPLAY_TABLE as f32,
                "table-row" => DISPLAY_TABLE_ROW as f32,
                "table-cell" => DISPLAY_TABLE_CELL as f32,
                _ => DISPLAY_BLOCK as f32,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_WIDTH | PROP_HEIGHT | PROP_FONT_SIZE | PROP_LINE_HEIGHT |
        PROP_BORDER_RADIUS | PROP_OPACITY | PROP_FLEX_GROW | PROP_FLEX_SHRINK => {
            let v = parse_length(value)?;
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_FONT_WEIGHT => {
            let v = match value {
                "normal" => 400.0,
                "bold" => 700.0,
                _ => value.parse::<f32>().ok()?,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_TEXT_ALIGN => {
            let v = match value {
                "left" => TEXT_LEFT as f32,
                "center" => TEXT_CENTER as f32,
                "right" => TEXT_RIGHT as f32,
                _ => TEXT_LEFT as f32,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_FLEX_DIRECTION => {
            let v = match value {
                "row" => FLEX_ROW as f32,
                "column" => FLEX_COLUMN as f32,
                _ => FLEX_ROW as f32,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_JUSTIFY_CONTENT => {
            let v = match value {
                "flex-start" | "start" => JUSTIFY_START as f32,
                "center" => JUSTIFY_CENTER as f32,
                "flex-end" | "end" => JUSTIFY_END as f32,
                "space-between" => JUSTIFY_SPACE_BETWEEN as f32,
                "space-around" => JUSTIFY_SPACE_AROUND as f32,
                _ => JUSTIFY_START as f32,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_ALIGN_ITEMS => {
            let v = match value {
                "flex-start" | "start" => ALIGN_START as f32,
                "center" => ALIGN_CENTER as f32,
                "flex-end" | "end" => ALIGN_END as f32,
                "stretch" => ALIGN_STRETCH as f32,
                _ => ALIGN_STRETCH as f32,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_MARGIN | PROP_PADDING | PROP_BORDER_WIDTH => {
            parse_box_values(value)
        }
        PROP_COLOR | PROP_BACKGROUND | PROP_BORDER_COLOR => {
            parse_color(value)
        }
        PROP_POSITION => {
            let v = match value {
                "static" => POSITION_STATIC as f32,
                "relative" => POSITION_RELATIVE as f32,
                "absolute" => POSITION_ABSOLUTE as f32,
                "fixed" => POSITION_FIXED as f32,
                _ => POSITION_STATIC as f32,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_TOP | PROP_RIGHT | PROP_BOTTOM | PROP_LEFT => {
            let v = if value == "auto" {
                OFFSET_AUTO
            } else {
                parse_length_or_percent(value)?
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_Z_INDEX => {
            let v: i32 = value.trim().parse().ok()?;
            Some([v as f32, 0.0, 0.0, 0.0])
        }
        PROP_OVERFLOW => {
            let v = parse_overflow_value(value)?;
            // Shorthand sets both x and y
            Some([v as f32, v as f32, 0.0, 0.0])
        }
        PROP_OVERFLOW_X => {
            let v = parse_overflow_value(value)?;
            Some([v as f32, 0.0, 0.0, 0.0])
        }
        PROP_OVERFLOW_Y => {
            let v = parse_overflow_value(value)?;
            Some([v as f32, 0.0, 0.0, 0.0])
        }
        PROP_BORDER_COLLAPSE => {
            let v = match value.trim() {
                "collapse" => 1.0,
                "separate" => 0.0,
                _ => 0.0,
            };
            Some([v, 0.0, 0.0, 0.0])
        }
        PROP_BORDER_SPACING => {
            let v = parse_length(value)?;
            Some([v, 0.0, 0.0, 0.0])
        }
        _ => None,
    }
}

fn parse_overflow_value(value: &str) -> Option<u32> {
    match value.trim() {
        "visible" => Some(OVERFLOW_VISIBLE),
        "hidden" => Some(OVERFLOW_HIDDEN),
        "scroll" => Some(OVERFLOW_SCROLL),
        "auto" => Some(OVERFLOW_AUTO),
        "clip" => Some(OVERFLOW_HIDDEN), // clip behaves like hidden
        _ => None,
    }
}

/// Parse box-shadow CSS property
/// Returns multiple StyleDefs for shadow geometry, color, and inset flag
fn parse_box_shadow(value: &str) -> Vec<StyleDef> {
    let value = value.trim();

    // Handle "none"
    if value == "none" {
        return vec![];  // No shadows
    }

    let mut defs = Vec::new();

    // Split multiple shadows by comma (handle nested parens for rgb/rgba)
    let shadows: Vec<&str> = split_shadows(value);

    for (idx, shadow) in shadows.iter().enumerate() {
        if idx >= 4 {
            break;  // Max 4 shadows
        }

        if let Some((geom, color, inset)) = parse_single_shadow(shadow.trim()) {
            // Geometry: offset_x, offset_y, blur, spread
            defs.push(StyleDef {
                property_id: PROP_BOX_SHADOW,
                values: [
                    geom.0 + (idx as f32 * 1000.0),  // Encode shadow index in offset
                    geom.1,
                    geom.2,
                    geom.3,
                ],
            });

            // Color: RGBA
            defs.push(StyleDef {
                property_id: PROP_BOX_SHADOW_COLOR,
                values: [
                    color.0 + (idx as f32 * 10.0),  // Encode shadow index
                    color.1,
                    color.2,
                    color.3,
                ],
            });

            // Inset flag
            if inset {
                defs.push(StyleDef {
                    property_id: PROP_BOX_SHADOW_INSET,
                    values: [idx as f32, 1.0, 0.0, 0.0],
                });
            }
        }
    }

    defs
}

/// Split shadow definitions by comma, handling nested parentheses
fn split_shadows(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0;

    for (i, c) in value.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                result.push(&value[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < value.len() {
        result.push(&value[start..]);
    }

    result
}

/// Parse a single box-shadow value
/// Returns (offset_x, offset_y, blur, spread), (r, g, b, a), inset
fn parse_single_shadow(value: &str) -> Option<((f32, f32, f32, f32), (f32, f32, f32, f32), bool)> {
    let mut inset = false;
    let mut lengths: Vec<f32> = Vec::new();
    let mut color = (0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32);  // Default: black

    // Tokenize the shadow value
    let mut chars = value.chars().peekable();
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            chars.next();
        } else if c == '(' {
            // Collect function call (rgb, rgba, etc.)
            current.push(chars.next().unwrap());
            while let Some(&c2) = chars.peek() {
                current.push(chars.next().unwrap());
                if c2 == ')' {
                    break;
                }
            }
        } else {
            current.push(chars.next().unwrap());
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    for token in &tokens {
        let token = token.trim();

        if token == "inset" {
            inset = true;
        } else if token.starts_with("rgb") || token.starts_with("rgba") {
            color = parse_css_color_fn(token);
        } else if token.starts_with('#') {
            color = parse_hex_color(token);
        } else if let Some(len) = parse_shadow_length(token) {
            lengths.push(len);
        } else {
            // Try named color
            if let Some(c) = parse_named_color(token) {
                color = c;
            }
        }
    }

    if lengths.len() < 2 {
        return None;  // Need at least offset-x and offset-y
    }

    let offset_x = lengths[0];
    let offset_y = lengths[1];
    let blur = lengths.get(2).copied().unwrap_or(0.0);
    let spread = lengths.get(3).copied().unwrap_or(0.0);

    Some(((offset_x, offset_y, blur, spread), color, inset))
}

fn parse_shadow_length(value: &str) -> Option<f32> {
    let value = value.trim();
    if value == "0" {
        return Some(0.0);
    }

    let num_str = value
        .trim_end_matches("px")
        .trim_end_matches("em")
        .trim_end_matches("rem");

    num_str.parse::<f32>().ok()
}

fn parse_css_color_fn(value: &str) -> (f32, f32, f32, f32) {
    // Parse rgb(r, g, b) or rgba(r, g, b, a)
    let inner = value
        .trim_start_matches("rgba(")
        .trim_start_matches("rgb(")
        .trim_end_matches(')');

    let parts: Vec<&str> = inner.split(',').collect();

    let r = parts.get(0).and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.0) / 255.0;
    let g = parts.get(1).and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.0) / 255.0;
    let b = parts.get(2).and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.0) / 255.0;
    let a = parts.get(3).and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(1.0);

    (r, g, b, a)
}

fn parse_hex_color(value: &str) -> (f32, f32, f32, f32) {
    let hex = value.trim_start_matches('#');

    if hex.len() == 3 {
        // #RGB
        let r = u8::from_str_radix(&hex[0..1], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[1..2], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[2..3], 16).unwrap_or(0);
        ((r * 17) as f32 / 255.0, (g * 17) as f32 / 255.0, (b * 17) as f32 / 255.0, 1.0)
    } else if hex.len() == 6 {
        // #RRGGBB
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    }
}

fn parse_named_color(name: &str) -> Option<(f32, f32, f32, f32)> {
    match name.to_lowercase().as_str() {
        "black" => Some((0.0, 0.0, 0.0, 1.0)),
        "white" => Some((1.0, 1.0, 1.0, 1.0)),
        "red" => Some((1.0, 0.0, 0.0, 1.0)),
        "green" => Some((0.0, 0.5, 0.0, 1.0)),
        "blue" => Some((0.0, 0.0, 1.0, 1.0)),
        "yellow" => Some((1.0, 1.0, 0.0, 1.0)),
        "cyan" => Some((0.0, 1.0, 1.0, 1.0)),
        "magenta" => Some((1.0, 0.0, 1.0, 1.0)),
        "orange" => Some((1.0, 0.65, 0.0, 1.0)),
        "purple" => Some((0.5, 0.0, 0.5, 1.0)),
        "pink" => Some((1.0, 0.75, 0.8, 1.0)),
        "gray" | "grey" => Some((0.5, 0.5, 0.5, 1.0)),
        "transparent" => Some((0.0, 0.0, 0.0, 0.0)),
        _ => None,
    }
}

/// Parse CSS gradient (linear-gradient or radial-gradient)
fn parse_gradient(value: &str) -> Vec<StyleDef> {
    let mut defs = Vec::new();
    let value = value.trim();

    // Detect gradient type
    let (gradient_type, inner) = if value.starts_with("linear-gradient(") {
        (GRADIENT_LINEAR, value.trim_start_matches("linear-gradient(").trim_end_matches(')'))
    } else if value.starts_with("radial-gradient(") {
        (GRADIENT_RADIAL, value.trim_start_matches("radial-gradient(").trim_end_matches(')'))
    } else {
        return defs;  // Not a gradient
    };

    // Add gradient type
    defs.push(StyleDef {
        property_id: PROP_GRADIENT_TYPE,
        values: [gradient_type as f32, 0.0, 0.0, 0.0],
    });

    // Parse gradient content
    let parts: Vec<&str> = split_gradient_parts(inner);

    let mut angle = 180.0_f32;  // Default: to bottom
    let mut color_stops: Vec<((f32, f32, f32, f32), f32)> = Vec::new();
    let mut stop_idx = 0;

    for part in parts {
        let part = part.trim();

        // Check for direction keywords or angle
        if part.starts_with("to ") {
            angle = parse_gradient_direction(part);
        } else if part.ends_with("deg") {
            if let Ok(a) = part.trim_end_matches("deg").parse::<f32>() {
                angle = a;
            }
        } else {
            // Parse color stop
            if let Some((color, pos)) = parse_color_stop(part) {
                color_stops.push((color, pos));
            }
        }
    }

    // Add angle
    defs.push(StyleDef {
        property_id: PROP_GRADIENT_ANGLE,
        values: [angle, 0.0, 0.0, 0.0],
    });

    // Auto-calculate positions if not specified
    let stop_count = color_stops.len();
    for (idx, (color, mut pos)) in color_stops.into_iter().enumerate() {
        if pos < 0.0 {
            // Position not specified, calculate automatically
            pos = if stop_count == 1 {
                0.5
            } else {
                idx as f32 / (stop_count - 1) as f32
            };
        }

        // Encode: idx in values[0], pos in values[1], encode color in values[2] and values[3]
        // Pack RGBA into two floats: values[2] = R + G*256, values[3] = B + A*256 (scaled)
        defs.push(StyleDef {
            property_id: PROP_GRADIENT_STOP,
            values: [
                idx as f32,
                pos,
                color.0 + color.1 * 256.0,  // Encode R, G
                color.2 + color.3 * 256.0,  // Encode B, A
            ],
        });
        stop_idx += 1;

        if stop_idx >= 8 {
            break;  // Max 8 stops
        }
    }

    defs
}

/// Split gradient parts by comma, handling nested parentheses
fn split_gradient_parts(value: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0;

    for (i, c) in value.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                result.push(&value[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < value.len() {
        result.push(&value[start..]);
    }

    result
}

/// Parse gradient direction keyword
fn parse_gradient_direction(dir: &str) -> f32 {
    match dir.trim() {
        "to top" => 0.0,
        "to right" => 90.0,
        "to bottom" => 180.0,
        "to left" => 270.0,
        "to top right" | "to right top" => 45.0,
        "to bottom right" | "to right bottom" => 135.0,
        "to bottom left" | "to left bottom" => 225.0,
        "to top left" | "to left top" => 315.0,
        _ => 180.0,  // Default: to bottom
    }
}

/// Parse a color stop (color and optional position)
fn parse_color_stop(value: &str) -> Option<((f32, f32, f32, f32), f32)> {
    let value = value.trim();
    let mut color = (0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32);
    let mut position = -1.0_f32;  // -1 means auto

    // Try to split into color and position
    // Color can be: named, #hex, rgb(), rgba()
    if value.starts_with("rgb") {
        // Find the closing paren
        if let Some(end) = value.find(')') {
            color = parse_css_color_fn(&value[..=end]);
            let rest = value[end + 1..].trim();
            if !rest.is_empty() {
                position = parse_percentage_or_length(rest);
            }
        }
    } else if value.starts_with('#') {
        // Hex color, possibly followed by position
        let parts: Vec<&str> = value.split_whitespace().collect();
        color = parse_hex_color(parts[0]);
        if parts.len() > 1 {
            position = parse_percentage_or_length(parts[1]);
        }
    } else {
        // Named color, possibly followed by position
        let parts: Vec<&str> = value.split_whitespace().collect();
        if let Some(c) = parse_named_color(parts[0]) {
            color = c;
            if parts.len() > 1 {
                position = parse_percentage_or_length(parts[1]);
            }
        } else {
            return None;
        }
    }

    Some((color, position))
}

/// Parse percentage (returns 0.0-1.0) or length
fn parse_percentage_or_length(value: &str) -> f32 {
    let value = value.trim();
    if value.ends_with('%') {
        value.trim_end_matches('%').parse::<f32>().unwrap_or(0.0) / 100.0
    } else if value.ends_with("px") {
        // For now, treat px as percentage of some default (this is a simplification)
        value.trim_end_matches("px").parse::<f32>().unwrap_or(0.0) / 100.0
    } else {
        value.parse::<f32>().unwrap_or(0.0)
    }
}

fn parse_length(value: &str) -> Option<f32> {
    let value = value.trim();
    if value == "auto" || value == "0" {
        return Some(0.0);
    }

    // Strip units
    let num_str = value
        .trim_end_matches("px")
        .trim_end_matches("em")
        .trim_end_matches("rem")
        .trim_end_matches("%");

    num_str.parse::<f32>().ok()
}

/// Parse length value, returning percentage as negative value
/// (for layout engine to interpret as percentage of containing block)
fn parse_length_or_percent(value: &str) -> Option<f32> {
    let value = value.trim();
    if value == "auto" {
        return Some(OFFSET_AUTO);
    }
    if value == "0" {
        return Some(0.0);
    }

    // Check for percentage
    if value.ends_with('%') {
        let num_str = value.trim_end_matches('%');
        // Store percentage as negative value (convention for layout engine)
        // -10.0 means 10%
        return num_str.parse::<f32>().ok().map(|v| -v);
    }

    // Strip px/em/rem units
    let num_str = value
        .trim_end_matches("px")
        .trim_end_matches("em")
        .trim_end_matches("rem");

    num_str.parse::<f32>().ok()
}

fn parse_box_values(value: &str) -> Option<[f32; 4]> {
    let parts: Vec<f32> = value
        .split_whitespace()
        .filter_map(|p| parse_length(p))
        .collect();

    match parts.len() {
        1 => Some([parts[0], parts[0], parts[0], parts[0]]),
        2 => Some([parts[0], parts[1], parts[0], parts[1]]),
        3 => Some([parts[0], parts[1], parts[2], parts[1]]),
        4 => Some([parts[0], parts[1], parts[2], parts[3]]),
        _ => None,
    }
}

fn parse_color(value: &str) -> Option<[f32; 4]> {
    let value = value.trim().to_lowercase();

    // Named colors
    match value.as_str() {
        "red" => return Some([1.0, 0.0, 0.0, 1.0]),
        "green" => return Some([0.0, 0.5, 0.0, 1.0]),
        "blue" => return Some([0.0, 0.0, 1.0, 1.0]),
        "white" => return Some([1.0, 1.0, 1.0, 1.0]),
        "black" => return Some([0.0, 0.0, 0.0, 1.0]),
        "yellow" => return Some([1.0, 1.0, 0.0, 1.0]),
        "cyan" => return Some([0.0, 1.0, 1.0, 1.0]),
        "magenta" => return Some([1.0, 0.0, 1.0, 1.0]),
        "gray" | "grey" => return Some([0.5, 0.5, 0.5, 1.0]),
        "orange" => return Some([1.0, 0.65, 0.0, 1.0]),
        "purple" => return Some([0.5, 0.0, 0.5, 1.0]),
        "pink" => return Some([1.0, 0.75, 0.8, 1.0]),
        "transparent" => return Some([0.0, 0.0, 0.0, 0.0]),
        _ => {}
    }

    // Hex colors
    if value.starts_with('#') {
        let hex = &value[1..];
        let (r, g, b, a) = match hex.len() {
            3 => {
                let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
                let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
                let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
                (r, g, b, 255)
            }
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                (r, g, b, 255)
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
                (r, g, b, a)
            }
            _ => return None,
        };
        return Some([
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        ]);
    }

    // rgb() / rgba()
    if value.starts_with("rgb") {
        let inner = value
            .trim_start_matches("rgba(")
            .trim_start_matches("rgb(")
            .trim_end_matches(')');
        let parts: Vec<f32> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();

        return match parts.len() {
            3 => Some([parts[0] / 255.0, parts[1] / 255.0, parts[2] / 255.0, 1.0]),
            4 => Some([parts[0] / 255.0, parts[1] / 255.0, parts[2] / 255.0, parts[3]]),
            _ => None,
        };
    }

    None
}

// Note: Inline style extraction and parsing now happens entirely on GPU
// in the Metal shader (parse_inline_style_gpu function in style.metal)

/// Metal shader source for the style resolver
const STYLE_SHADER: &str = include_str!("style.metal");

/// GPU-Native CSS Style Resolver
pub struct GpuStyler {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    resolve_pipeline: ComputePipelineState,
    #[allow(dead_code)]
    inherit_pipeline: ComputePipelineState,  // TODO: Re-enable when proper inheritance is implemented

    // Input buffers (from parser)
    element_buffer: Buffer,
    html_buffer: Buffer,
    token_buffer: Buffer,

    // Style buffers
    selector_buffer: Buffer,
    style_def_buffer: Buffer,

    // Output buffer
    computed_buffer: Buffer,

    // Count buffers
    element_count_buffer: Buffer,
    selector_count_buffer: Buffer,
    // Note: Inline style parsing happens entirely on GPU in the Metal shader
}

impl GpuStyler {
    /// Create a new GPU style resolver
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(STYLE_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile style shader: {}", e))?;

        // Create pipelines
        let resolve_fn = library
            .get_function("resolve_styles", None)
            .map_err(|e| format!("Failed to find resolve_styles: {}", e))?;
        let inherit_fn = library
            .get_function("apply_inheritance", None)
            .map_err(|e| format!("Failed to find apply_inheritance: {}", e))?;

        let resolve_pipeline = device
            .new_compute_pipeline_state_with_function(&resolve_fn)
            .map_err(|e| format!("Failed to create resolve pipeline: {}", e))?;
        let inherit_pipeline = device
            .new_compute_pipeline_state_with_function(&inherit_fn)
            .map_err(|e| format!("Failed to create inherit pipeline: {}", e))?;

        // Allocate buffers
        let max_elements = 16384usize;
        let max_tokens = 65536usize;
        let max_html = 1024 * 1024usize;

        let element_buffer = device.new_buffer(
            (max_elements * std::mem::size_of::<Element>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let html_buffer = device.new_buffer(
            max_html as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let token_buffer = device.new_buffer(
            (max_tokens * std::mem::size_of::<Token>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let selector_buffer = device.new_buffer(
            (MAX_SELECTORS * std::mem::size_of::<Selector>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let style_def_buffer = device.new_buffer(
            (MAX_STYLE_DEFS * std::mem::size_of::<StyleDef>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let computed_buffer = device.new_buffer(
            (max_elements * std::mem::size_of::<ComputedStyle>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let element_count_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let selector_count_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        // Note: Inline style parsing happens entirely on GPU in the Metal shader

        Ok(Self {
            device: device.clone(),
            command_queue,
            resolve_pipeline,
            inherit_pipeline,
            element_buffer,
            html_buffer,
            token_buffer,
            selector_buffer,
            style_def_buffer,
            computed_buffer,
            element_count_buffer,
            selector_count_buffer,
        })
    }

    /// Resolve styles for elements
    pub fn resolve_styles(
        &mut self,
        elements: &[Element],
        tokens: &[Token],
        html: &[u8],
        stylesheet: &Stylesheet,
    ) -> Vec<ComputedStyle> {
        if elements.is_empty() {
            return Vec::new();
        }

        let element_count = elements.len().min(16384);
        let token_count = tokens.len().min(65536);
        let html_len = html.len().min(1024 * 1024);
        let selector_count = stylesheet.selectors.len().min(MAX_SELECTORS);
        let style_def_count = stylesheet.style_defs.len().min(MAX_STYLE_DEFS);

        // Copy elements
        unsafe {
            std::ptr::copy_nonoverlapping(
                elements.as_ptr(),
                self.element_buffer.contents() as *mut Element,
                element_count,
            );
        }

        // Copy tokens
        unsafe {
            std::ptr::copy_nonoverlapping(
                tokens.as_ptr(),
                self.token_buffer.contents() as *mut Token,
                token_count,
            );
        }

        // Copy HTML
        unsafe {
            std::ptr::copy_nonoverlapping(
                html.as_ptr(),
                self.html_buffer.contents() as *mut u8,
                html_len,
            );
        }

        // Copy selectors
        unsafe {
            std::ptr::copy_nonoverlapping(
                stylesheet.selectors.as_ptr(),
                self.selector_buffer.contents() as *mut Selector,
                selector_count,
            );
        }

        // Copy style defs
        unsafe {
            std::ptr::copy_nonoverlapping(
                stylesheet.style_defs.as_ptr(),
                self.style_def_buffer.contents() as *mut StyleDef,
                style_def_count,
            );
        }

        // Set counts
        unsafe {
            *(self.element_count_buffer.contents() as *mut u32) = element_count as u32;
            *(self.selector_count_buffer.contents() as *mut u32) = selector_count as u32;
        }

        // Initialize computed styles to default
        // Note: Inline style parsing happens entirely on GPU in the Metal shader
        unsafe {
            let ptr = self.computed_buffer.contents() as *mut ComputedStyle;
            for i in 0..element_count {
                *ptr.add(i) = ComputedStyle::default();
            }
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1: Resolve styles
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.resolve_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.html_buffer), 0);
            encoder.set_buffer(2, Some(&self.token_buffer), 0);
            encoder.set_buffer(3, Some(&self.selector_buffer), 0);
            encoder.set_buffer(4, Some(&self.style_def_buffer), 0);
            encoder.set_buffer(5, Some(&self.computed_buffer), 0);
            encoder.set_buffer(6, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(7, Some(&self.selector_count_buffer), 0);
            // Inline styles are now parsed directly on GPU in the Metal shader

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 2: Apply inheritance
        // Uses properties_set bitmask to only inherit properties that weren't explicitly set
        // Run multiple passes to handle deep nesting (grandparent -> parent -> child)
        {
            // For proper inheritance in deeply nested trees, we run multiple passes
            // Each pass propagates inheritance one level deeper
            // We use log2(max_depth) passes which handles trees up to 2^passes deep
            const MAX_INHERITANCE_PASSES: u64 = 8;  // Handles trees up to 256 levels deep

            for _ in 0..MAX_INHERITANCE_PASSES {
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.inherit_pipeline);
                encoder.set_buffer(0, Some(&self.element_buffer), 0);
                encoder.set_buffer(1, Some(&self.computed_buffer), 0);
                encoder.set_buffer(2, Some(&self.element_count_buffer), 0);

                let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(THREAD_COUNT, 1, 1),
                );
                encoder.end_encoding();
            }
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let computed_ptr = self.computed_buffer.contents() as *const ComputedStyle;
        (0..element_count)
            .map(|i| unsafe { *computed_ptr.add(i) })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tokenizer::GpuTokenizer;
    use super::super::parser::{GpuParser, ELEM_P, ELEM_DIV};

    fn setup() -> (GpuTokenizer, GpuParser, GpuStyler) {
        let device = Device::system_default().expect("No Metal device");
        let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
        let parser = GpuParser::new(&device).expect("Failed to create parser");
        let styler = GpuStyler::new(&device).expect("Failed to create styler");
        (tokenizer, parser, styler)
    }

    #[test]
    fn test_tag_selector() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div><p>text</p></div>";
        let css = "p { color: red; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        // Find p element's style
        let p_idx = elements.iter().position(|e| e.element_type == ELEM_P).unwrap();
        let p_style = &styles[p_idx];

        // Red = [1.0, 0.0, 0.0, 1.0]
        assert!((p_style.color[0] - 1.0).abs() < 0.01, "Expected red, got {:?}", p_style.color);
        assert!((p_style.color[1] - 0.0).abs() < 0.01);
        assert!((p_style.color[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_class_selector() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div class=\"highlight\">text</div>";
        let css = ".highlight { background: yellow; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        let div_style = &styles[0];
        // Yellow = [1.0, 1.0, 0.0, 1.0]
        assert!((div_style.background_color[0] - 1.0).abs() < 0.01, "Expected yellow");
        assert!((div_style.background_color[1] - 1.0).abs() < 0.01);
        assert!((div_style.background_color[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_id_selector() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div id=\"main\">text</div>";
        let css = "#main { background: blue; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        let div_style = &styles[0];
        // Blue = [0.0, 0.0, 1.0, 1.0]
        assert!((div_style.background_color[2] - 1.0).abs() < 0.01, "Expected blue");
    }

    #[test]
    fn test_specificity() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<p class=\"special\">text</p>";
        let css = "p { color: blue; } .special { color: red; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        // Class selector has higher specificity, so red wins
        let p_style = &styles[0];
        assert!((p_style.color[0] - 1.0).abs() < 0.01, "Expected red (class wins), got {:?}", p_style.color);
    }

    #[test]
    fn test_cascade_order() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<p>text</p>";
        let css = "p { margin: 10px; } p { margin: 20px; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        // Later rule wins (same specificity)
        let p_style = &styles[0];
        assert!((p_style.margin[0] - 20.0).abs() < 0.01, "Expected 20px margin");
    }

    #[test]
    fn test_box_model() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div>text</div>";
        let css = "div { margin: 10px 20px; padding: 5px; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        let div_style = &styles[0];
        // margin: 10px 20px = top/bottom 10, left/right 20
        assert!((div_style.margin[0] - 10.0).abs() < 0.01);
        assert!((div_style.margin[1] - 20.0).abs() < 0.01);
        assert!((div_style.margin[2] - 10.0).abs() < 0.01);
        assert!((div_style.margin[3] - 20.0).abs() < 0.01);

        // padding: 5px = all sides 5
        assert!((div_style.padding[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_flexbox() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div>text</div>";
        let css = "div { display: flex; flex-direction: column; justify-content: center; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        let div_style = &styles[0];
        assert_eq!(div_style.display, DISPLAY_FLEX);
        assert_eq!(div_style.flex_direction, FLEX_COLUMN);
        assert_eq!(div_style.justify_content, JUSTIFY_CENTER);
    }

    #[test]
    fn test_hex_color() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div>text</div>";
        let css = "div { color: #ff5500; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        let div_style = &styles[0];
        assert!((div_style.color[0] - 1.0).abs() < 0.01);      // ff = 255 = 1.0
        assert!((div_style.color[1] - 0.333).abs() < 0.02);    // 55 = 85 ≈ 0.333
        assert!((div_style.color[2] - 0.0).abs() < 0.01);      // 00 = 0 = 0.0
    }

    #[test]
    fn test_multiple_classes() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = b"<div class=\"foo bar\">text</div>";
        let css = ".foo { margin: 10px; } .bar { padding: 5px; }";

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        let div_style = &styles[0];
        // Both classes should match
        assert!((div_style.margin[0] - 10.0).abs() < 0.01);
        assert!((div_style.padding[0] - 5.0).abs() < 0.01);
    }

    fn generate_html_with_classes(count: usize) -> Vec<u8> {
        let mut html = Vec::new();
        html.extend_from_slice(b"<div>");
        for i in 0..count {
            html.extend_from_slice(format!("<p class=\"item item-{}\">Text {}</p>", i % 10, i).as_bytes());
        }
        html.extend_from_slice(b"</div>");
        html
    }

    fn generate_css(selector_count: usize) -> String {
        let mut css = String::new();
        for i in 0..selector_count {
            css.push_str(&format!(".item-{} {{ margin: {}px; }}\n", i % 10, i));
        }
        css
    }

    #[test]
    fn test_performance_1k_elements_10_selectors() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = generate_html_with_classes(500);
        let css = generate_css(10);

        let tokens = tokenizer.tokenize(&html);
        let (elements, _) = parser.parse(&tokens, &html);
        let stylesheet = Stylesheet::parse(&css);

        // Warmup
        let _ = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);

        let start = std::time::Instant::now();
        let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
        let elapsed = start.elapsed();

        println!(
            "~1K elements × 10 selectors: {} styles in {:?}",
            styles.len(),
            elapsed
        );
        assert!(
            elapsed.as_millis() < 10,
            "Should style ~1K elements in <10ms, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_performance_5k_elements_50_selectors() {
        let (mut tokenizer, mut parser, mut styler) = setup();
        let html = generate_html_with_classes(2500);
        let css = generate_css(50);

        let tokens = tokenizer.tokenize(&html);
        let (elements, _) = parser.parse(&tokens, &html);
        let stylesheet = Stylesheet::parse(&css);

        // Warmup
        let _ = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);

        let start = std::time::Instant::now();
        let styles = styler.resolve_styles(&elements, &tokens, &html, &stylesheet);
        let elapsed = start.elapsed();

        println!(
            "~5K elements × 50 selectors: {} styles in {:?}",
            styles.len(),
            elapsed
        );
        assert!(
            elapsed.as_millis() < 20,
            "Should style ~5K elements in <20ms, took {:?}",
            elapsed
        );
    }
}
