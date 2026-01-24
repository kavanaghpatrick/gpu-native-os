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

// Display values
pub const DISPLAY_NONE: u32 = 0;
pub const DISPLAY_BLOCK: u32 = 1;
pub const DISPLAY_INLINE: u32 = 2;
pub const DISPLAY_FLEX: u32 = 3;
pub const DISPLAY_INLINE_BLOCK: u32 = 4;

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

// Buffer sizes
pub const MAX_SELECTORS: usize = 1024;
pub const MAX_STYLE_DEFS: usize = 4096;
const THREAD_COUNT: u64 = 1024;

/// A CSS selector
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Selector {
    pub selector_type: u32,
    pub hash: u32,
    pub specificity: u32,
    pub style_start: u32,
    pub style_count: u32,
    pub _padding: [u32; 3],
}

/// A single CSS property definition
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StyleDef {
    pub property_id: u32,
    pub values: [f32; 4],
}

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
    pub _padding: [f32; 2],
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

            // Parse selector type and hash
            let (selector_type, hash, specificity) = parse_selector(selector_str);

            // Parse style definitions
            let style_start = style_defs.len() as u32;
            let mut style_count = 0u32;

            for prop in properties_str.split(';') {
                let prop = prop.trim();
                if prop.is_empty() {
                    continue;
                }

                if let Some((name, value)) = prop.split_once(':') {
                    if let Some(def) = parse_property(name.trim(), value.trim()) {
                        style_defs.push(def);
                        style_count += 1;
                    }
                }
            }

            if style_count > 0 {
                selectors.push(Selector {
                    selector_type,
                    hash,
                    specificity,
                    style_start,
                    style_count,
                    _padding: [0; 3],
                });
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

fn parse_selector(s: &str) -> (u32, u32, u32) {
    let s = s.trim();
    if s == "*" {
        (SEL_UNIVERSAL, 0, 0)
    } else if s.starts_with('.') {
        let class_name = &s[1..];
        (SEL_CLASS, hash_string(class_name), 10)
    } else if s.starts_with('#') {
        let id_name = &s[1..];
        (SEL_ID, hash_string(id_name), 100)
    } else {
        // Tag selector
        (SEL_TAG, hash_string(s), 1)
    }
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
        _ => None,
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

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 2: Apply inheritance (disabled for now)
        // TODO: Implement proper CSS inheritance that only inherits properties
        // that weren't explicitly set. Current implementation overwrites explicitly
        // set values which breaks selector-based styling.
        // {
        //     let encoder = command_buffer.new_compute_command_encoder();
        //     encoder.set_compute_pipeline_state(&self.inherit_pipeline);
        //     encoder.set_buffer(0, Some(&self.element_buffer), 0);
        //     encoder.set_buffer(1, Some(&self.computed_buffer), 0);
        //     encoder.set_buffer(2, Some(&self.element_count_buffer), 0);
        //
        //     let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
        //     encoder.dispatch_thread_groups(
        //         MTLSize::new(threadgroups, 1, 1),
        //         MTLSize::new(THREAD_COUNT, 1, 1),
        //     );
        //     encoder.end_encoding();
        // }

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
