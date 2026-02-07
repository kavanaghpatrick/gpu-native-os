// GPU-Native Document Browser
//
// Implements the full HTML rendering pipeline as a GpuApp:
// - All computation happens on GPU
// - Single command buffer per frame
// - No CPU involvement except initial HTML load and frame submission

use super::app::{GpuApp, PipelineMode, APP_SHADER_HEADER, SLOT_APP_START, Text, colors};
use super::memory::{FrameState, InputEvent, InputEventType};
use super::vsync::FrameTiming;
use metal::*;

// ============================================================================
// Buffer Slots (starting from SLOT_APP_START = 3)
// ============================================================================

#[allow(dead_code)]
const SLOT_HTML: u64 = SLOT_APP_START;           // Raw HTML bytes
#[allow(dead_code)]
const SLOT_TOKENS: u64 = SLOT_APP_START + 1;     // Tokenizer output
#[allow(dead_code)]
const SLOT_ELEMENTS: u64 = SLOT_APP_START + 2;   // Parser output (DOM tree)
#[allow(dead_code)]
const SLOT_STYLES: u64 = SLOT_APP_START + 3;     // Computed styles
#[allow(dead_code)]
const SLOT_LAYOUT: u64 = SLOT_APP_START + 4;     // Layout boxes
#[allow(dead_code)]
const SLOT_VERTICES: u64 = SLOT_APP_START + 5;   // Paint output (render vertices)
#[allow(dead_code)]
const SLOT_COUNTS: u64 = SLOT_APP_START + 6;     // Counts: token_count, element_count, vertex_count
#[allow(dead_code)]
const SLOT_STYLESHEET: u64 = SLOT_APP_START + 7; // CSS rules (selectors + style defs)

// ============================================================================
// Limits
// ============================================================================

const MAX_HTML_SIZE: usize = 1024 * 1024;    // 1MB HTML
const MAX_TOKENS: usize = 65536;              // 64K tokens
const MAX_ELEMENTS: usize = 16384;            // 16K elements
const MAX_VERTICES: usize = 65536 * 6;        // 64K quads (6 verts each)
const MAX_SELECTORS: usize = 1024;
const MAX_STYLE_DEFS: usize = 4096;

// ============================================================================
// GPU Structures (must match Metal shader)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DocumentParams {
    pub viewport_width: f32,
    pub viewport_height: f32,
    pub scroll_x: f32,
    pub scroll_y: f32,
    pub html_length: u32,
    pub selector_count: u32,
    pub _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DocumentCounts {
    pub token_count: u32,
    pub element_count: u32,
    pub vertex_count: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PaintVertex {
    pub position: [f32; 2],
    pub tex_coord: [f32; 2],
    pub color: [f32; 4],
    pub flags: u32,
    pub _padding: [u32; 3],
}

// ============================================================================
// DocumentApp
// ============================================================================

pub struct DocumentApp {
    name: String,

    // Pipelines
    compute_pipeline: ComputePipelineState,
    render_pipeline: RenderPipelineState,

    // Buffers
    html_buffer: Buffer,
    tokens_buffer: Buffer,
    elements_buffer: Buffer,
    styles_buffer: Buffer,
    layout_buffer: Buffer,
    vertices_buffer: Buffer,
    counts_buffer: Buffer,
    stylesheet_buffer: Buffer,
    params_buffer: Buffer,

    // State
    params: DocumentParams,
    vertex_count: usize,

    // Scroll state
    scroll_x: f32,
    scroll_y: f32,
}

impl DocumentApp {
    pub fn new(device: &Device, viewport_width: f32, viewport_height: f32) -> Result<Self, String> {
        // Compile unified shader
        let shader_source = format!("{}\n{}", APP_SHADER_HEADER, DOCUMENT_SHADER);
        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(&shader_source, &compile_options)
            .map_err(|e| format!("Failed to compile document shader: {}", e))?;

        // Compute pipeline (runs tokenize → parse → style → layout → paint)
        let compute_fn = library
            .get_function("document_compute", None)
            .map_err(|e| format!("Failed to find document_compute: {}", e))?;
        let compute_pipeline = device
            .new_compute_pipeline_state_with_function(&compute_fn)
            .map_err(|e| format!("Failed to create compute pipeline: {}", e))?;

        // Render pipeline
        let vertex_fn = library
            .get_function("document_vertex", None)
            .map_err(|e| format!("Failed to find document_vertex: {}", e))?;
        let fragment_fn = library
            .get_function("document_fragment", None)
            .map_err(|e| format!("Failed to find document_fragment: {}", e))?;

        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&vertex_fn));
        render_desc.set_fragment_function(Some(&fragment_fn));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .map_err(|e| format!("Failed to create render pipeline: {}", e))?;

        // Allocate buffers
        let html_buffer = device.new_buffer(
            MAX_HTML_SIZE as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tokens_buffer = device.new_buffer(
            (MAX_TOKENS * std::mem::size_of::<Token>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let elements_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<Element>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let styles_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<ComputedStyle>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let layout_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<LayoutBox>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertices_buffer = device.new_buffer(
            (MAX_VERTICES * std::mem::size_of::<PaintVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counts_buffer = device.new_buffer(
            std::mem::size_of::<DocumentCounts>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let stylesheet_buffer = device.new_buffer(
            ((MAX_SELECTORS * std::mem::size_of::<Selector>()) +
             (MAX_STYLE_DEFS * std::mem::size_of::<StyleDef>())) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params_buffer = device.new_buffer(
            std::mem::size_of::<DocumentParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize params
        let params = DocumentParams {
            viewport_width,
            viewport_height,
            scroll_x: 0.0,
            scroll_y: 0.0,
            html_length: 0,
            selector_count: 0,
            _padding: [0; 2],
        };

        Ok(Self {
            name: "DocumentBrowser".to_string(),
            compute_pipeline,
            render_pipeline,
            html_buffer,
            tokens_buffer,
            elements_buffer,
            styles_buffer,
            layout_buffer,
            vertices_buffer,
            counts_buffer,
            stylesheet_buffer,
            params_buffer,
            params,
            vertex_count: 0,
            scroll_x: 0.0,
            scroll_y: 0.0,
        })
    }

    /// Load HTML document (only CPU operation - copies bytes to GPU buffer)
    pub fn load_html(&mut self, html: &[u8]) {
        let len = html.len().min(MAX_HTML_SIZE);
        unsafe {
            std::ptr::copy_nonoverlapping(
                html.as_ptr(),
                self.html_buffer.contents() as *mut u8,
                len,
            );
        }
        self.params.html_length = len as u32;
    }

    /// Load CSS stylesheet (only CPU operation - copies to GPU buffer)
    pub fn load_css(&mut self, css: &str) {
        // Parse CSS into selectors and style defs
        let stylesheet = Stylesheet::parse(css);

        let selector_count = stylesheet.selectors.len().min(MAX_SELECTORS);
        let style_def_count = stylesheet.style_defs.len().min(MAX_STYLE_DEFS);

        unsafe {
            // Copy selectors
            std::ptr::copy_nonoverlapping(
                stylesheet.selectors.as_ptr(),
                self.stylesheet_buffer.contents() as *mut DocSelector,
                selector_count,
            );

            // Copy style defs after selectors
            let style_defs_offset = MAX_SELECTORS * std::mem::size_of::<DocSelector>();
            std::ptr::copy_nonoverlapping(
                stylesheet.style_defs.as_ptr(),
                (self.stylesheet_buffer.contents() as *mut u8).add(style_defs_offset) as *mut DocStyleDef,
                style_def_count,
            );
        }

        self.params.selector_count = selector_count as u32;
    }
}

impl GpuApp for DocumentApp {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_pipeline(&self) -> &ComputePipelineState {
        &self.compute_pipeline
    }

    fn render_pipeline(&self) -> &RenderPipelineState {
        &self.render_pipeline
    }

    fn vertices_buffer(&self) -> &Buffer {
        &self.vertices_buffer
    }

    fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    fn app_buffers(&self) -> Vec<&Buffer> {
        vec![
            &self.html_buffer,       // SLOT_APP_START + 0
            &self.tokens_buffer,     // SLOT_APP_START + 1
            &self.elements_buffer,   // SLOT_APP_START + 2
            &self.styles_buffer,     // SLOT_APP_START + 3
            &self.layout_buffer,     // SLOT_APP_START + 4
            &self.vertices_buffer,   // SLOT_APP_START + 5
            &self.counts_buffer,     // SLOT_APP_START + 6
            &self.stylesheet_buffer, // SLOT_APP_START + 7
        ]
    }

    fn params_buffer(&self) -> &Buffer {
        &self.params_buffer
    }

    fn update_params(&mut self, _frame_state: &FrameState, _delta_time: f32) {
        self.params.scroll_x = self.scroll_x;
        self.params.scroll_y = self.scroll_y;

        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.params as *const DocumentParams,
                self.params_buffer.contents() as *mut DocumentParams,
                1,
            );
        }
    }

    fn handle_input(&mut self, event: &InputEvent) {
        match event.event_type {
            x if x == InputEventType::MouseScroll as u16 => {
                self.scroll_y += event.delta[1] * 20.0;
                self.scroll_y = self.scroll_y.max(0.0);
            }
            _ => {}
        }
    }

    fn post_frame(&mut self, _timing: &FrameTiming) {
        // Read vertex count from GPU
        unsafe {
            let counts = *(self.counts_buffer.contents() as *const DocumentCounts);
            self.vertex_count = counts.vertex_count as usize;
        }
    }

    fn pipeline_mode(&self) -> PipelineMode {
        PipelineMode::LowLatency
    }

    fn clear_color(&self) -> MTLClearColor {
        MTLClearColor::new(1.0, 1.0, 1.0, 1.0) // White background
    }

    fn uses_text_rendering(&self) -> bool {
        true // Enable global text rendering system
    }

    fn render_text(&mut self, text: &mut Text) {
        // Render status bar with scroll position
        let status = format!(
            "Scroll: {:.0}px | HTML: {} bytes",
            self.scroll_y,
            self.params.html_length
        );
        text.add_text(&status, 10.0, self.params.viewport_height - 24.0, colors::DARK_GRAY);
    }
}

// ============================================================================
// Supporting Structures (must match existing document module)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Token {
    token_type: u32,
    start: u32,
    end: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Element {
    element_type: u32,
    parent: i32,
    first_child: i32,
    next_sibling: i32,
    text_start: u32,
    text_length: u32,
    token_index: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct LayoutBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    element_index: u32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ComputedStyle {
    display: u32,
    width: f32,
    height: f32,
    margin: [f32; 4],
    padding: [f32; 4],
    flex_direction: u32,
    justify_content: u32,
    align_items: u32,
    flex_grow: f32,
    flex_shrink: f32,
    color: [f32; 4],
    font_size: f32,
    line_height: f32,
    font_weight: u32,
    text_align: u32,
    background_color: [f32; 4],
    border_width: [f32; 4],
    border_color: [f32; 4],
    border_radius: f32,
    opacity: f32,
    position: u32,
    top: f32,
    right: f32,
    bottom: f32,
    left: f32,
    z_index: i32,
    properties_set: u32,
    overflow_x: u32,
    overflow_y: u32,
    shadow_count: u32,
    shadow_offset_x: [f32; 4],
    shadow_offset_y: [f32; 4],
    shadow_blur: [f32; 4],
    shadow_spread: [f32; 4],
    shadow_color: [f32; 16],
    shadow_inset: [u32; 4],
    gradient_type: u32,
    gradient_angle: f32,
    gradient_stop_count: u32,
    gradient_stop_colors: [f32; 32],
    gradient_stop_positions: [f32; 8],
    border_collapse: u32,
    border_spacing: f32,
    _padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Selector {
    selector_type: u32,
    hash: u32,
    specificity: u32,
    style_start: u32,
    style_count: u32,
    combinator: u32,
    next_part: i32,
    pseudo_type: u32,
    attr_name_hash: u32,
    attr_op: u32,
    attr_value_hash: u32,
    nth_a: i32,
    nth_b: i32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct StyleDef {
    property_id: u32,
    values: [f32; 4],
}

// Reuse existing document types (init-time only, not per-frame)
use super::document::{Stylesheet, Selector as DocSelector, StyleDef as DocStyleDef};

// ============================================================================
// Unified Metal Shader
// ============================================================================

const DOCUMENT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Structures
// ============================================================================

struct DocumentParams {
    float viewport_width;
    float viewport_height;
    float scroll_x;
    float scroll_y;
    uint html_length;
    uint selector_count;
    uint _padding[2];
};

struct DocumentCounts {
    atomic_uint token_count;
    atomic_uint element_count;
    atomic_uint vertex_count;
    uint _padding;
};

struct Token {
    uint token_type;
    uint start;
    uint end;
    uint _padding;
};

struct Element {
    uint element_type;
    int parent;
    int first_child;
    int next_sibling;
    uint text_start;
    uint text_length;
    uint token_index;
    uint _padding;
};

struct ComputedStyle {
    uint display;
    float width;
    float height;
    float margin[4];
    float padding[4];
    uint flex_direction;
    uint justify_content;
    uint align_items;
    float flex_grow;
    float flex_shrink;
    float color[4];
    float font_size;
    float line_height;
    uint font_weight;
    uint text_align;
    float background_color[4];
    float border_width[4];
    float border_color[4];
    float border_radius;
    float opacity;
    uint position;
    float top;
    float right_;
    float bottom;
    float left;
    int z_index;
    uint properties_set;
    uint overflow_x;
    uint overflow_y;
    uint shadow_count;
    float shadow_offset_x[4];
    float shadow_offset_y[4];
    float shadow_blur[4];
    float shadow_spread[4];
    float shadow_color[16];
    uint shadow_inset[4];
    uint gradient_type;
    float gradient_angle;
    uint gradient_stop_count;
    float gradient_stop_colors[32];
    float gradient_stop_positions[8];
    uint border_collapse;
    float border_spacing;
    float _padding[2];
};

struct LayoutBox {
    float x;
    float y;
    float width;
    float height;
    uint element_index;
    uint _padding[3];
};

struct PaintVertex {
    float2 position;
    float2 tex_coord;
    float4 color;
    uint flags;
    uint _padding[3];
};

struct Selector {
    uint selector_type;
    uint hash;
    uint specificity;
    uint style_start;
    uint style_count;
    uint combinator;
    int next_part;
    uint pseudo_type;
    uint attr_name_hash;
    uint attr_op;
    uint attr_value_hash;
    int nth_a;
    int nth_b;
    uint _padding[3];
};

struct StyleDef {
    uint property_id;
    float values[4];
};

// ============================================================================
// Constants
// ============================================================================

constant uint TOKEN_TAG_OPEN = 1;
constant uint TOKEN_TAG_CLOSE = 2;
constant uint TOKEN_TAG_SELF = 3;
constant uint TOKEN_TEXT = 4;

constant uint ELEM_TEXT = 100;
constant uint ELEM_DIV = 1;
constant uint ELEM_SPAN = 2;
constant uint ELEM_P = 3;

constant uint DISPLAY_NONE = 0;
constant uint DISPLAY_BLOCK = 1;
constant uint DISPLAY_INLINE = 2;

constant uint MAX_TOKENS = 65536;
constant uint MAX_ELEMENTS = 16384;

// ============================================================================
// Tokenizer (Phase 1)
// Each thread processes a chunk of HTML
// ============================================================================

kernel void tokenize(
    device const uint8_t* html [[buffer(3)]],       // SLOT_HTML
    device Token* tokens [[buffer(4)]],              // SLOT_TOKENS
    device DocumentCounts* counts [[buffer(9)]],     // SLOT_COUNTS
    constant DocumentParams& params [[buffer(2)]],   // SLOT_APP_PARAMS
    uint gid [[thread_position_in_grid]]
) {
    // Simple single-threaded tokenizer for now
    // TODO: Parallel tokenization with prefix sum
    if (gid != 0) return;

    uint html_len = params.html_length;
    uint token_idx = 0;
    uint i = 0;

    while (i < html_len && token_idx < MAX_TOKENS) {
        // Skip whitespace
        while (i < html_len && (html[i] == ' ' || html[i] == '\n' || html[i] == '\r' || html[i] == '\t')) {
            i++;
        }
        if (i >= html_len) break;

        if (html[i] == '<') {
            uint start = i;
            i++; // Skip <

            // Check for closing tag
            bool is_close = false;
            if (i < html_len && html[i] == '/') {
                is_close = true;
                i++;
            }

            // Find end of tag
            bool is_self_closing = false;
            while (i < html_len && html[i] != '>') {
                if (html[i] == '/' && i + 1 < html_len && html[i + 1] == '>') {
                    is_self_closing = true;
                }
                i++;
            }
            if (i < html_len) i++; // Skip >

            Token tok;
            tok.start = start;
            tok.end = i;
            tok._padding = 0;

            if (is_close) {
                tok.token_type = TOKEN_TAG_CLOSE;
            } else if (is_self_closing) {
                tok.token_type = TOKEN_TAG_SELF;
            } else {
                tok.token_type = TOKEN_TAG_OPEN;
            }

            tokens[token_idx++] = tok;
        } else {
            // Text content
            uint start = i;
            while (i < html_len && html[i] != '<') {
                i++;
            }

            if (i > start) {
                Token tok;
                tok.token_type = TOKEN_TEXT;
                tok.start = start;
                tok.end = i;
                tok._padding = 0;
                tokens[token_idx++] = tok;
            }
        }
    }

    atomic_store_explicit(&counts->token_count, token_idx, memory_order_relaxed);
}

// ============================================================================
// Parser (Phase 2)
// Builds DOM tree from tokens
// ============================================================================

// Hash function for tag names
uint hash_tag(device const uint8_t* html, uint start, uint end) {
    uint hash = 5381;
    for (uint i = start; i < end; i++) {
        uint8_t c = html[i];
        if (c >= 'A' && c <= 'Z') c += 32; // lowercase
        if (c == ' ' || c == '/' || c == '>') break;
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

uint get_element_type(device const uint8_t* html, uint start, uint end) {
    // Skip < and any /
    uint i = start + 1;
    if (html[i] == '/') i++;

    uint tag_start = i;
    while (i < end && html[i] != ' ' && html[i] != '>' && html[i] != '/') {
        i++;
    }

    uint hash = hash_tag(html, tag_start, i);

    // Common tags
    if (hash == 193486360) return ELEM_DIV;   // "div"
    if (hash == 2090770405) return ELEM_SPAN; // "span"
    if (hash == 177621) return ELEM_P;        // "p"

    return ELEM_DIV; // Default to div
}

kernel void parse(
    device const uint8_t* html [[buffer(3)]],
    device const Token* tokens [[buffer(4)]],
    device Element* elements [[buffer(5)]],
    device DocumentCounts* counts [[buffer(9)]],
    constant DocumentParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint token_count = atomic_load_explicit(&counts->token_count, memory_order_relaxed);
    uint elem_idx = 0;
    int stack[64];
    int stack_top = -1;

    // Create root element
    elements[elem_idx].element_type = ELEM_DIV;
    elements[elem_idx].parent = -1;
    elements[elem_idx].first_child = -1;
    elements[elem_idx].next_sibling = -1;
    elements[elem_idx].text_start = 0;
    elements[elem_idx].text_length = 0;
    elements[elem_idx].token_index = 0;
    stack[++stack_top] = elem_idx++;

    for (uint t = 0; t < token_count && elem_idx < MAX_ELEMENTS; t++) {
        Token tok = tokens[t];

        if (tok.token_type == TOKEN_TAG_OPEN || tok.token_type == TOKEN_TAG_SELF) {
            // Create element
            uint new_idx = elem_idx++;
            elements[new_idx].element_type = get_element_type(html, tok.start, tok.end);
            elements[new_idx].parent = stack_top >= 0 ? stack[stack_top] : -1;
            elements[new_idx].first_child = -1;
            elements[new_idx].next_sibling = -1;
            elements[new_idx].text_start = 0;
            elements[new_idx].text_length = 0;
            elements[new_idx].token_index = t;

            // Link to parent
            if (stack_top >= 0) {
                int parent = stack[stack_top];
                if (elements[parent].first_child < 0) {
                    elements[parent].first_child = new_idx;
                } else {
                    // Find last sibling (Issue #264: add cycle detection)
                    int sib = elements[parent].first_child;
                    int max_iter = 10000;  // Prevent infinite loop
                    while (elements[sib].next_sibling >= 0 && max_iter > 0) {
                        sib = elements[sib].next_sibling;
                        max_iter--;
                    }
                    if (max_iter > 0) elements[sib].next_sibling = new_idx;
                }
            }

            if (tok.token_type == TOKEN_TAG_OPEN) {
                stack[++stack_top] = new_idx;
            }
        } else if (tok.token_type == TOKEN_TAG_CLOSE) {
            if (stack_top > 0) stack_top--;
        } else if (tok.token_type == TOKEN_TEXT) {
            // Create text node
            uint new_idx = elem_idx++;
            elements[new_idx].element_type = ELEM_TEXT;
            elements[new_idx].parent = stack_top >= 0 ? stack[stack_top] : -1;
            elements[new_idx].first_child = -1;
            elements[new_idx].next_sibling = -1;
            elements[new_idx].text_start = tok.start;
            elements[new_idx].text_length = tok.end - tok.start;
            elements[new_idx].token_index = t;

            // Link to parent
            if (stack_top >= 0) {
                int parent = stack[stack_top];
                if (elements[parent].first_child < 0) {
                    elements[parent].first_child = new_idx;
                } else {
                    // Issue #264: add cycle detection
                    int sib = elements[parent].first_child;
                    int max_iter = 10000;
                    while (elements[sib].next_sibling >= 0 && max_iter > 0) {
                        sib = elements[sib].next_sibling;
                        max_iter--;
                    }
                    if (max_iter > 0) elements[sib].next_sibling = new_idx;
                }
            }
        }
    }

    atomic_store_explicit(&counts->element_count, elem_idx, memory_order_relaxed);
}

// ============================================================================
// Styler (Phase 3)
// Computes styles for each element (parallel)
// ============================================================================

ComputedStyle default_style() {
    ComputedStyle s;
    s.display = DISPLAY_BLOCK;
    s.width = 0;
    s.height = 0;
    for (int i = 0; i < 4; i++) {
        s.margin[i] = 0;
        s.padding[i] = 0;
        s.color[i] = i < 3 ? 0.0 : 1.0;
        s.background_color[i] = i < 3 ? 1.0 : 0.0; // Transparent white
        s.border_width[i] = 0;
        s.border_color[i] = 0;
    }
    s.font_size = 16.0;
    s.line_height = 1.2;
    s.font_weight = 400;
    s.text_align = 0;
    s.border_radius = 0;
    s.opacity = 1.0;
    s.position = 0;
    s.top = 0;
    s.right_ = 0;
    s.bottom = 0;
    s.left = 0;
    s.z_index = 0;
    s.properties_set = 0;
    s.overflow_x = 0;
    s.overflow_y = 0;
    s.shadow_count = 0;
    s.gradient_type = 0;
    s.border_collapse = 0;
    s.border_spacing = 0;
    return s;
}

// Parse inline style attribute on GPU
void parse_inline_style(
    device const uint8_t* html,
    uint start,
    uint end,
    thread ComputedStyle* style
) {
    // Find style=" in the tag
    uint i = start;
    while (i + 6 < end) {
        if (html[i] == 's' && html[i+1] == 't' && html[i+2] == 'y' &&
            html[i+3] == 'l' && html[i+4] == 'e' && html[i+5] == '=') {
            i += 6;
            char quote = html[i];
            if (quote == '"' || quote == '\'') {
                i++;
                uint style_start = i;
                while (i < end && html[i] != quote) i++;
                uint style_end = i;

                // Parse CSS declarations
                uint j = style_start;
                while (j < style_end) {
                    // Skip whitespace
                    while (j < style_end && (html[j] == ' ' || html[j] == ';')) j++;
                    if (j >= style_end) break;

                    // Get property name
                    uint prop_start = j;
                    while (j < style_end && html[j] != ':' && html[j] != ';') j++;
                    uint prop_end = j;
                    if (j < style_end && html[j] == ':') j++;

                    // Skip whitespace
                    while (j < style_end && html[j] == ' ') j++;

                    // Get value
                    uint val_start = j;
                    while (j < style_end && html[j] != ';') j++;
                    uint val_end = j;

                    // Trim trailing whitespace from value
                    while (val_end > val_start && html[val_end-1] == ' ') val_end--;

                    // Parse property
                    uint prop_hash = hash_tag(html, prop_start, prop_end);

                    // width
                    if (prop_hash == 261238937) {
                        float val = 0;
                        for (uint k = val_start; k < val_end && html[k] >= '0' && html[k] <= '9'; k++) {
                            val = val * 10 + (html[k] - '0');
                        }
                        style->width = val;
                        style->properties_set |= 1;
                    }
                    // height
                    else if (prop_hash == 2090324718) {
                        float val = 0;
                        for (uint k = val_start; k < val_end && html[k] >= '0' && html[k] <= '9'; k++) {
                            val = val * 10 + (html[k] - '0');
                        }
                        style->height = val;
                        style->properties_set |= 2;
                    }
                    // color
                    else if (prop_hash == 254419046) {
                        // Check for "red", "blue", etc.
                        uint val_hash = hash_tag(html, val_start, val_end);
                        if (val_hash == 193462384) { // red
                            style->color[0] = 1.0; style->color[1] = 0.0; style->color[2] = 0.0;
                        } else if (val_hash == 2090117158) { // blue
                            style->color[0] = 0.0; style->color[1] = 0.0; style->color[2] = 1.0;
                        } else if (val_hash == 248962580) { // green
                            style->color[0] = 0.0; style->color[1] = 0.5; style->color[2] = 0.0;
                        }
                        style->properties_set |= 4;
                    }
                    // display
                    else if (prop_hash == 249995787) {
                        uint val_hash = hash_tag(html, val_start, val_end);
                        if (val_hash == 2090511091) { // none
                            style->display = DISPLAY_NONE;
                        } else if (val_hash == 254760018) { // flex
                            style->display = 3;
                        }
                        style->properties_set |= 8;
                    }
                    // background-color
                    else if (prop_hash == 1908504991) {
                        uint val_hash = hash_tag(html, val_start, val_end);
                        if (val_hash == 193462384) { // red
                            style->background_color[0] = 1.0; style->background_color[1] = 0.0;
                            style->background_color[2] = 0.0; style->background_color[3] = 1.0;
                        } else if (val_hash == 2090117158) { // blue
                            style->background_color[0] = 0.0; style->background_color[1] = 0.0;
                            style->background_color[2] = 1.0; style->background_color[3] = 1.0;
                        }
                        // Handle hex colors #rrggbb
                        else if (html[val_start] == '#' && val_end - val_start >= 7) {
                            auto hex_digit = [](uint8_t c) -> float {
                                if (c >= '0' && c <= '9') return (c - '0') / 15.0;
                                if (c >= 'a' && c <= 'f') return (c - 'a' + 10) / 15.0;
                                if (c >= 'A' && c <= 'F') return (c - 'A' + 10) / 15.0;
                                return 0;
                            };
                            style->background_color[0] = hex_digit(html[val_start+1]) * 16 + hex_digit(html[val_start+2]);
                            style->background_color[0] /= 255.0;
                            style->background_color[1] = hex_digit(html[val_start+3]) * 16 + hex_digit(html[val_start+4]);
                            style->background_color[1] /= 255.0;
                            style->background_color[2] = hex_digit(html[val_start+5]) * 16 + hex_digit(html[val_start+6]);
                            style->background_color[2] /= 255.0;
                            style->background_color[3] = 1.0;
                        }
                        style->properties_set |= 16;
                    }
                    // margin (simple single value)
                    else if (prop_hash == 2090889948) {
                        float val = 0;
                        for (uint k = val_start; k < val_end && html[k] >= '0' && html[k] <= '9'; k++) {
                            val = val * 10 + (html[k] - '0');
                        }
                        style->margin[0] = style->margin[1] = style->margin[2] = style->margin[3] = val;
                        style->properties_set |= 32;
                    }
                }
                return;
            }
        }
        i++;
    }
}

kernel void compute_styles(
    device const uint8_t* html [[buffer(3)]],
    device const Token* tokens [[buffer(4)]],
    device const Element* elements [[buffer(5)]],
    device ComputedStyle* styles [[buffer(6)]],
    device DocumentCounts* counts [[buffer(9)]],
    constant DocumentParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint elem_count = atomic_load_explicit(&counts->element_count, memory_order_relaxed);
    if (gid >= elem_count) return;

    Element elem = elements[gid];
    ComputedStyle style = default_style();

    // Text nodes inherit from parent
    if (elem.element_type == ELEM_TEXT) {
        if (elem.parent >= 0) {
            style = styles[elem.parent];
        }
    } else {
        // Parse inline styles from HTML
        Token tok = tokens[elem.token_index];
        parse_inline_style(html, tok.start, tok.end, &style);
    }

    styles[gid] = style;
}

// ============================================================================
// Layout (Phase 4)
// Computes positions for each element (parallel with dependencies)
// ============================================================================

kernel void compute_layout(
    device const Element* elements [[buffer(5)]],
    device const ComputedStyle* styles [[buffer(6)]],
    device LayoutBox* boxes [[buffer(7)]],
    device DocumentCounts* counts [[buffer(9)]],
    constant DocumentParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint elem_count = atomic_load_explicit(&counts->element_count, memory_order_relaxed);
    if (gid >= elem_count) return;

    Element elem = elements[gid];
    ComputedStyle style = styles[gid];

    LayoutBox box;
    box.element_index = gid;

    // Simple block layout
    if (style.display == DISPLAY_NONE) {
        box.x = 0; box.y = 0; box.width = 0; box.height = 0;
    } else {
        // Default width to viewport, height to content or explicit
        box.width = style.width > 0 ? style.width : params.viewport_width;
        box.height = style.height > 0 ? style.height : style.font_size * style.line_height;

        // Position based on parent
        if (elem.parent >= 0) {
            LayoutBox parent_box = boxes[elem.parent];
            box.x = parent_box.x + style.margin[3];

            // Stack vertically (simplified)
            float y_offset = parent_box.y + style.margin[0];

            // Count previous siblings
            int sib = elements[elem.parent].first_child;
            while (sib >= 0 && uint(sib) != gid) {
                y_offset += boxes[sib].height + styles[sib].margin[2];
                sib = elements[sib].next_sibling;
            }
            box.y = y_offset;
        } else {
            box.x = style.margin[3];
            box.y = style.margin[0];
        }
    }

    boxes[gid] = box;
}

// ============================================================================
// Paint (Phase 5)
// Generates vertices for each visible element
// ============================================================================

kernel void paint(
    device const Element* elements [[buffer(5)]],
    device const ComputedStyle* styles [[buffer(6)]],
    device const LayoutBox* boxes [[buffer(7)]],
    device PaintVertex* vertices [[buffer(8)]],
    device DocumentCounts* counts [[buffer(9)]],
    constant DocumentParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint elem_count = atomic_load_explicit(&counts->element_count, memory_order_relaxed);
    if (gid >= elem_count) return;

    ComputedStyle style = styles[gid];
    if (style.display == DISPLAY_NONE) return;
    if (style.background_color[3] < 0.01) return; // Skip transparent

    LayoutBox box = boxes[gid];
    if (box.width <= 0 || box.height <= 0) return;

    // Viewport culling
    float scroll_y = params.scroll_y;
    if (box.y + box.height < scroll_y) return;
    if (box.y > scroll_y + params.viewport_height) return;

    // Allocate 6 vertices (2 triangles for quad)
    uint base = atomic_fetch_add_explicit(&counts->vertex_count, 6, memory_order_relaxed);

    // Convert to clip space
    float vw = params.viewport_width;
    float vh = params.viewport_height;

    float x0 = (box.x / vw) * 2.0 - 1.0;
    float y0 = 1.0 - ((box.y - scroll_y) / vh) * 2.0;
    float x1 = ((box.x + box.width) / vw) * 2.0 - 1.0;
    float y1 = 1.0 - ((box.y - scroll_y + box.height) / vh) * 2.0;

    float4 color = float4(style.background_color[0], style.background_color[1],
                          style.background_color[2], style.background_color[3]);

    // Triangle 1
    vertices[base + 0].position = float2(x0, y0);
    vertices[base + 0].color = color;
    vertices[base + 1].position = float2(x1, y0);
    vertices[base + 1].color = color;
    vertices[base + 2].position = float2(x0, y1);
    vertices[base + 2].color = color;

    // Triangle 2
    vertices[base + 3].position = float2(x1, y0);
    vertices[base + 3].color = color;
    vertices[base + 4].position = float2(x1, y1);
    vertices[base + 4].color = color;
    vertices[base + 5].position = float2(x0, y1);
    vertices[base + 5].color = color;
}

// ============================================================================
// Main Compute Kernel - Chains all phases
// ============================================================================

kernel void document_compute(
    device FrameState* frame [[buffer(0)]],
    device InputQueue* input [[buffer(1)]],
    constant DocumentParams& params [[buffer(2)]],
    device uint8_t* html [[buffer(3)]],
    device Token* tokens [[buffer(4)]],
    device Element* elements [[buffer(5)]],
    device ComputedStyle* styles [[buffer(6)]],
    device LayoutBox* boxes [[buffer(7)]],
    device PaintVertex* vertices [[buffer(8)]],
    device DocumentCounts* counts [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Reset counts at start
    if (gid == 0) {
        atomic_store_explicit(&counts->token_count, 0, memory_order_relaxed);
        atomic_store_explicit(&counts->element_count, 0, memory_order_relaxed);
        atomic_store_explicit(&counts->vertex_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 1: Tokenize (single thread for now)
    if (gid == 0) {
        uint html_len = params.html_length;
        uint token_idx = 0;
        uint i = 0;

        while (i < html_len && token_idx < MAX_TOKENS) {
            while (i < html_len && (html[i] == ' ' || html[i] == '\n' || html[i] == '\r' || html[i] == '\t')) i++;
            if (i >= html_len) break;

            if (html[i] == '<') {
                uint start = i++;
                bool is_close = (i < html_len && html[i] == '/');
                if (is_close) i++;
                bool is_self = false;
                while (i < html_len && html[i] != '>') {
                    if (html[i] == '/' && i + 1 < html_len && html[i + 1] == '>') is_self = true;
                    i++;
                }
                if (i < html_len) i++;

                tokens[token_idx].start = start;
                tokens[token_idx].end = i;
                tokens[token_idx].token_type = is_close ? TOKEN_TAG_CLOSE : (is_self ? TOKEN_TAG_SELF : TOKEN_TAG_OPEN);
                token_idx++;
            } else {
                uint start = i;
                while (i < html_len && html[i] != '<') i++;
                if (i > start) {
                    tokens[token_idx].token_type = TOKEN_TEXT;
                    tokens[token_idx].start = start;
                    tokens[token_idx].end = i;
                    token_idx++;
                }
            }
        }
        atomic_store_explicit(&counts->token_count, token_idx, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Parse (single thread for now - tree building)
    if (gid == 0) {
        uint token_count = atomic_load_explicit(&counts->token_count, memory_order_relaxed);
        uint elem_idx = 0;
        int stack[64];
        int stack_top = -1;

        elements[elem_idx].element_type = ELEM_DIV;
        elements[elem_idx].parent = -1;
        elements[elem_idx].first_child = -1;
        elements[elem_idx].next_sibling = -1;
        elements[elem_idx].token_index = 0;
        stack[++stack_top] = elem_idx++;

        for (uint t = 0; t < token_count && elem_idx < MAX_ELEMENTS; t++) {
            Token tok = tokens[t];

            if (tok.token_type == TOKEN_TAG_OPEN || tok.token_type == TOKEN_TAG_SELF) {
                uint new_idx = elem_idx++;
                elements[new_idx].element_type = get_element_type(html, tok.start, tok.end);
                elements[new_idx].parent = stack_top >= 0 ? stack[stack_top] : -1;
                elements[new_idx].first_child = -1;
                elements[new_idx].next_sibling = -1;
                elements[new_idx].token_index = t;

                if (stack_top >= 0) {
                    int parent = stack[stack_top];
                    if (elements[parent].first_child < 0) {
                        elements[parent].first_child = new_idx;
                    } else {
                        // Issue #264: add cycle detection
                        int sib = elements[parent].first_child;
                        int max_iter = 10000;
                        while (elements[sib].next_sibling >= 0 && max_iter > 0) { sib = elements[sib].next_sibling; max_iter--; }
                        if (max_iter > 0) elements[sib].next_sibling = new_idx;
                    }
                }

                if (tok.token_type == TOKEN_TAG_OPEN) stack[++stack_top] = new_idx;
            } else if (tok.token_type == TOKEN_TAG_CLOSE) {
                if (stack_top > 0) stack_top--;
            } else if (tok.token_type == TOKEN_TEXT) {
                uint new_idx = elem_idx++;
                elements[new_idx].element_type = ELEM_TEXT;
                elements[new_idx].parent = stack_top >= 0 ? stack[stack_top] : -1;
                elements[new_idx].first_child = -1;
                elements[new_idx].next_sibling = -1;
                elements[new_idx].text_start = tok.start;
                elements[new_idx].text_length = tok.end - tok.start;
                elements[new_idx].token_index = t;

                if (stack_top >= 0) {
                    int parent = stack[stack_top];
                    if (elements[parent].first_child < 0) {
                        elements[parent].first_child = new_idx;
                    } else {
                        // Issue #264: add cycle detection
                        int sib = elements[parent].first_child;
                        int max_iter = 10000;
                        while (elements[sib].next_sibling >= 0 && max_iter > 0) { sib = elements[sib].next_sibling; max_iter--; }
                        if (max_iter > 0) elements[sib].next_sibling = new_idx;
                    }
                }
            }
        }
        atomic_store_explicit(&counts->element_count, elem_idx, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 3: Style (parallel - each thread handles one element)
    uint elem_count = atomic_load_explicit(&counts->element_count, memory_order_relaxed);
    if (gid < elem_count) {
        Element elem = elements[gid];
        ComputedStyle style = default_style();

        if (elem.element_type != ELEM_TEXT) {
            Token tok = tokens[elem.token_index];
            parse_inline_style(html, tok.start, tok.end, &style);
        }

        styles[gid] = style;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 4: Layout (needs parent info, run multiple passes)
    for (uint pass = 0; pass < 8; pass++) {
        if (gid < elem_count) {
            Element elem = elements[gid];
            ComputedStyle style = styles[gid];

            LayoutBox box;
            box.element_index = gid;

            if (style.display == DISPLAY_NONE) {
                box.x = 0; box.y = 0; box.width = 0; box.height = 0;
            } else {
                box.width = style.width > 0 ? style.width : params.viewport_width - style.margin[1] - style.margin[3];
                box.height = style.height > 0 ? style.height : style.font_size * style.line_height;

                if (elem.parent >= 0) {
                    LayoutBox parent_box = boxes[elem.parent];
                    box.x = parent_box.x + style.margin[3];

                    float y_offset = parent_box.y + style.margin[0];
                    int sib = elements[elem.parent].first_child;
                    while (sib >= 0 && uint(sib) != gid) {
                        if (styles[sib].display != DISPLAY_NONE) {
                            y_offset += boxes[sib].height + styles[sib].margin[0] + styles[sib].margin[2];
                        }
                        sib = elements[sib].next_sibling;
                    }
                    box.y = y_offset;
                } else {
                    box.x = style.margin[3];
                    box.y = style.margin[0];
                }
            }

            boxes[gid] = box;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Phase 5: Paint (parallel)
    if (gid < elem_count) {
        ComputedStyle style = styles[gid];
        if (style.display == DISPLAY_NONE) return;
        if (style.background_color[3] < 0.01) return;

        LayoutBox box = boxes[gid];
        if (box.width <= 0 || box.height <= 0) return;

        float scroll_y = params.scroll_y;
        if (box.y + box.height < scroll_y) return;
        if (box.y > scroll_y + params.viewport_height) return;

        uint base = atomic_fetch_add_explicit(&counts->vertex_count, 6, memory_order_relaxed);

        float vw = params.viewport_width;
        float vh = params.viewport_height;

        float x0 = (box.x / vw) * 2.0 - 1.0;
        float y0 = 1.0 - ((box.y - scroll_y) / vh) * 2.0;
        float x1 = ((box.x + box.width) / vw) * 2.0 - 1.0;
        float y1 = 1.0 - ((box.y - scroll_y + box.height) / vh) * 2.0;

        float4 color = float4(style.background_color[0], style.background_color[1],
                              style.background_color[2], style.background_color[3]);

        vertices[base + 0].position = float2(x0, y0); vertices[base + 0].color = color;
        vertices[base + 1].position = float2(x1, y0); vertices[base + 1].color = color;
        vertices[base + 2].position = float2(x0, y1); vertices[base + 2].color = color;
        vertices[base + 3].position = float2(x1, y0); vertices[base + 3].color = color;
        vertices[base + 4].position = float2(x1, y1); vertices[base + 4].color = color;
        vertices[base + 5].position = float2(x0, y1); vertices[base + 5].color = color;
    }
}

// ============================================================================
// Render Shaders
// ============================================================================

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut document_vertex(
    device const PaintVertex* vertices [[buffer(0)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = float4(vertices[vid].position, 0.0, 1.0);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 document_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;
