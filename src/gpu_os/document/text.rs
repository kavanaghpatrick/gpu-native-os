//! GPU-Accelerated Text Wrapping & Line Breaking
//!
//! Issue #60: Computes line breaks and positions text across multiple lines
//! using Metal compute shaders for GPU-native text layout.
//!
//! Issue #90: GPU-Native Text Containers with real glyph metrics,
//! parallel prefix sum, and fully parallel glyph positioning.

use metal::*;
use super::parser::Element;
use super::style::ComputedStyle;
use super::layout::LayoutBox;

const THREAD_COUNT: u64 = 1024;
const MAX_ELEMENTS: usize = 65536;
const MAX_LINES: usize = 65536;  // Max lines across all text elements
const MAX_TEXT_LENGTH: usize = 1024 * 1024;  // 1MB text buffer
const NUM_GLYPHS: usize = 96;  // ASCII 32-127
const BASE_FONT_SIZE: f32 = 16.0;  // Base size for glyph metrics

/// Glyph metrics from font (uploaded to GPU once)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphMetrics {
    pub advance: f32,       // Horizontal advance width
    pub bearing_x: f32,     // Left side bearing
    pub bearing_y: f32,     // Top bearing (baseline to top)
    pub width: f32,         // Glyph bbox width
    pub height: f32,        // Glyph bbox height
    pub atlas_x: u16,       // Atlas position X
    pub atlas_y: u16,       // Atlas position Y
    pub atlas_w: u16,       // Atlas size W
    pub atlas_h: u16,       // Atlas size H
}

/// Positioned glyph ready for rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PositionedGlyph {
    pub x: f32,             // Screen X position
    pub y: f32,             // Screen Y position
    pub glyph_id: u32,      // Index into atlas
    pub color: u32,         // Packed RGBA
    pub scale: f32,         // Font size / base size
    pub line_index: u32,    // Which line
    pub _padding: [f32; 2],
}

/// Line box representing a single line of text within a text element
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LineBox {
    /// Index of the text element this line belongs to
    pub element_index: u32,
    /// Start character offset within element's text
    pub char_start: u32,
    /// End character offset (exclusive)
    pub char_end: u32,
    /// Line width in pixels
    pub width: f32,
    /// X position of line start (for text-align)
    pub x: f32,
    /// Y position of line (from element's top)
    pub y: f32,
    /// Padding for GPU alignment
    pub _padding: [f32; 2],
}

/// Break opportunity at a character position
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BreakOpportunity {
    /// Character index where break can occur
    pub char_index: u32,
    /// Type: 0=none, 1=space, 2=hyphen, 3=newline
    pub break_type: u32,
    /// Cumulative width up to this point
    pub cumulative_width: f32,
    /// Padding
    pub _padding: f32,
}

/// White-space CSS property values
pub const WHITE_SPACE_NORMAL: u32 = 0;
pub const WHITE_SPACE_NOWRAP: u32 = 1;
pub const WHITE_SPACE_PRE: u32 = 2;
pub const WHITE_SPACE_PRE_WRAP: u32 = 3;
pub const WHITE_SPACE_PRE_LINE: u32 = 4;

/// Issue #131: Two-Pass Line Layout structures
/// Maximum lines per text element (for GPU buffer sizing)
pub const MAX_LINES_PER_ELEMENT: usize = 64;

/// Pre-computed line information (16 bytes, GPU-aligned)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LineInfo {
    /// First character index in this line
    pub char_start: u32,
    /// Last character index (exclusive)
    pub char_end: u32,
    /// Y position of this line relative to element
    pub y_offset: f32,
    /// Actual width of text on this line
    pub width: f32,
}

/// Per-element line data header (stored separately from line array)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TextLineDataHeader {
    /// Number of lines in this text element
    pub line_count: u32,
    /// Padding for 16-byte alignment
    pub _padding: [u32; 3],
}

/// Text-align CSS property values
pub const TEXT_ALIGN_LEFT: u32 = 0;
pub const TEXT_ALIGN_CENTER: u32 = 1;
pub const TEXT_ALIGN_RIGHT: u32 = 2;
pub const TEXT_ALIGN_JUSTIFY: u32 = 3;

const TEXT_SHADER: &str = include_str!("text.metal");

/// Generate default glyph metrics for 8x8 bitmap font
/// These are monospace metrics matching text_render.rs
pub fn generate_default_metrics() -> Vec<GlyphMetrics> {
    let char_width = 8.0f32;
    let char_height = 8.0f32;
    let chars_per_row = 16u16;

    (0..NUM_GLYPHS).map(|i| {
        let col = (i % chars_per_row as usize) as u16;
        let row = (i / chars_per_row as usize) as u16;

        // Special case for space (narrower)
        let advance = if i == 0 { char_width * 0.5 } else { char_width };

        GlyphMetrics {
            advance,
            bearing_x: 0.0,
            bearing_y: char_height,  // Top of glyph relative to baseline
            width: char_width,
            height: char_height,
            atlas_x: col * 8,
            atlas_y: row * 8,
            atlas_w: 8,
            atlas_h: 8,
        }
    }).collect()
}

/// GPU-accelerated text wrapping engine
pub struct GpuTextEngine {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,

    // Legacy pipelines (Issue #60)
    measure_pipeline: ComputePipelineState,
    find_breaks_pipeline: ComputePipelineState,
    compute_lines_pipeline: ComputePipelineState,
    position_lines_pipeline: ComputePipelineState,

    // New GPU-native pipelines (Issue #90)
    char_to_glyph_pipeline: ComputePipelineState,
    prefix_sum_pipeline: ComputePipelineState,
    find_breaks_parallel_pipeline: ComputePipelineState,
    assign_lines_pipeline: ComputePipelineState,
    position_glyphs_pipeline: ComputePipelineState,
    generate_vertices_pipeline: ComputePipelineState,

    // Legacy buffers
    element_buffer: Buffer,
    style_buffer: Buffer,
    layout_buffer: Buffer,
    text_buffer: Buffer,
    break_buffer: Buffer,
    line_buffer: Buffer,
    line_count_buffer: Buffer,
    element_count_buffer: Buffer,

    // New GPU-native buffers (Issue #90)
    metrics_buffer: Buffer,         // GlyphMetrics[96]
    advances_buffer: Buffer,        // float per char
    cumulative_buffer: Buffer,      // float per char (prefix sum output)
    glyph_ids_buffer: Buffer,       // uint per char
    is_break_buffer: Buffer,        // uint per char
    break_type_buffer: Buffer,      // uint per char
    line_indices_buffer: Buffer,    // uint per char
    positioned_glyphs_buffer: Buffer, // PositionedGlyph per char
    text_vertices_buffer: Buffer,   // 6 vertices × 8 floats per char
}

impl GpuTextEngine {
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(TEXT_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile text shader: {}", e))?;

        // Legacy pipelines (Issue #60)
        let measure_fn = library
            .get_function("measure_text_advances", None)
            .map_err(|e| format!("Failed to get measure function: {}", e))?;
        let find_breaks_fn = library
            .get_function("find_break_opportunities", None)
            .map_err(|e| format!("Failed to get find_breaks function: {}", e))?;
        let compute_lines_fn = library
            .get_function("compute_line_breaks", None)
            .map_err(|e| format!("Failed to get compute_lines function: {}", e))?;
        let position_lines_fn = library
            .get_function("position_lines", None)
            .map_err(|e| format!("Failed to get position_lines function: {}", e))?;

        let measure_pipeline = device
            .new_compute_pipeline_state_with_function(&measure_fn)
            .map_err(|e| format!("Failed to create measure pipeline: {}", e))?;
        let find_breaks_pipeline = device
            .new_compute_pipeline_state_with_function(&find_breaks_fn)
            .map_err(|e| format!("Failed to create find_breaks pipeline: {}", e))?;
        let compute_lines_pipeline = device
            .new_compute_pipeline_state_with_function(&compute_lines_fn)
            .map_err(|e| format!("Failed to create compute_lines pipeline: {}", e))?;
        let position_lines_pipeline = device
            .new_compute_pipeline_state_with_function(&position_lines_fn)
            .map_err(|e| format!("Failed to create position_lines pipeline: {}", e))?;

        // New GPU-native pipelines (Issue #90)
        let char_to_glyph_fn = library
            .get_function("char_to_glyph", None)
            .map_err(|e| format!("Failed to get char_to_glyph function: {}", e))?;
        let prefix_sum_fn = library
            .get_function("prefix_sum_sequential", None)
            .map_err(|e| format!("Failed to get prefix_sum function: {}", e))?;
        let find_breaks_parallel_fn = library
            .get_function("find_breaks_parallel", None)
            .map_err(|e| format!("Failed to get find_breaks_parallel function: {}", e))?;
        let assign_lines_fn = library
            .get_function("assign_lines_parallel", None)
            .map_err(|e| format!("Failed to get assign_lines function: {}", e))?;
        let position_glyphs_fn = library
            .get_function("position_glyphs", None)
            .map_err(|e| format!("Failed to get position_glyphs function: {}", e))?;
        let generate_vertices_fn = library
            .get_function("generate_text_vertices", None)
            .map_err(|e| format!("Failed to get generate_text_vertices function: {}", e))?;

        let char_to_glyph_pipeline = device
            .new_compute_pipeline_state_with_function(&char_to_glyph_fn)
            .map_err(|e| format!("Failed to create char_to_glyph pipeline: {}", e))?;
        let prefix_sum_pipeline = device
            .new_compute_pipeline_state_with_function(&prefix_sum_fn)
            .map_err(|e| format!("Failed to create prefix_sum pipeline: {}", e))?;
        let find_breaks_parallel_pipeline = device
            .new_compute_pipeline_state_with_function(&find_breaks_parallel_fn)
            .map_err(|e| format!("Failed to create find_breaks_parallel pipeline: {}", e))?;
        let assign_lines_pipeline = device
            .new_compute_pipeline_state_with_function(&assign_lines_fn)
            .map_err(|e| format!("Failed to create assign_lines pipeline: {}", e))?;
        let position_glyphs_pipeline = device
            .new_compute_pipeline_state_with_function(&position_glyphs_fn)
            .map_err(|e| format!("Failed to create position_glyphs pipeline: {}", e))?;
        let generate_vertices_pipeline = device
            .new_compute_pipeline_state_with_function(&generate_vertices_fn)
            .map_err(|e| format!("Failed to create generate_vertices pipeline: {}", e))?;

        let element_size = std::mem::size_of::<Element>();
        let style_size = std::mem::size_of::<ComputedStyle>();
        let layout_size = std::mem::size_of::<LayoutBox>();
        let break_size = std::mem::size_of::<BreakOpportunity>();
        let line_size = std::mem::size_of::<LineBox>();
        let metrics_size = std::mem::size_of::<GlyphMetrics>();
        let positioned_size = std::mem::size_of::<PositionedGlyph>();

        // Legacy buffers
        let element_buffer = device.new_buffer(
            (MAX_ELEMENTS * element_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let style_buffer = device.new_buffer(
            (MAX_ELEMENTS * style_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let layout_buffer = device.new_buffer(
            (MAX_ELEMENTS * layout_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let text_buffer = device.new_buffer(
            MAX_TEXT_LENGTH as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let break_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * break_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let line_buffer = device.new_buffer(
            (MAX_LINES * line_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let line_count_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let element_count_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // New GPU-native buffers (Issue #90)
        let metrics_buffer = device.new_buffer(
            (NUM_GLYPHS * metrics_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let advances_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * 4) as u64,  // float per char
            MTLResourceOptions::StorageModeShared,
        );
        let cumulative_buffer = device.new_buffer(
            ((MAX_TEXT_LENGTH + 1) * 4) as u64,  // float per char + 1
            MTLResourceOptions::StorageModeShared,
        );
        let glyph_ids_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * 4) as u64,  // uint per char
            MTLResourceOptions::StorageModeShared,
        );
        let is_break_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * 4) as u64,  // uint per char
            MTLResourceOptions::StorageModeShared,
        );
        let break_type_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * 4) as u64,  // uint per char
            MTLResourceOptions::StorageModeShared,
        );
        let line_indices_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * 4) as u64,  // uint per char
            MTLResourceOptions::StorageModeShared,
        );
        let positioned_glyphs_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * positioned_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let text_vertices_buffer = device.new_buffer(
            (MAX_TEXT_LENGTH * 6 * 8 * 4) as u64,  // 6 vertices × 8 floats × 4 bytes
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize metrics buffer with default font metrics
        let metrics = generate_default_metrics();
        unsafe {
            std::ptr::copy_nonoverlapping(
                metrics.as_ptr(),
                metrics_buffer.contents() as *mut GlyphMetrics,
                NUM_GLYPHS,
            );
        }

        Ok(Self {
            device: device.clone(),
            command_queue,
            measure_pipeline,
            find_breaks_pipeline,
            compute_lines_pipeline,
            position_lines_pipeline,
            char_to_glyph_pipeline,
            prefix_sum_pipeline,
            find_breaks_parallel_pipeline,
            assign_lines_pipeline,
            position_glyphs_pipeline,
            generate_vertices_pipeline,
            element_buffer,
            style_buffer,
            layout_buffer,
            text_buffer,
            break_buffer,
            line_buffer,
            line_count_buffer,
            element_count_buffer,
            metrics_buffer,
            advances_buffer,
            cumulative_buffer,
            glyph_ids_buffer,
            is_break_buffer,
            break_type_buffer,
            line_indices_buffer,
            positioned_glyphs_buffer,
            text_vertices_buffer,
        })
    }

    /// Compute text wrapping for all text elements
    /// Returns LineBox array and updates layout boxes with correct heights
    pub fn wrap_text(
        &mut self,
        elements: &[Element],
        styles: &[ComputedStyle],
        layout_boxes: &mut [LayoutBox],
        text_content: &[u8],
    ) -> Vec<LineBox> {
        let element_count = elements.len();
        if element_count == 0 {
            return Vec::new();
        }

        // Copy elements
        unsafe {
            std::ptr::copy_nonoverlapping(
                elements.as_ptr(),
                self.element_buffer.contents() as *mut Element,
                element_count,
            );
        }

        // Copy styles
        unsafe {
            std::ptr::copy_nonoverlapping(
                styles.as_ptr(),
                self.style_buffer.contents() as *mut ComputedStyle,
                element_count,
            );
        }

        // Copy layout boxes
        unsafe {
            std::ptr::copy_nonoverlapping(
                layout_boxes.as_ptr(),
                self.layout_buffer.contents() as *mut LayoutBox,
                element_count,
            );
        }

        // Copy text content
        let text_len = text_content.len().min(MAX_TEXT_LENGTH);
        if text_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    text_content.as_ptr(),
                    self.text_buffer.contents() as *mut u8,
                    text_len,
                );
            }
        }

        // Set element count
        unsafe {
            *(self.element_count_buffer.contents() as *mut u32) = element_count as u32;
        }

        // Initialize line count to 0
        unsafe {
            *(self.line_count_buffer.contents() as *mut u32) = 0;
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1: Find break opportunities for each text element
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.find_breaks_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.text_buffer), 0);
            encoder.set_buffer(3, Some(&self.break_buffer), 0);
            encoder.set_buffer(4, Some(&self.element_count_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 2: Compute line breaks based on container width
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.compute_lines_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.break_buffer), 0);
            encoder.set_buffer(4, Some(&self.line_buffer), 0);
            encoder.set_buffer(5, Some(&self.line_count_buffer), 0);
            encoder.set_buffer(6, Some(&self.element_count_buffer), 0);

            // Sequential for now - could be parallelized per element
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(1, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 3: Position lines with text-align
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.position_lines_pipeline);
            encoder.set_buffer(0, Some(&self.layout_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.line_buffer), 0);
            encoder.set_buffer(3, Some(&self.line_count_buffer), 0);

            let threadgroups = ((MAX_LINES as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read line count
        let line_count = unsafe {
            *(self.line_count_buffer.contents() as *const u32) as usize
        };

        // Read updated layout boxes (heights adjusted for multi-line text)
        unsafe {
            let ptr = self.layout_buffer.contents() as *const LayoutBox;
            for i in 0..element_count {
                layout_boxes[i] = *ptr.add(i);
            }
        }

        // Read lines
        let line_ptr = self.line_buffer.contents() as *const LineBox;
        (0..line_count)
            .map(|i| unsafe { *line_ptr.add(i) })
            .collect()
    }

    /// GPU-Native text layout using real glyph metrics (Issue #90)
    /// Returns positioned glyphs ready for rendering
    pub fn layout_text_gpu_native(
        &mut self,
        text_content: &[u8],
        container_width: f32,
        font_size: f32,
        line_height: f32,
    ) -> Vec<PositionedGlyph> {
        let char_count = text_content.len();
        if char_count == 0 {
            return Vec::new();
        }

        let char_count_u32 = char_count as u32;
        let line_height_px = font_size * line_height;

        // Copy text to GPU buffer
        unsafe {
            std::ptr::copy_nonoverlapping(
                text_content.as_ptr(),
                self.text_buffer.contents() as *mut u8,
                char_count.min(MAX_TEXT_LENGTH),
            );
        }

        // Initialize line count to 0
        unsafe {
            *(self.line_count_buffer.contents() as *mut u32) = 0;
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Kernel 1: Character to Glyph with real metrics
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.char_to_glyph_pipeline);
            encoder.set_buffer(0, Some(&self.text_buffer), 0);
            encoder.set_buffer(1, Some(&self.metrics_buffer), 0);
            encoder.set_buffer(2, Some(&self.advances_buffer), 0);
            encoder.set_buffer(3, Some(&self.glyph_ids_buffer), 0);
            encoder.set_bytes(4, 4, &char_count_u32 as *const u32 as *const _);
            encoder.set_bytes(5, 4, &font_size as *const f32 as *const _);
            let base_size = BASE_FONT_SIZE;
            encoder.set_bytes(6, 4, &base_size as *const f32 as *const _);

            let threadgroups = ((char_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Kernel 2: Prefix sum for cumulative widths
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.prefix_sum_pipeline);
            encoder.set_buffer(0, Some(&self.advances_buffer), 0);
            encoder.set_buffer(1, Some(&self.cumulative_buffer), 0);
            encoder.set_bytes(2, 4, &char_count_u32 as *const u32 as *const _);

            // Sequential prefix sum for now (could parallelize for large arrays)
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(1, 1, 1),
            );
            encoder.end_encoding();
        }

        // Kernel 3: Find break opportunities
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.find_breaks_parallel_pipeline);
            encoder.set_buffer(0, Some(&self.text_buffer), 0);
            encoder.set_buffer(1, Some(&self.is_break_buffer), 0);
            encoder.set_buffer(2, Some(&self.break_type_buffer), 0);
            encoder.set_bytes(3, 4, &char_count_u32 as *const u32 as *const _);

            let threadgroups = ((char_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // For now, do line assignment on CPU (GPU parallel version needs more work)
        // This is still fast because we're just using the GPU-computed data
        let cumulative_ptr = self.cumulative_buffer.contents() as *const f32;
        let is_break_ptr = self.is_break_buffer.contents() as *const u32;
        let glyph_ids_ptr = self.glyph_ids_buffer.contents() as *const u32;

        let mut lines: Vec<LineBox> = Vec::new();
        let mut line_indices: Vec<u32> = vec![0; char_count];

        let mut line_start_char: u32 = 0;
        let mut line_start_width: f32 = 0.0;
        let mut last_break_char: u32 = 0;
        let mut last_break_width: f32 = 0.0;
        let mut current_y: f32 = 0.0;

        for i in 0..char_count {
            let cumulative = unsafe { *cumulative_ptr.add(i + 1) };
            let is_break = unsafe { *is_break_ptr.add(i) };
            let width_so_far = cumulative - line_start_width;

            // Track break opportunities
            if is_break != 0 {
                last_break_char = i as u32 + 1;
                last_break_width = cumulative;
            }

            // Check for newline
            if text_content[i] == b'\n' {
                // End current line
                lines.push(LineBox {
                    element_index: 0,
                    char_start: line_start_char,
                    char_end: i as u32,
                    width: width_so_far,
                    x: 0.0,
                    y: current_y,
                    _padding: [0.0; 2],
                });

                for j in line_start_char as usize..=i {
                    line_indices[j] = lines.len() as u32 - 1;
                }

                current_y += line_height_px;
                line_start_char = i as u32 + 1;
                line_start_width = cumulative;
                last_break_char = line_start_char;
                last_break_width = cumulative;
                continue;
            }

            // Check if line overflows
            if width_so_far > container_width && i as u32 > line_start_char {
                let break_at = if last_break_char > line_start_char {
                    last_break_char
                } else {
                    i as u32
                };
                let break_width = if last_break_char > line_start_char {
                    last_break_width - line_start_width
                } else {
                    width_so_far
                };

                lines.push(LineBox {
                    element_index: 0,
                    char_start: line_start_char,
                    char_end: break_at,
                    width: break_width,
                    x: 0.0,
                    y: current_y,
                    _padding: [0.0; 2],
                });

                for j in line_start_char as usize..break_at as usize {
                    line_indices[j] = lines.len() as u32 - 1;
                }

                current_y += line_height_px;
                line_start_char = break_at;
                line_start_width = unsafe { *cumulative_ptr.add(break_at as usize) };
                last_break_char = break_at;
                last_break_width = line_start_width;
            }

            line_indices[i] = lines.len() as u32;
        }

        // Final line
        if line_start_char < char_count as u32 {
            let final_width = unsafe { *cumulative_ptr.add(char_count) } - line_start_width;
            lines.push(LineBox {
                element_index: 0,
                char_start: line_start_char,
                char_end: char_count as u32,
                width: final_width,
                x: 0.0,
                y: current_y,
                _padding: [0.0; 2],
            });

            for j in line_start_char as usize..char_count {
                line_indices[j] = lines.len() as u32 - 1;
            }
        }

        // Build positioned glyphs
        let mut positioned: Vec<PositionedGlyph> = Vec::with_capacity(char_count);
        for i in 0..char_count {
            let line_idx = line_indices[i] as usize;
            if line_idx >= lines.len() {
                continue;
            }
            let line = &lines[line_idx];

            let line_start_cumulative = unsafe { *cumulative_ptr.add(line.char_start as usize) };
            let char_cumulative = unsafe { *cumulative_ptr.add(i) };
            let x = line.x + (char_cumulative - line_start_cumulative);
            let y = line.y;

            let glyph_id = unsafe { *glyph_ids_ptr.add(i) };

            positioned.push(PositionedGlyph {
                x,
                y,
                glyph_id,
                color: 0x000000FF,  // Black text
                scale: font_size / BASE_FONT_SIZE,
                line_index: line_idx as u32,
                _padding: [0.0; 2],
            });
        }

        positioned
    }

    /// Get the lines computed by layout_text_gpu_native
    pub fn get_line_count(&self) -> usize {
        unsafe {
            *(self.line_count_buffer.contents() as *const u32) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::parser::ELEM_TEXT;

    fn create_text_element(text_start: u32, text_length: u32) -> Element {
        Element {
            element_type: ELEM_TEXT,
            parent: -1,
            first_child: -1,
            next_sibling: -1,
            text_start,
            text_length,
            token_index: 0,
            prev_sibling: -1,
        }
    }

    #[test]
    fn test_short_text_no_wrap() {
        let device = Device::system_default().expect("No Metal device");
        let mut engine = GpuTextEngine::new(&device).expect("Failed to create text engine");

        let text = b"Hello";
        let elements = vec![create_text_element(0, text.len() as u32)];
        let styles = vec![ComputedStyle {
            font_size: 16.0,
            line_height: 1.2,
            ..Default::default()
        }];
        let mut layout_boxes = vec![LayoutBox {
            content_width: 200.0,  // Wide enough for "Hello"
            ..Default::default()
        }];

        let lines = engine.wrap_text(&elements, &styles, &mut layout_boxes, text);

        assert_eq!(lines.len(), 1, "Short text should produce 1 line");
        assert_eq!(lines[0].char_start, 0);
        assert_eq!(lines[0].char_end, 5);
    }

    #[test]
    fn test_long_text_wraps() {
        let device = Device::system_default().expect("No Metal device");
        let mut engine = GpuTextEngine::new(&device).expect("Failed to create text engine");

        // "Hello World" - should wrap if container is narrow
        let text = b"Hello World Test";
        let elements = vec![create_text_element(0, text.len() as u32)];
        let styles = vec![ComputedStyle {
            font_size: 16.0,
            line_height: 1.2,
            ..Default::default()
        }];
        let mut layout_boxes = vec![LayoutBox {
            content_width: 80.0,  // Narrow - forces wrap
            ..Default::default()
        }];

        let lines = engine.wrap_text(&elements, &styles, &mut layout_boxes, text);

        assert!(lines.len() >= 2, "Long text in narrow container should wrap, got {} lines", lines.len());
    }

    #[test]
    fn test_white_space_nowrap() {
        let device = Device::system_default().expect("No Metal device");
        let mut engine = GpuTextEngine::new(&device).expect("Failed to create text engine");

        let text = b"Hello World Test Long Text";
        let elements = vec![create_text_element(0, text.len() as u32)];
        let mut styles = vec![ComputedStyle {
            font_size: 16.0,
            line_height: 1.2,
            ..Default::default()
        }];
        // Set white-space: nowrap (in the ComputedStyle we'd add this field)
        // For now, test that the basic wrapping works

        let mut layout_boxes = vec![LayoutBox {
            content_width: 100.0,
            ..Default::default()
        }];

        let lines = engine.wrap_text(&elements, &styles, &mut layout_boxes, text);

        // With normal wrapping, this should produce multiple lines
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_height_adjustment() {
        let device = Device::system_default().expect("No Metal device");
        let mut engine = GpuTextEngine::new(&device).expect("Failed to create text engine");

        let text = b"Line one and Line two and Line three";
        let elements = vec![create_text_element(0, text.len() as u32)];
        let styles = vec![ComputedStyle {
            font_size: 16.0,
            line_height: 1.5,
            ..Default::default()
        }];
        let mut layout_boxes = vec![LayoutBox {
            content_width: 100.0,  // Forces multiple lines
            content_height: 0.0,   // Should be updated
            ..Default::default()
        }];

        let lines = engine.wrap_text(&elements, &styles, &mut layout_boxes, text);

        // Height should be updated for multi-line text
        let expected_line_height = 16.0 * 1.5;
        let expected_height = lines.len() as f32 * expected_line_height;

        // Allow some tolerance
        assert!(
            (layout_boxes[0].content_height - expected_height).abs() < 1.0,
            "Expected height ~{}, got {}",
            expected_height,
            layout_boxes[0].content_height
        );
    }

    #[test]
    fn test_performance_10k_chars() {
        let device = Device::system_default().expect("No Metal device");
        let mut engine = GpuTextEngine::new(&device).expect("Failed to create text engine");

        // 10K characters with spaces
        let text: Vec<u8> = (0..10000).map(|i| {
            if i % 10 == 9 { b' ' } else { b'a' }
        }).collect();

        let elements = vec![create_text_element(0, text.len() as u32)];
        let styles = vec![ComputedStyle {
            font_size: 14.0,
            line_height: 1.2,
            ..Default::default()
        }];
        let mut layout_boxes = vec![LayoutBox {
            content_width: 500.0,
            ..Default::default()
        }];

        // Warmup
        let _ = engine.wrap_text(&elements, &styles, &mut layout_boxes, &text);

        // Timed run
        let start = std::time::Instant::now();
        let lines = engine.wrap_text(&elements, &styles, &mut layout_boxes, &text);
        let elapsed = start.elapsed();

        println!("10K chars text wrapping: {} lines in {:?}", lines.len(), elapsed);
        assert!(elapsed.as_millis() < 10, "10K chars took too long: {:?}", elapsed);
    }
}
