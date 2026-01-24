//! GPU-Accelerated Text Wrapping & Line Breaking
//!
//! Issue #60: Computes line breaks and positions text across multiple lines
//! using Metal compute shaders for GPU-native text layout.

use metal::*;
use super::parser::Element;
use super::style::ComputedStyle;
use super::layout::LayoutBox;

const THREAD_COUNT: u64 = 1024;
const MAX_ELEMENTS: usize = 65536;
const MAX_LINES: usize = 65536;  // Max lines across all text elements
const MAX_TEXT_LENGTH: usize = 1024 * 1024;  // 1MB text buffer

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

/// Text-align CSS property values
pub const TEXT_ALIGN_LEFT: u32 = 0;
pub const TEXT_ALIGN_CENTER: u32 = 1;
pub const TEXT_ALIGN_RIGHT: u32 = 2;
pub const TEXT_ALIGN_JUSTIFY: u32 = 3;

const TEXT_SHADER: &str = include_str!("text.metal");

/// GPU-accelerated text wrapping engine
pub struct GpuTextEngine {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    measure_pipeline: ComputePipelineState,
    find_breaks_pipeline: ComputePipelineState,
    compute_lines_pipeline: ComputePipelineState,
    position_lines_pipeline: ComputePipelineState,
    element_buffer: Buffer,
    style_buffer: Buffer,
    layout_buffer: Buffer,
    text_buffer: Buffer,
    break_buffer: Buffer,
    line_buffer: Buffer,
    line_count_buffer: Buffer,
    element_count_buffer: Buffer,
}

impl GpuTextEngine {
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(TEXT_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile text shader: {}", e))?;

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

        let element_size = std::mem::size_of::<Element>();
        let style_size = std::mem::size_of::<ComputedStyle>();
        let layout_size = std::mem::size_of::<LayoutBox>();
        let break_size = std::mem::size_of::<BreakOpportunity>();
        let line_size = std::mem::size_of::<LineBox>();

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
        // One break opportunity per character
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

        Ok(Self {
            device: device.clone(),
            command_queue,
            measure_pipeline,
            find_breaks_pipeline,
            compute_lines_pipeline,
            position_lines_pipeline,
            element_buffer,
            style_buffer,
            layout_buffer,
            text_buffer,
            break_buffer,
            line_buffer,
            line_count_buffer,
            element_count_buffer,
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
            _padding: 0,
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
