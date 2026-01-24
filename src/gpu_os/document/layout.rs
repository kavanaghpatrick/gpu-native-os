//! GPU-Accelerated Layout Engine
//!
//! Pass 4 of the document pipeline: computes element positions and dimensions
//! based on computed styles using CSS box model and flexbox layout.

use metal::*;
use super::parser::Element;
use super::style::ComputedStyle;

const THREAD_COUNT: u64 = 1024;
const MAX_ELEMENTS: usize = 65536;

/// Layout box containing position and dimensions for an element
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LayoutBox {
    /// X position (relative to parent, then absolute after finalize)
    pub x: f32,
    /// Y position (relative to parent, then absolute after finalize)
    pub y: f32,
    /// Total width (border box)
    pub width: f32,
    /// Total height (border box)
    pub height: f32,
    /// Content box X (absolute after finalize)
    pub content_x: f32,
    /// Content box Y (absolute after finalize)
    pub content_y: f32,
    /// Content box width
    pub content_width: f32,
    /// Content box height
    pub content_height: f32,
    /// Scroll width (for overflow)
    pub scroll_width: f32,
    /// Scroll height (for overflow)
    pub scroll_height: f32,
    /// Padding for GPU alignment
    pub _padding: [f32; 6],
}

/// Viewport dimensions
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub width: f32,
    pub height: f32,
    /// Padding for GPU alignment
    pub _padding: [f32; 2],
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            _padding: [0.0; 2],
        }
    }
}

const LAYOUT_SHADER: &str = include_str!("layout.metal");

/// GPU-accelerated layout engine
pub struct GpuLayoutEngine {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    // Original pipelines (kept for compatibility)
    intrinsic_pipeline: ComputePipelineState,
    block_layout_pipeline: ComputePipelineState,
    #[allow(dead_code)]
    children_pipeline: ComputePipelineState,  // Legacy - kept for fallback
    #[allow(dead_code)]
    finalize_pipeline: ComputePipelineState,  // Legacy - kept for fallback
    // New level-parallel pipelines (Issue #89)
    compute_depths_pipeline: ComputePipelineState,
    sum_heights_pipeline: ComputePipelineState,
    position_siblings_pipeline: ComputePipelineState,
    finalize_level_pipeline: ComputePipelineState,
    // Buffers
    element_buffer: Buffer,
    style_buffer: Buffer,
    layout_buffer: Buffer,
    element_count_buffer: Buffer,
    viewport_buffer: Buffer,
    // New buffers for level-parallel layout
    depth_buffer: Buffer,
    max_depth_buffer: Buffer,
    current_level_buffer: Buffer,
}

impl GpuLayoutEngine {
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(LAYOUT_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile layout shader: {}", e))?;

        // Original pipelines
        let intrinsic_fn = library
            .get_function("compute_intrinsic_sizes", None)
            .map_err(|e| format!("Failed to get intrinsic function: {}", e))?;
        let block_layout_fn = library
            .get_function("compute_block_layout", None)
            .map_err(|e| format!("Failed to get block_layout function: {}", e))?;
        let children_fn = library
            .get_function("layout_children_sequential", None)
            .map_err(|e| format!("Failed to get children function: {}", e))?;
        let finalize_fn = library
            .get_function("finalize_positions", None)
            .map_err(|e| format!("Failed to get finalize function: {}", e))?;

        // New level-parallel pipelines (Issue #89)
        let compute_depths_fn = library
            .get_function("layout_compute_depths", None)
            .map_err(|e| format!("Failed to get compute_depths function: {}", e))?;
        let sum_heights_fn = library
            .get_function("layout_sum_heights", None)
            .map_err(|e| format!("Failed to get sum_heights function: {}", e))?;
        let position_siblings_fn = library
            .get_function("layout_position_siblings", None)
            .map_err(|e| format!("Failed to get position_siblings function: {}", e))?;
        let finalize_level_fn = library
            .get_function("layout_finalize_level", None)
            .map_err(|e| format!("Failed to get finalize_level function: {}", e))?;

        let intrinsic_pipeline = device
            .new_compute_pipeline_state_with_function(&intrinsic_fn)
            .map_err(|e| format!("Failed to create intrinsic pipeline: {}", e))?;
        let block_layout_pipeline = device
            .new_compute_pipeline_state_with_function(&block_layout_fn)
            .map_err(|e| format!("Failed to create block_layout pipeline: {}", e))?;
        let children_pipeline = device
            .new_compute_pipeline_state_with_function(&children_fn)
            .map_err(|e| format!("Failed to create children pipeline: {}", e))?;
        let finalize_pipeline = device
            .new_compute_pipeline_state_with_function(&finalize_fn)
            .map_err(|e| format!("Failed to create finalize pipeline: {}", e))?;

        let compute_depths_pipeline = device
            .new_compute_pipeline_state_with_function(&compute_depths_fn)
            .map_err(|e| format!("Failed to create compute_depths pipeline: {}", e))?;
        let sum_heights_pipeline = device
            .new_compute_pipeline_state_with_function(&sum_heights_fn)
            .map_err(|e| format!("Failed to create sum_heights pipeline: {}", e))?;
        let position_siblings_pipeline = device
            .new_compute_pipeline_state_with_function(&position_siblings_fn)
            .map_err(|e| format!("Failed to create position_siblings pipeline: {}", e))?;
        let finalize_level_pipeline = device
            .new_compute_pipeline_state_with_function(&finalize_level_fn)
            .map_err(|e| format!("Failed to create finalize_level pipeline: {}", e))?;

        let element_size = std::mem::size_of::<Element>();
        let style_size = std::mem::size_of::<ComputedStyle>();
        let layout_size = std::mem::size_of::<LayoutBox>();

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
        let element_count_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let viewport_buffer = device.new_buffer(
            std::mem::size_of::<Viewport>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // New buffers for level-parallel layout
        let depth_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let max_depth_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let current_level_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device: device.clone(),
            command_queue,
            intrinsic_pipeline,
            block_layout_pipeline,
            children_pipeline,
            finalize_pipeline,
            compute_depths_pipeline,
            sum_heights_pipeline,
            position_siblings_pipeline,
            finalize_level_pipeline,
            element_buffer,
            style_buffer,
            layout_buffer,
            element_count_buffer,
            viewport_buffer,
            depth_buffer,
            max_depth_buffer,
            current_level_buffer,
        })
    }

    /// Compute layout for all elements using level-parallel algorithm
    pub fn compute_layout(
        &mut self,
        elements: &[Element],
        styles: &[ComputedStyle],
        viewport: Viewport,
    ) -> Vec<LayoutBox> {
        let element_count = elements.len();
        assert!(element_count <= MAX_ELEMENTS, "Too many elements");
        assert_eq!(elements.len(), styles.len(), "Elements and styles must match");

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


        // Set element count
        unsafe {
            *(self.element_count_buffer.contents() as *mut u32) = element_count as u32;
        }

        // Set viewport
        unsafe {
            *(self.viewport_buffer.contents() as *mut Viewport) = viewport;
        }

        // Initialize layout boxes
        unsafe {
            let ptr = self.layout_buffer.contents() as *mut LayoutBox;
            for i in 0..element_count {
                *ptr.add(i) = LayoutBox::default();
            }
        }

        // Initialize max_depth to 0
        unsafe {
            *(self.max_depth_buffer.contents() as *mut u32) = 0;
        }

        let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);

        // Pass 1: Compute intrinsic sizes (all threads)
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.intrinsic_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(4, Some(&self.viewport_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }


        // Pass 2: Compute block layout (all threads)
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.block_layout_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(4, Some(&self.viewport_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Pass 3a: Compute tree depths (all threads)
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.compute_depths_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.depth_buffer), 0);
            encoder.set_buffer(2, Some(&self.max_depth_buffer), 0);
            encoder.set_buffer(3, Some(&self.element_count_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Read max_depth from depth buffer (atomic in GPU doesn't seem reliable)
        let max_depth = unsafe {
            let depth_ptr = self.depth_buffer.contents() as *const u32;
            (0..element_count).map(|i| *depth_ptr.add(i)).max().unwrap_or(0)
        };

        // Pass 3b: Sum child heights (bottom-up, level by level)
        for level in (0..=max_depth).rev() {
            unsafe {
                *(self.current_level_buffer.contents() as *mut u32) = level;
            }
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.sum_heights_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.depth_buffer), 0);
            encoder.set_buffer(4, Some(&self.current_level_buffer), 0);
            encoder.set_buffer(5, Some(&self.element_count_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Pass 3c + 3d: Position siblings then finalize, level by level (top-down)
        // Must interleave: finalize for level N needs level N-1's absolute positions
        for level in 0..=max_depth {
            unsafe {
                *(self.current_level_buffer.contents() as *mut u32) = level;
            }

            // Position siblings at this level (relative coords)
            {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.position_siblings_pipeline);
                encoder.set_buffer(0, Some(&self.element_buffer), 0);
                encoder.set_buffer(1, Some(&self.style_buffer), 0);
                encoder.set_buffer(2, Some(&self.layout_buffer), 0);
                encoder.set_buffer(3, Some(&self.depth_buffer), 0);
                encoder.set_buffer(4, Some(&self.current_level_buffer), 0);
                encoder.set_buffer(5, Some(&self.element_count_buffer), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(THREAD_COUNT, 1, 1),
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            // Finalize absolute positions at this level
            {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.finalize_level_pipeline);
                encoder.set_buffer(0, Some(&self.element_buffer), 0);
                encoder.set_buffer(1, Some(&self.style_buffer), 0);
                encoder.set_buffer(2, Some(&self.layout_buffer), 0);
                encoder.set_buffer(3, Some(&self.depth_buffer), 0);
                encoder.set_buffer(4, Some(&self.current_level_buffer), 0);
                encoder.set_buffer(5, Some(&self.element_count_buffer), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(THREAD_COUNT, 1, 1),
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
        }

        // Read results
        let layout_ptr = self.layout_buffer.contents() as *const LayoutBox;
        (0..element_count)
            .map(|i| unsafe { *layout_ptr.add(i) })
            .collect()
    }
}

// Display constants re-exported from style module
pub use super::style::{
    DISPLAY_NONE, DISPLAY_BLOCK, DISPLAY_INLINE, DISPLAY_FLEX, DISPLAY_INLINE_BLOCK,
    FLEX_ROW, FLEX_COLUMN,
    JUSTIFY_START, JUSTIFY_CENTER, JUSTIFY_END, JUSTIFY_SPACE_BETWEEN, JUSTIFY_SPACE_AROUND,
    ALIGN_START, ALIGN_CENTER, ALIGN_END, ALIGN_STRETCH,
};

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tokenizer::GpuTokenizer;
    use super::super::parser::GpuParser;
    use super::super::style::{GpuStyler, Stylesheet};

    fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine) {
        let device = Device::system_default().expect("No Metal device");
        let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
        let parser = GpuParser::new(&device).expect("Failed to create parser");
        let styler = GpuStyler::new(&device).expect("Failed to create styler");
        let layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");
        (tokenizer, parser, styler, layout)
    }

    fn process_html(html: &[u8], css: &str, viewport: Viewport) -> Vec<LayoutBox> {
        let (mut tokenizer, mut parser, mut styler, mut layout) = setup();

        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
        layout.compute_layout(&elements, &styles, viewport)
    }

    #[test]
    fn test_explicit_size() {
        let html = b"<div>text</div>";
        let css = "div { width: 100px; height: 50px; }";
        let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

        let boxes = process_html(html, css, viewport);

        assert!(boxes.len() >= 1, "Expected at least 1 element");
        let div_box = &boxes[0];
        assert!((div_box.width - 100.0).abs() < 0.01, "Expected width 100, got {}", div_box.width);
        assert!((div_box.height - 50.0).abs() < 0.01, "Expected height 50, got {}", div_box.height);
    }

    #[test]
    fn test_auto_width() {
        let html = b"<div>text</div>";
        let css = "div { height: 50px; }";  // No width specified
        let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

        let boxes = process_html(html, css, viewport);

        let div_box = &boxes[0];
        // Block element should fill parent width
        assert!((div_box.width - 800.0).abs() < 0.01, "Expected width 800, got {}", div_box.width);
    }

    #[test]
    fn test_padding() {
        let html = b"<div>text</div>";
        let css = "div { width: 100px; height: 50px; padding: 10px; }";
        let viewport = Viewport::default();

        let boxes = process_html(html, css, viewport);

        let div_box = &boxes[0];
        assert!((div_box.width - 100.0).abs() < 0.01);
        assert!((div_box.content_width - 80.0).abs() < 0.01, "Expected content_width 80, got {}", div_box.content_width);
        assert!((div_box.content_x - 10.0).abs() < 0.01, "Expected content_x 10, got {}", div_box.content_x);
    }

    #[test]
    fn test_margin() {
        let html = b"<div>text</div>";
        let css = "div { width: 100px; height: 50px; margin: 20px; }";
        let viewport = Viewport::default();

        let boxes = process_html(html, css, viewport);

        let div_box = &boxes[0];
        assert!((div_box.x - 20.0).abs() < 0.01, "Expected x 20, got {}", div_box.x);
        assert!((div_box.y - 20.0).abs() < 0.01, "Expected y 20, got {}", div_box.y);
    }

    #[test]
    fn test_nested_layout() {
        let html = b"<div><span>text</span></div>";
        let css = "div { width: 200px; height: 100px; padding: 10px; } span { width: 50px; height: 30px; }";
        let viewport = Viewport::default();

        let (mut tokenizer, mut parser, mut styler, mut layout) = setup();
        let tokens = tokenizer.tokenize(html);
        let (elements, _) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);

        println!("Elements ({}):", elements.len());
        for (i, e) in elements.iter().enumerate() {
            println!("  [{}] type={:2} parent={:2} first_child={:2} next={:2}",
                i, e.element_type, e.parent, e.first_child, e.next_sibling);
        }
        println!("Styles:");
        for (i, s) in styles.iter().enumerate() {
            println!("  [{}] w={:.0} h={:.0} p=[{:.0},{:.0},{:.0},{:.0}]",
                i, s.width, s.height, s.padding[0], s.padding[1], s.padding[2], s.padding[3]);
        }

        let boxes = layout.compute_layout(&elements, &styles, viewport);

        println!("Layout boxes:");
        for (i, b) in boxes.iter().enumerate() {
            println!("  [{}] x={:.1} y={:.1} w={:.1} h={:.1} cx={:.1} cy={:.1}",
                i, b.x, b.y, b.width, b.height, b.content_x, b.content_y);
        }

        // div at index 0, span at index 1, text at index 2
        assert!(boxes.len() >= 2, "Expected at least 2 elements");
        let div_box = &boxes[0];
        let span_box = &boxes[1];

        // Span should be positioned within div's content box
        assert!(span_box.x >= div_box.content_x, "Span x {} should be >= div content_x {}", span_box.x, div_box.content_x);
    }

    #[test]
    fn test_flex_row() {
        let html = b"<div><span>A</span><span>B</span></div>";
        let css = "div { display: flex; width: 200px; height: 50px; } span { width: 50px; height: 30px; }";
        let viewport = Viewport::default();

        let boxes = process_html(html, css, viewport);

        // Elements: div(0), span(1), text(2), span(3), text(4)
        assert!(boxes.len() >= 4, "Expected at least 4 elements, got {}", boxes.len());

        // First span should be at start
        // Second span should be to the right of first
        let span1 = &boxes[1];
        let span2 = &boxes[3];
        assert!(span2.x > span1.x, "Second span x {} should be > first span x {}", span2.x, span1.x);
    }

    #[test]
    fn test_flex_justify_center() {
        let html = b"<div><span>A</span></div>";
        let css = "div { display: flex; justify-content: center; width: 100px; height: 50px; } span { width: 30px; height: 20px; }";
        let viewport = Viewport::default();

        let boxes = process_html(html, css, viewport);

        let div_box = &boxes[0];
        let span_box = &boxes[1];

        // Span should be centered: x = (100 - 30) / 2 = 35
        let expected_x = div_box.content_x + (div_box.content_width - span_box.width) / 2.0;
        assert!(
            (span_box.x - expected_x).abs() < 1.0,
            "Expected span at ~{}, got {}", expected_x, span_box.x
        );
    }

    #[test]
    fn test_flex_column() {
        let html = b"<div><span>A</span><span>B</span></div>";
        let css = "div { display: flex; flex-direction: column; width: 100px; height: 100px; } span { width: 50px; height: 30px; }";
        let viewport = Viewport::default();

        let boxes = process_html(html, css, viewport);

        let span1 = &boxes[1];
        let span2 = &boxes[3];

        // In column layout, second span should be below first
        assert!(span2.y > span1.y, "Second span y {} should be > first span y {}", span2.y, span1.y);
    }

    #[test]
    fn test_display_none() {
        let html = b"<div>visible</div><div>hidden</div>";
        let css = "div:nth-child(2) { display: none; }";  // Won't match but test structure
        let viewport = Viewport::default();

        // Just ensure no crash with display none
        let boxes = process_html(html, css, viewport);
        assert!(!boxes.is_empty());
    }

    #[test]
    fn test_performance_1k_elements() {
        let device = Device::system_default().expect("No Metal device");
        let mut layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");

        // Generate 1K elements
        let element_count = 1000;
        let mut elements = Vec::with_capacity(element_count);
        let mut styles = Vec::with_capacity(element_count);

        // Root element
        elements.push(Element {
            element_type: 1,  // div
            parent: -1,
            first_child: 1,
            next_sibling: -1,
            text_start: 0,
            text_length: 0,
            token_index: 0,
            _padding: 0,
        });
        styles.push(ComputedStyle {
            display: DISPLAY_BLOCK,
            width: 800.0,
            height: 0.0,  // auto
            ..Default::default()
        });

        // Child elements
        for i in 1..element_count {
            elements.push(Element {
                element_type: 1,  // div
                parent: 0,
                first_child: -1,
                next_sibling: if i < element_count - 1 { (i + 1) as i32 } else { -1 },
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            });
            styles.push(ComputedStyle {
                display: DISPLAY_BLOCK,
                width: 0.0,  // auto
                height: 50.0,
                ..Default::default()
            });
        }
        elements[0].first_child = 1;

        let viewport = Viewport::default();

        // Warmup
        let _ = layout.compute_layout(&elements, &styles, viewport);

        // Timed run
        let start = std::time::Instant::now();
        let boxes = layout.compute_layout(&elements, &styles, viewport);
        let elapsed = start.elapsed();

        println!("~1K elements layout: {} boxes in {:?}", boxes.len(), elapsed);

        assert_eq!(boxes.len(), element_count);
        assert!(elapsed.as_millis() < 10, "1K layout took too long: {:?}", elapsed);
    }

    #[test]
    fn test_performance_5k_elements() {
        let device = Device::system_default().expect("No Metal device");
        let mut layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");

        // Generate 5K elements in a flat structure
        let element_count = 5000;
        let mut elements = Vec::with_capacity(element_count);
        let mut styles = Vec::with_capacity(element_count);

        // Root element
        elements.push(Element {
            element_type: 1,
            parent: -1,
            first_child: 1,
            next_sibling: -1,
            text_start: 0,
            text_length: 0,
            token_index: 0,
            _padding: 0,
        });
        styles.push(ComputedStyle {
            display: DISPLAY_FLEX,
            flex_direction: FLEX_COLUMN,
            width: 800.0,
            height: 0.0,
            ..Default::default()
        });

        // Child elements
        for i in 1..element_count {
            elements.push(Element {
                element_type: 1,
                parent: 0,
                first_child: -1,
                next_sibling: if i < element_count - 1 { (i + 1) as i32 } else { -1 },
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            });
            styles.push(ComputedStyle {
                display: DISPLAY_BLOCK,
                width: 0.0,
                height: 30.0,
                ..Default::default()
            });
        }

        let viewport = Viewport::default();

        // Warmup
        let _ = layout.compute_layout(&elements, &styles, viewport);

        // Timed run
        let start = std::time::Instant::now();
        let boxes = layout.compute_layout(&elements, &styles, viewport);
        let elapsed = start.elapsed();

        println!("~5K elements layout: {} boxes in {:?}", boxes.len(), elapsed);

        assert_eq!(boxes.len(), element_count);
        assert!(elapsed.as_millis() < 50, "5K layout took too long: {:?}", elapsed);
    }
}
