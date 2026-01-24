//! GPU-Accelerated Layout Engine
//!
//! Pass 4 of the document pipeline: computes element positions and dimensions
//! based on computed styles using CSS box model and flexbox layout.

use metal::*;
use super::parser::Element;
use super::style::ComputedStyle;

const THREAD_COUNT: u64 = 1024;
const MAX_ELEMENTS: usize = 65536;

/// Issue #128: Cumulative height info for O(1) sibling positioning
/// Must match Metal struct layout
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CumulativeInfo {
    pub y_offset: f32,           // Cumulative Y position
    pub prev_margin_bottom: f32, // Previous sibling's bottom margin (for margin collapsing)
    pub flags: u32,              // Bit 0: has_visible_sibling
    pub _padding: f32,
}

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
    #[allow(dead_code)]
    compute_depths_pipeline: ComputePipelineState,  // Legacy O(depth) - kept for fallback
    sum_heights_pipeline: ComputePipelineState,
    position_siblings_pipeline: ComputePipelineState,
    finalize_level_pipeline: ComputePipelineState,
    // Width propagation and text height calculation (fixes whitespace issue)
    propagate_widths_text_pipeline: ComputePipelineState,
    // O(1) depth buffer pipelines (Issue #130)
    init_depths_pipeline: ComputePipelineState,
    propagate_depths_pipeline: ComputePipelineState,
    // Issue #128: O(1) cumulative heights pipeline
    compute_cumulative_heights_pipeline: ComputePipelineState,
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
    // O(1) depth buffer support (Issue #130)
    changed_buffer: Buffer,  // Atomic counter for level propagation
    depths_valid: bool,      // Cache invalidation flag
    // Text buffer for wrapped height calculation
    text_buffer: Buffer,
    // Issue #128: O(1) cumulative heights buffer
    cumulative_buffer: Buffer,
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
        let propagate_widths_text_fn = library
            .get_function("propagate_widths_and_text", None)
            .map_err(|e| format!("Failed to get propagate_widths_and_text function: {}", e))?;

        // O(1) depth buffer pipelines (Issue #130)
        let init_depths_fn = library
            .get_function("init_depths", None)
            .map_err(|e| format!("Failed to get init_depths function: {}", e))?;
        let propagate_depths_fn = library
            .get_function("propagate_depths", None)
            .map_err(|e| format!("Failed to get propagate_depths function: {}", e))?;

        // Issue #128: O(1) cumulative heights pipeline
        let compute_cumulative_heights_fn = library
            .get_function("compute_cumulative_heights", None)
            .map_err(|e| format!("Failed to get compute_cumulative_heights function: {}", e))?;

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
        let propagate_widths_text_pipeline = device
            .new_compute_pipeline_state_with_function(&propagate_widths_text_fn)
            .map_err(|e| format!("Failed to create propagate_widths_text pipeline: {}", e))?;

        // O(1) depth buffer pipelines (Issue #130)
        let init_depths_pipeline = device
            .new_compute_pipeline_state_with_function(&init_depths_fn)
            .map_err(|e| format!("Failed to create init_depths pipeline: {}", e))?;
        let propagate_depths_pipeline = device
            .new_compute_pipeline_state_with_function(&propagate_depths_fn)
            .map_err(|e| format!("Failed to create propagate_depths pipeline: {}", e))?;

        // Issue #128: O(1) cumulative heights pipeline
        let compute_cumulative_heights_pipeline = device
            .new_compute_pipeline_state_with_function(&compute_cumulative_heights_fn)
            .map_err(|e| format!("Failed to create compute_cumulative_heights pipeline: {}", e))?;

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

        // Text buffer for wrapped height calculation (512KB max)
        let text_buffer = device.new_buffer(
            512 * 1024,
            MTLResourceOptions::StorageModeShared,
        );

        // Changed counter for level-parallel depth propagation (Issue #130)
        let changed_buffer = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Issue #128: Cumulative heights buffer for O(1) sibling positioning
        let cumulative_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<CumulativeInfo>()) as u64,
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
            propagate_widths_text_pipeline,
            init_depths_pipeline,
            propagate_depths_pipeline,
            compute_cumulative_heights_pipeline,
            element_buffer,
            style_buffer,
            layout_buffer,
            element_count_buffer,
            viewport_buffer,
            depth_buffer,
            max_depth_buffer,
            current_level_buffer,
            changed_buffer,
            depths_valid: false,
            text_buffer,
            cumulative_buffer,
        })
    }

    /// Compute layout for all elements using level-parallel algorithm
    pub fn compute_layout(
        &mut self,
        elements: &[Element],
        styles: &[ComputedStyle],
        text: &[u8],  // Text buffer for wrapped height calculation
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

        // Copy text buffer for wrapped height calculation
        if !text.is_empty() {
            let text_len = text.len().min(512 * 1024); // Cap at buffer size
            unsafe {
                std::ptr::copy_nonoverlapping(
                    text.as_ptr(),
                    self.text_buffer.contents() as *mut u8,
                    text_len,
                );
            }
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
        // Now includes text buffer for wrapped height calculation
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.block_layout_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(4, Some(&self.viewport_buffer), 0);
            encoder.set_buffer(5, Some(&self.text_buffer), 0);  // Text for wrapped height
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Pass 3a: Compute tree depths using O(1) level-parallel algorithm (Issue #130)
        // Phase 1: Initialize depths (roots = 0, others = 0xFFFFFFFF)
        {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.init_depths_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.depth_buffer), 0);
            encoder.set_buffer(2, Some(&self.element_count_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Phase 2: Propagate depths level by level
        let mut level = 0u32;
        let max_depth = loop {
            // Reset changed counter
            unsafe {
                *(self.changed_buffer.contents() as *mut u32) = 0;
            }

            // Set current level
            unsafe {
                *(self.current_level_buffer.contents() as *mut u32) = level;
            }

            // Dispatch propagation for this level
            {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.propagate_depths_pipeline);
                encoder.set_buffer(0, Some(&self.element_buffer), 0);
                encoder.set_buffer(1, Some(&self.depth_buffer), 0);
                encoder.set_buffer(2, Some(&self.current_level_buffer), 0);
                encoder.set_buffer(3, Some(&self.element_count_buffer), 0);
                encoder.set_buffer(4, Some(&self.changed_buffer), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(THREAD_COUNT, 1, 1),
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            // Check if any elements were updated
            let changed = unsafe { *(self.changed_buffer.contents() as *const u32) };
            if changed == 0 {
                break level;  // All depths computed
            }

            level += 1;
            if level > 256 {
                panic!("Tree depth exceeds maximum (256)");
            }
        };

        // Mark depths as valid (for cache purposes)
        self.depths_valid = true;

        // Pass NEW: Propagate widths and calculate text heights (TOP-DOWN by level)
        // This fixes the whitespace issue by calculating text heights AFTER parent widths are known
        for level in 0..=max_depth {
            unsafe {
                *(self.current_level_buffer.contents() as *mut u32) = level;
            }
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.propagate_widths_text_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.layout_buffer), 0);
            encoder.set_buffer(3, Some(&self.depth_buffer), 0);
            encoder.set_buffer(4, Some(&self.current_level_buffer), 0);
            encoder.set_buffer(5, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(6, Some(&self.viewport_buffer), 0);
            encoder.set_buffer(7, Some(&self.text_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

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
        // Issue #128: First compute cumulative heights, then use O(1) lookup in position_siblings
        for level in 0..=max_depth {
            unsafe {
                *(self.current_level_buffer.contents() as *mut u32) = level;
            }

            // Issue #128: Compute cumulative heights for this level (O(1) per element)
            {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.compute_cumulative_heights_pipeline);
                encoder.set_buffer(0, Some(&self.element_buffer), 0);
                encoder.set_buffer(1, Some(&self.style_buffer), 0);
                encoder.set_buffer(2, Some(&self.layout_buffer), 0);
                encoder.set_buffer(3, Some(&self.cumulative_buffer), 0);
                encoder.set_buffer(4, Some(&self.depth_buffer), 0);
                encoder.set_buffer(5, Some(&self.current_level_buffer), 0);
                encoder.set_buffer(6, Some(&self.element_count_buffer), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(threadgroups, 1, 1),
                    MTLSize::new(THREAD_COUNT, 1, 1),
                );
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            // Position siblings at this level using O(1) cumulative lookup
            {
                let command_buffer = self.command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.position_siblings_pipeline);
                encoder.set_buffer(0, Some(&self.element_buffer), 0);
                encoder.set_buffer(1, Some(&self.style_buffer), 0);
                encoder.set_buffer(2, Some(&self.layout_buffer), 0);
                encoder.set_buffer(3, Some(&self.depth_buffer), 0);
                encoder.set_buffer(4, Some(&self.cumulative_buffer), 0);  // Issue #128: O(1) lookup
                encoder.set_buffer(5, Some(&self.current_level_buffer), 0);
                encoder.set_buffer(6, Some(&self.element_count_buffer), 0);
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

    /// Invalidate the depth buffer cache.
    /// Call this when the tree structure changes (elements added/removed/reparented).
    #[allow(dead_code)]
    pub fn invalidate_depths(&mut self) {
        self.depths_valid = false;
    }

    /// Check if depth buffer is valid (for debugging/testing)
    #[allow(dead_code)]
    pub fn depths_valid(&self) -> bool {
        self.depths_valid
    }

    /// Get a copy of the depth buffer (for testing)
    #[allow(dead_code)]
    pub fn get_depths(&self, count: usize) -> Vec<u32> {
        let depth_ptr = self.depth_buffer.contents() as *const u32;
        (0..count)
            .map(|i| unsafe { *depth_ptr.add(i) })
            .collect()
    }
}

// Display constants re-exported from style module

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tokenizer::GpuTokenizer;
    use super::super::parser::GpuParser;
    use super::super::style::{
        GpuStyler, Stylesheet,
        DISPLAY_BLOCK, DISPLAY_FLEX, FLEX_COLUMN,
    };

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
        layout.compute_layout(&elements, &styles, html, viewport)
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

        let boxes = layout.compute_layout(&elements, &styles, html, viewport);

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
            prev_sibling: -1,  // Issue #128
            text_start: 0,
            text_length: 0,
            token_index: 0,
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
                prev_sibling: if i > 1 { (i - 1) as i32 } else { -1 },  // Issue #128
                text_start: 0,
                text_length: 0,
                token_index: 0,
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
        let _ = layout.compute_layout(&elements, &styles, &[], viewport);

        // Timed run
        let start = std::time::Instant::now();
        let boxes = layout.compute_layout(&elements, &styles, &[], viewport);
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
            prev_sibling: -1,  // Issue #128
            text_start: 0,
            text_length: 0,
            token_index: 0,
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
                prev_sibling: if i > 1 { (i - 1) as i32 } else { -1 },  // Issue #128
                text_start: 0,
                text_length: 0,
                token_index: 0,
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
        let _ = layout.compute_layout(&elements, &styles, &[], viewport);

        // Timed run
        let start = std::time::Instant::now();
        let boxes = layout.compute_layout(&elements, &styles, &[], viewport);
        let elapsed = start.elapsed();

        println!("~5K elements layout: {} boxes in {:?}", boxes.len(), elapsed);

        assert_eq!(boxes.len(), element_count);
        assert!(elapsed.as_millis() < 50, "5K layout took too long: {:?}", elapsed);
    }
}
