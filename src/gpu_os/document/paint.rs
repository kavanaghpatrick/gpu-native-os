//! GPU-Accelerated Paint (Vertex Generation)
//!
//! Pass 5 of the document pipeline: converts layout boxes and computed styles
//! into GPU vertices ready for rendering.

use metal::*;
use super::parser::Element;
use super::style::ComputedStyle;
use super::layout::{LayoutBox, Viewport};

const THREAD_COUNT: u64 = 1024;
const MAX_ELEMENTS: usize = 65536;
const MAX_VERTICES: usize = MAX_ELEMENTS * 64;  // Up to 64 vertices per element

/// Vertex flags
pub const FLAG_BACKGROUND: u32 = 1;
pub const FLAG_BORDER: u32 = 2;
pub const FLAG_TEXT: u32 = 4;

/// Paint vertex for GPU rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PaintVertex {
    /// Screen position in NDC [-1, 1]
    pub position: [f32; 2],
    /// Texture coordinate for text/images
    pub tex_coord: [f32; 2],
    /// RGBA color
    pub color: [f32; 4],
    /// Rendering flags
    pub flags: u32,
    _padding: [u32; 3],
}

/// Paint command for batched rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PaintCommand {
    /// Element index this command belongs to
    pub element_index: u32,
    /// Start index in vertex buffer
    pub vertex_start: u32,
    /// Number of vertices
    pub vertex_count: u32,
    /// Texture ID (0 for solid color)
    pub texture_id: u32,
}

const PAINT_SHADER: &str = include_str!("paint.metal");

/// GPU-accelerated paint engine (vertex generation)
pub struct GpuPaintEngine {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    count_pipeline: ComputePipelineState,
    offset_pipeline: ComputePipelineState,
    background_pipeline: ComputePipelineState,
    border_pipeline: ComputePipelineState,
    text_pipeline: ComputePipelineState,
    element_buffer: Buffer,
    layout_buffer: Buffer,
    style_buffer: Buffer,
    text_buffer: Buffer,
    vertex_count_buffer: Buffer,
    vertex_offset_buffer: Buffer,
    vertex_buffer: Buffer,
    element_count_buffer: Buffer,
    viewport_buffer: Buffer,
}

impl GpuPaintEngine {
    pub fn new(device: &Device) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(PAINT_SHADER, &compile_options)
            .map_err(|e| format!("Failed to compile paint shader: {}", e))?;

        let count_fn = library
            .get_function("count_vertices", None)
            .map_err(|e| format!("Failed to get count function: {}", e))?;
        let offset_fn = library
            .get_function("compute_offsets", None)
            .map_err(|e| format!("Failed to get offset function: {}", e))?;
        let background_fn = library
            .get_function("generate_background_vertices", None)
            .map_err(|e| format!("Failed to get background function: {}", e))?;
        let border_fn = library
            .get_function("generate_border_vertices", None)
            .map_err(|e| format!("Failed to get border function: {}", e))?;
        let text_fn = library
            .get_function("generate_text_vertices", None)
            .map_err(|e| format!("Failed to get text function: {}", e))?;

        let count_pipeline = device
            .new_compute_pipeline_state_with_function(&count_fn)
            .map_err(|e| format!("Failed to create count pipeline: {}", e))?;
        let offset_pipeline = device
            .new_compute_pipeline_state_with_function(&offset_fn)
            .map_err(|e| format!("Failed to create offset pipeline: {}", e))?;
        let background_pipeline = device
            .new_compute_pipeline_state_with_function(&background_fn)
            .map_err(|e| format!("Failed to create background pipeline: {}", e))?;
        let border_pipeline = device
            .new_compute_pipeline_state_with_function(&border_fn)
            .map_err(|e| format!("Failed to create border pipeline: {}", e))?;
        let text_pipeline = device
            .new_compute_pipeline_state_with_function(&text_fn)
            .map_err(|e| format!("Failed to create text pipeline: {}", e))?;

        let element_size = std::mem::size_of::<Element>();
        let layout_size = std::mem::size_of::<LayoutBox>();
        let style_size = std::mem::size_of::<ComputedStyle>();
        let vertex_size = std::mem::size_of::<PaintVertex>();

        let element_buffer = device.new_buffer(
            (MAX_ELEMENTS * element_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let layout_buffer = device.new_buffer(
            (MAX_ELEMENTS * layout_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let style_buffer = device.new_buffer(
            (MAX_ELEMENTS * style_size) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let text_buffer = device.new_buffer(
            (1024 * 1024) as u64,  // 1MB text buffer
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_count_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_offset_buffer = device.new_buffer(
            (MAX_ELEMENTS * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vertex_buffer = device.new_buffer(
            (MAX_VERTICES * vertex_size) as u64,
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

        Ok(Self {
            device: device.clone(),
            command_queue,
            count_pipeline,
            offset_pipeline,
            background_pipeline,
            border_pipeline,
            text_pipeline,
            element_buffer,
            layout_buffer,
            style_buffer,
            text_buffer,
            vertex_count_buffer,
            vertex_offset_buffer,
            vertex_buffer,
            element_count_buffer,
            viewport_buffer,
        })
    }

    /// Generate vertices for all elements
    pub fn paint(
        &mut self,
        elements: &[Element],
        boxes: &[LayoutBox],
        styles: &[ComputedStyle],
        text_content: &[u8],
        viewport: Viewport,
    ) -> Vec<PaintVertex> {
        let element_count = elements.len();
        assert!(element_count <= MAX_ELEMENTS, "Too many elements");
        assert_eq!(elements.len(), boxes.len(), "Elements and boxes must match");
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

        // Copy layout boxes
        unsafe {
            std::ptr::copy_nonoverlapping(
                boxes.as_ptr(),
                self.layout_buffer.contents() as *mut LayoutBox,
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

        // Copy text content
        let text_len = text_content.len().min(1024 * 1024);
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

        // Set viewport
        unsafe {
            *(self.viewport_buffer.contents() as *mut Viewport) = viewport;
        }

        // Initialize vertex counts to 0
        unsafe {
            let ptr = self.vertex_count_buffer.contents() as *mut u32;
            for i in 0..element_count {
                *ptr.add(i) = 0;
            }
        }

        // Initialize vertices to 0
        unsafe {
            let ptr = self.vertex_buffer.contents() as *mut PaintVertex;
            for i in 0..MAX_VERTICES {
                *ptr.add(i) = PaintVertex::default();
            }
        }

        let command_buffer = self.command_queue.new_command_buffer();

        // Pass 1: Count vertices per element
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.count_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.style_buffer), 0);
            encoder.set_buffer(2, Some(&self.vertex_count_buffer), 0);
            encoder.set_buffer(3, Some(&self.element_count_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 2: Compute vertex offsets (prefix sum)
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.offset_pipeline);
            encoder.set_buffer(0, Some(&self.vertex_count_buffer), 0);
            encoder.set_buffer(1, Some(&self.vertex_offset_buffer), 0);
            encoder.set_buffer(2, Some(&self.element_count_buffer), 0);

            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(1, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 3: Generate background vertices
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.background_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.layout_buffer), 0);
            encoder.set_buffer(2, Some(&self.style_buffer), 0);
            encoder.set_buffer(3, Some(&self.vertex_offset_buffer), 0);
            encoder.set_buffer(4, Some(&self.vertex_buffer), 0);
            encoder.set_buffer(5, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(6, Some(&self.viewport_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 4: Generate border vertices
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.border_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.layout_buffer), 0);
            encoder.set_buffer(2, Some(&self.style_buffer), 0);
            encoder.set_buffer(3, Some(&self.vertex_offset_buffer), 0);
            encoder.set_buffer(4, Some(&self.vertex_count_buffer), 0);
            encoder.set_buffer(5, Some(&self.vertex_buffer), 0);
            encoder.set_buffer(6, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(7, Some(&self.viewport_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        // Pass 5: Generate text vertices
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.text_pipeline);
            encoder.set_buffer(0, Some(&self.element_buffer), 0);
            encoder.set_buffer(1, Some(&self.layout_buffer), 0);
            encoder.set_buffer(2, Some(&self.style_buffer), 0);
            encoder.set_buffer(3, Some(&self.text_buffer), 0);
            encoder.set_buffer(4, Some(&self.vertex_offset_buffer), 0);
            encoder.set_buffer(5, Some(&self.vertex_count_buffer), 0);
            encoder.set_buffer(6, Some(&self.vertex_buffer), 0);
            encoder.set_buffer(7, Some(&self.element_count_buffer), 0);
            encoder.set_buffer(8, Some(&self.viewport_buffer), 0);

            let threadgroups = ((element_count as u64 + THREAD_COUNT - 1) / THREAD_COUNT).max(1);
            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(THREAD_COUNT, 1, 1),
            );
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Calculate total vertices
        let counts_ptr = self.vertex_count_buffer.contents() as *const u32;
        let offsets_ptr = self.vertex_offset_buffer.contents() as *const u32;
        let total_vertices: u32 = unsafe {
            let last_offset = *offsets_ptr.add(element_count - 1);
            let last_count = *counts_ptr.add(element_count - 1);
            last_offset + last_count
        };

        // Read vertices
        let vertex_ptr = self.vertex_buffer.contents() as *const PaintVertex;
        (0..total_vertices as usize)
            .map(|i| unsafe { *vertex_ptr.add(i) })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tokenizer::GpuTokenizer;
    use super::super::parser::GpuParser;
    use super::super::style::{GpuStyler, Stylesheet};
    use super::super::layout::GpuLayoutEngine;

    fn setup() -> (GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine) {
        let device = Device::system_default().expect("No Metal device");
        let tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
        let parser = GpuParser::new(&device).expect("Failed to create parser");
        let styler = GpuStyler::new(&device).expect("Failed to create styler");
        let layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");
        let paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");
        (tokenizer, parser, styler, layout, paint)
    }

    fn process_html(html: &[u8], css: &str, viewport: Viewport) -> Vec<PaintVertex> {
        let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();

        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);
        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
        let boxes = layout.compute_layout(&elements, &styles, viewport);
        paint.paint(&elements, &boxes, &styles, &text, viewport)
    }

    #[test]
    fn test_background_vertices() {
        let device = Device::system_default().expect("No Metal device");
        let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

        // Single element with red background
        let elements = vec![Element {
            element_type: 1,  // div
            parent: -1,
            first_child: -1,
            next_sibling: -1,
            text_start: 0,
            text_length: 0,
            token_index: 0,
            _padding: 0,
        }];
        let boxes = vec![LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
            content_x: 0.0,
            content_y: 0.0,
            content_width: 100.0,
            content_height: 50.0,
            ..Default::default()
        }];
        let styles = vec![ComputedStyle {
            display: 1,  // DISPLAY_BLOCK
            width: 100.0,
            height: 50.0,
            background_color: [1.0, 0.0, 0.0, 1.0],  // Red
            opacity: 1.0,
            ..Default::default()
        }];

        let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };
        let vertices = paint.paint(&elements, &boxes, &styles, &[], viewport);

        // Should have at least 4 vertices for the background quad
        let bg_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_BACKGROUND).collect();
        assert!(bg_vertices.len() >= 4, "Expected at least 4 background vertices, got {}", bg_vertices.len());

        // Check color is red
        let first = &bg_vertices[0];
        assert!((first.color[0] - 1.0).abs() < 0.01, "Expected red, got {:?}", first.color);
    }

    #[test]
    fn test_border_vertices() {
        let device = Device::system_default().expect("No Metal device");
        let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

        let elements = vec![Element {
            element_type: 1,
            parent: -1,
            first_child: -1,
            next_sibling: -1,
            text_start: 0,
            text_length: 0,
            token_index: 0,
            _padding: 0,
        }];
        let boxes = vec![LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 50.0,
            content_x: 2.0,
            content_y: 2.0,
            content_width: 96.0,
            content_height: 46.0,
            ..Default::default()
        }];
        let styles = vec![ComputedStyle {
            display: 1,
            width: 100.0,
            height: 50.0,
            border_width: [2.0, 2.0, 2.0, 2.0],  // 2px all sides
            border_color: [0.0, 0.0, 0.0, 1.0],  // Black
            opacity: 1.0,
            ..Default::default()
        }];

        let viewport = Viewport::default();
        let vertices = paint.paint(&elements, &boxes, &styles, &[], viewport);

        // Should have border vertices (4 sides * 4 vertices = 16)
        let border_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_BORDER).collect();
        assert!(border_vertices.len() >= 16, "Expected at least 16 border vertices, got {}", border_vertices.len());
    }

    #[test]
    fn test_text_vertices() {
        let device = Device::system_default().expect("No Metal device");
        let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

        // Text element with "Hello" (5 characters)
        let text = b"Hello";
        let elements = vec![Element {
            element_type: 100,  // ELEM_TEXT
            parent: -1,
            first_child: -1,
            next_sibling: -1,
            text_start: 0,
            text_length: 5,
            token_index: 0,
            _padding: 0,
        }];
        let boxes = vec![LayoutBox {
            x: 0.0,
            y: 0.0,
            width: 50.0,
            height: 16.0,
            content_x: 0.0,
            content_y: 0.0,
            content_width: 50.0,
            content_height: 16.0,
            ..Default::default()
        }];
        let styles = vec![ComputedStyle {
            display: 2,  // DISPLAY_INLINE (text)
            color: [0.0, 0.0, 1.0, 1.0],  // Blue
            font_size: 16.0,
            opacity: 1.0,
            ..Default::default()
        }];

        let viewport = Viewport::default();
        let vertices = paint.paint(&elements, &boxes, &styles, text, viewport);

        // Text vertices (5 chars = 5 * 4 = 20 vertices)
        let text_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_TEXT).collect();
        assert!(text_vertices.len() >= 20, "Expected at least 20 text vertices, got {}", text_vertices.len());

        // Check text color is blue
        if !text_vertices.is_empty() {
            let first = &text_vertices[0];
            assert!((first.color[2] - 1.0).abs() < 0.01, "Expected blue, got {:?}", first.color);
        }
    }

    #[test]
    fn test_ndc_coordinates() {
        let html = b"<div>text</div>";
        let css = "div { width: 100px; height: 50px; background: red; }";
        let viewport = Viewport { width: 800.0, height: 600.0, _padding: [0.0; 2] };

        let vertices = process_html(html, css, viewport);

        // All vertices should be in NDC range [-1, 1]
        for v in &vertices {
            if v.flags != 0 {  // Skip uninitialized vertices
                assert!(v.position[0] >= -1.0 && v.position[0] <= 1.0,
                    "X position {} out of NDC range", v.position[0]);
                assert!(v.position[1] >= -1.0 && v.position[1] <= 1.0,
                    "Y position {} out of NDC range", v.position[1]);
            }
        }
    }

    #[test]
    fn test_empty_document() {
        let html = b"";
        let css = "";
        let viewport = Viewport::default();

        let (mut tokenizer, mut parser, mut styler, mut layout, mut paint) = setup();
        let tokens = tokenizer.tokenize(html);
        let (elements, text) = parser.parse(&tokens, html);

        if elements.is_empty() {
            return;  // Empty is OK
        }

        let stylesheet = Stylesheet::parse(css);
        let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
        let boxes = layout.compute_layout(&elements, &styles, viewport);
        let vertices = paint.paint(&elements, &boxes, &styles, &text, viewport);

        // Should not crash, may or may not have vertices
        assert!(vertices.len() < 1000000, "Too many vertices for empty doc");
    }

    #[test]
    fn test_multiple_elements() {
        let device = Device::system_default().expect("No Metal device");
        let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

        // Three elements with backgrounds
        let elements = vec![
            Element {
                element_type: 1,  // div
                parent: -1,
                first_child: 1,
                next_sibling: -1,
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            },
            Element {
                element_type: 2,  // span
                parent: 0,
                first_child: -1,
                next_sibling: 2,
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            },
            Element {
                element_type: 2,  // span
                parent: 0,
                first_child: -1,
                next_sibling: -1,
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            },
        ];
        let boxes = vec![
            LayoutBox { x: 0.0, y: 0.0, width: 200.0, height: 50.0, ..Default::default() },
            LayoutBox { x: 0.0, y: 0.0, width: 80.0, height: 30.0, ..Default::default() },
            LayoutBox { x: 80.0, y: 0.0, width: 80.0, height: 30.0, ..Default::default() },
        ];
        let styles = vec![
            ComputedStyle { display: 1, background_color: [0.5, 0.5, 0.5, 1.0], opacity: 1.0, ..Default::default() },
            ComputedStyle { display: 2, background_color: [1.0, 1.0, 1.0, 1.0], opacity: 1.0, ..Default::default() },
            ComputedStyle { display: 2, background_color: [1.0, 1.0, 1.0, 1.0], opacity: 1.0, ..Default::default() },
        ];

        let viewport = Viewport::default();
        let vertices = paint.paint(&elements, &boxes, &styles, &[], viewport);

        // Should have 3 background quads (3 * 4 = 12 vertices)
        let bg_vertices: Vec<_> = vertices.iter().filter(|v| v.flags == FLAG_BACKGROUND).collect();
        assert!(bg_vertices.len() >= 12, "Expected at least 12 background vertices (3 elements x 4), got {}", bg_vertices.len());
    }

    #[test]
    fn test_performance_1k_elements() {
        let device = Device::system_default().expect("No Metal device");
        let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

        // Generate 1K elements
        let element_count = 1000;
        let mut elements = Vec::with_capacity(element_count);
        let mut boxes = Vec::with_capacity(element_count);
        let mut styles = Vec::with_capacity(element_count);

        for i in 0..element_count {
            elements.push(Element {
                element_type: 1,  // div
                parent: if i == 0 { -1 } else { 0 },
                first_child: -1,
                next_sibling: if i > 0 && i < element_count - 1 { (i + 1) as i32 } else { -1 },
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            });
            boxes.push(LayoutBox {
                x: (i % 10) as f32 * 80.0,
                y: (i / 10) as f32 * 60.0,
                width: 70.0,
                height: 50.0,
                content_x: (i % 10) as f32 * 80.0 + 5.0,
                content_y: (i / 10) as f32 * 60.0 + 5.0,
                content_width: 60.0,
                content_height: 40.0,
                ..Default::default()
            });
            styles.push(ComputedStyle {
                display: 1,  // DISPLAY_BLOCK
                width: 70.0,
                height: 50.0,
                background_color: [0.5, 0.5, 0.5, 1.0],
                border_width: [1.0, 1.0, 1.0, 1.0],
                border_color: [0.0, 0.0, 0.0, 1.0],
                opacity: 1.0,
                ..Default::default()
            });
        }

        let viewport = Viewport::default();
        let text = Vec::new();

        // Warmup
        let _ = paint.paint(&elements, &boxes, &styles, &text, viewport);

        // Timed run
        let start = std::time::Instant::now();
        let vertices = paint.paint(&elements, &boxes, &styles, &text, viewport);
        let elapsed = start.elapsed();

        println!("~1K elements paint: {} vertices in {:?}", vertices.len(), elapsed);

        // Each element: 4 background + 16 border = 20 vertices
        assert!(vertices.len() >= element_count * 4, "Expected at least {} vertices", element_count * 4);
        // Note: First run may be slow due to GPU compilation overhead
        assert!(elapsed.as_millis() < 50, "1K paint took too long: {:?}", elapsed);
    }

    #[test]
    fn test_performance_5k_elements() {
        let device = Device::system_default().expect("No Metal device");
        let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

        // Generate 5K elements
        let element_count = 5000;
        let mut elements = Vec::with_capacity(element_count);
        let mut boxes = Vec::with_capacity(element_count);
        let mut styles = Vec::with_capacity(element_count);

        for i in 0..element_count {
            elements.push(Element {
                element_type: 1,
                parent: if i == 0 { -1 } else { 0 },
                first_child: -1,
                next_sibling: if i > 0 && i < element_count - 1 { (i + 1) as i32 } else { -1 },
                text_start: 0,
                text_length: 0,
                token_index: 0,
                _padding: 0,
            });
            boxes.push(LayoutBox {
                x: (i % 100) as f32 * 8.0,
                y: (i / 100) as f32 * 12.0,
                width: 7.0,
                height: 10.0,
                content_x: (i % 100) as f32 * 8.0,
                content_y: (i / 100) as f32 * 12.0,
                content_width: 7.0,
                content_height: 10.0,
                ..Default::default()
            });
            styles.push(ComputedStyle {
                display: 1,
                width: 7.0,
                height: 10.0,
                background_color: [0.3, 0.6, 0.9, 1.0],
                opacity: 1.0,
                ..Default::default()
            });
        }

        let viewport = Viewport::default();
        let text = Vec::new();

        // Warmup
        let _ = paint.paint(&elements, &boxes, &styles, &text, viewport);

        // Timed run
        let start = std::time::Instant::now();
        let vertices = paint.paint(&elements, &boxes, &styles, &text, viewport);
        let elapsed = start.elapsed();

        println!("~5K elements paint: {} vertices in {:?}", vertices.len(), elapsed);

        assert!(vertices.len() >= element_count * 4);
        assert!(elapsed.as_millis() < 50, "5K paint took too long: {:?}", elapsed);
    }
}
