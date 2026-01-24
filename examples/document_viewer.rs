//! GPU Document Viewer
//!
//! Renders HTML documents using the GPU-native document pipeline with text rendering.
//!
//! Run with:
//!   cargo run --release --example document_viewer                    # Default demo HTML
//!   cargo run --release --example document_viewer -- https://example.com  # Load from URL

use cocoa::{appkit::NSView, base::id as cocoa_id};
use std::env;
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::document::{
    GpuLayoutEngine, GpuPaintEngine, GpuParser, GpuStyler, GpuTokenizer, PaintVertex, Stylesheet,
    Viewport, FLAG_TEXT, FLAG_BACKGROUND, FLAG_BORDER,
};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer};
use std::mem;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WINDOW_WIDTH: u32 = 1024;
const WINDOW_HEIGHT: u32 = 768;

// ============================================================================
// Shader for rendering document backgrounds
// ============================================================================

const DOCUMENT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float4 color;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

struct Uniforms {
    float scroll_y;
    float _pad[3];
};

vertex VertexOut document_vertex(
    const device Vertex* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    Vertex v = vertices[vid];
    VertexOut out;
    out.position = float4(v.position.x, v.position.y + uniforms.scroll_y * 2.0, 0.0, 1.0);
    out.color = v.color;
    return out;
}

fragment float4 document_fragment(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;

// ============================================================================
// Document Data
// ============================================================================

/// Simple vertex for document rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct Uniforms {
    scroll_y: f32,
    _pad: [f32; 3],
}

/// Text item to render
struct TextItem {
    text: String,
    x: f32,
    y: f32,
    color: u32,
    scale: f32,
}

// ============================================================================
// Document Viewer
// ============================================================================

struct DocumentViewer {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    render_pipeline: RenderPipelineState,
    vertex_buffer: Buffer,
    uniform_buffer: Buffer,
    vertex_count: usize,
    text_items: Vec<TextItem>,
    font: BitmapFont,
    text_renderer: TextRenderer,
    scroll_y: f32,
    document_height: f32,
}

impl DocumentViewer {
    fn new(device: Device) -> Result<Self, String> {
        println!("\n=== Initializing Document Viewer ===");

        let command_queue = device.new_command_queue();

        // Create render pipeline
        let library = device
            .new_library_with_source(DOCUMENT_SHADER, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile shader: {}", e))?;

        let vertex_fn = library
            .get_function("document_vertex", None)
            .map_err(|e| format!("Failed to get vertex function: {}", e))?;
        let fragment_fn = library
            .get_function("document_fragment", None)
            .map_err(|e| format!("Failed to get fragment function: {}", e))?;

        let pipeline_desc = RenderPipelineDescriptor::new();
        pipeline_desc.set_vertex_function(Some(&vertex_fn));
        pipeline_desc.set_fragment_function(Some(&fragment_fn));

        let attachment = pipeline_desc.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        // Add missing alpha blend factors (identified by agents 1 & 7)
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);

        let render_pipeline = device
            .new_render_pipeline_state(&pipeline_desc)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        // Process document
        let (vertices, text_items, document_height) = process_document(&device);

        println!(
            "Generated {} background vertices, {} text items",
            vertices.len(),
            text_items.len()
        );

        // Create buffers
        let vertex_buffer = device.new_buffer_with_data(
            vertices.as_ptr() as *const _,
            (vertices.len() * mem::size_of::<Vertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let uniform_buffer = device.new_buffer(
            mem::size_of::<Uniforms>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create text rendering
        let font = BitmapFont::new(&device);
        let text_renderer =
            TextRenderer::new(&device, 8192).map_err(|e| format!("Failed to create text renderer: {}", e))?;

        println!("=== Document Viewer Ready ===\n");
        println!("Controls:");
        println!("  Up/Down arrows or scroll: Scroll document");
        println!("  Escape: Quit\n");

        Ok(Self {
            device,
            command_queue,
            render_pipeline,
            vertex_buffer,
            uniform_buffer,
            vertex_count: vertices.len(),
            text_items,
            font,
            text_renderer,
            scroll_y: 0.0,
            document_height,
        })
    }

    fn render(&mut self, drawable: &MetalDrawableRef) {
        // Update uniforms
        unsafe {
            let ptr = self.uniform_buffer.contents() as *mut Uniforms;
            *ptr = Uniforms {
                scroll_y: self.scroll_y,
                _pad: [0.0; 3],
            };
        }

        // Prepare text with scroll offset
        self.text_renderer.clear();
        for item in &self.text_items {
            let y = item.y + self.scroll_y * WINDOW_HEIGHT as f32;
            // Only add visible text
            if y > -50.0 && y < WINDOW_HEIGHT as f32 + 50.0 {
                self.text_renderer
                    .add_text_scaled(&item.text, item.x, y, item.color, item.scale);
            }
        }

        let command_buffer = self.command_queue.new_command_buffer();

        let render_pass_desc = RenderPassDescriptor::new();
        let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.93, 0.93, 0.93, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(render_pass_desc);

        // Draw document backgrounds
        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.vertex_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&self.uniform_buffer), 0);
        // Draw document backgrounds - restored to test in isolation (no text rendering)
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, self.vertex_count as u64);
        encoder.end_encoding();

        // Draw text - create new render pass with LoadAction::Load to preserve backgrounds
        let text_render_pass_desc = RenderPassDescriptor::new();
        let text_color_attachment = text_render_pass_desc.color_attachments().object_at(0).unwrap();
        text_color_attachment.set_texture(Some(drawable.texture()));
        text_color_attachment.set_load_action(MTLLoadAction::Load);
        text_color_attachment.set_store_action(MTLStoreAction::Store);

        let text_encoder = command_buffer.new_render_command_encoder(&text_render_pass_desc);
        self.text_renderer.render(
            &text_encoder,
            &self.font,
            WINDOW_WIDTH as f32,
            WINDOW_HEIGHT as f32,
        );
        text_encoder.end_encoding();

        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }

    fn scroll(&mut self, delta: f32) {
        let max_scroll = (self.document_height - WINDOW_HEIGHT as f32).max(0.0) / WINDOW_HEIGHT as f32;
        self.scroll_y = (self.scroll_y + delta).clamp(-max_scroll, 0.0);
    }
}

// ============================================================================
// HTML Source
// ============================================================================

fn get_html_source() -> Vec<u8> {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        let url = &args[1];
        if url.starts_with("http://") || url.starts_with("https://") {
            println!("Fetching HTML from: {}", url);
            match ureq::get(url).call() {
                Ok(response) => {
                    match response.into_string() {
                        Ok(body) => {
                            println!("Fetched {} bytes", body.len());
                            return body.into_bytes();
                        }
                        Err(e) => {
                            println!("Failed to read response: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("Failed to fetch URL: {}", e);
                }
            }
            println!("Falling back to default HTML...");
        } else {
            // Try to read as file
            if let Ok(content) = std::fs::read(url) {
                println!("Loaded {} bytes from file: {}", content.len(), url);
                return content;
            }
        }
    }

    // Default demo HTML
    br#"<!DOCTYPE html>
<html>
<body>
    <header>
        <h1>GPU Document Viewer</h1>
        <p>Rendered entirely on the GPU!</p>
    </header>

    <main>
        <section>
            <h2>Features</h2>
            <ul>
                <li>HTML Tokenization</li>
                <li>DOM Tree Construction</li>
                <li>CSS Style Resolution</li>
                <li>Layout Computation</li>
                <li>Vertex Generation</li>
            </ul>
        </section>

        <section>
            <h2>Performance</h2>
            <p>All processing happens on the GPU using Metal compute shaders.</p>
            <p>This enables massive parallelism for document rendering.</p>
        </section>

        <div>
            <h3>Built with Rust and Metal</h3>
            <p>A GPU-native approach to document rendering.</p>
        </div>
    </main>

    <footer>
        <p>GPU Native OS Project</p>
    </footer>
</body>
</html>"#.to_vec()
}

// ============================================================================
// Document Processing
// ============================================================================

fn process_document(device: &Device) -> (Vec<Vertex>, Vec<TextItem>, f32) {
    // Create pipeline stages
    let mut tokenizer = GpuTokenizer::new(device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(device).expect("Failed to create parser");
    let mut styler = GpuStyler::new(device).expect("Failed to create styler");
    let mut layout = GpuLayoutEngine::new(device).expect("Failed to create layout engine");
    let mut paint = GpuPaintEngine::new(device).expect("Failed to create paint engine");

    // Get HTML source (from URL, file, or default)
    let html = get_html_source();
    let html = html.as_slice();

    // Default CSS stylesheet (simple styling for any HTML)
    let css = "
        * { margin: 0; padding: 0; }
        body { background: #e0e0e0; padding: 20px; }
        header { background: #2c3e50; padding: 30px; margin: 0 0 20px 0; }
        h1 { font-size: 36px; margin: 0 0 10px 0; background: #1a252f; padding: 10px; }
        h2 { font-size: 24px; margin: 20px 0 10px 0; background: #34495e; padding: 8px; }
        h3 { font-size: 20px; margin: 10px 0; background: #4a6785; padding: 6px; }
        p { font-size: 16px; margin: 10px 0; background: #bdc3c7; padding: 5px; }
        main { padding: 20px; background: #ecf0f1; }
        section { margin: 20px 0; background: #d5dbdb; padding: 10px; }
        ul { margin: 10px 0 10px 30px; background: #aeb6bf; padding: 10px; }
        li { margin: 5px 0; font-size: 16px; background: #85929e; padding: 4px; }
        div { background: #3498db; padding: 20px; margin: 20px 0; }
        footer { background: #2c3e50; padding: 15px; margin: 20px 0 0 0; }
    ";

    let viewport = Viewport {
        width: WINDOW_WIDTH as f32,
        height: WINDOW_HEIGHT as f32,
        _padding: [0.0; 2],
    };

    // Process through pipeline
    println!("Processing document through GPU pipeline...");

    let start = std::time::Instant::now();
    let tokens = tokenizer.tokenize(html);
    let (elements, text_buffer) = parser.parse(&tokens, html);
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let boxes = layout.compute_layout(&elements, &styles, viewport);
    let paint_vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);

    let elapsed = start.elapsed();
    println!("Pipeline time: {:?}", elapsed);

    // Verify paint vertices are valid (no uninitialized data)
    let mut flag_counts = std::collections::HashMap::new();
    for pv in &paint_vertices {
        *flag_counts.entry(pv.flags).or_insert(0) += 1;
    }
    if flag_counts.contains_key(&0) {
        println!("  WARNING: {} uninitialized (flag=0) vertices!", flag_counts.get(&0).unwrap());
    }

    // Convert to simple vertices (backgrounds only) and extract text items
    let mut vertices = Vec::new();
    let mut text_items = Vec::new();
    let mut document_height: f32 = 0.0;

    // Helper function to validate vertex data
    let is_valid_position = |p: &[f32; 2]| -> bool {
        // NDC positions should be in reasonable range (allow some overflow for scrolling)
        let valid_range = -10.0..10.0;
        valid_range.contains(&p[0]) && valid_range.contains(&p[1])
            && !p[0].is_nan() && !p[1].is_nan()
            && !p[0].is_infinite() && !p[1].is_infinite()
    };

    let is_valid_color = |c: &[f32; 4]| -> bool {
        // Colors should be in [0, 1] range
        c.iter().all(|&v| v >= 0.0 && v <= 1.0 && !v.is_nan() && !v.is_infinite())
    };

    let colors_similar = |c1: &[f32; 4], c2: &[f32; 4]| -> bool {
        // For solid backgrounds, all vertices should have the same color
        const EPSILON: f32 = 0.01;
        (c1[0] - c2[0]).abs() < EPSILON
            && (c1[1] - c2[1]).abs() < EPSILON
            && (c1[2] - c2[2]).abs() < EPSILON
            && (c1[3] - c2[3]).abs() < EPSILON
    };

    // Process paint vertices - handle variable-length quads properly
    // The paint kernel outputs vertices in groups, but borders use 16 vertices (4 per side)
    // We need to process each quad by checking that all 4 vertices have consistent flags
    let mut i = 0;
    let mut skipped_invalid = 0;
    while i + 3 < paint_vertices.len() {
        let v0 = &paint_vertices[i];
        let v1 = &paint_vertices[i + 1];
        let v2 = &paint_vertices[i + 2];
        let v3 = &paint_vertices[i + 3];

        // Check flag consistency - all 4 vertices in a valid quad must have same flag
        let flags_match = v0.flags == v1.flags && v1.flags == v2.flags && v2.flags == v3.flags;

        // Skip if flags don't match (misaligned data) or flag is 0 (uninitialized)
        if !flags_match || v0.flags == 0 {
            i += 1; // Advance by 1 to try to realign
            continue;
        }

        if v0.flags == FLAG_TEXT {
            // This is a text glyph - extract text info
            // The tex_coord encodes the character code
            let char_code = ((v0.tex_coord[0] * 16.0) as u32) + ((v0.tex_coord[1] * 16.0) as u32) * 16;
            if char_code >= 32 && char_code < 128 {
                // Convert NDC position to screen coordinates
                let x = (v0.position[0] + 1.0) * 0.5 * WINDOW_WIDTH as f32;
                let y = (1.0 - v0.position[1]) * 0.5 * WINDOW_HEIGHT as f32;

                // Pack color as RGBA
                let r = (v0.color[0] * 255.0) as u32;
                let g = (v0.color[1] * 255.0) as u32;
                let b = (v0.color[2] * 255.0) as u32;
                let a = (v0.color[3] * 255.0) as u32;
                let color = (r << 24) | (g << 16) | (b << 8) | a;

                text_items.push(TextItem {
                    text: char::from_u32(char_code).unwrap_or(' ').to_string(),
                    x,
                    y,
                    color,
                    scale: 1.0,
                });
            }
            i += 4;
        } else if v0.flags == FLAG_BACKGROUND || v0.flags == FLAG_BORDER {
            // Validate ALL vertex positions and colors
            let positions_valid = is_valid_position(&v0.position)
                && is_valid_position(&v1.position)
                && is_valid_position(&v2.position)
                && is_valid_position(&v3.position);

            let colors_valid = is_valid_color(&v0.color)
                && is_valid_color(&v1.color)
                && is_valid_color(&v2.color)
                && is_valid_color(&v3.color);

            // For backgrounds, all vertices should have the same color (solid fill)
            let colors_consistent = v0.flags != FLAG_BACKGROUND || (
                colors_similar(&v0.color, &v1.color)
                && colors_similar(&v1.color, &v2.color)
                && colors_similar(&v2.color, &v3.color)
            );

            // Skip if data validation fails
            if !positions_valid || !colors_valid || !colors_consistent {
                skipped_invalid += 1;
                i += 4;
                continue;
            }

            // Skip if color alpha is zero (invisible)
            if v0.color[3] <= 0.0 {
                i += 4;
                continue;
            }

            let to_vertex = |pv: &PaintVertex| Vertex {
                position: pv.position,
                color: pv.color,
            };

            // Triangle 1: v0, v1, v2
            vertices.push(to_vertex(v0));
            vertices.push(to_vertex(v1));
            vertices.push(to_vertex(v2));

            // Triangle 2: v0, v2, v3
            vertices.push(to_vertex(v0));
            vertices.push(to_vertex(v2));
            vertices.push(to_vertex(v3));

            // Track document height
            for v in [v0, v1, v2, v3] {
                let screen_y = (1.0 - v.position[1]) * 0.5 * WINDOW_HEIGHT as f32;
                document_height = document_height.max(screen_y);
            }

            i += 4;
        } else {
            // Unknown flag - skip this vertex and try to realign
            i += 1;
        }
    }

    if skipped_invalid > 0 {
        println!("  Skipped {} invalid quads", skipped_invalid);
    }

    // Element types to skip (script, style contain non-visible text)
    const ELEM_STYLE: u32 = 107;
    const ELEM_SCRIPT: u32 = 108;

    // Extract text from elements and position them
    let mut real_text_items = Vec::new();
    for (i, elem) in elements.iter().enumerate() {
        if elem.element_type == 100 && elem.text_length > 0 {
            // Text node - skip if parent is script or style
            let parent_idx = elem.parent as usize;
            if parent_idx < elements.len() {
                let parent_type = elements[parent_idx].element_type;
                if parent_type == ELEM_SCRIPT || parent_type == ELEM_STYLE {
                    continue; // Skip script/style content
                }
            }

            let start = elem.text_start as usize;
            let end = (elem.text_start + elem.text_length) as usize;
            if end <= text_buffer.len() {
                let text = String::from_utf8_lossy(&text_buffer[start..end]).to_string();
                let text = text.trim().to_string();

                if !text.is_empty() {
                    let bx = &boxes[i];
                    let style = &styles[i];

                    // Convert from layout coords to screen coords
                    let x = bx.x;
                    let y = bx.y;

                    // Determine text color based on parent's background
                    let parent_idx = elem.parent as usize;
                    let is_dark_bg = if parent_idx < styles.len() {
                        let bg = &styles[parent_idx].background_color;
                        // Simple luminance check
                        (bg[0] * 0.299 + bg[1] * 0.587 + bg[2] * 0.114) < 0.5
                    } else {
                        false
                    };

                    let color = if is_dark_bg {
                        0xFFFFFFFF // White text on dark background
                    } else {
                        0x222222FF // Dark text on light background
                    };

                    // Scale based on font size
                    let scale = (style.font_size / 16.0).max(0.5).min(3.0);

                    real_text_items.push(TextItem {
                        text,
                        x,
                        y,
                        color,
                        scale,
                    });

                    document_height = document_height.max(y + style.font_size * 1.2);
                }
            }
        }
    }

    println!(
        "  Backgrounds: {} vertices, Text: {} items",
        vertices.len(),
        real_text_items.len()
    );

    (vertices, real_text_items, document_height)
}

// ============================================================================
// Window Application
// ============================================================================

struct DocumentViewerApp {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    viewer: Option<DocumentViewer>,
}

impl DocumentViewerApp {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            viewer: None,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("========================================");
        println!("     GPU DOCUMENT VIEWER");
        println!("========================================");
        println!("GPU: {}", device.name());

        // Set up Metal layer
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                    let view = appkit_handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        let viewer = DocumentViewer::new(device).expect("Failed to create Document Viewer");

        self.window = Some(window);
        self.layer = Some(layer);
        self.viewer = Some(viewer);
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let viewer = self.viewer.as_mut().unwrap();

        if let Some(drawable) = layer.next_drawable() {
            viewer.render(drawable);
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }
    }

    fn scroll(&mut self, delta: f32) {
        if let Some(viewer) = &mut self.viewer {
            viewer.scroll(delta);
        }
    }
}

impl ApplicationHandler for DocumentViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title("GPU Document Viewer");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                println!("\n========================================");
                println!("            VIEWER CLOSED");
                println!("========================================");
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y * 0.05,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.0005,
                };
                self.scroll(scroll_amount);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{Key, NamedKey};
                if event.state == winit::event::ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::ArrowUp) => self.scroll(0.05),
                        Key::Named(NamedKey::ArrowDown) => self.scroll(-0.05),
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    println!("\nStarting GPU Document Viewer...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = DocumentViewerApp::new();
    event_loop.run_app(&mut app).unwrap();
}
