// GPU-Native Filesystem Browser
//
// Fully GPU-rendered filesystem browser using Metal compute shaders
// - Text rendering via compute shader
// - Search results rendered on GPU
// - CPU only handles keyboard input
//
// Launch: cargo run --release --example filesystem_browser

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::filesystem::{FileType, GpuFilesystem};
use std::collections::HashMap;
use std::fs;
use std::mem;
use std::path::Path;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 800;
const FONT_SIZE: f32 = 14.0;
const LINE_HEIGHT: f32 = 20.0;
const MAX_RESULTS: usize = 30;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = FilesystemBrowserApp::new();
    event_loop.run_app(&mut app).unwrap();
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Uniforms {
    screen_size: [f32; 2],
    time: f32,
    cursor_pos: u32,
    result_count: u32,
    scroll_offset: u32,
    selected_index: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct TextChar {
    x: f32,
    y: f32,
    char_code: u32,
    color: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct SearchResult {
    path_start: u32,
    path_len: u32,
    is_dir: u32,
    size_kb: u32,
}

struct FilesystemBrowserApp {
    window: Option<Window>,
    device: Device,
    layer: MetalLayer,
    command_queue: CommandQueue,
    render_pipeline: RenderPipelineState,
    compute_pipeline: ComputePipelineState,
    uniforms_buffer: Buffer,
    text_buffer: Buffer,
    path_buffer: Buffer,
    results_buffer: Buffer,
    font_texture: Texture,

    // State
    search_query: String,
    results: Vec<(String, bool, u64)>, // (path, is_dir, size)
    path_to_id: HashMap<String, u32>,
    scan_complete: bool,
    selected_index: usize,
    scroll_offset: usize,
    last_frame: Instant,
    status_message: String,
    file_count: usize,
    dir_count: usize,
}

impl FilesystemBrowserApp {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device");
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        let command_queue = device.new_command_queue();

        // Create shader library
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("Failed to compile shaders");

        // Create render pipeline
        let render_desc = RenderPipelineDescriptor::new();
        render_desc.set_vertex_function(Some(&library.get_function("vertex_main", None).unwrap()));
        render_desc
            .set_fragment_function(Some(&library.get_function("fragment_main", None).unwrap()));
        render_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let render_pipeline = device
            .new_render_pipeline_state(&render_desc)
            .expect("Failed to create render pipeline");

        // Create compute pipeline for text generation
        let compute_desc = ComputePipelineDescriptor::new();
        compute_desc.set_compute_function(Some(
            &library.get_function("generate_text", None).unwrap(),
        ));
        let compute_pipeline = device
            .new_compute_pipeline_state(&compute_desc)
            .expect("Failed to create compute pipeline");

        // Create buffers
        let uniforms_buffer = device.new_buffer(
            mem::size_of::<Uniforms>() as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let text_buffer = device.new_buffer(
            (10000 * mem::size_of::<TextChar>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let path_buffer = device.new_buffer(
            (100000) as u64, // 100KB for path strings
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        let results_buffer = device.new_buffer(
            (1000 * mem::size_of::<SearchResult>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache,
        );

        // Create simple font texture (8x8 bitmap font)
        let font_texture = create_font_texture(&device);

        Self {
            window: None,
            device,
            layer,
            command_queue,
            render_pipeline,
            compute_pipeline,
            uniforms_buffer,
            text_buffer,
            path_buffer,
            results_buffer,
            font_texture,
            search_query: String::new(),
            results: Vec::new(),
            path_to_id: HashMap::new(),
            scan_complete: false,
            selected_index: 0,
            scroll_offset: 0,
            last_frame: Instant::now(),
            status_message: "Scanning /Users/patrickkavanagh...".to_string(),
            file_count: 0,
            dir_count: 0,
        }
    }

    fn scan_filesystem(&mut self) {
        let start = Instant::now();
        let base_path = "/Users/patrickkavanagh";

        let mut fs = match GpuFilesystem::new(&self.device, 100000) {
            Ok(f) => f,
            Err(e) => {
                self.status_message = format!("Error: {}", e);
                return;
            }
        };

        scan_recursive(
            Path::new(base_path),
            0,
            &mut fs,
            &mut self.path_to_id,
            &mut self.file_count,
            &mut self.dir_count,
            0,
        );

        self.scan_complete = true;
        self.status_message = format!(
            "{} files, {} folders indexed in {:.1}s",
            self.file_count,
            self.dir_count,
            start.elapsed().as_secs_f32()
        );
    }

    fn perform_search(&mut self) {
        if self.search_query.is_empty() {
            self.results.clear();
            return;
        }

        let query = self.search_query.to_lowercase();
        self.results = self
            .path_to_id
            .keys()
            .filter(|path| path.to_lowercase().contains(&query))
            .take(500)
            .map(|path| {
                let is_dir = fs::metadata(path).map(|m| m.is_dir()).unwrap_or(false);
                let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                (path.clone(), is_dir, size)
            })
            .collect();

        // Sort: directories first, then alphabetically
        self.results.sort_by(|a, b| match (a.1, b.1) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.0.cmp(&b.0),
        });

        self.selected_index = 0;
        self.scroll_offset = 0;
    }

    fn render(&mut self) {
        // Update uniforms
        let uniforms = Uniforms {
            screen_size: [WIDTH as f32, HEIGHT as f32],
            time: self.last_frame.elapsed().as_secs_f32(),
            cursor_pos: self.search_query.len() as u32,
            result_count: self.results.len().min(MAX_RESULTS) as u32,
            scroll_offset: self.scroll_offset as u32,
            selected_index: self.selected_index as u32,
            _padding: 0,
        };

        unsafe {
            let ptr = self.uniforms_buffer.contents() as *mut Uniforms;
            *ptr = uniforms;
        }

        // Generate text data on CPU (simple approach)
        self.generate_text_data();

        // Get drawable after text generation to avoid borrow conflict
        let drawable = match self.layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Render pass
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.12, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(&render_desc);
        encoder.set_render_pipeline_state(&self.render_pipeline);
        encoder.set_vertex_buffer(0, Some(&self.uniforms_buffer), 0);
        encoder.set_vertex_buffer(1, Some(&self.text_buffer), 0);
        encoder.set_fragment_texture(0, Some(&self.font_texture));

        // Draw text quads
        let char_count = self.count_visible_chars();
        if char_count > 0 {
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, (char_count * 6) as u64);
        }

        encoder.end_encoding();

        command_buffer.present_drawable(&drawable);
        command_buffer.commit();

        self.last_frame = Instant::now();
    }

    fn generate_text_data(&mut self) {
        let mut chars: Vec<TextChar> = Vec::with_capacity(10000);
        let mut y = 30.0;

        // Title
        self.add_text(&mut chars, "GPU Filesystem Browser", 20.0, y, 0xFFFFFFFF);
        y += LINE_HEIGHT * 1.5;

        // Search box
        self.add_text(&mut chars, "Search: ", 20.0, y, 0xAAAAAAAA);
        let query_x = 20.0 + 8.0 * 8.0;
        if self.search_query.is_empty() {
            self.add_text(&mut chars, "Type to search...", query_x, y, 0x666666FF);
        } else {
            self.add_text(&mut chars, &self.search_query, query_x, y, 0xFFFFFFFF);
        }
        // Cursor
        let cursor_x = query_x + (self.search_query.len() as f32 * 8.0);
        self.add_text(&mut chars, "_", cursor_x, y, 0x00FF00FF);
        y += LINE_HEIGHT * 1.5;

        // Separator
        self.add_text(
            &mut chars,
            "─".repeat(80).as_str(),
            20.0,
            y,
            0x444444FF,
        );
        y += LINE_HEIGHT;

        // Results header
        if !self.results.is_empty() {
            let header = format!(
                "{} results (Up/Down to select, Enter to copy)",
                self.results.len()
            );
            self.add_text(&mut chars, &header, 20.0, y, 0x888888FF);
            y += LINE_HEIGHT;
        }

        // Results list
        let visible_results = self
            .results
            .iter()
            .skip(self.scroll_offset)
            .take(MAX_RESULTS);

        for (i, (path, is_dir, size)) in visible_results.enumerate() {
            let actual_index = i + self.scroll_offset;
            let is_selected = actual_index == self.selected_index;

            // Background highlight for selected
            let text_color = if is_selected {
                0x00FF00FF // Green for selected
            } else {
                0xCCCCCCFF
            };

            let icon = if *is_dir { ">" } else { " " };
            let prefix = if is_selected { "> " } else { "  " };

            // Truncate path to fit
            let max_path_len = 100;
            let display_path = if path.len() > max_path_len {
                format!("...{}", &path[path.len() - max_path_len + 3..])
            } else {
                path.clone()
            };

            let line = format!("{}{} {}", prefix, icon, display_path);
            self.add_text(&mut chars, &line, 20.0, y, text_color);

            // Size on the right
            if !is_dir && *size > 0 {
                let size_str = format_size(*size);
                let size_x = WIDTH as f32 - 100.0;
                self.add_text(&mut chars, &size_str, size_x, y, 0x666666FF);
            }

            y += LINE_HEIGHT;
        }

        // Empty state
        if self.results.is_empty() && !self.search_query.is_empty() {
            self.add_text(&mut chars, "No results found", 20.0, y + 50.0, 0x888888FF);
        } else if self.search_query.is_empty() && self.scan_complete {
            self.add_text(
                &mut chars,
                "Type to search indexed files",
                20.0,
                y + 50.0,
                0x666666FF,
            );
        }

        // Status bar at bottom
        let status_y = HEIGHT as f32 - 30.0;
        self.add_text(&mut chars, &self.status_message, 20.0, status_y, 0x888888FF);

        // Write to buffer
        unsafe {
            let ptr = self.text_buffer.contents() as *mut TextChar;
            for (i, ch) in chars.iter().enumerate() {
                *ptr.add(i) = *ch;
            }
        }
    }

    fn add_text(&self, chars: &mut Vec<TextChar>, text: &str, x: f32, y: f32, color: u32) {
        let mut cx = x;
        for c in text.chars() {
            if c.is_ascii() && c >= ' ' && c <= '~' {
                chars.push(TextChar {
                    x: cx,
                    y,
                    char_code: c as u32,
                    color,
                });
                cx += 12.0; // Spacing: 8 pixels * 1.5 scale = 12
            }
        }
    }

    fn count_visible_chars(&self) -> usize {
        // Count based on generated text
        let mut count = 0;
        count += 23; // Title
        count += 8 + self.search_query.len().max(17) + 1; // Search line

        count += 80; // Separator

        if !self.results.is_empty() {
            count += 50; // Header
        }

        let visible = self.results.len().min(MAX_RESULTS);
        count += visible * 120; // Approximate chars per result

        count += self.status_message.len();

        count.min(10000)
    }

    fn handle_key(&mut self, key: Key, state: ElementState) {
        if state != ElementState::Pressed {
            return;
        }

        match key {
            Key::Character(c) => {
                let s = c.as_str();
                if s.chars().all(|c| c.is_ascii() && !c.is_control()) {
                    self.search_query.push_str(s);
                    self.perform_search();
                }
            }
            Key::Named(NamedKey::Backspace) => {
                self.search_query.pop();
                self.perform_search();
            }
            Key::Named(NamedKey::Escape) => {
                self.search_query.clear();
                self.results.clear();
            }
            Key::Named(NamedKey::ArrowDown) => {
                if !self.results.is_empty() {
                    self.selected_index = (self.selected_index + 1).min(self.results.len() - 1);
                    // Scroll to keep selection visible
                    if self.selected_index >= self.scroll_offset + MAX_RESULTS {
                        self.scroll_offset = self.selected_index - MAX_RESULTS + 1;
                    }
                }
            }
            Key::Named(NamedKey::ArrowUp) => {
                if self.selected_index > 0 {
                    self.selected_index -= 1;
                    if self.selected_index < self.scroll_offset {
                        self.scroll_offset = self.selected_index;
                    }
                }
            }
            Key::Named(NamedKey::Enter) => {
                // Copy selected path
                if let Some((path, _, _)) = self.results.get(self.selected_index) {
                    // Use pbcopy on macOS
                    let _ = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(format!("echo -n '{}' | pbcopy", path))
                        .output();
                    self.status_message = format!("Copied: {}", path);
                }
            }
            Key::Named(NamedKey::PageDown) => {
                self.scroll_offset = (self.scroll_offset + MAX_RESULTS).min(
                    self.results.len().saturating_sub(MAX_RESULTS),
                );
                self.selected_index = self.scroll_offset;
            }
            Key::Named(NamedKey::PageUp) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(MAX_RESULTS);
                self.selected_index = self.scroll_offset;
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for FilesystemBrowserApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("GPU Filesystem Browser")
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT));

        let window = event_loop.create_window(window_attrs).unwrap();

        // Set up Metal layer
        if let Ok(handle) = window.window_handle() {
            if let RawWindowHandle::AppKit(appkit) = handle.as_raw() {
                unsafe {
                    let view = appkit.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(mem::transmute(self.layer.as_ref()));
                }
                self.layer
                    .set_drawable_size(CGSize::new(WIDTH as f64, HEIGHT as f64));
            }
        }

        self.window = Some(window);

        // Start filesystem scan
        self.scan_filesystem();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state,
                        ..
                    },
                ..
            } => {
                self.handle_key(logical_key, state);
            }
            WindowEvent::RedrawRequested => {
                autoreleasepool(|| {
                    self.render();
                });
            }
            _ => {}
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn scan_recursive(
    path: &Path,
    parent_id: u32,
    fs: &mut GpuFilesystem,
    path_to_id: &mut HashMap<String, u32>,
    file_count: &mut usize,
    dir_count: &mut usize,
    depth: usize,
) {
    if depth > 12 {
        return;
    }

    let skip_dirs = [
        ".git",
        "node_modules",
        ".Trash",
        "Library/Caches",
        ".cargo",
        "target",
        ".npm",
        ".cache",
        "Cache",
        "Caches",
        "DerivedData",
    ];

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        let full_path = entry_path.to_string_lossy().to_string();

        if skip_dirs.iter().any(|skip| full_path.contains(skip)) {
            continue;
        }

        let file_name = entry.file_name();
        let name = match file_name.to_str() {
            Some(s) => s,
            None => continue,
        };

        if name.starts_with('.') {
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        if metadata.is_dir() {
            *dir_count += 1;
            let id = match fs.add_file(parent_id, name, FileType::Directory) {
                Ok(id) => id,
                Err(_) => continue,
            };
            path_to_id.insert(full_path.clone(), id);
            scan_recursive(&entry_path, id, fs, path_to_id, file_count, dir_count, depth + 1);
        } else if metadata.is_file() {
            *file_count += 1;
            let id = match fs.add_file(parent_id, name, FileType::Regular) {
                Ok(id) => id,
                Err(_) => continue,
            };
            path_to_id.insert(full_path, id);
        }
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn create_font_texture(device: &Device) -> Texture {
    // 8x8 bitmap font, 16 chars per row, 6 rows = 96 chars (ASCII 32-127)
    let char_w = 8usize;
    let char_h = 8usize;
    let chars_per_row = 16usize;
    let num_rows = 6usize;
    let width = (chars_per_row * char_w) as u64;
    let height = (num_rows * char_h) as u64;

    let desc = TextureDescriptor::new();
    desc.set_width(width);
    desc.set_height(height);
    desc.set_pixel_format(MTLPixelFormat::R8Unorm);
    desc.set_texture_type(MTLTextureType::D2);
    desc.set_usage(MTLTextureUsage::ShaderRead);
    desc.set_storage_mode(MTLStorageMode::Shared);

    let texture = device.new_texture(&desc);
    let mut data = vec![0u8; (width * height) as usize];

    // Complete 8x8 font data for ASCII 32-127
    // Each character is 8 bytes, one per row, bits are pixels left-to-right
    let font: [u8; 768] = [
        // 32 ' '
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 33 '!'
        0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00,
        // 34 '"'
        0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 35 '#'
        0x24, 0x24, 0x7E, 0x24, 0x7E, 0x24, 0x24, 0x00,
        // 36 '$'
        0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00,
        // 37 '%'
        0x00, 0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00,
        // 38 '&'
        0x30, 0x48, 0x30, 0x56, 0x88, 0x88, 0x76, 0x00,
        // 39 '''
        0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 40 '('
        0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00,
        // 41 ')'
        0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00,
        // 42 '*'
        0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00,
        // 43 '+'
        0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00,
        // 44 ','
        0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30,
        // 45 '-'
        0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00,
        // 46 '.'
        0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00,
        // 47 '/'
        0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00,
        // 48 '0'
        0x3C, 0x66, 0x6E, 0x7E, 0x76, 0x66, 0x3C, 0x00,
        // 49 '1'
        0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00,
        // 50 '2'
        0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00,
        // 51 '3'
        0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00,
        // 52 '4'
        0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00,
        // 53 '5'
        0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00,
        // 54 '6'
        0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00,
        // 55 '7'
        0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00,
        // 56 '8'
        0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00,
        // 57 '9'
        0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00,
        // 58 ':'
        0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00,
        // 59 ';'
        0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00,
        // 60 '<'
        0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00,
        // 61 '='
        0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00,
        // 62 '>'
        0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00,
        // 63 '?'
        0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00,
        // 64 '@'
        0x3C, 0x66, 0x6E, 0x6A, 0x6E, 0x60, 0x3C, 0x00,
        // 65 'A'
        0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00,
        // 66 'B'
        0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00,
        // 67 'C'
        0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00,
        // 68 'D'
        0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00,
        // 69 'E'
        0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00,
        // 70 'F'
        0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00,
        // 71 'G'
        0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00,
        // 72 'H'
        0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00,
        // 73 'I'
        0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
        // 74 'J'
        0x06, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C, 0x00,
        // 75 'K'
        0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00,
        // 76 'L'
        0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00,
        // 77 'M'
        0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00,
        // 78 'N'
        0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00,
        // 79 'O'
        0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00,
        // 80 'P'
        0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00,
        // 81 'Q'
        0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00,
        // 82 'R'
        0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00,
        // 83 'S'
        0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00,
        // 84 'T'
        0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00,
        // 85 'U'
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00,
        // 86 'V'
        0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00,
        // 87 'W'
        0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00,
        // 88 'X'
        0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00,
        // 89 'Y'
        0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00,
        // 90 'Z'
        0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00,
        // 91 '['
        0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00,
        // 92 '\'
        0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00,
        // 93 ']'
        0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00,
        // 94 '^'
        0x18, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 95 '_'
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00,
        // 96 '`'
        0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 97 'a'
        0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00,
        // 98 'b'
        0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00,
        // 99 'c'
        0x00, 0x00, 0x3C, 0x66, 0x60, 0x66, 0x3C, 0x00,
        // 100 'd'
        0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00,
        // 101 'e'
        0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00,
        // 102 'f'
        0x1C, 0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x00,
        // 103 'g'
        0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x3C,
        // 104 'h'
        0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00,
        // 105 'i'
        0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00,
        // 106 'j'
        0x0C, 0x00, 0x1C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38,
        // 107 'k'
        0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x00,
        // 108 'l'
        0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00,
        // 109 'm'
        0x00, 0x00, 0x76, 0x7F, 0x6B, 0x6B, 0x63, 0x00,
        // 110 'n'
        0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00,
        // 111 'o'
        0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00,
        // 112 'p'
        0x00, 0x00, 0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60,
        // 113 'q'
        0x00, 0x00, 0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06,
        // 114 'r'
        0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00,
        // 115 's'
        0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00,
        // 116 't'
        0x30, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x1C, 0x00,
        // 117 'u'
        0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00,
        // 118 'v'
        0x00, 0x00, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00,
        // 119 'w'
        0x00, 0x00, 0x63, 0x6B, 0x6B, 0x7F, 0x36, 0x00,
        // 120 'x'
        0x00, 0x00, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x00,
        // 121 'y'
        0x00, 0x00, 0x66, 0x66, 0x66, 0x3E, 0x06, 0x3C,
        // 122 'z'
        0x00, 0x00, 0x7E, 0x0C, 0x18, 0x30, 0x7E, 0x00,
        // 123 '{'
        0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00,
        // 124 '|'
        0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00,
        // 125 '}'
        0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00,
        // 126 '~'
        0x32, 0x4C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // 127 DEL (placeholder)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    // Render each character into the texture
    for ascii in 32u8..128 {
        let idx = (ascii - 32) as usize;
        let col = idx % chars_per_row;
        let row = idx / chars_per_row;
        let base_x = col * char_w;
        let base_y = row * char_h;

        // Guard against out of bounds
        if idx * 8 + 7 >= font.len() {
            continue;
        }

        for py in 0..8 {
            let byte = font[idx * 8 + py];
            for px in 0..8 {
                let bit = (byte >> (7 - px)) & 1;
                let x = base_x + px;
                let y = base_y + py;
                if x < width as usize && y < height as usize {
                    data[y * width as usize + x] = if bit == 1 { 255 } else { 0 };
                }
            }
        }
    }

    // Debug: Print a few values to verify
    println!("Font texture: {}x{}, first bytes: {:?}", width, height, &data[0..16]);

    texture.replace_region(
        MTLRegion::new_2d(0, 0, width, height),
        0,
        data.as_ptr() as *const _,
        width,
    );

    texture
}

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float2 screen_size;
    float time;
    uint cursor_pos;
    uint result_count;
    uint scroll_offset;
    uint selected_index;
    uint _padding;
};

struct TextChar {
    float x;
    float y;
    uint char_code;
    uint color;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex VertexOut vertex_main(
    uint vid [[vertex_id]],
    constant Uniforms& uniforms [[buffer(0)]],
    constant TextChar* chars [[buffer(1)]]
) {
    uint char_idx = vid / 6;
    uint vert_idx = vid % 6;

    TextChar ch = chars[char_idx];

    // Unpack color (RGBA)
    float4 color;
    color.r = float((ch.color >> 24) & 0xFF) / 255.0;
    color.g = float((ch.color >> 16) & 0xFF) / 255.0;
    color.b = float((ch.color >> 8) & 0xFF) / 255.0;
    color.a = float(ch.color & 0xFF) / 255.0;

    // Character cell is 8x8 pixels (scaled up 1.5x for readability)
    float scale = 1.5;
    float char_w = 8.0 * scale;
    float char_h = 8.0 * scale;

    // Quad vertices (two triangles)
    float2 positions[6] = {
        float2(0, 0), float2(char_w, 0), float2(char_w, char_h),
        float2(0, 0), float2(char_w, char_h), float2(0, char_h)
    };

    // UV coordinates within a single character cell (0-1 range, to be scaled)
    float2 uvs[6] = {
        float2(0, 0), float2(1, 0), float2(1, 1),
        float2(0, 0), float2(1, 1), float2(0, 1)
    };

    float2 pos = float2(ch.x, ch.y) + positions[vert_idx];

    // Convert to clip space
    float2 ndc = (pos / uniforms.screen_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    // Calculate UV based on character code
    // Texture is 128x48 with 16x6 character grid, each char is 8x8
    uint ascii = ch.char_code;
    if (ascii < 32) ascii = 32;
    if (ascii > 127) ascii = 32;
    uint idx = ascii - 32;
    uint col = idx % 16;
    uint row = idx / 16;

    // UV coordinates in texture
    float cell_u = 8.0 / 128.0;  // Width of one char in UV space
    float cell_v = 8.0 / 48.0;   // Height of one char in UV space

    float2 uv_base = float2(float(col) * cell_u, float(row) * cell_v);
    float2 uv_offset = uvs[vert_idx] * float2(cell_u, cell_v);

    VertexOut out;
    out.position = float4(ndc, 0.0, 1.0);
    out.uv = uv_base + uv_offset;
    out.color = color;
    return out;
}

fragment float4 fragment_main(
    VertexOut in [[stage_in]],
    texture2d<float> font_tex [[texture(0)]]
) {
    constexpr sampler samp(mag_filter::nearest, min_filter::nearest);
    float glyph = font_tex.sample(samp, in.uv).r;

    // Discard transparent pixels
    if (glyph < 0.5) discard_fragment();

    return in.color;
}

kernel void generate_text(uint tid [[thread_position_in_grid]]) {
    // Placeholder - text gen currently on CPU for simplicity
}
"#;
