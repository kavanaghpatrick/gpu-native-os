// GPU-Native Filesystem Browser
//
// Fully GPU-rendered filesystem browser using the text_render library
// - Text rendering via GPU shaders (text_render.rs)
// - Search results rendered on GPU
// - CPU only handles keyboard input
//
// Launch: cargo run --release --example filesystem_browser

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::filesystem::{FileType, GpuFilesystem};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer, colors};
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
const MAX_RESULTS: usize = 30;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = FilesystemBrowserApp::new();
    event_loop.run_app(&mut app).unwrap();
}

struct FilesystemBrowserApp {
    window: Option<Window>,
    device: Device,
    layer: MetalLayer,
    command_queue: CommandQueue,

    // Text rendering (from library)
    font: BitmapFont,
    text_renderer: TextRenderer,

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

        // Create text rendering components from library
        let font = BitmapFont::new(&device);
        let text_renderer = TextRenderer::new(&device, 20000)
            .expect("Failed to create text renderer");

        Self {
            window: None,
            device,
            layer,
            command_queue,
            font,
            text_renderer,
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
        // Clear text renderer for new frame
        self.text_renderer.clear();

        // Build the UI text
        self.build_ui();

        // Get drawable
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

        // Render all text using the library
        self.text_renderer.render(
            encoder,
            &self.font,
            WIDTH as f32,
            HEIGHT as f32,
        );

        encoder.end_encoding();

        command_buffer.present_drawable(&drawable);
        command_buffer.commit();

        self.last_frame = Instant::now();
    }

    fn build_ui(&mut self) {
        let line_height = self.text_renderer.line_height();
        let mut y = 30.0;

        // Title
        self.text_renderer.add_text("GPU Filesystem Browser", 20.0, y, colors::WHITE);
        y += line_height * 1.2;

        // Search box
        self.text_renderer.add_text("Search: ", 20.0, y, colors::GRAY);
        let query_x = 20.0 + self.text_renderer.text_width("Search: ");

        if self.search_query.is_empty() {
            self.text_renderer.add_text("Type to search...", query_x, y, colors::DARK_GRAY);
        } else {
            self.text_renderer.add_text(&self.search_query, query_x, y, colors::WHITE);
        }

        // Cursor
        let cursor_x = query_x + self.text_renderer.text_width(&self.search_query);
        self.text_renderer.add_text("_", cursor_x, y, colors::GREEN);
        y += line_height * 1.2;

        // Separator
        self.text_renderer.add_text(&"-".repeat(60), 20.0, y, colors::DARK_GRAY);
        y += line_height;

        // Results header
        if !self.results.is_empty() {
            let header = format!(
                "{} results (Up/Down to select, Enter to copy)",
                self.results.len()
            );
            self.text_renderer.add_text(&header, 20.0, y, colors::GRAY);
            y += line_height;
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

            let text_color = if is_selected {
                colors::GREEN
            } else {
                colors::LIGHT_GRAY
            };

            let icon = if *is_dir { ">" } else { " " };
            let prefix = if is_selected { "> " } else { "  " };

            // Truncate path to fit
            let max_path_len = 80;
            let display_path = if path.len() > max_path_len {
                format!("...{}", &path[path.len() - max_path_len + 3..])
            } else {
                path.clone()
            };

            let line = format!("{}{} {}", prefix, icon, display_path);
            self.text_renderer.add_text(&line, 20.0, y, text_color);

            // Size on the right
            if !is_dir && *size > 0 {
                let size_str = format_size(*size);
                let size_x = WIDTH as f32 - 120.0;
                self.text_renderer.add_text(&size_str, size_x, y, colors::DARK_GRAY);
            }

            y += line_height;
        }

        // Empty state
        if self.results.is_empty() && !self.search_query.is_empty() {
            self.text_renderer.add_text("No results found", 20.0, y + 50.0, colors::GRAY);
        } else if self.search_query.is_empty() && self.scan_complete {
            self.text_renderer.add_text(
                "Type to search indexed files",
                20.0,
                y + 50.0,
                colors::DARK_GRAY,
            );
        }

        // Status bar at bottom
        let status_y = HEIGHT as f32 - 30.0;
        self.text_renderer.add_text(&self.status_message, 20.0, status_y, colors::GRAY);
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
                if let Some((path, _, _)) = self.results.get(self.selected_index) {
                    let _ = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(format!("echo -n '{}' | pbcopy", path))
                        .output();
                    self.status_message = format!("Copied: {}", path);
                }
            }
            Key::Named(NamedKey::PageDown) => {
                self.scroll_offset = (self.scroll_offset + MAX_RESULTS)
                    .min(self.results.len().saturating_sub(MAX_RESULTS));
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
        self.scan_filesystem();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state, .. },
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
        ".git", "node_modules", ".Trash", "Library/Caches", ".cargo",
        "target", ".npm", ".cache", "Cache", "Caches", "DerivedData",
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
