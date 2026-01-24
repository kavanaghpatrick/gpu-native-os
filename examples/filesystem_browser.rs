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
use rust_experiment::gpu_os::filesystem::{FileType, GpuFilesystem, GpuPathSearch};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer, colors};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::mem;
use std::path::Path;
use std::time::Instant;

const INDEX_FILE: &str = "/Users/patrickkavanagh/.filesystem_index.txt";
const MAX_GPU_PATHS: usize = 3_000_000; // Support up to 3M paths on GPU
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

    // GPU Search
    gpu_search: Option<GpuPathSearch>,

    // State
    search_query: String,
    results: Vec<(String, i32, u64)>, // (path, match_score, size)
    path_to_id: HashMap<String, u32>,
    scan_complete: bool,
    selected_index: usize,
    scroll_offset: usize,
    last_frame: Instant,
    status_message: String,
    file_count: usize,
    dir_count: usize,
    last_search_time_us: u64, // Track GPU search timing
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

        // Create GPU search engine
        let gpu_search = GpuPathSearch::new(&device, MAX_GPU_PATHS)
            .map_err(|e| eprintln!("Warning: GPU search init failed: {}", e))
            .ok();

        Self {
            window: None,
            device,
            layer,
            command_queue,
            font,
            text_renderer,
            gpu_search,
            search_query: String::new(),
            results: Vec::new(),
            path_to_id: HashMap::new(),
            scan_complete: false,
            selected_index: 0,
            scroll_offset: 0,
            last_frame: Instant::now(),
            status_message: "Scanning entire filesystem...".to_string(),
            file_count: 0,
            dir_count: 0,
            last_search_time_us: 0,
        }
    }

    fn scan_filesystem(&mut self) {
        let start = Instant::now();

        // Try to load from cache first
        if let Some((paths, file_count, dir_count)) = load_index() {
            self.path_to_id = paths;
            self.file_count = file_count;
            self.dir_count = dir_count;

            // Load paths into GPU
            let gpu_start = Instant::now();
            if let Some(ref mut gpu_search) = self.gpu_search {
                let path_list: Vec<String> = self.path_to_id.keys().cloned().collect();
                if let Err(e) = gpu_search.add_paths(&path_list) {
                    eprintln!("GPU path load error: {}", e);
                }
            }
            let gpu_load_time = gpu_start.elapsed();

            self.scan_complete = true;
            self.status_message = format!(
                "{} files, {} folders loaded from cache in {:.1}s (GPU: {:.0}ms)",
                self.file_count,
                self.dir_count,
                start.elapsed().as_secs_f32(),
                gpu_load_time.as_secs_f64() * 1000.0
            );
            return;
        }

        // No cache, scan filesystem
        let base_path = "/";

        let mut fs = match GpuFilesystem::new(&self.device, 3000000) {
            Ok(f) => f,
            Err(e) => {
                self.status_message = format!("Error: {}", e);
                return;
            }
        };

        let mut skipped = 0usize;
        let mut errors = 0usize;
        let mut depth_limited = 0usize;
        scan_recursive(
            Path::new(base_path),
            0,
            &mut fs,
            &mut self.path_to_id,
            &mut self.file_count,
            &mut self.dir_count,
            &mut skipped,
            &mut errors,
            &mut depth_limited,
            0,
        );

        self.scan_complete = true;

        let total = self.file_count + self.dir_count;

        // Load paths into GPU
        let gpu_start = Instant::now();
        if let Some(ref mut gpu_search) = self.gpu_search {
            let path_list: Vec<String> = self.path_to_id.keys().cloned().collect();
            if let Err(e) = gpu_search.add_paths(&path_list) {
                eprintln!("GPU path load error: {}", e);
            }
        }
        let gpu_load_time = gpu_start.elapsed();

        // Save to cache
        save_index(&self.path_to_id, self.file_count, self.dir_count);

        self.status_message = format!(
            "{} files + {} folders = {} in {:.1}s (GPU: {:.0}ms) | skip:{} err:{}",
            self.file_count,
            self.dir_count,
            total,
            start.elapsed().as_secs_f32(),
            gpu_load_time.as_secs_f64() * 1000.0,
            skipped,
            errors
        );
    }

    fn perform_search(&mut self) {
        if self.search_query.is_empty() {
            self.results.clear();
            self.last_search_time_us = 0;
            return;
        }

        let search_start = Instant::now();

        // Use GPU search if available
        if let Some(ref gpu_search) = self.gpu_search {
            // GPU parallel fuzzy search across all paths
            let matches = gpu_search.search(&self.search_query, 500);

            self.last_search_time_us = search_start.elapsed().as_micros() as u64;

            // Convert to results with file metadata
            self.results = matches
                .iter()
                .filter_map(|(idx, score)| {
                    let path = gpu_search.get_path(*idx)?;
                    let metadata = fs::metadata(path).ok()?;
                    // Skip directories - only show files
                    if metadata.is_dir() {
                        return None;
                    }
                    Some((path.to_string(), *score, metadata.len()))
                })
                .collect();
        } else {
            // Fallback: CPU search (shouldn't happen)
            self.results.clear();
            self.last_search_time_us = 0;
        }

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

        // Results header with GPU timing
        if !self.results.is_empty() {
            let timing = if self.last_search_time_us > 1000 {
                format!("{:.1}ms GPU", self.last_search_time_us as f64 / 1000.0)
            } else {
                format!("{}us GPU", self.last_search_time_us)
            };
            let header = format!(
                "{} results ({}) - Up/Down, Enter to copy",
                self.results.len(),
                timing
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

        for (i, (path, _score, size)) in visible_results.enumerate() {
            let actual_index = i + self.scroll_offset;
            let is_selected = actual_index == self.selected_index;

            let text_color = if is_selected {
                colors::GREEN
            } else {
                colors::LIGHT_GRAY
            };

            let prefix = if is_selected { "> " } else { "  " };

            // Truncate path to fit
            let max_path_len = 80;
            let display_path = if path.len() > max_path_len {
                format!("...{}", &path[path.len() - max_path_len + 3..])
            } else {
                path.clone()
            };

            let line = format!("{}  {}", prefix, display_path);
            self.text_renderer.add_text(&line, 20.0, y, text_color);

            // Size on the right
            if *size > 0 {
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
            Key::Named(NamedKey::Space) => {
                self.search_query.push(' ');
                self.perform_search();
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

fn save_index(paths: &HashMap<String, u32>, file_count: usize, dir_count: usize) -> bool {
    let file = match fs::File::create(INDEX_FILE) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut writer = BufWriter::new(file);

    // Write header
    if writeln!(writer, "{} {}", file_count, dir_count).is_err() {
        return false;
    }

    // Write paths
    for (path, id) in paths {
        if writeln!(writer, "{} {}", id, path).is_err() {
            return false;
        }
    }

    true
}

fn load_index() -> Option<(HashMap<String, u32>, usize, usize)> {
    let file = fs::File::open(INDEX_FILE).ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let header = lines.next()?.ok()?;
    let mut parts = header.split_whitespace();
    let file_count: usize = parts.next()?.parse().ok()?;
    let dir_count: usize = parts.next()?.parse().ok()?;

    // Read paths
    let mut paths = HashMap::new();
    for line in lines {
        let line = line.ok()?;
        let space_idx = line.find(' ')?;
        let id: u32 = line[..space_idx].parse().ok()?;
        let path = line[space_idx + 1..].to_string();
        paths.insert(path, id);
    }

    Some((paths, file_count, dir_count))
}

fn scan_recursive(
    path: &Path,
    parent_id: u32,
    fs: &mut GpuFilesystem,
    path_to_id: &mut HashMap<String, u32>,
    file_count: &mut usize,
    dir_count: &mut usize,
    skipped: &mut usize,
    errors: &mut usize,
    depth_limited: &mut usize,
    depth: usize,
) {
    if depth > 20 {
        *depth_limited += 1;
        return;
    }

    let skip_dirs = [
        ".git", "node_modules", ".Trash", "Library/Caches", ".cargo",
        "target", ".npm", ".cache", "Cache", "Caches", "DerivedData",
        "/System", "/private/var", "/dev", "/Volumes", "/cores",
        "/Library/Application Support/MobileSync", // iOS backups
    ];

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => {
            *errors += 1;
            return;
        }
    };

    for entry in entries.flatten() {
        let entry_path = entry.path();
        let full_path = entry_path.to_string_lossy().to_string();

        if skip_dirs.iter().any(|skip| full_path.contains(skip)) {
            *skipped += 1;
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

        // Use symlink_metadata to NOT follow symlinks
        let metadata = match entry.path().symlink_metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Skip symlinks entirely - they cause double-counting
        if metadata.file_type().is_symlink() {
            *skipped += 1;
            continue;
        }

        if metadata.is_dir() {
            *dir_count += 1;
            let id = match fs.add_file(parent_id, name, FileType::Directory) {
                Ok(id) => id,
                Err(_) => {
                    *errors += 1;
                    continue;
                }
            };
            path_to_id.insert(full_path.clone(), id);
            scan_recursive(&entry_path, id, fs, path_to_id, file_count, dir_count, skipped, errors, depth_limited, depth + 1);
        } else if metadata.is_file() {
            *file_count += 1;
            let id = match fs.add_file(parent_id, name, FileType::Regular) {
                Ok(id) => id,
                Err(_) => {
                    *errors += 1;
                    continue;
                }
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
