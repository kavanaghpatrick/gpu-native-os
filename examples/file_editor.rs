// GPU-Native File Editor
//
// Combined filesystem browser + text editor
// - Search for files with GPU-accelerated path matching
// - Press Enter to open file in GPU text editor
// - Press Escape to return to file browser
//
// Launch: cargo run --release --example file_editor

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::filesystem::GpuPathSearch;
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer, colors};
use rust_experiment::gpu_os::text_editor::{
    TextEditor, EDIT_MOVE_LEFT, EDIT_MOVE_RIGHT, EDIT_MOVE_UP, EDIT_MOVE_DOWN,
    EDIT_MOVE_HOME, EDIT_MOVE_END,
};
use rust_experiment::gpu_os::app::GpuRuntime;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey, ModifiersState},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

const INDEX_FILE: &str = "/Users/patrickkavanagh/.filesystem_index.txt";
const MAX_GPU_PATHS: usize = 500_000;
const WIDTH: u32 = 1200;
const HEIGHT: u32 = 800;
const MAX_RESULTS: usize = 25;

/// App mode - either browsing files or editing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Browse,
    Edit,
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = FileEditorApp::new();
    event_loop.run_app(&mut app).unwrap();
}

struct FileEditorApp {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<CommandQueue>,

    // Text rendering for browser UI
    font: Option<BitmapFont>,
    text_renderer: Option<TextRenderer>,

    // GPU Search
    gpu_search: Option<GpuPathSearch>,

    // Text Editor
    runtime: Option<GpuRuntime>,
    editor: Option<TextEditor>,

    // State
    mode: AppMode,
    search_query: String,
    modifiers: ModifiersState,
    results: Vec<(usize, i32)>, // (path_index, match_score)
    all_paths: Vec<String>,    // Store paths for lookup
    all_sizes: Vec<u64>,       // Store sizes for display
    scan_complete: bool,
    selected_index: usize,
    scroll_offset: usize,
    last_frame: Instant,
    status_message: String,
    file_count: usize,
    current_file: Option<String>,
    search_pending: bool,
    last_search_time_us: u64,
}

impl FileEditorApp {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            command_queue: None,
            font: None,
            text_renderer: None,
            gpu_search: None,
            runtime: None,
            editor: None,
            mode: AppMode::Browse,
            search_query: String::new(),
            modifiers: ModifiersState::empty(),
            results: Vec::new(),
            all_paths: Vec::new(),
            all_sizes: Vec::new(),
            scan_complete: false,
            selected_index: 0,
            scroll_offset: 0,
            last_frame: Instant::now(),
            status_message: "Loading filesystem index...".to_string(),
            file_count: 0,
            current_file: None,
            search_pending: false,
            last_search_time_us: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("GPU-Native File Editor");
        println!("======================");
        println!("GPU: {}", device.name());

        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            if let Ok(handle) = window.window_handle() {
                if let RawWindowHandle::AppKit(appkit_handle) = handle.as_raw() {
                    let view = appkit_handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(objc::runtime::YES);
                    view.setLayer(layer.as_ref() as *const _ as *mut _);
                }
            }
        }

        layer.set_drawable_size(CGSize::new(WIDTH as f64, HEIGHT as f64));

        let command_queue = device.new_command_queue();
        let font = BitmapFont::new(&device);
        let text_renderer = TextRenderer::new(&device, 20000)
            .expect("Failed to create TextRenderer");

        // Initialize text editor
        let runtime = GpuRuntime::new(device.clone());
        let editor = TextEditor::new(&device).expect("Failed to create TextEditor");

        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.font = Some(font);
        self.text_renderer = Some(text_renderer);
        self.runtime = Some(runtime);
        self.editor = Some(editor);
        self.window = Some(window);

        // Load filesystem index
        self.load_index();
    }

    fn load_index(&mut self) {
        if !Path::new(INDEX_FILE).exists() {
            self.status_message = format!("Index not found. Run: find ~ -type f > {}", INDEX_FILE);
            return;
        }

        let file = fs::File::open(INDEX_FILE).expect("Failed to open index");
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Skip header line (file_count dir_count)
        if let Some(Ok(_header)) = lines.next() {
            // Header parsed, continue
        }

        // Read paths - format is "id path"
        // Don't check file sizes during load (too slow for 500K files)
        for line in lines.take(MAX_GPU_PATHS) {
            if let Ok(line) = line {
                // Split on first space: "id path"
                if let Some(space_idx) = line.find(' ') {
                    let path = &line[space_idx + 1..];
                    self.all_paths.push(path.to_string());
                    self.all_sizes.push(0); // Don't fetch size during load
                }
            }
        }

        self.file_count = self.all_paths.len();

        if self.file_count == 0 {
            self.status_message = "No files found in index".to_string();
            return;
        }

        // Initialize GPU search
        let device = self.device.as_ref().unwrap();
        let mut gpu_search = GpuPathSearch::new(device, MAX_GPU_PATHS)
            .expect("Failed to create GPU search");
        gpu_search.add_paths(&self.all_paths).expect("Failed to add paths");
        self.gpu_search = Some(gpu_search);

        self.scan_complete = true;
        self.status_message = format!("{} files indexed. Type to search...", self.file_count);
    }

    fn perform_search(&mut self) {
        if !self.scan_complete || self.search_query.is_empty() {
            self.results.clear();
            return;
        }

        let gpu_search = match &self.gpu_search {
            Some(s) => s,
            None => return,
        };

        let start = Instant::now();
        let matches = gpu_search.search(&self.search_query, 200);
        self.last_search_time_us = start.elapsed().as_micros() as u64;

        self.results = matches
            .into_iter()
            .filter(|(_, score)| *score > 0)
            .collect();

        self.selected_index = 0;
        self.scroll_offset = 0;
    }

    fn open_selected_file(&mut self) {
        if let Some((path_idx, _)) = self.results.get(self.selected_index) {
            if let Some(path) = self.all_paths.get(*path_idx) {
                let path = path.clone();
                let device = self.device.as_ref().unwrap().clone();

                if let Some(editor) = &mut self.editor {
                    // Try GPU-Direct I/O first (Metal 3), fall back to CPU
                    let result = editor.load_file_gpu_direct(&device, &path)
                        .or_else(|_| editor.load_file(&path));

                    match result {
                        Ok(()) => {
                            self.current_file = Some(path.clone());
                            self.mode = AppMode::Edit;
                            self.status_message = format!("Opened: {}", path);
                        }
                        Err(e) => {
                            self.status_message = format!("Error: {}", e);
                        }
                    }
                }
            }
        }
    }

    fn render(&mut self) {
        let layer = match self.layer.as_ref() {
            Some(l) => l,
            None => return,
        };
        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        let mode = self.mode;

        match mode {
            AppMode::Browse => {
                // Browse mode rendering
                let text_renderer = self.text_renderer.as_mut().unwrap();
                let command_queue = self.command_queue.as_ref().unwrap();
                let font = self.font.as_ref().unwrap();

                text_renderer.clear();

                let line_height = 22.0;
                let mut y = 20.0;

                // Title
                text_renderer.add_text("GPU File Editor - Browse Mode", 20.0, y, colors::CYAN);
                y += line_height;

                // Search box
                let search_display = if self.search_query.is_empty() {
                    "Type to search files... (Enter to open, Esc to clear)".to_string()
                } else {
                    self.search_query.clone()
                };
                text_renderer.add_text(&format!("> {}_", search_display), 20.0, y, colors::GREEN);
                y += line_height * 1.5;

                // Status
                text_renderer.add_text(&self.status_message, 20.0, y, colors::GRAY);
                y += line_height * 1.5;

                // Results header
                if !self.results.is_empty() {
                    let timing = if self.last_search_time_us > 1000 {
                        format!("{:.1}ms", self.last_search_time_us as f64 / 1000.0)
                    } else {
                        format!("{}us", self.last_search_time_us)
                    };
                    text_renderer.add_text(
                        &format!("{} results ({}) - Up/Down to select, Enter to open", self.results.len(), timing),
                        20.0, y, colors::GRAY
                    );
                    y += line_height;
                }

                // Results list
                let visible_results: Vec<_> = self.results
                    .iter()
                    .skip(self.scroll_offset)
                    .take(MAX_RESULTS)
                    .collect();

                for (i, (path_idx, _score)) in visible_results.iter().enumerate() {
                    let actual_index = i + self.scroll_offset;
                    let is_selected = actual_index == self.selected_index;

                    let text_color = if is_selected { colors::GREEN } else { colors::LIGHT_GRAY };
                    let prefix = if is_selected { "> " } else { "  " };

                    let path = self.all_paths.get(*path_idx).map(|s| s.as_str()).unwrap_or("???");
                    let size = self.all_sizes.get(*path_idx).copied().unwrap_or(0);

                    // Truncate path for display
                    let display_path = if path.len() > 90 {
                        format!("...{}", &path[path.len()-87..])
                    } else {
                        path.to_string()
                    };

                    let size_str = if size > 1_000_000 {
                        format!("{:.1}MB", size as f64 / 1_000_000.0)
                    } else if size > 1_000 {
                        format!("{:.1}KB", size as f64 / 1_000.0)
                    } else {
                        format!("{}B", size)
                    };

                    text_renderer.add_text(
                        &format!("{}{} ({})", prefix, display_path, size_str),
                        20.0, y, text_color
                    );
                    y += line_height;
                }

                // Render
                let command_buffer = command_queue.new_command_buffer();

                let render_pass = {
                    let desc = RenderPassDescriptor::new();
                    let attachment = desc.color_attachments().object_at(0).unwrap();
                    attachment.set_texture(Some(drawable.texture()));
                    attachment.set_load_action(MTLLoadAction::Clear);
                    attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));
                    attachment.set_store_action(MTLStoreAction::Store);
                    desc
                };

                let encoder = command_buffer.new_render_command_encoder(&render_pass);
                text_renderer.render(&encoder, font, WIDTH as f32, HEIGHT as f32);
                encoder.end_encoding();

                command_buffer.present_drawable(&drawable);
                command_buffer.commit();
            }
            AppMode::Edit => {
                // Edit mode rendering - just run the editor frame
                let runtime = self.runtime.as_mut().unwrap();
                let editor = self.editor.as_mut().unwrap();
                runtime.run_frame(editor, &drawable);
                // Note: run_frame already presents and commits the drawable
            }
        }

        self.last_frame = Instant::now();
    }

    fn handle_key_browse(&mut self, key: Key) {
        match key {
            Key::Character(c) => {
                self.search_query.push_str(&c);
                self.search_pending = true;
            }
            Key::Named(NamedKey::Backspace) => {
                self.search_query.pop();
                self.search_pending = true;
            }
            Key::Named(NamedKey::Escape) => {
                self.search_query.clear();
                self.results.clear();
                self.search_pending = false;
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
                self.open_selected_file();
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

        if self.search_pending {
            self.perform_search();
            self.search_pending = false;
        }
    }

    fn handle_key_edit(&mut self, event: &KeyEvent) {
        let editor = match &mut self.editor {
            Some(e) => e,
            None => return,
        };

        // Escape returns to browse mode
        if event.logical_key == Key::Named(NamedKey::Escape) {
            self.mode = AppMode::Browse;
            self.status_message = "Returned to browse mode".to_string();
            return;
        }

        match &event.logical_key {
            Key::Character(c) => {
                for ch in c.chars() {
                    editor.insert_char(ch);
                }
            }
            Key::Named(NamedKey::Backspace) => editor.delete_back(),
            Key::Named(NamedKey::Delete) => editor.delete_forward(),
            Key::Named(NamedKey::Enter) => editor.newline(),
            Key::Named(NamedKey::ArrowLeft) => editor.move_cursor(EDIT_MOVE_LEFT),
            Key::Named(NamedKey::ArrowRight) => editor.move_cursor(EDIT_MOVE_RIGHT),
            Key::Named(NamedKey::ArrowUp) => editor.move_cursor(EDIT_MOVE_UP),
            Key::Named(NamedKey::ArrowDown) => editor.move_cursor(EDIT_MOVE_DOWN),
            Key::Named(NamedKey::Home) => editor.move_cursor(EDIT_MOVE_HOME),
            Key::Named(NamedKey::End) => editor.move_cursor(EDIT_MOVE_END),
            _ => {}
        }
    }
}

impl ApplicationHandler for FileEditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("GPU File Editor")
                .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT));
            let window = event_loop.create_window(attrs).unwrap();
            self.initialize(window);
            // Request initial redraw
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. },
                ..
            } if self.mode == AppMode::Browse => {
                self.handle_key_browse(logical_key);
            }
            WindowEvent::KeyboardInput {
                event,
                ..
            } if self.mode == AppMode::Edit && event.state == ElementState::Pressed => {
                self.handle_key_edit(&event);
            }
            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
