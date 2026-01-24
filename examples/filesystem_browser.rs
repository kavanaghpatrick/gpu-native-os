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
use rust_experiment::gpu_os::filesystem::{
    FileType, GpuFilesystem, GpuPathSearch,
    GpuStreamingSearch, parse_query_words, STREAM_CHUNK_SIZE,
};
use rust_experiment::gpu_os::content_search::{GpuContentSearch, ContentMatch, SearchOptions};
use rust_experiment::gpu_os::batch_io::GpuBatchLoader;
use rust_experiment::gpu_os::duplicate_finder::{GpuDuplicateFinder, DuplicateGroup};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer, colors};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::mem;
use std::path::Path;
use std::time::Instant;

const INDEX_FILE: &str = "/Users/patrickkavanagh/.filesystem_index.txt";
// 10M paths = ~2.5GB GPU memory - easily fits on Apple Silicon with unified memory
// This eliminates the need for streaming search in most cases
const MAX_GPU_PATHS: usize = 10_000_000;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey, ModifiersState},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Search mode - path search, content search, or duplicate finder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Path,       // Search file paths (default)
    Content,    // Search file contents (grep-like)
    Duplicates, // Find duplicate files
}

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

    // GPU Search (path search)
    gpu_search: Option<GpuPathSearch>,

    // GPU Streaming Search (for large filesystems with 3M+ paths)
    streaming_search: Option<GpuStreamingSearch>,
    use_streaming: bool,  // True when path count exceeds MAX_GPU_PATHS
    all_paths: Vec<String>,  // Store all paths for streaming search path lookups

    // GPU Content Search (grep-like)
    content_search: Option<GpuContentSearch>,
    content_results: Vec<ContentMatch>,

    // GPU Duplicate Finder
    duplicate_finder: Option<GpuDuplicateFinder>,
    duplicate_groups: Vec<DuplicateGroup>,

    // State
    search_query: String,
    search_mode: SearchMode,
    modifiers: ModifiersState,
    results: Vec<(String, i32, u64)>, // (path, match_score, size)
    path_to_id: HashMap<String, u32>,
    scan_complete: bool,
    content_files_loaded: bool,
    duplicates_scanned: bool,
    selected_index: usize,
    scroll_offset: usize,
    last_frame: Instant,
    status_message: String,
    file_count: usize,
    dir_count: usize,
    last_search_time_us: u64, // Track GPU search timing

    // Debouncing - only search after user stops typing
    last_keystroke: Instant,
    search_pending: bool,
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

        // Create GPU path search engine
        let gpu_search = GpuPathSearch::new(&device, MAX_GPU_PATHS)
            .map_err(|e| eprintln!("Warning: GPU search init failed: {}", e))
            .ok();

        // Create GPU content search engine
        // Content search with larger capacity for more files
        let content_search = GpuContentSearch::new(&device, 50_000)
            .map_err(|e| eprintln!("Warning: GPU content search init failed: {}", e))
            .ok();

        // Create GPU duplicate finder
        let duplicate_finder = GpuDuplicateFinder::new(&device, 50000)
            .map_err(|e| eprintln!("Warning: GPU duplicate finder init failed: {}", e))
            .ok();

        // Create GPU streaming search for large filesystems
        let streaming_search = GpuStreamingSearch::new(&device)
            .map_err(|e| eprintln!("Warning: GPU streaming search init failed: {}", e))
            .ok();

        Self {
            window: None,
            device,
            layer,
            command_queue,
            font,
            text_renderer,
            gpu_search,
            streaming_search,
            use_streaming: false,
            all_paths: Vec::new(),
            content_search,
            content_results: Vec::new(),
            duplicate_finder,
            duplicate_groups: Vec::new(),
            search_query: String::new(),
            search_mode: SearchMode::Path,
            modifiers: ModifiersState::empty(),
            results: Vec::new(),
            path_to_id: HashMap::new(),
            scan_complete: false,
            content_files_loaded: false,
            duplicates_scanned: false,
            selected_index: 0,
            scroll_offset: 0,
            last_frame: Instant::now(),
            status_message: "Scanning entire filesystem...".to_string(),
            file_count: 0,
            dir_count: 0,
            last_search_time_us: 0,
            last_keystroke: Instant::now(),
            search_pending: false,
        }
    }

    fn scan_filesystem(&mut self) {
        let start = Instant::now();

        // Try to load from cache first
        if let Some((paths, file_count, dir_count)) = load_index() {
            self.path_to_id = paths;
            self.file_count = file_count;
            self.dir_count = dir_count;

            let path_count = self.path_to_id.len();
            let gpu_start = Instant::now();

            // Check if we need streaming mode (too many paths for regular GPU search)
            if path_count > MAX_GPU_PATHS {
                // Use streaming search - don't load all paths into GPU memory
                self.use_streaming = true;
                self.all_paths = self.path_to_id.keys().cloned().collect();
                println!("Using streaming search for {} paths (exceeds {} limit)",
                    path_count, MAX_GPU_PATHS);

                self.scan_complete = true;
                self.status_message = format!(
                    "{} files, {} folders - STREAMING MODE ({}M paths) - loaded in {:.1}s",
                    self.file_count,
                    self.dir_count,
                    path_count / 1_000_000,
                    start.elapsed().as_secs_f32()
                );
                return;
            }

            // Regular GPU search - load paths into GPU buffer
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
        let path_count = self.path_to_id.len();
        let gpu_start = Instant::now();

        // Check if we need streaming mode (too many paths for regular GPU search)
        if path_count > MAX_GPU_PATHS {
            self.use_streaming = true;
            self.all_paths = self.path_to_id.keys().cloned().collect();
            println!("Using streaming search for {} paths (exceeds {} limit)",
                path_count, MAX_GPU_PATHS);

            // Save to cache
            save_index(&self.path_to_id, self.file_count, self.dir_count);

            self.status_message = format!(
                "{} files + {} folders = {} - STREAMING MODE - in {:.1}s | skip:{} err:{}",
                self.file_count,
                self.dir_count,
                total,
                start.elapsed().as_secs_f32(),
                skipped,
                errors
            );
            return;
        }

        // Load paths into GPU
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

    fn toggle_search_mode(&mut self) {
        self.search_mode = match self.search_mode {
            SearchMode::Path => {
                // Switch to content mode - load files if needed
                if !self.content_files_loaded {
                    self.load_content_files();
                }
                SearchMode::Content
            }
            SearchMode::Content => SearchMode::Path,
            SearchMode::Duplicates => SearchMode::Path,
        };
        self.search_query.clear();
        self.results.clear();
        self.content_results.clear();
        self.selected_index = 0;
        self.scroll_offset = 0;
        self.status_message = format!(
            "Search mode: {} (Ctrl+Shift+F to toggle)",
            match self.search_mode {
                SearchMode::Path => "Path Search",
                SearchMode::Content => "Content Search (grep)",
                SearchMode::Duplicates => "Duplicates",
            }
        );
    }

    fn run_duplicate_scan(&mut self) {
        self.search_mode = SearchMode::Duplicates;
        self.search_query.clear();
        self.results.clear();
        self.content_results.clear();
        self.selected_index = 0;
        self.scroll_offset = 0;
        self.status_message = "Scanning for duplicate files...".to_string();

        let scan_start = Instant::now();

        if let Some(ref mut finder) = self.duplicate_finder {
            // Scan home directory for demo
            let scan_path = std::path::Path::new("/Users/patrickkavanagh");

            match finder.scan_directory(scan_path) {
                Ok(result) => {
                    self.status_message = format!(
                        "Found {} files with potential duplicates, hashing...",
                        result.files_scanned
                    );

                    // Compute hashes
                    if let Err(e) = finder.compute_hashes() {
                        self.status_message = format!("Hash error: {}", e);
                        return;
                    }

                    // Find duplicates
                    self.duplicate_groups = finder.find_duplicates();
                    let total_wasted = finder.total_wasted_bytes(&self.duplicate_groups);

                    self.last_search_time_us = scan_start.elapsed().as_micros() as u64;
                    self.duplicates_scanned = true;

                    // Convert to results format for display
                    self.results = self.duplicate_groups
                        .iter()
                        .map(|g| {
                            let display = format!(
                                "{} files ({} each) - {} wasted",
                                g.files.len(),
                                format_size(g.file_size),
                                format_size(g.wasted_bytes)
                            );
                            (display, g.files.len() as i32, g.wasted_bytes)
                        })
                        .collect();

                    self.status_message = format!(
                        "{} duplicate groups, {} wasted ({:.1}ms GPU)",
                        self.duplicate_groups.len(),
                        format_size(total_wasted),
                        self.last_search_time_us as f64 / 1000.0
                    );
                }
                Err(e) => {
                    self.status_message = format!("Scan error: {}", e);
                }
            }
        } else {
            self.status_message = "Duplicate finder not available".to_string();
        }
    }

    fn load_content_files(&mut self) {
        // Load files for content searching using GPU-direct I/O (MTLIOCommandQueue)
        // This bypasses CPU entirely for maximum performance
        self.status_message = "Loading files via GPU-direct I/O...".to_string();

        if let Some(ref mut content_search) = self.content_search {
            // Get searchable file paths from the path index (up to 10K files)
            let paths: Vec<std::path::PathBuf> = self.path_to_id
                .keys()
                .filter(|p| {
                    // Only include text-like files
                    let ext = std::path::Path::new(p)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    matches!(ext, "rs" | "py" | "js" | "ts" | "json" | "toml" | "yaml" | "yml" |
                            "md" | "txt" | "c" | "h" | "cpp" | "hpp" | "go" | "java" | "swift" |
                            "rb" | "sh" | "bash" | "zsh" | "css" | "html" | "xml" | "sql")
                })
                .take(10_000)  // Increased from 1000 - GPU can handle more
                .map(|s| std::path::PathBuf::from(s))
                .collect();

            let load_start = Instant::now();

            // Try GPU-direct I/O first (MTLIOCommandQueue - bypasses CPU)
            if let Some(batch_loader) = GpuBatchLoader::new(&self.device) {
                if let Some(batch_result) = batch_loader.load_batch(&paths) {
                    match content_search.load_from_batch(&batch_result) {
                        Ok(chunks) => {
                            self.content_files_loaded = true;
                            let load_time = load_start.elapsed();
                            self.status_message = format!(
                                "GPU-direct: {} files ({} chunks, {:.1}MB) in {:.0}ms",
                                content_search.file_count(),
                                chunks,
                                batch_result.total_bytes as f64 / (1024.0 * 1024.0),
                                load_time.as_secs_f64() * 1000.0
                            );
                            return;
                        }
                        Err(e) => {
                            eprintln!("GPU batch load failed: {}", e);
                        }
                    }
                }
            }

            // Fallback: CPU-based loading (slower but always works)
            let path_refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
            match content_search.load_files(&path_refs) {
                Ok(chunks) => {
                    self.content_files_loaded = true;
                    let load_time = load_start.elapsed();
                    self.status_message = format!(
                        "CPU fallback: {} files ({} chunks) in {:.0}ms",
                        content_search.file_count(),
                        chunks,
                        load_time.as_secs_f64() * 1000.0
                    );
                }
                Err(e) => {
                    self.status_message = format!("Error loading files: {}", e);
                }
            }
        }
    }

    fn perform_search(&mut self) {
        if self.search_query.is_empty() {
            self.results.clear();
            self.content_results.clear();
            self.last_search_time_us = 0;
            return;
        }

        let search_start = Instant::now();

        match self.search_mode {
            SearchMode::Path => {
                if self.use_streaming {
                    // Use GPU streaming search for large filesystems
                    self.perform_streaming_search();
                    self.last_search_time_us = search_start.elapsed().as_micros() as u64;
                } else if let Some(ref gpu_search) = self.gpu_search {
                    // Use regular GPU path search
                    let matches = gpu_search.search(&self.search_query, 500);
                    self.last_search_time_us = search_start.elapsed().as_micros() as u64;

                    self.results = matches
                        .iter()
                        .filter_map(|(idx, score)| {
                            let path = gpu_search.get_path(*idx)?;
                            let metadata = fs::metadata(path).ok()?;
                            if metadata.is_dir() {
                                return None;
                            }
                            Some((path.to_string(), *score, metadata.len()))
                        })
                        .collect();
                }
            }
            SearchMode::Content => {
                // Use GPU content search
                if let Some(ref content_search) = self.content_search {
                    let options = SearchOptions::default();
                    self.content_results = content_search.search(&self.search_query, &options);
                    self.last_search_time_us = search_start.elapsed().as_micros() as u64;

                    // Convert content matches to display format
                    self.results = self.content_results
                        .iter()
                        .map(|m| {
                            let display = format!("{}:{}", m.file_path, m.line_number);
                            (display, m.line_number as i32, 0)
                        })
                        .collect();
                }
            }
            SearchMode::Duplicates => {
                // Duplicates mode doesn't use the search query
                // Results are already populated by run_duplicate_scan
                return;
            }
        }

        self.selected_index = 0;
        self.scroll_offset = 0;
    }

    /// Perform streaming search for large filesystems (3M+ paths)
    /// Processes paths in 50K chunks to avoid GPU memory exhaustion
    fn perform_streaming_search(&mut self) {
        let Some(ref mut streaming) = self.streaming_search else {
            self.status_message = "Streaming search not available".to_string();
            return;
        };

        // Parse query into words
        let query_words = parse_query_words(&self.search_query);
        if query_words.is_empty() {
            return;
        }

        // Reset streaming search state
        streaming.reset();

        // Process all paths in chunks
        let chunk_size = STREAM_CHUNK_SIZE;
        let total_paths = self.all_paths.len();
        let mut chunk_offset = 0;
        let mut chunks_processed = 0;

        while chunk_offset < total_paths {
            let chunk_end = (chunk_offset + chunk_size).min(total_paths);
            let chunk: Vec<String> = self.all_paths[chunk_offset..chunk_end].to_vec();

            streaming.process_chunk(&chunk, chunk_offset, &query_words);

            chunk_offset = chunk_end;
            chunks_processed += 1;
        }

        // Get sorted results
        let stream_results = streaming.get_results();

        // Convert to display format
        self.results = stream_results
            .iter()
            .filter_map(|r| {
                let path = self.all_paths.get(r.path_index as usize)?;
                let metadata = fs::metadata(path).ok()?;
                if metadata.is_dir() {
                    return None;
                }
                Some((path.clone(), r.score, metadata.len()))
            })
            .take(500)  // Limit results for display
            .collect();

        let (total_searched, _) = streaming.stats();
        self.status_message = format!(
            "Streaming: {} results from {}M paths ({} chunks)",
            self.results.len(),
            total_searched / 1_000_000,
            chunks_processed
        );
    }

    fn render(&mut self) {
        static mut FRAME_COUNT: u64 = 0;
        static mut LAST_DEBUG_SECS: u64 = 0;

        unsafe {
            FRAME_COUNT += 1;
        }

        // Debounce: only search after 150ms of no typing
        const DEBOUNCE_MS: u128 = 150;
        if self.search_pending && self.last_keystroke.elapsed().as_millis() >= DEBOUNCE_MS {
            self.search_pending = false;
            self.perform_search();
        }

        // Clear text renderer for new frame
        self.text_renderer.clear();

        // Build the UI text
        self.build_ui();

        // DEBUG: Print text stats every 60 frames
        unsafe {
            if FRAME_COUNT % 60 == 1 {
                println!("[DEBUG] Frame {} | Chars: {} | Scale: {}",
                    FRAME_COUNT,
                    self.text_renderer.char_count(),
                    self.text_renderer.scale
                );
            }
        }

        // Get drawable
        let drawable = match self.layer.next_drawable() {
            Some(d) => d,
            None => {
                println!("[DEBUG] No drawable available!");
                return;
            }
        };

        // DEBUG: Check drawable texture dimensions
        unsafe {
            if FRAME_COUNT == 1 {
                let tex = drawable.texture();
                println!("[DEBUG] Drawable texture: {}x{} format={:?}",
                    tex.width(), tex.height(), tex.pixel_format());
            }
        }

        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();

        // Render pass
        let render_desc = RenderPassDescriptor::new();
        let color_attachment = render_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(drawable.texture()));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        // DEBUG: Use bright red clear color to verify render pass works
        color_attachment.set_clear_color(MTLClearColor::new(0.3, 0.0, 0.0, 1.0));
        color_attachment.set_store_action(MTLStoreAction::Store);

        // Render text using bitmap font
        let encoder = command_buffer.new_render_command_encoder(&render_desc);
        if self.text_renderer.char_count() > 0 {
            self.text_renderer.render(
                &encoder,
                &self.font,
                WIDTH as f32,
                HEIGHT as f32,
            );
        }
        encoder.end_encoding();

        command_buffer.present_drawable(&drawable);
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // DEBUG: After GPU is done, check vertex data
        unsafe {
            if FRAME_COUNT == 3 {
                self.text_renderer.debug_atlas_info();
                self.text_renderer.debug_dump_vertices(18);
            }
        }

        self.last_frame = Instant::now();
    }

    fn build_ui(&mut self) {
        let line_height = self.text_renderer.line_height();
        let mut y = 30.0;

        // Title with search mode indicator
        let mode_str = match self.search_mode {
            SearchMode::Path => "[Path]",
            SearchMode::Content => "[Content]",
            SearchMode::Duplicates => "[Duplicates]",
        };
        let title = format!("GPU Filesystem Browser {}", mode_str);
        self.text_renderer.add_text(&title, 20.0, y, colors::WHITE);

        // Mode hint on right side
        let hint = "Ctrl+Shift+F: toggle | Ctrl+D: duplicates";
        let hint_x = WIDTH as f32 - self.text_renderer.text_width(hint) - 20.0;
        self.text_renderer.add_text(hint, hint_x, y, colors::DARK_GRAY);
        y += line_height * 1.2;

        // Search box with mode-specific label (not shown in Duplicates mode)
        if self.search_mode != SearchMode::Duplicates {
            let search_label = match self.search_mode {
                SearchMode::Path => "Search: ",
                SearchMode::Content => "Grep: ",
                SearchMode::Duplicates => "",
            };
            self.text_renderer.add_text(search_label, 20.0, y, colors::GRAY);
            let query_x = 20.0 + self.text_renderer.text_width(search_label);

            if self.search_query.is_empty() {
                self.text_renderer.add_text("Type to search...", query_x, y, colors::DARK_GRAY);
            } else {
                self.text_renderer.add_text(&self.search_query, query_x, y, colors::WHITE);
            }

            // Cursor
            let cursor_x = query_x + self.text_renderer.text_width(&self.search_query);
            self.text_renderer.add_text("_", cursor_x, y, colors::GREEN);
            y += line_height * 1.2;
        } else {
            // In duplicates mode, show status instead of search box
            self.text_renderer.add_text("Duplicate files found:", 20.0, y, colors::GRAY);
            y += line_height * 1.2;
        }

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

            // Truncate path but ALWAYS show full filename
            // Leave room for file size column (10 chars) on right
            let max_path_len = 70;
            let display_path = if path.len() > max_path_len {
                // Find the filename (after last /)
                let filename_start = path.rfind('/').map(|i| i + 1).unwrap_or(0);
                let filename = &path[filename_start..];
                let dir_path = &path[..filename_start];

                // Calculate how much space we have for directory
                let available_for_dir = max_path_len.saturating_sub(filename.len() + 4); // 4 for ".../"

                if available_for_dir > 10 && dir_path.len() > available_for_dir {
                    // Truncate directory, keep filename
                    format!("...{}{}", &dir_path[dir_path.len() - available_for_dir..], filename)
                } else if filename.len() > max_path_len - 3 {
                    // Filename itself is too long, truncate it
                    format!("...{}", &filename[filename.len() - (max_path_len - 3)..])
                } else {
                    // Just show filename with ... prefix
                    format!(".../{}", filename)
                }
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

        // Check for Ctrl+Shift+F to toggle content search mode
        if self.modifiers.control_key() && self.modifiers.shift_key() {
            if let Key::Character(c) = &key {
                if c.as_str().eq_ignore_ascii_case("f") {
                    self.toggle_search_mode();
                    return;
                }
            }
        }

        // Check for Ctrl+D to run duplicate scan
        if self.modifiers.control_key() && !self.modifiers.shift_key() {
            if let Key::Character(c) = &key {
                if c.as_str().eq_ignore_ascii_case("d") {
                    self.run_duplicate_scan();
                    return;
                }
            }
        }

        match key {
            Key::Character(c) => {
                let s = c.as_str();
                if s.chars().all(|c| c.is_ascii() && !c.is_control()) {
                    self.search_query.push_str(s);
                    self.last_keystroke = Instant::now();
                    self.search_pending = true;
                }
            }
            Key::Named(NamedKey::Space) => {
                self.search_query.push(' ');
                self.last_keystroke = Instant::now();
                self.search_pending = true;
            }
            Key::Named(NamedKey::Backspace) => {
                self.search_query.pop();
                self.last_keystroke = Instant::now();
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
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
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
