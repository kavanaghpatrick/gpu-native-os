// GPU Filesystem Browser with Visual UI
//
// Interactive filesystem browser with Metal-rendered UI

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::filesystem::{GpuFilesystem, FileType};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct FilesystemBrowser {
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<CommandQueue>,
    pipeline: Option<RenderPipelineState>,
    filesystem: Option<Arc<Mutex<GpuFilesystem>>>,

    // UI state
    search_input: String,
    search_results: Vec<SearchResult>,
    scan_root: String,
    scan_complete: bool,
    total_files: usize,

    // Timing
    last_search_time: f64,
}

#[derive(Clone)]
struct SearchResult {
    path: String,
    inode: u32,
    is_hit: bool,
}

impl FilesystemBrowser {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            layer: None,
            command_queue: None,
            pipeline: None,
            filesystem: None,
            search_input: String::new(),
            search_results: Vec::new(),
            scan_root: std::env::current_dir()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
            scan_complete: false,
            total_files: 0,
            last_search_time: 0.0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("GPU Filesystem Browser");
        println!("GPU: {}", device.name());

        // Set up Metal layer
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            let size = window.inner_size();
            layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

            match window.window_handle().unwrap().as_raw() {
                RawWindowHandle::AppKit(handle) => {
                    let view = handle.ns_view.as_ptr() as cocoa_id;
                    view.setWantsLayer(YES);
                    view.setLayer(std::mem::transmute(layer.as_ref()));
                }
                _ => panic!("Unsupported platform"),
            }
        }

        let command_queue = device.new_command_queue();

        // Create simple render pipeline
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("Failed to compile shaders");

        let pipeline_desc = RenderPipelineDescriptor::new();
        let vert = library.get_function("vertex_main", None).unwrap();
        let frag = library.get_function("fragment_main", None).unwrap();
        pipeline_desc.set_vertex_function(Some(&vert));
        pipeline_desc.set_fragment_function(Some(&frag));
        pipeline_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let pipeline = device
            .new_render_pipeline_state(&pipeline_desc)
            .expect("Failed to create pipeline");

        println!("\n📁 Scanning directory: {}", self.scan_root);
        println!("Please wait...\n");

        // Load filesystem in background
        let scan_root = self.scan_root.clone();
        let filesystem = Arc::new(Mutex::new(
            GpuFilesystem::new(&device, 50_000).expect("Failed to create filesystem"),
        ));

        let fs_clone = filesystem.clone();
        std::thread::spawn(move || {
            let mut stats = ScanStats {
                files: 0,
                dirs: 0,
            };

            let path = Path::new(&scan_root);
            let mut fs = fs_clone.lock().unwrap();
            scan_directory(path, 0, &mut *fs, &mut stats, 0);

            println!("✅ Scan complete: {} files, {} dirs", stats.files, stats.dirs);
        });

        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.pipeline = Some(pipeline);
        self.filesystem = Some(filesystem);
        self.window = Some(window);
    }

    fn render(&mut self) {
        let Some(ref layer) = self.layer else { return };
        let Some(ref command_queue) = self.command_queue else { return };
        let Some(ref pipeline) = self.pipeline else { return };

        autoreleasepool(|| {
            let drawable = match layer.next_drawable() {
                Some(d) => d,
                None => return,
            };

            let render_pass_descriptor = RenderPassDescriptor::new();
            let color_attachment = render_pass_descriptor
                .color_attachments()
                .object_at(0)
                .unwrap();

            color_attachment.set_texture(Some(drawable.texture()));
            color_attachment.set_load_action(MTLLoadAction::Clear);
            color_attachment.set_clear_color(MTLClearColor::new(0.1, 0.1, 0.15, 1.0));
            color_attachment.set_store_action(MTLStoreAction::Store);

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_render_command_encoder(&render_pass_descriptor);

            encoder.set_render_pipeline_state(pipeline);

            // Draw UI quads here (simplified - just clear for now)
            encoder.end_encoding();

            command_buffer.present_drawable(drawable);
            command_buffer.commit();
            command_buffer.wait_until_completed(); // CRITICAL: Prevent GPU queue overflow
        });
    }

    fn handle_key(&mut self, key: Key, state: ElementState) {
        if state != ElementState::Pressed {
            return;
        }

        match key {
            Key::Character(c) => {
                self.search_input.push_str(&c);
                self.perform_search();
            }
            Key::Named(NamedKey::Backspace) => {
                self.search_input.pop();
                self.perform_search();
            }
            Key::Named(NamedKey::Enter) => {
                self.perform_search();
            }
            Key::Named(NamedKey::Escape) => {
                self.search_input.clear();
                self.search_results.clear();
            }
            _ => {}
        }

        println!("\nSearch: {}", self.search_input);
        if !self.search_results.is_empty() {
            println!("Results ({}):", self.search_results.len());
            for result in &self.search_results {
                let marker = if result.is_hit { "⚡" } else { "🔍" };
                println!("  {} {} → inode {}", marker, result.path, result.inode);
            }
            println!("Search time: {:.1}µs", self.last_search_time * 1000.0);
        }
    }

    fn perform_search(&mut self) {
        if self.search_input.is_empty() {
            self.search_results.clear();
            return;
        }

        let Some(ref fs) = self.filesystem else { return };

        // Try to lock filesystem (might be scanning)
        let Ok(mut fs) = fs.try_lock() else {
            return;
        };

        let search_path = if self.search_input.starts_with('/') {
            self.search_input.clone()
        } else {
            format!("/{}", self.search_input)
        };

        let start = Instant::now();
        self.search_results.clear();

        match fs.lookup_path(&search_path) {
            Ok(inode) => {
                self.search_results.push(SearchResult {
                    path: search_path,
                    inode,
                    is_hit: true,
                });
            }
            Err(_) => {
                // Try partial matches
                // For demo, just show "not found"
            }
        }

        self.last_search_time = start.elapsed().as_secs_f64() * 1_000_000.0;
    }
}

impl ApplicationHandler for FilesystemBrowser {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("GPU Filesystem Browser")
            .with_inner_size(LogicalSize::new(800, 600));

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // CRITICAL: Wrap ENTIRE event handler in autoreleasepool to drain Metal objects
        autoreleasepool(|| {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.render();
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    self.handle_key(event.logical_key, event.state);
                    if let Some(ref window) = self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::Resized(size) => {
                    if let Some(ref layer) = self.layer {
                        layer.set_drawable_size(CGSize::new(
                            size.width as f64,
                            size.height as f64,
                        ));
                    }
                }
                _ => {}
            }
        });
    }
}

struct ScanStats {
    files: usize,
    dirs: usize,
}

fn scan_directory(
    path: &Path,
    parent_inode: u32,
    fs: &mut GpuFilesystem,
    stats: &mut ScanStats,
    depth: usize,
) {
    if depth > 15 {
        return;
    }

    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else { continue };

        if name.len() > 20 {
            continue;
        }

        let Ok(metadata) = entry.metadata() else { continue };

        let file_type = if metadata.is_dir() {
            FileType::Directory
        } else {
            FileType::Regular
        };

        match fs.add_file(parent_inode, name, file_type) {
            Ok(inode_id) => {
                if metadata.is_dir() {
                    stats.dirs += 1;
                    scan_directory(&entry.path(), inode_id, fs, stats, depth + 1);
                } else {
                    stats.files += 1;
                }
            }
            Err(_) => {}
        }
    }
}

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
    VertexOut out;
    out.position = float4(0.0, 0.0, 0.0, 1.0);
    out.color = float4(1.0, 1.0, 1.0, 1.0);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
"#;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          GPU Filesystem Browser - Visual UI             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    println!("Instructions:");
    println!("  • Window will open and load current directory into GPU");
    println!("  • Type to search (e.g., 'src', 'Cargo.toml')");
    println!("  • ESC to clear search");
    println!("  • Check terminal for results\n");
    println!("Loading...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = FilesystemBrowser::new();
    event_loop.run_app(&mut app).unwrap();
}
