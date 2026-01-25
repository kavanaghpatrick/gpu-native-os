//! GPU Desktop Environment Demo
//!
//! A complete desktop environment running on GPU compute shaders.
//!
//! Usage:
//!   cargo run --release --example gpu_desktop
//!
//! Controls:
//! - Click and drag window title bars to move windows
//! - Click window buttons (close/minimize/maximize)
//! - Resize windows by dragging edges/corners
//! - Click dock items to launch apps (placeholder)
//! - Escape to quit

use metal::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowAttributes, WindowId};

use rust_experiment::gpu_os::desktop::{GpuDesktop, DesktopApp, AppRenderContext, AppInputEvent, AppEventType, KeyModifiers, FileBrowserApp, TerminalApp, DocumentViewerApp, TextEditorApp};
use rust_experiment::gpu_os::text_render::{BitmapFont, TextRenderer};

use std::time::Instant;

/// Simple calculator app for demo
struct CalculatorApp {
    display: String,
    result: f64,
}

impl CalculatorApp {
    fn new() -> Self {
        Self {
            display: "0".to_string(),
            result: 0.0,
        }
    }
}

impl DesktopApp for CalculatorApp {
    fn name(&self) -> &str {
        "Calculator"
    }

    fn icon_index(&self) -> u32 {
        1
    }

    fn preferred_size(&self) -> (f32, f32) {
        (280.0, 380.0)
    }

    fn render(&mut self, _ctx: &mut AppRenderContext) {
        // App would render calculator UI here
        // For now, just a placeholder
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        match event.event_type {
            AppEventType::MouseDown => {
                // Would handle button clicks
                true
            }
            AppEventType::KeyDown => {
                // Would handle number/operator keys
                true
            }
            _ => false,
        }
    }
}

/// Notes app for demo
struct NotesApp {
    text: String,
}

impl NotesApp {
    fn new() -> Self {
        Self {
            text: "Hello, GPU Desktop!".to_string(),
        }
    }
}

impl DesktopApp for NotesApp {
    fn name(&self) -> &str {
        "Notes"
    }

    fn icon_index(&self) -> u32 {
        2
    }

    fn preferred_size(&self) -> (f32, f32) {
        (400.0, 300.0)
    }

    fn render(&mut self, _ctx: &mut AppRenderContext) {
        // Would render text editor UI
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        match event.event_type {
            AppEventType::Character(c) => {
                self.text.push(c);
                true
            }
            AppEventType::KeyDown => {
                // Handle backspace, etc.
                true
            }
            _ => false,
        }
    }

    fn has_unsaved_changes(&self) -> bool {
        !self.text.is_empty()
    }
}

struct App {
    window: Option<Window>,
    device: Device,
    command_queue: CommandQueue,
    layer: MetalLayer,
    desktop: Option<GpuDesktop>,
    font: Option<BitmapFont>,
    text_renderer: Option<TextRenderer>,
    start_time: Instant,
    last_frame: Instant,
    frame_count: u64,
    mouse_x: f32,
    mouse_y: f32,
}

impl App {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device found");
        let command_queue = device.new_command_queue();
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        Self {
            window: None,
            device,
            command_queue,
            layer,
            desktop: None,
            font: None,
            text_renderer: None,
            start_time: Instant::now(),
            last_frame: Instant::now(),
            frame_count: 0,
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    fn render(&mut self) {
        let Some(window) = &self.window else { return };
        let Some(drawable) = self.layer.next_drawable() else { return };
        let Some(desktop) = &mut self.desktop else { return };
        let Some(text_renderer) = &mut self.text_renderer else { return };
        let Some(font) = &self.font else { return };

        let size = window.inner_size();
        let width = size.width as f32;
        let height = size.height as f32;

        // Calculate delta time
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Update desktop
        desktop.update(delta);

        // Create render pass
        let texture = drawable.texture();
        let render_pass_desc = RenderPassDescriptor::new();
        let color_attachment = render_pass_desc.color_attachments().object_at(0).unwrap();
        color_attachment.set_texture(Some(texture));
        color_attachment.set_load_action(MTLLoadAction::Clear);
        color_attachment.set_store_action(MTLStoreAction::Store);
        color_attachment.set_clear_color(MTLClearColor::new(
            desktop.background_color[0] as f64,
            desktop.background_color[1] as f64,
            desktop.background_color[2] as f64,
            1.0,
        ));

        let cmd = self.command_queue.new_command_buffer();
        let encoder = cmd.new_render_command_encoder(&render_pass_desc);

        // Render desktop (windows + dock)
        desktop.render(encoder);

        // Status text at bottom of screen (above dock)
        text_renderer.clear();
        let status_y = height - 80.0;  // Above dock
        text_renderer.add_text(
            &format!("{} windows | {:.1} FPS | ESC to quit",
                desktop.state.window_count,
                1.0 / delta.max(0.001)
            ),
            10.0, status_y,
            0x888888FF,
        );

        text_renderer.render(encoder, font, width, height);

        encoder.end_encoding();
        cmd.present_drawable(drawable);
        cmd.commit();

        self.frame_count += 1;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title("GPU Desktop Environment")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));

        let window = event_loop.create_window(attrs).expect("Failed to create window");
        let size = window.inner_size();

        // Set up Metal layer
        unsafe {
            match window.raw_window_handle() {
                Ok(RawWindowHandle::AppKit(handle)) => {
                    use cocoa::appkit::NSView;
                    use cocoa::base::id;
                    use objc::runtime::YES;

                    let view = handle.ns_view.as_ptr() as id;
                    view.setWantsLayer(YES);
                    view.setLayer(std::mem::transmute(self.layer.as_ref()));
                }
                _ => panic!("Unsupported platform"),
            }
        }

        self.layer.set_drawable_size(core_graphics_types::geometry::CGSize::new(
            size.width as f64,
            size.height as f64,
        ));

        // Initialize desktop
        let width = size.width as f32;
        let height = size.height as f32;

        let mut desktop = GpuDesktop::new(
            &self.device,
            width,
            height,
            MTLPixelFormat::BGRA8Unorm,
        ).expect("Failed to create GPU desktop");

        // Register apps
        let files_id = desktop.apps.register("Files", 0);
        let terminal_id = desktop.apps.register("Terminal", 3);
        let docs_id = desktop.apps.register("Documents", 2);
        let editor_id = desktop.apps.register("Editor", 4);

        // Add apps to dock
        desktop.dock.state.add_item(files_id, "Files", 0);
        desktop.dock.state.add_item(terminal_id, "Terminal", 3);
        desktop.dock.state.add_item(docs_id, "Documents", 2);
        desktop.dock.state.add_item(editor_id, "Editor", 4);

        // Launch File Browser app
        let file_browser = Box::new(FileBrowserApp::new());
        if let Err(e) = desktop.launch_app(files_id, file_browser) {
            eprintln!("Failed to launch file browser: {}", e);
        }

        // Launch Terminal app
        let terminal = Box::new(TerminalApp::new());
        if let Err(e) = desktop.launch_app(terminal_id, terminal) {
            eprintln!("Failed to launch terminal: {}", e);
        }

        // Launch Document Viewer app
        let mut doc_viewer = DocumentViewerApp::new();
        doc_viewer.set_title("Welcome");
        doc_viewer.load_html(r#"
            <h1>GPU Desktop Environment</h1>
            <p>Welcome to the GPU-native desktop environment. This system runs
            entirely on GPU compute shaders.</p>

            <h2>Features</h2>
            <ul>
                <li>Window management with GPU compute kernels</li>
                <li>Compositor with shadows and window chrome</li>
                <li>macOS-style dock with magnification</li>
                <li>Integrated applications</li>
            </ul>

            <h2>Applications</h2>
            <p><b>Files</b> - Browse the filesystem</p>
            <p><b>Terminal</b> - GPU Shell with filesystem queries</p>
            <p><b>Documents</b> - View HTML documents</p>
            <p><b>Editor</b> - Edit text files</p>

            <p><em>Use arrow keys to scroll this document.</em></p>
        "#);
        if let Err(e) = desktop.launch_app(docs_id, Box::new(doc_viewer)) {
            eprintln!("Failed to launch document viewer: {}", e);
        }

        // Launch Text Editor app
        let text_editor = Box::new(TextEditorApp::new());
        if let Err(e) = desktop.launch_app(editor_id, text_editor) {
            eprintln!("Failed to launch text editor: {}", e);
        }

        self.desktop = Some(desktop);

        // Initialize text rendering
        let font = BitmapFont::new(&self.device);
        let text_renderer = TextRenderer::new(&self.device, 1024)
            .expect("Failed to create text renderer");
        self.font = Some(font);
        self.text_renderer = Some(text_renderer);

        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::RedrawRequested => {
                self.render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            WindowEvent::Resized(size) => {
                self.layer.set_drawable_size(core_graphics_types::geometry::CGSize::new(
                    size.width as f64,
                    size.height as f64,
                ));

                if let Some(desktop) = &mut self.desktop {
                    desktop.state.screen_width = size.width as f32;
                    desktop.state.screen_height = size.height as f32;
                    desktop.dock.state.config.screen_width = size.width as f32;
                    desktop.dock.state.config.screen_height = size.height as f32;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_x = position.x as f32;
                self.mouse_y = position.y as f32;

                if let Some(desktop) = &mut self.desktop {
                    desktop.on_mouse_move(self.mouse_x, self.mouse_y);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let btn = match button {
                    MouseButton::Left => 0,
                    MouseButton::Right => 1,
                    MouseButton::Middle => 2,
                    _ => 0,
                };

                if let Some(desktop) = &mut self.desktop {
                    match state {
                        ElementState::Pressed => {
                            desktop.on_mouse_down(self.mouse_x, self.mouse_y, btn);
                        }
                        ElementState::Released => {
                            desktop.on_mouse_up(self.mouse_x, self.mouse_y, btn);
                        }
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                // ESC to quit
                if event.state == ElementState::Pressed {
                    if event.physical_key == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) {
                        event_loop.exit();
                        return;
                    }
                }

                // Forward keyboard input to desktop
                if let Some(desktop) = &mut self.desktop {
                    // Convert winit key code to macOS virtual key code
                    let key_code = match event.physical_key {
                        winit::keyboard::PhysicalKey::Code(code) => {
                            use winit::keyboard::KeyCode;
                            match code {
                                KeyCode::KeyA => 0x00,
                                KeyCode::KeyS => 0x01,
                                KeyCode::KeyD => 0x02,
                                KeyCode::KeyF => 0x03,
                                KeyCode::KeyH => 0x04,
                                KeyCode::KeyG => 0x05,
                                KeyCode::KeyZ => 0x06,
                                KeyCode::KeyX => 0x07,
                                KeyCode::KeyC => 0x08,
                                KeyCode::KeyV => 0x09,
                                KeyCode::KeyB => 0x0B,
                                KeyCode::KeyQ => 0x0C,
                                KeyCode::KeyW => 0x0D,
                                KeyCode::KeyE => 0x0E,
                                KeyCode::KeyR => 0x0F,
                                KeyCode::KeyY => 0x10,
                                KeyCode::KeyT => 0x11,
                                KeyCode::Digit1 => 0x12,
                                KeyCode::Digit2 => 0x13,
                                KeyCode::Digit3 => 0x14,
                                KeyCode::Digit4 => 0x15,
                                KeyCode::Digit6 => 0x16,
                                KeyCode::Digit5 => 0x17,
                                KeyCode::Digit9 => 0x19,
                                KeyCode::Digit7 => 0x1A,
                                KeyCode::Digit8 => 0x1C,
                                KeyCode::Digit0 => 0x1D,
                                KeyCode::KeyO => 0x1F,
                                KeyCode::KeyU => 0x20,
                                KeyCode::KeyI => 0x22,
                                KeyCode::KeyP => 0x23,
                                KeyCode::Enter => 0x24,
                                KeyCode::KeyL => 0x25,
                                KeyCode::KeyJ => 0x26,
                                KeyCode::KeyK => 0x28,
                                KeyCode::Semicolon => 0x29,
                                KeyCode::Backslash => 0x2A,
                                KeyCode::Comma => 0x2B,
                                KeyCode::Slash => 0x2C,
                                KeyCode::KeyN => 0x2D,
                                KeyCode::KeyM => 0x2E,
                                KeyCode::Period => 0x2F,
                                KeyCode::Tab => 0x30,
                                KeyCode::Space => 0x31,
                                KeyCode::Backquote => 0x32,
                                KeyCode::Backspace => 0x33,
                                KeyCode::Delete => 0x75,
                                KeyCode::Home => 0x73,
                                KeyCode::End => 0x77,
                                KeyCode::PageUp => 0x74,
                                KeyCode::PageDown => 0x79,
                                KeyCode::ArrowLeft => 0x7B,
                                KeyCode::ArrowRight => 0x7C,
                                KeyCode::ArrowDown => 0x7D,
                                KeyCode::ArrowUp => 0x7E,
                                _ => 0xFF,
                            }
                        }
                        _ => 0xFF,
                    };

                    if key_code != 0xFF && event.state == ElementState::Pressed {
                        let modifiers = KeyModifiers::default();
                        desktop.on_key_down(key_code, modifiers);

                        // Handle character input for printable keys
                        if let Some(text) = &event.text {
                            for c in text.chars() {
                                if c.is_ascii() && !c.is_control() {
                                    let char_event = AppInputEvent {
                                        event_type: AppEventType::Character(c),
                                        key_code: 0,
                                        mouse_x: 0.0,
                                        mouse_y: 0.0,
                                        mouse_button: 0,
                                        modifiers: KeyModifiers::default(),
                                    };
                                    desktop.apps.dispatch_input(desktop.state.focused_window, &char_event);
                                }
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    println!("GPU Desktop Environment Demo");
    println!("============================");
    println!("- Drag window title bars to move");
    println!("- Drag edges/corners to resize");
    println!("- Click close button (red) to close windows");
    println!("- Click minimize (yellow) to minimize");
    println!("- Click maximize (green) to maximize");
    println!("- ESC to quit");
    println!();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
