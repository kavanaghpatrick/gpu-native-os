// GPU-Native Text Editor Demo
//
// A text editor running entirely on the GPU using the GpuApp framework.
//
// Controls:
// - Type to insert characters
// - Arrow keys to move cursor
// - Backspace/Delete to delete
// - Enter for newline
// - Home/End for line start/end

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::*;
use objc::{rc::autoreleasepool, runtime::YES};
use rust_experiment::gpu_os::app::GpuRuntime;
use rust_experiment::gpu_os::text_editor::{
    TextEditor, EDIT_MOVE_LEFT, EDIT_MOVE_RIGHT, EDIT_MOVE_UP, EDIT_MOVE_DOWN,
    EDIT_MOVE_HOME, EDIT_MOVE_END,
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::{Window, WindowId},
};

struct TextEditorDemo {
    window: Option<Window>,
    layer: Option<MetalLayer>,
    runtime: Option<GpuRuntime>,
    editor: Option<TextEditor>,
    window_size: (u32, u32),
    frame_count: u64,
}

impl TextEditorDemo {
    fn new() -> Self {
        Self {
            window: None,
            layer: None,
            runtime: None,
            editor: None,
            window_size: (1024, 768),
            frame_count: 0,
        }
    }

    fn initialize(&mut self, window: Window) {
        let device = Device::system_default().expect("No Metal device found");
        println!("GPU-Native Text Editor");
        println!("======================");
        println!("GPU: {}", device.name());
        println!();
        println!("Controls:");
        println!("  Type        - Insert characters");
        println!("  Arrows      - Move cursor");
        println!("  Backspace   - Delete backward");
        println!("  Delete      - Delete forward");
        println!("  Enter       - New line");
        println!("  Home/End    - Line start/end");
        println!();

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
        self.window_size = (size.width, size.height);
        layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

        let runtime = GpuRuntime::new(device.clone());
        let editor = TextEditor::new(&device).expect("Failed to create TextEditor");

        self.window = Some(window);
        self.layer = Some(layer);
        self.runtime = Some(runtime);
        self.editor = Some(editor);
    }

    fn render(&mut self) {
        let layer = self.layer.as_ref().unwrap();
        let runtime = self.runtime.as_mut().unwrap();
        let editor = self.editor.as_mut().unwrap();

        let drawable = match layer.next_drawable() {
            Some(d) => d,
            None => return,
        };

        runtime.run_frame(editor, &drawable);

        self.frame_count += 1;

        // Print stats occasionally
        if self.frame_count % 120 == 0 {
            let doc = editor.document();
            let layout = editor.layout();
            println!(
                "Ln {}, Col {} | {} chars | {} lines | Frame {}",
                layout.cursor_line + 1,
                layout.cursor_column + 1,
                doc.char_count,
                layout.line_count,
                self.frame_count
            );
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.window_size = (width, height);
        if let Some(layer) = &self.layer {
            layer.set_drawable_size(CGSize::new(width as f64, height as f64));
        }
    }

    fn handle_key(&mut self, key: Key) {
        let editor = match &mut self.editor {
            Some(e) => e,
            None => return,
        };

        match key {
            Key::Named(NamedKey::ArrowLeft) => editor.move_cursor(EDIT_MOVE_LEFT),
            Key::Named(NamedKey::ArrowRight) => editor.move_cursor(EDIT_MOVE_RIGHT),
            Key::Named(NamedKey::ArrowUp) => editor.move_cursor(EDIT_MOVE_UP),
            Key::Named(NamedKey::ArrowDown) => editor.move_cursor(EDIT_MOVE_DOWN),
            Key::Named(NamedKey::Home) => editor.move_cursor(EDIT_MOVE_HOME),
            Key::Named(NamedKey::End) => editor.move_cursor(EDIT_MOVE_END),
            Key::Named(NamedKey::Backspace) => editor.delete_back(),
            Key::Named(NamedKey::Delete) => editor.delete_forward(),
            Key::Named(NamedKey::Enter) => editor.newline(),
            Key::Character(ref c) => {
                // Insert printable characters
                for ch in c.chars() {
                    if ch.is_ascii() && !ch.is_control() {
                        editor.insert_char(ch);
                    }
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for TextEditorDemo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::LogicalSize::new(1024, 768))
            .with_title("GPU-Native Text Editor");

        let window = event_loop.create_window(window_attrs).unwrap();
        self.initialize(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        autoreleasepool(|| match event {
            WindowEvent::CloseRequested => {
                if let Some(editor) = &self.editor {
                    let doc = editor.document();
                    println!();
                    println!("Final Stats:");
                    println!("  Characters: {}", doc.char_count);
                    println!("  Frames: {}", self.frame_count);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.resize(new_size.width.max(1), new_size.height.max(1));
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.handle_key(event.logical_key);
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
    println!("Starting GPU-Native Text Editor...\n");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut demo = TextEditorDemo::new();
    event_loop.run_app(&mut demo).unwrap();
}
