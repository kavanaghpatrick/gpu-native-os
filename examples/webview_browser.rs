//! Simple WebView browser using wry
//! Uses system WebKit on macOS (same engine as Safari)
//!
//! Run: cargo run --example webview_browser -- [URL]

use wry::{
    application::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    },
    webview::WebViewBuilder,
};

fn main() -> wry::Result<()> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string());

    println!("Opening: {}", url);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("GPU-OS WebView Browser")
        .with_inner_size(wry::application::dpi::LogicalSize::new(1200.0, 800.0))
        .build(&event_loop)?;

    let _webview = WebViewBuilder::new(window)?
        .with_url(&url)?
        .build()?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
}
