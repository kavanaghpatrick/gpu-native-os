//! GPU Document Viewer App
//!
//! A simple document viewer integrated with the GPU desktop.
//! Displays HTML/text documents with basic styling and scrolling.

use metal::*;

use crate::gpu_os::desktop::app::{DesktopApp, AppRenderContext, AppInputEvent, AppEventType};
use crate::gpu_os::text_render::{BitmapFont, TextRenderer, colors};

/// Parsed line with color information
#[derive(Clone)]
struct DocumentLine {
    text: String,
    color: u32,
    indent: usize,
}

/// GPU Document Viewer Application
pub struct DocumentViewerApp {
    /// Document title
    title: String,
    /// Parsed document lines
    lines: Vec<DocumentLine>,
    /// Scroll offset (in lines)
    scroll_offset: usize,
    /// Visible lines
    visible_lines: usize,
    /// Text renderer
    text_renderer: Option<TextRenderer>,
    /// Font
    font: Option<BitmapFont>,
}

impl DocumentViewerApp {
    pub fn new() -> Self {
        Self {
            title: "Document".to_string(),
            lines: Vec::new(),
            scroll_offset: 0,
            visible_lines: 20,
            text_renderer: None,
            font: None,
        }
    }

    /// Load HTML content (basic parsing)
    pub fn load_html(&mut self, html: &str) {
        self.lines.clear();
        self.scroll_offset = 0;

        let mut current_color = colors::BLACK;
        let mut in_heading = false;
        let mut indent = 0;

        // Simple HTML parser - extract text and basic formatting
        let mut chars = html.chars().peekable();
        let mut current_text = String::new();

        while let Some(c) = chars.next() {
            if c == '<' {
                // Output accumulated text
                if !current_text.trim().is_empty() {
                    self.lines.push(DocumentLine {
                        text: current_text.trim().to_string(),
                        color: current_color,
                        indent,
                    });
                }
                current_text.clear();

                // Parse tag
                let mut tag = String::new();
                while let Some(&tc) = chars.peek() {
                    if tc == '>' {
                        chars.next();
                        break;
                    }
                    tag.push(chars.next().unwrap());
                }

                let tag_lower = tag.to_lowercase();
                let tag_name = tag_lower.split_whitespace().next().unwrap_or("");

                match tag_name {
                    "h1" | "h2" | "h3" => {
                        in_heading = true;
                        current_color = colors::BLUE;
                    }
                    "/h1" | "/h2" | "/h3" => {
                        in_heading = false;
                        current_color = colors::BLACK;
                        self.lines.push(DocumentLine {
                            text: "".to_string(),
                            color: colors::BLACK,
                            indent: 0,
                        });
                    }
                    "p" => {
                        current_color = colors::BLACK;
                    }
                    "/p" => {
                        self.lines.push(DocumentLine {
                            text: "".to_string(),
                            color: colors::BLACK,
                            indent: 0,
                        });
                    }
                    "b" | "strong" => {
                        current_color = colors::BLACK;  // Bold text in black
                    }
                    "/b" | "/strong" => {
                        current_color = if in_heading { colors::BLUE } else { colors::BLACK };
                    }
                    "em" | "i" => {
                        current_color = colors::GRAY;
                    }
                    "/em" | "/i" => {
                        current_color = if in_heading { colors::BLUE } else { colors::BLACK };
                    }
                    "code" => {
                        current_color = colors::GREEN;
                    }
                    "/code" => {
                        current_color = colors::BLACK;
                    }
                    "li" => {
                        indent = 2;
                        current_text.push_str("* ");
                    }
                    "/li" => {
                        indent = 0;
                    }
                    "br" | "br/" => {
                        self.lines.push(DocumentLine {
                            text: current_text.trim().to_string(),
                            color: current_color,
                            indent,
                        });
                        current_text.clear();
                    }
                    "title" => {
                        // Skip title content - grab for window title
                    }
                    "/title" => {}
                    _ => {}
                }
            } else if c == '\n' || c == '\r' {
                // Treat newlines as spaces
                if !current_text.ends_with(' ') && !current_text.is_empty() {
                    current_text.push(' ');
                }
            } else {
                current_text.push(c);
            }
        }

        // Output any remaining text
        if !current_text.trim().is_empty() {
            self.lines.push(DocumentLine {
                text: current_text.trim().to_string(),
                color: current_color,
                indent,
            });
        }

        // Add demo content if empty
        if self.lines.is_empty() {
            self.load_demo_content();
        }
    }

    /// Load demo content
    fn load_demo_content(&mut self) {
        self.lines = vec![
            DocumentLine { text: "GPU Document Viewer".to_string(), color: colors::BLUE, indent: 0 },
            DocumentLine { text: "".to_string(), color: colors::BLACK, indent: 0 },
            DocumentLine { text: "Welcome to the GPU-native document viewer.".to_string(), color: colors::BLACK, indent: 0 },
            DocumentLine { text: "This component renders HTML documents using".to_string(), color: colors::BLACK, indent: 0 },
            DocumentLine { text: "GPU-accelerated text rendering.".to_string(), color: colors::BLACK, indent: 0 },
            DocumentLine { text: "".to_string(), color: colors::BLACK, indent: 0 },
            DocumentLine { text: "Features:".to_string(), color: colors::CYAN, indent: 0 },
            DocumentLine { text: "* Basic HTML parsing".to_string(), color: colors::BLACK, indent: 2 },
            DocumentLine { text: "* Text styling (headings, bold, italic)".to_string(), color: colors::BLACK, indent: 2 },
            DocumentLine { text: "* Keyboard scrolling (up/down arrows)".to_string(), color: colors::BLACK, indent: 2 },
            DocumentLine { text: "* Mouse wheel scrolling".to_string(), color: colors::BLACK, indent: 2 },
            DocumentLine { text: "".to_string(), color: colors::BLACK, indent: 0 },
            DocumentLine { text: "Use the arrow keys to scroll.".to_string(), color: colors::GRAY, indent: 0 },
        ];
    }

    /// Set document title
    pub fn set_title(&mut self, title: &str) {
        self.title = title.to_string();
    }

    /// Ensure scroll position is valid
    fn clamp_scroll(&mut self) {
        let max_scroll = self.lines.len().saturating_sub(self.visible_lines);
        if self.scroll_offset > max_scroll {
            self.scroll_offset = max_scroll;
        }
    }

    /// Truncate text to fit within max characters
    fn truncate_text(text: &str, max_chars: usize) -> String {
        if text.len() <= max_chars {
            text.to_string()
        } else if max_chars < 4 {
            ".".repeat(max_chars.min(3))
        } else {
            format!("{}...", &text[..max_chars - 3])
        }
    }
}

impl Default for DocumentViewerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopApp for DocumentViewerApp {
    fn name(&self) -> &str {
        &self.title
    }

    fn icon_index(&self) -> u32 {
        2  // Document icon
    }

    fn preferred_size(&self) -> (f32, f32) {
        (500.0, 400.0)
    }

    fn init(&mut self, device: &Device) -> Result<(), String> {
        // Create text renderer
        let font = BitmapFont::new(device);
        let text_renderer = TextRenderer::new(device, 4096)
            .map_err(|e| format!("Failed to create text renderer: {}", e))?;

        self.font = Some(font);
        self.text_renderer = Some(text_renderer);

        // Load demo content
        self.load_demo_content();

        Ok(())
    }

    fn render(&mut self, ctx: &mut AppRenderContext) {
        let line_height = 16.0;
        let padding = 10.0;
        let char_width = 12.0;  // 8.0 base * 1.5 scale
        let scrollbar_width = 20.0;  // Reserve space for scroll indicator

        // Calculate visible lines and clamp scroll before borrowing text_renderer
        self.visible_lines = ((ctx.height - padding * 2.0) / line_height) as usize;
        self.clamp_scroll();

        let Some(text_renderer) = &mut self.text_renderer else { return };
        let Some(font) = &self.font else { return };

        text_renderer.clear();

        // Offset for window position
        let ox = ctx.window_x;
        let oy = ctx.window_y;

        // Calculate max characters for text (accounting for scrollbar area)
        // Add extra padding (20px) for border/shadow visual margin
        let text_area_width = ctx.width - padding * 2.0 - scrollbar_width - 20.0;

        // Render document lines
        let start = self.scroll_offset;
        let end = (start + self.visible_lines).min(self.lines.len());

        for (idx, line) in self.lines[start..end].iter().enumerate() {
            let y = padding + idx as f32 * line_height;
            let x = padding + line.indent as f32 * char_width;

            // Calculate max chars for this line (accounting for indent)
            let indent_width = line.indent as f32 * char_width;
            let available_width = text_area_width - indent_width;
            let max_chars = (available_width / char_width) as usize;

            let display_text = Self::truncate_text(&line.text, max_chars);
            text_renderer.add_text(&display_text, ox + x, oy + y, line.color);
        }

        // Scroll indicator
        if self.lines.len() > self.visible_lines {
            let scroll_pct = if self.lines.len() <= self.visible_lines {
                0.0
            } else {
                self.scroll_offset as f32 / (self.lines.len() - self.visible_lines) as f32
            };

            let indicator = format!("{}/{}", start + 1, self.lines.len());
            text_renderer.add_text(
                &indicator,
                ox + ctx.width - 60.0,
                oy + ctx.height - padding - line_height,
                colors::GRAY,
            );

            // Simple scroll bar
            let bar_height = ctx.height - padding * 2.0;
            let bar_y = padding + scroll_pct * (bar_height - 20.0);
            text_renderer.add_text("|", ox + ctx.width - 12.0, oy + bar_y, colors::DARK_GRAY);
        }

        // Render text using full screen dimensions
        text_renderer.render(ctx.encoder, font, ctx.screen_width, ctx.screen_height);
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        match event.event_type {
            AppEventType::KeyDown => {
                match event.key_code {
                    // Up arrow
                    0x7E => {
                        if self.scroll_offset > 0 {
                            self.scroll_offset -= 1;
                        }
                        true
                    }
                    // Down arrow
                    0x7D => {
                        let max_scroll = self.lines.len().saturating_sub(self.visible_lines);
                        if self.scroll_offset < max_scroll {
                            self.scroll_offset += 1;
                        }
                        true
                    }
                    // Page Up
                    0x74 => {
                        self.scroll_offset = self.scroll_offset.saturating_sub(self.visible_lines);
                        true
                    }
                    // Page Down
                    0x79 => {
                        let max_scroll = self.lines.len().saturating_sub(self.visible_lines);
                        self.scroll_offset = (self.scroll_offset + self.visible_lines).min(max_scroll);
                        true
                    }
                    // Home
                    0x73 => {
                        self.scroll_offset = 0;
                        true
                    }
                    // End
                    0x77 => {
                        self.scroll_offset = self.lines.len().saturating_sub(self.visible_lines);
                        true
                    }
                    _ => false,
                }
            }
            AppEventType::Scroll { delta_y, .. } => {
                if delta_y > 0.0 && self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                } else if delta_y < 0.0 {
                    let max_scroll = self.lines.len().saturating_sub(self.visible_lines);
                    if self.scroll_offset < max_scroll {
                        self.scroll_offset += 1;
                    }
                }
                true
            }
            _ => false,
        }
    }
}
