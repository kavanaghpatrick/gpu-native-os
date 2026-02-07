//! GPU Text Editor App
//!
//! A simple text editor integrated with the GPU desktop.
//! Supports basic text editing, cursor movement, and file operations.

use metal::*;

use crate::gpu_os::desktop::app::{DesktopApp, AppRenderContext, AppInputEvent, AppEventType};
use crate::gpu_os::text_render::{BitmapFont, TextRenderer, colors};

/// GPU Text Editor Application
pub struct TextEditorApp {
    /// Document title/filename
    title: String,
    /// Text lines
    lines: Vec<String>,
    /// Cursor row
    cursor_row: usize,
    /// Cursor column
    cursor_col: usize,
    /// Scroll offset (in lines)
    scroll_offset: usize,
    /// Visible lines
    visible_lines: usize,
    /// Has unsaved changes
    modified: bool,
    /// Text renderer
    text_renderer: Option<TextRenderer>,
    /// Font
    font: Option<BitmapFont>,
    /// Cursor blink state
    cursor_visible: bool,
    /// Blink timer
    blink_timer: f32,
}

impl TextEditorApp {
    pub fn new() -> Self {
        Self {
            title: "Untitled".to_string(),
            lines: vec!["".to_string()],
            cursor_row: 0,
            cursor_col: 0,
            scroll_offset: 0,
            visible_lines: 20,
            modified: false,
            text_renderer: None,
            font: None,
            cursor_visible: true,
            blink_timer: 0.0,
        }
    }

    /// Load text content
    pub fn load_text(&mut self, text: &str) {
        self.lines = text.lines().map(|s| s.to_string()).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor_row = 0;
        self.cursor_col = 0;
        self.scroll_offset = 0;
        self.modified = false;
    }

    /// Set title
    pub fn set_title(&mut self, title: &str) {
        self.title = title.to_string();
    }

    /// Get text content
    pub fn get_text(&self) -> String {
        self.lines.join("\n")
    }

    /// Insert character at cursor
    fn insert_char(&mut self, c: char) {
        if self.cursor_row < self.lines.len() {
            let line = &mut self.lines[self.cursor_row];
            if self.cursor_col <= line.len() {
                line.insert(self.cursor_col, c);
                self.cursor_col += 1;
                self.modified = true;
            }
        }
    }

    /// Delete character before cursor
    fn backspace(&mut self) {
        if self.cursor_col > 0 && self.cursor_row < self.lines.len() {
            let line = &mut self.lines[self.cursor_row];
            self.cursor_col -= 1;
            line.remove(self.cursor_col);
            self.modified = true;
        } else if self.cursor_col == 0 && self.cursor_row > 0 {
            // Join with previous line
            let current_line = self.lines.remove(self.cursor_row);
            self.cursor_row -= 1;
            self.cursor_col = self.lines[self.cursor_row].len();
            self.lines[self.cursor_row].push_str(&current_line);
            self.modified = true;
        }
    }

    /// Delete character at cursor
    fn delete(&mut self) {
        if self.cursor_row < self.lines.len() {
            let line_len = self.lines[self.cursor_row].len();
            if self.cursor_col < line_len {
                self.lines[self.cursor_row].remove(self.cursor_col);
                self.modified = true;
            } else if self.cursor_row + 1 < self.lines.len() {
                // Join with next line
                let next_line = self.lines.remove(self.cursor_row + 1);
                self.lines[self.cursor_row].push_str(&next_line);
                self.modified = true;
            }
        }
    }

    /// Insert new line at cursor
    fn new_line(&mut self) {
        if self.cursor_row < self.lines.len() {
            let current = &self.lines[self.cursor_row];
            let (left, right) = current.split_at(self.cursor_col.min(current.len()));
            let new_line = right.to_string();
            self.lines[self.cursor_row] = left.to_string();
            self.cursor_row += 1;
            self.lines.insert(self.cursor_row, new_line);
            self.cursor_col = 0;
            self.modified = true;
        }
    }

    /// Move cursor
    fn move_cursor(&mut self, delta_row: i32, delta_col: i32) {
        // Move row
        if delta_row < 0 && self.cursor_row > 0 {
            self.cursor_row -= (-delta_row) as usize;
        } else if delta_row > 0 {
            self.cursor_row = (self.cursor_row + delta_row as usize).min(self.lines.len() - 1);
        }

        // Move column
        if delta_col < 0 && self.cursor_col > 0 {
            self.cursor_col = self.cursor_col.saturating_sub((-delta_col) as usize);
        } else if delta_col > 0 {
            self.cursor_col += delta_col as usize;
        }

        // Clamp cursor to line length
        if self.cursor_row < self.lines.len() {
            self.cursor_col = self.cursor_col.min(self.lines[self.cursor_row].len());
        }

        // Ensure cursor is visible
        self.ensure_visible();
    }

    /// Ensure cursor is visible
    fn ensure_visible(&mut self) {
        if self.cursor_row < self.scroll_offset {
            self.scroll_offset = self.cursor_row;
        } else if self.cursor_row >= self.scroll_offset + self.visible_lines {
            self.scroll_offset = self.cursor_row - self.visible_lines + 1;
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

impl Default for TextEditorApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopApp for TextEditorApp {
    fn name(&self) -> &str {
        &self.title
    }

    fn icon_index(&self) -> u32 {
        4  // Text file icon
    }

    fn preferred_size(&self) -> (f32, f32) {
        (500.0, 350.0)
    }

    fn init(&mut self, device: &Device) -> Result<(), String> {
        let font = BitmapFont::new(device);
        let text_renderer = TextRenderer::new(device, 4096)
            .map_err(|e| format!("Failed to create text renderer: {}", e))?;

        self.font = Some(font);
        self.text_renderer = Some(text_renderer);

        // Load demo content
        self.load_text("Welcome to GPU Text Editor\n\nThis is a simple text editor running on the GPU desktop.\n\nFeatures:\n- Text editing with cursor\n- Line insertion/deletion\n- Keyboard navigation\n- Cursor blinking\n\nStart typing to edit this document.");

        Ok(())
    }

    fn update(&mut self, delta_time: f32) {
        // Cursor blink
        self.blink_timer += delta_time;
        if self.blink_timer > 0.5 {
            self.blink_timer = 0.0;
            self.cursor_visible = !self.cursor_visible;
        }
    }

    fn render(&mut self, ctx: &mut AppRenderContext) {
        let line_height = 14.0;
        let padding = 8.0;
        let char_width = 12.0;  // 8.0 base * 1.5 scale
        let gutter_width = 32.0;  // Space for line numbers

        // Calculate visible lines before borrowing
        self.visible_lines = ((ctx.height - padding * 2.0) / line_height) as usize;
        self.ensure_visible();

        let Some(text_renderer) = &mut self.text_renderer else { return };
        let Some(font) = &self.font else { return };

        text_renderer.clear();

        let ox = ctx.window_x;
        let oy = ctx.window_y;

        // Calculate max characters for text content (excluding gutter)
        // Account for: gutter, right padding, and visual margin for borders/shadows
        let effective_width = ctx.width - gutter_width - padding - 40.0;
        let max_text_chars = (effective_width / char_width).max(10.0) as usize;

        // Render visible lines
        let start = self.scroll_offset;
        let end = (start + self.visible_lines).min(self.lines.len());

        for (idx, line) in self.lines[start..end].iter().enumerate() {
            let y = padding + idx as f32 * line_height;
            let line_num = start + idx + 1;

            // Line number
            let num_str = format!("{:3}", line_num);
            let num_color = if start + idx == self.cursor_row {
                colors::BLUE  // Highlight current line number
            } else {
                colors::GRAY
            };
            text_renderer.add_text(&num_str, ox + padding, oy + y, num_color);

            // Line content - truncated to fit window
            let text_color = colors::BLACK;
            let display_line = Self::truncate_text(line, max_text_chars);
            text_renderer.add_text(&display_line, ox + gutter_width, oy + y, text_color);
        }

        // Render cursor (only if within visible text area)
        if self.cursor_visible && self.cursor_row >= start && self.cursor_row < end {
            if self.cursor_col <= max_text_chars {
                let cursor_screen_row = self.cursor_row - start;
                let cursor_y = padding + cursor_screen_row as f32 * line_height;
                let cursor_x = gutter_width + self.cursor_col as f32 * char_width;
                text_renderer.add_text("_", ox + cursor_x, oy + cursor_y, colors::CYAN);
            }
        }

        // Status bar (truncated to fit within window)
        let status_y = ctx.height - padding - line_height;
        let status = format!(
            "Ln {}, Col {} {}",
            self.cursor_row + 1,
            self.cursor_col + 1,
            if self.modified { "[Modified]" } else { "" }
        );
        let max_status_chars = ((ctx.width - padding * 2.0 - 40.0) / char_width).max(10.0) as usize;
        let display_status = Self::truncate_text(&status, max_status_chars);
        text_renderer.add_text(&display_status, ox + padding, oy + status_y, colors::GRAY);
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        match event.event_type {
            AppEventType::Character(c) => {
                if c.is_ascii() && !c.is_control() {
                    self.insert_char(c);
                    self.cursor_visible = true;
                    self.blink_timer = 0.0;
                }
                true
            }
            AppEventType::KeyDown => {
                match event.key_code {
                    // Return/Enter
                    0x24 => {
                        self.new_line();
                        true
                    }
                    // Backspace
                    0x33 => {
                        self.backspace();
                        true
                    }
                    // Delete
                    0x75 => {
                        self.delete();
                        true
                    }
                    // Left arrow
                    0x7B => {
                        self.move_cursor(0, -1);
                        self.cursor_visible = true;
                        self.blink_timer = 0.0;
                        true
                    }
                    // Right arrow
                    0x7C => {
                        self.move_cursor(0, 1);
                        self.cursor_visible = true;
                        self.blink_timer = 0.0;
                        true
                    }
                    // Up arrow
                    0x7E => {
                        self.move_cursor(-1, 0);
                        self.cursor_visible = true;
                        self.blink_timer = 0.0;
                        true
                    }
                    // Down arrow
                    0x7D => {
                        self.move_cursor(1, 0);
                        self.cursor_visible = true;
                        self.blink_timer = 0.0;
                        true
                    }
                    // Home (Fn+Left on Mac)
                    0x73 => {
                        self.cursor_col = 0;
                        self.cursor_visible = true;
                        true
                    }
                    // End (Fn+Right on Mac)
                    0x77 => {
                        if self.cursor_row < self.lines.len() {
                            self.cursor_col = self.lines[self.cursor_row].len();
                        }
                        self.cursor_visible = true;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn has_unsaved_changes(&self) -> bool {
        self.modified
    }
}
