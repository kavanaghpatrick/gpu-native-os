//! GPU Terminal App
//!
//! A terminal emulator integrated with the GPU desktop, using the GPU Shell.

use metal::*;

use crate::gpu_os::desktop::app::{DesktopApp, AppRenderContext, AppInputEvent, AppEventType};
use crate::gpu_os::text_render::{BitmapFont, TextRenderer, colors};
use crate::gpu_os::shell::GpuShell;

/// Output line in terminal
#[derive(Clone)]
pub struct OutputLine {
    pub text: String,
    pub color: u32,
}

/// GPU Terminal Application
pub struct TerminalApp {
    /// GPU Shell instance
    shell: Option<GpuShell>,
    /// Command history
    history: Vec<String>,
    /// Current history index
    history_index: usize,
    /// Current input line
    current_line: String,
    /// Output buffer
    output: Vec<OutputLine>,
    /// Scroll offset
    scroll_offset: usize,
    /// Max output lines
    max_output: usize,
    /// Text renderer
    text_renderer: Option<TextRenderer>,
    /// Font
    font: Option<BitmapFont>,
    /// Cursor position in current line
    cursor_pos: usize,
    /// Cursor blink state
    cursor_visible: bool,
    /// Time since last blink
    blink_timer: f32,
}

impl TerminalApp {
    pub fn new() -> Self {
        Self {
            shell: None,
            history: Vec::new(),
            history_index: 0,
            current_line: String::new(),
            output: vec![
                OutputLine {
                    text: "GPU Shell v0.1".to_string(),
                    color: colors::GREEN,
                },
                OutputLine {
                    text: "Type 'help' for commands".to_string(),
                    color: colors::DARK_GRAY,
                },
                OutputLine {
                    text: "".to_string(),
                    color: colors::BLACK,
                },
            ],
            scroll_offset: 0,
            max_output: 1000,
            text_renderer: None,
            font: None,
            cursor_pos: 0,
            cursor_visible: true,
            blink_timer: 0.0,
        }
    }

    /// Execute current command
    fn execute_command(&mut self) {
        let cmd = self.current_line.trim().to_string();

        if cmd.is_empty() {
            self.add_output("", colors::BLACK);
            return;
        }

        // Add to history
        if self.history.last() != Some(&cmd) {
            self.history.push(cmd.clone());
        }
        self.history_index = self.history.len();

        // Show command in output
        self.add_output(&format!("gpu> {}", cmd), colors::CYAN);

        // Handle built-in commands
        match cmd.as_str() {
            "help" => {
                self.add_output("Commands:", colors::GREEN);
                self.add_output("  files <path>  - List files in path", colors::DARK_GRAY);
                self.add_output("  files <path> | where <cond> | sort | head", colors::DARK_GRAY);
                self.add_output("  search <pattern> <path>", colors::DARK_GRAY);
                self.add_output("  stats - Show cache statistics", colors::DARK_GRAY);
                self.add_output("  clear - Clear screen", colors::DARK_GRAY);
                self.add_output("  help - Show this help", colors::DARK_GRAY);
            }
            "clear" => {
                self.output.clear();
            }
            "stats" => {
                if let Some(ref shell) = self.shell {
                    let (hits, misses, cached, index_hits) = shell.cache_stats();
                    self.add_output(&format!("Cache: {} hits, {} misses, {} cached", hits, misses, cached), colors::BLACK);
                    self.add_output(&format!("Index: {} loads", index_hits), colors::BLACK);
                } else {
                    self.add_output("Shell not initialized", colors::RED);
                }
            }
            _ => {
                // Execute via shell
                if let Some(ref mut shell) = self.shell {
                    match shell.execute(&cmd) {
                        Ok(result) => {
                            // Format result
                            let output = format!("{:?}", result);
                            for line in output.lines().take(50) {
                                self.add_output(line, colors::BLACK);
                            }
                        }
                        Err(e) => {
                            self.add_output(&format!("Error: {}", e), colors::RED);
                        }
                    }
                } else {
                    self.add_output("Shell not initialized", colors::RED);
                }
            }
        }

        self.current_line.clear();
        self.cursor_pos = 0;
    }

    /// Add output line
    fn add_output(&mut self, text: &str, color: u32) {
        self.output.push(OutputLine {
            text: text.to_string(),
            color,
        });

        // Trim if too many lines
        while self.output.len() > self.max_output {
            self.output.remove(0);
        }

        // Auto-scroll to bottom
        self.scroll_to_bottom();
    }

    /// Scroll to bottom of output
    fn scroll_to_bottom(&mut self) {
        // Will be calculated during render
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

impl Default for TerminalApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopApp for TerminalApp {
    fn name(&self) -> &str {
        "Terminal"
    }

    fn icon_index(&self) -> u32 {
        3  // Terminal icon
    }

    fn preferred_size(&self) -> (f32, f32) {
        (600.0, 400.0)
    }

    fn init(&mut self, device: &Device) -> Result<(), String> {
        // Create text renderer
        let font = BitmapFont::new(device);
        let text_renderer = TextRenderer::new(device, 4096)
            .map_err(|e| format!("Failed to create text renderer: {}", e))?;

        self.font = Some(font);
        self.text_renderer = Some(text_renderer);

        // Initialize GPU shell
        match GpuShell::new() {
            Ok(shell) => {
                self.shell = Some(shell);
                self.add_output("GPU Shell initialized", colors::GREEN);
            }
            Err(e) => {
                self.add_output(&format!("Warning: {}", e), colors::YELLOW);
            }
        }

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
        let Some(text_renderer) = &mut self.text_renderer else { return };
        let Some(font) = &self.font else { return };

        text_renderer.clear();

        let line_height = 14.0;
        let padding = 8.0;
        let char_width = 12.0;  // 8.0 base * 1.5 scale

        // Offset for window position - text coords are in screen space
        let ox = ctx.window_x;
        let oy = ctx.window_y;

        // Calculate max characters that fit in window
        // Add extra padding (20px) for border/shadow visual margin
        let effective_width = ctx.width - padding * 2.0 - 20.0;
        let max_chars = (effective_width / char_width).max(10.0) as usize;

        // Dark terminal background is handled by window content color

        // Calculate visible lines
        let available_height = ctx.height - padding * 2.0 - line_height;  // Reserve for input
        let visible_lines = (available_height / line_height) as usize;

        // Scroll to show latest output
        let total_lines = self.output.len();
        let start = if total_lines > visible_lines {
            total_lines - visible_lines
        } else {
            0
        };

        // Render output (truncated to fit window)
        let mut y = padding;
        for line in self.output.iter().skip(start) {
            let display_text = Self::truncate_text(&line.text, max_chars);
            text_renderer.add_text(&display_text, ox + padding, oy + y, line.color);
            y += line_height;
        }

        // Render input line with prompt
        let input_y = ctx.height - padding - line_height;
        let prompt = "gpu> ";
        text_renderer.add_text(prompt, ox + padding, oy + input_y, colors::CYAN);

        let input_x = padding + (prompt.len() as f32 * char_width);
        let input_max_chars = max_chars.saturating_sub(prompt.len());
        let display_input = Self::truncate_text(&self.current_line, input_max_chars);
        text_renderer.add_text(&display_input, ox + input_x, oy + input_y, colors::BLACK);

        // Render cursor (only if within visible area)
        if self.cursor_visible && self.cursor_pos <= input_max_chars {
            let cursor_x = input_x + (self.cursor_pos as f32 * char_width);
            text_renderer.add_text("_", ox + cursor_x, oy + input_y, colors::BLACK);
        }

        // Render text using full screen dimensions
        text_renderer.render(ctx.encoder, font, ctx.screen_width, ctx.screen_height);
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        match event.event_type {
            AppEventType::Character(c) => {
                if c.is_ascii() && !c.is_control() {
                    self.current_line.insert(self.cursor_pos, c);
                    self.cursor_pos += 1;
                    self.cursor_visible = true;
                    self.blink_timer = 0.0;
                }
                true
            }
            AppEventType::KeyDown => {
                match event.key_code {
                    // Return/Enter
                    0x24 => {
                        self.execute_command();
                        true
                    }
                    // Backspace
                    0x33 => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                            self.current_line.remove(self.cursor_pos);
                        }
                        true
                    }
                    // Delete
                    0x75 => {
                        if self.cursor_pos < self.current_line.len() {
                            self.current_line.remove(self.cursor_pos);
                        }
                        true
                    }
                    // Left arrow
                    0x7B => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                        true
                    }
                    // Right arrow
                    0x7C => {
                        if self.cursor_pos < self.current_line.len() {
                            self.cursor_pos += 1;
                        }
                        true
                    }
                    // Up arrow - history
                    0x7E => {
                        if !self.history.is_empty() && self.history_index > 0 {
                            self.history_index -= 1;
                            self.current_line = self.history[self.history_index].clone();
                            self.cursor_pos = self.current_line.len();
                        }
                        true
                    }
                    // Down arrow - history
                    0x7D => {
                        if self.history_index < self.history.len() {
                            self.history_index += 1;
                            if self.history_index < self.history.len() {
                                self.current_line = self.history[self.history_index].clone();
                            } else {
                                self.current_line.clear();
                            }
                            self.cursor_pos = self.current_line.len();
                        }
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}
