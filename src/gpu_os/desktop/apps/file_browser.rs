//! GPU File Browser App
//!
//! A file browser integrated with the GPU desktop, using the GPU filesystem.

use metal::*;
use std::path::Path;

use crate::gpu_os::desktop::app::{DesktopApp, AppRenderContext, AppInputEvent, AppEventType};
use crate::gpu_os::text_render::{BitmapFont, TextRenderer, colors};

/// File entry in the browser
#[derive(Clone)]
pub struct FileEntry {
    pub name: String,
    pub path: String,
    pub is_dir: bool,
    pub size: u64,
}

/// GPU File Browser Application
pub struct FileBrowserApp {
    /// Current directory path
    current_path: String,
    /// Files in current directory
    entries: Vec<FileEntry>,
    /// Currently selected index
    selected: usize,
    /// Scroll offset
    scroll_offset: usize,
    /// Visible rows
    visible_rows: usize,
    /// Text renderer
    text_renderer: Option<TextRenderer>,
    /// Font
    font: Option<BitmapFont>,
    /// Whether initialized
    initialized: bool,
}

impl FileBrowserApp {
    pub fn new() -> Self {
        let home = dirs::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "/".to_string());

        Self {
            current_path: home,
            entries: Vec::new(),
            selected: 0,
            scroll_offset: 0,
            visible_rows: 20,
            text_renderer: None,
            font: None,
            initialized: false,
        }
    }

    /// Load entries for current directory
    fn load_directory(&mut self) {
        self.entries.clear();
        self.selected = 0;
        self.scroll_offset = 0;

        // Add parent directory entry
        if self.current_path != "/" {
            self.entries.push(FileEntry {
                name: "..".to_string(),
                path: Path::new(&self.current_path)
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| "/".to_string()),
                is_dir: true,
                size: 0,
            });
        }

        // Read directory
        if let Ok(read_dir) = std::fs::read_dir(&self.current_path) {
            let mut entries: Vec<FileEntry> = read_dir
                .filter_map(|e| e.ok())
                .map(|e| {
                    let path = e.path();
                    let metadata = e.metadata().ok();
                    FileEntry {
                        name: e.file_name().to_string_lossy().to_string(),
                        path: path.to_string_lossy().to_string(),
                        is_dir: path.is_dir(),
                        size: metadata.map(|m| m.len()).unwrap_or(0),
                    }
                })
                .filter(|e| !e.name.starts_with('.'))  // Hide dotfiles
                .collect();

            // Sort: directories first, then alphabetically
            entries.sort_by(|a, b| {
                match (a.is_dir, b.is_dir) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                }
            });

            self.entries.extend(entries);
        }
    }

    /// Navigate to path
    pub fn navigate_to(&mut self, path: &str) {
        self.current_path = path.to_string();
        self.load_directory();
    }

    /// Open selected item
    fn open_selected(&mut self) {
        if let Some(entry) = self.entries.get(self.selected) {
            if entry.is_dir {
                self.navigate_to(&entry.path.clone());
            }
            // TODO: For files, emit an event to open in appropriate app
        }
    }

    /// Format file size
    fn format_size(size: u64) -> String {
        if size < 1024 {
            format!("{} B", size)
        } else if size < 1024 * 1024 {
            format!("{:.1} KB", size as f64 / 1024.0)
        } else if size < 1024 * 1024 * 1024 {
            format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    /// Ensure selection is visible
    fn ensure_visible(&mut self) {
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        } else if self.selected >= self.scroll_offset + self.visible_rows {
            self.scroll_offset = self.selected - self.visible_rows + 1;
        }
    }
}

impl Default for FileBrowserApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DesktopApp for FileBrowserApp {
    fn name(&self) -> &str {
        "Files"
    }

    fn icon_index(&self) -> u32 {
        0  // Folder icon
    }

    fn preferred_size(&self) -> (f32, f32) {
        (500.0, 400.0)
    }

    fn init(&mut self, device: &Device) -> Result<(), String> {
        // Create text renderer
        let font = BitmapFont::new(device);
        let text_renderer = TextRenderer::new(device, 2048)
            .map_err(|e| format!("Failed to create text renderer: {}", e))?;

        self.font = Some(font);
        self.text_renderer = Some(text_renderer);
        self.initialized = true;

        // Load initial directory
        self.load_directory();

        Ok(())
    }

    fn render(&mut self, ctx: &mut AppRenderContext) {
        let Some(text_renderer) = &mut self.text_renderer else { return };
        let Some(font) = &self.font else { return };

        text_renderer.clear();

        let line_height = 16.0;
        let padding = 8.0;
        let mut y = padding;

        // Offset for window position - text coords are in screen space
        let ox = ctx.window_x;
        let oy = ctx.window_y;

        // Header: current path
        text_renderer.add_text(
            &self.current_path,
            ox + padding,
            oy + y,
            colors::BLUE,
        );
        y += line_height + 4.0;

        // Separator
        text_renderer.add_text(
            &"-".repeat(50),
            ox + padding,
            oy + y,
            colors::DARK_GRAY,
        );
        y += line_height;

        // Calculate visible rows based on content height
        self.visible_rows = ((ctx.height - y - padding) / line_height) as usize;

        // File listing
        let start = self.scroll_offset;
        let end = (start + self.visible_rows).min(self.entries.len());

        for (idx, entry) in self.entries[start..end].iter().enumerate() {
            let actual_idx = start + idx;
            let is_selected = actual_idx == self.selected;

            // Selection highlight - use dark colors for light background
            let color = if is_selected {
                colors::BLUE  // Blue highlight for selection
            } else if entry.is_dir {
                colors::GREEN  // Green for directories
            } else {
                colors::DARK_GRAY  // Dark gray for files
            };

            // Icon
            let icon = if entry.is_dir { ">" } else { " " };

            // Name (truncate if too long)
            let max_name_len = 40;
            let name = if entry.name.len() > max_name_len {
                format!("{}...", &entry.name[..max_name_len - 3])
            } else {
                entry.name.clone()
            };

            // Size (for files)
            let size_str = if entry.is_dir {
                "".to_string()
            } else {
                Self::format_size(entry.size)
            };

            // Selection background indicator
            let prefix = if is_selected { "> " } else { "  " };

            let line = format!("{}{} {}  {}", prefix, icon, name, size_str);
            text_renderer.add_text(&line, ox + padding, oy + y, color);

            y += line_height;
        }

        // Scrollbar indicator
        if self.entries.len() > self.visible_rows {
            let scroll_info = format!(
                "[{}-{} of {}]",
                start + 1,
                end,
                self.entries.len()
            );
            text_renderer.add_text(
                &scroll_info,
                ox + ctx.width - 120.0,
                oy + padding,
                colors::GRAY,
            );
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
                        if self.selected > 0 {
                            self.selected -= 1;
                            self.ensure_visible();
                        }
                        true
                    }
                    // Down arrow
                    0x7D => {
                        if self.selected < self.entries.len().saturating_sub(1) {
                            self.selected += 1;
                            self.ensure_visible();
                        }
                        true
                    }
                    // Return/Enter
                    0x24 => {
                        self.open_selected();
                        true
                    }
                    // Backspace - go up
                    0x33 => {
                        if self.current_path != "/" {
                            let parent = Path::new(&self.current_path)
                                .parent()
                                .map(|p| p.to_string_lossy().to_string());
                            if let Some(parent_path) = parent {
                                self.navigate_to(&parent_path);
                            }
                        }
                        true
                    }
                    _ => false,
                }
            }
            AppEventType::MouseDown => {
                // Calculate which row was clicked
                let line_height = 16.0;
                let header_height = 36.0;  // Path + separator
                let row = ((event.mouse_y - header_height) / line_height) as usize;

                if row < self.visible_rows {
                    let clicked_idx = self.scroll_offset + row;
                    if clicked_idx < self.entries.len() {
                        if self.selected == clicked_idx {
                            // Double-click effect: open
                            self.open_selected();
                        } else {
                            self.selected = clicked_idx;
                        }
                    }
                }
                true
            }
            AppEventType::Scroll { delta_y, .. } => {
                if delta_y > 0.0 && self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                } else if delta_y < 0.0 && self.scroll_offset + self.visible_rows < self.entries.len() {
                    self.scroll_offset += 1;
                }
                true
            }
            _ => false,
        }
    }
}
