//! GPU Menu Bar System
//!
//! A macOS-style menu bar rendered at the top of the screen.
//! Displays app menu, system menus, and status items.

use metal::*;

use crate::gpu_os::text_render::{BitmapFont, TextRenderer, colors};

/// Menu bar height in pixels
pub const MENU_BAR_HEIGHT: f32 = 24.0;

/// A menu item
#[derive(Clone)]
pub struct MenuItem {
    pub label: String,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub separator: bool,
    pub submenu: Option<Vec<MenuItem>>,
}

impl MenuItem {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            shortcut: None,
            enabled: true,
            separator: false,
            submenu: None,
        }
    }

    pub fn with_shortcut(mut self, shortcut: &str) -> Self {
        self.shortcut = Some(shortcut.to_string());
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    pub fn separator() -> Self {
        Self {
            label: String::new(),
            shortcut: None,
            enabled: false,
            separator: true,
            submenu: None,
        }
    }

    pub fn with_submenu(mut self, items: Vec<MenuItem>) -> Self {
        self.submenu = Some(items);
        self
    }
}

/// A menu in the menu bar
#[derive(Clone)]
pub struct Menu {
    pub title: String,
    pub items: Vec<MenuItem>,
    pub x: f32,
    pub width: f32,
}

impl Menu {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            items: Vec::new(),
            x: 0.0,
            width: 0.0,
        }
    }

    pub fn add_item(mut self, item: MenuItem) -> Self {
        self.items.push(item);
        self
    }
}

/// Status item in the right side of menu bar
#[derive(Clone)]
pub struct StatusItem {
    pub icon: String,  // Single character or emoji
    pub label: String,
    pub x: f32,
    pub width: f32,
}

impl StatusItem {
    pub fn new(icon: &str, label: &str) -> Self {
        Self {
            icon: icon.to_string(),
            label: label.to_string(),
            x: 0.0,
            width: 0.0,
        }
    }
}

/// Menu bar state
pub struct MenuBarState {
    /// Screen width
    pub screen_width: f32,
    /// Currently active app name
    pub app_name: String,
    /// Menus
    pub menus: Vec<Menu>,
    /// Status items (right side)
    pub status_items: Vec<StatusItem>,
    /// Currently open menu index (-1 if none)
    pub open_menu: i32,
    /// Hovered menu item in open menu
    pub hovered_item: i32,
    /// Is menu bar visible
    pub visible: bool,
}

impl MenuBarState {
    pub fn new(screen_width: f32) -> Self {
        let mut state = Self {
            screen_width,
            app_name: "Finder".to_string(),
            menus: Vec::new(),
            status_items: Vec::new(),
            open_menu: -1,
            hovered_item: -1,
            visible: true,
        };

        // Default menus
        state.add_default_menus();
        state.add_default_status_items();
        state.layout();

        state
    }

    fn add_default_menus(&mut self) {
        // Apple menu (app name)
        self.menus.push(
            Menu::new("")
                .add_item(MenuItem::new("About This Mac"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("System Settings..."))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("Sleep"))
                .add_item(MenuItem::new("Restart..."))
                .add_item(MenuItem::new("Shut Down..."))
        );

        // File menu
        self.menus.push(
            Menu::new("File")
                .add_item(MenuItem::new("New Window").with_shortcut("Cmd+N"))
                .add_item(MenuItem::new("New Tab").with_shortcut("Cmd+T"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("Open...").with_shortcut("Cmd+O"))
                .add_item(MenuItem::new("Close Window").with_shortcut("Cmd+W"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("Save").with_shortcut("Cmd+S"))
                .add_item(MenuItem::new("Save As...").with_shortcut("Cmd+Shift+S"))
        );

        // Edit menu
        self.menus.push(
            Menu::new("Edit")
                .add_item(MenuItem::new("Undo").with_shortcut("Cmd+Z"))
                .add_item(MenuItem::new("Redo").with_shortcut("Cmd+Shift+Z"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("Cut").with_shortcut("Cmd+X"))
                .add_item(MenuItem::new("Copy").with_shortcut("Cmd+C"))
                .add_item(MenuItem::new("Paste").with_shortcut("Cmd+V"))
                .add_item(MenuItem::new("Select All").with_shortcut("Cmd+A"))
        );

        // View menu
        self.menus.push(
            Menu::new("View")
                .add_item(MenuItem::new("as Icons"))
                .add_item(MenuItem::new("as List"))
                .add_item(MenuItem::new("as Columns"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("Show Sidebar"))
                .add_item(MenuItem::new("Show Preview"))
        );

        // Window menu
        self.menus.push(
            Menu::new("Window")
                .add_item(MenuItem::new("Minimize").with_shortcut("Cmd+M"))
                .add_item(MenuItem::new("Zoom"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("Bring All to Front"))
        );

        // Help menu
        self.menus.push(
            Menu::new("Help")
                .add_item(MenuItem::new("Search"))
                .add_item(MenuItem::separator())
                .add_item(MenuItem::new("GPU Desktop Help"))
        );
    }

    fn add_default_status_items(&mut self) {
        self.status_items.push(StatusItem::new("@", ""));  // Wifi-ish
        self.status_items.push(StatusItem::new("#", "100%"));  // Battery
        self.status_items.push(StatusItem::new("", "12:00"));  // Clock
    }

    /// Update layout after changes
    pub fn layout(&mut self) {
        let char_width = 8.0;
        let padding = 12.0;

        // Layout menus from left
        let mut x = padding;

        // Apple logo placeholder
        x += char_width + padding;

        // App name
        let app_width = self.app_name.len() as f32 * char_width + padding * 2.0;
        x += app_width;

        // Menus
        for menu in &mut self.menus {
            if menu.title.is_empty() {
                continue; // Skip apple menu in layout (it's the apple symbol)
            }
            menu.x = x;
            menu.width = menu.title.len() as f32 * char_width + padding * 2.0;
            x += menu.width;
        }

        // Layout status items from right
        let mut right_x = self.screen_width - padding;

        for item in self.status_items.iter_mut().rev() {
            let width = (item.icon.len() + item.label.len()) as f32 * char_width + padding;
            right_x -= width;
            item.x = right_x;
            item.width = width;
        }
    }

    /// Set active app name
    pub fn set_app_name(&mut self, name: &str) {
        self.app_name = name.to_string();
        self.layout();
    }

    /// Handle click on menu bar
    pub fn handle_click(&mut self, x: f32, y: f32) -> Option<(usize, usize)> {
        if y > MENU_BAR_HEIGHT {
            // Click outside menu bar
            if self.open_menu >= 0 {
                // Check if click is in open menu dropdown
                // For now, just close menu
                self.open_menu = -1;
                self.hovered_item = -1;
            }
            return None;
        }

        // Check menus
        for (idx, menu) in self.menus.iter().enumerate() {
            if x >= menu.x && x < menu.x + menu.width {
                if self.open_menu == idx as i32 {
                    self.open_menu = -1;
                } else {
                    self.open_menu = idx as i32;
                }
                return None;
            }
        }

        self.open_menu = -1;
        None
    }

    /// Handle mouse move
    pub fn handle_mouse_move(&mut self, x: f32, y: f32) {
        if self.open_menu < 0 {
            return;
        }

        // Check if hovering over different menu title
        for (idx, menu) in self.menus.iter().enumerate() {
            if y <= MENU_BAR_HEIGHT && x >= menu.x && x < menu.x + menu.width {
                self.open_menu = idx as i32;
                self.hovered_item = -1;
                return;
            }
        }

        // Check hovering over menu items
        if self.open_menu >= 0 {
            let menu = &self.menus[self.open_menu as usize];
            let dropdown_x = menu.x;
            let dropdown_y = MENU_BAR_HEIGHT;
            let item_height = 22.0;

            for (idx, _item) in menu.items.iter().enumerate() {
                let item_y = dropdown_y + idx as f32 * item_height;
                if y >= item_y && y < item_y + item_height && x >= dropdown_x && x < dropdown_x + 200.0 {
                    self.hovered_item = idx as i32;
                    return;
                }
            }
        }

        self.hovered_item = -1;
    }
}

/// GPU Menu Bar Renderer
pub struct GpuMenuBar {
    pub state: MenuBarState,
    text_renderer: TextRenderer,
    font: BitmapFont,
}

impl GpuMenuBar {
    pub fn new(device: &Device, screen_width: f32) -> Result<Self, String> {
        let font = BitmapFont::new(device);
        let text_renderer = TextRenderer::new(device, 1024)
            .map_err(|e| format!("Failed to create text renderer: {}", e))?;

        Ok(Self {
            state: MenuBarState::new(screen_width),
            text_renderer,
            font,
        })
    }

    /// Render the menu bar
    pub fn render(&mut self, encoder: &RenderCommandEncoderRef, screen_width: f32, screen_height: f32) {
        if !self.state.visible {
            return;
        }

        self.text_renderer.clear();

        let padding = 12.0;
        let char_width = 8.0;
        let text_y = 6.0;

        // Menu bar background is rendered by compositor

        // Apple menu (using @ as placeholder)
        self.text_renderer.add_text("@", padding, text_y, colors::WHITE);

        // App name (bold)
        let app_x = padding + char_width + padding;
        self.text_renderer.add_text(&self.state.app_name, app_x, text_y, colors::WHITE);

        // Menu titles
        for (idx, menu) in self.state.menus.iter().enumerate() {
            if menu.title.is_empty() {
                continue;
            }

            let color = if self.state.open_menu == idx as i32 {
                colors::CYAN
            } else {
                colors::WHITE
            };

            self.text_renderer.add_text(&menu.title, menu.x + padding / 2.0, text_y, color);
        }

        // Status items
        for item in &self.state.status_items {
            let text = format!("{}{}", item.icon, item.label);
            self.text_renderer.add_text(&text, item.x, text_y, colors::WHITE);
        }

        // Render open menu dropdown
        if self.state.open_menu >= 0 {
            let menu = &self.state.menus[self.state.open_menu as usize];
            let dropdown_x = menu.x;
            let dropdown_y = MENU_BAR_HEIGHT;
            let item_height = 22.0;

            for (idx, item) in menu.items.iter().enumerate() {
                let item_y = dropdown_y + idx as f32 * item_height;

                if item.separator {
                    // Draw separator line
                    self.text_renderer.add_text("----------", dropdown_x + 4.0, item_y + 8.0, colors::DARK_GRAY);
                } else {
                    let color = if !item.enabled {
                        colors::DARK_GRAY
                    } else if self.state.hovered_item == idx as i32 {
                        colors::CYAN
                    } else {
                        colors::WHITE
                    };

                    self.text_renderer.add_text(&item.label, dropdown_x + 16.0, item_y + 4.0, color);

                    if let Some(ref shortcut) = item.shortcut {
                        self.text_renderer.add_text(shortcut, dropdown_x + 150.0, item_y + 4.0, colors::GRAY);
                    }
                }
            }
        }

        self.text_renderer.render(encoder, &self.font, screen_width, screen_height);
    }
}
