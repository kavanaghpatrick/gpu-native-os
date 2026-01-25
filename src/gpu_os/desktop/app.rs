//! GPU Desktop Application Framework
//!
//! Provides a trait and registry for applications running in the desktop environment.
//! Apps can render to windows and handle input events.
//!
//! # Example
//! ```ignore
//! struct Calculator;
//!
//! impl DesktopApp for Calculator {
//!     fn name(&self) -> &str { "Calculator" }
//!     fn render(&mut self, ctx: &mut AppRenderContext) { ... }
//!     fn handle_input(&mut self, event: &AppInputEvent) -> bool { ... }
//! }
//!
//! let mut registry = AppRegistry::new();
//! registry.register(Box::new(Calculator::new()));
//! ```

use metal::*;
use std::collections::HashMap;

use super::types::*;

/// Input event for desktop apps
#[derive(Clone, Copy, Debug)]
pub struct AppInputEvent {
    /// Event type
    pub event_type: AppEventType,
    /// Key code (for keyboard events)
    pub key_code: u32,
    /// Mouse position relative to window content
    pub mouse_x: f32,
    pub mouse_y: f32,
    /// Mouse button (0=left, 1=right, 2=middle)
    pub mouse_button: u8,
    /// Key modifiers
    pub modifiers: KeyModifiers,
}

/// Event types for apps
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppEventType {
    /// Key pressed
    KeyDown,
    /// Key released
    KeyUp,
    /// Character typed
    Character(char),
    /// Mouse button pressed
    MouseDown,
    /// Mouse button released
    MouseUp,
    /// Mouse moved
    MouseMove,
    /// Mouse scrolled
    Scroll { delta_x: f32, delta_y: f32 },
    /// Window gained focus
    FocusGained,
    /// Window lost focus
    FocusLost,
    /// Window resized
    Resize { width: f32, height: f32 },
}

/// Key modifiers
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyModifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub cmd: bool,
}

/// Render context provided to apps during rendering
pub struct AppRenderContext<'a> {
    /// The command encoder to use for rendering
    pub encoder: &'a RenderCommandEncoderRef,
    /// Window content area dimensions
    pub width: f32,
    pub height: f32,
    /// Time since last frame
    pub delta_time: f32,
    /// Total time since app started
    pub total_time: f32,
    /// Frame number
    pub frame: u64,
    /// Screen position of window content (for global positioning)
    pub window_x: f32,
    pub window_y: f32,
    /// Full screen dimensions
    pub screen_width: f32,
    pub screen_height: f32,
}

/// Trait for desktop applications
///
/// Implement this trait to create applications that run in the GPU desktop.
pub trait DesktopApp: Send {
    /// Application name (shown in title bar and dock)
    fn name(&self) -> &str;

    /// Application icon index in icon atlas (for dock)
    fn icon_index(&self) -> u32 {
        0  // Default icon
    }

    /// Initialize the application
    ///
    /// Called when the app is first launched. Use this to create GPU resources.
    fn init(&mut self, _device: &Device) -> Result<(), String> {
        Ok(())
    }

    /// Render the application content
    ///
    /// Called every frame when the app's window is visible.
    fn render(&mut self, ctx: &mut AppRenderContext);

    /// Handle an input event
    ///
    /// Return true if the event was consumed, false to let it propagate.
    fn handle_input(&mut self, event: &AppInputEvent) -> bool;

    /// Called when the app should update its state
    ///
    /// Called every frame, even when the window is not visible.
    fn update(&mut self, _delta_time: f32) {}

    /// Called when the app is about to be closed
    ///
    /// Return true to allow closing, false to prevent (e.g., unsaved changes).
    fn should_close(&mut self) -> bool {
        true
    }

    /// Called when the app is closed
    fn on_close(&mut self) {}

    /// Get preferred window size
    fn preferred_size(&self) -> (f32, f32) {
        (400.0, 300.0)
    }

    /// Whether the app has unsaved changes
    fn has_unsaved_changes(&self) -> bool {
        false
    }
}

/// Registered application info
pub struct RegisteredApp {
    pub id: u32,
    pub name: String,
    pub icon_index: u32,
}

/// Running application instance
pub struct AppInstance {
    /// Unique instance ID
    pub id: u32,
    /// App type ID
    pub app_id: u32,
    /// Window ID
    pub window_id: u32,
    /// The app implementation
    pub app: Box<dyn DesktopApp>,
    /// Time since app started
    pub running_time: f32,
    /// Frame counter
    pub frame_count: u64,
}

/// Application registry and manager
pub struct AppRegistry {
    /// Registered app types
    registered: HashMap<u32, RegisteredApp>,
    /// Running app instances
    instances: Vec<AppInstance>,
    /// Next app type ID
    next_app_id: u32,
    /// Next instance ID
    next_instance_id: u32,
}

impl AppRegistry {
    /// Create a new app registry
    pub fn new() -> Self {
        Self {
            registered: HashMap::new(),
            instances: Vec::new(),
            next_app_id: 1,
            next_instance_id: 1,
        }
    }

    /// Register an app type
    ///
    /// Returns the app type ID for launching instances.
    pub fn register(&mut self, name: &str, icon_index: u32) -> u32 {
        let id = self.next_app_id;
        self.next_app_id += 1;

        self.registered.insert(id, RegisteredApp {
            id,
            name: name.to_string(),
            icon_index,
        });

        id
    }

    /// Get registered app info
    pub fn get_registered(&self, app_id: u32) -> Option<&RegisteredApp> {
        self.registered.get(&app_id)
    }

    /// List all registered apps
    pub fn list_registered(&self) -> Vec<&RegisteredApp> {
        self.registered.values().collect()
    }

    /// Launch an app instance
    ///
    /// Returns the instance ID if successful.
    pub fn launch(
        &mut self,
        app_id: u32,
        window_id: u32,
        app: Box<dyn DesktopApp>,
        device: &Device,
    ) -> Result<u32, String> {
        // Verify app is registered
        if !self.registered.contains_key(&app_id) {
            return Err(format!("App {} not registered", app_id));
        }

        let instance_id = self.next_instance_id;
        self.next_instance_id += 1;

        let mut instance = AppInstance {
            id: instance_id,
            app_id,
            window_id,
            app,
            running_time: 0.0,
            frame_count: 0,
        };

        // Initialize the app
        instance.app.init(device)?;

        self.instances.push(instance);

        Ok(instance_id)
    }

    /// Close an app instance
    pub fn close(&mut self, instance_id: u32) -> bool {
        if let Some(idx) = self.instances.iter().position(|i| i.id == instance_id) {
            let instance = &mut self.instances[idx];
            if instance.app.should_close() {
                instance.app.on_close();
                self.instances.remove(idx);
                return true;
            }
        }
        false
    }

    /// Close app instance by window ID
    pub fn close_by_window(&mut self, window_id: u32) -> bool {
        if let Some(idx) = self.instances.iter().position(|i| i.window_id == window_id) {
            let instance = &mut self.instances[idx];
            if instance.app.should_close() {
                instance.app.on_close();
                self.instances.remove(idx);
                return true;
            }
        }
        false
    }

    /// Get app instance by ID
    pub fn get_instance(&self, instance_id: u32) -> Option<&AppInstance> {
        self.instances.iter().find(|i| i.id == instance_id)
    }

    /// Get app instance by ID (mutable)
    pub fn get_instance_mut(&mut self, instance_id: u32) -> Option<&mut AppInstance> {
        self.instances.iter_mut().find(|i| i.id == instance_id)
    }

    /// Get app instance by window ID
    pub fn get_by_window(&self, window_id: u32) -> Option<&AppInstance> {
        self.instances.iter().find(|i| i.window_id == window_id)
    }

    /// Get app instance by window ID (mutable)
    pub fn get_by_window_mut(&mut self, window_id: u32) -> Option<&mut AppInstance> {
        self.instances.iter_mut().find(|i| i.window_id == window_id)
    }

    /// Update all running apps
    pub fn update_all(&mut self, delta_time: f32) {
        for instance in &mut self.instances {
            instance.running_time += delta_time;
            instance.app.update(delta_time);
        }
    }

    /// Render app for a window
    pub fn render_for_window(
        &mut self,
        window_id: u32,
        ctx: &mut AppRenderContext,
    ) {
        if let Some(instance) = self.instances.iter_mut().find(|i| i.window_id == window_id) {
            instance.frame_count += 1;
            instance.app.render(ctx);
        }
    }

    /// Dispatch input event to app for window
    pub fn dispatch_input(&mut self, window_id: u32, event: &AppInputEvent) -> bool {
        if let Some(instance) = self.instances.iter_mut().find(|i| i.window_id == window_id) {
            return instance.app.handle_input(event);
        }
        false
    }

    /// Get number of running instances of an app type
    pub fn instance_count(&self, app_id: u32) -> usize {
        self.instances.iter().filter(|i| i.app_id == app_id).count()
    }

    /// List all running instances
    pub fn list_instances(&self) -> &[AppInstance] {
        &self.instances
    }

    /// Check if any app has unsaved changes
    pub fn any_unsaved_changes(&self) -> bool {
        self.instances.iter().any(|i| i.app.has_unsaved_changes())
    }
}

impl Default for AppRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple test app that renders a colored rectangle
pub struct TestApp {
    name: String,
    color: [f32; 4],
    clicks: u32,
}

impl TestApp {
    pub fn new(name: &str, color: [f32; 4]) -> Self {
        Self {
            name: name.to_string(),
            color,
            clicks: 0,
        }
    }
}

impl DesktopApp for TestApp {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&mut self, _ctx: &mut AppRenderContext) {
        // In real implementation, would render colored content
        // For now, this is just a placeholder
    }

    fn handle_input(&mut self, event: &AppInputEvent) -> bool {
        match event.event_type {
            AppEventType::MouseDown => {
                self.clicks += 1;
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_registry() {
        let mut registry = AppRegistry::new();

        // Register an app
        let calc_id = registry.register("Calculator", 1);
        let notes_id = registry.register("Notes", 2);

        assert_eq!(registry.list_registered().len(), 2);
        assert_eq!(registry.get_registered(calc_id).unwrap().name, "Calculator");
        assert_eq!(registry.get_registered(notes_id).unwrap().name, "Notes");
    }

    #[test]
    fn test_app_launch() {
        let device = Device::system_default().expect("No Metal device");
        let mut registry = AppRegistry::new();

        let app_id = registry.register("Test", 0);
        let app = Box::new(TestApp::new("Test", [1.0, 0.0, 0.0, 1.0]));

        let instance_id = registry.launch(app_id, 1, app, &device).unwrap();

        assert_eq!(registry.instance_count(app_id), 1);
        assert!(registry.get_instance(instance_id).is_some());
        assert!(registry.get_by_window(1).is_some());
    }

    #[test]
    fn test_app_close() {
        let device = Device::system_default().expect("No Metal device");
        let mut registry = AppRegistry::new();

        let app_id = registry.register("Test", 0);
        let app = Box::new(TestApp::new("Test", [1.0, 0.0, 0.0, 1.0]));

        let instance_id = registry.launch(app_id, 1, app, &device).unwrap();
        assert_eq!(registry.instance_count(app_id), 1);

        registry.close(instance_id);
        assert_eq!(registry.instance_count(app_id), 0);
    }

    #[test]
    fn test_input_dispatch() {
        let device = Device::system_default().expect("No Metal device");
        let mut registry = AppRegistry::new();

        let app_id = registry.register("Test", 0);
        let app = Box::new(TestApp::new("Test", [1.0, 0.0, 0.0, 1.0]));

        registry.launch(app_id, 1, app, &device).unwrap();

        let event = AppInputEvent {
            event_type: AppEventType::MouseDown,
            key_code: 0,
            mouse_x: 10.0,
            mouse_y: 10.0,
            mouse_button: 0,
            modifiers: KeyModifiers::default(),
        };

        let handled = registry.dispatch_input(1, &event);
        assert!(handled);
    }
}
