//! Desktop Applications
//!
//! Integrated applications for the GPU desktop environment.

pub mod document_viewer;
pub mod file_browser;
pub mod terminal;
pub mod text_editor;

pub use document_viewer::DocumentViewerApp;
pub use file_browser::FileBrowserApp;
pub use terminal::TerminalApp;
pub use text_editor::TextEditorApp;
