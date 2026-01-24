//! GPU-Native Document Processing
//!
//! This module contains the GPU-accelerated document processing pipeline:
//! - Tokenizer: HTML bytes → Token stream
//! - Parser: Tokens → Element tree
//! - Style: Elements × Selectors → Computed styles
//! - Layout: Elements + Styles → Positions (coming soon)
//! - Paint: Layout → Vertices (coming soon)

//! GPU-Native Document Processing
//!
//! This module contains the GPU-accelerated document processing pipeline:
//! - Tokenizer: HTML bytes → Token stream
//! - Parser: Tokens → Element tree
//! - Style: Elements × Selectors → Computed styles
//! - Layout: Elements + Styles → Positions
//! - Paint: Layout → Vertices

mod tokenizer;
mod parser;
mod style;
mod layout;
mod paint;
mod image;
mod text;
mod hit_test;
mod link;
mod navigation;

pub use tokenizer::*;
pub use parser::*;
pub use style::*;
// Re-export layout types but not the duplicated constants
pub use layout::{LayoutBox, Viewport, GpuLayoutEngine};
pub use paint::*;
pub use image::*;
pub use text::*;
pub use hit_test::*;
pub use link::*;
pub use navigation::*;
