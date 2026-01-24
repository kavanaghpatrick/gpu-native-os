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

pub use tokenizer::*;
pub use parser::*;
pub use style::*;
// Re-export layout types but not the duplicated constants
pub use layout::{LayoutBox, Viewport, GpuLayoutEngine};
pub use paint::*;
