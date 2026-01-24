//! GPU-Native Document Processing
//!
//! This module contains the GPU-accelerated document processing pipeline:
//! - Tokenizer: HTML bytes → Token stream
//! - Parser: Tokens → Element tree
//! - Style: Elements × Selectors → Computed styles
//! - Layout: Elements + Styles → Positions (coming soon)
//! - Paint: Layout → Vertices (coming soon)

mod tokenizer;
mod parser;
mod style;

pub use tokenizer::*;
pub use parser::*;
pub use style::*;
