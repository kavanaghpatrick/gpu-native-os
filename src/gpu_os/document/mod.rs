//! GPU-Native Document Processing
//!
//! This module contains the GPU-accelerated document processing pipeline:
//! - Tokenizer: HTML bytes → Token stream
//! - Parser: Tokens → Element tree
//! - Style: Elements × Selectors → Computed styles (coming soon)
//! - Layout: Elements + Styles → Positions (coming soon)
//! - Paint: Layout → Vertices (coming soon)

mod tokenizer;
mod parser;

pub use tokenizer::*;
pub use parser::*;
