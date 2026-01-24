//! GPU Vector Rasterizer - Issue #34
//!
//! GPU-accelerated vector graphics rendering for bezier paths, fills, strokes, and gradients.
//!
//! Architecture:
//! - PathBuilder: CPU-side path construction
//! - VectorRenderer: GPU compute tessellation + fragment AA rendering
//!
//! Example:
//! ```ignore
//! let mut path = PathBuilder::new();
//! path.move_to(100.0, 100.0)
//!     .line_to(200.0, 100.0)
//!     .line_to(150.0, 200.0)
//!     .close();
//!
//! let mut renderer = VectorRenderer::new(&device)?;
//! renderer.fill(&path.build(), Color::RED);
//! renderer.render(encoder, width, height);
//! ```

mod path;
mod rasterizer;

pub use path::{Path, PathBuilder, PathCommand, PathSegment};
pub use rasterizer::{Color, FillRule, Paint, VectorRenderer};
