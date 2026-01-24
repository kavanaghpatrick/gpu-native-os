//! GPU Vector Rasterizer - Issue #34 + #35
//!
//! GPU-accelerated vector graphics rendering for bezier paths, fills, strokes, and gradients.
//!
//! Architecture:
//! - PathBuilder: CPU-side path construction
//! - VectorRenderer: GPU compute tessellation + fragment AA rendering
//!
//! Features (Issue #34 - Phase 1):
//! - Solid color fills
//! - Bezier curve tessellation (quadratic and cubic)
//! - Stroke expansion
//!
//! Features (Issue #35 - Gradients):
//! - Linear gradients
//! - Radial gradients
//! - Multi-stop gradients
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
//!
//! // Solid fill
//! renderer.fill(&path.build(), Paint::Solid(Color::RED));
//!
//! // Linear gradient
//! let gradient = LinearGradient::two_color([0.0, 0.0], [200.0, 0.0], Color::RED, Color::BLUE);
//! renderer.fill(&path.build(), Paint::Linear(gradient));
//!
//! renderer.render(encoder, width, height);
//! ```

mod path;
mod rasterizer;

pub use path::{Path, PathBuilder, PathCommand, PathSegment};
pub use rasterizer::{
    AntiAliasMode, Color, FillRule, GradientStop, LinearGradient, Paint, RadialGradient,
    VectorRenderer,
};
