// SDF Text Engine - Issue #33
//
// GPU-Native Signed Distance Field text rendering for resolution-independent,
// scalable text at any size.
//
// Architecture:
// - atlas_data.rs: Pre-baked SDF atlas (generated once, embedded in binary)
// - embedded_renderer.rs: GPU-native rendering using embedded atlas
// - font.rs: TTF parsing for atlas generation tool only
//
// All runtime rendering is 100% GPU - no CPU work per frame.

pub mod font;
pub mod generator;
pub mod atlas;
pub mod renderer;
pub mod atlas_data;
pub mod embedded_renderer;

pub use font::{SdfFont, FontError, GlyphOutline, PathSegment, Rect};
pub use generator::{SdfGenerator, SdfBitmap};
pub use atlas::{SdfAtlas, AtlasGlyph, AtlasError};
pub use renderer::{SdfTextRenderer, TextMetrics, GlyphInstance};
pub use embedded_renderer::{EmbeddedSdfRenderer, SdfTextVertex};
