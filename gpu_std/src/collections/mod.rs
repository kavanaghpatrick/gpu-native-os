//! GPU-Native Collections
//!
//! THE GPU IS THE COMPUTER.
//!
//! This module provides GPU-optimized data structures that work efficiently
//! with SIMD execution and avoid divergence.
//!
//! # Key Design Principles
//!
//! - **O(1) guaranteed**: All operations have constant time to avoid SIMD divergence
//! - **Cache-aligned**: Data structures align to 128-byte Apple Silicon cache lines
//! - **Lock-free reads**: Concurrent read access without synchronization
//! - **Batch-friendly**: Operations designed for bulk processing

mod hashmap;

pub use hashmap::HashMap;
pub use hashmap::Entry;
pub use hashmap::OccupiedEntry;
pub use hashmap::VacantEntry;
