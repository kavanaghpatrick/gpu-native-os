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
//!
//! # Available Collections
//!
//! | Collection | Algorithm | Lookup | Insert | Delete | Best For |
//! |------------|-----------|--------|--------|--------|----------|
//! | `HashMap` | Cuckoo | O(1) | O(1)* | O(1) | Key-value storage |
//! | `HashSet` | Cuckoo | O(1) | O(1)* | O(1) | Membership tests |
//! | `VecDeque` | Ring Buffer | O(1) | O(1) | O(1) | Queues, buffers |
//! | `BinaryHeap` | Tournament | O(1) | O(log n) | O(log n) | Priority queues |
//! | `LinkedList` | Pool+Freelist | O(n) | O(1) | O(1) | Splice, LRU cache |
//!
//! *Amortized due to eviction chains
//!
//! # GPU-Specific Considerations
//!
//! ## Why These Algorithms?
//!
//! Traditional data structure algorithms cause SIMD divergence on GPU:
//!
//! ```text
//! CPU: Each thread finishes independently
//!   Thread 1: 2 steps → done
//!   Thread 2: 5 steps → done
//!   Thread 3: 3 steps → done
//!
//! GPU: All 32 threads must wait for slowest
//!   Thread 1: 2 steps → wait 3 more
//!   Thread 2: 5 steps → done
//!   Thread 3: 3 steps → wait 2 more
//!   Result: 32 threads × 5 steps = wasted work
//! ```
//!
//! Our implementations use O(1) or deterministic-depth algorithms:
//!
//! - **Cuckoo Hashing**: Exactly 2 lookups per key, always
//! - **Ring Buffer**: Bitwise AND for wrapping, no branches
//! - **Tournament Tree**: Exactly log(n) levels, every time
//! - **Pool Allocation**: O(1) freelist operations
//!
//! ## Memory Layout
//!
//! All structures use 128-byte cache-line alignment for Apple Silicon:
//!
//! ```rust,ignore
//! #[repr(C, align(128))]
//! struct CacheAlignedBucket<T> {
//!     data: [T; ELEMENTS_PER_LINE],
//!     // ...
//! }
//! ```
//!
//! ## No Lazy Iteration
//!
//! GPU cannot do lazy iteration. All `.iter()` methods return `Vec<T>`:
//!
//! ```rust,ignore
//! // Returns Vec<T>, not Iterator
//! let items = map.iter();
//! ```

mod hashmap;
mod hashset;
mod vecdeque;
mod binary_heap;
mod linked_list;

// HashMap exports
pub use hashmap::HashMap;
pub use hashmap::Entry;
pub use hashmap::OccupiedEntry;
pub use hashmap::VacantEntry;

// HashSet exports
pub use hashset::HashSet;

// VecDeque exports
pub use vecdeque::VecDeque;

// BinaryHeap exports
pub use binary_heap::BinaryHeap;

// LinkedList exports
pub use linked_list::LinkedList;
pub use linked_list::Cursor;
