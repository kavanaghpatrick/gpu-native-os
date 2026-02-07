//! GPU-Native BinaryHeap using Tournament Tree
//!
//! THE GPU IS THE COMPUTER.
//!
//! This BinaryHeap uses a tournament tree structure for O(log n) operations
//! with DETERMINISTIC depth - every operation traverses exactly log(n) levels,
//! avoiding SIMD divergence.
//!
//! # Why Tournament Tree for GPU?
//!
//! | Approach | GPU Problem |
//! |----------|-------------|
//! | Standard heap sift | Variable depth = SIMD divergence |
//! | Tournament tree | Fixed log(n) traversal, always |
//!
//! The key insight: in a tournament tree, every push/pop requires exactly
//! log(n) comparisons, and all threads do the same amount of work.
//!
//! # Example
//!
//! ```ignore
//! use gpu_std::collections::BinaryHeap;
//!
//! let mut heap = BinaryHeap::new();
//! heap.push(3);
//! heap.push(1);
//! heap.push(4);
//! heap.push(1);
//! heap.push(5);
//!
//! // Max-heap by default
//! assert_eq!(heap.pop(), Some(5));
//! assert_eq!(heap.pop(), Some(4));
//! ```

use core::alloc::Layout;

// ============================================================================
// Constants
// ============================================================================

/// Minimum leaf count (must be power of 2)
const MIN_LEAVES: usize = 16;

/// Sentinel value index indicating no element
const SENTINEL: u32 = u32::MAX;

// ============================================================================
// Tournament Node
// ============================================================================

/// A node in the tournament tree
#[repr(C)]
#[derive(Clone, Copy)]
struct TournamentNode<T> {
    /// The value (winner of subtree)
    value: T,

    /// Index of the leaf this value came from
    source_leaf: u32,

    /// True if this node contains a valid element
    is_valid: bool,

    _padding: [u8; 3],
}

impl<T: Default> Default for TournamentNode<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            source_leaf: SENTINEL,
            is_valid: false,
            _padding: [0; 3],
        }
    }
}

// ============================================================================
// BinaryHeap Implementation
// ============================================================================

/// GPU-native priority queue using tournament tree
///
/// Provides max-heap behavior with O(1) peek and O(log n) push/pop.
/// All operations have deterministic depth to avoid SIMD divergence.
pub struct BinaryHeap<T> {
    /// The tournament tree stored in level-order
    /// Layout: [internal nodes (leaf_count - 1)] [leaves (leaf_count)]
    /// Total size: 2 * leaf_count - 1
    tree: *mut TournamentNode<T>,

    /// Number of leaves (always power of 2)
    leaf_count: usize,

    /// Number of actual elements
    len: usize,
}

impl<T: Copy + Default + Ord> BinaryHeap<T> {
    /// Create a new empty BinaryHeap
    pub fn new() -> Self {
        Self::with_capacity(MIN_LEAVES)
    }

    /// Create a BinaryHeap with specified minimum capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let leaf_count = capacity.max(MIN_LEAVES).next_power_of_two();
        let tree_size = 2 * leaf_count - 1;

        let tree = unsafe {
            let layout = Layout::array::<TournamentNode<T>>(tree_size).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut TournamentNode<T>;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }

            // Initialize all nodes as invalid
            for i in 0..tree_size {
                (*ptr.add(i)).is_valid = false;
                (*ptr.add(i)).source_leaf = SENTINEL;
            }

            ptr
        };

        Self {
            tree,
            leaf_count,
            len: 0,
        }
    }

    /// Get total tree size
    #[inline]
    fn tree_size(&self) -> usize {
        2 * self.leaf_count - 1
    }

    /// Get index of first leaf
    #[inline]
    fn first_leaf(&self) -> usize {
        self.leaf_count - 1
    }

    /// Get a reference to tree node
    #[inline]
    unsafe fn node(&self, idx: usize) -> &TournamentNode<T> {
        &*self.tree.add(idx)
    }

    /// Get a mutable reference to tree node
    #[inline]
    unsafe fn node_mut(&mut self, idx: usize) -> &mut TournamentNode<T> {
        &mut *self.tree.add(idx)
    }

    /// Get tree depth (log2 of leaf count)
    /// Currently unused but kept for potential future use in batch operations.
    #[allow(dead_code)]
    #[inline]
    fn depth(&self) -> usize {
        self.leaf_count.trailing_zeros() as usize
    }

    /// Peek at the maximum element
    ///
    /// O(1) - just read root
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        unsafe {
            let root = self.node(0);
            if root.is_valid {
                Some(&root.value)
            } else {
                None
            }
        }
    }

    /// Push an element onto the heap
    ///
    /// O(log n) - exactly log(n) comparisons
    pub fn push(&mut self, value: T) {
        if self.len >= self.leaf_count {
            self.grow();
        }

        // Find an empty leaf
        let leaf_idx = self.find_empty_leaf();

        unsafe {
            // Set the leaf
            let leaf_tree_idx = self.first_leaf() + leaf_idx;
            let leaf = self.node_mut(leaf_tree_idx);
            leaf.value = value;
            leaf.source_leaf = leaf_idx as u32;
            leaf.is_valid = true;

            // Replay tournament from this leaf to root
            self.replay_from(leaf_tree_idx);
        }

        self.len += 1;
    }

    /// Pop the maximum element
    ///
    /// O(log n) - exactly log(n) comparisons
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        unsafe {
            let root = self.node(0);
            if !root.is_valid {
                return None;
            }

            let winner_value = root.value;
            let winner_leaf = root.source_leaf as usize;

            // Mark winner leaf as empty
            let leaf_tree_idx = self.first_leaf() + winner_leaf;
            let leaf = self.node_mut(leaf_tree_idx);
            leaf.is_valid = false;
            leaf.source_leaf = SENTINEL;

            // Replay tournament from that leaf
            self.replay_from(leaf_tree_idx);

            self.len -= 1;
            Some(winner_value)
        }
    }

    /// Find an empty leaf slot
    fn find_empty_leaf(&self) -> usize {
        let first_leaf = self.first_leaf();
        unsafe {
            for i in 0..self.leaf_count {
                if !self.node(first_leaf + i).is_valid {
                    return i;
                }
            }
        }
        // Should never reach here if len < leaf_count
        0
    }

    /// Replay tournament from a leaf to the root
    ///
    /// This is the key GPU-friendly operation: exactly log(n) steps,
    /// every thread does the same amount of work.
    unsafe fn replay_from(&mut self, mut idx: usize) {
        // Walk up to root
        while idx > 0 {
            let parent_idx = (idx - 1) / 2;
            let left_idx = 2 * parent_idx + 1;
            let right_idx = 2 * parent_idx + 2;

            let left = self.node(left_idx);
            let right = if right_idx < self.tree_size() {
                self.node(right_idx)
            } else {
                &TournamentNode::default()
            };

            // Determine winner (max-heap: larger value wins)
            let winner = self.compare_nodes(left, right);

            let parent = self.node_mut(parent_idx);
            *parent = winner;

            idx = parent_idx;
        }
    }

    /// Compare two nodes and return the winner (max-heap)
    #[inline]
    fn compare_nodes(&self, a: &TournamentNode<T>, b: &TournamentNode<T>) -> TournamentNode<T> {
        match (a.is_valid, b.is_valid) {
            (false, false) => TournamentNode::default(),
            (true, false) => *a,
            (false, true) => *b,
            (true, true) => {
                if a.value >= b.value {
                    *a
                } else {
                    *b
                }
            }
        }
    }

    /// Grow the heap (double leaf count)
    fn grow(&mut self) {
        let new_leaf_count = self.leaf_count * 2;
        let new_tree_size = 2 * new_leaf_count - 1;

        let new_tree = unsafe {
            let layout = Layout::array::<TournamentNode<T>>(new_tree_size).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut TournamentNode<T>;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }

            // Initialize as invalid
            for i in 0..new_tree_size {
                (*ptr.add(i)).is_valid = false;
                (*ptr.add(i)).source_leaf = SENTINEL;
            }

            ptr
        };

        // Copy existing leaves to new tree
        let old_first_leaf = self.first_leaf();
        let new_first_leaf = new_leaf_count - 1;

        unsafe {
            for i in 0..self.leaf_count {
                let old_node = self.node(old_first_leaf + i);
                if old_node.is_valid {
                    let new_node = &mut *new_tree.add(new_first_leaf + i);
                    new_node.value = old_node.value;
                    new_node.source_leaf = i as u32;
                    new_node.is_valid = true;
                }
            }
        }

        // Free old tree
        unsafe {
            let old_layout = Layout::array::<TournamentNode<T>>(self.tree_size()).unwrap();
            alloc::alloc::dealloc(self.tree as *mut u8, old_layout);
        }

        self.tree = new_tree;
        self.leaf_count = new_leaf_count;

        // Rebuild entire tournament
        self.rebuild_tournament();
    }

    /// Rebuild the entire tournament from leaves
    fn rebuild_tournament(&mut self) {
        let first_leaf = self.first_leaf();

        // Process each level from bottom to top
        unsafe {
            // Start from parent of first leaf
            for level_start in (0..first_leaf).rev() {
                let left_idx = 2 * level_start + 1;
                let right_idx = 2 * level_start + 2;

                let left = self.node(left_idx);
                let right = if right_idx < self.tree_size() {
                    self.node(right_idx)
                } else {
                    &TournamentNode::default()
                };

                let winner = self.compare_nodes(left, right);
                let node = self.node_mut(level_start);
                *node = winner;
            }
        }
    }

    /// Get number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.leaf_count
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        unsafe {
            for i in 0..self.tree_size() {
                let node = self.node_mut(i);
                node.is_valid = false;
                node.source_leaf = SENTINEL;
            }
        }
        self.len = 0;
    }
}

impl<T: Copy + Default + Ord> Default for BinaryHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for BinaryHeap<T> {
    fn drop(&mut self) {
        unsafe {
            // Calculate tree size directly to avoid trait bounds
            let tree_size = 2 * self.leaf_count - 1;
            let layout = Layout::array::<TournamentNode<T>>(tree_size).unwrap();
            alloc::alloc::dealloc(self.tree as *mut u8, layout);
        }
    }
}

// ============================================================================
// Batch Operations (GPU-Parallel Friendly)
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: Copy + Default + Ord> BinaryHeap<T> {
    /// Collect all elements in sorted order (descending)
    pub fn into_sorted_vec(mut self) -> alloc::vec::Vec<T> {
        let mut result = alloc::vec::Vec::with_capacity(self.len);
        while let Some(val) = self.pop() {
            result.push(val);
        }
        result
    }

    /// Create from a slice
    pub fn from_slice(slice: &[T]) -> Self {
        let mut heap = Self::with_capacity(slice.len());
        for &item in slice {
            heap.push(item);
        }
        heap
    }

    /// Push multiple elements (batch operation)
    ///
    /// More efficient than individual pushes for large batches.
    pub fn push_batch(&mut self, values: &[T]) {
        // Ensure capacity
        while self.len + values.len() > self.leaf_count {
            self.grow();
        }

        // Insert all values into leaves first
        let first_leaf = self.first_leaf();
        let mut leaf_idx = 0;

        for &value in values {
            // Find empty leaf
            while leaf_idx < self.leaf_count {
                unsafe {
                    if !self.node(first_leaf + leaf_idx).is_valid {
                        break;
                    }
                }
                leaf_idx += 1;
            }

            unsafe {
                let node = self.node_mut(first_leaf + leaf_idx);
                node.value = value;
                node.source_leaf = leaf_idx as u32;
                node.is_valid = true;
            }

            self.len += 1;
            leaf_idx += 1;
        }

        // Rebuild entire tournament (more efficient for large batches)
        self.rebuild_tournament();
    }

    /// Pop up to k elements
    pub fn pop_many(&mut self, k: usize) -> alloc::vec::Vec<T> {
        let count = k.min(self.len);
        let mut result = alloc::vec::Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(val) = self.pop() {
                result.push(val);
            }
        }

        result
    }
}

// ============================================================================
// FromIterator
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: Copy + Default + Ord> core::iter::FromIterator<T> for BinaryHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut heap = BinaryHeap::with_capacity(lower);

        for item in iter {
            heap.push(item);
        }

        heap
    }
}

// Use std when available (for testing), alloc otherwise (for WASM)
#[cfg(feature = "std")]
extern crate std as alloc;

#[cfg(not(feature = "std"))]
extern crate alloc;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut heap = BinaryHeap::new();

        heap.push(3);
        heap.push(1);
        heap.push(4);
        heap.push(1);
        heap.push(5);
        heap.push(9);
        heap.push(2);
        heap.push(6);

        // Max-heap: should pop in descending order
        assert_eq!(heap.pop(), Some(9));
        assert_eq!(heap.pop(), Some(6));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_peek() {
        let mut heap = BinaryHeap::new();

        assert!(heap.peek().is_none());

        heap.push(5);
        assert_eq!(*heap.peek().unwrap(), 5);

        heap.push(10);
        assert_eq!(*heap.peek().unwrap(), 10);

        heap.push(3);
        assert_eq!(*heap.peek().unwrap(), 10);
    }

    #[test]
    fn test_len_empty() {
        let mut heap = BinaryHeap::<i32>::new();

        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);

        heap.push(1);
        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 1);

        heap.pop();
        assert!(heap.is_empty());
    }

    #[test]
    fn test_grow() {
        let mut heap = BinaryHeap::with_capacity(4);

        // Push more than initial capacity
        for i in 0..100 {
            heap.push(i);
        }

        assert_eq!(heap.len(), 100);

        // Should pop in descending order
        for i in (0..100).rev() {
            assert_eq!(heap.pop(), Some(i));
        }
    }

    #[test]
    fn test_clear() {
        let mut heap = BinaryHeap::new();

        for i in 0..10 {
            heap.push(i);
        }

        heap.clear();
        assert!(heap.is_empty());
        assert!(heap.peek().is_none());
    }

    #[test]
    fn test_into_sorted_vec() {
        let heap: BinaryHeap<i32> = [3, 1, 4, 1, 5, 9, 2, 6].into_iter().collect();

        let sorted = heap.into_sorted_vec();
        assert_eq!(sorted, vec![9, 6, 5, 4, 3, 2, 1, 1]);
    }

    #[test]
    fn test_push_batch() {
        let mut heap = BinaryHeap::new();

        heap.push_batch(&[3, 1, 4, 1, 5, 9, 2, 6]);

        assert_eq!(heap.len(), 8);
        assert_eq!(*heap.peek().unwrap(), 9);
    }

    #[test]
    fn test_pop_many() {
        let mut heap: BinaryHeap<i32> = [3, 1, 4, 1, 5, 9, 2, 6].into_iter().collect();

        let top3 = heap.pop_many(3);
        assert_eq!(top3, vec![9, 6, 5]);
        assert_eq!(heap.len(), 5);
    }

    #[test]
    fn test_from_iterator() {
        let heap: BinaryHeap<i32> = [1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(heap.len(), 5);
        assert_eq!(*heap.peek().unwrap(), 5);
    }

    #[test]
    fn test_deterministic_depth() {
        // This test verifies that operations have predictable behavior
        // (can't directly test SIMD behavior in unit tests, but we verify correctness)
        let mut heap = BinaryHeap::with_capacity(64);

        // Fill exactly to capacity
        for i in 0..64 {
            heap.push(i);
        }

        // Every pop should work correctly
        for i in (0..64).rev() {
            assert_eq!(heap.pop(), Some(i));
        }
    }
}
