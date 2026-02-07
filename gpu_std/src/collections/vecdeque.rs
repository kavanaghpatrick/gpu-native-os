//! GPU-Native VecDeque using Ring Buffer
//!
//! THE GPU IS THE COMPUTER.
//!
//! This VecDeque uses a power-of-2 ring buffer for O(1) operations at both ends.
//! Index calculations are branchless (bitwise AND with mask).
//!
//! # Why Ring Buffer for GPU?
//!
//! - No memory moves on push/pop (O(1) guaranteed)
//! - Index wrapping is branchless (idx & mask)
//! - Predictable memory access pattern
//! - All operations have constant-time complexity
//!
//! # Example
//!
//! ```ignore
//! use gpu_std::collections::VecDeque;
//!
//! let mut deque = VecDeque::new();
//! deque.push_back(1);
//! deque.push_back(2);
//! deque.push_front(0);
//!
//! assert_eq!(deque.pop_front(), Some(0));
//! assert_eq!(deque.pop_back(), Some(2));
//! assert_eq!(deque.len(), 1);
//! ```

use core::alloc::Layout;

// ============================================================================
// Constants
// ============================================================================

/// Minimum capacity (must be power of 2)
const MIN_CAPACITY: usize = 16;

// ============================================================================
// VecDeque Implementation
// ============================================================================

/// GPU-native double-ended queue using ring buffer
///
/// All operations are O(1) with no memory moves.
/// Capacity is always a power of 2 for branchless index wrapping.
pub struct VecDeque<T> {
    /// Data buffer (always power-of-2 capacity)
    buffer: *mut T,

    /// Capacity (always power of 2)
    capacity: usize,

    /// Head index (where next pop_front reads from)
    head: usize,

    /// Length (number of elements)
    len: usize,
}

impl<T: Copy + Default> VecDeque<T> {
    /// Create a new empty VecDeque
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    /// Create a VecDeque with specified minimum capacity
    ///
    /// Actual capacity will be rounded up to next power of 2.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(MIN_CAPACITY).next_power_of_two();

        let buffer = unsafe {
            let layout = Layout::array::<T>(capacity).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut T;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }
            ptr
        };

        Self {
            buffer,
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Get the mask for index wrapping (capacity - 1)
    #[inline(always)]
    fn mask(&self) -> usize {
        self.capacity - 1
    }

    /// Wrap an index to be within buffer bounds
    #[inline(always)]
    fn wrap(&self, index: usize) -> usize {
        index & self.mask()
    }

    /// Get the tail index (one past the last element)
    #[inline(always)]
    fn tail(&self) -> usize {
        self.wrap(self.head.wrapping_add(self.len))
    }

    /// Push an element to the back
    ///
    /// O(1) - no memory moves
    #[inline]
    pub fn push_back(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow();
        }

        let tail = self.tail();
        unsafe {
            *self.buffer.add(tail) = value;
        }
        self.len += 1;
    }

    /// Push an element to the front
    ///
    /// O(1) - no memory moves
    #[inline]
    pub fn push_front(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow();
        }

        self.head = self.wrap(self.head.wrapping_sub(1));
        unsafe {
            *self.buffer.add(self.head) = value;
        }
        self.len += 1;
    }

    /// Pop an element from the front
    ///
    /// O(1) - no memory moves
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        let value = unsafe { *self.buffer.add(self.head) };
        self.head = self.wrap(self.head.wrapping_add(1));
        self.len -= 1;
        Some(value)
    }

    /// Pop an element from the back
    ///
    /// O(1) - no memory moves
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        let tail = self.tail();
        Some(unsafe { *self.buffer.add(tail) })
    }

    /// Get a reference to the front element
    #[inline]
    pub fn front(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        Some(unsafe { &*self.buffer.add(self.head) })
    }

    /// Get a mutable reference to the front element
    #[inline]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }
        Some(unsafe { &mut *self.buffer.add(self.head) })
    }

    /// Get a reference to the back element
    #[inline]
    pub fn back(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let back_idx = self.wrap(self.head.wrapping_add(self.len - 1));
        Some(unsafe { &*self.buffer.add(back_idx) })
    }

    /// Get a mutable reference to the back element
    #[inline]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            return None;
        }
        let back_idx = self.wrap(self.head.wrapping_add(self.len - 1));
        Some(unsafe { &mut *self.buffer.add(back_idx) })
    }

    /// Get a reference to an element by index
    ///
    /// O(1) random access
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let actual_idx = self.wrap(self.head.wrapping_add(index));
        Some(unsafe { &*self.buffer.add(actual_idx) })
    }

    /// Get a mutable reference to an element by index
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let actual_idx = self.wrap(self.head.wrapping_add(index));
        Some(unsafe { &mut *self.buffer.add(actual_idx) })
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
        self.capacity
    }

    /// Clear all elements
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Grow the buffer (double capacity)
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;

        let new_buffer = unsafe {
            let layout = Layout::array::<T>(new_capacity).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut T;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }
            ptr
        };

        // Copy elements to new buffer in order
        unsafe {
            for i in 0..self.len {
                let src_idx = self.wrap(self.head.wrapping_add(i));
                *new_buffer.add(i) = *self.buffer.add(src_idx);
            }

            // Free old buffer
            let old_layout = Layout::array::<T>(self.capacity).unwrap();
            alloc::alloc::dealloc(self.buffer as *mut u8, old_layout);
        }

        self.buffer = new_buffer;
        self.capacity = new_capacity;
        self.head = 0;
    }

    /// Swap elements at two indices
    pub fn swap(&mut self, i: usize, j: usize) {
        if i >= self.len || j >= self.len {
            return;
        }

        let idx_i = self.wrap(self.head.wrapping_add(i));
        let idx_j = self.wrap(self.head.wrapping_add(j));

        unsafe {
            let tmp = *self.buffer.add(idx_i);
            *self.buffer.add(idx_i) = *self.buffer.add(idx_j);
            *self.buffer.add(idx_j) = tmp;
        }
    }

    /// Rotate left by n positions
    ///
    /// Moves the first n elements to the end.
    /// O(n) time complexity.
    pub fn rotate_left(&mut self, n: usize) {
        if self.len <= 1 {
            return;
        }
        let n = n % self.len;
        if n == 0 {
            return;
        }

        // Pop n elements from front and push them to back
        for _ in 0..n {
            if let Some(val) = self.pop_front() {
                self.push_back(val);
            }
        }
    }

    /// Rotate right by n positions
    ///
    /// Moves the last n elements to the front.
    /// O(n) time complexity.
    pub fn rotate_right(&mut self, n: usize) {
        if self.len <= 1 {
            return;
        }
        let n = n % self.len;
        if n == 0 {
            return;
        }

        // Pop n elements from back and push them to front
        for _ in 0..n {
            if let Some(val) = self.pop_back() {
                self.push_front(val);
            }
        }
    }
}

impl<T: Copy + Default> Default for VecDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for VecDeque<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<T>(self.capacity).unwrap();
            alloc::alloc::dealloc(self.buffer as *mut u8, layout);
        }
    }
}

// ============================================================================
// Iteration (returns Vec, not lazy iterator - GPU can't do lazy)
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: Copy + Default> VecDeque<T> {
    /// Collect all elements into a Vec (front to back order)
    pub fn iter(&self) -> alloc::vec::Vec<T> {
        let mut result = alloc::vec::Vec::with_capacity(self.len);
        for i in 0..self.len {
            result.push(*self.get(i).unwrap());
        }
        result
    }

    /// Drain all elements, returning them as a Vec
    pub fn drain(&mut self) -> alloc::vec::Vec<T> {
        let result = self.iter();
        self.clear();
        result
    }

    /// Create from a slice
    pub fn from_slice(slice: &[T]) -> Self {
        let mut deque = Self::with_capacity(slice.len());
        for &item in slice {
            deque.push_back(item);
        }
        deque
    }
}

// ============================================================================
// Index Implementation
// ============================================================================

impl<T: Copy + Default> core::ops::Index<usize> for VecDeque<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("VecDeque index out of bounds")
    }
}

impl<T: Copy + Default> core::ops::IndexMut<usize> for VecDeque<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("VecDeque index out of bounds")
    }
}

// ============================================================================
// FromIterator
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: Copy + Default> core::iter::FromIterator<T> for VecDeque<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut deque = VecDeque::with_capacity(lower);

        for item in iter {
            deque.push_back(item);
        }

        deque
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
    fn test_push_pop_back() {
        let mut deque = VecDeque::new();

        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        assert_eq!(deque.len(), 3);
        assert_eq!(deque.pop_back(), Some(3));
        assert_eq!(deque.pop_back(), Some(2));
        assert_eq!(deque.pop_back(), Some(1));
        assert_eq!(deque.pop_back(), None);
    }

    #[test]
    fn test_push_pop_front() {
        let mut deque = VecDeque::new();

        deque.push_front(1);
        deque.push_front(2);
        deque.push_front(3);

        assert_eq!(deque.len(), 3);
        assert_eq!(deque.pop_front(), Some(3));
        assert_eq!(deque.pop_front(), Some(2));
        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.pop_front(), None);
    }

    #[test]
    fn test_mixed_operations() {
        let mut deque = VecDeque::new();

        deque.push_back(2);
        deque.push_front(1);
        deque.push_back(3);
        deque.push_front(0);

        // Order should be: 0, 1, 2, 3
        assert_eq!(deque.pop_front(), Some(0));
        assert_eq!(deque.pop_back(), Some(3));
        assert_eq!(deque.pop_front(), Some(1));
        assert_eq!(deque.pop_back(), Some(2));
    }

    #[test]
    fn test_front_back() {
        let mut deque = VecDeque::new();

        assert!(deque.front().is_none());
        assert!(deque.back().is_none());

        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        assert_eq!(*deque.front().unwrap(), 1);
        assert_eq!(*deque.back().unwrap(), 3);

        *deque.front_mut().unwrap() = 10;
        *deque.back_mut().unwrap() = 30;

        assert_eq!(*deque.front().unwrap(), 10);
        assert_eq!(*deque.back().unwrap(), 30);
    }

    #[test]
    fn test_random_access() {
        let mut deque = VecDeque::new();

        for i in 0..10 {
            deque.push_back(i);
        }

        for i in 0..10 {
            assert_eq!(*deque.get(i).unwrap(), i);
        }

        assert!(deque.get(10).is_none());

        // Test indexing
        assert_eq!(deque[5], 5);
    }

    #[test]
    fn test_grow() {
        let mut deque = VecDeque::with_capacity(4);

        // Push more than initial capacity
        for i in 0..100 {
            deque.push_back(i);
        }

        assert_eq!(deque.len(), 100);

        // Verify order preserved
        for i in 0..100 {
            assert_eq!(*deque.get(i).unwrap(), i);
        }
    }

    #[test]
    fn test_wrap_around() {
        let mut deque = VecDeque::with_capacity(16);

        // Fill halfway, then pop some, then push more to cause wrap
        for i in 0..8 {
            deque.push_back(i);
        }

        for _ in 0..4 {
            deque.pop_front();
        }

        // Now head is at index 4, push more to wrap around
        for i in 8..20 {
            deque.push_back(i);
        }

        // Verify correct order
        for (i, expected) in (4..20).enumerate() {
            assert_eq!(*deque.get(i).unwrap(), expected);
        }
    }

    #[test]
    fn test_rotate() {
        let mut deque: VecDeque<i32> = [1, 2, 3, 4, 5].into_iter().collect();

        deque.rotate_left(2);
        assert_eq!(deque.iter(), vec![3, 4, 5, 1, 2]);

        deque.rotate_right(2);
        assert_eq!(deque.iter(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_swap() {
        let mut deque: VecDeque<i32> = [1, 2, 3, 4, 5].into_iter().collect();

        deque.swap(1, 3);
        assert_eq!(deque.iter(), vec![1, 4, 3, 2, 5]);
    }

    #[test]
    fn test_iter() {
        let mut deque = VecDeque::new();
        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        let v = deque.iter();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_from_iterator() {
        let deque: VecDeque<i32> = [1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(deque.len(), 5);
        assert_eq!(deque.iter(), vec![1, 2, 3, 4, 5]);
    }
}
