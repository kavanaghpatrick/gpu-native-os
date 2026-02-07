//! GPU-Native LinkedList using Pool-Based Allocation
//!
//! THE GPU IS THE COMPUTER.
//!
//! This LinkedList uses a contiguous array pool with index-based links,
//! avoiding pointer chasing and providing O(1) allocation via freelist.
//!
//! # Why Pool-Based for GPU?
//!
//! | Approach | GPU Problem |
//! |----------|-------------|
//! | Pointer-based | Pointer chasing = unpredictable memory |
//! | Per-node alloc | System call overhead, fragmentation |
//! | **Pool + indices** | Contiguous memory, O(1) alloc |
//!
//! # Example
//!
//! ```ignore
//! use gpu_std::collections::LinkedList;
//!
//! let mut list = LinkedList::new();
//! list.push_back(1);
//! list.push_back(2);
//! list.push_front(0);
//!
//! assert_eq!(list.pop_front(), Some(0));
//! assert_eq!(list.pop_back(), Some(2));
//! ```

use core::alloc::Layout;

// ============================================================================
// Constants
// ============================================================================

/// Initial pool capacity
const INITIAL_CAPACITY: usize = 64;

/// Null index marker
const NULL_INDEX: u32 = u32::MAX;

// ============================================================================
// Pool Node
// ============================================================================

/// A node in the pool-based linked list
#[repr(C)]
#[derive(Clone, Copy)]
struct PoolNode<T> {
    /// The stored value
    value: T,

    /// Index of next node (NULL_INDEX = end)
    next: u32,

    /// Index of previous node (NULL_INDEX = end)
    prev: u32,

    /// Flags: bit 0 = is_free
    flags: u32,
}

impl<T: Default> PoolNode<T> {
    const FLAG_FREE: u32 = 1;

    /// Check if node is in the free list
    /// Currently unused but kept for debugging purposes.
    #[allow(dead_code)]
    #[inline]
    fn is_free(&self) -> bool {
        self.flags & Self::FLAG_FREE != 0
    }

    #[inline]
    fn set_free(&mut self) {
        self.flags |= Self::FLAG_FREE;
    }

    #[inline]
    fn set_used(&mut self) {
        self.flags &= !Self::FLAG_FREE;
    }
}

impl<T: Default> Default for PoolNode<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            next: NULL_INDEX,
            prev: NULL_INDEX,
            flags: PoolNode::<T>::FLAG_FREE,
        }
    }
}

// ============================================================================
// LinkedList Implementation
// ============================================================================

/// GPU-native doubly-linked list using pool allocation
///
/// All nodes are stored in a contiguous array for cache efficiency.
/// Allocation is O(1) via freelist, no system calls in hot path.
pub struct LinkedList<T> {
    /// Pool of all nodes
    pool: *mut PoolNode<T>,

    /// Pool capacity
    capacity: usize,

    /// Head of the freelist (first free node)
    free_head: u32,

    /// Head of the actual list
    list_head: u32,

    /// Tail of the actual list
    list_tail: u32,

    /// Number of active elements
    len: usize,
}

impl<T: Copy + Default> LinkedList<T> {
    /// Create a new empty LinkedList
    pub fn new() -> Self {
        Self::with_capacity(INITIAL_CAPACITY)
    }

    /// Create a LinkedList with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(4);

        let pool = unsafe {
            let layout = Layout::array::<PoolNode<T>>(capacity).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut PoolNode<T>;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }

            // Initialize freelist: each node points to next
            for i in 0..capacity {
                let node = &mut *ptr.add(i);
                node.next = if i + 1 < capacity {
                    (i + 1) as u32
                } else {
                    NULL_INDEX
                };
                node.prev = NULL_INDEX;
                node.flags = PoolNode::<T>::FLAG_FREE;
            }

            ptr
        };

        Self {
            pool,
            capacity,
            free_head: 0,
            list_head: NULL_INDEX,
            list_tail: NULL_INDEX,
            len: 0,
        }
    }

    /// Get a reference to a node
    #[inline]
    unsafe fn node(&self, idx: u32) -> &PoolNode<T> {
        &*self.pool.add(idx as usize)
    }

    /// Get a mutable reference to a node
    #[inline]
    unsafe fn node_mut(&mut self, idx: u32) -> &mut PoolNode<T> {
        &mut *self.pool.add(idx as usize)
    }

    /// Allocate a node from the freelist
    ///
    /// O(1) - pop from freelist head
    fn alloc_node(&mut self) -> u32 {
        if self.free_head == NULL_INDEX {
            self.grow();
        }

        let idx = self.free_head;
        unsafe {
            let node_ptr = self.pool.add(idx as usize);
            self.free_head = (*node_ptr).next;
            (*node_ptr).set_used();
            (*node_ptr).next = NULL_INDEX;
            (*node_ptr).prev = NULL_INDEX;
        }
        idx
    }

    /// Free a node back to the freelist
    ///
    /// O(1) - push to freelist head
    fn free_node(&mut self, idx: u32) {
        let old_free_head = self.free_head;
        unsafe {
            let node_ptr = self.pool.add(idx as usize);
            (*node_ptr).set_free();
            (*node_ptr).next = old_free_head;
            (*node_ptr).prev = NULL_INDEX;
        }
        self.free_head = idx;
    }

    /// Push an element to the front
    ///
    /// O(1) - allocate node and link
    pub fn push_front(&mut self, value: T) {
        let idx = self.alloc_node();
        let old_head = self.list_head;

        unsafe {
            let node_ptr = self.pool.add(idx as usize);
            (*node_ptr).value = value;
            (*node_ptr).next = old_head;
            (*node_ptr).prev = NULL_INDEX;
        }

        if old_head != NULL_INDEX {
            unsafe {
                let old_head_ptr = self.pool.add(old_head as usize);
                (*old_head_ptr).prev = idx;
            }
        } else {
            self.list_tail = idx;
        }

        self.list_head = idx;
        self.len += 1;
    }

    /// Push an element to the back
    ///
    /// O(1) - allocate node and link
    pub fn push_back(&mut self, value: T) {
        let idx = self.alloc_node();
        let old_tail = self.list_tail;

        unsafe {
            let node_ptr = self.pool.add(idx as usize);
            (*node_ptr).value = value;
            (*node_ptr).next = NULL_INDEX;
            (*node_ptr).prev = old_tail;
        }

        if old_tail != NULL_INDEX {
            unsafe {
                let old_tail_ptr = self.pool.add(old_tail as usize);
                (*old_tail_ptr).next = idx;
            }
        } else {
            self.list_head = idx;
        }

        self.list_tail = idx;
        self.len += 1;
    }

    /// Pop an element from the front
    ///
    /// O(1) - unlink and free
    pub fn pop_front(&mut self) -> Option<T> {
        if self.list_head == NULL_INDEX {
            return None;
        }

        let idx = self.list_head;
        let value = unsafe {
            let node = self.node(idx);
            let value = node.value;
            let next = node.next;

            self.list_head = next;
            if next != NULL_INDEX {
                self.node_mut(next).prev = NULL_INDEX;
            } else {
                self.list_tail = NULL_INDEX;
            }

            value
        };

        self.free_node(idx);
        self.len -= 1;
        Some(value)
    }

    /// Pop an element from the back
    ///
    /// O(1) - unlink and free
    pub fn pop_back(&mut self) -> Option<T> {
        if self.list_tail == NULL_INDEX {
            return None;
        }

        let idx = self.list_tail;
        let value = unsafe {
            let node = self.node(idx);
            let value = node.value;
            let prev = node.prev;

            self.list_tail = prev;
            if prev != NULL_INDEX {
                self.node_mut(prev).next = NULL_INDEX;
            } else {
                self.list_head = NULL_INDEX;
            }

            value
        };

        self.free_node(idx);
        self.len -= 1;
        Some(value)
    }

    /// Get a reference to the front element
    pub fn front(&self) -> Option<&T> {
        if self.list_head == NULL_INDEX {
            return None;
        }
        unsafe { Some(&self.node(self.list_head).value) }
    }

    /// Get a mutable reference to the front element
    pub fn front_mut(&mut self) -> Option<&mut T> {
        if self.list_head == NULL_INDEX {
            return None;
        }
        unsafe { Some(&mut self.node_mut(self.list_head).value) }
    }

    /// Get a reference to the back element
    pub fn back(&self) -> Option<&T> {
        if self.list_tail == NULL_INDEX {
            return None;
        }
        unsafe { Some(&self.node(self.list_tail).value) }
    }

    /// Get a mutable reference to the back element
    pub fn back_mut(&mut self) -> Option<&mut T> {
        if self.list_tail == NULL_INDEX {
            return None;
        }
        unsafe { Some(&mut self.node_mut(self.list_tail).value) }
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

    /// Clear all elements
    pub fn clear(&mut self) {
        // Return all nodes to freelist
        let mut current = self.list_head;
        while current != NULL_INDEX {
            let next = unsafe { self.node(current).next };
            self.free_node(current);
            current = next;
        }

        self.list_head = NULL_INDEX;
        self.list_tail = NULL_INDEX;
        self.len = 0;
    }

    /// Grow the pool (double capacity)
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;

        let new_pool = unsafe {
            let layout = Layout::array::<PoolNode<T>>(new_capacity).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut PoolNode<T>;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }

            // Copy existing nodes
            for i in 0..self.capacity {
                *ptr.add(i) = *self.pool.add(i);
            }

            // Initialize new nodes in freelist
            for i in self.capacity..new_capacity {
                let node = &mut *ptr.add(i);
                node.next = if i + 1 < new_capacity {
                    (i + 1) as u32
                } else {
                    self.free_head // Connect to old freelist
                };
                node.prev = NULL_INDEX;
                node.flags = PoolNode::<T>::FLAG_FREE;
            }

            ptr
        };

        // Free old pool
        unsafe {
            let old_layout = Layout::array::<PoolNode<T>>(self.capacity).unwrap();
            alloc::alloc::dealloc(self.pool as *mut u8, old_layout);
        }

        self.free_head = self.capacity as u32; // New freelist head
        self.pool = new_pool;
        self.capacity = new_capacity;
    }

    /// Check if list contains a value
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        let mut current = self.list_head;
        while current != NULL_INDEX {
            unsafe {
                let node = self.node(current);
                if node.value == *value {
                    return true;
                }
                current = node.next;
            }
        }
        false
    }
}

impl<T: Copy + Default> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<PoolNode<T>>(self.capacity).unwrap();
            alloc::alloc::dealloc(self.pool as *mut u8, layout);
        }
    }
}

// ============================================================================
// Cursor API for Traversal
// ============================================================================

/// A cursor for traversing the linked list
pub struct Cursor<'a, T> {
    list: &'a LinkedList<T>,
    current: u32,
}

impl<'a, T: Copy + Default> Cursor<'a, T> {
    /// Create a cursor starting at the front
    pub fn new(list: &'a LinkedList<T>) -> Self {
        Self {
            list,
            current: list.list_head,
        }
    }

    /// Get the current element
    pub fn current(&self) -> Option<&T> {
        if self.current == NULL_INDEX {
            return None;
        }
        unsafe { Some(&self.list.node(self.current).value) }
    }

    /// Move to the next element
    pub fn move_next(&mut self) -> bool {
        if self.current == NULL_INDEX {
            return false;
        }
        unsafe {
            self.current = self.list.node(self.current).next;
        }
        self.current != NULL_INDEX
    }

    /// Move to the previous element
    pub fn move_prev(&mut self) -> bool {
        if self.current == NULL_INDEX {
            return false;
        }
        unsafe {
            self.current = self.list.node(self.current).prev;
        }
        self.current != NULL_INDEX
    }
}

// ============================================================================
// Iteration (returns Vec, not lazy iterator - GPU can't do lazy)
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: Copy + Default> LinkedList<T> {
    /// Collect all elements into a Vec
    pub fn iter(&self) -> alloc::vec::Vec<T> {
        let mut result = alloc::vec::Vec::with_capacity(self.len);
        let mut current = self.list_head;

        while current != NULL_INDEX {
            unsafe {
                let node = self.node(current);
                result.push(node.value);
                current = node.next;
            }
        }

        result
    }

    /// Collect all elements in reverse order
    pub fn iter_rev(&self) -> alloc::vec::Vec<T> {
        let mut result = alloc::vec::Vec::with_capacity(self.len);
        let mut current = self.list_tail;

        while current != NULL_INDEX {
            unsafe {
                let node = self.node(current);
                result.push(node.value);
                current = node.prev;
            }
        }

        result
    }

    /// Append another list to the back
    pub fn append(&mut self, other: &mut LinkedList<T>) {
        if other.is_empty() {
            return;
        }

        // Add all elements from other
        for value in other.iter() {
            self.push_back(value);
        }

        other.clear();
    }
}

// ============================================================================
// FromIterator
// ============================================================================

#[cfg(feature = "alloc")]
impl<T: Copy + Default> core::iter::FromIterator<T> for LinkedList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut list = LinkedList::with_capacity(lower.max(INITIAL_CAPACITY));

        for item in iter {
            list.push_back(item);
        }

        list
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
    fn test_push_pop_front() {
        let mut list = LinkedList::new();

        list.push_front(1);
        list.push_front(2);
        list.push_front(3);

        assert_eq!(list.len(), 3);
        assert_eq!(list.pop_front(), Some(3));
        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn test_push_pop_back() {
        let mut list = LinkedList::new();

        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.len(), 3);
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.pop_back(), Some(1));
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn test_mixed_operations() {
        let mut list = LinkedList::new();

        list.push_back(2);
        list.push_front(1);
        list.push_back(3);
        list.push_front(0);

        // Order: 0, 1, 2, 3
        assert_eq!(list.pop_front(), Some(0));
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_back(), Some(2));
        assert!(list.is_empty());
    }

    #[test]
    fn test_front_back() {
        let mut list = LinkedList::new();

        assert!(list.front().is_none());
        assert!(list.back().is_none());

        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(*list.front().unwrap(), 1);
        assert_eq!(*list.back().unwrap(), 3);

        *list.front_mut().unwrap() = 10;
        *list.back_mut().unwrap() = 30;

        assert_eq!(*list.front().unwrap(), 10);
        assert_eq!(*list.back().unwrap(), 30);
    }

    #[test]
    fn test_grow() {
        let mut list = LinkedList::with_capacity(4);

        // Push more than initial capacity
        for i in 0..100 {
            list.push_back(i);
        }

        assert_eq!(list.len(), 100);

        // Verify order preserved
        for i in 0..100 {
            assert_eq!(list.pop_front(), Some(i));
        }
    }

    #[test]
    fn test_clear() {
        let mut list = LinkedList::new();

        for i in 0..10 {
            list.push_back(i);
        }

        list.clear();
        assert!(list.is_empty());
        assert!(list.front().is_none());

        // Can still add elements after clear
        list.push_back(42);
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_contains() {
        let mut list = LinkedList::new();

        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert!(list.contains(&2));
        assert!(!list.contains(&4));
    }

    #[test]
    fn test_iter() {
        let mut list = LinkedList::new();

        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.iter(), vec![1, 2, 3]);
        assert_eq!(list.iter_rev(), vec![3, 2, 1]);
    }

    #[test]
    fn test_cursor() {
        let mut list = LinkedList::new();

        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let mut cursor = Cursor::new(&list);

        assert_eq!(*cursor.current().unwrap(), 1);
        assert!(cursor.move_next());
        assert_eq!(*cursor.current().unwrap(), 2);
        assert!(cursor.move_next());
        assert_eq!(*cursor.current().unwrap(), 3);
        assert!(!cursor.move_next());
    }

    #[test]
    fn test_from_iterator() {
        let list: LinkedList<i32> = [1, 2, 3, 4, 5].into_iter().collect();
        assert_eq!(list.len(), 5);
        assert_eq!(list.iter(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reuse_freed_nodes() {
        let mut list = LinkedList::with_capacity(4);

        // Fill and empty multiple times
        for round in 0..5 {
            for i in 0..10 {
                list.push_back(round * 10 + i);
            }
            list.clear();
        }

        // Should still work
        list.push_back(999);
        assert_eq!(list.pop_front(), Some(999));
    }
}
