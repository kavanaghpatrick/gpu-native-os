//! GPU-Native HashSet using Cuckoo Hashing
//!
//! THE GPU IS THE COMPUTER.
//!
//! This HashSet wraps HashMap<K, ()> to provide O(1) guaranteed operations.
//! All the SIMD-divergence-avoiding properties of HashMap are inherited.
//!
//! # Example
//!
//! ```ignore
//! use gpu_std::collections::HashSet;
//!
//! let mut set = HashSet::new();
//! set.insert(1);
//! set.insert(2);
//! set.insert(1); // duplicate, returns false
//!
//! assert!(set.contains(&1));
//! assert_eq!(set.len(), 2);
//! ```

use core::hash::Hash;
use super::HashMap;

/// GPU-native HashSet using Cuckoo hashing
///
/// Provides O(1) guaranteed lookup with exactly 2 table accesses.
/// This is a thin wrapper around `HashMap<K, ()>`.
pub struct HashSet<K> {
    inner: HashMap<K, ()>,
}

impl<K> HashSet<K>
where
    K: Hash + Eq + Copy + Default,
{
    /// Create a new empty HashSet
    #[inline]
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Create a HashSet with specified capacity
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
        }
    }

    /// Check if the set contains a key
    ///
    /// O(1) guaranteed - exactly 2 bucket lookups plus stash check.
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }

    /// Insert a key into the set
    ///
    /// Returns `true` if the key was newly inserted, `false` if it already existed.
    #[inline]
    pub fn insert(&mut self, key: K) -> bool {
        self.inner.insert(key, ()).is_none()
    }

    /// Remove a key from the set
    ///
    /// Returns `true` if the key was present and removed.
    #[inline]
    pub fn remove(&mut self, key: &K) -> bool {
        self.inner.remove(key).is_some()
    }

    /// Get number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if set is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear all elements
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get capacity (approximate)
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

impl<K> Default for HashSet<K>
where
    K: Hash + Eq + Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Set Operations (Batch-Friendly)
// ============================================================================

#[cfg(feature = "alloc")]
impl<K> HashSet<K>
where
    K: Hash + Eq + Copy + Default,
{
    /// Collect all keys into a Vec
    pub fn iter(&self) -> alloc::vec::Vec<K> {
        self.inner.keys()
    }

    /// Union: elements in either set
    ///
    /// GPU-friendly: creates new set with batch inserts
    pub fn union(&self, other: &HashSet<K>) -> HashSet<K> {
        let mut result = HashSet::with_capacity(self.len() + other.len());

        // Insert all from self
        for key in self.iter() {
            result.insert(key);
        }

        // Insert all from other (duplicates ignored)
        for key in other.iter() {
            result.insert(key);
        }

        result
    }

    /// Intersection: elements in both sets
    ///
    /// GPU-friendly: iterates smaller set, checks larger
    pub fn intersection(&self, other: &HashSet<K>) -> HashSet<K> {
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };

        let mut result = HashSet::with_capacity(smaller.len());

        for key in smaller.iter() {
            if larger.contains(&key) {
                result.insert(key);
            }
        }

        result
    }

    /// Difference: elements in self but not in other
    pub fn difference(&self, other: &HashSet<K>) -> HashSet<K> {
        let mut result = HashSet::with_capacity(self.len());

        for key in self.iter() {
            if !other.contains(&key) {
                result.insert(key);
            }
        }

        result
    }

    /// Symmetric difference: elements in either but not both
    pub fn symmetric_difference(&self, other: &HashSet<K>) -> HashSet<K> {
        let mut result = HashSet::with_capacity(self.len() + other.len());

        for key in self.iter() {
            if !other.contains(&key) {
                result.insert(key);
            }
        }

        for key in other.iter() {
            if !self.contains(&key) {
                result.insert(key);
            }
        }

        result
    }

    /// Check if self is a subset of other
    pub fn is_subset(&self, other: &HashSet<K>) -> bool {
        if self.len() > other.len() {
            return false;
        }

        for key in self.iter() {
            if !other.contains(&key) {
                return false;
            }
        }

        true
    }

    /// Check if self is a superset of other
    #[inline]
    pub fn is_superset(&self, other: &HashSet<K>) -> bool {
        other.is_subset(self)
    }

    /// Check if sets are disjoint (no common elements)
    pub fn is_disjoint(&self, other: &HashSet<K>) -> bool {
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };

        for key in smaller.iter() {
            if larger.contains(&key) {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// FromIterator
// ============================================================================

#[cfg(feature = "alloc")]
impl<K> core::iter::FromIterator<K> for HashSet<K>
where
    K: Hash + Eq + Copy + Default,
{
    fn from_iter<I: IntoIterator<Item = K>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut set = HashSet::with_capacity(lower);

        for key in iter {
            set.insert(key);
        }

        set
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
    fn test_basic_operations() {
        let mut set = HashSet::new();

        assert!(set.is_empty());
        assert_eq!(set.len(), 0);

        // Insert
        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(set.insert(3));
        assert!(!set.insert(1)); // Duplicate

        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());

        // Contains
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(!set.contains(&4));

        // Remove
        assert!(set.remove(&2));
        assert!(!set.remove(&2)); // Already removed
        assert_eq!(set.len(), 2);

        // Clear
        set.clear();
        assert!(set.is_empty());
    }

    #[test]
    fn test_set_operations() {
        let mut a = HashSet::new();
        a.insert(1);
        a.insert(2);
        a.insert(3);

        let mut b = HashSet::new();
        b.insert(2);
        b.insert(3);
        b.insert(4);

        // Union
        let union = a.union(&b);
        assert_eq!(union.len(), 4);
        assert!(union.contains(&1));
        assert!(union.contains(&2));
        assert!(union.contains(&3));
        assert!(union.contains(&4));

        // Intersection
        let intersection = a.intersection(&b);
        assert_eq!(intersection.len(), 2);
        assert!(intersection.contains(&2));
        assert!(intersection.contains(&3));

        // Difference
        let diff = a.difference(&b);
        assert_eq!(diff.len(), 1);
        assert!(diff.contains(&1));

        // Symmetric difference
        let sym_diff = a.symmetric_difference(&b);
        assert_eq!(sym_diff.len(), 2);
        assert!(sym_diff.contains(&1));
        assert!(sym_diff.contains(&4));
    }

    #[test]
    fn test_subset_superset() {
        let mut small = HashSet::new();
        small.insert(1);
        small.insert(2);

        let mut large = HashSet::new();
        large.insert(1);
        large.insert(2);
        large.insert(3);

        assert!(small.is_subset(&large));
        assert!(!large.is_subset(&small));
        assert!(large.is_superset(&small));
        assert!(!small.is_superset(&large));

        // Equal sets
        let mut equal = HashSet::new();
        equal.insert(1);
        equal.insert(2);
        assert!(small.is_subset(&equal));
        assert!(equal.is_subset(&small));
    }

    #[test]
    fn test_disjoint() {
        let mut a = HashSet::new();
        a.insert(1);
        a.insert(2);

        let mut b = HashSet::new();
        b.insert(3);
        b.insert(4);

        assert!(a.is_disjoint(&b));

        b.insert(2);
        assert!(!a.is_disjoint(&b));
    }

    #[test]
    fn test_from_iterator() {
        let set: HashSet<i32> = [1, 2, 3, 2, 1].into_iter().collect();
        assert_eq!(set.len(), 3);
    }
}
