//! GPU-Native HashMap using Cuckoo Hashing
//!
//! THE GPU IS THE COMPUTER.
//!
//! This HashMap uses Cuckoo hashing which provides O(1) guaranteed lookup time.
//! This is critical for GPU execution where variable-time operations cause
//! SIMD divergence and waste threads.
//!
//! # Why Cuckoo Hashing?
//!
//! | Approach | GPU Problem |
//! |----------|-------------|
//! | Linear probing | Variable iterations = SIMD divergence |
//! | Chaining | Pointer chasing = cache misses |
//! | Robin Hood | Variable work = SIMD divergence |
//! | **Cuckoo** | Exactly 2 lookups, always |
//!
//! # Implementation Details
//!
//! - Two hash tables with different hash functions
//! - 16 entries per bucket (128 bytes = Apple Silicon cache line)
//! - Bounded eviction chains to prevent infinite loops
//! - Small stash for overflow handling
//!
//! # Example
//!
//! ```ignore
//! use gpu_std::collections::HashMap;
//!
//! let mut map = HashMap::new();
//! map.insert(1, "hello");
//! map.insert(2, "world");
//!
//! assert_eq!(map.get(&1), Some(&"hello"));
//! assert_eq!(map.len(), 2);
//! ```

use core::hash::{Hash, Hasher};

// ============================================================================
// Constants
// ============================================================================

/// Entries per bucket - sized for 128-byte cache line with 8-byte entries
const BUCKET_SIZE: usize = 16;

/// Default number of buckets per table
const DEFAULT_BUCKETS: usize = 64;

/// Maximum eviction chain length before using stash
const MAX_EVICTIONS: usize = 64;

/// Stash size for overflow entries
const STASH_SIZE: usize = 16;

// ============================================================================
// Hash Functions
// ============================================================================

/// FxHash - fast, simple, GPU-friendly hash
///
/// This is a multiply-XOR hash that's very fast and works well for
/// integer keys. Not cryptographically secure but perfect for hash tables.
#[derive(Clone, Copy, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const K: u64 = 0x517cc1b727220a95;

    #[inline]
    pub fn new() -> Self {
        Self { hash: 0 }
    }

    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.rotate_left(5) ^ word;
        self.hash = self.hash.wrapping_mul(Self::K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Process 8 bytes at a time
        let mut chunks = bytes.chunks_exact(8);
        for chunk in chunks.by_ref() {
            let word = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            self.add(word);
        }

        // Process remaining bytes
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut word = 0u64;
            for (i, &byte) in remainder.iter().enumerate() {
                word |= (byte as u64) << (i * 8);
            }
            self.add(word);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// PCG hash for generating second hash function
/// This is branchless and has excellent distribution.
#[inline]
fn pcg_hash(input: u64) -> u64 {
    let state = input.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(0x14057b7ef767814f);
    let word = ((state >> ((state >> 59) + 5)) ^ state).wrapping_mul(0xaef17502108ef2d9);
    (word >> 43) ^ word
}

/// Compute first hash (using FxHash)
#[inline]
fn hash1<K: Hash>(key: &K) -> u64 {
    let mut hasher = FxHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Compute second hash (using PCG on first hash)
#[inline]
fn hash2<K: Hash>(key: &K) -> u64 {
    pcg_hash(hash1(key))
}

// ============================================================================
// Entry Types
// ============================================================================

/// A single entry in the hash table
#[repr(C)]
#[derive(Clone, Copy)]
struct CuckooEntry<K, V> {
    key: K,
    value: V,
}

impl<K: Default, V: Default> Default for CuckooEntry<K, V> {
    fn default() -> Self {
        Self {
            key: K::default(),
            value: V::default(),
        }
    }
}

/// A bucket containing multiple entries (sized for cache line)
#[repr(C, align(128))]
struct CuckooBucket<K, V> {
    entries: [CuckooEntry<K, V>; BUCKET_SIZE],
    occupancy: u16,  // Bitmap of occupied slots
    _padding: [u8; 14],  // Pad to 128 bytes
}

impl<K: Default + Copy, V: Default + Copy> Default for CuckooBucket<K, V> {
    fn default() -> Self {
        Self {
            entries: [CuckooEntry::default(); BUCKET_SIZE],
            occupancy: 0,
            _padding: [0; 14],
        }
    }
}

impl<K: Default + Copy, V: Default + Copy> CuckooBucket<K, V> {
    /// Find empty slot in bucket, returns index or None
    #[inline]
    fn find_empty(&self) -> Option<usize> {
        // Check occupancy bitmap
        for i in 0..BUCKET_SIZE {
            if self.occupancy & (1 << i) == 0 {
                return Some(i);
            }
        }
        None
    }

    /// Mark slot as occupied
    #[inline]
    fn set_occupied(&mut self, idx: usize) {
        self.occupancy |= 1 << idx;
    }

    /// Mark slot as empty
    #[inline]
    fn set_empty(&mut self, idx: usize) {
        self.occupancy &= !(1 << idx);
    }

    /// Check if slot is occupied
    #[inline]
    fn is_occupied(&self, idx: usize) -> bool {
        self.occupancy & (1 << idx) != 0
    }
}

// ============================================================================
// HashMap Implementation
// ============================================================================

/// GPU-native HashMap using Cuckoo hashing
///
/// Provides O(1) guaranteed lookup with exactly 2 table accesses.
/// Uses two hash tables with different hash functions and handles
/// collisions by "kicking out" existing entries to their alternate location.
pub struct HashMap<K, V> {
    /// First hash table
    table1: *mut CuckooBucket<K, V>,
    /// Second hash table
    table2: *mut CuckooBucket<K, V>,
    /// Stash for entries that couldn't be placed
    stash: *mut CuckooEntry<K, V>,
    /// Number of buckets in each table
    num_buckets: usize,
    /// Number of entries
    len: usize,
    /// Number of entries in stash
    stash_len: usize,
}

impl<K, V> HashMap<K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    /// Create a new empty HashMap
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_BUCKETS * BUCKET_SIZE)
    }

    /// Create a HashMap with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let num_buckets = (capacity + BUCKET_SIZE - 1) / BUCKET_SIZE;
        let num_buckets = num_buckets.max(4); // Minimum 4 buckets

        // Allocate tables
        let table1 = Self::alloc_buckets(num_buckets);
        let table2 = Self::alloc_buckets(num_buckets);
        let stash = Self::alloc_stash();

        Self {
            table1,
            table2,
            stash,
            num_buckets,
            len: 0,
            stash_len: 0,
        }
    }

    /// Allocate bucket array
    fn alloc_buckets(count: usize) -> *mut CuckooBucket<K, V> {
        unsafe {
            let layout = core::alloc::Layout::array::<CuckooBucket<K, V>>(count).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut CuckooBucket<K, V>;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }
            ptr
        }
    }

    /// Allocate stash array
    fn alloc_stash() -> *mut CuckooEntry<K, V> {
        unsafe {
            let layout = core::alloc::Layout::array::<CuckooEntry<K, V>>(STASH_SIZE).unwrap();
            let ptr = alloc::alloc::alloc_zeroed(layout) as *mut CuckooEntry<K, V>;
            if ptr.is_null() {
                alloc::alloc::handle_alloc_error(layout);
            }
            ptr
        }
    }

    /// Get bucket index for first table
    #[inline]
    fn bucket1(&self, key: &K) -> usize {
        (hash1(key) as usize) % self.num_buckets
    }

    /// Get bucket index for second table
    #[inline]
    fn bucket2(&self, key: &K) -> usize {
        (hash2(key) as usize) % self.num_buckets
    }

    /// Get reference to bucket in table 1
    #[inline]
    unsafe fn get_bucket1(&self, idx: usize) -> &CuckooBucket<K, V> {
        &*self.table1.add(idx)
    }

    /// Get mutable reference to bucket in table 1
    #[inline]
    unsafe fn get_bucket1_mut(&mut self, idx: usize) -> &mut CuckooBucket<K, V> {
        &mut *self.table1.add(idx)
    }

    /// Get reference to bucket in table 2
    #[inline]
    unsafe fn get_bucket2(&self, idx: usize) -> &CuckooBucket<K, V> {
        &*self.table2.add(idx)
    }

    /// Get mutable reference to bucket in table 2
    #[inline]
    unsafe fn get_bucket2_mut(&mut self, idx: usize) -> &mut CuckooBucket<K, V> {
        &mut *self.table2.add(idx)
    }

    /// Look up a key in the map
    ///
    /// This is O(1) guaranteed - exactly 2 bucket lookups plus stash check.
    pub fn get(&self, key: &K) -> Option<&V> {
        unsafe {
            // Check table 1
            let b1 = self.bucket1(key);
            let bucket1 = self.get_bucket1(b1);
            for i in 0..BUCKET_SIZE {
                if bucket1.is_occupied(i) && bucket1.entries[i].key == *key {
                    return Some(&bucket1.entries[i].value);
                }
            }

            // Check table 2
            let b2 = self.bucket2(key);
            let bucket2 = self.get_bucket2(b2);
            for i in 0..BUCKET_SIZE {
                if bucket2.is_occupied(i) && bucket2.entries[i].key == *key {
                    return Some(&bucket2.entries[i].value);
                }
            }

            // Check stash
            for i in 0..self.stash_len {
                let entry = &*self.stash.add(i);
                if entry.key == *key {
                    return Some(&entry.value);
                }
            }

            None
        }
    }

    /// Get mutable reference to value
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        // Use raw pointers to avoid borrow conflicts
        let b1 = self.bucket1(key);
        let b2 = self.bucket2(key);
        let table1 = self.table1;
        let table2 = self.table2;
        let stash = self.stash;
        let stash_len = self.stash_len;

        unsafe {
            // Check table 1
            let bucket1 = &mut *table1.add(b1);
            for i in 0..BUCKET_SIZE {
                if bucket1.is_occupied(i) && bucket1.entries[i].key == *key {
                    return Some(&mut bucket1.entries[i].value);
                }
            }

            // Check table 2
            let bucket2 = &mut *table2.add(b2);
            for i in 0..BUCKET_SIZE {
                if bucket2.is_occupied(i) && bucket2.entries[i].key == *key {
                    return Some(&mut bucket2.entries[i].value);
                }
            }

            // Check stash
            for i in 0..stash_len {
                let entry = &mut *stash.add(i);
                if entry.key == *key {
                    return Some(&mut entry.value);
                }
            }

            None
        }
    }

    /// Insert a key-value pair
    ///
    /// Returns the old value if the key was already present.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key already exists and update
        if let Some(existing) = self.get_mut(&key) {
            let old = *existing;
            *existing = value;
            return Some(old);
        }

        // Try to insert into table 1
        unsafe {
            let b1 = self.bucket1(&key);
            let bucket1 = self.get_bucket1_mut(b1);
            if let Some(slot) = bucket1.find_empty() {
                bucket1.entries[slot] = CuckooEntry { key, value };
                bucket1.set_occupied(slot);
                self.len += 1;
                return None;
            }

            // Try to insert into table 2
            let b2 = self.bucket2(&key);
            let bucket2 = self.get_bucket2_mut(b2);
            if let Some(slot) = bucket2.find_empty() {
                bucket2.entries[slot] = CuckooEntry { key, value };
                bucket2.set_occupied(slot);
                self.len += 1;
                return None;
            }
        }

        // Both buckets full - need to evict
        self.insert_with_eviction(key, value);
        None
    }

    /// Insert with eviction chain
    fn insert_with_eviction(&mut self, mut key: K, mut value: V) {
        let mut use_table1 = true;

        for _ in 0..MAX_EVICTIONS {
            unsafe {
                if use_table1 {
                    let b1 = self.bucket1(&key);
                    let bucket1 = self.get_bucket1_mut(b1);

                    // Pick a victim (first occupied slot)
                    for i in 0..BUCKET_SIZE {
                        if bucket1.is_occupied(i) {
                            // Swap with victim
                            let victim_key = bucket1.entries[i].key;
                            let victim_value = bucket1.entries[i].value;
                            bucket1.entries[i] = CuckooEntry { key, value };
                            key = victim_key;
                            value = victim_value;
                            break;
                        }
                    }
                } else {
                    let b2 = self.bucket2(&key);
                    let bucket2 = self.get_bucket2_mut(b2);

                    // Try to place the evicted entry
                    if let Some(slot) = bucket2.find_empty() {
                        bucket2.entries[slot] = CuckooEntry { key, value };
                        bucket2.set_occupied(slot);
                        self.len += 1;
                        return;
                    }

                    // Evict from table 2
                    for i in 0..BUCKET_SIZE {
                        if bucket2.is_occupied(i) {
                            let victim_key = bucket2.entries[i].key;
                            let victim_value = bucket2.entries[i].value;
                            bucket2.entries[i] = CuckooEntry { key, value };
                            key = victim_key;
                            value = victim_value;
                            break;
                        }
                    }
                }
                use_table1 = !use_table1;

                // Check if evicted entry can go to its alternate location
                if use_table1 {
                    let b1 = self.bucket1(&key);
                    let bucket1 = self.get_bucket1_mut(b1);
                    if let Some(slot) = bucket1.find_empty() {
                        bucket1.entries[slot] = CuckooEntry { key, value };
                        bucket1.set_occupied(slot);
                        self.len += 1;
                        return;
                    }
                }
            }
        }

        // Eviction chain too long - use stash
        self.add_to_stash(key, value);
    }

    /// Add entry to stash (overflow buffer)
    fn add_to_stash(&mut self, key: K, value: V) {
        // Maximum resize attempts before giving up (prevents infinite loops)
        const MAX_RESIZE_ATTEMPTS: usize = 16;

        if self.stash_len < STASH_SIZE {
            unsafe {
                let entry = &mut *self.stash.add(self.stash_len);
                entry.key = key;
                entry.value = value;
                self.stash_len += 1;
                self.len += 1;
            }
        } else {
            // Stash full - resize and retry WITHOUT recursion
            // Keep resizing until this entry fits (bounded to prevent infinite loop)
            for _attempt in 0..MAX_RESIZE_ATTEMPTS {
                self.resize();
                // Try to place directly into table 1
                let b1 = self.bucket1(&key);
                unsafe {
                    let bucket1 = self.get_bucket1_mut(b1);
                    if let Some(slot) = bucket1.find_empty() {
                        bucket1.entries[slot] = CuckooEntry { key, value };
                        bucket1.set_occupied(slot);
                        self.len += 1;
                        return;
                    }
                }
                // Try table 2
                let b2 = self.bucket2(&key);
                unsafe {
                    let bucket2 = self.get_bucket2_mut(b2);
                    if let Some(slot) = bucket2.find_empty() {
                        bucket2.entries[slot] = CuckooEntry { key, value };
                        bucket2.set_occupied(slot);
                        self.len += 1;
                        return;
                    }
                }
                // Both full after resize, try stash
                if self.stash_len < STASH_SIZE {
                    unsafe {
                        let entry = &mut *self.stash.add(self.stash_len);
                        entry.key = key;
                        entry.value = value;
                        self.stash_len += 1;
                        self.len += 1;
                    }
                    return;
                }
                // Still no room, resize again
            }
            // CRITICAL: Exceeded maximum resize attempts - hash distribution failure
            panic!("HashMap insert failed: exceeded {} resize attempts. Hash distribution failure.", MAX_RESIZE_ATTEMPTS);
        }
    }

    /// Try to insert without using stash or recursion - returns true if successful
    fn try_insert_direct(&mut self, key: K, value: V) -> bool {
        // Try table 1
        let b1 = self.bucket1(&key);
        unsafe {
            let bucket1 = self.get_bucket1_mut(b1);
            if let Some(slot) = bucket1.find_empty() {
                bucket1.entries[slot] = CuckooEntry { key, value };
                bucket1.set_occupied(slot);
                self.len += 1;
                return true;
            }
        }
        // Try table 2
        let b2 = self.bucket2(&key);
        unsafe {
            let bucket2 = self.get_bucket2_mut(b2);
            if let Some(slot) = bucket2.find_empty() {
                bucket2.entries[slot] = CuckooEntry { key, value };
                bucket2.set_occupied(slot);
                self.len += 1;
                return true;
            }
        }
        false
    }

    /// Resize the hash tables (double capacity) and rehash all entries
    /// Uses iterative rehashing to avoid recursion
    fn resize(&mut self) {
        let old_table1 = self.table1;
        let old_table2 = self.table2;
        let old_stash = self.stash;
        let old_num_buckets = self.num_buckets;
        let old_stash_len = self.stash_len;

        // Double the number of buckets
        let new_num_buckets = old_num_buckets * 2;

        // Allocate new tables
        self.table1 = Self::alloc_buckets(new_num_buckets);
        self.table2 = Self::alloc_buckets(new_num_buckets);
        self.stash = Self::alloc_stash();
        self.num_buckets = new_num_buckets;
        self.len = 0;
        self.stash_len = 0;

        // Rehash all entries from old table 1 using direct insert (no recursion)
        unsafe {
            for i in 0..old_num_buckets {
                let bucket = &*old_table1.add(i);
                for j in 0..BUCKET_SIZE {
                    if bucket.is_occupied(j) {
                        let entry = &bucket.entries[j];
                        self.rehash_entry(entry.key, entry.value);
                    }
                }
            }

            // Rehash all entries from old table 2
            for i in 0..old_num_buckets {
                let bucket = &*old_table2.add(i);
                for j in 0..BUCKET_SIZE {
                    if bucket.is_occupied(j) {
                        let entry = &bucket.entries[j];
                        self.rehash_entry(entry.key, entry.value);
                    }
                }
            }

            // Rehash entries from old stash
            for i in 0..old_stash_len {
                let entry = &*old_stash.add(i);
                self.rehash_entry(entry.key, entry.value);
            }

            // Free old tables
            let layout1 = core::alloc::Layout::array::<CuckooBucket<K, V>>(old_num_buckets).unwrap();
            alloc::alloc::dealloc(old_table1 as *mut u8, layout1);

            let layout2 = core::alloc::Layout::array::<CuckooBucket<K, V>>(old_num_buckets).unwrap();
            alloc::alloc::dealloc(old_table2 as *mut u8, layout2);

            let layout_stash = core::alloc::Layout::array::<CuckooEntry<K, V>>(STASH_SIZE).unwrap();
            alloc::alloc::dealloc(old_stash as *mut u8, layout_stash);
        }
    }

    /// Rehash a single entry during resize - does NOT trigger recursive resize
    /// Uses eviction but overflow goes to stash without further resize
    fn rehash_entry(&mut self, key: K, value: V) {
        // Try direct insert first
        if self.try_insert_direct(key, value) {
            return;
        }

        // Need eviction - similar to insert_with_eviction but no recursive resize
        let mut cur_key = key;
        let mut cur_value = value;
        let mut use_table1 = true;

        for _ in 0..MAX_EVICTIONS {
            unsafe {
                if use_table1 {
                    let b1 = self.bucket1(&cur_key);
                    let bucket1 = self.get_bucket1_mut(b1);

                    // Try to find empty slot
                    if let Some(slot) = bucket1.find_empty() {
                        bucket1.entries[slot] = CuckooEntry { key: cur_key, value: cur_value };
                        bucket1.set_occupied(slot);
                        self.len += 1;
                        return;
                    }

                    // Evict first occupied entry
                    for i in 0..BUCKET_SIZE {
                        if bucket1.is_occupied(i) {
                            let victim_key = bucket1.entries[i].key;
                            let victim_value = bucket1.entries[i].value;
                            bucket1.entries[i] = CuckooEntry { key: cur_key, value: cur_value };
                            cur_key = victim_key;
                            cur_value = victim_value;
                            break;
                        }
                    }
                } else {
                    let b2 = self.bucket2(&cur_key);
                    let bucket2 = self.get_bucket2_mut(b2);

                    // Try to find empty slot
                    if let Some(slot) = bucket2.find_empty() {
                        bucket2.entries[slot] = CuckooEntry { key: cur_key, value: cur_value };
                        bucket2.set_occupied(slot);
                        self.len += 1;
                        return;
                    }

                    // Evict first occupied entry
                    for i in 0..BUCKET_SIZE {
                        if bucket2.is_occupied(i) {
                            let victim_key = bucket2.entries[i].key;
                            let victim_value = bucket2.entries[i].value;
                            bucket2.entries[i] = CuckooEntry { key: cur_key, value: cur_value };
                            cur_key = victim_key;
                            cur_value = victim_value;
                            break;
                        }
                    }
                }
                use_table1 = !use_table1;
            }
        }

        // Eviction chain exhausted - use stash (no resize during rehash)
        if self.stash_len < STASH_SIZE {
            unsafe {
                let entry = &mut *self.stash.add(self.stash_len);
                entry.key = cur_key;
                entry.value = cur_value;
                self.stash_len += 1;
                self.len += 1;
            }
        } else {
            // CRITICAL: Never silently drop data - this violates project principles.
            // If we reach here, our hash function has catastrophic distribution.
            panic!("HashMap rehash failed: stash overflow after resize. Hash distribution failure.");
        }
    }

    /// Remove a key from the map
    pub fn remove(&mut self, key: &K) -> Option<V> {
        unsafe {
            // Check table 1
            let b1 = self.bucket1(key);
            let bucket1 = self.get_bucket1_mut(b1);
            for i in 0..BUCKET_SIZE {
                if bucket1.is_occupied(i) && bucket1.entries[i].key == *key {
                    let value = bucket1.entries[i].value;
                    bucket1.set_empty(i);
                    self.len -= 1;
                    return Some(value);
                }
            }

            // Check table 2
            let b2 = self.bucket2(key);
            let bucket2 = self.get_bucket2_mut(b2);
            for i in 0..BUCKET_SIZE {
                if bucket2.is_occupied(i) && bucket2.entries[i].key == *key {
                    let value = bucket2.entries[i].value;
                    bucket2.set_empty(i);
                    self.len -= 1;
                    return Some(value);
                }
            }

            // Check stash
            for i in 0..self.stash_len {
                let entry = &*self.stash.add(i);
                if entry.key == *key {
                    let value = entry.value;
                    // Move last stash entry to this slot
                    if i < self.stash_len - 1 {
                        let last = &*self.stash.add(self.stash_len - 1);
                        let dest = &mut *self.stash.add(i);
                        dest.key = last.key;
                        dest.value = last.value;
                    }
                    self.stash_len -= 1;
                    self.len -= 1;
                    return Some(value);
                }
            }

            None
        }
    }

    /// Check if key exists in map
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Get number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if map is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        unsafe {
            // Clear table 1
            for i in 0..self.num_buckets {
                let bucket = self.get_bucket1_mut(i);
                bucket.occupancy = 0;
            }

            // Clear table 2
            for i in 0..self.num_buckets {
                let bucket = self.get_bucket2_mut(i);
                bucket.occupancy = 0;
            }

            self.len = 0;
            self.stash_len = 0;
        }
    }

    /// Get capacity (approximate - actual depends on hash distribution)
    pub fn capacity(&self) -> usize {
        self.num_buckets * BUCKET_SIZE * 2 + STASH_SIZE
    }

    /// Entry API for in-place modification
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        // Check if key exists
        unsafe {
            let b1 = self.bucket1(&key);
            let bucket1 = self.get_bucket1(b1);
            for i in 0..BUCKET_SIZE {
                if bucket1.is_occupied(i) && bucket1.entries[i].key == key {
                    return Entry::Occupied(OccupiedEntry {
                        map: self,
                        key,
                    });
                }
            }

            let b2 = self.bucket2(&key);
            let bucket2 = self.get_bucket2(b2);
            for i in 0..BUCKET_SIZE {
                if bucket2.is_occupied(i) && bucket2.entries[i].key == key {
                    return Entry::Occupied(OccupiedEntry {
                        map: self,
                        key,
                    });
                }
            }

            for i in 0..self.stash_len {
                let entry = &*self.stash.add(i);
                if entry.key == key {
                    return Entry::Occupied(OccupiedEntry {
                        map: self,
                        key,
                    });
                }
            }
        }

        Entry::Vacant(VacantEntry {
            map: self,
            key,
        })
    }
}

impl<K, V> Default for HashMap<K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for HashMap<K, V> {
    fn drop(&mut self) {
        unsafe {
            // Free table 1
            let layout1 = core::alloc::Layout::array::<CuckooBucket<K, V>>(self.num_buckets).unwrap();
            alloc::alloc::dealloc(self.table1 as *mut u8, layout1);

            // Free table 2
            let layout2 = core::alloc::Layout::array::<CuckooBucket<K, V>>(self.num_buckets).unwrap();
            alloc::alloc::dealloc(self.table2 as *mut u8, layout2);

            // Free stash
            let layout_stash = core::alloc::Layout::array::<CuckooEntry<K, V>>(STASH_SIZE).unwrap();
            alloc::alloc::dealloc(self.stash as *mut u8, layout_stash);
        }
    }
}

// ============================================================================
// Entry API
// ============================================================================

/// Entry in the map, either occupied or vacant
pub enum Entry<'a, K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    /// Insert value if vacant, or return reference to existing
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Insert with function if vacant
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Get key reference
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => &entry.key,
            Entry::Vacant(entry) => &entry.key,
        }
    }
}

/// An occupied entry
pub struct OccupiedEntry<'a, K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    map: &'a mut HashMap<K, V>,
    key: K,
}

impl<'a, K, V> OccupiedEntry<'a, K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    /// Get reference to value
    pub fn get(&self) -> &V {
        self.map.get(&self.key).unwrap()
    }

    /// Get mutable reference to value
    pub fn get_mut(&mut self) -> &mut V {
        self.map.get_mut(&self.key).unwrap()
    }

    /// Convert to mutable reference
    pub fn into_mut(self) -> &'a mut V {
        self.map.get_mut(&self.key).unwrap()
    }

    /// Get the key
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Remove entry and return value
    pub fn remove(self) -> V {
        self.map.remove(&self.key).unwrap()
    }
}

/// A vacant entry
pub struct VacantEntry<'a, K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    map: &'a mut HashMap<K, V>,
    key: K,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    /// Insert value and return mutable reference
    pub fn insert(self, value: V) -> &'a mut V {
        self.map.insert(self.key, value);
        self.map.get_mut(&self.key).unwrap()
    }

    /// Get the key
    pub fn key(&self) -> &K {
        &self.key
    }
}

// ============================================================================
// Iteration (returns Vec, not lazy iterator - GPU can't do lazy)
// ============================================================================

#[cfg(feature = "alloc")]
impl<K, V> HashMap<K, V>
where
    K: Hash + Eq + Copy + Default,
    V: Copy + Default,
{
    /// Collect all key-value pairs into a Vec
    pub fn iter(&self) -> alloc::vec::Vec<(K, V)> {
        let mut result = alloc::vec::Vec::with_capacity(self.len);

        unsafe {
            // Collect from table 1
            for i in 0..self.num_buckets {
                let bucket = self.get_bucket1(i);
                for j in 0..BUCKET_SIZE {
                    if bucket.is_occupied(j) {
                        result.push((bucket.entries[j].key, bucket.entries[j].value));
                    }
                }
            }

            // Collect from table 2
            for i in 0..self.num_buckets {
                let bucket = self.get_bucket2(i);
                for j in 0..BUCKET_SIZE {
                    if bucket.is_occupied(j) {
                        result.push((bucket.entries[j].key, bucket.entries[j].value));
                    }
                }
            }

            // Collect from stash
            for i in 0..self.stash_len {
                let entry = &*self.stash.add(i);
                result.push((entry.key, entry.value));
            }
        }

        result
    }

    /// Collect all keys into a Vec
    pub fn keys(&self) -> alloc::vec::Vec<K> {
        self.iter().into_iter().map(|(k, _)| k).collect()
    }

    /// Collect all values into a Vec
    pub fn values(&self) -> alloc::vec::Vec<V> {
        self.iter().into_iter().map(|(_, v)| v).collect()
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
    fn test_basic_insert_get() {
        let mut map = HashMap::new();

        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), Some(&"two"));
        assert_eq!(map.get(&3), Some(&"three"));
        assert_eq!(map.get(&4), None);
    }

    #[test]
    fn test_update_existing() {
        let mut map = HashMap::new();

        map.insert(1, 100);
        assert_eq!(map.get(&1), Some(&100));

        map.insert(1, 200);
        assert_eq!(map.get(&1), Some(&200));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut map = HashMap::new();

        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        assert_eq!(map.remove(&2), Some("two"));
        assert_eq!(map.get(&2), None);
        assert_eq!(map.len(), 2);

        assert_eq!(map.remove(&2), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_contains_key() {
        let mut map = HashMap::new();

        map.insert("key1", 1);
        map.insert("key2", 2);

        assert!(map.contains_key(&"key1"));
        assert!(map.contains_key(&"key2"));
        assert!(!map.contains_key(&"key3"));
    }

    #[test]
    fn test_len_and_empty() {
        let mut map: HashMap<i32, i32> = HashMap::new();

        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        map.insert(1, 1);
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);

        map.insert(2, 2);
        assert_eq!(map.len(), 2);

        map.remove(&1);
        assert_eq!(map.len(), 1);

        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = HashMap::new();

        for i in 0..100 {
            map.insert(i, i * 10);
        }

        assert_eq!(map.len(), 100);

        map.clear();

        assert!(map.is_empty());
        assert_eq!(map.get(&50), None);
    }

    #[test]
    fn test_many_entries() {
        let mut map = HashMap::new();

        // Insert many entries to trigger resizing
        for i in 0..1000 {
            map.insert(i, i * 2);
        }

        assert_eq!(map.len(), 1000);

        // Verify all entries
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_eviction_chain() {
        // Insert keys that likely hash to same buckets to test eviction
        let mut map = HashMap::with_capacity(16);

        for i in 0..100 {
            map.insert(i, i);
        }

        // All should be retrievable
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_stash_usage() {
        // Fill map to near capacity to force stash usage
        let mut map = HashMap::with_capacity(8);

        for i in 0..200 {
            map.insert(i, i);
        }

        // All should be retrievable (some may be in stash)
        for i in 0..200 {
            assert_eq!(map.get(&i), Some(&i), "Failed to find key {}", i);
        }
    }

    #[test]
    fn test_get_mut() {
        let mut map = HashMap::new();

        map.insert(1, 100);

        if let Some(value) = map.get_mut(&1) {
            *value = 200;
        }

        assert_eq!(map.get(&1), Some(&200));
    }

    #[test]
    fn test_entry_api_vacant() {
        let mut map: HashMap<i32, i32> = HashMap::new();

        match map.entry(1) {
            Entry::Vacant(e) => {
                e.insert(100);
            }
            Entry::Occupied(_) => panic!("Expected vacant entry"),
        }

        assert_eq!(map.get(&1), Some(&100));
    }

    #[test]
    fn test_entry_api_occupied() {
        let mut map = HashMap::new();
        map.insert(1, 100);

        match map.entry(1) {
            Entry::Vacant(_) => panic!("Expected occupied entry"),
            Entry::Occupied(e) => {
                assert_eq!(*e.get(), 100);
            }
        }
    }

    #[test]
    fn test_or_insert() {
        let mut map: HashMap<i32, i32> = HashMap::new();

        let v = map.entry(1).or_insert(100);
        assert_eq!(*v, 100);

        let v = map.entry(1).or_insert(200);
        assert_eq!(*v, 100); // Should not change
    }

    #[test]
    fn test_or_insert_with() {
        let mut map: HashMap<i32, i32> = HashMap::new();

        let v = map.entry(1).or_insert_with(|| 42 * 2);
        assert_eq!(*v, 84);

        // Counter to track if closure is called
        let mut called = false;
        let v = map.entry(1).or_insert_with(|| {
            called = true;
            999
        });
        assert_eq!(*v, 84);
        assert!(!called); // Closure should not be called
    }

    #[test]
    fn test_bulk_insert() {
        let pairs = [(1, "one"), (2, "two"), (3, "three")];
        let mut map: HashMap<i32, &str> = HashMap::new();
        for (k, v) in pairs {
            map.insert(k, v);
        }

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), Some(&"two"));
        assert_eq!(map.get(&3), Some(&"three"));
    }

    #[test]
    fn test_iter() {
        let mut map = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        let items = map.iter();
        assert_eq!(items.len(), 3);

        // Check all items are present (order not guaranteed)
        let keys: alloc::vec::Vec<i32> = items.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_keys() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");

        let keys = map.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let mut map = HashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        let values = map.values();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&10));
        assert!(values.contains(&20));
        assert!(values.contains(&30));
    }

    #[test]
    fn test_capacity() {
        let map: HashMap<i32, i32> = HashMap::with_capacity(100);
        // Capacity should be at least what we requested
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn test_string_keys() {
        // Test with different key types
        let mut map = HashMap::new();
        map.insert("hello", 1);
        map.insert("world", 2);

        assert_eq!(map.get(&"hello"), Some(&1));
        assert_eq!(map.get(&"world"), Some(&2));
    }

    #[test]
    fn test_zero_key() {
        let mut map = HashMap::new();
        map.insert(0i32, 100);

        assert_eq!(map.get(&0), Some(&100));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_negative_keys() {
        let mut map = HashMap::new();
        map.insert(-1, "negative one");
        map.insert(-100, "negative hundred");
        map.insert(0, "zero");
        map.insert(1, "one");

        assert_eq!(map.get(&-1), Some(&"negative one"));
        assert_eq!(map.get(&-100), Some(&"negative hundred"));
        assert_eq!(map.get(&0), Some(&"zero"));
        assert_eq!(map.get(&1), Some(&"one"));
    }

    #[test]
    fn test_resize_preserves_entries() {
        let mut map = HashMap::with_capacity(8);

        // Insert enough to trigger multiple resizes
        for i in 0..500 {
            map.insert(i, i * 3);
        }

        // All entries should be preserved
        for i in 0..500 {
            assert_eq!(
                map.get(&i),
                Some(&(i * 3)),
                "Entry {} missing after resize",
                i
            );
        }
    }
}
