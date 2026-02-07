# GPU-Native Collections Design Document

**THE GPU IS THE COMPUTER.**

This document specifies the design of GPU-optimized data structures for `gpu_std`. All designs prioritize O(1) operations to avoid SIMD divergence.

---

## Table of Contents

1. [HashSet](#1-hashset)
2. [BTreeMap](#2-btreemap--sorted-array-with-implicit-btree)
3. [VecDeque](#3-vecdeque--ring-buffer)
4. [BinaryHeap](#4-binaryheap--tournament-tree)
5. [LinkedList](#5-linkedlist--pool-based-freelist)
6. [Common Patterns](#common-patterns)

---

## 1. HashSet

### Algorithm: Cuckoo Hashing (HashMap wrapper)

This is trivially implemented as `HashMap<K, ()>` with a thin wrapper API.

### GPU Considerations

- **Zero additional SIMD divergence** - inherits HashMap's O(1) guarantees
- **Same cache-line alignment** (128-byte buckets)
- **No value storage overhead** - `()` is zero-sized

### Struct Layout

```rust
/// GPU-native HashSet using Cuckoo hashing
///
/// O(1) guaranteed contains/insert/remove - exactly 2 bucket lookups.
pub struct HashSet<K> {
    inner: HashMap<K, ()>,
}
```

### API

```rust
impl<K: Hash + Eq + Copy + Default> HashSet<K> {
    pub fn new() -> Self;
    pub fn with_capacity(capacity: usize) -> Self;

    // O(1) guaranteed
    pub fn contains(&self, key: &K) -> bool;
    pub fn insert(&mut self, key: K) -> bool;  // true if new
    pub fn remove(&mut self, key: &K) -> bool; // true if existed

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn clear(&mut self);

    // Set operations (batch-friendly)
    pub fn union(&self, other: &HashSet<K>) -> HashSet<K>;
    pub fn intersection(&self, other: &HashSet<K>) -> HashSet<K>;
    pub fn difference(&self, other: &HashSet<K>) -> HashSet<K>;
}
```

### Implementation Complexity

- **Trivial** - ~50 lines wrapping HashMap
- **Priority: High** - Common use case, minimal effort

---

## 2. BTreeMap / Sorted Array with Implicit B+Tree

### Algorithm: Sorted Array with Pre-computed Lookup Index

**Why NOT a traditional B+Tree:**
- Tree traversal is O(log n) with variable steps = SIMD divergence
- Pointer chasing between nodes = cache misses
- Node splits during insertion = unpredictable work

**GPU-Optimal Solution: Sorted Array + Implicit B+Tree Index**

Store elements in a sorted array. Pre-compute a B+tree-style index that provides O(1) lookup into the array segment containing the target key.

### The Insight

B+Tree is really just a way to skip large portions of a sorted array. We can achieve the same effect with pre-computed "jump tables" at different granularities:

```
Level 2 (coarse):  [0]              [256]            [512]
                    |                 |                |
Level 1 (medium):  [0][64][128][192] [256]...        [512]...
                    |   |    |    |
Level 0 (data):    Sorted array of all elements
```

Each level tells you which segment to look in. Jump from level to level in O(1).

### Struct Layout

```rust
/// Cache-aligned bucket for index levels
#[repr(C, align(128))]
struct IndexBucket {
    keys: [K; 16],        // Pivot keys for this bucket
    offsets: [u32; 16],   // Offsets into next level
    count: u8,            // Number of valid entries
    _padding: [u8; 63],   // Pad to 128 bytes
}

/// GPU-native ordered map using sorted array with lookup index
pub struct BTreeMap<K, V> {
    /// Sorted array of key-value pairs (the actual data)
    data: *mut Entry<K, V>,
    data_len: usize,
    data_capacity: usize,

    /// Index levels (pre-computed B+tree structure)
    /// Level 0: points into data every 16 elements
    /// Level 1: points into level 0 every 16 buckets
    /// Level 2: points into level 1 every 16 buckets
    index_levels: [*mut IndexBucket; 4],  // Up to 16^4 = 65536 elements per level
    level_counts: [usize; 4],

    /// Total number of elements
    len: usize,
}
```

### Lookup Algorithm (O(1) for practical sizes)

```rust
fn get(&self, key: &K) -> Option<&V> {
    // Traverse index levels (max 4 levels = 4 reads)
    let mut offset = 0;
    for level in (0..4).rev() {
        if self.level_counts[level] > 0 {
            let bucket = &self.index_levels[level][offset / 16];
            // Linear scan within 16-element bucket (unrolled, branchless)
            offset = bucket.find_child(key);
        }
    }

    // Final linear scan within 16-element data segment
    let segment = &self.data[offset..offset + 16];
    segment.binary_search_branchless(key)
}
```

### Why This Works for GPU

1. **Fixed traversal depth** (4 levels max) - no SIMD divergence
2. **Cache-line aligned buckets** (128 bytes) - single memory transaction
3. **Branchless linear scan** within buckets - all threads do same work
4. **Pre-computed structure** - no work during lookup

### Insertion Strategy

Insertions are batched and trigger a full rebuild:

```rust
/// Batch insert - O(n log n) rebuild, but amortized O(1) per element
fn insert_batch(&mut self, entries: &[(K, V)]) {
    // 1. Merge new entries with existing data (sorted merge)
    // 2. Rebuild index levels from scratch
    // This is GPU-parallel: N threads can build N segments
}

/// Single insert - deferred to batch
fn insert(&mut self, key: K, value: V) {
    self.pending.push((key, value));
    if self.pending.len() >= BATCH_THRESHOLD {
        self.flush_pending();
    }
}
```

### GPU-Specific Considerations

- **Read-heavy workloads**: Excellent (O(1) lookup)
- **Write-heavy workloads**: Use pending buffer, batch rebuilds
- **Iteration**: O(n) - just walk sorted array
- **Range queries**: O(1) to find start, O(k) to iterate k elements

### Implementation Complexity

- **Medium** - ~300 lines
- **Priority: Medium** - Useful for ordered iteration, but HashMap covers most cases

---

## 3. VecDeque / Ring Buffer

### Algorithm: Power-of-2 Ring Buffer with Atomic Head/Tail

Ring buffers are naturally GPU-friendly because:
- No memory moves on push/pop
- Index calculation is branchless (bitwise AND)
- Head/tail updates are single atomics

### Struct Layout

```rust
/// GPU-native double-ended queue using ring buffer
///
/// O(1) push/pop at both ends, no memory moves.
#[repr(C)]
pub struct VecDeque<T> {
    /// Data buffer (always power-of-2 capacity)
    buffer: *mut T,

    /// Capacity mask (capacity - 1, for bitwise AND)
    mask: usize,

    /// Head index (where next pop_front reads from)
    /// Wrapped via `head & mask`
    head: usize,

    /// Tail index (where next push_back writes to)
    /// `len = tail.wrapping_sub(head) & mask`
    tail: usize,
}
```

### Key Operations

```rust
impl<T: Copy + Default> VecDeque<T> {
    /// O(1) - no branching
    #[inline]
    pub fn push_back(&mut self, value: T) {
        let idx = self.tail & self.mask;
        unsafe { *self.buffer.add(idx) = value; }
        self.tail = self.tail.wrapping_add(1);
    }

    /// O(1) - no branching
    #[inline]
    pub fn push_front(&mut self, value: T) {
        self.head = self.head.wrapping_sub(1);
        let idx = self.head & self.mask;
        unsafe { *self.buffer.add(idx) = value; }
    }

    /// O(1) - no branching
    #[inline]
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() { return None; }
        let idx = self.head & self.mask;
        let value = unsafe { *self.buffer.add(idx) };
        self.head = self.head.wrapping_add(1);
        Some(value)
    }

    /// O(1) - no branching
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() { return None; }
        self.tail = self.tail.wrapping_sub(1);
        let idx = self.tail & self.mask;
        Some(unsafe { *self.buffer.add(idx) })
    }

    /// O(1) random access
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() { return None; }
        let idx = (self.head.wrapping_add(index)) & self.mask;
        Some(unsafe { &*self.buffer.add(idx) })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tail.wrapping_sub(self.head)
    }
}
```

### GPU-Specific Considerations

- **Power-of-2 capacity**: Required for branchless index wrapping
- **Resize strategy**: Double capacity, copy with parallel threads
- **Multi-producer (optional)**: Use atomic CAS for tail updates

### Concurrent Variant (SPSC Queue)

For lock-free single-producer single-consumer:

```rust
pub struct SpscQueue<T> {
    buffer: *mut T,
    mask: usize,
    head: AtomicUsize,  // Consumer reads/writes
    tail: AtomicUsize,  // Producer reads/writes
    _cache_pad: [u8; 64], // Prevent false sharing
}
```

### Implementation Complexity

- **Simple** - ~100 lines
- **Priority: High** - Essential for queues, buffers, sliding windows

---

## 4. BinaryHeap / Tournament Tree

### Algorithm: Tournament Tree with Level-Order Array

**Why NOT traditional binary heap:**
- `sift_up` / `sift_down` are O(log n) with variable depth = SIMD divergence
- Parent/child traversal has unpredictable memory access

**GPU-Optimal Solution: Tournament Tree**

A tournament tree is a complete binary tree where each internal node stores the "winner" (min or max) of its children. The key insight: we can pre-compute all internal nodes, making peek O(1) and parallel updates O(log n) but with ALL threads doing the same amount of work.

### The Structure

```
Tournament tree for 8 elements (indices in array):
                    [0]              <- winner of all (root)
                   /   \
                [1]     [2]          <- winners of pairs
               /   \   /   \
             [3]  [4] [5]  [6]       <- winners of leaves
            /  \ /  \ /  \ /  \
           [L0][L1][L2][L3][L4][L5][L6][L7]  <- leaves (actual data)
```

Stored in level-order array: `[root, level1..., level2..., leaves...]`

### Struct Layout

```rust
/// Tournament tree node
#[repr(C)]
struct TournamentNode<T> {
    value: T,
    source_leaf: u32,  // Which leaf this value came from
    _padding: u32,
}

/// GPU-native priority queue using tournament tree
pub struct BinaryHeap<T> {
    /// Level-order array: internal nodes followed by leaves
    /// For N leaves, there are N-1 internal nodes
    /// Total size: 2*N - 1
    tree: *mut TournamentNode<T>,

    /// Number of leaves (always power of 2)
    leaf_count: usize,

    /// Number of actual elements (may be < leaf_count)
    len: usize,

    /// Comparison function (true if a < b for min-heap)
    is_less: fn(&T, &T) -> bool,
}
```

### Key Operations

```rust
impl<T: Copy + Ord + Default> BinaryHeap<T> {
    /// O(1) - just read root
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 { return None; }
        Some(&self.internal_nodes()[0].value)
    }

    /// O(log n) - but all threads do exactly log(n) steps
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 { return None; }

        let root = self.internal_nodes()[0];
        let winner_value = root.value;
        let winner_leaf = root.source_leaf as usize;

        // Mark winner leaf as "infinity" (removed)
        self.leaves_mut()[winner_leaf].value = T::MAX;

        // Replay tournament from that leaf to root
        // This is O(log n) but DETERMINISTIC - no branching on data
        self.replay_from_leaf(winner_leaf);

        self.len -= 1;
        Some(winner_value)
    }

    /// O(log n) - replay tournament
    pub fn push(&mut self, value: T) {
        // Find an empty leaf slot
        let leaf_idx = self.find_empty_leaf();
        self.leaves_mut()[leaf_idx].value = value;
        self.leaves_mut()[leaf_idx].source_leaf = leaf_idx as u32;

        // Replay tournament from that leaf
        self.replay_from_leaf(leaf_idx);

        self.len += 1;
    }

    /// Deterministic O(log n) replay
    fn replay_from_leaf(&mut self, mut leaf_idx: usize) {
        let internal_start = self.leaf_count - 1;
        let mut node_idx = internal_start + leaf_idx;

        // Walk up the tree, updating winners
        while node_idx > 0 {
            let parent_idx = (node_idx - 1) / 2;
            let sibling_idx = if node_idx % 2 == 1 { node_idx + 1 } else { node_idx - 1 };

            // Compare and update parent (branchless)
            let left = self.tree_node(if node_idx < sibling_idx { node_idx } else { sibling_idx });
            let right = self.tree_node(if node_idx < sibling_idx { sibling_idx } else { node_idx });

            let winner = if (self.is_less)(&left.value, &right.value) { left } else { right };
            self.set_tree_node(parent_idx, winner);

            node_idx = parent_idx;
        }
    }
}
```

### GPU-Specific Considerations

- **Deterministic depth**: All operations traverse exactly log(n) levels
- **Parallel replay**: Multiple threads can replay different leaves simultaneously (for batch operations)
- **Memory locality**: Level-order array has good cache behavior
- **Power-of-2 size**: Required for simple index calculations

### Batch Operations (GPU-Parallel)

```rust
/// Pop k elements in parallel - O(log n) depth with k threads
pub fn pop_batch(&mut self, k: usize) -> Vec<T> {
    // All k winners are at leaves pointed to by top-k internal nodes
    // This is more complex but enables true parallelism
}

/// Push k elements in parallel
pub fn push_batch(&mut self, values: &[T]) {
    // Assign values to leaves in parallel
    // Replay tournament in parallel (each thread handles one path)
}
```

### Implementation Complexity

- **Medium** - ~200 lines
- **Priority: Medium** - Useful for scheduling, top-k queries

---

## 5. LinkedList / Pool-Based Freelist

### Algorithm: Index-Based Pool with Freelist

**Why NOT pointer-based linked list:**
- Pointer chasing = unpredictable memory access
- Allocation per node = fragmentation
- No spatial locality

**GPU-Optimal Solution: Array Pool with Index Links**

Store all nodes in a contiguous array. Use indices instead of pointers. Maintain a freelist for O(1) allocation.

### Struct Layout

```rust
/// A node in the pool-based linked list
#[repr(C)]
struct PoolNode<T> {
    value: T,
    next: u32,     // Index of next node (u32::MAX = end)
    prev: u32,     // Index of prev node (u32::MAX = end)
    _flags: u32,   // Bit 0: is_free
}

/// GPU-native doubly-linked list using pool allocation
pub struct LinkedList<T> {
    /// Contiguous array of all nodes
    pool: *mut PoolNode<T>,
    pool_capacity: usize,

    /// Head of the freelist (index, not pointer)
    free_head: u32,

    /// Head/tail of the actual list
    list_head: u32,  // u32::MAX if empty
    list_tail: u32,

    /// Number of active elements
    len: usize,
}

const NULL_INDEX: u32 = u32::MAX;
```

### Key Operations

```rust
impl<T: Copy + Default> LinkedList<T> {
    /// O(1) - pop from freelist
    fn alloc_node(&mut self) -> u32 {
        if self.free_head == NULL_INDEX {
            self.grow_pool();
        }
        let idx = self.free_head;
        let node = unsafe { &mut *self.pool.add(idx as usize) };
        self.free_head = node.next;
        node._flags &= !1; // Mark as not free
        idx
    }

    /// O(1) - push to freelist
    fn free_node(&mut self, idx: u32) {
        let node = unsafe { &mut *self.pool.add(idx as usize) };
        node.next = self.free_head;
        node._flags |= 1; // Mark as free
        self.free_head = idx;
    }

    /// O(1) push_front
    pub fn push_front(&mut self, value: T) {
        let idx = self.alloc_node();
        let node = unsafe { &mut *self.pool.add(idx as usize) };
        node.value = value;
        node.next = self.list_head;
        node.prev = NULL_INDEX;

        if self.list_head != NULL_INDEX {
            let old_head = unsafe { &mut *self.pool.add(self.list_head as usize) };
            old_head.prev = idx;
        } else {
            self.list_tail = idx;
        }
        self.list_head = idx;
        self.len += 1;
    }

    /// O(1) push_back
    pub fn push_back(&mut self, value: T) {
        let idx = self.alloc_node();
        let node = unsafe { &mut *self.pool.add(idx as usize) };
        node.value = value;
        node.next = NULL_INDEX;
        node.prev = self.list_tail;

        if self.list_tail != NULL_INDEX {
            let old_tail = unsafe { &mut *self.pool.add(self.list_tail as usize) };
            old_tail.next = idx;
        } else {
            self.list_head = idx;
        }
        self.list_tail = idx;
        self.len += 1;
    }

    /// O(1) pop_front
    pub fn pop_front(&mut self) -> Option<T> {
        if self.list_head == NULL_INDEX { return None; }

        let idx = self.list_head;
        let node = unsafe { &*self.pool.add(idx as usize) };
        let value = node.value;

        self.list_head = node.next;
        if self.list_head != NULL_INDEX {
            let new_head = unsafe { &mut *self.pool.add(self.list_head as usize) };
            new_head.prev = NULL_INDEX;
        } else {
            self.list_tail = NULL_INDEX;
        }

        self.free_node(idx);
        self.len -= 1;
        Some(value)
    }

    /// O(1) splice at known position
    pub fn splice_after(&mut self, pos: u32, value: T) {
        // Insert after node at index `pos`
        // O(1) because we have direct index access
    }
}
```

### Cursor API for Traversal

Since linked list traversal is inherently O(n), provide a cursor for explicit iteration:

```rust
pub struct Cursor<'a, T> {
    list: &'a LinkedList<T>,
    current: u32,
}

impl<'a, T: Copy> Iterator for Cursor<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == NULL_INDEX { return None; }
        let node = unsafe { &*self.list.pool.add(self.current as usize) };
        self.current = node.next;
        Some(&node.value)
    }
}
```

### GPU-Specific Considerations

- **Index-based**: No pointer chasing, indices are 32-bit
- **Pool allocation**: All nodes contiguous in memory
- **Freelist**: O(1) alloc/free with no system calls
- **Cache-friendly**: Sequential iteration hits cache well

### When to Use LinkedList

| Use Case | Recommendation |
|----------|----------------|
| FIFO queue | Use VecDeque instead |
| LIFO stack | Use Vec instead |
| Priority queue | Use BinaryHeap instead |
| O(1) splice/insert at known position | LinkedList is appropriate |
| LRU cache | LinkedList + HashMap for O(1) move-to-front |

### Implementation Complexity

- **Simple** - ~150 lines
- **Priority: Low** - Most use cases better served by other structures

---

## Common Patterns

### Cache-Line Alignment

All structures use 128-byte alignment for Apple Silicon cache lines:

```rust
#[repr(C, align(128))]
struct CacheAlignedBucket<T> {
    data: [T; ELEMENTS_PER_LINE],
    metadata: u64,
    _padding: [u8; PADDING_SIZE],
}
```

### Batch Operations

Every structure should support batch variants:

```rust
// Single operation (may be deferred)
fn insert(&mut self, key: K, value: V);

// Batch operation (immediately executed, GPU-parallel)
fn insert_batch(&mut self, entries: &[(K, V)]);
```

### No Lazy Iterators

GPU cannot do lazy iteration. All iteration returns materialized collections:

```rust
// WRONG for GPU
fn iter(&self) -> impl Iterator<Item = &T>;

// CORRECT for GPU
fn iter(&self) -> Vec<&T>;  // or Vec<T> if Copy
fn to_vec(&self) -> Vec<T>;
```

### Default Implementations

Use `Default` trait requirement to enable batch operations:

```rust
// Enables filling buffers with default values
where T: Copy + Default
```

### Size Requirements

Prefer `Copy` types to avoid drop glue complexity:

```rust
// Preferred bounds for GPU collections
where K: Hash + Eq + Copy + Default
where V: Copy + Default
```

---

## Implementation Priority

| Structure | Complexity | Priority | Reason |
|-----------|------------|----------|--------|
| HashSet | Trivial | **HIGH** | Common, wraps existing HashMap |
| VecDeque | Simple | **HIGH** | Essential for queues/buffers |
| BinaryHeap | Medium | **MEDIUM** | Scheduling, top-k |
| LinkedList | Simple | **LOW** | Rarely needed, other options better |
| BTreeMap | Medium | **LOW** | HashMap covers most cases |

---

## Summary

| Structure | Algorithm | Lookup | Insert | Delete | GPU Benefit |
|-----------|-----------|--------|--------|--------|-------------|
| HashSet | Cuckoo | O(1) | O(1)* | O(1) | No divergence |
| BTreeMap | Sorted+Index | O(1)** | O(n)*** | O(n)*** | Ordered iteration |
| VecDeque | Ring Buffer | O(1) | O(1) | O(1) | Zero memory moves |
| BinaryHeap | Tournament | O(1) | O(log n) | O(log n) | Deterministic depth |
| LinkedList | Pool+Freelist | O(n) | O(1)**** | O(1)**** | O(1) splice |

*Amortized due to eviction chains
**Constant for practical sizes (4 level traversal max)
***Batch rebuild, amortized O(1)
****At known position

All designs follow the core GPU principle: **trade memory for constant-time access, avoid variable-length operations that cause SIMD divergence.**
