# Issue #130: O(1) Depth Buffer for Layout

## Problem Statement

The current `layout_compute_depths` kernel walks up the parent chain for every element to compute its depth. This is O(depth) per element.

**Current Code (O(depth)):**
```metal
// layout.metal layout_compute_depths
while (parent >= 0 && d < 256) {
    d++;
    parent = elements[parent].parent;  // O(depth) per element!
}
```

**Impact:** For a 100-level deep tree with 10K elements, this performs 10K × 50 = 500K parent lookups on average.

## Solution: Level-Parallel Depth Computation

Compute depths in a single top-down pass where each level depends only on the previous level (already computed).

### Key Insight

If we process level-by-level from root:
- Level 0 (root): depth = 0
- Level 1 (children of root): depth = parent.depth + 1 = 1
- Level N: depth = parent.depth + 1

Each level's depth can be computed in O(1) because parent's depth is already known!

### Algorithm

**Phase 1: Identify root nodes (depth 0)**
```metal
kernel void init_depths(
    device const Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    if (elements[gid].parent < 0) {
        depths[gid] = 0;  // Root node
    } else {
        depths[gid] = 0xFFFFFFFF;  // Not yet computed
    }
}
```

**Phase 2: Level-parallel propagation**
```metal
// Process one level at a time
kernel void propagate_depths(
    device const Element* elements [[buffer(0)]],
    device uint* depths [[buffer(1)]],
    constant uint& current_level [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    device atomic_uint* changed [[buffer(4)]],  // Did any element change?
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != 0xFFFFFFFF) return;  // Already computed

    int parent = elements[gid].parent;
    if (parent < 0) return;

    uint parent_depth = depths[parent];
    if (parent_depth == current_level) {
        // Parent is at current level, so we're at current_level + 1
        depths[gid] = current_level + 1;
        atomic_fetch_add_explicit(changed, 1, memory_order_relaxed);
    }
}
```

**Rust dispatch loop:**
```rust
fn compute_depths(&self, encoder: &ComputeCommandEncoderRef) {
    // Phase 1: Init roots
    self.dispatch_init_depths(encoder);

    // Phase 2: Propagate level by level
    let mut level = 0u32;
    loop {
        // Reset changed counter
        self.reset_changed_counter();

        // Dispatch propagation for this level
        encoder.set_compute_pipeline_state(&self.propagate_depths_pipeline);
        encoder.set_buffer(0, Some(&self.elements_buffer), 0);
        encoder.set_buffer(1, Some(&self.depths_buffer), 0);
        encoder.set_bytes(2, &level);
        encoder.set_bytes(3, &self.element_count);
        encoder.set_buffer(4, Some(&self.changed_buffer), 0);

        let threadgroups = (self.element_count as u64 + 255) / 256;
        encoder.dispatch_threadgroups(
            MTLSize::new(threadgroups, 1, 1),
            MTLSize::new(256, 1, 1),
        );

        // Check if any elements were updated
        encoder.memory_barrier_with_scope(MTLBarrierScope::Buffers);

        let changed = self.read_changed_counter();
        if changed == 0 {
            break;  // All depths computed
        }

        level += 1;
        if level > 256 {
            panic!("Tree depth exceeds maximum");
        }
    }

    self.max_depth = level;
}
```

### Alternative: Single-Pass with Atomics (Simpler but potentially slower)

```metal
// Alternative: Use atomics to handle dependencies
kernel void compute_depths_atomic(
    device const Element* elements [[buffer(0)]],
    device atomic_uint* depths [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;

    // Walk up to root, computing depth
    uint d = 0;
    int current = gid;

    while (current >= 0) {
        int parent = elements[current].parent;
        if (parent < 0) {
            // Reached root - propagate depth back down
            break;
        }
        current = parent;
        d++;
    }

    // Store depth (atomic to handle races)
    atomic_store_explicit(&depths[gid], d, memory_order_relaxed);
}
```

### Optimization: Persistent Depth Buffer

Once computed, depths only change when tree structure changes. Cache the depth buffer:

```rust
pub struct GpuLayout {
    depths_buffer: Buffer,
    depths_valid: bool,  // Invalidate on tree structure change

    fn invalidate_depths(&mut self) {
        self.depths_valid = false;
    }

    fn ensure_depths(&mut self, encoder: &ComputeCommandEncoderRef) {
        if !self.depths_valid {
            self.compute_depths(encoder);
            self.depths_valid = true;
        }
    }
}
```

## Usage in Other Kernels

The depth buffer enables O(1) depth access everywhere:

```metal
// Before: O(depth) per access
uint depth = 0;
int p = elements[gid].parent;
while (p >= 0) { depth++; p = elements[p].parent; }

// After: O(1)
uint depth = depths[gid];
```

## Benchmarks

### Test Cases

1. **Deep tree:** 1000 levels, 10 elements per level
2. **Wide tree:** 10 levels, 1000 elements per level
3. **Realistic DOM:** Mixed structure, ~1000 elements

### Expected Performance

| Tree Structure | Current O(depth) | New O(1) lookup | Speedup |
|---------------|------------------|-----------------|---------|
| 1000 deep | ~5ms compute + ~500μs/lookup | ~10ms compute + ~1μs/lookup | 500x per lookup |
| 10 deep | ~50μs compute + ~5μs/lookup | ~100μs compute + ~1μs/lookup | 5x per lookup |

### Benchmark Code

```rust
#[test]
fn benchmark_depth_computation() {
    let device = Device::system_default().unwrap();

    // Create deep tree
    let elements = create_deep_tree(1000, 10);  // 1000 levels, 10 per level

    // Benchmark depth computation
    let compute_time = benchmark_depth_compute(&device, &elements);

    // Benchmark depth lookups (1000 random elements)
    let old_lookup_time = benchmark_walk_to_root(&device, &elements, 1000);
    let new_lookup_time = benchmark_buffer_lookup(&device, &elements, 1000);

    println!("Compute: {:.2}ms, Walk lookup: {:.2}μs/op, Buffer lookup: {:.2}μs/op",
        compute_time * 1e3, old_lookup_time * 1e6, new_lookup_time * 1e6);
}
```

## Memory Overhead

- Depth buffer: 4 bytes per element
- 10K elements = 40 KB
- Negligible compared to element data

## Success Criteria

1. **Correctness:** Depths match parent-chain computation
2. **Performance:** Depth lookup is O(1) - constant time regardless of tree depth
3. **Memory:** 4 bytes per element overhead
4. **Cache validity:** Properly invalidate when tree changes

## Implementation Steps

1. Add `depths_buffer` and `depths_valid` to `GpuLayout`
2. Create `init_depths` and `propagate_depths` Metal kernels
3. Implement level-parallel depth computation in Rust
4. Add depth buffer access to existing layout kernels
5. Add cache invalidation on tree structure changes
6. Add tests and benchmarks
