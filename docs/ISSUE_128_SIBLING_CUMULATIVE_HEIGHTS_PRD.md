# Issue #128: O(1) Sibling Cumulative Heights for Layout

## Problem Statement

The current `layout_position_siblings` kernel walks the entire sibling chain for each element to calculate Y positions. This is O(siblings) per element, causing severe SIMD divergence on GPU.

**Current Code (O(siblings)):**
```metal
// layout.metal lines 698-729
int sib = elements[parent].first_child;
while (sib >= 0 && sib != int(gid)) {
    float sib_height = boxes[sib].height;
    y += sib_height;
    sib = elements[sib].next_sibling;  // O(siblings) per element!
}
```

**Impact:** With 100 siblings, each element does 50 iterations on average. All 32 threads in a SIMD group execute ALL iterations (GPU executes both branches), making this effectively O(siblings * 32).

## Solution: Pre-computed Cumulative Height Buffer

Pre-compute cumulative heights in a level-parallel pass, then use O(1) lookup.

### Data Structure

```rust
/// Cumulative height for each element (sum of all preceding siblings' heights)
/// cumulative_height[i] = sum(boxes[j].height for j in preceding_siblings(i))
pub struct CumulativeHeightBuffer {
    buffer: Buffer,  // device [f32; element_count]
}
```

### Algorithm

**Phase 1: Compute per-element heights (already done in existing layout)**

**Phase 2: Parallel prefix sum per sibling chain**

```metal
// New kernel: compute_cumulative_heights
// Runs AFTER heights are computed, BEFORE positioning
kernel void compute_cumulative_heights(
    device const Element* elements [[buffer(0)]],
    device const LayoutBox* boxes [[buffer(1)]],
    device float* cumulative_heights [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    constant uint& current_level [[buffer(4)]],
    constant uint& element_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    Element elem = elements[gid];
    int parent = elem.parent;

    if (parent < 0) {
        cumulative_heights[gid] = 0;
        return;
    }

    // Find my sibling index using SIMD ballot
    // Thread with first_child == gid has index 0
    int first = elements[parent].first_child;

    if (gid == first) {
        cumulative_heights[gid] = 0;  // First child has no preceding siblings
    } else {
        // Use prev_sibling to get predecessor's cumulative + height
        int prev = elem.prev_sibling;
        if (prev >= 0) {
            cumulative_heights[gid] = cumulative_heights[prev] + boxes[prev].height;
        } else {
            cumulative_heights[gid] = 0;
        }
    }
}
```

**Phase 3: O(1) lookup in positioning kernel**

```metal
// Modified layout_position_siblings - now O(1)
kernel void layout_position_siblings_fast(
    device const Element* elements [[buffer(0)]],
    device const ComputedStyle* styles [[buffer(1)]],
    device LayoutBox* boxes [[buffer(2)]],
    device const uint* depths [[buffer(3)]],
    device const float* cumulative_heights [[buffer(4)]],  // NEW
    constant uint& current_level [[buffer(5)]],
    constant uint& element_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) return;
    if (depths[gid] != current_level) return;

    // O(1) lookup replaces O(siblings) loop!
    float y = cumulative_heights[gid] + styles[gid].margin[0];

    boxes[gid].y = y;
    boxes[gid].x = styles[gid].margin[3];
}
```

### Handling Margin Collapsing

The current code handles CSS margin collapsing. We need to incorporate this into the cumulative buffer:

```metal
// Extended: cumulative buffer stores collapsed margin info
struct CumulativeInfo {
    float y_offset;           // Cumulative height
    float prev_margin_bottom; // For margin collapsing
};
```

## Rust-Side Changes

```rust
// In layout.rs
impl GpuLayout {
    fn compute_cumulative_heights(&self, encoder: &ComputeCommandEncoderRef) {
        // Dispatch compute_cumulative_heights kernel per level
        for level in 0..self.max_depth {
            encoder.set_compute_pipeline_state(&self.cumulative_pipeline);
            encoder.set_buffer(0, Some(&self.elements_buffer), 0);
            encoder.set_buffer(1, Some(&self.boxes_buffer), 0);
            encoder.set_buffer(2, Some(&self.cumulative_heights_buffer), 0);
            encoder.set_buffer(3, Some(&self.depths_buffer), 0);
            encoder.set_bytes(4, &(level as u32));
            encoder.set_bytes(5, &(self.element_count as u32));

            let threadgroups = (self.element_count as u64 + 255) / 256;
            encoder.dispatch_threadgroups(
                MTLSize::new(threadgroups, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
    }
}
```

## Benchmarks

### Test Cases

1. **Wide tree (many siblings):** 10 levels, 100 siblings per node = 10^10 relationships
2. **Deep tree (few siblings):** 100 levels, 2 siblings per node
3. **Mixed tree:** Realistic HTML document structure

### Expected Performance

| Tree Type | Current (O(siblings)) | New (O(1)) | Speedup |
|-----------|----------------------|------------|---------|
| 100 siblings/node | ~500μs | ~20μs | 25x |
| 10 siblings/node | ~50μs | ~20μs | 2.5x |
| 2 siblings/node | ~25μs | ~20μs | 1.25x |

### Benchmark Code

```rust
#[test]
fn benchmark_sibling_positioning() {
    let device = Device::system_default().unwrap();

    // Create wide tree: 1000 elements, 100 siblings per parent
    let elements = create_wide_tree(1000, 100);

    // Benchmark current O(siblings) approach
    let old_time = benchmark_old_positioning(&device, &elements);

    // Benchmark new O(1) approach
    let new_time = benchmark_new_positioning(&device, &elements);

    println!("Old: {:.2}μs, New: {:.2}μs, Speedup: {:.1}x",
        old_time * 1e6, new_time * 1e6, old_time / new_time);

    assert!(new_time < old_time, "New approach should be faster");
}
```

## Success Criteria

1. **Correctness:** Layout output identical to current implementation
2. **Performance:** ≥10x speedup for trees with ≥50 siblings per node
3. **Memory:** Additional buffer size = element_count * 4 bytes (acceptable)

## Implementation Steps

1. Add `cumulative_heights_buffer` to `GpuLayout` struct
2. Create `compute_cumulative_heights` Metal kernel
3. Modify `layout_position_siblings` to use O(1) lookup
4. Update layout dispatch sequence
5. Add tests and benchmarks
6. Verify visual correctness with document viewer
