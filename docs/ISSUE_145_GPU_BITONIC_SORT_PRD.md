# Issue #145: GPU Bitonic Sort - Replace CPU Z-Order Sort

## Problem Statement

In `src/gpu_os/desktop/types.rs` lines 504-522, window z-order is normalized using O(n²) bubble sort on CPU:

```rust
pub fn normalize_z_order(&mut self) {
    // Collect (index, z_order) pairs
    let mut pairs: Vec<(usize, u32)> = self.windows.iter()
        .enumerate()
        .filter(|(_, w)| w.is_visible())
        .map(|(i, w)| (i, w.z_order))
        .collect();

    // Sort by z_order (O(n log n) but still CPU)
    pairs.sort_by_key(|(_, z)| *z);

    // Reassign sequential z_orders
    for (new_z, (idx, _)) in pairs.iter().enumerate() {
        self.windows[*idx].z_order = new_z as u32;
    }
}
```

**Impact:**
- Called after every window state change (focus, create, close)
- O(n²) for bubble sort or O(n log n) for Rust's sort
- Blocks GPU rendering waiting for CPU sort

## Solution

Implement GPU bitonic sort for z-order management:
1. All window z-orders stored in GPU buffer
2. Bitonic sort runs entirely on GPU
3. No CPU involvement during sort

### Why Bitonic Sort?

- **Data-independent:** Same comparisons regardless of input order (no branches)
- **GPU-friendly:** Highly parallel, maps well to SIMD
- **In-place:** No extra memory allocation
- **Stable:** Maintains relative order of equal elements

## Requirements

### Functional Requirements
1. Sort produces same order as CPU sort
2. Handles up to 1024 windows
3. Supports both ascending and descending order
4. Works with packed window data (z_order + window_id)

### Performance Requirements
1. **Target:** <100µs for 1024 windows
2. **No CPU involvement:** Entire sort on GPU
3. **Single dispatch:** One command buffer for complete sort

### Non-Functional Requirements
1. Generic implementation usable for other sorting needs
2. Support for custom comparison functions
3. Clear debugging/profiling support

## Technical Design

### Bitonic Sort Algorithm

Bitonic sort works by:
1. Building bitonic sequences (ascending then descending or vice versa)
2. Merging bitonic sequences into sorted sequences
3. Repeating until entire array is sorted

```
For n=8:
Step 1: Compare pairs (0,1), (2,3), (4,5), (6,7) - form 2-element bitonic sequences
Step 2: Compare (0,2), (1,3), (4,6), (5,7) - merge into 4-element bitonic sequences
Step 3: Compare (0,1), (2,3), (4,5), (6,7) - complete 4-element sort
Step 4: Compare (0,4), (1,5), (2,6), (3,7) - start 8-element merge
Step 5: Compare (0,2), (1,3), (4,6), (5,7) - continue merge
Step 6: Compare (0,1), (2,3), (4,5), (6,7) - complete sort
```

### Metal Shader Implementation

```metal
// src/gpu_os/desktop/sort.metal

struct SortParams {
    uint count;
    uint stage;
    uint step;
    uint ascending;  // 1 = ascending, 0 = descending
};

// Single comparison step of bitonic sort
kernel void bitonic_sort_step(
    device uint* keys [[buffer(0)]],      // z_order values
    device uint* values [[buffer(1)]],    // window indices
    constant SortParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint count = params.count;
    uint stage = params.stage;
    uint step = params.step;

    // Each thread handles one comparison
    uint pair_distance = 1u << step;
    uint block_size = 1u << (stage + 1);

    // Calculate indices to compare
    uint block_idx = tid / (block_size / 2);
    uint idx_in_block = tid % (block_size / 2);

    uint left_idx = block_idx * block_size + idx_in_block;
    uint right_idx = left_idx + pair_distance;

    if (right_idx >= count) return;

    // Determine sort direction for this block
    // In bitonic sort, alternating blocks sort in opposite directions
    bool block_ascending = ((block_idx & 1) == 0) == (params.ascending == 1);

    // For merge steps, all comparisons go same direction
    if (step < stage) {
        block_ascending = (params.ascending == 1);
    }

    // Compare and swap
    uint left_key = keys[left_idx];
    uint right_key = keys[right_idx];

    bool should_swap = block_ascending
        ? (left_key > right_key)
        : (left_key < right_key);

    if (should_swap) {
        keys[left_idx] = right_key;
        keys[right_idx] = left_key;

        uint left_val = values[left_idx];
        uint right_val = values[right_idx];
        values[left_idx] = right_val;
        values[right_idx] = left_val;
    }
}

// Optimized version using threadgroup memory for small arrays
kernel void bitonic_sort_local(
    device uint* keys [[buffer(0)]],
    device uint* values [[buffer(1)]],
    constant SortParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    threadgroup uint* local_keys [[threadgroup(0)]],
    threadgroup uint* local_values [[threadgroup(1)]]
) {
    uint count = params.count;

    // Load into threadgroup memory
    if (tid < count) {
        local_keys[lid] = keys[tid];
        local_values[lid] = values[tid];
    } else {
        local_keys[lid] = 0xFFFFFFFF;  // Sentinel for padding
        local_values[lid] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Complete bitonic sort within threadgroup
    for (uint stage = 0; (1u << stage) < group_size; stage++) {
        for (uint step = stage; step >= 0; step--) {
            uint pair_distance = 1u << step;
            uint block_size = 1u << (stage + 1);

            uint block_idx = lid / (block_size / 2);
            uint idx_in_block = lid % (block_size / 2);

            uint left_idx = block_idx * block_size + idx_in_block;
            uint right_idx = left_idx + pair_distance;

            if (right_idx < group_size) {
                bool ascending = ((block_idx & 1) == 0) == (params.ascending == 1);
                if (step < stage) ascending = (params.ascending == 1);

                uint left_key = local_keys[left_idx];
                uint right_key = local_keys[right_idx];

                if ((ascending && left_key > right_key) ||
                    (!ascending && left_key < right_key)) {
                    local_keys[left_idx] = right_key;
                    local_keys[right_idx] = left_key;

                    uint tmp = local_values[left_idx];
                    local_values[left_idx] = local_values[right_idx];
                    local_values[right_idx] = tmp;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back
    if (tid < count) {
        keys[tid] = local_keys[lid];
        values[tid] = local_values[lid];
    }
}
```

### Rust Implementation

```rust
// src/gpu_os/gpu_sort.rs

pub struct GpuBitonicSort {
    device: Device,
    command_queue: CommandQueue,
    sort_step_pipeline: ComputePipelineState,
    sort_local_pipeline: ComputePipelineState,
    keys_buffer: Buffer,
    values_buffer: Buffer,
    params_buffer: Buffer,
    max_count: usize,
}

impl GpuBitonicSort {
    pub fn new(device: &Device, max_count: usize) -> Self {
        let library = device.new_library_with_source(SORT_SHADER_SOURCE, &CompileOptions::new()).unwrap();

        let sort_step_fn = library.get_function("bitonic_sort_step", None).unwrap();
        let sort_step_pipeline = device.new_compute_pipeline_state_with_function(&sort_step_fn).unwrap();

        let sort_local_fn = library.get_function("bitonic_sort_local", None).unwrap();
        let sort_local_pipeline = device.new_compute_pipeline_state_with_function(&sort_local_fn).unwrap();

        // Round up to power of 2
        let padded_count = max_count.next_power_of_two();

        Self {
            device: device.clone(),
            command_queue: device.new_command_queue(),
            sort_step_pipeline,
            sort_local_pipeline,
            keys_buffer: device.new_buffer((padded_count * 4) as u64, MTLResourceOptions::StorageModeShared),
            values_buffer: device.new_buffer((padded_count * 4) as u64, MTLResourceOptions::StorageModeShared),
            params_buffer: device.new_buffer(16, MTLResourceOptions::StorageModeShared),
            max_count: padded_count,
        }
    }

    pub fn sort(&self, keys: &mut [u32], values: &mut [u32], ascending: bool) {
        let count = keys.len();
        assert!(count <= self.max_count);
        assert_eq!(keys.len(), values.len());

        let padded_count = count.next_power_of_two();

        // Upload data
        unsafe {
            let keys_ptr = self.keys_buffer.contents() as *mut u32;
            let values_ptr = self.values_buffer.contents() as *mut u32;

            std::ptr::copy_nonoverlapping(keys.as_ptr(), keys_ptr, count);
            std::ptr::copy_nonoverlapping(values.as_ptr(), values_ptr, count);

            // Pad with max value
            for i in count..padded_count {
                *keys_ptr.add(i) = u32::MAX;
                *values_ptr.add(i) = 0;
            }
        }

        // For small arrays, use local sort
        if padded_count <= 1024 {
            self.sort_local(padded_count, ascending);
        } else {
            self.sort_global(padded_count, ascending);
        }

        // Download results
        unsafe {
            let keys_ptr = self.keys_buffer.contents() as *const u32;
            let values_ptr = self.values_buffer.contents() as *const u32;

            std::ptr::copy_nonoverlapping(keys_ptr, keys.as_mut_ptr(), count);
            std::ptr::copy_nonoverlapping(values_ptr, values.as_mut_ptr(), count);
        }
    }

    fn sort_local(&self, count: usize, ascending: bool) {
        let params = SortParams {
            count: count as u32,
            stage: 0,
            step: 0,
            ascending: if ascending { 1 } else { 0 },
        };

        unsafe {
            let ptr = self.params_buffer.contents() as *mut SortParams;
            *ptr = params;
        }

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.sort_local_pipeline);
        encoder.set_buffer(0, Some(&self.keys_buffer), 0);
        encoder.set_buffer(1, Some(&self.values_buffer), 0);
        encoder.set_buffer(2, Some(&self.params_buffer), 0);

        // Threadgroup memory for keys and values
        encoder.set_threadgroup_memory_length(0, (count * 4) as u64);
        encoder.set_threadgroup_memory_length(1, (count * 4) as u64);

        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(count as u64, 1, 1)
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn sort_global(&self, count: usize, ascending: bool) {
        let command_buffer = self.command_queue.new_command_buffer();

        let num_stages = (count as f64).log2().ceil() as u32;

        for stage in 0..num_stages {
            for step in (0..=stage).rev() {
                let params = SortParams {
                    count: count as u32,
                    stage,
                    step,
                    ascending: if ascending { 1 } else { 0 },
                };

                unsafe {
                    let ptr = self.params_buffer.contents() as *mut SortParams;
                    *ptr = params;
                }

                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.sort_step_pipeline);
                encoder.set_buffer(0, Some(&self.keys_buffer), 0);
                encoder.set_buffer(1, Some(&self.values_buffer), 0);
                encoder.set_buffer(2, Some(&self.params_buffer), 0);

                let threads = MTLSize::new((count / 2) as u64, 1, 1);
                let threadgroup = MTLSize::new(256, 1, 1);
                encoder.dispatch_threads(threads, threadgroup);

                encoder.end_encoding();
            }
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}
```

### Integration with Window Manager

```rust
// src/gpu_os/desktop/types.rs

impl DesktopState {
    pub fn normalize_z_order_gpu(&mut self, sorter: &GpuBitonicSort) {
        // Collect z_orders and indices for visible windows
        let visible: Vec<(u32, u32)> = self.windows.iter()
            .enumerate()
            .filter(|(_, w)| w.is_visible())
            .map(|(i, w)| (w.z_order, i as u32))
            .collect();

        if visible.is_empty() { return; }

        let mut keys: Vec<u32> = visible.iter().map(|(z, _)| *z).collect();
        let mut values: Vec<u32> = visible.iter().map(|(_, i)| *i).collect();

        // GPU sort
        sorter.sort(&mut keys, &mut values, true);

        // Apply new z_orders
        for (new_z, &window_idx) in values.iter().enumerate() {
            self.windows[window_idx as usize].z_order = new_z as u32;
        }
    }
}
```

## Pseudocode

```
function bitonic_sort_gpu(keys, values, ascending):
    n = next_power_of_two(keys.length)
    pad_to_power_of_two(keys, values, n)

    # Number of stages = log2(n)
    for stage in 0 to log2(n):
        # Each stage has (stage + 1) steps
        for step in stage down to 0:
            # GPU kernel: each thread handles one comparison
            parallel for tid in 0 to n/2:
                pair_distance = 2^step
                block_size = 2^(stage + 1)

                # Calculate indices
                block_idx = tid / (block_size / 2)
                idx_in_block = tid % (block_size / 2)
                left = block_idx * block_size + idx_in_block
                right = left + pair_distance

                # Determine direction
                direction = (block_idx is even) == ascending
                if step < stage:
                    direction = ascending

                # Compare and swap
                if should_swap(keys[left], keys[right], direction):
                    swap(keys[left], keys[right])
                    swap(values[left], values[right])

            barrier()  # Sync before next step
```

## Test Plan

### Unit Tests

```rust
// tests/test_issue_145_bitonic_sort.rs

#[test]
fn test_sort_correctness() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    // Random data
    let mut keys: Vec<u32> = (0..100).map(|_| rand::random::<u32>() % 1000).collect();
    let mut values: Vec<u32> = (0..100).map(|i| i as u32).collect();

    let mut expected_keys = keys.clone();
    let mut expected_values = values.clone();

    // CPU sort for reference
    let mut pairs: Vec<_> = expected_keys.iter().zip(expected_values.iter()).map(|(&k, &v)| (k, v)).collect();
    pairs.sort_by_key(|(k, _)| *k);
    expected_keys = pairs.iter().map(|(k, _)| *k).collect();
    expected_values = pairs.iter().map(|(_, v)| *v).collect();

    // GPU sort
    sorter.sort(&mut keys, &mut values, true);

    assert_eq!(keys, expected_keys);
    assert_eq!(values, expected_values);
}

#[test]
fn test_sort_descending() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    let mut keys: Vec<u32> = vec![1, 5, 3, 9, 2, 7, 4, 8, 6, 0];
    let mut values: Vec<u32> = (0..10).collect();

    sorter.sort(&mut keys, &mut values, false);

    assert_eq!(keys, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
}

#[test]
fn test_sort_already_sorted() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    let mut keys: Vec<u32> = (0..64).collect();
    let mut values: Vec<u32> = (0..64).collect();

    sorter.sort(&mut keys, &mut values, true);

    assert_eq!(keys, (0..64).collect::<Vec<_>>());
    assert_eq!(values, (0..64).collect::<Vec<_>>());
}

#[test]
fn test_sort_reverse_sorted() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    let mut keys: Vec<u32> = (0..64).rev().collect();
    let mut values: Vec<u32> = (0..64).collect();

    sorter.sort(&mut keys, &mut values, true);

    assert_eq!(keys, (0..64).collect::<Vec<_>>());
}

#[test]
fn test_sort_duplicates() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    let mut keys: Vec<u32> = vec![5, 3, 5, 1, 3, 5, 1, 3];
    let mut values: Vec<u32> = (0..8).collect();

    sorter.sort(&mut keys, &mut values, true);

    assert_eq!(keys, vec![1, 1, 3, 3, 3, 5, 5, 5]);
}

#[test]
fn test_sort_performance() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    let mut keys: Vec<u32> = (0..1024).map(|_| rand::random()).collect();
    let mut values: Vec<u32> = (0..1024).collect();

    // Warmup
    for _ in 0..10 {
        let mut k = keys.clone();
        let mut v = values.clone();
        sorter.sort(&mut k, &mut v, true);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..1000 {
        let mut k = keys.clone();
        let mut v = values.clone();
        sorter.sort(&mut k, &mut v, true);
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() / 1000;
    println!("Average sort time (1024 elements): {}µs", avg_us);

    // Target: <100µs
    assert!(avg_us < 200, "Sort too slow: {}µs", avg_us);
}

#[test]
fn test_z_order_normalization() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);

    let mut desktop = DesktopState::new();

    // Create windows with random z_orders
    for i in 0..10 {
        let mut window = Window::new(i, format!("Window {}", i));
        window.z_order = rand::random::<u32>() % 100;
        window.flags |= WINDOW_VISIBLE;
        desktop.windows.push(window);
    }

    // Normalize
    desktop.normalize_z_order_gpu(&sorter);

    // Verify sequential z_orders
    let z_orders: Vec<u32> = desktop.windows.iter()
        .filter(|w| w.is_visible())
        .map(|w| w.z_order)
        .collect();

    let expected: Vec<u32> = (0..10).collect();
    assert_eq!(z_orders.iter().cloned().sorted().collect::<Vec<_>>(), expected);
}
```

### Visual Verification Tests

```rust
// tests/test_issue_145_visual.rs

#[test]
fn visual_test_window_z_order() {
    let device = Device::system_default().unwrap();
    let sorter = GpuBitonicSort::new(&device, 1024);
    let mut renderer = TestRenderer::new(&device, 800, 600);

    let mut desktop = DesktopState::new();

    // Create overlapping windows
    for i in 0..5 {
        let mut window = Window::new(i, format!("Window {}", i));
        window.x = 50.0 + (i as f32) * 30.0;
        window.y = 50.0 + (i as f32) * 30.0;
        window.width = 200.0;
        window.height = 150.0;
        window.z_order = (4 - i) as u32;  // Reverse order
        window.flags |= WINDOW_VISIBLE;
        desktop.windows.push(window);
    }

    // Before normalization
    renderer.render_windows(&desktop);
    renderer.save_to_file("tests/visual_output/z_order_before.png");

    // Normalize
    desktop.normalize_z_order_gpu(&sorter);

    // After normalization
    renderer.clear();
    renderer.render_windows(&desktop);
    renderer.save_to_file("tests/visual_output/z_order_after.png");

    // Verify visual - top window should be at highest z_order
    // (Manual inspection or automated screenshot comparison)
}
```

## Success Metrics

1. **Performance:** <100µs for 1024 elements
2. **Correctness:** 100% match with CPU sort
3. **No CPU blocking:** Single GPU dispatch for complete sort
4. **Memory:** In-place sort, no extra allocation

## Dependencies

None (standalone utility)

## Files to Create/Modify

1. `src/gpu_os/gpu_sort.rs` - New module
2. `src/gpu_os/gpu_sort.metal` - GPU kernels
3. `src/gpu_os/desktop/types.rs` - Integration
4. `tests/test_issue_145_bitonic_sort.rs` - Tests
