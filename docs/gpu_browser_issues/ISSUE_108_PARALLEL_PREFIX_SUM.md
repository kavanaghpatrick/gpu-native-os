# Issue #108: GPU Parallel Prefix Sum

## Summary
Replace single-threaded prefix sum in paint pipeline with parallel GPU implementation, eliminating a major bottleneck discovered in both our code and Servo/WebRender.

## Motivation
Current `compute_offsets` kernel in `paint.metal` uses only thread 0:

```metal
kernel void compute_offsets(...) {
    if (gid != 0) return;  // Only thread 0 works!
    uint offset = 0;
    for (uint i = 0; i < element_count; i++) {
        vertex_offsets[i] = offset;
        offset += vertex_counts[i];
    }
}
```

This serializes vertex allocation, limiting throughput. For 10K elements, this loop alone can take 1-2ms on GPU, negating parallelism gains.

## Algorithm: Blelloch Scan

The Blelloch algorithm achieves O(n) work in O(log n) steps:

```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]

Up-sweep (reduce):
Step 1: [3, 4, 7, 7, 4, 5, 6, 9]
Step 2: [3, 4, 7, 11, 4, 5, 6, 14]
Step 3: [3, 4, 7, 11, 4, 5, 6, 25]

Down-sweep (distribute):
Step 1: [3, 4, 7, 11, 4, 5, 6, 0]
Step 2: [3, 4, 7, 0, 4, 5, 6, 11]
Step 3: [3, 0, 7, 4, 4, 11, 6, 16]
Step 4: [0, 3, 4, 11, 11, 15, 16, 22]

Output: [0, 3, 4, 11, 11, 15, 16, 22]
        (exclusive prefix sum)
```

## Metal Implementation

```metal
// src/gpu_os/document/prefix_sum.metal

#include <metal_stdlib>
using namespace metal;

// Threadgroup size (must be power of 2)
constant uint BLOCK_SIZE = 256;

//=============================================================================
// Single-block prefix sum (for <= 256 elements)
//=============================================================================

kernel void prefix_sum_single_block(
    device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup uint temp[BLOCK_SIZE * 2];

    // Load input into shared memory
    uint ai = lid;
    uint bi = lid + BLOCK_SIZE;

    uint offset = 1;

    // Load data
    temp[ai] = (ai < n) ? input[ai] : 0;
    temp[bi] = (bi < n) ? input[bi] : 0;

    // Up-sweep (reduce) phase
    for (uint d = BLOCK_SIZE; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }

    // Clear last element (for exclusive scan)
    if (lid == 0) {
        temp[BLOCK_SIZE * 2 - 1] = 0;
    }

    // Down-sweep phase
    for (uint d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            uint t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output
    if (ai < n) output[ai] = temp[ai];
    if (bi < n) output[bi] = temp[bi];
}

//=============================================================================
// Multi-block prefix sum (for > 256 elements)
//=============================================================================

// Phase 1: Per-block scan + save block totals
kernel void prefix_sum_phase1(
    device uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device uint* block_sums [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup uint temp[BLOCK_SIZE * 2];

    uint block_offset = group_id * BLOCK_SIZE * 2;
    uint ai = lid;
    uint bi = lid + BLOCK_SIZE;

    // Load data
    temp[ai] = (block_offset + ai < n) ? input[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < n) ? input[block_offset + bi] : 0;

    uint offset = 1;

    // Up-sweep
    for (uint d = BLOCK_SIZE; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset *= 2;
    }

    // Save block sum and clear for exclusive scan
    if (lid == 0) {
        block_sums[group_id] = temp[BLOCK_SIZE * 2 - 1];
        temp[BLOCK_SIZE * 2 - 1] = 0;
    }

    // Down-sweep
    for (uint d = 1; d < BLOCK_SIZE * 2; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai_idx = offset * (2 * lid + 1) - 1;
            uint bi_idx = offset * (2 * lid + 2) - 1;
            uint t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output
    if (block_offset + ai < n) output[block_offset + ai] = temp[ai];
    if (block_offset + bi < n) output[block_offset + bi] = temp[bi];
}

// Phase 2: Scan block sums (recursive or single-block)
// Uses prefix_sum_single_block if block count <= 256
// Otherwise recursive call to prefix_sum_phase1

// Phase 3: Add block sums to elements
kernel void prefix_sum_phase3(
    device uint* data [[buffer(0)]],
    device uint* block_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    if (gid >= n) return;
    if (group_id == 0) return;  // First block doesn't need adjustment

    data[gid] += block_sums[group_id];
}

//=============================================================================
// Simple atomic-based prefix sum (alternative, simpler but slower)
//=============================================================================

kernel void prefix_sum_atomic(
    device uint* counts [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    device atomic_uint* running_total [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint my_count = counts[gid];
    uint my_offset = atomic_fetch_add_explicit(running_total, my_count, memory_order_relaxed);
    offsets[gid] = my_offset;
}

//=============================================================================
// Segmented prefix sum (for variable-length records)
//=============================================================================

kernel void segmented_prefix_sum(
    device uint* values [[buffer(0)]],
    device uint* segment_flags [[buffer(1)]],  // 1 = start of segment
    device uint* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint temp_val[BLOCK_SIZE * 2];
    threadgroup uint temp_flag[BLOCK_SIZE * 2];

    // ... similar to above but resets at segment boundaries
}
```

## Rust Integration

```rust
// src/gpu_os/document/prefix_sum.rs

pub struct GPUPrefixSum {
    device: Device,
    single_block_pipeline: ComputePipelineState,
    phase1_pipeline: ComputePipelineState,
    phase3_pipeline: ComputePipelineState,
    block_sums_buffer: Buffer,
}

impl GPUPrefixSum {
    const BLOCK_SIZE: usize = 256;

    pub fn new(device: &Device, library: &Library, max_elements: usize) -> Self {
        let single_block_fn = library.get_function("prefix_sum_single_block", None).unwrap();
        let phase1_fn = library.get_function("prefix_sum_phase1", None).unwrap();
        let phase3_fn = library.get_function("prefix_sum_phase3", None).unwrap();

        let block_count = (max_elements + Self::BLOCK_SIZE * 2 - 1) / (Self::BLOCK_SIZE * 2);
        let block_sums_buffer = device.new_buffer(
            (block_count * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            device: device.clone(),
            single_block_pipeline: device.new_compute_pipeline_state_with_function(&single_block_fn).unwrap(),
            phase1_pipeline: device.new_compute_pipeline_state_with_function(&phase1_fn).unwrap(),
            phase3_pipeline: device.new_compute_pipeline_state_with_function(&phase3_fn).unwrap(),
            block_sums_buffer,
        }
    }

    pub fn scan(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        if count <= Self::BLOCK_SIZE as u32 * 2 {
            // Single block
            encoder.set_compute_pipeline_state(&self.single_block_pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_bytes(2, 4, &count as *const _ as *const _);

            encoder.dispatch_threadgroups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(Self::BLOCK_SIZE as u64, 1, 1),
            );
        } else {
            // Multi-block
            let block_count = (count as usize + Self::BLOCK_SIZE * 2 - 1) / (Self::BLOCK_SIZE * 2);

            // Phase 1: Per-block scan
            encoder.set_compute_pipeline_state(&self.phase1_pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_buffer(2, Some(&self.block_sums_buffer), 0);
            encoder.set_bytes(3, 4, &count as *const _ as *const _);

            encoder.dispatch_threadgroups(
                MTLSize::new(block_count as u64, 1, 1),
                MTLSize::new(Self::BLOCK_SIZE as u64, 1, 1),
            );

            encoder.memory_barrier_with_resources(&[&self.block_sums_buffer, output]);

            // Phase 2: Scan block sums (recursive)
            self.scan_block_sums(encoder, block_count as u32);

            encoder.memory_barrier_with_resources(&[&self.block_sums_buffer]);

            // Phase 3: Add block sums
            encoder.set_compute_pipeline_state(&self.phase3_pipeline);
            encoder.set_buffer(0, Some(output), 0);
            encoder.set_buffer(1, Some(&self.block_sums_buffer), 0);
            encoder.set_bytes(2, 4, &count as *const _ as *const _);

            let threads = MTLSize::new(count as u64, 1, 1);
            encoder.dispatch_threads(threads, MTLSize::new(Self::BLOCK_SIZE as u64, 1, 1));
        }
    }

    fn scan_block_sums(&self, encoder: &ComputeCommandEncoderRef, block_count: u32) {
        // For simplicity, use single-block scan if block_count <= 512
        // For very large arrays, would need recursive multi-block
        encoder.set_compute_pipeline_state(&self.single_block_pipeline);
        encoder.set_buffer(0, Some(&self.block_sums_buffer), 0);
        encoder.set_buffer(1, Some(&self.block_sums_buffer), 0);  // In-place
        encoder.set_bytes(2, 4, &block_count as *const _ as *const _);

        encoder.dispatch_threadgroups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(Self::BLOCK_SIZE as u64, 1, 1),
        );
    }
}
```

## Updated Paint Pipeline

```rust
// Update paint.rs to use parallel prefix sum

impl PaintPipeline {
    pub fn paint(
        &self,
        command_buffer: &CommandBufferRef,
        layout: &Buffer,
        styles: &Buffer,
        node_count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        // Phase 1: Count vertices per element
        encoder.set_compute_pipeline_state(&self.count_pipeline);
        encoder.set_buffer(0, Some(layout), 0);
        encoder.set_buffer(1, Some(styles), 0);
        encoder.set_buffer(2, Some(&self.vertex_counts_buffer), 0);
        encoder.set_bytes(3, 4, &node_count as *const _ as *const _);

        let threads = MTLSize::new(node_count as u64, 1, 1);
        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));

        encoder.memory_barrier_with_resources(&[&self.vertex_counts_buffer]);

        // Phase 2: PARALLEL prefix sum (was single-threaded!)
        self.prefix_sum.scan(
            encoder,
            &self.vertex_counts_buffer,
            &self.vertex_offsets_buffer,
            node_count,
        );

        encoder.memory_barrier_with_resources(&[&self.vertex_offsets_buffer]);

        // Phase 3: Generate vertices
        encoder.set_compute_pipeline_state(&self.generate_pipeline);
        encoder.set_buffer(0, Some(layout), 0);
        encoder.set_buffer(1, Some(styles), 0);
        encoder.set_buffer(2, Some(&self.vertex_offsets_buffer), 0);
        encoder.set_buffer(3, Some(&self.vertices_buffer), 0);
        encoder.set_bytes(4, 4, &node_count as *const _ as *const _);

        encoder.dispatch_threads(threads, MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }
}
```

## Benchmarks

```rust
fn bench_prefix_sum(c: &mut Criterion) {
    let device = Device::system_default().unwrap();
    let library = device.new_library_with_source(SHADER_SOURCE, &Default::default()).unwrap();
    let prefix_sum = GPUPrefixSum::new(&device, &library, 1_000_000);

    let mut group = c.benchmark_group("prefix_sum");

    for size in [256, 1024, 10_000, 100_000, 1_000_000] {
        // Generate random input
        let input: Vec<u32> = (0..size).map(|_| rand::random::<u32>() % 100).collect();
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer(
            (size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        group.bench_function(format!("sequential_{}", size), |b| {
            b.iter(|| {
                // Current single-threaded approach
                let input = input.clone();
                let mut output = vec![0u32; size];
                let mut sum = 0u32;
                for i in 0..size {
                    output[i] = sum;
                    sum += input[i];
                }
                output
            })
        });

        group.bench_function(format!("parallel_gpu_{}", size), |b| {
            b.iter(|| {
                let cmd = queue.new_command_buffer();
                let encoder = cmd.new_compute_command_encoder();
                prefix_sum.scan(&encoder, &input_buffer, &output_buffer, size as u32);
                encoder.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            })
        });
    }
}
```

### Expected Results

| Elements | Sequential | Parallel GPU | Speedup |
|----------|------------|--------------|---------|
| 256 | 0.001ms | 0.005ms | 0.2x (overhead) |
| 1K | 0.005ms | 0.006ms | 0.8x |
| 10K | 0.05ms | 0.01ms | 5x |
| 100K | 0.5ms | 0.02ms | 25x |
| 1M | 5ms | 0.1ms | 50x |

## Tests

```rust
#[test]
fn test_small_prefix_sum() {
    let input = vec![1, 2, 3, 4, 5];
    let output = gpu_prefix_sum(&input);

    // Exclusive prefix sum
    assert_eq!(output, vec![0, 1, 3, 6, 10]);
}

#[test]
fn test_power_of_two() {
    let input: Vec<u32> = (0..256).collect();
    let output = gpu_prefix_sum(&input);

    // Verify against CPU
    let expected = cpu_prefix_sum(&input);
    assert_eq!(output, expected);
}

#[test]
fn test_non_power_of_two() {
    let input: Vec<u32> = (0..1000).map(|i| i % 10).collect();
    let output = gpu_prefix_sum(&input);

    let expected = cpu_prefix_sum(&input);
    assert_eq!(output, expected);
}

#[test]
fn test_large() {
    let input: Vec<u32> = (0..100_000).map(|_| rand::random::<u32>() % 100).collect();
    let output = gpu_prefix_sum(&input);

    let expected = cpu_prefix_sum(&input);
    assert_eq!(output, expected);
}

#[test]
fn test_all_zeros() {
    let input = vec![0u32; 1000];
    let output = gpu_prefix_sum(&input);

    assert!(output.iter().all(|&x| x == 0));
}

#[test]
fn test_all_ones() {
    let input = vec![1u32; 1000];
    let output = gpu_prefix_sum(&input);

    for (i, &val) in output.iter().enumerate() {
        assert_eq!(val, i as u32);
    }
}

fn cpu_prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut output = Vec::with_capacity(input.len());
    let mut sum = 0;
    for &x in input {
        output.push(sum);
        sum += x;
    }
    output
}
```

## Acceptance Criteria

- [ ] Correct results for all input sizes
- [ ] Handles non-power-of-two sizes
- [ ] Speedup ≥5x for 10K+ elements
- [ ] Integrated into paint pipeline
- [ ] Works with vertex count allocation
- [ ] No race conditions or data hazards

## Dependencies

- None (foundational algorithm)

## Blocks

- All GPU pipeline stages that need prefix sum
- Issue #106: GPU Layout Engine (for allocations)
