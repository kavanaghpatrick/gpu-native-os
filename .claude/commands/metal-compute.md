# Metal Compute: GPU as General-Purpose Processor

You are an expert in using Apple Metal for **general-purpose computation**, treating the GPU as a massively parallel CPU replacement. This is NOT about graphics, shaders for rendering, or game engines. This is about using Metal compute kernels to replace CPU workloads entirely.

## Core Philosophy: THE GPU IS THE COMPUTER

The CPU is an I/O coprocessor. The GPU is where computation happens. Every CPU touchpoint is technical debt.

### Mental Model Shift

| CPU Thinking | GPU Thinking |
|--------------|--------------|
| "Process items in a loop" | "Launch one thread per item" |
| "Use a thread pool" | "Use a threadgroup" |
| "Share data via mutex" | "Share data via atomics" |
| "Read file, then process" | "GPU loads file directly (MTLIOCommandQueue)" |
| "CPU coordinates work" | "GPU self-schedules via atomics" |

---

## Apple Silicon Architecture Fundamentals

### Unified Memory (Zero-Copy)

Apple Silicon shares memory between CPU and GPU. There is NO PCIe bus bottleneck.

```
M1/M2/M3/M4 Architecture:
┌─────────────────────────────────────┐
│         Unified Memory              │
│  (Shared by CPU, GPU, Neural Engine)│
│         400 GB/s bandwidth          │
└─────────────────────────────────────┘
        ↑           ↑           ↑
      CPU         GPU        Neural
     Cores       Cores       Engine
```

**Key Implication**: Use `StorageModeShared` for data both CPU and GPU access. There's no copy - it's the same memory.

```rust
// Rust: Create shared buffer (zero-copy between CPU/GPU)
let buffer = device.new_buffer(
    size,
    MTLResourceOptions::StorageModeShared
);

// CPU writes directly
let ptr = buffer.contents() as *mut MyData;
unsafe { *ptr = my_data; }

// GPU reads same memory - no copy!
encoder.set_buffer(0, Some(&buffer), 0);
```

### SIMD Groups (Waves/Wavefronts)

Apple GPUs execute threads in **SIMD groups of 32 threads** (like NVIDIA warps, AMD wavefronts).

```
Threadgroup (up to 1024 threads)
├── SIMD Group 0 (32 threads) ─── Execute in lockstep
├── SIMD Group 1 (32 threads) ─── Execute in lockstep
├── SIMD Group 2 (32 threads) ─── Execute in lockstep
└── ...
```

**Critical Rule**: All 32 threads in a SIMD group execute the SAME instruction. Branch divergence wastes cycles.

```metal
// BAD: Branch divergence - all 32 threads execute BOTH branches
if (tid % 2 == 0) {
    do_even_work();  // 16 threads active, 16 masked
} else {
    do_odd_work();   // 16 threads active, 16 masked
}
// Total: 64 instruction slots for 32 threads of work

// GOOD: Uniform control flow - all threads take same path
uint work_id = tid / 2;
bool is_even = (tid % 2 == 0);
float result = is_even ? even_data[work_id] : odd_data[work_id];
// All threads execute same instruction with different data
```

---

## Core Patterns for General-Purpose Compute

### Pattern 1: One Thread Per Work Item

Replace CPU loops with GPU threads.

```rust
// CPU approach (sequential)
for item in items.iter() {
    process(item);  // 1M iterations = 1M sequential ops
}

// GPU approach (parallel)
// Launch 1M threads, each processes one item
let thread_count = items.len();
let threadgroup_size = 256;
let threadgroups = (thread_count + 255) / 256;

encoder.dispatch_threadgroups(
    MTLSize::new(threadgroups as u64, 1, 1),
    MTLSize::new(threadgroup_size as u64, 1, 1)
);
```

```metal
kernel void process_items(
    device Item* items [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;  // Bounds check for partial threadgroups

    // Each thread processes exactly one item
    Item item = items[tid];
    items[tid] = process(item);
}
```

### Pattern 2: Atomic Work Queues (GPU Self-Scheduling)

Let the GPU schedule its own work without CPU involvement.

```metal
// Work queue in device memory
struct WorkQueue {
    atomic_uint head;      // Next item to dequeue
    atomic_uint tail;      // Next slot to enqueue
    uint capacity;
    WorkItem items[MAX_ITEMS];
};

kernel void worker_kernel(
    device WorkQueue* queue [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread tries to claim work
    while (true) {
        // Atomically claim next work item
        uint slot = atomic_fetch_add_explicit(
            &queue->head, 1, memory_order_relaxed
        );

        // Check if queue is exhausted
        uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);
        if (slot >= tail) break;

        // Process the work item
        WorkItem work = queue->items[slot % queue->capacity];
        process_work(work);

        // Optionally enqueue new work (producer-consumer)
        if (should_spawn_child_work(work)) {
            uint new_slot = atomic_fetch_add_explicit(
                &queue->tail, 1, memory_order_relaxed
            );
            queue->items[new_slot % queue->capacity] = child_work;
        }
    }
}
```

### Pattern 3: Parallel Reduction (SIMD-Aware)

Reduce an array to a single value using SIMD group operations.

```metal
kernel void parallel_sum(
    device float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    float value = input[tid];

    // Step 1: Reduce within SIMD group (no shared memory needed!)
    // simd_sum uses shuffle operations - extremely fast
    float simd_total = simd_sum(value);

    // Step 2: Lane 0 of each SIMD group has the sum
    if (simd_lane == 0) {
        // Atomic add to global result
        atomic_fetch_add_explicit(output, simd_total, memory_order_relaxed);
    }
}
```

### Pattern 4: Prefix Sum (Scan)

Build cumulative sums - foundation for many parallel algorithms.

```metal
// Simplified Hillis-Steele scan within threadgroup
kernel void prefix_sum(
    device float* data [[buffer(0)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Load to shared memory
    shared[local_id] = data[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele scan
    for (uint offset = 1; offset < group_size; offset *= 2) {
        float val = 0;
        if (local_id >= offset) {
            val = shared[local_id - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[local_id] += val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    data[tid] = shared[local_id];
}
```

### Pattern 5: Hash Tables (Cuckoo Hashing for O(1))

Linear probing causes SIMD divergence. Cuckoo hashing is O(1) guaranteed.

```metal
constant uint EMPTY = 0xFFFFFFFF;

struct CuckooTable {
    uint keys1[CAPACITY];
    uint values1[CAPACITY];
    uint keys2[CAPACITY];
    uint values2[CAPACITY];
};

// O(1) lookup - exactly 2 probes, always
uint cuckoo_lookup(device CuckooTable* table, uint key) {
    uint h1 = hash1(key) & (CAPACITY - 1);
    uint h2 = hash2(key) & (CAPACITY - 1);

    // Check table 1
    if (table->keys1[h1] == key) {
        return table->values1[h1];
    }

    // Check table 2
    if (table->keys2[h2] == key) {
        return table->values2[h2];
    }

    return EMPTY;  // Not found
}
```

---

## Metal-Specific Constraints and Workarounds

### Constraint 1: No Device-Scope Barriers

Metal cannot synchronize between threadgroups within a single dispatch. This is the biggest difference from CUDA/Vulkan.

**Workaround**: Use multiple dispatches with CPU coordination, or design algorithms that don't require cross-threadgroup sync.

```rust
// Multi-pass approach for algorithms needing global sync
for pass in 0..num_passes {
    encoder.dispatch_threadgroups(...);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();  // Global sync point
    // Next pass sees all previous writes
}
```

### Constraint 2: Only memory_order_relaxed for Atomics

Metal atomics don't support acquire/release semantics.

**Workaround**: Use multiple dispatches for ordering, or accept relaxed ordering where safe.

```metal
// Relaxed atomics are fine for:
// - Counters
// - Work queue indices
// - Reduction accumulators

// Be careful with:
// - Producer-consumer with data dependencies
// - Lock-free data structures requiring ordering
```

### Constraint 3: Truly Infinite Loops Are Blocked (Apple Silicon M4)

Kernels with `while(true)` without numeric bounds never execute on M4.

**Workaround**: Use bounded iteration with CPU re-dispatch (pseudo-persistent kernels).

```metal
// BAD: Never executes on M4
kernel void infinite_worker(...) {
    while (true) {  // BLOCKED
        do_work();
    }
}

// GOOD: Bounded iteration, CPU re-dispatches
kernel void bounded_worker(
    device atomic_uint* shutdown [[buffer(0)]],
    ...
) {
    for (uint i = 0; i < 1000000u; i++) {  // ~0.5s of work
        if (atomic_load_explicit(shutdown, memory_order_relaxed)) break;
        do_work();
    }
    // Kernel completes, CPU re-dispatches
}
```

```rust
// Rust: Pseudo-persistent kernel pattern
thread::spawn(move || {
    while !shutdown.load(Ordering::Acquire) {
        let cmd = queue.new_command_buffer();
        // ... encode bounded_worker kernel ...
        cmd.commit();
        cmd.wait_until_completed();  // ~0.5s
        // Loop continues, re-dispatching
    }
});
```

### Constraint 4: 32-bit Atomics Only

Metal only provides `atomic_int` and `atomic_uint` (32-bit).

**Workaround**: Use 64-bit values as two 32-bit atomics, or use double-wide CAS patterns.

```metal
// 64-bit counter using two 32-bit atomics
struct Counter64 {
    atomic_uint lo;
    atomic_uint hi;
};

void increment_64(device Counter64* c) {
    uint old_lo = atomic_fetch_add_explicit(&c->lo, 1, memory_order_relaxed);
    if (old_lo == 0xFFFFFFFF) {  // Overflow
        atomic_fetch_add_explicit(&c->hi, 1, memory_order_relaxed);
    }
}
```

---

## GPU-Direct File I/O (MTLIOCommandQueue)

Metal 3 allows GPU to load files directly without CPU involvement.

```rust
// Create I/O command queue
let io_queue = device.new_io_command_queue(
    &MTLIOCommandQueueDescriptor::new()
)?;

// Open file handle
let file_handle = device.new_io_file_handle(&file_url)?;

// Create destination buffer
let buffer = device.new_buffer(file_size, MTLResourceOptions::StorageModePrivate);

// Create I/O command buffer
let io_cmd = io_queue.new_command_buffer();

// Encode load command (GPU loads file directly)
io_cmd.load_buffer(&buffer, 0, file_size, &file_handle, 0);

// Synchronize with compute work using shared events
let event = device.new_shared_event();
io_cmd.signal_event(&event, 1);

// Compute command buffer waits for I/O
compute_cmd.wait_for_event(&event, 1);
compute_cmd.encode_compute(...);  // Now buffer has file contents
```

---

## Memory Safety Patterns (From Our Bug Analysis)

### Always Pass Buffer Sizes

Every buffer access must be bounds-checked.

```metal
// BAD: No bounds check
kernel void unsafe_kernel(device float* data [[buffer(0)]]) {
    data[tid] = 0;  // OOB if tid >= actual size
}

// GOOD: Size parameter enables validation
kernel void safe_kernel(
    device float* data [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;  // Bounds check
    data[tid] = 0;
}
```

### Use Atomics with Bounds Validation

```metal
// BAD: Atomic gives slot, but no bounds check
uint slot = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
buffer[slot] = value;  // May be OOB

// GOOD: Validate atomic result
uint slot = atomic_fetch_add_explicit(&counter, 1, memory_order_relaxed);
if (slot < max_slots) {
    buffer[slot] = value;
} else {
    // Revert allocation
    atomic_fetch_sub_explicit(&counter, 1, memory_order_relaxed);
}
```

### State Machine Completeness

Every defined state must be used. Unused states cause race conditions.

```metal
// Define ALL states
constant uint STATE_EMPTY = 0;
constant uint STATE_INSERTING = 1;  // MUST USE THIS
constant uint STATE_OCCUPIED = 2;

// Correct insertion with transitional state
uint expected = STATE_EMPTY;
if (atomic_compare_exchange_weak_explicit(
    &slot->state, &expected, STATE_INSERTING,  // Transitional
    memory_order_relaxed, memory_order_relaxed
)) {
    slot->key = key;
    slot->value = value;
    // Now mark as occupied (visible to readers)
    atomic_store_explicit(&slot->state, STATE_OCCUPIED, memory_order_relaxed);
}
```

---

## Struct Alignment (Rust <-> Metal)

### The packed_float3 Rule

Metal `float3` is 16 bytes. Rust `[f32; 3]` is 12 bytes. They don't match.

```rust
// Rust struct
#[repr(C)]
struct Vertex {
    position: [f32; 3],  // 12 bytes
    _pad: f32,           // 4 bytes - EXPLICIT PADDING
    color: [f32; 4],     // 16 bytes
}
// Total: 32 bytes
```

```metal
// Metal struct - MUST match exactly
struct Vertex {
    packed_float3 position;  // 12 bytes (NOT float3!)
    float _pad;              // 4 bytes
    float4 color;            // 16 bytes
};
// Total: 32 bytes
```

### Buffer Binding Contract

Document bindings explicitly to prevent mismatches.

```rust
/// Kernel: process_data
/// Buffer bindings:
/// - 0: device Data* data
/// - 1: constant uint& count
/// - 2: device Result* results
fn dispatch_process_data(&self) {
    encoder.set_buffer(0, Some(&self.data_buffer), 0);
    encoder.set_buffer(1, Some(&self.count_buffer), 0);
    encoder.set_buffer(2, Some(&self.results_buffer), 0);
}
```

---

## Performance Optimization Hierarchy

### 1. Maximize Occupancy

More threads in flight = more latency hiding.

```metal
// Declare max threads to help compiler optimize register usage
[[max_total_threads_per_threadgroup(256)]]
kernel void optimized_kernel(...) { ... }
```

### 2. Prefer SIMD Operations Over Shared Memory

On Apple GPUs (Family 9+), SIMD shuffles are faster than threadgroup memory.

```metal
// Instead of shared memory reduction
float simd_total = simd_sum(my_value);  // Uses shuffle, no memory

// Compare values across SIMD group
float simd_max = simd_max(my_value);
float simd_min = simd_min(my_value);

// Broadcast from lane 0
float broadcast = simd_broadcast(my_value, 0);
```

### 3. Use 16-bit Types

Half precision doubles register availability.

```metal
// Use half when full precision not needed
half4 color = half4(texture.sample(sampler, uv));
half result = dot(color.rgb, half3(0.299h, 0.587h, 0.114h));
```

### 4. Minimize Atomics

Profile first - moderate use is fine, heavy use bottlenecks.

```metal
// BAD: Global atomic per thread
atomic_fetch_add_explicit(&global_counter, 1, memory_order_relaxed);

// GOOD: SIMD group reduction first, then one atomic per group
uint count = simd_sum(my_contribution);
if (simd_lane == 0) {
    atomic_fetch_add_explicit(&global_counter, count, memory_order_relaxed);
}
```

---

## When GPU Compute Makes Sense

| Use Case | GPU Advantage | Speedup |
|----------|---------------|---------|
| Matrix multiplication | Massive parallelism | 50-100x |
| Image/signal processing | Regular data access | 10-50x |
| Hashing/searching | Parallel probes | 5-20x |
| Sorting (large arrays) | Parallel merge/radix | 5-15x |
| Tree traversal | Depends on structure | 1-5x |
| Sequential algorithms | N/A - CPU better | 0.1-0.5x |

**Rule of Thumb**: If you can't express work as "one thread per item," the GPU may not help.

---

## Debug Tips

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Kernel never executes | `while(true)` on M4 | Use bounded iteration |
| Silent data corruption | OOB buffer access | Add bounds checks |
| Vertex data scrambled | `float3` vs `packed_float3` | Use packed types |
| Race conditions | Unused state machine state | Implement all states |
| Wrong results | SIMD divergence | Make control flow uniform |
| GPU hangs | Deadlock from barrier in divergent flow | Barriers must be uniform |

---

## Sources and References

- [Metal Overview - Apple Developer](https://developer.apple.com/metal/)
- [Discover Metal 4 - WWDC25](https://developer.apple.com/videos/play/wwdc2025/205/)
- [Metal Performance Best Practices](https://developer.apple.com/videos/play/tech-talks/111373/)
- [Optimizing Parallel Reduction in Metal for M1](https://kieber-emmons.medium.com/optimizing-parallel-reduction-in-metal-for-apple-m1-8e8677b49b01)
- [Apple G13 GPU Architecture Reference](https://dougallj.github.io/applegpu/docs.html)
- [Metal by Example: Introduction to Compute](https://metalbyexample.com/introduction-to-compute/)
- [A note on Metal shader converter (barrier limitations)](https://raphlinus.github.io/gpu/2023/06/12/shader-converter.html)
- [Load resources faster with Metal 3 - WWDC22](https://developer.apple.com/videos/play/wwdc2022/10104/)
- [Understanding Unified Memory on Apple Silicon](https://www.oreateai.com/blog/understanding-unified-memory-the-heart-of-apple-silicon-performance/)
- [SIMT Execution Model - ACM SIGGRAPH](https://blog.siggraph.org/2026/01/simd-started-it-simt-improved-it.html/)
