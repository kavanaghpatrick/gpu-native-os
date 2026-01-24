# Issue #133: Persistent Search Kernel - Eliminate Dispatch Overhead

## Problem Statement

Every search invocation has fixed overhead:
1. Create command buffer (~0.5ms)
2. Encode compute command (~0.2ms)
3. Commit and wait (~0.3ms)
4. Result readback (~0.5ms)

**Total overhead: ~1.5ms per search**

For tiny directories (6.7 MB), this overhead is significant:
- GPU kernel: ~2ms
- Dispatch overhead: ~1.5ms
- **43% overhead!**

## Solution: Persistent Kernel with Work Queue

Keep a GPU kernel running continuously, fed by a work queue:

```
Traditional:  [dispatch][kernel][wait] [dispatch][kernel][wait] [dispatch]...
Persistent:   [========== kernel polling work queue ==========]
              CPU pushes work → GPU processes → CPU reads results
```

**Expected improvement:**
- Tiny: 10.5ms → ~6ms (eliminate per-search dispatch)
- Repeated searches: Near-zero latency

## Architecture

### Work Queue Design

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU-RESIDENT WORK QUEUE                  │
├─────────────────────────────────────────────────────────────┤
│  work_queue[0]: { pattern: "TODO", data_ptr, status: DONE } │
│  work_queue[1]: { pattern: "fn",   data_ptr, status: READY }│ ← GPU processing
│  work_queue[2]: { pattern: "",     data_ptr, status: EMPTY }│ ← CPU can write
│  work_queue[3]: { pattern: "",     data_ptr, status: EMPTY }│
├─────────────────────────────────────────────────────────────┤
│  head: 1 (GPU reads from here)                              │
│  tail: 2 (CPU writes to here)                               │
│  shutdown: false                                            │
└─────────────────────────────────────────────────────────────┘
```

### Data Structures

```rust
/// Work item for persistent kernel
#[repr(C)]
pub struct SearchWorkItem {
    pub pattern: [u8; 64],       // Search pattern (null-terminated)
    pub pattern_len: u32,        // Pattern length
    pub case_sensitive: u32,     // 0 = case-insensitive
    pub data_buffer_id: u32,     // Which data buffer to search
    pub status: AtomicU32,       // 0=empty, 1=ready, 2=processing, 3=done
    pub result_count: AtomicU32, // Number of matches found
    pub _padding: [u32; 2],      // Alignment
}

/// Persistent kernel control block
#[repr(C)]
pub struct PersistentKernelControl {
    pub head: AtomicU32,         // GPU reads from here
    pub tail: AtomicU32,         // CPU writes to here
    pub shutdown: AtomicU32,     // Signal kernel to exit
    pub heartbeat: AtomicU32,    // GPU increments to prove it's alive
}

/// GPU-resident work queue
pub struct PersistentSearchQueue {
    control: Buffer,             // PersistentKernelControl
    work_items: Buffer,          // [SearchWorkItem; QUEUE_SIZE]
    results: Buffer,             // Search results buffer
    data_buffers: Vec<Buffer>,   // Pre-loaded data buffers
}
```

### Metal Kernel

```metal
// Persistent kernel - runs continuously until shutdown
kernel void persistent_search_kernel(
    device PersistentKernelControl* control [[buffer(0)]],
    device SearchWorkItem* work_queue [[buffer(1)]],
    device SearchResult* results [[buffer(2)]],
    device const uchar4* data_buffers [[buffer(3)]],  // Array of data buffers
    constant uint& queue_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Only thread 0 manages queue, others wait for work
    threadgroup uint current_work_idx;
    threadgroup bool has_work;
    threadgroup SearchWorkItem current_work;

    while (true) {
        // Thread 0 checks for shutdown and new work
        if (tid == 0) {
            // Check shutdown flag
            if (atomic_load_explicit(&control->shutdown, memory_order_relaxed)) {
                has_work = false;
                // Signal all threads to exit
            } else {
                // Heartbeat - prove we're alive
                atomic_fetch_add_explicit(&control->heartbeat, 1, memory_order_relaxed);

                // Check for work
                uint head = atomic_load_explicit(&control->head, memory_order_acquire);
                uint tail = atomic_load_explicit(&control->tail, memory_order_acquire);

                if (head != tail) {
                    uint idx = head % queue_size;
                    uint status = atomic_load_explicit(&work_queue[idx].status, memory_order_acquire);

                    if (status == 1) {  // READY
                        // Claim this work item
                        atomic_store_explicit(&work_queue[idx].status, 2, memory_order_release);  // PROCESSING
                        current_work_idx = idx;
                        current_work = work_queue[idx];
                        has_work = true;
                    } else {
                        has_work = false;
                    }
                } else {
                    has_work = false;
                }
            }
        }

        // Sync all threads
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (!has_work) {
            // No work - spin briefly then check again
            // Could use simd_ballot to reduce power
            for (int i = 0; i < 1000; i++) {
                // Spin wait - could be optimized with sleep
            }
            continue;
        }

        // All threads participate in search
        device const uchar4* data = data_buffers + current_work.data_buffer_id * DATA_BUFFER_SIZE;

        // Existing vectorized search logic
        uint local_matches = 0;
        uint chunk_start = gid * 64;  // 64 bytes per thread

        // ... search logic from existing kernel ...

        // Aggregate results
        uint total_matches = simd_sum(local_matches);

        if (tid == 0) {
            atomic_store_explicit(&work_queue[current_work_idx].result_count, total_matches, memory_order_release);
            atomic_store_explicit(&work_queue[current_work_idx].status, 3, memory_order_release);  // DONE

            // Advance head
            atomic_fetch_add_explicit(&control->head, 1, memory_order_release);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
```

### Rust API

```rust
impl PersistentSearchQueue {
    /// Create and start persistent kernel
    pub fn new(device: &Device) -> Self {
        let control = device.new_buffer(
            std::mem::size_of::<PersistentKernelControl>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize control block
        unsafe {
            let ctrl = control.contents() as *mut PersistentKernelControl;
            (*ctrl).head.store(0, Ordering::Relaxed);
            (*ctrl).tail.store(0, Ordering::Relaxed);
            (*ctrl).shutdown.store(0, Ordering::Relaxed);
            (*ctrl).heartbeat.store(0, Ordering::Relaxed);
        }

        let queue = Self {
            control,
            work_items: device.new_buffer(/*...*/),
            results: device.new_buffer(/*...*/),
            data_buffers: Vec::new(),
        };

        // Start persistent kernel (runs until shutdown)
        queue.start_kernel(device);

        queue
    }

    /// Submit search (non-blocking, returns immediately)
    pub fn submit_search(&self, pattern: &str, data_buffer_id: u32) -> SearchHandle {
        let tail = unsafe {
            let ctrl = self.control.contents() as *mut PersistentKernelControl;
            (*ctrl).tail.fetch_add(1, Ordering::AcqRel)
        };

        let idx = tail % QUEUE_SIZE;

        // Write work item
        unsafe {
            let item = (self.work_items.contents() as *mut SearchWorkItem).add(idx as usize);
            (*item).pattern[..pattern.len()].copy_from_slice(pattern.as_bytes());
            (*item).pattern_len = pattern.len() as u32;
            (*item).data_buffer_id = data_buffer_id;
            (*item).result_count.store(0, Ordering::Relaxed);
            (*item).status.store(1, Ordering::Release);  // READY
        }

        SearchHandle { queue: self, idx }
    }

    /// Wait for search to complete
    pub fn wait_result(&self, handle: SearchHandle) -> u32 {
        loop {
            let status = unsafe {
                let item = (self.work_items.contents() as *const SearchWorkItem).add(handle.idx as usize);
                (*item).status.load(Ordering::Acquire)
            };

            if status == 3 {  // DONE
                return unsafe {
                    let item = (self.work_items.contents() as *const SearchWorkItem).add(handle.idx as usize);
                    (*item).result_count.load(Ordering::Acquire)
                };
            }

            std::hint::spin_loop();
        }
    }

    /// Shutdown persistent kernel
    pub fn shutdown(&self) {
        unsafe {
            let ctrl = self.control.contents() as *mut PersistentKernelControl;
            (*ctrl).shutdown.store(1, Ordering::Release);
        }
    }
}

impl Drop for PersistentSearchQueue {
    fn drop(&mut self) {
        self.shutdown();
    }
}
```

## Benchmarks

### Test Cases

1. **Single search latency** - Dispatch overhead eliminated
2. **Burst searches** - 100 searches in rapid succession
3. **Interactive search** - Keystroke-by-keystroke (search-as-you-type)

### Expected Results

| Scenario | Current | Persistent | Speedup |
|----------|---------|------------|---------|
| Tiny single search | 10.5ms | ~6ms | 1.75x |
| 100 searches (same data) | 1,050ms | ~200ms | 5x |
| Search-as-you-type (10 chars) | 105ms total | ~60ms total | 1.75x |

### Benchmark Code

```rust
#[test]
fn benchmark_persistent_kernel() {
    let device = Device::system_default().unwrap();

    // Load test data once
    let data = load_test_data(&device, ".");

    // Traditional: new dispatch per search
    let trad_start = Instant::now();
    for _ in 0..100 {
        let mut searcher = GpuContentSearch::new(&device, 1).unwrap();
        searcher.load_data(&data).unwrap();
        let _ = searcher.search("TODO", false);
    }
    let trad_time = trad_start.elapsed();

    // Persistent: reuse running kernel
    let persistent = PersistentSearchQueue::new(&device);
    persistent.load_data(0, &data);

    let pers_start = Instant::now();
    for _ in 0..100 {
        let handle = persistent.submit_search("TODO", 0);
        let _ = persistent.wait_result(handle);
    }
    let pers_time = pers_start.elapsed();

    println!("Traditional: {:.1}ms, Persistent: {:.1}ms, Speedup: {:.2}x",
        trad_time.as_secs_f64() * 1000.0,
        pers_time.as_secs_f64() * 1000.0,
        trad_time.as_secs_f64() / pers_time.as_secs_f64());
}
```

## Success Criteria

1. **Correctness:** Same results as traditional dispatch
2. **Latency:** ≥40% reduction in single-search overhead
3. **Throughput:** ≥3x improvement for burst searches
4. **Stability:** Kernel runs reliably for 10+ minutes
5. **Clean shutdown:** No hangs or resource leaks

## Implementation Steps

1. Add `SearchWorkItem` and `PersistentKernelControl` structs
2. Create persistent kernel Metal shader
3. Implement work queue management in Rust
4. Add `submit_search` / `wait_result` API
5. Implement clean shutdown with timeout
6. Add `--persistent` flag to `gpu_ripgrep`
7. Benchmark and tune spin-wait vs sleep
8. Consider making persistent the default for interactive use
