// GPU Work Queue - GPU as Computing Unit
//
// THE GPU IS THE COMPUTER. The CPU is just an I/O handler.
//
// This module implements a work queue model where:
// 1. GPU pulls work from a queue in device memory
// 2. GPU decides what to do next (not the host)
// 3. State persists across execution quanta
// 4. Host only intervenes for I/O

use metal::*;
use std::time::Instant;

// ============================================================================
// Work Queue Data Structures
// ============================================================================

/// Work item types the GPU can process
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorkType {
    Noop = 0,
    Compute = 1,      // Generic compute work
    Search = 2,       // Search through data
    Transform = 3,    // Transform data in-place
    Aggregate = 4,    // Reduce/aggregate values
    Custom = 5,       // User-defined work
}

/// A single work item in the queue
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkItem {
    pub work_type: u32,
    pub priority: u32,
    pub data_offset: u32,    // Offset into shared data buffer
    pub data_size: u32,      // Size of data to process
    pub param0: u32,         // Work-specific parameters
    pub param1: u32,
    pub param2: u32,
    pub param3: u32,
}

impl Default for WorkItem {
    fn default() -> Self {
        Self {
            work_type: WorkType::Noop as u32,
            priority: 0,
            data_offset: 0,
            data_size: 0,
            param0: 0,
            param1: 0,
            param2: 0,
            param3: 0,
        }
    }
}

/// GPU-side queue state (persistent across execution quanta)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct QueueState {
    // Queue pointers (atomically updated by GPU)
    pub head: u32,           // Next item to dequeue
    pub tail: u32,           // Next slot for enqueue
    pub capacity: u32,       // Queue size
    pub _pad0: u32,

    // Execution state
    pub items_processed: u32,
    pub current_item: u32,
    pub execution_phase: u32,  // 0=idle, 1=dequeue, 2=execute, 3=complete
    pub _pad1: u32,

    // Checkpoint for restart
    pub checkpoint_iteration: u32,
    pub checkpoint_valid: u32,  // Magic number when valid
    pub last_quantum_start: u32,  // For tracking execution time
    pub quantum_count: u32,       // How many quanta executed

    // Statistics
    pub total_items: u32,
    pub items_in_flight: u32,
    pub errors: u32,
    pub _pad2: u32,
}

/// Result from a work item
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkResult {
    pub work_id: u32,
    pub status: u32,       // 0=pending, 1=success, 2=error
    pub result_value: u32,
    pub cycles_used: u32,
}

// ============================================================================
// GPU Work Queue Implementation
// ============================================================================

const QUEUE_CAPACITY: usize = 4096;
const MAX_RESULTS: usize = 4096;
const CHECKPOINT_MAGIC: u32 = 0xC0FFEE42;

/// Shader that implements the GPU work queue processor
const WORK_QUEUE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Work types
constant uint WORK_NOOP = 0;
constant uint WORK_COMPUTE = 1;
constant uint WORK_SEARCH = 2;
constant uint WORK_TRANSFORM = 3;
constant uint WORK_AGGREGATE = 4;

struct WorkItem {
    uint work_type;
    uint priority;
    uint data_offset;
    uint data_size;
    uint param0;
    uint param1;
    uint param2;
    uint param3;
};

struct QueueState {
    // Atomic queue pointers
    atomic_uint head;
    atomic_uint tail;
    uint capacity;
    uint _pad0;

    // Execution state
    atomic_uint items_processed;
    atomic_uint current_item;
    atomic_uint execution_phase;
    uint _pad1;

    // Checkpoint
    atomic_uint checkpoint_iteration;
    atomic_uint checkpoint_valid;
    atomic_uint last_quantum_start;
    atomic_uint quantum_count;

    // Statistics
    atomic_uint total_items;
    atomic_uint items_in_flight;
    atomic_uint errors;
    uint _pad2;
};

struct WorkResult {
    uint work_id;
    uint status;
    uint result_value;
    uint cycles_used;
};

// ============================================================================
// GPU PULLS WORK - Not pushed by host
// ============================================================================

// Dequeue a work item (returns index, or 0xFFFFFFFF if empty)
inline uint dequeue_work(device QueueState* queue) {
    uint head = atomic_load_explicit(&queue->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&queue->tail, memory_order_relaxed);

    if (head == tail) {
        return 0xFFFFFFFF;  // Queue empty
    }

    // Try to claim this item
    uint expected = head;
    uint new_head = (head + 1) % queue->capacity;

    if (atomic_compare_exchange_weak_explicit(
        &queue->head, &expected, new_head,
        memory_order_relaxed, memory_order_relaxed
    )) {
        atomic_fetch_add_explicit(&queue->items_in_flight, 1, memory_order_relaxed);
        return head;
    }

    return 0xFFFFFFFF;  // Contention, try again
}

// Complete a work item
inline void complete_work(
    device QueueState* queue,
    device WorkResult* results,
    uint work_id,
    uint result_value
) {
    // Record result
    results[work_id].work_id = work_id;
    results[work_id].status = 1;  // Success
    results[work_id].result_value = result_value;

    // Update counters
    atomic_fetch_add_explicit(&queue->items_processed, 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(&queue->items_in_flight, 1, memory_order_relaxed);
}

// ============================================================================
// WORK EXECUTION - GPU decides what to do
// ============================================================================

// Execute a compute work item
inline uint execute_compute(
    WorkItem item,
    device float* data,
    uint tid
) {
    uint offset = item.data_offset;
    uint size = item.data_size;
    uint iterations = item.param0;

    if (iterations == 0) iterations = 1000;

    // Parallel computation over data range
    uint work_per_thread = (size + 255) / 256;
    uint my_start = tid * work_per_thread;
    uint my_end = min(my_start + work_per_thread, size);

    float sum = 0.0;
    for (uint i = my_start; i < my_end; i++) {
        float val = data[offset + i];
        // Some actual work
        for (uint j = 0; j < iterations; j++) {
            val = val * 1.001 + 0.001;
        }
        sum += val;
    }

    return as_type<uint>(sum);
}

// Execute a search work item
inline uint execute_search(
    WorkItem item,
    device float* data,
    uint tid,
    threadgroup atomic_uint* found_idx
) {
    uint offset = item.data_offset;
    uint size = item.data_size;
    float target = as_type<float>(item.param0);
    float epsilon = as_type<float>(item.param1);
    if (epsilon == 0.0) epsilon = 0.001;

    // Parallel search
    uint work_per_thread = (size + 255) / 256;
    uint my_start = tid * work_per_thread;
    uint my_end = min(my_start + work_per_thread, size);

    for (uint i = my_start; i < my_end; i++) {
        float val = data[offset + i];
        if (abs(val - target) < epsilon) {
            atomic_store_explicit(found_idx, i, memory_order_relaxed);
            return i;
        }
    }

    return 0xFFFFFFFF;  // Not found
}

// Execute a transform work item
inline uint execute_transform(
    WorkItem item,
    device float* data,
    uint tid
) {
    uint offset = item.data_offset;
    uint size = item.data_size;
    float scale = as_type<float>(item.param0);
    float bias = as_type<float>(item.param1);
    if (scale == 0.0) scale = 1.0;

    // Parallel transform in-place
    uint work_per_thread = (size + 255) / 256;
    uint my_start = tid * work_per_thread;
    uint my_end = min(my_start + work_per_thread, size);

    for (uint i = my_start; i < my_end; i++) {
        data[offset + i] = data[offset + i] * scale + bias;
    }

    return my_end - my_start;  // Items transformed
}

// Execute an aggregate work item
inline uint execute_aggregate(
    WorkItem item,
    device float* data,
    uint tid,
    threadgroup float* partial_sums
) {
    uint offset = item.data_offset;
    uint size = item.data_size;

    // Parallel reduction - each thread sums its portion
    uint work_per_thread = (size + 255) / 256;
    uint my_start = tid * work_per_thread;
    uint my_end = min(my_start + work_per_thread, size);

    float sum = 0.0;
    for (uint i = my_start; i < my_end; i++) {
        sum += data[offset + i];
    }

    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < 256) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return as_type<uint>(partial_sums[0]);
}

// ============================================================================
// MAIN WORK LOOP - GPU runs continuously
// ============================================================================

kernel void work_queue_processor(
    device QueueState* queue [[buffer(0)]],
    device WorkItem* items [[buffer(1)]],
    device WorkResult* results [[buffer(2)]],
    device float* shared_data [[buffer(3)]],
    constant uint& max_items_per_quantum [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Threadgroup-local storage for work coordination
    threadgroup uint work_idx;
    threadgroup WorkItem current_work;
    threadgroup atomic_uint tg_result;
    threadgroup float partial_sums[256];

    // Track quantum start
    if (tid == 0 && tgid == 0) {
        atomic_fetch_add_explicit(&queue->quantum_count, 1, memory_order_relaxed);
        atomic_store_explicit(&queue->execution_phase, 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // MAIN WORK LOOP - GPU decides how much work to do
    uint items_this_quantum = 0;

    while (items_this_quantum < max_items_per_quantum) {
        // Thread 0 of threadgroup 0 dequeues work
        if (tid == 0 && tgid == 0) {
            work_idx = dequeue_work(queue);
            if (work_idx != 0xFFFFFFFF) {
                current_work = items[work_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if we got work
        if (work_idx == 0xFFFFFFFF) {
            // Queue empty - exit quantum
            break;
        }

        // All threads execute the work item
        uint result = 0;

        if (current_work.work_type == WORK_COMPUTE) {
            result = execute_compute(current_work, shared_data, tid);
        } else if (current_work.work_type == WORK_SEARCH) {
            result = execute_search(current_work, shared_data, tid, &tg_result);
        } else if (current_work.work_type == WORK_TRANSFORM) {
            result = execute_transform(current_work, shared_data, tid);
        } else if (current_work.work_type == WORK_AGGREGATE) {
            result = execute_aggregate(current_work, shared_data, tid, partial_sums);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 records completion
        if (tid == 0 && tgid == 0) {
            complete_work(queue, results, work_idx, result);

            // Checkpoint periodically (every 100 items)
            items_this_quantum++;
            if (items_this_quantum % 100 == 0) {
                atomic_store_explicit(&queue->checkpoint_iteration,
                    atomic_load_explicit(&queue->items_processed, memory_order_relaxed),
                    memory_order_relaxed);
                atomic_store_explicit(&queue->checkpoint_valid, 0xC0FFEE42, memory_order_relaxed);
            }
        }

        threadgroup_barrier(mem_flags::mem_device);
    }

    // Mark quantum complete
    if (tid == 0 && tgid == 0) {
        atomic_store_explicit(&queue->execution_phase, 0, memory_order_relaxed);
    }
}

// ============================================================================
// QUEUE MANAGEMENT - Host can enqueue, GPU processes
// ============================================================================

kernel void enqueue_work(
    device QueueState* queue [[buffer(0)]],
    device WorkItem* items [[buffer(1)]],
    constant WorkItem* new_items [[buffer(2)]],
    constant uint& num_new_items [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_new_items) return;

    // Claim a slot in the queue
    uint tail = atomic_fetch_add_explicit(&queue->tail, 1, memory_order_relaxed);
    uint slot = tail % queue->capacity;

    // Write the work item
    items[slot] = new_items[tid];

    // Update total count
    atomic_fetch_add_explicit(&queue->total_items, 1, memory_order_relaxed);
}
"#;

pub struct GpuWorkQueue {
    device: Device,
    queue: CommandQueue,

    // Pipelines
    processor_pipeline: ComputePipelineState,
    enqueue_pipeline: ComputePipelineState,

    // Persistent buffers (GPU-resident state)
    queue_state: Buffer,
    work_items: Buffer,
    results: Buffer,
    shared_data: Buffer,

    // Configuration
    max_items_per_quantum: u32,
}

impl GpuWorkQueue {
    pub fn new(device: &Device, data_size: usize) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        // Compile shaders
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(WORK_QUEUE_SHADER, &options)
            .map_err(|e| format!("Shader compile failed: {}", e))?;

        let processor_fn = library
            .get_function("work_queue_processor", None)
            .expect("work_queue_processor not found");
        let enqueue_fn = library
            .get_function("enqueue_work", None)
            .expect("enqueue_work not found");

        let processor_pipeline = device
            .new_compute_pipeline_state_with_function(&processor_fn)
            .map_err(|e| format!("Pipeline failed: {}", e))?;
        let enqueue_pipeline = device
            .new_compute_pipeline_state_with_function(&enqueue_fn)
            .map_err(|e| format!("Pipeline failed: {}", e))?;

        // Create persistent buffers
        let queue_state = device.new_buffer(
            std::mem::size_of::<QueueState>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let work_items = device.new_buffer(
            (QUEUE_CAPACITY * std::mem::size_of::<WorkItem>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let results = device.new_buffer(
            (MAX_RESULTS * std::mem::size_of::<WorkResult>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let shared_data = device.new_buffer(
            (data_size * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize queue state
        unsafe {
            let state = queue_state.contents() as *mut QueueState;
            (*state).head = 0;
            (*state).tail = 0;
            (*state).capacity = QUEUE_CAPACITY as u32;
            (*state).items_processed = 0;
            (*state).current_item = 0;
            (*state).execution_phase = 0;
            (*state).checkpoint_iteration = 0;
            (*state).checkpoint_valid = 0;
            (*state).quantum_count = 0;
            (*state).total_items = 0;
            (*state).items_in_flight = 0;
            (*state).errors = 0;
        }

        Ok(Self {
            device: device.clone(),
            queue: command_queue,
            processor_pipeline,
            enqueue_pipeline,
            queue_state,
            work_items,
            results,
            shared_data,
            max_items_per_quantum: 1000,  // Default: process up to 1000 items per quantum
        })
    }

    /// Initialize shared data buffer with values
    pub fn init_data(&self, values: &[f32]) {
        unsafe {
            let ptr = self.shared_data.contents() as *mut f32;
            for (i, &v) in values.iter().enumerate() {
                *ptr.add(i) = v;
            }
        }
    }

    /// Enqueue work items (host pushes to queue, GPU will pull)
    pub fn enqueue(&self, items: &[WorkItem]) {
        if items.is_empty() {
            return;
        }

        // Create temporary buffer for new items
        let items_buf = self.device.new_buffer_with_data(
            items.as_ptr() as *const _,
            (items.len() * std::mem::size_of::<WorkItem>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let num_items = items.len() as u32;
        let num_buf = self.device.new_buffer_with_data(
            &num_items as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.enqueue_pipeline);
        enc.set_buffer(0, Some(&self.queue_state), 0);
        enc.set_buffer(1, Some(&self.work_items), 0);
        enc.set_buffer(2, Some(&items_buf), 0);
        enc.set_buffer(3, Some(&num_buf), 0);

        let threads = MTLSize::new(items.len() as u64, 1, 1);
        let threadgroup_size = MTLSize::new(64.min(items.len() as u64), 1, 1);
        enc.dispatch_threads(threads, threadgroup_size);
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();
    }

    /// Execute one quantum of work (GPU processes work from queue)
    /// Returns: (items_processed_this_quantum, total_items_processed, queue_empty)
    pub fn execute_quantum(&self) -> (u32, u32, bool) {
        let _start = Instant::now();

        // Get current state
        let (before_processed, _before_head, _before_tail) = unsafe {
            let state = self.queue_state.contents() as *const QueueState;
            ((*state).items_processed, (*state).head, (*state).tail)
        };

        // Create config buffer
        let max_items_buf = self.device.new_buffer_with_data(
            &self.max_items_per_quantum as *const _ as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.processor_pipeline);
        enc.set_buffer(0, Some(&self.queue_state), 0);
        enc.set_buffer(1, Some(&self.work_items), 0);
        enc.set_buffer(2, Some(&self.results), 0);
        enc.set_buffer(3, Some(&self.shared_data), 0);
        enc.set_buffer(4, Some(&max_items_buf), 0);

        // Single threadgroup with 256 threads (GPU decides how much work internally)
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();

        cmd.commit();
        cmd.wait_until_completed();

        // Get new state
        let (after_processed, after_head, after_tail) = unsafe {
            let state = self.queue_state.contents() as *const QueueState;
            ((*state).items_processed, (*state).head, (*state).tail)
        };

        let items_this_quantum = after_processed - before_processed;
        let queue_empty = after_head == after_tail;

        (items_this_quantum, after_processed, queue_empty)
    }

    /// Get queue statistics
    pub fn get_stats(&self) -> QueueStats {
        unsafe {
            let state = self.queue_state.contents() as *const QueueState;
            QueueStats {
                items_processed: (*state).items_processed,
                items_pending: ((*state).tail.wrapping_sub((*state).head)) % (*state).capacity,
                items_in_flight: (*state).items_in_flight,
                quantum_count: (*state).quantum_count,
                checkpoint_valid: (*state).checkpoint_valid == CHECKPOINT_MAGIC,
                checkpoint_iteration: (*state).checkpoint_iteration,
            }
        }
    }

    /// Get result for a work item
    pub fn get_result(&self, work_id: u32) -> Option<WorkResult> {
        if work_id as usize >= MAX_RESULTS {
            return None;
        }

        unsafe {
            let results = self.results.contents() as *const WorkResult;
            let result = *results.add(work_id as usize);
            if result.status != 0 {
                Some(result)
            } else {
                None
            }
        }
    }

    /// Run work queue until empty (multiple quanta)
    pub fn run_until_empty(&self) -> RunStats {
        let start = Instant::now();
        let mut total_quanta = 0;
        let mut total_items = 0;

        loop {
            let (items, _, empty) = self.execute_quantum();
            total_quanta += 1;
            total_items += items;

            if empty {
                break;
            }
        }

        RunStats {
            total_quanta,
            total_items,
            total_time_us: start.elapsed().as_secs_f64() * 1_000_000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueStats {
    pub items_processed: u32,
    pub items_pending: u32,
    pub items_in_flight: u32,
    pub quantum_count: u32,
    pub checkpoint_valid: bool,
    pub checkpoint_iteration: u32,
}

#[derive(Debug, Clone)]
pub struct RunStats {
    pub total_quanta: u32,
    pub total_items: u32,
    pub total_time_us: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_queue_creation() {
        let device = Device::system_default().expect("No Metal device");
        let queue = GpuWorkQueue::new(&device, 1024).expect("Failed to create work queue");

        let stats = queue.get_stats();
        assert_eq!(stats.items_processed, 0);
        assert_eq!(stats.items_pending, 0);
    }

    #[test]
    fn test_enqueue_and_process() {
        let device = Device::system_default().expect("No Metal device");
        let queue = GpuWorkQueue::new(&device, 1024).expect("Failed to create work queue");

        // Initialize data
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        queue.init_data(&data);

        // Enqueue compute work
        let work_items: Vec<WorkItem> = (0..10).map(|i| WorkItem {
            work_type: WorkType::Compute as u32,
            priority: 0,
            data_offset: 0,
            data_size: 100,
            param0: 100,  // iterations
            param1: 0,
            param2: 0,
            param3: 0,
        }).collect();

        queue.enqueue(&work_items);

        let stats = queue.get_stats();
        assert_eq!(stats.items_pending, 10);

        // Process
        let run_stats = queue.run_until_empty();

        assert_eq!(run_stats.total_items, 10);
        println!("Processed {} items in {} quanta, {:.1}µs total",
            run_stats.total_items, run_stats.total_quanta, run_stats.total_time_us);
    }

    #[test]
    fn test_state_persistence() {
        let device = Device::system_default().expect("No Metal device");
        let queue = GpuWorkQueue::new(&device, 1024).expect("Failed to create work queue");

        // Initialize data
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        queue.init_data(&data);

        // Enqueue work in batches, process, verify state persists
        for batch in 0..3 {
            let work_items: Vec<WorkItem> = (0..5).map(|_| WorkItem {
                work_type: WorkType::Compute as u32,
                priority: 0,
                data_offset: 0,
                data_size: 100,
                param0: 50,
                ..Default::default()
            }).collect();

            queue.enqueue(&work_items);
            queue.run_until_empty();
        }

        let final_stats = queue.get_stats();
        assert_eq!(final_stats.items_processed, 15);  // 3 batches × 5 items
        assert!(final_stats.quantum_count >= 3);  // At least one quantum per batch

        println!("State persisted across {} quanta, {} items processed",
            final_stats.quantum_count, final_stats.items_processed);
    }
}
