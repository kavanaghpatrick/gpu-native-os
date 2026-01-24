// GPU Profiler - Measure exactly where persistence ends
//
// THE GPU IS THE COMPUTER. We need hard data, not guesses.

use metal::*;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, AtomicBool};

// ============================================================================
// Profiling Data Structures
// ============================================================================

/// GPU-side timing data written by shaders
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimestamp {
    pub kernel_start: u64,      // First thread's timestamp
    pub kernel_end: u64,        // Last thread's timestamp
    pub iterations: u32,        // How many iterations completed
    pub checkpoint_count: u32,  // How many checkpoints written
}

/// CPU-side timing for dispatch overhead
#[derive(Clone, Debug)]
pub struct DispatchTiming {
    pub encode_start: Instant,
    pub encode_end: Instant,
    pub commit_time: Instant,
    pub signal_received: Instant,

    // Derived metrics
    pub encode_us: f64,
    pub dispatch_to_signal_us: f64,
    pub total_us: f64,
}

/// Comprehensive profiling results
#[derive(Clone, Debug, Default)]
pub struct ProfileResults {
    pub dispatch_timings: Vec<DispatchTiming>,
    pub gpu_timestamps: Vec<GpuTimestamp>,

    // Aggregate stats
    pub avg_encode_us: f64,
    pub avg_dispatch_us: f64,
    pub avg_gpu_kernel_us: f64,
    pub min_dispatch_gap_us: f64,
    pub max_dispatch_gap_us: f64,

    // Persistence metrics
    pub state_survived: bool,
    pub cache_hit_rate: f64,
}

// ============================================================================
// Profiler Implementation
// ============================================================================

pub struct GpuProfiler {
    device: Device,
    queue: CommandQueue,

    // Timing buffers
    timestamp_buffer: Buffer,
    state_buffer: Buffer,

    // Pre-encoded command buffers for rapid dispatch
    pipeline: ComputePipelineState,

    // CPU-side tracking
    dispatch_count: AtomicU64,
    is_profiling: AtomicBool,
}

/// Shader that measures internal GPU timing and tests state persistence
const PROFILER_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GpuTimestamp {
    atomic_uint kernel_start_lo;
    atomic_uint kernel_start_hi;
    atomic_uint kernel_end_lo;
    atomic_uint kernel_end_hi;
    atomic_uint iterations;
    atomic_uint checkpoint_count;
    uint _pad[2];
};

struct PersistentState {
    atomic_uint counter;        // Increments each dispatch
    atomic_uint magic;          // Check if state survived
    uint data[256];             // Test data persistence
};

// Get GPU timestamp (architecture-specific)
inline uint2 get_timestamp() {
    // Metal doesn't expose raw GPU timestamps in shaders
    // We use atomic increments as a proxy for ordering
    return uint2(0, 0);  // Placeholder
}

kernel void profiler_kernel(
    device GpuTimestamp* timing [[buffer(0)]],
    device PersistentState* state [[buffer(1)]],
    constant uint& iteration_limit [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint total_threads [[threads_per_grid]]
) {
    // First thread records "start"
    if (tid == 0) {
        atomic_fetch_add_explicit(&timing->iterations, 1, memory_order_relaxed);
    }

    // Check magic number - did state survive from last dispatch?
    if (tid == 0) {
        uint magic = atomic_load_explicit(&state->magic, memory_order_relaxed);
        if (magic == 0xDEADBEEF) {
            // State survived! Increment counter
            atomic_fetch_add_explicit(&state->counter, 1, memory_order_relaxed);
        } else {
            // First dispatch or state was cleared
            atomic_store_explicit(&state->magic, 0xDEADBEEF, memory_order_relaxed);
            atomic_store_explicit(&state->counter, 1, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Simulate work with internal loop
    uint work = 0;
    for (uint i = 0; i < iteration_limit; i++) {
        work += tid * i;
        // Write to state to test persistence
        if (tid < 256) {
            state->data[tid] = work;
        }
    }

    // Last thread records "end"
    if (tid == total_threads - 1) {
        atomic_fetch_add_explicit(&timing->checkpoint_count, 1, memory_order_relaxed);
    }
}

// Measure dispatch gap - how long between kernel end and next kernel start
kernel void gap_detector(
    device atomic_uint* sequence [[buffer(0)]],
    device uint* timestamps [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        uint seq = atomic_fetch_add_explicit(sequence, 1, memory_order_relaxed);
        timestamps[seq] = seq;  // Just record ordering
    }
}
"#;

impl GpuProfiler {
    pub fn new(device: &Device) -> Result<Self, String> {
        let queue = device.new_command_queue();

        // Compile profiler shader
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(PROFILER_SHADER, &options)
            .map_err(|e| format!("Shader compile failed: {}", e))?;

        let function = library
            .get_function("profiler_kernel", None)
            .expect("Function not found");

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline failed: {}", e))?;

        // Create timing buffer
        let timestamp_buffer = device.new_buffer(
            std::mem::size_of::<GpuTimestamp>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create persistent state buffer
        let state_buffer = device.new_buffer(
            1024,  // PersistentState size
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize state buffer
        unsafe {
            let ptr = state_buffer.contents() as *mut u32;
            for i in 0..256 {
                *ptr.add(i) = 0;
            }
        }

        Ok(Self {
            device: device.clone(),
            queue,
            timestamp_buffer,
            state_buffer,
            pipeline,
            dispatch_count: AtomicU64::new(0),
            is_profiling: AtomicBool::new(false),
        })
    }

    /// Measure pure dispatch overhead (no GPU work)
    pub fn measure_dispatch_overhead(&self, iterations: usize) -> Vec<DispatchTiming> {
        let mut timings = Vec::with_capacity(iterations);

        // Warmup
        for _ in 0..10 {
            let cmd = self.queue.new_command_buffer();
            cmd.commit();
            cmd.wait_until_completed();
        }

        for _ in 0..iterations {
            let encode_start = Instant::now();

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&self.timestamp_buffer), 0);
            enc.set_buffer(1, Some(&self.state_buffer), 0);

            let iteration_limit: u32 = 1;  // Minimal work
            let limit_buf = self.device.new_buffer_with_data(
                &iteration_limit as *const _ as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_buffer(2, Some(&limit_buf), 0);

            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            enc.end_encoding();

            let encode_end = Instant::now();

            cmd.commit();
            let commit_time = Instant::now();

            cmd.wait_until_completed();
            let signal_received = Instant::now();

            let encode_us = encode_end.duration_since(encode_start).as_secs_f64() * 1_000_000.0;
            let dispatch_to_signal_us = signal_received.duration_since(commit_time).as_secs_f64() * 1_000_000.0;
            let total_us = signal_received.duration_since(encode_start).as_secs_f64() * 1_000_000.0;

            timings.push(DispatchTiming {
                encode_start,
                encode_end,
                commit_time,
                signal_received,
                encode_us,
                dispatch_to_signal_us,
                total_us,
            });
        }

        timings
    }

    /// Measure gap between consecutive dispatches
    pub fn measure_dispatch_gap(&self, iterations: usize) -> (f64, f64, f64) {
        let mut gaps = Vec::with_capacity(iterations);

        let sequence_buf = self.device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let _timestamps_buf = self.device.new_buffer(
            (iterations * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize
        unsafe {
            *(sequence_buf.contents() as *mut u32) = 0;
        }

        // Rapid-fire dispatches
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&self.timestamp_buffer), 0);
            enc.set_buffer(1, Some(&self.state_buffer), 0);

            let iteration_limit: u32 = 1;
            let limit_buf = self.device.new_buffer_with_data(
                &iteration_limit as *const _ as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_buffer(2, Some(&limit_buf), 0);

            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(64, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            gaps.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }

        // Calculate inter-dispatch gaps
        let mut inter_gaps: Vec<f64> = Vec::new();
        for i in 1..gaps.len() {
            inter_gaps.push(gaps[i] - gaps[i-1]);
        }

        if inter_gaps.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let sum: f64 = inter_gaps.iter().sum();
        let avg = sum / inter_gaps.len() as f64;
        let min = inter_gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = inter_gaps.iter().cloned().fold(0.0, f64::max);

        (min, avg, max)
    }

    /// Test if state persists across dispatches
    pub fn test_state_persistence(&self, dispatches: usize) -> (bool, u32) {
        // Clear state
        unsafe {
            let ptr = self.state_buffer.contents() as *mut u32;
            *ptr = 0;  // counter
            *ptr.add(1) = 0;  // magic (NOT 0xDEADBEEF)
        }

        // Multiple dispatches
        for _ in 0..dispatches {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&self.timestamp_buffer), 0);
            enc.set_buffer(1, Some(&self.state_buffer), 0);

            let iteration_limit: u32 = 100;
            let limit_buf = self.device.new_buffer_with_data(
                &iteration_limit as *const _ as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_buffer(2, Some(&limit_buf), 0);

            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Check state
        let (counter, magic) = unsafe {
            let ptr = self.state_buffer.contents() as *const u32;
            (*ptr, *ptr.add(1))
        };

        let survived = magic == 0xDEADBEEF && counter == dispatches as u32;
        (survived, counter)
    }

    /// Measure how many iterations we can do before timeout
    pub fn find_safe_iteration_limit(&self) -> u32 {
        let test_limits = [100, 1000, 10000, 100000, 1000000, 10000000];

        for &limit in &test_limits {
            let start = Instant::now();

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&self.timestamp_buffer), 0);
            enc.set_buffer(1, Some(&self.state_buffer), 0);

            let limit_buf = self.device.new_buffer_with_data(
                &limit as *const _ as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_buffer(2, Some(&limit_buf), 0);

            enc.dispatch_thread_groups(
                MTLSize::new(256, 1, 1),  // More threadgroups
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            println!("  {} iterations: {:.2}ms", limit, elapsed_ms);

            // If we're over 1 second, we're getting close to timeout
            if elapsed_ms > 1000.0 {
                // Return previous limit as "safe"
                let idx = test_limits.iter().position(|&x| x == limit).unwrap();
                if idx > 0 {
                    return test_limits[idx - 1];
                }
                return limit / 10;
            }
        }

        // All limits were safe
        *test_limits.last().unwrap()
    }

    /// Run comprehensive profiling
    pub fn run_full_profile(&self) -> ProfileResults {
        println!("\n=== GPU PROFILER: MEASURING PERSISTENCE BOUNDARIES ===\n");

        // 1. Dispatch overhead
        println!("1. Measuring dispatch overhead (100 iterations)...");
        let dispatch_timings = self.measure_dispatch_overhead(100);

        let avg_encode: f64 = dispatch_timings.iter().map(|t| t.encode_us).sum::<f64>()
            / dispatch_timings.len() as f64;
        let avg_dispatch: f64 = dispatch_timings.iter().map(|t| t.dispatch_to_signal_us).sum::<f64>()
            / dispatch_timings.len() as f64;
        let avg_total: f64 = dispatch_timings.iter().map(|t| t.total_us).sum::<f64>()
            / dispatch_timings.len() as f64;

        println!("   Encode:   {:.1}µs avg", avg_encode);
        println!("   Dispatch: {:.1}µs avg", avg_dispatch);
        println!("   Total:    {:.1}µs avg", avg_total);

        // 2. Dispatch gap
        println!("\n2. Measuring inter-dispatch gap (1000 rapid dispatches)...");
        let (min_gap, avg_gap, max_gap) = self.measure_dispatch_gap(1000);
        println!("   Min gap:  {:.1}µs", min_gap);
        println!("   Avg gap:  {:.1}µs", avg_gap);
        println!("   Max gap:  {:.1}µs", max_gap);

        // 3. State persistence
        println!("\n3. Testing state persistence across 10 dispatches...");
        let (survived, counter) = self.test_state_persistence(10);
        println!("   State survived: {}", survived);
        println!("   Counter value:  {} (expected 10)", counter);

        // 4. Safe iteration limit
        println!("\n4. Finding safe iteration limit before timeout...");
        let safe_limit = self.find_safe_iteration_limit();
        println!("   Safe limit: {} iterations", safe_limit);

        println!("\n=== PROFILING COMPLETE ===\n");

        ProfileResults {
            dispatch_timings,
            gpu_timestamps: vec![],
            avg_encode_us: avg_encode,
            avg_dispatch_us: avg_dispatch,
            avg_gpu_kernel_us: avg_total,
            min_dispatch_gap_us: min_gap,
            max_dispatch_gap_us: max_gap,
            state_survived: survived,
            cache_hit_rate: 0.0,  // TODO: implement
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_profiler_creation() {
        let device = Device::system_default().expect("No Metal device");
        let profiler = GpuProfiler::new(&device).expect("Failed to create profiler");
        assert!(profiler.dispatch_count.load(Ordering::SeqCst) == 0);
    }

    #[test]
    fn run_full_profile() {
        let device = Device::system_default().expect("No Metal device");
        let profiler = GpuProfiler::new(&device).expect("Failed to create profiler");
        let results = profiler.run_full_profile();

        // Basic sanity checks
        assert!(results.avg_encode_us > 0.0);
        assert!(results.avg_dispatch_us > 0.0);
        assert!(results.state_survived, "State should persist across dispatches");
    }
}
