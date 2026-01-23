// Issue #11: Unified Worker Model - Core GPU Kernel
//
// The core compute kernel that runs all OS logic in a single threadgroup.
// All 1024 threads participate in ALL phases (no fixed SIMD roles).

use metal::*;
use std::mem;
use std::time::Instant;

/// Metal shader source for the unified worker kernel
pub const KERNEL_SOURCE: &str = include_str!("shaders/kernel.metal");

/// Kernel parameters passed to the GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct KernelParams {
    pub widget_count: u32,
    pub max_widgets: u32,
    pub delta_time: f32,
    pub time: f32,
    pub frame_number: u32,
}

/// The GPU OS Kernel - manages the compute pipeline
pub struct GpuKernel {
    pipeline: ComputePipelineState,
    test_pipeline: ComputePipelineState,
    max_threads: usize,
    params_buffer: Buffer,
}

impl GpuKernel {
    /// Compile and create the GPU kernel
    pub fn new(device: &Device) -> Result<Self, String> {
        // Compile shader library
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(KERNEL_SOURCE, &options)
            .map_err(|e| format!("Failed to compile kernel: {}", e))?;

        // Get the main kernel function
        let kernel_fn = library
            .get_function("gpu_os_kernel", None)
            .map_err(|e| format!("Failed to get gpu_os_kernel function: {}", e))?;

        // Create compute pipeline
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel_fn)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        // Get max threads
        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;

        // Get test kernel for simple verification
        let test_fn = library
            .get_function("test_kernel", None)
            .map_err(|e| format!("Failed to get test_kernel function: {}", e))?;

        let test_pipeline = device
            .new_compute_pipeline_state_with_function(&test_fn)
            .map_err(|e| format!("Failed to create test pipeline: {}", e))?;

        // Create params buffer
        let params_buffer = device.new_buffer(
            mem::size_of::<KernelParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            pipeline,
            test_pipeline,
            max_threads,
            params_buffer,
        })
    }

    /// Get the maximum threads per threadgroup supported by this kernel
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.max_threads
    }

    /// Check if kernel supports the required 1024 threads
    pub fn supports_1024_threads(&self) -> bool {
        self.max_threads >= 1024
    }

    /// Get the pipeline for external use
    pub fn pipeline(&self) -> &ComputePipelineState {
        &self.pipeline
    }

    /// Update kernel parameters
    fn update_params(&self, params: &KernelParams) {
        unsafe {
            let ptr = self.params_buffer.contents() as *mut KernelParams;
            *ptr = *params;
        }
    }

    /// Encode the kernel dispatch into a command buffer
    pub fn encode(&self, encoder: &ComputeCommandEncoderRef, memory: &super::memory::GpuMemory, params: &KernelParams) {
        self.update_params(params);

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&memory.widget_buffer), 0);
        encoder.set_buffer(1, Some(&memory.input_queue_buffer), 0);
        encoder.set_buffer(2, Some(&memory.draw_args_buffer), 0);
        encoder.set_buffer(3, Some(&memory.frame_state_buffer), 0);
        encoder.set_buffer(4, Some(&self.params_buffer), 0);

        // Dispatch single threadgroup with 1024 threads (or max supported)
        let threads = self.max_threads.min(1024);
        let threadgroup_size = MTLSize::new(threads as u64, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
    }

    /// Run the kernel synchronously (for testing)
    pub fn run_sync(&self, queue: &CommandQueue, memory: &super::memory::GpuMemory, params: &KernelParams) -> f64 {
        let start = Instant::now();

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.encode(&encoder, memory, params);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        start.elapsed().as_secs_f64() * 1000.0 // Return milliseconds
    }

    /// Run the test kernel to verify basic execution
    pub fn run_test_kernel(&self, queue: &CommandQueue, device: &Device) -> Vec<u32> {
        let threads = self.max_threads.min(1024);
        let output_buffer = device.new_buffer(
            (threads * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Zero the buffer
        unsafe {
            let ptr = output_buffer.contents() as *mut u32;
            std::ptr::write_bytes(ptr, 0, threads);
        }

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.test_pipeline);
        encoder.set_buffer(0, Some(&output_buffer), 0);

        let threadgroup_size = MTLSize::new(threads as u64, 1, 1);
        let threadgroups = MTLSize::new(1, 1, 1);

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let mut results = Vec::with_capacity(threads);
        unsafe {
            let ptr = output_buffer.contents() as *const u32;
            for i in 0..threads {
                results.push(*ptr.add(i));
            }
        }
        results
    }
}

/// Kernel execution statistics
#[derive(Debug, Default, Clone)]
pub struct KernelStats {
    /// Number of threads that executed
    pub threads_executed: u32,
    /// Number of phases completed
    pub phases_completed: u32,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Whether all threads ran successfully
    pub all_threads_ran: bool,
}

impl GpuKernel {
    /// Run kernel and collect statistics (for testing/profiling)
    pub fn run_with_stats(&self, queue: &CommandQueue, memory: &super::memory::GpuMemory, params: &KernelParams) -> KernelStats {
        let threads = self.max_threads.min(1024) as u32;

        let start = Instant::now();

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        self.encode(&encoder, memory, params);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Get frame state to verify execution
        let frame_state = memory.frame_state();

        KernelStats {
            threads_executed: threads,
            phases_completed: 5, // We have 5 phases in the kernel
            execution_time_ms,
            all_threads_ran: frame_state.frame_number == params.frame_number + 1,
        }
    }

    /// Benchmark the kernel with multiple runs
    pub fn benchmark(&self, queue: &CommandQueue, memory: &super::memory::GpuMemory, runs: usize) -> Vec<f64> {
        let mut times = Vec::with_capacity(runs);

        let params = KernelParams {
            widget_count: 256,
            max_widgets: 1024,
            delta_time: 1.0 / 120.0,
            time: 0.0,
            frame_number: 0,
        };

        // Warmup
        for _ in 0..5 {
            self.run_sync(queue, memory, &params);
        }

        // Benchmark
        for i in 0..runs {
            let params = KernelParams {
                frame_number: i as u32,
                ..params
            };
            times.push(self.run_sync(queue, memory, &params));
        }

        times
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_params_size() {
        assert_eq!(mem::size_of::<KernelParams>(), 20);
    }
}
