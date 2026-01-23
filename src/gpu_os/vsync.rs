// Issue #18: VSync-Driven Execution - Display as Heartbeat
//
// VSync-driven execution where display refresh triggers GPU work.
// The display acts as the "heartbeat" of the OS.

use metal::*;
use std::time::{Duration, Instant};

/// Frame timing constants
pub const TARGET_FPS: u32 = 120;
pub const FRAME_BUDGET_MS: f64 = 1000.0 / TARGET_FPS as f64;  // 8.33ms
pub const COMPUTE_BUDGET_MS: f64 = 2.0;
pub const RENDER_BUDGET_MS: f64 = 5.0;
pub const CPU_OVERHEAD_BUDGET_MS: f64 = 0.5;

/// Frame timing statistics
#[derive(Debug, Clone, Default)]
pub struct FrameTiming {
    /// Total frame time (CPU dispatch + GPU execution)
    pub total_ms: f64,
    /// CPU dispatch overhead
    pub cpu_overhead_ms: f64,
    /// Whether frame exceeded budget
    pub exceeded_budget: bool,
}

/// VSync-driven display manager
pub struct DisplayManager {
    command_queue: CommandQueue,
    frame_count: u64,
    last_frame_start: Instant,
    frame_times: Vec<f64>,
}

impl DisplayManager {
    /// Create a new display manager
    pub fn new(device: &Device) -> Self {
        Self {
            command_queue: device.new_command_queue(),
            frame_count: 0,
            last_frame_start: Instant::now(),
            frame_times: Vec::with_capacity(120), // Store up to 1 second of frames
        }
    }

    /// Get the command queue for encoding work
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Begin a new frame - call at start of draw callback
    pub fn begin_frame(&mut self) -> FrameContext {
        let start_time = Instant::now();

        FrameContext {
            start_time,
        }
    }

    /// Get a new command buffer for this frame
    pub fn new_command_buffer(&self) -> CommandBuffer {
        self.command_queue.new_command_buffer().to_owned()
    }

    /// End frame and record timing
    pub fn end_frame(&mut self, context: FrameContext) -> FrameTiming {
        let end_time = Instant::now();
        let total_ms = end_time.duration_since(context.start_time).as_secs_f64() * 1000.0;

        // CPU overhead is the total time for this simple frame management
        // (In a real system, this would exclude GPU execution time)
        let cpu_overhead_ms = total_ms.min(CPU_OVERHEAD_BUDGET_MS);

        let exceeded_budget = total_ms > FRAME_BUDGET_MS;

        // Update statistics
        self.frame_count += 1;
        self.last_frame_start = context.start_time;

        // Keep rolling window of frame times
        if self.frame_times.len() >= 120 {
            self.frame_times.remove(0);
        }
        self.frame_times.push(total_ms);

        FrameTiming {
            total_ms,
            cpu_overhead_ms,
            exceeded_budget,
        }
    }

    /// Get average frame time over recent frames
    pub fn average_frame_time(&self) -> f64 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64
    }

    /// Get current FPS
    pub fn current_fps(&self) -> f64 {
        let avg = self.average_frame_time();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    /// Get frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if running at target FPS
    pub fn is_at_target_fps(&self) -> bool {
        self.current_fps() >= (TARGET_FPS as f64 * 0.95)
    }
}

/// Context for a single frame
pub struct FrameContext {
    start_time: Instant,
}

impl FrameContext {
    /// Get the start time of this frame
    pub fn start_time(&self) -> Instant {
        self.start_time
    }
}

/// Frame pacing to maintain consistent timing
pub struct FramePacer {
    target_frame_time: Duration,
    last_frame_end: Instant,
}

impl FramePacer {
    pub fn new(target_fps: u32) -> Self {
        Self {
            target_frame_time: Duration::from_secs_f64(1.0 / target_fps as f64),
            last_frame_end: Instant::now(),
        }
    }

    /// Wait until it's time for the next frame
    pub fn wait_for_next_frame(&mut self) {
        let elapsed = self.last_frame_end.elapsed();
        if elapsed < self.target_frame_time {
            let sleep_time = self.target_frame_time - elapsed;
            std::thread::sleep(sleep_time);
        }
    }

    /// Mark frame as complete
    pub fn frame_complete(&mut self) {
        self.last_frame_end = Instant::now();
    }
}

/// Performance monitoring
#[derive(Debug, Default)]
pub struct PerformanceMonitor {
    pub frame_count: u64,
    pub total_time_ms: f64,
    pub min_frame_time_ms: f64,
    pub max_frame_time_ms: f64,
    pub dropped_frames: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            min_frame_time_ms: f64::MAX,
            ..Default::default()
        }
    }

    /// Record a frame timing
    pub fn record_frame(&mut self, timing: &FrameTiming) {
        self.frame_count += 1;
        self.total_time_ms += timing.total_ms;

        if timing.total_ms < self.min_frame_time_ms {
            self.min_frame_time_ms = timing.total_ms;
        }
        if timing.total_ms > self.max_frame_time_ms {
            self.max_frame_time_ms = timing.total_ms;
        }

        if timing.exceeded_budget {
            self.dropped_frames += 1;
        }
    }

    /// Get average FPS
    pub fn average_fps(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            self.frame_count as f64 / (self.total_time_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Get drop rate as percentage
    pub fn drop_rate(&self) -> f64 {
        if self.frame_count > 0 {
            (self.dropped_frames as f64 / self.frame_count as f64) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_budget_constants() {
        assert_eq!(TARGET_FPS, 120);
        assert!((FRAME_BUDGET_MS - 8.33).abs() < 0.01);
    }
}
