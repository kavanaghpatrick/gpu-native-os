# Issue #142: Async Frame Callbacks - Enable Non-Blocking Frame Completion

## Problem Statement

In `src/gpu_os/app.rs` line 297, LowLatency mode (the default) blocks the CPU on every frame:

```rust
if app.pipeline_mode() == PipelineMode::LowLatency {
    if let Some(prev) = &self.previous_command_buffer {
        prev.wait_until_completed();  // BLOCKS 1-2ms per frame
    }
}
```

This means:
- CPU is idle for 1-2ms every 8.33ms frame (120fps budget)
- **12-24% of CPU time is spent waiting**
- Cannot start preparing next frame while GPU renders current
- Inconsistent behavior: HighThroughput mode uses async, LowLatency doesn't

## Solution

Enable async completion callbacks in all pipeline modes while maintaining frame synchronization for input latency.

### Key Insight

We need to separate two concerns:
1. **Frame synchronization:** Ensuring input events are processed with correct frame timing
2. **CPU blocking:** Whether CPU waits for GPU or uses callbacks

Current code conflates these: LowLatency = sync wait, HighThroughput = async callback.

**New approach:** Both modes use async callbacks, but LowLatency mode processes input before submitting the frame that will handle it.

## Requirements

### Functional Requirements
1. All pipeline modes use `add_completed_handler()` instead of `wait_until_completed()`
2. Frame timing metrics (GPU start/end times) available in all modes
3. Input latency in LowLatency mode unchanged or improved
4. Support for triple buffering (3 frames in flight)

### Performance Requirements
1. Eliminate 1-2ms CPU idle time per frame
2. Enable CPU to prepare frame N+1 while GPU renders frame N
3. No increase in input-to-display latency

### Non-Functional Requirements
1. Thread-safe completion handlers
2. Clear error handling for GPU failures
3. Profiling/debugging support maintained

## Technical Design

### Async Completion with SharedEvent

```rust
// src/gpu_os/app.rs

use metal::MTLSharedEvent;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct GpuApp {
    // Existing fields...

    // New: Async frame tracking
    shared_event: SharedEvent,
    frame_counter: AtomicU64,
    completed_frame: AtomicU64,
    frame_timings: crossbeam::queue::ArrayQueue<FrameTiming>,

    // Triple buffering support
    command_buffers: [Option<CommandBuffer>; 3],
    current_buffer_index: usize,
}

impl GpuApp {
    pub fn render_frame_async(&mut self, app: &mut impl App) -> FrameHandle {
        let frame_id = self.frame_counter.fetch_add(1, Ordering::SeqCst);
        let buffer_index = (frame_id % 3) as usize;

        // Wait if this buffer slot is still in use (3 frames in flight max)
        self.wait_for_buffer_slot(buffer_index);

        // Build command buffer
        let command_buffer = self.build_frame_command_buffer(app);

        // Signal when complete
        command_buffer.encode_signal_event(&self.shared_event, frame_id + 1);

        // Async completion handler
        let timing_queue = self.frame_timings.clone();
        let completed_frame = self.completed_frame.clone();
        let frame_id_copy = frame_id;

        let handler = block::ConcreteBlock::new(move |cmd_buf: &CommandBufferRef| {
            let timing = FrameTiming {
                frame_id: frame_id_copy,
                gpu_start: cmd_buf.gpu_start_time(),
                gpu_end: cmd_buf.gpu_end_time(),
                kernel_start: cmd_buf.kernel_start_time(),
                kernel_end: cmd_buf.kernel_end_time(),
            };

            // Non-blocking push (drops if full)
            let _ = timing_queue.push(timing);

            // Update completed frame counter
            completed_frame.fetch_max(frame_id_copy, Ordering::SeqCst);
        });

        command_buffer.add_completed_handler(&handler.copy());
        command_buffer.commit();

        self.command_buffers[buffer_index] = Some(command_buffer.to_owned());

        FrameHandle { frame_id }
    }

    fn wait_for_buffer_slot(&self, slot: usize) {
        // Only wait if we're 3+ frames ahead
        let target_frame = self.frame_counter.load(Ordering::SeqCst).saturating_sub(2);

        if self.completed_frame.load(Ordering::SeqCst) < target_frame {
            // Use SharedEvent wait instead of command buffer wait
            self.shared_event.wait_until_signaled(target_frame + 1);
        }
    }

    pub fn get_frame_timing(&self, frame_id: u64) -> Option<FrameTiming> {
        // Check if frame is complete
        if self.completed_frame.load(Ordering::SeqCst) >= frame_id {
            // Search timing queue
            // Note: In production, use a proper lookup structure
            self.frame_timings.iter().find(|t| t.frame_id == frame_id).cloned()
        } else {
            None
        }
    }

    pub fn frames_in_flight(&self) -> u64 {
        self.frame_counter.load(Ordering::SeqCst) -
        self.completed_frame.load(Ordering::SeqCst)
    }
}
```

### Input Latency Preservation

```rust
// For LowLatency mode, ensure input is processed before frame submission

impl GpuApp {
    pub fn render_frame_low_latency(&mut self, app: &mut impl App) -> FrameHandle {
        // Process all pending input BEFORE building frame
        self.process_pending_input(app);

        // Now build and submit frame (async)
        self.render_frame_async(app)
    }

    fn process_pending_input(&mut self, app: &mut impl App) {
        // Drain input queue
        while let Some(event) = self.input_queue.pop() {
            app.handle_input(event);
        }

        // Update app state synchronously
        app.update();
    }
}
```

### Frame Pacing with VSync

```rust
// Integrate with display link for frame pacing

impl GpuApp {
    pub fn run_with_vsync<F>(&mut self, mut frame_callback: F)
    where
        F: FnMut(&mut Self) -> bool
    {
        let display_link = CVDisplayLink::new();

        display_link.set_output_callback(move |_, _, _, _, _| {
            // Called at VSync interval

            // Check frame budget
            let in_flight = self.frames_in_flight();
            if in_flight >= 2 {
                // Skip frame to catch up
                return;
            }

            // Render new frame
            if !frame_callback(self) {
                display_link.stop();
            }
        });

        display_link.start();
    }
}
```

## Pseudocode

```
class AsyncFrameRenderer:
    shared_event: MTLSharedEvent
    frame_counter: atomic<u64> = 0
    completed_frame: atomic<u64> = 0
    buffers: CommandBuffer[3]  # Triple buffering

    function render_frame_async(app):
        frame_id = atomic_increment(frame_counter)
        slot = frame_id % 3

        # Wait if slot is still in use (max 3 frames in flight)
        wait_for_slot(slot)

        # Build frame
        cmd = build_command_buffer(app)

        # Signal completion
        cmd.encode_signal_event(shared_event, frame_id + 1)

        # Async callback
        cmd.add_completed_handler(lambda:
            record_timing(frame_id, cmd.gpu_times())
            atomic_max(completed_frame, frame_id)
        )

        cmd.commit()
        buffers[slot] = cmd

        return FrameHandle(frame_id)

    function wait_for_slot(slot):
        target = frame_counter - 2
        if completed_frame < target:
            shared_event.wait_until(target + 1)

    function frames_in_flight():
        return frame_counter - completed_frame
```

## Test Plan

### Unit Tests

```rust
// tests/test_issue_142_async_frames.rs

#[test]
fn test_async_completion_callback_fires() {
    let device = Device::system_default().unwrap();
    let mut app = TestApp::new(&device);

    let (tx, rx) = std::sync::mpsc::channel();

    app.set_completion_callback(move |frame_id, timing| {
        tx.send((frame_id, timing)).unwrap();
    });

    // Render a frame
    let handle = app.render_frame_async();

    // Wait for completion (should happen within 100ms)
    let (frame_id, timing) = rx.recv_timeout(Duration::from_millis(100)).unwrap();

    assert_eq!(frame_id, handle.frame_id);
    assert!(timing.gpu_end > timing.gpu_start);
}

#[test]
fn test_triple_buffering() {
    let device = Device::system_default().unwrap();
    let mut app = TestApp::new(&device);

    // Submit 3 frames rapidly
    let h1 = app.render_frame_async();
    let h2 = app.render_frame_async();
    let h3 = app.render_frame_async();

    assert_eq!(app.frames_in_flight(), 3);

    // 4th frame should block until one completes
    let start = Instant::now();
    let h4 = app.render_frame_async();
    let elapsed = start.elapsed();

    // Should have waited for at least one frame
    assert!(elapsed.as_micros() > 100, "Should have blocked");
    assert!(app.frames_in_flight() <= 3);
}

#[test]
fn test_no_wait_until_completed_in_render_path() {
    // Verify we never call wait_until_completed() during normal rendering

    let device = Device::system_default().unwrap();
    let mut app = TestApp::new(&device);

    // Hook to detect blocking calls
    let blocked = Arc::new(AtomicBool::new(false));
    let blocked_clone = blocked.clone();

    app.set_sync_detector(move || {
        blocked_clone.store(true, Ordering::SeqCst);
    });

    // Render 100 frames
    for _ in 0..100 {
        app.render_frame_async();
    }

    // Wait for all to complete
    app.wait_for_all_frames();

    assert!(!blocked.load(Ordering::SeqCst), "Detected blocking wait");
}

#[test]
fn test_frame_timing_accuracy() {
    let device = Device::system_default().unwrap();
    let mut app = TestApp::new(&device);

    let handle = app.render_frame_async();
    app.wait_for_frame(handle.frame_id);

    let timing = app.get_frame_timing(handle.frame_id).unwrap();

    // GPU times should be reasonable
    assert!(timing.gpu_start > 0.0);
    assert!(timing.gpu_end > timing.gpu_start);
    assert!(timing.gpu_end - timing.gpu_start < 0.1); // <100ms
}

#[test]
fn test_input_latency_low_latency_mode() {
    let device = Device::system_default().unwrap();
    let mut app = TestApp::new(&device);
    app.set_pipeline_mode(PipelineMode::LowLatency);

    // Inject input event
    let input_time = Instant::now();
    app.inject_input(InputEvent::MouseMove { x: 100.0, y: 100.0 });

    // Render frame
    let handle = app.render_frame_low_latency();
    app.wait_for_frame(handle.frame_id);

    // Check that input was processed in this frame
    let frame_state = app.get_frame_state(handle.frame_id);
    assert_eq!(frame_state.mouse_x, 100.0);
    assert_eq!(frame_state.mouse_y, 100.0);

    // Input-to-frame latency should be minimal
    let timing = app.get_frame_timing(handle.frame_id).unwrap();
    let latency = Duration::from_secs_f64(timing.gpu_start) - input_time.elapsed();

    // Should be <1 frame (8.33ms at 120fps)
    assert!(latency.as_millis() < 10, "Input latency too high: {:?}", latency);
}
```

### Visual Verification Tests

```rust
// tests/test_issue_142_visual.rs

#[test]
fn visual_test_smooth_animation_async() {
    let device = Device::system_default().unwrap();
    let mut app = AnimationTestApp::new(&device);
    let mut renderer = TestRenderer::new(&device, 800, 600);

    // Capture 60 frames of animation
    let mut frames = Vec::new();

    for i in 0..60 {
        app.set_animation_time(i as f32 / 60.0);
        let handle = app.render_frame_async();
        app.wait_for_frame(handle.frame_id);

        let screenshot = renderer.capture_frame();
        frames.push(screenshot);
    }

    // Verify smooth motion (no frame drops or stutters)
    for i in 1..60 {
        let diff = image_diff(&frames[i-1], &frames[i]);

        // Each frame should be slightly different (animation moving)
        assert!(diff > 0.001, "Frame {} identical to previous (stutter?)", i);

        // But not too different (no frame skips)
        assert!(diff < 0.1, "Frame {} too different (frame skip?)", i);
    }

    // Save animation as GIF for manual inspection
    save_as_gif(&frames, "tests/visual_output/async_animation.gif");
}

#[test]
fn visual_test_frame_pacing() {
    let device = Device::system_default().unwrap();
    let mut app = TestApp::new(&device);

    // Record frame intervals
    let mut intervals = Vec::new();
    let mut last_time = Instant::now();

    for _ in 0..120 {
        let handle = app.render_frame_async();
        app.wait_for_frame(handle.frame_id);

        let now = Instant::now();
        intervals.push(now - last_time);
        last_time = now;
    }

    // Calculate statistics
    let avg = intervals.iter().map(|d| d.as_micros()).sum::<u128>() / 120;
    let variance = intervals.iter()
        .map(|d| (d.as_micros() as i128 - avg as i128).pow(2))
        .sum::<i128>() / 120;
    let std_dev = (variance as f64).sqrt();

    println!("Frame interval: avg={}µs, std_dev={:.1}µs", avg, std_dev);

    // Variance should be low (consistent frame pacing)
    assert!(std_dev < 1000.0, "Frame pacing too variable: std_dev={}", std_dev);

    // Generate frame timing visualization
    let mut chart = FrameTimingChart::new(800, 200);
    for (i, interval) in intervals.iter().enumerate() {
        chart.add_bar(i, interval.as_micros() as f32);
    }
    chart.save("tests/visual_output/frame_pacing.png");
}
```

## Success Metrics

1. **Zero blocking waits:** No `wait_until_completed()` in render path
2. **CPU utilization:** Eliminate 1-2ms idle time per frame
3. **Frame timing:** <1ms variance in frame intervals
4. **Input latency:** Unchanged in LowLatency mode

## Dependencies

None (standalone improvement)

## Files to Modify

1. `src/gpu_os/app.rs` - Main async implementation
2. `src/gpu_os/vsync.rs` - VSync integration
3. `tests/test_issue_142_async_frames.rs` - Unit tests
4. `tests/test_issue_142_visual.rs` - Visual tests
