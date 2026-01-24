#!/bin/bash
# GPU-Native Filesystem Profiling Guide
#
# How to profile the GPU filesystem with Xcode Instruments

cat << 'EOF'
# GPU Filesystem Profiling Guide

## Quick Start

Run the automated profiling suite:
```bash
# Basic performance profile
cargo run --release --example filesystem_profile

# GPU vs CPU comparison
cargo run --release --example filesystem_cpu_comparison

# Path lookup demo with timing
cargo run --release --example filesystem_path_lookup
```

## Xcode Instruments Profiling (Advanced)

### 1. Metal System Trace

**Captures**: GPU timeline, command buffer execution, shader occupancy

```bash
# Build with debug symbols
cargo build --release

# Open in Instruments
instruments -t "Metal System Trace" \
    ./target/release/examples/filesystem_profile
```

**What to Look For**:
- Command buffer latency (should be ~200µs)
- GPU utilization (should be near 100% during compute)
- Memory bandwidth (should hit ~400GB/s on M4 Pro)
- Shader execution time (should be <10µs for path lookup)

### 2. GPU Frame Capture

**Note**: Requires running in a window (not console)

```bash
# For visual demos with Metal debugging
cargo run --release --example filesystem_browser_window

# Then: Xcode → Debug → Capture GPU Frame
```

**What to Analyze**:
- Buffer bindings (verify all 7 buffers bound)
- Threadgroup configuration (should be 1 × 1 × 1024)
- Register usage
- Memory access patterns

### 3. Time Profiler

**Captures**: CPU timing breakdown

```bash
instruments -t "Time Profiler" \
    ./target/release/examples/filesystem_profile
```

**What to Look For**:
- Time spent in `lookup_path` method
- Metal API overhead
- Command buffer creation time
- String operations (path splitting)

## Manual Profiling Techniques

### Measure GPU Kernel Time Only

Add Metal event profiling:

```rust
use metal::MTLCaptureManager;

// Before dispatch
let start_event = command_buffer.make_event();
encoder.signal_event(&start_event);

// After dispatch
let end_event = command_buffer.make_event();
encoder.signal_event(&end_event);

// Measure difference (GPU timeline)
let gpu_time = end_event.timestamp() - start_event.timestamp();
```

### Measure Memory Bandwidth

```rust
let bytes_read = (
    inode_count * size_of::<InodeCompact>() +
    entry_count * size_of::<DirEntryCompact>()
) as f64;

let bytes_written = size_of::<u32>() * 2; // result + status

let total_bytes = bytes_read + bytes_written;
let bandwidth_gb_s = (total_bytes / gpu_time_seconds) / 1e9;

println!("Memory bandwidth: {:.2} GB/s", bandwidth_gb_s);
```

### Measure GPU Occupancy

**Theoretical Max**:
- M4 Pro: 20 GPU cores × 1024 threads = 20,480 threads
- Our kernel: 1 threadgroup × 1024 threads = 1,024 threads
- **Occupancy: 5%** (could dispatch 20 threadgroups in parallel!)

**Optimization**: Batch multiple directories into one dispatch

## Performance Metrics Reference

### Good Performance Indicators

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Command buffer latency | <100µs | 100-500µs | >500µs |
| GPU utilization | >80% | 50-80% | <50% |
| Memory bandwidth | >300GB/s | 100-300GB/s | <100GB/s |
| Threadgroup occupancy | >50% | 10-50% | <10% |
| Cache hit rate | >90% | 70-90% | <70% |

### Current Performance (Phase 1 MVP)

| Metric | Current | Target (Phase 2) |
|--------|---------|------------------|
| Command buffer latency | ~200µs | <50µs (pooling) |
| Lookup latency (avg) | 200-400µs | 2-5µs (batching + cache) |
| Throughput | 5,000 ops/sec | 500,000 ops/sec |
| GPU utilization | ~5% | >50% (batching) |
| Memory bandwidth | ~50GB/s | >300GB/s |

## Common Issues and Solutions

### Issue: High Latency (>500µs per lookup)

**Diagnosis**:
```bash
# Check if debug mode
cargo run --example filesystem_profile  # SLOW
cargo run --release --example filesystem_profile  # FAST
```

**Solutions**:
- Always use `--release` for benchmarks
- Check for GPU thermal throttling
- Verify no other GPU-intensive apps running

### Issue: Low Throughput (<1000 ops/sec)

**Diagnosis**:
```rust
// Add timing around GPU dispatch
let start = Instant::now();
command_buffer.wait_until_completed();
let dispatch_time = start.elapsed();

if dispatch_time > Duration::from_micros(500) {
    println!("WARNING: Slow GPU dispatch!");
}
```

**Solutions**:
- Implement batching (multiple lookups per dispatch)
- Use async dispatch (don't wait)
- Add CPU-side cache for hot paths

### Issue: GPU Not Being Used

**Diagnosis**:
```bash
# Monitor GPU usage while running
sudo powermetrics --samplers gpu_power -i 100

# Should show GPU active during benchmark
```

**Solutions**:
- Verify Metal device is system_default()
- Check that compute pipeline is being executed
- Ensure not falling back to CPU path

## Benchmarking Best Practices

### 1. Warmup

Always run a warmup iteration:
```rust
// Warmup
for _ in 0..10 {
    let _ = fs.lookup_path("/test");
}

// Now benchmark
let start = Instant::now();
for _ in 0..1000 {
    let _ = fs.lookup_path("/test");
}
let duration = start.elapsed();
```

### 2. Statistical Significance

Run multiple iterations and report statistics:
```rust
let mut timings = Vec::new();
for _ in 0..100 {
    let start = Instant::now();
    let _ = fs.lookup_path(path);
    timings.push(start.elapsed());
}

timings.sort();
let min = timings[0];
let median = timings[timings.len() / 2];
let p95 = timings[(timings.len() * 95) / 100];
let max = timings[timings.len() - 1];
```

### 3. Isolate Variables

Test one thing at a time:
```rust
// Good: Test path depth impact in isolation
test_depth_1();
test_depth_2();
test_depth_3();

// Bad: Mixed test
test_various_depths_and_sizes();
```

### 4. Measure End-to-End

Include realistic workflow:
```rust
// Simulate real application usage
fn simulate_file_browse() {
    fs.lookup_path("/usr/bin");  // cd command
    fs.list_directory(42);        // ls command
    fs.lookup_path("/usr/bin/git"); // stat command
}
```

## Profiling Checklist

Before reporting performance numbers:

- [ ] Compiled with `--release`
- [ ] Warmup iterations completed
- [ ] Multiple runs (100+ iterations)
- [ ] Statistical analysis (min/avg/median/p95/max)
- [ ] GPU confirmed active (powermetrics)
- [ ] System not under load (Activity Monitor)
- [ ] Thermal throttling not occurring
- [ ] Results reproducible

## Advanced: Metal Debugging

### Enable Metal Validation

```bash
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=0  # Assert on errors
cargo run --example filesystem_profile
```

### Metal Shader Debugger

1. Open Xcode
2. Run example in debug mode
3. Xcode → Debug → Capture GPU Frame
4. Select compute kernel
5. Click "Debug" to step through shader code

### GPU Timeline Analysis

Use `MTLCaptureManager` for programmatic capture:
```rust
use metal::MTLCaptureManager;

let capture_manager = MTLCaptureManager::shared();
let descriptor = MTLCaptureDescriptor::new();
descriptor.set_destination(MTLCaptureDestination::DeveloperTools);
capture_manager.start_capture(&descriptor);

// Run benchmark here

capture_manager.stop_capture();
```

## Quick Reference Commands

```bash
# Run all benchmarks
./run_all_benchmarks.sh

# Profile with Instruments
instruments -t "Metal System Trace" \
    ./target/release/examples/filesystem_profile

# Monitor GPU usage
sudo powermetrics --samplers gpu_power -i 100

# Check build mode
file ./target/debug/examples/filesystem_profile    # debug
file ./target/release/examples/filesystem_profile  # release

# Clean and rebuild
cargo clean && cargo build --release

# Run single benchmark
cargo run --release --example filesystem_profile 2>/dev/null
```

## Expected Results (Apple M4 Pro)

```
Individual Lookup: ~200-400µs
Batch Throughput: ~5,000 ops/sec
GPU Utilization: ~5%
Memory Bandwidth: ~50 GB/s
Command Buffer Overhead: ~200µs

With Phase 2 Optimizations:
Individual Lookup: ~2-5µs
Batch Throughput: ~500,000 ops/sec
GPU Utilization: ~50%
Memory Bandwidth: ~300 GB/s
Command Buffer Overhead: ~20µs (pooling)
```
EOF
