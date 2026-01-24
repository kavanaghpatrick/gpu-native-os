# Issue #109: Benchmarking Framework

## Summary
Create a comprehensive benchmarking framework to measure and compare GPU browser performance against **Servo** (the real baseline) and naive CPU implementations.

## Motivation
We need empirical data to:
1. **Validate GPU speedup vs Servo** - Servo uses Rayon parallelization, represents state-of-the-art CPU-parallel browser
2. Identify remaining bottlenecks
3. Track performance regressions
4. Prove GPU approach is worthwhile

## Why Benchmark Against Servo?

Servo is the correct baseline because:
- Uses **Rayon** for parallel layout (not naive sequential)
- Uses **Stylo** (same CSS engine as Firefox)
- Uses **WebRender** (GPU-accelerated compositing)
- Represents ~10 years of browser optimization work

If we can't beat Servo, there's no point in the GPU approach.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmarking Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Benchmark Targets:                                              │
│  ├── Servo (via libservo)      ← PRIMARY BASELINE                │
│  ├── GPU-Native (our impl)     ← WHAT WE'RE TESTING              │
│  └── Naive CPU (sequential)    ← SANITY CHECK                    │
│                                                                  │
│  Test Pages:                                                     │
│  ├── test_pages/simple.html         (baseline, ~50 elements)    │
│  ├── test_pages/medium.html         (~500 elements)             │
│  ├── test_pages/complex.html        (~2000 elements)            │
│  ├── test_pages/wikipedia.html      (real-world, ~5000 elements)│
│  ├── test_pages/flexbox_heavy.html  (flex layout stress)        │
│  ├── test_pages/text_heavy.html     (text measurement stress)   │
│  └── test_pages/deeply_nested.html  (20+ depth levels)          │
│                                                                  │
│  Per-Stage Benchmarks:                                           │
│  ├── Selector Matching      (GPU vs Servo/Stylo)                 │
│  ├── Cascade Resolution     (GPU vs Servo/Stylo)                 │
│  ├── Layout Computation     (GPU vs Servo/Layout)                │
│  ├── Text Shaping           (GPU vs Servo/Fonts)                 │
│  ├── Paint Generation       (GPU vs Servo/WebRender)             │
│  └── Full Pipeline          (end-to-end first render)            │
│                                                                  │
│  Metrics:                                                        │
│  ├── Time (µs/ms)                                                │
│  ├── Throughput (elements/sec, glyphs/sec)                       │
│  ├── Memory (peak, resident)                                     │
│  ├── GPU utilization (Metal Performance HUD)                     │
│  └── Frame timing (variance, 99th percentile)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Servo Benchmark Harness

### Option 1: libservo as Dependency (Preferred)

Build Servo from source and link libservo for direct API access:

```rust
// benches/servo_baseline.rs

use servo::{Servo, ServoBuilder, WebView, WebViewBuilder};
use std::time::Instant;

pub struct ServoBenchmark {
    servo: Servo,
    webview: WebView,
}

impl ServoBenchmark {
    pub fn new() -> Self {
        let servo = ServoBuilder::default().build();
        let webview = WebViewBuilder::new(&servo, rendering_context.clone())
            .build();
        Self { servo, webview }
    }

    /// Measure time for Servo to complete initial layout
    pub fn measure_first_layout(&mut self, html: &str) -> Duration {
        // Load HTML via data URL
        let data_url = format!("data:text/html,{}", urlencoding::encode(html));

        let start = Instant::now();

        self.webview.load_url(Url::parse(&data_url).unwrap());

        // Spin event loop until layout complete
        while !self.webview.layout_complete() {
            self.servo.spin_event_loop();
        }

        start.elapsed()
    }

    /// Measure time for subsequent relayout (e.g., after resize)
    pub fn measure_relayout(&mut self, new_size: (u32, u32)) -> Duration {
        let start = Instant::now();

        self.webview.resize(new_size);

        while !self.webview.layout_complete() {
            self.servo.spin_event_loop();
        }

        start.elapsed()
    }
}
```

### Option 2: Instrument Servo Source

Add timing instrumentation to Servo's layout code:

```rust
// In /tmp/servo-src/components/layout/layout_impl.rs

use std::time::Instant;

pub fn perform_layout(...) {
    let total_start = Instant::now();

    // Existing style recalc
    let style_start = Instant::now();
    self.recalc_style(...);
    let style_time = style_start.elapsed();

    // Existing layout
    let layout_start = Instant::now();
    self.perform_layout_internal(...);
    let layout_time = layout_start.elapsed();

    // Existing paint
    let paint_start = Instant::now();
    self.build_display_list(...);
    let paint_time = paint_start.elapsed();

    // Report timings
    BENCHMARK_SINK.report(BenchmarkEvent {
        total: total_start.elapsed(),
        style: style_time,
        layout: layout_time,
        paint: paint_time,
        node_count: self.node_count,
        rule_count: self.rule_count,
    });
}
```

### Option 3: External Timing via DevTools

Use Servo's DevTools protocol to measure layout timing:

```rust
// Connect to Servo DevTools
let client = DevToolsClient::connect("127.0.0.1:6000")?;

// Enable performance tracing
client.send("Performance.enable", json!({}))?;

// Navigate to test page
client.send("Page.navigate", json!({ "url": test_url }))?;

// Wait for load and collect metrics
let metrics = client.recv_until("Page.loadEventFired")?;
let layout_time = metrics["layoutDuration"].as_f64().unwrap();
```

## Test Pages

### simple.html
```html
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 20px; font-family: sans-serif; }
        .container { padding: 10px; background: #f0f0f0; }
        h1 { color: #333; }
        p { line-height: 1.5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Simple Test Page</h1>
        <p>This is a simple paragraph with some text.</p>
        <p>Another paragraph for testing basic layout.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
            <li>List item 3</li>
        </ul>
    </div>
</body>
</html>
```

### flexbox_heavy.html
```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .flex-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .flex-item {
            flex: 1 1 100px;
            min-width: 100px;
            height: 50px;
            background: #4CAF50;
        }
        .nested-flex {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
</head>
<body>
    <!-- Generate 200 flex containers with 5 items each = 1000 flex items -->
    <script>
        for (let i = 0; i < 200; i++) {
            document.write(`
                <div class="flex-container">
                    <div class="flex-item nested-flex">
                        <span>A</span><span>B</span><span>C</span>
                    </div>
                    <div class="flex-item">Item ${i*5+1}</div>
                    <div class="flex-item">Item ${i*5+2}</div>
                    <div class="flex-item">Item ${i*5+3}</div>
                    <div class="flex-item">Item ${i*5+4}</div>
                </div>
            `);
        }
    </script>
</body>
</html>
```

## Benchmark Implementation

```rust
// benches/browser_benchmarks.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;

mod benchmarks {
    use super::*;

    pub fn html_parsing(c: &mut Criterion) {
        let mut group = c.benchmark_group("html_parsing");
        group.measurement_time(Duration::from_secs(10));

        let test_pages = load_test_pages();

        for (name, html) in &test_pages {
            let size = html.len();
            group.throughput(Throughput::Bytes(size as u64));

            group.bench_with_input(
                BenchmarkId::new("html5ever", name),
                html,
                |b, html| b.iter(|| GPUTreeSink::parse(html))
            );
        }

        group.finish();
    }

    pub fn selector_matching(c: &mut Criterion) {
        let mut group = c.benchmark_group("selector_matching");
        group.measurement_time(Duration::from_secs(10));

        let device = Device::system_default().unwrap();
        let test_cases = prepare_selector_tests(&device);

        for tc in &test_cases {
            group.throughput(Throughput::Elements(
                (tc.node_count * tc.rule_count) as u64
            ));

            group.bench_with_input(
                BenchmarkId::new("cpu", &tc.name),
                tc,
                |b, tc| b.iter(|| cpu_match_selectors(&tc.doc, &tc.css))
            );

            group.bench_with_input(
                BenchmarkId::new("gpu", &tc.name),
                tc,
                |b, tc| {
                    b.iter(|| {
                        let cmd = tc.queue.new_command_buffer();
                        tc.matcher.match_all(&cmd, &tc.gpu_dom, &tc.gpu_css);
                        cmd.commit();
                        cmd.wait_until_completed();
                    })
                }
            );
        }

        group.finish();
    }

    pub fn layout_computation(c: &mut Criterion) {
        let mut group = c.benchmark_group("layout");
        group.measurement_time(Duration::from_secs(10));

        let device = Device::system_default().unwrap();
        let test_cases = prepare_layout_tests(&device);

        for tc in &test_cases {
            group.throughput(Throughput::Elements(tc.node_count as u64));

            group.bench_with_input(
                BenchmarkId::new("cpu", &tc.name),
                tc,
                |b, tc| b.iter(|| cpu_layout(&tc.doc, &tc.styles, tc.viewport))
            );

            group.bench_with_input(
                BenchmarkId::new("gpu", &tc.name),
                tc,
                |b, tc| {
                    b.iter(|| {
                        let cmd = tc.queue.new_command_buffer();
                        tc.layout_engine.layout(
                            &cmd, &tc.gpu_dom, &tc.computed_styles, tc.viewport
                        );
                        cmd.commit();
                        cmd.wait_until_completed();
                    })
                }
            );
        }

        group.finish();
    }

    pub fn text_measurement(c: &mut Criterion) {
        let mut group = c.benchmark_group("text_measurement");

        let device = Device::system_default().unwrap();
        let test_texts = vec![
            ("short", "Hello, World!"),
            ("paragraph", include_str!("../test_text/paragraph.txt")),
            ("article", include_str!("../test_text/article.txt")),
            ("book_chapter", include_str!("../test_text/book_chapter.txt")),
        ];

        for (name, text) in test_texts {
            group.throughput(Throughput::Elements(text.chars().count() as u64));

            group.bench_with_input(
                BenchmarkId::new("cpu", name),
                &text,
                |b, text| b.iter(|| cpu_measure_text(text, 16.0, 400.0))
            );

            // GPU setup...
            group.bench_with_input(
                BenchmarkId::new("gpu", name),
                &text,
                |b, _| b.iter(|| gpu_measure_text(&text_buffer, &runs_buffer))
            );
        }

        group.finish();
    }

    pub fn prefix_sum(c: &mut Criterion) {
        let mut group = c.benchmark_group("prefix_sum");

        let device = Device::system_default().unwrap();
        let prefix_sum = GPUPrefixSum::new(&device, &library, 1_000_000);

        for size in [256, 1024, 4096, 16384, 65536, 262144, 1048576] {
            let input: Vec<u32> = (0..size).map(|_| rand::random::<u32>() % 100).collect();

            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new("sequential", size),
                &input,
                |b, input| b.iter(|| cpu_prefix_sum(input))
            );

            let input_buffer = device.new_buffer_with_data(/*...*/);
            let output_buffer = device.new_buffer(/*...*/);

            group.bench_with_input(
                BenchmarkId::new("gpu_parallel", size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        let cmd = queue.new_command_buffer();
                        let encoder = cmd.new_compute_command_encoder();
                        prefix_sum.scan(&encoder, &input_buffer, &output_buffer, size as u32);
                        encoder.end_encoding();
                        cmd.commit();
                        cmd.wait_until_completed();
                    })
                }
            );
        }

        group.finish();
    }

    pub fn full_pipeline(c: &mut Criterion) {
        let mut group = c.benchmark_group("full_pipeline");
        group.measurement_time(Duration::from_secs(30));
        group.sample_size(50);

        let device = Device::system_default().unwrap();
        let test_pages = load_test_pages();

        for (name, html) in &test_pages {
            group.bench_with_input(
                BenchmarkId::new("end_to_end", name),
                html,
                |b, html| {
                    b.iter(|| {
                        // Parse HTML
                        let doc = GPUTreeSink::parse(html);
                        let gpu_dom = GPUDom::from_document(&device, &doc);

                        // Parse CSS
                        let css = extract_styles(&doc);
                        let gpu_css = GPUCSS::from_parser(&device, &css);

                        // Selector matching
                        let cmd = queue.new_command_buffer();
                        matcher.match_all(&cmd, &gpu_dom, &gpu_css);

                        // Cascade resolution
                        cascade.resolve(&cmd, &gpu_dom, &matcher, &gpu_css, viewport);

                        // Layout
                        layout_engine.layout(&cmd, &gpu_dom, &computed_styles, viewport);

                        // Paint
                        paint.generate(&cmd, &layout_buffer, &styles_buffer);

                        cmd.commit();
                        cmd.wait_until_completed();
                    })
                }
            );
        }

        group.finish();
    }

    pub fn frame_timing(c: &mut Criterion) {
        let mut group = c.benchmark_group("frame_timing");
        group.measurement_time(Duration::from_secs(60));
        group.sample_size(1000);

        let device = Device::system_default().unwrap();

        // Pre-load Wikipedia page
        let html = include_str!("../test_pages/wikipedia.html");
        let doc = GPUTreeSink::parse(html);
        let gpu_dom = GPUDom::from_document(&device, &doc);
        // ... setup pipeline

        group.bench_function("wikipedia_frame", |b| {
            b.iter(|| {
                // Simulate frame with scroll change
                let scroll_offset = rand::random::<f32>() * 1000.0;

                let cmd = queue.new_command_buffer();

                // Re-layout (if needed)
                // Re-paint
                paint.generate(&cmd, &layout_buffer, &styles_buffer);

                // Render
                render(&cmd, &vertices, &viewport);

                cmd.commit();
                cmd.wait_until_completed();
            })
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    benchmarks::html_parsing,
    benchmarks::selector_matching,
    benchmarks::layout_computation,
    benchmarks::text_measurement,
    benchmarks::prefix_sum,
    benchmarks::full_pipeline,
    benchmarks::frame_timing,
);

criterion_main!(benches);
```

## Results Dashboard

```rust
// src/bin/benchmark_report.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    name: String,
    variant: String,
    mean_ns: f64,
    std_dev_ns: f64,
    throughput: Option<f64>,
    samples: usize,
}

#[derive(Serialize, Deserialize)]
struct BenchmarkReport {
    timestamp: String,
    git_commit: String,
    system_info: SystemInfo,
    results: Vec<BenchmarkResult>,
}

fn generate_report() -> BenchmarkReport {
    // Run criterion benchmarks
    let output = std::process::Command::new("cargo")
        .args(&["bench", "--", "--save-baseline", "current"])
        .output()
        .expect("Failed to run benchmarks");

    // Parse criterion output
    let results = parse_criterion_output(&output.stdout);

    // Calculate speedups
    let mut speedups = HashMap::new();
    for r in &results {
        if r.variant == "cpu" {
            let gpu_result = results.iter().find(|x| x.name == r.name && x.variant == "gpu");
            if let Some(gpu) = gpu_result {
                speedups.insert(r.name.clone(), r.mean_ns / gpu.mean_ns);
            }
        }
    }

    // Print summary
    println!("\n=== GPU Browser Benchmark Results ===\n");
    println!("| Benchmark | CPU (ms) | GPU (ms) | Speedup |");
    println!("|-----------|----------|----------|---------|");

    for (name, speedup) in &speedups {
        let cpu = results.iter().find(|r| r.name == *name && r.variant == "cpu").unwrap();
        let gpu = results.iter().find(|r| r.name == *name && r.variant == "gpu").unwrap();

        println!(
            "| {} | {:.3} | {:.3} | {:.1}x |",
            name,
            cpu.mean_ns / 1_000_000.0,
            gpu.mean_ns / 1_000_000.0,
            speedup
        );
    }

    BenchmarkReport {
        timestamp: chrono::Utc::now().to_rfc3339(),
        git_commit: get_git_commit(),
        system_info: get_system_info(),
        results,
    }
}

fn main() {
    let report = generate_report();

    // Save JSON report
    let json = serde_json::to_string_pretty(&report).unwrap();
    std::fs::write("benchmark_results.json", &json).unwrap();

    // Append to CSV for tracking over time
    append_to_csv(&report, "benchmark_history.csv");

    println!("\nReport saved to benchmark_results.json");
}
```

## Integration with CI

```yaml
# .github/workflows/benchmark.yml

name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: macos-latest  # Need Metal support

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run benchmarks
        run: cargo bench --features=benchmark

      - name: Generate report
        run: cargo run --bin benchmark_report

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: |
            benchmark_results.json
            target/criterion/

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('benchmark_results.json'));
            // Generate comment with results table
            // ...
```

## Servo Comparison Benchmarks

```rust
// benches/servo_comparison.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_vs_servo(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_servo");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(20);  // Servo startup is expensive

    let test_pages = vec![
        ("simple", include_str!("../test_pages/simple.html")),
        ("medium", include_str!("../test_pages/medium.html")),
        ("wikipedia", include_str!("../test_pages/wikipedia.html")),
        ("flexbox", include_str!("../test_pages/flexbox_heavy.html")),
    ];

    // Initialize Servo once
    let mut servo_bench = ServoBenchmark::new();

    // Initialize GPU pipeline once
    let device = Device::system_default().unwrap();
    let gpu_pipeline = GPUBrowserPipeline::new(&device);

    for (name, html) in &test_pages {
        // Servo baseline (Rayon-parallelized)
        group.bench_with_input(
            BenchmarkId::new("servo", name),
            html,
            |b, html| {
                b.iter(|| servo_bench.measure_first_layout(html))
            }
        );

        // Our GPU implementation
        group.bench_with_input(
            BenchmarkId::new("gpu_native", name),
            html,
            |b, html| {
                b.iter(|| gpu_pipeline.full_render(html))
            }
        );

        // Naive sequential (sanity check)
        group.bench_with_input(
            BenchmarkId::new("naive_cpu", name),
            html,
            |b, html| {
                b.iter(|| naive_sequential_render(html))
            }
        );
    }

    group.finish();
}

fn bench_layout_vs_servo(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_vs_servo");

    // Pre-parse HTML and CSS to isolate layout timing
    let html = include_str!("../test_pages/wikipedia.html");
    let doc = parse_html(html);
    let styles = parse_css(&doc);

    // Servo layout (via instrumented build)
    group.bench_function("servo_layout", |b| {
        b.iter(|| servo_layout_only(&doc, &styles))
    });

    // GPU layout
    let gpu_dom = GPUDom::from_document(&device, &doc);
    let gpu_styles = GPUComputedStyles::from(&styles);

    group.bench_function("gpu_layout", |b| {
        b.iter(|| {
            let cmd = queue.new_command_buffer();
            layout_engine.layout(&cmd, &gpu_dom, &gpu_styles, viewport);
            cmd.commit();
            cmd.wait_until_completed();
        })
    });

    group.finish();
}

criterion_group!(
    servo_benches,
    bench_vs_servo,
    bench_layout_vs_servo,
);
```

## Expected Results Table (vs Servo)

**Key insight**: Servo already uses Rayon, so we're competing against optimized parallel CPU code, not naive sequential. Speedups will be smaller but still significant.

| Component | Page | Naive CPU | Servo (Rayon) | GPU-Native | vs Servo |
|-----------|------|-----------|---------------|------------|----------|
| **Selector Matching** | | | | | |
| | simple (50 nodes) | 0.5ms | 0.2ms | 0.05ms | 4x |
| | wikipedia (5K nodes) | 500ms | 50ms | 2ms | **25x** |
| **Cascade Resolution** | | | | | |
| | simple | 0.1ms | 0.05ms | 0.02ms | 2.5x |
| | wikipedia | 50ms | 10ms | 1ms | **10x** |
| **Layout** | | | | | |
| | simple | 0.5ms | 0.3ms | 0.1ms | 3x |
| | wikipedia | 100ms | 20ms | 3ms | **7x** |
| | flexbox_heavy | 50ms | 10ms | 0.8ms | **12x** |
| **Text Shaping** | | | | | |
| | paragraph | 0.5ms | 0.2ms | 0.02ms | **10x** |
| | article (10K chars) | 10ms | 3ms | 0.1ms | **30x** |
| **Paint Generation** | | | | | |
| | simple | 0.2ms | 0.1ms | 0.05ms | 2x |
| | wikipedia | 10ms | 5ms | 0.5ms | **10x** |
| **Full Pipeline** | | | | | |
| | simple.html | 2ms | 1ms | 0.3ms | 3x |
| | wikipedia.html | 700ms | 90ms | 10ms | **9x** |

### Why GPU Wins Even vs Rayon

1. **Selector Matching**: O(rules × nodes) - Rayon parallelizes by node, but GPU parallelizes by rule×node pairs
2. **Layout**: GPU can process all siblings simultaneously; Rayon still has tree-walk overhead
3. **Text**: GPU SDF atlas lookup is O(1) per glyph; CPU needs HarfBuzz shaping
4. **Paint**: GPU atomic allocation beats Rayon's work-stealing for vertex generation

### Where Servo Might Win

1. **Small pages**: GPU dispatch overhead dominates for <100 elements
2. **Complex CSS**: Servo's Stylo handles full CSS spec; our subset may have correctness gaps
3. **Memory bandwidth**: Very large pages might hit GPU memory limits

## Servo Build Instructions

To build Servo with benchmarking support:

```bash
# Clone Servo
git clone https://github.com/servo/servo /tmp/servo-src
cd /tmp/servo-src

# Build with profiling
./mach build --release --features=profiling

# Run benchmarks
./mach run --release -- --benchmark test_pages/wikipedia.html
```

## Instrumented Servo Metrics

Add this to Cargo.toml to get Servo's internal metrics:

```toml
[dependencies]
libservo = { path = "/tmp/servo-src/components/servo", features = ["tracing"] }
```

Then access metrics:

```rust
// Enable tracing subscriber
tracing_subscriber::fmt::init();

// Servo will emit spans like:
// servo::layout::perform_layout{node_count=5000, duration_ms=20}
// servo::style::recalc_style{rule_count=3000, duration_ms=10}
```

## Acceptance Criteria

- [ ] All benchmarks run without errors
- [ ] Criterion reports generated correctly
- [ ] CPU vs GPU comparisons available
- [ ] Results reproducible (low variance)
- [ ] CI integration working
- [ ] Historical tracking in place
- [ ] Memory benchmarks included
- [ ] Frame timing benchmarks pass 60fps

## Dependencies

- All other GPU browser issues (for complete benchmarking)

## Blocks

- None (final validation step)
