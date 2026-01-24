//! Visual QA System for GPU Document Viewer
//!
//! Automated testing that validates rendered output without screenshots.
//! Run with: cargo run --release --example visual_qa [url]

use metal::*;

use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine,
    Stylesheet, Viewport, PaintVertex, LayoutBox,
    FLAG_BACKGROUND, FLAG_BORDER, FLAG_TEXT, FLAG_IMAGE,
};

// ============================================================================
// QA Result Types
// ============================================================================

#[derive(Debug, Default)]
struct QAResults {
    total_vertices: usize,
    background_vertices: usize,
    border_vertices: usize,
    text_vertices: usize,
    image_vertices: usize,
    uninitialized_vertices: usize,

    nan_positions: usize,
    inf_positions: usize,
    out_of_bounds_positions: usize,
    invalid_colors: usize,
    garbage_colors: usize,

    elements_parsed: usize,
    text_bytes: usize,
    layout_boxes: usize,
    negative_dimensions: usize,

    tokenize_ms: f64,
    parse_ms: f64,
    style_ms: f64,
    layout_ms: f64,
    paint_ms: f64,
    total_ms: f64,
}

impl QAResults {
    fn is_pass(&self) -> bool {
        self.nan_positions == 0 &&
        self.inf_positions == 0 &&
        self.out_of_bounds_positions == 0 &&
        self.invalid_colors == 0 &&
        self.garbage_colors == 0 &&
        self.uninitialized_vertices == 0 &&
        self.negative_dimensions == 0
    }

    fn print_report(&self, url: &str) {
        let sep = "=".repeat(60);
        println!("\n{}", sep);
        println!("VISUAL QA REPORT: {}", url);
        println!("{}\n", sep);

        let status = if self.is_pass() { "PASS" } else { "FAIL" };
        println!("Status: {}\n", status);

        println!("Vertex Analysis:");
        println!("  Total vertices:      {:>6}", self.total_vertices);
        println!("  - Background:        {:>6}", self.background_vertices);
        println!("  - Border:            {:>6}", self.border_vertices);
        println!("  - Text:              {:>6}", self.text_vertices);
        println!("  - Image:             {:>6}", self.image_vertices);
        println!("  - Uninitialized:     {:>6} {}", self.uninitialized_vertices,
            if self.uninitialized_vertices > 0 { "<-- WARN" } else { "" });

        println!("\nIssues Detected:");
        println!("  NaN positions:       {:>6} {}", self.nan_positions,
            if self.nan_positions > 0 { "FAIL" } else { "ok" });
        println!("  Inf positions:       {:>6} {}", self.inf_positions,
            if self.inf_positions > 0 { "FAIL" } else { "ok" });
        println!("  Out-of-bounds pos:   {:>6} {}", self.out_of_bounds_positions,
            if self.out_of_bounds_positions > 0 { "FAIL" } else { "ok" });
        println!("  Invalid colors:      {:>6} {}", self.invalid_colors,
            if self.invalid_colors > 0 { "FAIL" } else { "ok" });
        println!("  Garbage colors:      {:>6} {}", self.garbage_colors,
            if self.garbage_colors > 0 { "FAIL" } else { "ok" });
        println!("  Negative dimensions: {:>6} {}", self.negative_dimensions,
            if self.negative_dimensions > 0 { "FAIL" } else { "ok" });

        println!("\nDocument Statistics:");
        println!("  Elements parsed:     {:>6}", self.elements_parsed);
        println!("  Text bytes:          {:>6}", self.text_bytes);
        println!("  Layout boxes:        {:>6}", self.layout_boxes);

        println!("\nPipeline Timing:");
        println!("  Tokenize:    {:>7.2}ms", self.tokenize_ms);
        println!("  Parse:       {:>7.2}ms", self.parse_ms);
        println!("  Style:       {:>7.2}ms", self.style_ms);
        println!("  Layout:      {:>7.2}ms", self.layout_ms);
        println!("  Paint:       {:>7.2}ms", self.paint_ms);
        println!("  ---------------------");
        println!("  Total:       {:>7.2}ms", self.total_ms);

        println!("\n{}\n", sep);
    }
}

// ============================================================================
// Visual QA Engine
// ============================================================================

fn is_garbage_color(r: f32, g: f32, b: f32) -> bool {
    let is_bright_magenta = r > 0.8 && g < 0.3 && b > 0.8;
    let is_bright_cyan = r < 0.3 && g > 0.8 && b > 0.8;
    let is_bright_yellow = r > 0.8 && g > 0.8 && b < 0.3;
    let is_neon_green = r < 0.3 && g > 0.9 && b < 0.3;
    is_bright_magenta || is_bright_cyan || is_bright_yellow || is_neon_green
}

fn analyze_vertices(vertices: &[PaintVertex]) -> QAResults {
    let mut results = QAResults::default();
    results.total_vertices = vertices.len();

    for v in vertices {
        match v.flags {
            f if f == FLAG_BACKGROUND => results.background_vertices += 1,
            f if f == FLAG_BORDER => results.border_vertices += 1,
            f if f == FLAG_TEXT => results.text_vertices += 1,
            f if f == FLAG_IMAGE => results.image_vertices += 1,
            0 => results.uninitialized_vertices += 1,
            _ => {}
        }

        let px = v.position[0];
        let py = v.position[1];

        if px.is_nan() || py.is_nan() {
            results.nan_positions += 1;
        }
        if px.is_infinite() || py.is_infinite() {
            results.inf_positions += 1;
        }
        if px < -10.0 || px > 10.0 || py < -10.0 || py > 10.0 {
            results.out_of_bounds_positions += 1;
        }

        let [r, g, b, a] = v.color;
        if r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0 || a < 0.0 || a > 1.0 {
            results.invalid_colors += 1;
        }
        if r.is_nan() || g.is_nan() || b.is_nan() || a.is_nan() {
            results.invalid_colors += 1;
        }
        if a > 0.1 && is_garbage_color(r, g, b) {
            results.garbage_colors += 1;
        }
    }

    results
}

fn analyze_layout(boxes: &[LayoutBox]) -> (usize, usize) {
    let mut negative_dims = 0;
    for b in boxes {
        if b.width < 0.0 || b.height < 0.0 {
            negative_dims += 1;
        }
    }
    (boxes.len(), negative_dims)
}

fn run_qa(html: &[u8], css: &str, url: &str) -> QAResults {
    let device = Device::system_default().expect("No Metal device");

    let mut tokenizer = GpuTokenizer::new(&device).expect("tokenizer");
    let mut parser = GpuParser::new(&device).expect("parser");
    let mut styler = GpuStyler::new(&device).expect("styler");
    let mut layout = GpuLayoutEngine::new(&device).expect("layout");
    let mut paint = GpuPaintEngine::new(&device).expect("paint");

    let viewport = Viewport {
        width: 1024.0,
        height: 768.0,
        _padding: [0.0; 2],
    };

    let total_start = std::time::Instant::now();

    let t0 = std::time::Instant::now();
    let tokens = tokenizer.tokenize(html);
    let tokenize_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = std::time::Instant::now();
    let (elements, text_buffer) = parser.parse(&tokens, html);
    let parse_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = std::time::Instant::now();
    let stylesheet = Stylesheet::parse(css);
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let style_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = std::time::Instant::now();
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let layout_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t0 = std::time::Instant::now();
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);
    let paint_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    let mut results = analyze_vertices(&vertices);
    results.elements_parsed = elements.len();
    results.text_bytes = text_buffer.len();
    let (layout_count, neg_dims) = analyze_layout(&boxes);
    results.layout_boxes = layout_count;
    results.negative_dimensions = neg_dims;

    results.tokenize_ms = tokenize_ms;
    results.parse_ms = parse_ms;
    results.style_ms = style_ms;
    results.layout_ms = layout_ms;
    results.paint_ms = paint_ms;
    results.total_ms = total_ms;

    results.print_report(url);
    results
}

// ============================================================================
// Test HTML Sources
// ============================================================================

fn default_html() -> Vec<u8> {
    br#"<!DOCTYPE html>
<html>
<head><style>
body { margin: 20px; background: #f5f5f5; }
.header { background: #2c3e50; color: white; padding: 20px; }
.card { background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }
h1 { color: #e74c3c; }
h2 { color: #2980b9; }
</style></head>
<body>
<div class="header"><h1>GPU Document Viewer</h1><p>Rendered entirely on the GPU!</p></div>
<div class="card"><h2>Features</h2><ul>
<li>HTML Tokenization</li><li>DOM Tree Construction</li><li>CSS Style Resolution</li>
<li>Layout Computation</li><li>Vertex Generation</li></ul></div>
<div class="card"><h2>Performance</h2>
<p>All processing happens on the GPU using Metal compute shaders.</p></div>
</body></html>"#.to_vec()
}

fn stress_test_html() -> Vec<u8> {
    let mut html = String::from(r#"<!DOCTYPE html><html><head><style>
.box { width: 50px; height: 30px; margin: 5px; display: inline-block; }
.red { background: red; } .green { background: green; }
.blue { background: blue; } .yellow { background: yellow; }
</style></head><body><h1>Stress Test: 200 Elements</h1>"#);

    for i in 0..200 {
        let color = match i % 4 { 0 => "red", 1 => "green", 2 => "blue", _ => "yellow" };
        html.push_str(&format!(r#"<div class="box {}">{}</div>"#, color, i));
    }
    html.push_str("</body></html>");
    html.into_bytes()
}

fn nested_html() -> Vec<u8> {
    br#"<!DOCTYPE html><html><head><style>
.level1 { background: #ffcccc; padding: 20px; margin: 10px; }
.level2 { background: #ccffcc; padding: 15px; margin: 10px; }
.level3 { background: #ccccff; padding: 10px; margin: 10px; }
.level4 { background: #ffffcc; padding: 5px; margin: 5px; }
</style></head><body>
<h1>Nested Elements Test</h1>
<div class="level1">Level 1
<div class="level2">Level 2
<div class="level3">Level 3
<div class="level4">Level 4 - Deepest</div>
<div class="level4">Level 4 - Sibling</div>
</div></div></div>
</body></html>"#.to_vec()
}

fn inline_styles_html() -> Vec<u8> {
    br#"<!DOCTYPE html><html><body>
<h1>Inline Styles Test</h1>
<div style="width: 300px; height: 100px; background: #ff5733; margin: 20px; padding: 10px;">
<p style="color: white;">White text on orange</p></div>
<div style="width: 200px; height: 50px; background: #3498db; margin: 20px;">
<span style="color: #ecf0f1;">Light text on blue</span></div>
<div style="border: 5px solid #2ecc71; padding: 20px; margin: 20px;">Green border box</div>
<div style="display: none;">This should be hidden</div>
</body></html>"#.to_vec()
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let sep = "#".repeat(60);
    println!("\n{}", sep);
    println!("#  GPU DOCUMENT VIEWER - VISUAL QA SYSTEM");
    println!("{}\n", sep);

    let args: Vec<String> = std::env::args().collect();
    let mut all_pass = true;

    if args.len() > 1 {
        let url = &args[1];
        let html = if url.starts_with("http") {
            match ureq::get(url).call() {
                Ok(resp) => resp.into_string().unwrap_or_default().into_bytes(),
                Err(e) => {
                    println!("Failed to fetch {}: {}", url, e);
                    return;
                }
            }
        } else {
            std::fs::read(url).unwrap_or_else(|_| {
                println!("Failed to read file: {}", url);
                std::process::exit(1);
            })
        };

        let results = run_qa(&html, "", url);
        all_pass = results.is_pass();
    } else {
        println!("Running built-in test suite...\n");

        let tests = [
            ("Default HTML", default_html()),
            ("Stress Test (200 elements)", stress_test_html()),
            ("Nested Elements", nested_html()),
            ("Inline Styles", inline_styles_html()),
        ];

        for (name, html) in tests {
            let results = run_qa(&html, "", name);
            if !results.is_pass() {
                all_pass = false;
            }
        }
    }

    println!("\n{}", sep);
    if all_pass {
        println!("#  FINAL RESULT: ALL TESTS PASSED");
    } else {
        println!("#  FINAL RESULT: SOME TESTS FAILED");
    }
    println!("{}\n", sep);

    std::process::exit(if all_pass { 0 } else { 1 });
}
