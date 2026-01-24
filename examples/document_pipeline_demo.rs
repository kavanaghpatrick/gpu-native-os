//! GPU Document Pipeline Demo
//!
//! Demonstrates the complete GPU-native document processing pipeline:
//! HTML bytes → Tokens → Elements → Styles → Layout → Vertices
//!
//! Run with: cargo run --release --example document_pipeline_demo

use metal::Device;
use rust_experiment::gpu_os::document::{
    GpuTokenizer, GpuParser, GpuStyler, GpuLayoutEngine, GpuPaintEngine,
    Stylesheet, Viewport, FLAG_BACKGROUND, FLAG_BORDER, FLAG_TEXT,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("       GPU-Native Document Pipeline Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Initialize GPU
    let device = Device::system_default().expect("No Metal GPU device found");
    println!("✓ GPU: {}\n", device.name());

    // Create pipeline stages
    let mut tokenizer = GpuTokenizer::new(&device).expect("Failed to create tokenizer");
    let mut parser = GpuParser::new(&device).expect("Failed to create parser");
    let mut styler = GpuStyler::new(&device).expect("Failed to create styler");
    let mut layout = GpuLayoutEngine::new(&device).expect("Failed to create layout engine");
    let mut paint = GpuPaintEngine::new(&device).expect("Failed to create paint engine");

    println!("✓ Pipeline stages initialized\n");

    // Sample HTML document
    let html = b"<!DOCTYPE html>
<html>
<head>
    <title>GPU Document Demo</title>
</head>
<body>
    <header>
        <h1>Welcome to GPU-Native Rendering</h1>
        <nav>
            <a href=\"/about\">About</a>
            <a href=\"/features\">Features</a>
            <a href=\"/demo\">Demo</a>
        </nav>
    </header>

    <main>
        <section id=\"about\">
            <h2>About This Demo</h2>
            <p>This demonstrates a complete GPU-accelerated document pipeline.</p>
            <p>All processing happens on the GPU using Metal compute shaders.</p>
        </section>

        <section id=\"features\">
            <h2>Features</h2>
            <ul>
                <li>Parallel HTML tokenization</li>
                <li>GPU tree construction</li>
                <li>CSS selector matching</li>
                <li>Flexbox layout</li>
                <li>Vertex generation</li>
            </ul>
        </section>

        <div class=\"card highlight\">
            <h3>Performance</h3>
            <p>Processing 5K elements in under 50ms!</p>
        </div>
    </main>

    <footer>
        <p>Built with Rust and Metal</p>
    </footer>
</body>
</html>";

    // Sample CSS
    let css = r#"
        * { margin: 0; padding: 0; }
        body { background: #f0f0f0; color: #333; font-size: 16px; }
        header { background: #2c3e50; padding: 20px; }
        h1 { color: white; font-size: 32px; }
        nav { display: flex; }
        nav a { color: #3498db; margin: 10px; }
        main { padding: 20px; }
        section { margin: 20px 0; }
        h2 { color: #2c3e50; font-size: 24px; margin: 10px 0; }
        p { line-height: 1.6; margin: 10px 0; }
        ul { margin: 10px 20px; }
        li { margin: 5px 0; }
        .card { background: white; padding: 20px; border-radius: 8px; }
        .highlight { border: 2px solid #3498db; }
        footer { background: #34495e; color: white; padding: 10px; text-align: center; }
    "#;

    let viewport = Viewport {
        width: 1024.0,
        height: 768.0,
        _padding: [0.0; 2],
    };

    println!("Input:");
    println!("  HTML: {} bytes", html.len());
    println!("  CSS:  {} bytes", css.len());
    println!("  Viewport: {}x{}\n", viewport.width, viewport.height);

    // ═══════════════════════════════════════════════════════════════
    // PASS 1: Tokenization
    // ═══════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("Pass 1: Tokenization (HTML bytes → Tokens)");
    println!("───────────────────────────────────────────────────────────────");

    let start = std::time::Instant::now();
    let tokens = tokenizer.tokenize(html);
    let tokenize_time = start.elapsed();

    println!("  ✓ {} tokens in {:?}", tokens.len(), tokenize_time);
    println!("  Throughput: {:.1} MB/s\n",
        html.len() as f64 / tokenize_time.as_secs_f64() / 1_000_000.0);

    // ═══════════════════════════════════════════════════════════════
    // PASS 2: Parsing
    // ═══════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("Pass 2: Parsing (Tokens → Element Tree)");
    println!("───────────────────────────────────────────────────────────────");

    let start = std::time::Instant::now();
    let (elements, text_buffer) = parser.parse(&tokens, html);
    let parse_time = start.elapsed();

    println!("  ✓ {} elements in {:?}", elements.len(), parse_time);
    println!("  Text buffer: {} bytes", text_buffer.len());

    // Count element types
    let text_nodes = elements.iter().filter(|e| e.element_type == 100).count();
    let element_nodes = elements.len() - text_nodes;
    println!("  Elements: {}, Text nodes: {}\n", element_nodes, text_nodes);

    // ═══════════════════════════════════════════════════════════════
    // PASS 3: Style Resolution
    // ═══════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("Pass 3: Style Resolution (Elements × CSS → Computed Styles)");
    println!("───────────────────────────────────────────────────────────────");

    let stylesheet = Stylesheet::parse(css);
    println!("  Parsed {} CSS selectors", stylesheet.selectors.len());

    let start = std::time::Instant::now();
    let styles = styler.resolve_styles(&elements, &tokens, html, &stylesheet);
    let style_time = start.elapsed();

    println!("  ✓ {} computed styles in {:?}", styles.len(), style_time);

    // Sample some styles
    if !styles.is_empty() {
        let body_style = &styles[0];
        println!("  Sample (body): bg={:?}, font_size={}",
            body_style.background_color, body_style.font_size);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // PASS 4: Layout
    // ═══════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("Pass 4: Layout (Styles → Positions)");
    println!("───────────────────────────────────────────────────────────────");

    let start = std::time::Instant::now();
    let boxes = layout.compute_layout(&elements, &styles, &text_buffer, viewport);
    let layout_time = start.elapsed();

    println!("  ✓ {} layout boxes in {:?}", boxes.len(), layout_time);

    // Sample some layouts
    if boxes.len() > 1 {
        let first = &boxes[0];
        println!("  Sample (root): pos=({:.0}, {:.0}), size=({:.0}x{:.0})",
            first.x, first.y, first.width, first.height);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // PASS 5: Paint (Vertex Generation)
    // ═══════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────────");
    println!("Pass 5: Paint (Layout → GPU Vertices)");
    println!("───────────────────────────────────────────────────────────────");

    let start = std::time::Instant::now();
    let vertices = paint.paint(&elements, &boxes, &styles, &text_buffer, viewport);
    let paint_time = start.elapsed();

    println!("  ✓ {} vertices in {:?}", vertices.len(), paint_time);

    // Count by type
    let bg_count = vertices.iter().filter(|v| v.flags == FLAG_BACKGROUND).count();
    let border_count = vertices.iter().filter(|v| v.flags == FLAG_BORDER).count();
    let text_count = vertices.iter().filter(|v| v.flags == FLAG_TEXT).count();

    println!("  Backgrounds: {} vertices ({} quads)", bg_count, bg_count / 4);
    println!("  Borders: {} vertices", border_count);
    println!("  Text: {} vertices ({} glyphs)\n", text_count, text_count / 4);

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════
    let total_time = tokenize_time + parse_time + style_time + layout_time + paint_time;

    println!("═══════════════════════════════════════════════════════════════");
    println!("                         SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Pipeline Stage          Time         Output");
    println!("  ─────────────────────────────────────────────────────────────");
    println!("  1. Tokenize         {:>8.2?}    {:>6} tokens", tokenize_time, tokens.len());
    println!("  2. Parse            {:>8.2?}    {:>6} elements", parse_time, elements.len());
    println!("  3. Style            {:>8.2?}    {:>6} styles", style_time, styles.len());
    println!("  4. Layout           {:>8.2?}    {:>6} boxes", layout_time, boxes.len());
    println!("  5. Paint            {:>8.2?}    {:>6} vertices", paint_time, vertices.len());
    println!("  ─────────────────────────────────────────────────────────────");
    println!("  TOTAL               {:>8.2?}", total_time);
    println!();
    println!("  Throughput: {:.1} KB/s (HTML to vertices)",
        html.len() as f64 / total_time.as_secs_f64() / 1000.0);
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Ready to render {} vertices to screen!", vertices.len());
    println!("═══════════════════════════════════════════════════════════════");
}
